#!/usr/bin/env python3
"""Paired static-mask validation for the aggressive TeaCache budget probe.

The experimental unit is one latent-seed offset, not one generated image. Runs
with the same dataset seed and latent offset are paired across forced masks and,
when available, a plain TeaCache threshold comparator.
"""

import argparse
import glob
import math
import os
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_budget_probe import _budget_dirs, _f  # noqa: E402
from analyze_teacache_sweeps import _load  # noqa: E402

Run = Dict[str, object]
RunMap = Dict[int, Run]


_PAIR_CONFIG_KEYS = (
    "seed",
    "n_prompts",
    "batch_size",
    "num_steps",
    "total_images",
    "dataset_start_index",
    "generation_start_index",
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_run(path: str, expected_offset: Optional[int] = None
              ) -> Tuple[Optional[Run], Optional[str]]:
    payload = _load(path)
    if payload is None:
        return None, f"unreadable results: {path}"
    config = payload.get("config", {})
    aggregate = payload.get("aggregate", {})
    try:
        offset = int(config.get("latent_seed_offset", 0))
    except (TypeError, ValueError):
        return None, f"invalid latent_seed_offset: {path}"
    if expected_offset is not None and offset != expected_offset:
        return None, (f"directory offset {expected_offset} != config offset "
                      f"{offset}: {path}")
    return {
        "path": path,
        "offset": offset,
        "config": config,
        "aggregate": aggregate,
        "fid": aggregate.get("fid"),
        "is_mean": aggregate.get("is_mean"),
        "total_calc": aggregate.get("total_calc"),
    }, None


def _discover_replica_dir(run_dir: str) -> Tuple[RunMap, List[str]]:
    runs: RunMap = {}
    warnings: List[str] = []
    root_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(root_path):
        run, warning = _load_run(root_path, expected_offset=0)
        if warning:
            warnings.append(warning)
        elif run is not None:
            runs[0] = run
    for path in sorted(glob.glob(os.path.join(
            run_dir, "rep_*", "results.json"))):
        label = os.path.basename(os.path.dirname(path))
        try:
            expected = int(label[len("rep_"):])
        except ValueError:
            warnings.append(f"invalid replica directory: {path}")
            continue
        run, warning = _load_run(path, expected_offset=expected)
        if warning:
            warnings.append(warning)
            continue
        assert run is not None
        if expected in runs:
            warnings.append(f"duplicate latent offset {expected}: {path}")
            continue
        runs[expected] = run
    return runs, warnings


def discover_arm_runs(budget_dir: str) -> Tuple[Dict[str, RunMap], List[str]]:
    arms: Dict[str, RunMap] = {}
    warnings: List[str] = []
    pattern = os.path.join(budget_dir, "equalflops", "arm_*")
    for arm_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(arm_dir):
            continue
        arm = os.path.basename(arm_dir)[len("arm_"):]
        runs, arm_warnings = _discover_replica_dir(arm_dir)
        warnings.extend(arm_warnings)
        if runs:
            arms[arm] = runs
    return arms, warnings


def discover_threshold_runs(root: str) -> Tuple[Dict[str, RunMap], List[str]]:
    thresholds: Dict[str, RunMap] = {}
    warnings: List[str] = []
    for threshold_dir in sorted(glob.glob(os.path.join(
            root, "threshold", "thresh_*"))):
        if not os.path.isdir(threshold_dir):
            continue
        label = os.path.basename(threshold_dir)[len("thresh_"):]
        runs, threshold_warnings = _discover_replica_dir(threshold_dir)
        warnings.extend(threshold_warnings)
        if runs:
            thresholds[label] = runs
    return thresholds, warnings


def _pair_compatible(left: Run, right: Run) -> Tuple[bool, Optional[str]]:
    left_cfg = left["config"]
    right_cfg = right["config"]
    assert isinstance(left_cfg, dict) and isinstance(right_cfg, dict)
    for key in _PAIR_CONFIG_KEYS:
        if left_cfg.get(key) != right_cfg.get(key):
            return False, (f"offset {left['offset']}: config.{key} differs "
                           f"({left_cfg.get(key)!r} vs {right_cfg.get(key)!r})")
    return True, None


def paired_deltas(left: RunMap, right: RunMap, metric: str
                  ) -> Tuple[List[Tuple[int, float]], List[str]]:
    deltas: List[Tuple[int, float]] = []
    warnings: List[str] = []
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    if left_only:
        warnings.append(f"left-only latent offsets: {left_only}")
    if right_only:
        warnings.append(f"right-only latent offsets: {right_only}")
    for offset in sorted(set(left) & set(right)):
        compatible, warning = _pair_compatible(left[offset], right[offset])
        if not compatible:
            assert warning is not None
            warnings.append(warning)
            continue
        left_value = left[offset].get(metric)
        right_value = right[offset].get(metric)
        if not _finite(left_value) or not _finite(right_value):
            warnings.append(f"offset {offset}: non-finite {metric}")
            continue
        deltas.append((offset, float(left_value) - float(right_value)))
    return deltas, warnings


def exact_sign_p(values: Iterable[float], direction: str) -> Optional[float]:
    """One-sided exact sign-test p, dropping exact ties.

    ``direction='less'`` tests for a negative median; ``'greater'`` tests for a
    positive median.
    """
    values = [float(value) for value in values if float(value) != 0.0]
    if not values:
        return None
    if direction == "less":
        successes = sum(value < 0.0 for value in values)
    elif direction == "greater":
        successes = sum(value > 0.0 for value in values)
    else:
        raise ValueError("direction must be 'less' or 'greater'")
    n = len(values)
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / (2 ** n)


def paired_stats(deltas: Sequence[Tuple[int, float]]) -> Dict[str, object]:
    values = [value for _, value in deltas]
    n = len(values)
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if n >= 2 else None
    se = sd / math.sqrt(n) if sd is not None else None
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(values) if values else None,
        "sd": sd,
        "se": se,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "positive": sum(value > 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
    }


def calc_per_trajectory(run: Run) -> Optional[float]:
    """Return calc steps from the final generation recorded by the runner.

    ``run_dit`` resets TeaCache's decision history before every batch, so the
    aggregate ``total_calc`` is already one trajectory's count rather than a
    sum over all batches.
    """
    total_calc = run.get("total_calc")
    if not _finite(total_calc):
        return None
    return float(total_calc)


def nearest_threshold(thresholds: Dict[str, RunMap], target_calc: float
                      ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    candidates: List[Tuple[float, float, str]] = []
    for label, runs in thresholds.items():
        calc_values = [calc for calc in (
            calc_per_trajectory(run) for run in runs.values()) if calc is not None]
        if not calc_values:
            continue
        mean_calc = statistics.fmean(calc_values)
        candidates.append((abs(mean_calc - target_calc), mean_calc, label))
    if not candidates:
        return None, None, None
    distance, mean_calc, label = min(candidates)
    return label, mean_calc, distance


def _fmt(value: object, spec: str = ".3f") -> str:
    return _f(value, spec)


def _render_warnings(warnings: Iterable[str]) -> None:
    for warning in dict.fromkeys(warnings):
        print(f"  [WARN] {warning}")


def _render_pair(left_name: str, right_name: str, left: RunMap,
                 right: RunMap, fid_margin: float, is_margin: float,
                 min_pairs: int) -> Tuple[bool, bool]:
    print(f"  comparison: {left_name} - {right_name} (negative FID is better; "
          "positive IS is better)")
    fid_deltas, fid_warnings = paired_deltas(left, right, "fid")
    is_deltas, is_warnings = paired_deltas(left, right, "is_mean")
    _render_warnings(fid_warnings + is_warnings)

    print("    offset      delta_FID       delta_IS")
    fid_by_offset = dict(fid_deltas)
    is_by_offset = dict(is_deltas)
    for offset in sorted(set(fid_by_offset) | set(is_by_offset)):
        print(f"    {offset:>6} {_fmt(fid_by_offset.get(offset), '.3f'):>14} "
              f"{_fmt(is_by_offset.get(offset), '.3f'):>14}")

    fid_stats = paired_stats(fid_deltas)
    is_stats = paired_stats(is_deltas)
    print("    paired summary")
    print(f"      FID: n={fid_stats['n']} mean={_fmt(fid_stats['mean'])} "
          f"sd={_fmt(fid_stats['sd'])} se={_fmt(fid_stats['se'])} "
          f"range={_fmt(fid_stats['minimum'])}..{_fmt(fid_stats['maximum'])}")
    print(f"       IS: n={is_stats['n']} mean={_fmt(is_stats['mean'])} "
          f"sd={_fmt(is_stats['sd'])} se={_fmt(is_stats['se'])} "
          f"range={_fmt(is_stats['minimum'])}..{_fmt(is_stats['maximum'])}")

    if fid_stats["n"] < min_pairs or is_stats["n"] < min_pairs:
        print(f"    VERDICT: INSUFFICIENT EVIDENCE (need >= {min_pairs} complete "
              "paired latent offsets)")
        return False, False

    fid_adjusted = [value - fid_margin for _, value in fid_deltas]
    is_adjusted = [value - is_margin for _, value in is_deltas]
    fid_p = exact_sign_p(fid_adjusted, "less")
    is_p = exact_sign_p(is_adjusted, "greater")
    fid_pass = fid_p is not None and fid_p <= 0.05
    is_pass = is_p is not None and is_p <= 0.05
    print(f"    FID non-inferiority: margin={fid_margin:.3f}, "
          f"p={_fmt(fid_p, '.5f')} -> {'PASS' if fid_pass else 'FAIL'}")
    print(f"    IS superiority: margin={is_margin:.3f}, "
          f"p={_fmt(is_p, '.5f')} -> {'PASS' if is_pass else 'FAIL'}")
    if fid_pass and is_pass:
        print(f"    VERDICT: {left_name} is FID-non-inferior and IS-superior")
    else:
        print("    VERDICT: UNDECIDED — do not force a static winner")
    return fid_pass, is_pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired FID/IS verdict for static TeaCache mask replicas")
    parser.add_argument("probe_dir")
    parser.add_argument("--left", default="uniform")
    parser.add_argument(
        "--right", default="geometric",
        help="comparison arm ID, or 'all' for every discovered arm")
    parser.add_argument("--fid-margin", type=float, default=2.0)
    parser.add_argument("--is-margin", type=float, default=0.0)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--calc-tolerance", type=float, default=1.0)
    args = parser.parse_args(argv)
    if (args.fid_margin < 0 or args.is_margin < 0 or
            args.min_pairs < 2 or args.calc_tolerance < 0):
        parser.error("margins/tolerance must be non-negative and min-pairs >= 2")

    root = os.path.abspath(args.probe_dir)
    thresholds, threshold_warnings = discover_threshold_runs(root)
    print("== Paired static TeaCache mask verdict ==")
    print(f"  probe dir: {root}")
    _render_warnings(threshold_warnings)

    budgets = _budget_dirs(root)
    if not budgets:
        print("  INSUFFICIENT EVIDENCE: no k<K> budget directories")
        return 0

    all_pass = True
    any_complete = False
    comparator_all_pass = True
    comparator_pending = False
    for budget, budget_dir in budgets:
        print("")
        print(f"== budget k{budget} ==")
        arms, warnings = discover_arm_runs(budget_dir)
        _render_warnings(warnings)
        if args.left not in arms:
            print(f"  INSUFFICIENT EVIDENCE: need arm_{args.left}")
            all_pass = False
            continue
        right_names = (
            sorted(name for name in arms if name != args.left)
            if args.right == "all" else [args.right]
        )
        if not right_names:
            print("  INSUFFICIENT EVIDENCE: no comparison arms found")
            all_pass = False
            continue
        for right_name in right_names:
            if right_name not in arms:
                print(f"  INSUFFICIENT EVIDENCE: need arm_{right_name}")
                all_pass = False
                continue
            fid_pass, is_pass = _render_pair(
                args.left, right_name, arms[args.left], arms[right_name],
                args.fid_margin, args.is_margin, args.min_pairs)
            complete_pairs = len(
                set(arms[args.left]) & set(arms[right_name]))
            any_complete = any_complete or complete_pairs >= args.min_pairs
            all_pass = all_pass and fid_pass and is_pass

        if not thresholds:
            print("  plain TeaCache comparator: not run")
            continue
        label, mean_calc, distance = nearest_threshold(thresholds, budget)
        if label is None:
            print("  plain TeaCache comparator: no run has usable total_calc")
            comparator_pending = True
            continue
        assert mean_calc is not None and distance is not None
        print(f"  nearest plain TeaCache: thresh={label} "
              f"calc/trajectory={mean_calc:.2f} distance={distance:.2f}")
        if distance > args.calc_tolerance:
            print(f"    UNMATCHED: distance exceeds tolerance "
                  f"{args.calc_tolerance:.2f}; do not claim equal compute")
            comparator_pending = True
            continue
        threshold_runs = thresholds[label]
        paired_count = len(set(arms[args.left]) & set(threshold_runs))
        if paired_count < args.min_pairs:
            root_run = threshold_runs.get(0)
            if root_run is not None:
                print(f"    descriptive only: {args.left} FID="
                      f"{_fmt(arms[args.left][0].get('fid'), '.2f')} vs "
                      f"threshold FID={_fmt(root_run.get('fid'), '.2f')}; "
                      "replicate the matched threshold before inference")
            comparator_pending = True
            continue
        threshold_fid_pass, threshold_is_pass = _render_pair(
            args.left, f"threshold_{label}", arms[args.left], threshold_runs,
            args.fid_margin, args.is_margin, args.min_pairs)
        comparator_all_pass = (comparator_all_pass and threshold_fid_pass and
                               threshold_is_pass)

    print("")
    print("== RECOMMENDATION ==")
    if not any_complete:
        print(f"  INSUFFICIENT EVIDENCE: need >= {args.min_pairs} paired latent "
              "offsets for both arms.")
    elif not all_pass:
        print("  NO MASK WINNER: at least one paired mask gate failed. Keep the "
              "top masks tied or add evidence; do not resume the bandit.")
    elif thresholds and comparator_pending:
        print(f"  MASK WINNER: {args.left} passes the paired mask gates, but the "
              "equal-compute plain TeaCache comparator still needs paired runs.")
    elif thresholds and not comparator_all_pass:
        print("  NO STATIC WINNER: the selected mask did not pass the paired "
              "equal-compute plain TeaCache gates.")
    elif thresholds:
        print(f"  STATIC WINNER: {args.left} passes both mask and equal-compute "
              "plain TeaCache gates.")
    else:
        print(f"  MASK WINNER: {args.left} passes FID non-inferiority and IS "
              "superiority; no plain TeaCache comparator was supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
