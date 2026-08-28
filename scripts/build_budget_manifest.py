#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesize equal-FLOPs refresh-mask manifests at an ARBITRARY budget.

``scripts/build_covr_manifest.py --method teacache-mask`` derives its masks
from an observed SpecA action-audit, so it can only emit masks whose refresh
cardinality the audit actually contains — otherwise
``experiments/covr_analysis.py`` raises "no source trajectory can satisfy the
template constraints at the target refresh cardinality". That makes the
aggressive-budget probe (push calc from 13/50 down to ~6-8/50 and re-measure
arm FID spread) impossible to set up from the existing audit.

This script builds the masks combinatorially instead: given ``--num-steps``
and ``--refresh-count`` it lays the calc steps out along the early<->late
allocation axis, which is the axis an offline probe of the existing audit
found per-image crossover on. No audit, no cardinality floor.

Every arm carries EXACTLY ``--refresh-count`` calc steps, so the equal-FLOPs
invariant the arm-spread verdict depends on holds by construction.
Use ``--random-count N --random-seed S`` to append N reproducible random-null
arms per budget. The legacy invocation still emits only the four deterministic
layouts and keeps ``uniform`` as the baseline.

Usage:
  python scripts/build_budget_manifest.py --num-steps 50 --refresh-count 8 \
      --output /tmp/budget_k8.json
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerators.covr_bandit import AccelerationStrategy, StrategyManifest


def _place(lo: int, hi: int, count: int) -> List[int]:
    """``count`` integer positions spread evenly across ``[lo, hi]``."""
    if count <= 0:
        return []
    if count == 1:
        return [int(round((lo + hi) / 2))]
    step = (hi - lo) / (count - 1)
    return [int(round(lo + i * step)) for i in range(count)]


def _fill_to(steps: List[int], k: int, num_steps: int) -> List[int]:
    """Top ``steps`` up to exactly ``k`` distinct entries.

    Rounding collisions in :func:`_place` can leave fewer than ``k`` calc
    steps. Missing ones are added greedily at the position farthest from any
    existing calc step, which preserves the layout's intent (and keeps the
    result deterministic).
    """
    chosen = sorted(set(s for s in steps if 0 <= s < num_steps))
    while len(chosen) > k:
        # Drop the entry whose removal costs the least coverage: the one with
        # the closest neighbour, scanning from the dense end.
        gaps = [(chosen[i + 1] - chosen[i - 1], i)
                for i in range(1, len(chosen) - 1)]
        chosen.pop(min(gaps)[1] if gaps else len(chosen) - 1)
    while len(chosen) < k:
        best, best_d = None, -1
        for cand in range(num_steps):
            if cand in chosen:
                continue
            d = min(abs(cand - c) for c in chosen) if chosen else num_steps
            if d > best_d:
                best, best_d = cand, d
        chosen.append(best)
        chosen.sort()
    return chosen


def _layouts(num_steps: int, k: int, prefix: int) -> List[Tuple[str, List[int]]]:
    """Four calc-step layouts spanning the early<->late allocation axis."""
    head = list(range(min(prefix, k)))          # teacache requires mask[0]
    rest = k - len(head)
    lo = len(head)
    hi = num_steps - 1
    mid = (lo + hi) // 2
    # Geometric: spacing grows, so calc thins out as denoising proceeds.
    geo: List[int] = []
    if rest > 0:
        span = hi - lo
        geo = [lo + int(round(span * (i / rest) ** 1.8))
               for i in range(1, rest + 1)]
    return [
        ("front_loaded", head + _place(lo, mid, rest)),
        ("uniform", head + _place(lo, hi, rest)),
        ("back_loaded", head + _place(mid, hi, rest)),
        ("geometric", head + geo),
    ]


def _random_layouts(
        num_steps: int, k: int, prefix: int, count: int, seed: int,
        ) -> List[Tuple[str, List[int]]]:
    """Return reproducible random masks with the same prefix and budget."""
    if count <= 0:
        return []
    prefix_steps = list(range(min(prefix, k)))
    available = list(range(len(prefix_steps), num_steps))
    rest = k - len(prefix_steps)
    if rest > len(available):
        raise ValueError("refresh_count leaves no room for random mask sampling")

    rng = random.Random(seed)
    layouts = []
    seen = set()
    max_unique = math.comb(len(available), rest)
    if count > max_unique:
        raise ValueError(
            f"random_count={count} exceeds {max_unique} unique masks for "
            f"num_steps={num_steps}, refresh_count={k}, prefix={prefix}")
    while len(layouts) < count:
        selected = tuple(sorted(prefix_steps + rng.sample(available, rest)))
        if selected in seen:
            continue
        seen.add(selected)
        layouts.append((f"random_{len(layouts):02d}", list(selected)))
    return layouts


def _max_taylor_gap(mask: Tuple[bool, ...]) -> int:
    worst = run = 0
    for flag in mask:
        run = 0 if flag else run + 1
        worst = max(worst, run)
    return worst


def _float_token(value: float) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def _parse_threshold_arm(value: str, parser) -> Tuple[float, float]:
    try:
        threshold_text, skip_rate_text = value.split(":", 1)
        threshold = float(threshold_text)
        expected_skip_rate = float(skip_rate_text)
    except (TypeError, ValueError):
        parser.error(
            "--threshold-arm must use THRESHOLD:EXPECTED_SKIP_RATE")
    if threshold <= 0.0:
        parser.error("threshold values must be positive")
    if not 0.0 <= expected_skip_rate < 1.0:
        parser.error("expected skip rates must be in [0, 1)")
    return threshold, expected_skip_rate


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build TeaCache threshold and/or fixed-mask strategy arms")
    ap.add_argument("--output", required=True, help="manifest json path")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--refresh-count", type=int, default=None,
                    help="legacy single fixed-mask calc budget")
    ap.add_argument("--refresh-counts", type=int, nargs="*", default=[],
                    help="fixed-mask calc budgets; emits four patterns per budget")
    ap.add_argument("--random-count", type=int, default=0,
                    help="deterministic random masks per fixed budget; disabled "
                         "by default")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="seed for deterministic random masks")
    ap.add_argument(
        "--threshold-arm", action="append", default=[],
        metavar="THRESHOLD:EXPECTED_SKIP_RATE",
        help="dynamic TeaCache arm and its calibrated expected skip rate; "
             "repeat for multiple thresholds")
    ap.add_argument("--method", default="teacache",
                    help="registered accelerator method for every arm")
    ap.add_argument("--mandatory-prefix", type=int, default=3,
                    help="leading steps forced to calc (mask[0] is required)")
    ap.add_argument("--max-taylor-gap", type=int, default=None,
                    help="reject arms exceeding this cache gap (default: "
                         "report only — the constraint is often infeasible "
                         "at aggressive budgets)")
    ap.add_argument(
        "--baseline-arm", default=None,
        help="conservative baseline arm (default: uniform for legacy masks, "
             "threshold 0.25 when present, otherwise the first arm)")
    ap.add_argument("--version-key", default=None)
    args = ap.parse_args()

    ns = args.num_steps
    refresh_counts = []
    if args.refresh_count is not None:
        refresh_counts.append(args.refresh_count)
    refresh_counts.extend(args.refresh_counts)
    refresh_counts = list(dict.fromkeys(refresh_counts))
    threshold_arms = [
        _parse_threshold_arm(value, ap) for value in args.threshold_arm
    ]
    if not refresh_counts and not threshold_arms:
        ap.error("provide --refresh-count/--refresh-counts or --threshold-arm")
    if threshold_arms and args.method != "teacache":
        ap.error("--threshold-arm requires --method teacache")
    if args.random_count < 0:
        ap.error("--random-count must be non-negative")
    for k in refresh_counts:
        if not 1 <= k <= ns:
            ap.error(f"refresh counts must be in [1, {ns}]")

    strategies: List[AccelerationStrategy] = []
    for threshold, expected_skip_rate in threshold_arms:
        arm_id = f"threshold_{_float_token(threshold)}"
        expected_calc_count = max(2.0, ns * (1.0 - expected_skip_rate))
        print(
            f"dynamic arm {arm_id:<18} threshold={threshold:g}  "
            f"expected_skip={expected_skip_rate:.3f}  "
            f"expected_calc={expected_calc_count:.2f}/{ns}")
        strategies.append(AccelerationStrategy(
            strategy_id=arm_id,
            method="teacache",
            params={
                "rel_l1_thresh": threshold,
                "expected_skip_rate": expected_skip_rate,
                "num_steps": ns,
            },
            modeled_flops=expected_calc_count,
            source="teacache_threshold_manifest",
        ))

    mixed_ids = (
        bool(threshold_arms) or len(refresh_counts) > 1
        or args.random_count > 0
    )
    for k in refresh_counts:
        print(f"fixed-mask arms at calc={k}/{ns}, method={args.method}:")
        for name, steps in _layouts(ns, k, args.mandatory_prefix):
            calc = _fill_to(steps, k, ns)
            mask = tuple(i in set(calc) for i in range(ns))
            assert sum(mask) == k and mask[0], (name, sum(mask), mask[0])
            gap = _max_taylor_gap(mask)
            arm_id = f"pattern_{name}_k{k}" if mixed_ids else name
            if args.max_taylor_gap is not None and gap > args.max_taylor_gap:
                print(f"  {arm_id:<24} SKIPPED (max cache gap {gap} > "
                      f"{args.max_taylor_gap})")
                continue
            print(f"  {arm_id:<24} max_gap={gap:>2}  calc at {calc}")
            strategies.append(AccelerationStrategy(
                strategy_id=arm_id,
                method=args.method,
                params={
                    "refresh_mask": list(mask),
                    "refresh_count": k,
                    "num_steps": ns,
                },
                modeled_flops=float(k),
                source=f"budget_manifest_k{k}",
            ))
        for name, steps in _random_layouts(
                ns, k, args.mandatory_prefix, args.random_count,
                args.random_seed + k):
            calc = _fill_to(steps, k, ns)
            mask = tuple(i in set(calc) for i in range(ns))
            assert sum(mask) == k and mask[0], (name, sum(mask), mask[0])
            gap = _max_taylor_gap(mask)
            arm_id = f"pattern_{name}_k{k}" if mixed_ids else name
            if args.max_taylor_gap is not None and gap > args.max_taylor_gap:
                print(f"  {arm_id:<24} SKIPPED (max cache gap {gap} > "
                      f"{args.max_taylor_gap})")
                continue
            print(f"  {arm_id:<24} max_gap={gap:>2}  calc at {calc}")
            strategies.append(AccelerationStrategy(
                strategy_id=arm_id,
                method=args.method,
                params={
                    "refresh_mask": list(mask),
                    "refresh_count": k,
                    "num_steps": ns,
                    "random_seed": args.random_seed + k,
                },
                modeled_flops=float(k),
                source=f"budget_random_null_k{k}",
            ))

    ids = [s.strategy_id for s in strategies]
    if not ids:
        print("[FATAL] every arm was rejected — relax --max-taylor-gap",
              file=sys.stderr)
        return 1
    if args.baseline_arm is not None:
        if args.baseline_arm not in ids:
            ap.error(f"--baseline-arm is not present: {args.baseline_arm}")
        baseline = args.baseline_arm
    elif "threshold_0p25" in ids:
        baseline = "threshold_0p25"
    elif args.random_count > 0:
        uniform_ids = sorted(
            strategy_id for strategy_id in ids
            if strategy_id.startswith("pattern_uniform_k"))
        if not uniform_ids:
            raise RuntimeError("expected a uniform baseline arm")
        baseline = uniform_ids[0]
    elif "uniform" in ids:
        baseline = "uniform"
    else:
        baseline = ids[0]

    default_version = f"strategy-n{ns}-{args.method}"
    source_groups = [f"synthetic_budget_k{k}" for k in refresh_counts]
    if threshold_arms:
        source_groups.insert(0, "teacache_thresholds")
    manifest = StrategyManifest(
        version_key=args.version_key or default_version,
        num_steps=ns,
        baseline_strategy_id=baseline,
        strategies=tuple(strategies),
        source_groups=tuple(source_groups),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    manifest.save(args.output)
    print(f"\nbaseline arm: {baseline}")
    print(f"wrote {len(strategies)} arms -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
