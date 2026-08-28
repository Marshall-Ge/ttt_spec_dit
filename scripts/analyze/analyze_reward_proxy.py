#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-proxy verdict for the COVR forced single-arm sweep.

Question: does the bandit's cheap reward rank the equal-FLOPs arms the same
way FID does? ``scripts/sweep_budget_probe.sh`` now runs every forced arm
with sentinel telemetry on, so each arm's results.json carries both a reward
estimate (``aggregate.covr_reward_telemetry``) and an FID truth
(``aggregate.fid``).

Design — why stratified, and why a within-budget permutation test:
  * each budget has 4 arms -> 4 points. A Spearman p from an asymptotic
    formula on 4 points is meaningless (4! = 24 arrangements, best possible
    p = 1/24 ~ 0.042), so we never print one.
  * budgets differ hugely in FID magnitude (budget main effect), so we rank
    WITHIN each budget and combine ranks across budgets. The null shuffles
    arm labels ONLY inside each budget (24 per budget; 24^3 = 13824 for the
    default 3-budget probe), which keeps the budget main effect out of the
    statistic entirely.

Two rewards, mutually exclusive per run (run_dit.py: the terminal block
requires ``covr_sentinel_start_idx is None``; H-step requires it not None):
  * terminal (horizon 0): mean MSE at the last step vs a full shadow pass.
  * h-step   (horizon > 0): mean transition-defect numerator over an H-step
    full rollout from a hashed start.
Both are LOSSES (smaller = better), same direction as FID.

Structural-degeneration check (terminal mode only): if an arm's mask calc's
the FINAL step, the accelerated last step IS the full computation, so the
shadow-full comparison is structurally identical and the loss is ~0 by
construction. Arms like that tie at zero and a bandit can only pick among
them arbitrarily — the hypothesized explanation for the 92.1% single-arm
convergence. This script verifies it directly from the manifest masks, and
only reports the correlation story on top of non-degenerate evidence.

Expected layout (written by ``scripts/sweep_budget_probe.sh``)::

    <out>/k<K>/manifest.json
    <out>/k<K>/equalflops/arm_<id>/results.json
    <out>/k<K>/equalflops/noise_<seed>/results.json   (ignored here)

Analysis only; always exits 0. The RECOMMENDATION block carries the
reasoning, not just a label.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analyze.analyze_teacache_sweeps import _load          # noqa: E402
from scripts.analyze.analyze_budget_probe import _budget_dirs, _f  # noqa: E402

# Terminal and H-step reward fields in aggregate.covr_reward_telemetry.
_REWARD_FIELDS = {
    "terminal": ("terminal_fidelity_loss_mean", "terminal_fidelity_loss_n",
                 "terminal_fidelity_loss_std"),
    "hstep": ("h_step_numerator_mean", "h_step_numerator_n",
              "h_step_denominator_mean"),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _manifest_info(manifest_path: str) -> Optional[Dict[str, object]]:
    """(arm_id -> refresh_mask) plus num_steps, straight from the JSON."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    masks = {}
    for strategy in manifest.get("strategies", []):
        params = strategy.get("params", {})
        mask = params.get("refresh_mask")
        if mask is not None:
            masks[str(strategy["strategy_id"])] = tuple(bool(v) for v in mask)
    return {
        "num_steps": int(manifest.get("num_steps", 0)),
        "masks": masks,
    }


def _arm_row(results: dict, num_steps_hint: int) -> Dict[str, object]:
    """Extract one arm's FID + reward telemetry + session info."""
    agg = results.get("aggregate", {}) or {}
    cfg = results.get("config", {}) or {}
    tele = agg.get("covr_reward_telemetry") or {}
    horizon = tele.get("sentinel_horizon")
    if horizon is None:
        horizon = int(cfg.get("covr_sentinel_horizon", 0))
    mode = "hstep" if horizon and horizon > 0 else "terminal"
    mean_key, n_key, aux_key = _REWARD_FIELDS[mode]
    row: Dict[str, object] = {
        "mode": mode,
        "horizon": horizon,
        "fid": agg.get("fid"),
        "reward": tele.get(mean_key),
        "reward_n": tele.get(n_key),
        "aux": tele.get(aux_key),
        "sentinel_count": tele.get("sentinel_count"),
        "sentinel_skipped": tele.get("sentinel_skipped"),
        "sentinel_rate": tele.get("sentinel_rate"),
        "session_id": cfg.get("covr_session_id"),
        "num_steps": int(cfg.get("num_steps") or num_steps_hint or 0),
        "forced_id": tele.get("forced_strategy_id"),
        "label": None,
        "last_calc": None,
        "cache_distance": None,
        "mask_final_calc": False,
    }
    return row


def _mask_stats(mask: Tuple[bool, ...], num_steps: int
                ) -> Tuple[Optional[int], Optional[int]]:
    """(last calc step index, cache distance to the final step)."""
    last_calc = -1
    for index, flag in enumerate(mask):
        if flag:
            last_calc = index
    if last_calc < 0:
        return None, None
    return last_calc, (num_steps - 1) - last_calc


def _budget_rows(budget_dir: str, num_steps_hint: int
                 ) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    """Load all arm_*/results.json rows under budget_dir; list of problems."""
    rows: Dict[str, Dict[str, object]] = {}
    problems: List[str] = []
    for path in sorted(glob.glob(os.path.join(
            budget_dir, "equalflops", "arm_*", "results.json"))):
        label = os.path.basename(os.path.dirname(path))[len("arm_"):]
        results = _load(path)
        if results is None:
            problems.append(f"{label}: unreadable results.json")
            continue
        row = _arm_row(results, num_steps_hint)
        row["label"] = label
        rows[label] = row
    return rows, problems


# --------------------------------------------------------------------------
# Ranking / permutation machinery
# --------------------------------------------------------------------------

def _average_ranks(values: List[float]) -> List[float]:
    """Rank (1 = best/smallest) with average ranks for ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while (j + 1 < len(order)
               and values[order[j + 1]] == values[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> float:
    """Spearman rho over average ranks (0.0 if either side has no spread)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra, rb = _average_ranks(a), _average_ranks(b)
    mean = (len(a) + 1) / 2.0
    va = sum((x - mean) ** 2 for x in ra)
    vb = sum((x - mean) ** 2 for x in rb)
    if va == 0.0 or vb == 0.0:
        return 0.0
    cov = sum((x - mean) * (y - mean) for x, y in zip(ra, rb))
    return cov / math.sqrt(va * vb)


def _permutation_p(fid_lists: List[List[float]],
                   reward_lists: List[List[float]],
                   max_perms: Optional[int],
                   perm_seed: int) -> Tuple[float, str]:
    """One-sided p for reward-vs-FID rank concordance.

    Null: within each budget, shuffle the observed reward values across the
    arm labels (each budget contributes 4! = 24 assignments; independent
    across budgets). Statistic: sum of per-budget Spearman rho.
    """
    buckets = []
    observed = 0.0
    for fid, reward in zip(fid_lists, reward_lists):
        observed += _spearman(fid, reward)
        perms = [tuple(reward[i] for i in permutation)
                 for permutation in itertools.permutations(range(len(reward)))]
        buckets.append([(p, _spearman(fid, list(p))) for p in perms])
    total = 1
    for bucket in buckets:
        total *= len(bucket)
    if max_perms is not None and total > max_perms:
        rng = random.Random(perm_seed)
        count, ge = 0, 0
        for _ in range(max_perms):
            stat = sum(rng.choice(bucket)[1] for bucket in buckets)
            count += 1
            if stat >= observed - 1e-12:
                ge += 1
        return ge / count, f"sampled {count} of {total} permutations (seed {perm_seed})"
    ge = 0
    for combo in itertools.product(*buckets):
        stat = sum(contribution for _, contribution in combo)
        if stat >= observed - 1e-12:
            ge += 1
    return ge / total, f"enumerated all {total} permutations"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _print_arm_table(rows: List[Dict[str, object]]) -> None:
    print(f"    {'arm':<14} {'FID':>7} {'FID_r':>5} {'reward':>9} "
          f"{'r_n':>4} {'rew_r':>5} {'lastCalc':>8} {'dist':>4} "
          f"{'sent(skip)':>11}")
    for row in rows:
        reward_s = (_f(row["reward"], ".4g") if row["reward"] is not None
                    else "n/a")
        print(f"    {str(row['label']):<14} {_f(row['fid'], '.2f'):>7} "
              f"{_f(row['fid_rank'], '.0f'):>5} {reward_s:>9} "
              f"{_f(row['reward_n'], '.0f'):>4} "
              f"{_f(row['reward_rank'], '.1f'):>5} "
              f"{_f(row['last_calc'], 'd'):>8} "
              f"{_f(row['cache_distance'], 'd'):>4} "
              f"{int(row['sentinel_count'] or 0)}"
              f"({int(row['sentinel_skipped'] or 0)})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reward-vs-FID ranking verdict (COVR hypothesis C)")
    ap.add_argument("probe_dir", help="root containing k<K>/ subdirs")
    ap.add_argument("--abs-zero-tol", type=float, default=1e-6,
                    help="absolute threshold for 'reward ~ 0' (degenerate)")
    ap.add_argument("--min-reward-n", type=int, default=2,
                    help="reward observations per arm below this = thin "
                         "evidence")
    ap.add_argument("--max-perm", type=int, default=100000,
                    help="cap for the permutation null (sampled beyond)")
    ap.add_argument("--perm-seed", type=int, default=0)
    args = ap.parse_args()

    budgets = _budget_dirs(args.probe_dir)
    if not budgets:
        print(f"no k<K>/ budget dirs under {args.probe_dir}")
        return 0

    print("== COVR reward proxy: does the reward rank arms like FID? ==")
    print(f"  budgets found: {', '.join(f'k{k}' for k, _ in budgets)}")
    print(f"  abs-zero tolerance for 'reward ~ 0': {args.abs_zero_tol:g}")

    budget_arms: List[Tuple[int, List[Dict[str, object]]]] = []
    unusable: List[Tuple[int, List[str]]] = []
    excluded: List[Tuple[int, str]] = []
    modes_mixed = False

    for budget, budget_dir in budgets:
        manifest = _manifest_info(os.path.join(budget_dir, "manifest.json"))
        num_steps = int(manifest["num_steps"]) if manifest else 0
        rows, problems = _budget_rows(budget_dir, num_steps)
        if manifest is None or not manifest["masks"]:
            excluded.append((budget, "manifest.json missing or has no masks"))
            continue
        for label, row in rows.items():
            row["budget"] = budget
            mask = manifest["masks"].get(label)
            if mask is None:
                problems.append(f"{label}: no refresh_mask in manifest")
                continue
            row["last_calc"], row["cache_distance"] = _mask_stats(
                mask, num_steps)
            row["mask_final_calc"] = bool(mask[-1]) if mask else False

        print("")
        print(f"== budget K={budget}/{num_steps}: per-arm FID vs reward ==")
        if not rows:
            print("  (no forced-arm runs found)")
            unusable.append((budget, ["no forced-arm runs"]))
            continue
        modes = {row["mode"] for row in rows.values()}
        if len(modes) > 1:
            modes_mixed = True
        ordered = sorted(rows.values(), key=lambda r: r["label"])
        for row in ordered:
            print(f"  {row['label']}: mode={row['mode']} "
                  f"horizon={row['horizon']} "
                  f"FID={_f(row['fid'], '.2f')} "
                  f"reward={_f(row['reward'], '.4g')} "
                  f"n={_f(row['reward_n'], 'd')} "
                  f"lastCalc={_f(row['last_calc'], 'd')} "
                  f"distToEnd={_f(row['cache_distance'], 'd')}")
        # Session consistency: sentinel hashing keys off session id.
        sessions = {row["session_id"] for row in rows.values()
                    if row["session_id"] is not None}
        if len(sessions) > 1:
            problems.append(
                f"arms do NOT share a --covr-session-id "
                f"({sorted(sessions)}); each arm sampled different sentinel "
                f"trajectories — the cross-arm reward comparison is "
                f"confounded")
        elif len(sessions) == 1:
            print(f"  session id shared by all arms: {sessions.pop()}")
        if problems:
            print("  PROBLEMS:")
            for problem in problems:
                print(f"    - {problem}")
        # Usability: every manifest arm present, with reward + FID values.
        missing = [label for label in manifest["masks"] if label not in rows]
        if missing:
            unusable.append((budget, [f"missing arms: {sorted(missing)}"]))
            continue
        without_reward = [label for label, row in rows.items()
                          if row["reward"] is None or row["reward_n"] is None
                          or int(row["reward_n"]) < 1
                          or row["fid"] is None
                          or (isinstance(row["fid"], float)
                              and math.isnan(row["fid"]))]
        if without_reward:
            unusable.append((budget, [
                f"arms without usable reward/FID: {sorted(without_reward)}"]))
            continue
        if len(sessions) > 1:
            excluded.append((budget, "arms did not share a session id"))
            continue
        budget_arms.append((budget, ordered))

    # ------------------------------------------------------------------
    # Rank concordance over usable budgets
    # ------------------------------------------------------------------
    print("")
    print("== combined stratified rank concordance ==")
    if modes_mixed:
        print("  WARNING: terminal and hstep rewards coexist across arms — "
              "ranks mix two different reward definitions")
    p_value: Optional[float] = None
    observed: Optional[float] = None
    if not budget_arms:
        print("  no usable budget (every budget lacks all-arm reward data) — "
              "nothing to rank")
    else:
        fid_lists: List[List[float]] = []
        reward_lists: List[List[float]] = []
        print(f"  {'calc':>5} {'rho':>7} {'FID_r == rew_r':>16}")
        for budget, rows in budget_arms:
            fids = [float(row["fid"]) for row in rows]
            rewards = [float(row["reward"]) for row in rows]
            fid_lists.append(fids)
            reward_lists.append(rewards)
            fid_ranks = _average_ranks(fids)
            reward_ranks = _average_ranks(rewards)
            for row, fr, rr in zip(rows, fid_ranks, reward_ranks):
                row["fid_rank"], row["reward_rank"] = fr, rr
            exact = sum(1 for fr, rr in zip(fid_ranks, reward_ranks)
                        if fr == rr)
            print(f"  {budget:>5} {_spearman(fids, rewards):+7.3f} "
                  f"{exact}/{len(rows):>6}")
            _print_arm_table(rows)
        observed = sum(_spearman(f, r)
                       for f, r in zip(fid_lists, reward_lists))
        p_value, how = _permutation_p(
            fid_lists, reward_lists, args.max_perm, args.perm_seed)
        print(f"  observed statistic (sum of per-budget Spearman rho): "
              f"{observed:+.3f}")
        print(f"  null: arm labels shuffled within each budget only ({how})")
        print(f"  p (one-sided, S_perm >= S_obs): {p_value:.6g}")

    # ------------------------------------------------------------------
    # Structural degeneration of the terminal reward
    # ------------------------------------------------------------------
    print("")
    print("== structural degeneration of the terminal reward ==")
    terminal_rows = [row for _, rows in budget_arms for row in rows
                     if row["mode"] == "terminal"]
    if not terminal_rows:
        print("  (no terminal-mode arms with reward telemetry in this tree — "
              "only hstep runs were found)")
    else:
        zero_arms = [row for row in terminal_rows
                     if float(row["reward"]) <= args.abs_zero_tol]
        print(f"  arms with reward ~ 0 (|mean| <= {args.abs_zero_tol:g}):")
        for row in sorted(terminal_rows,
                          key=lambda r: (int(r["budget"]), str(r["label"]))):
            degenerate = float(row["reward"]) <= args.abs_zero_tol
            print(f"    {str(row['label']):<14} k{row['budget']:>2} "
                  f"lastCalc={_f(row['last_calc'], 'd'):>3} "
                  f"distToEnd={_f(row['cache_distance'], 'd'):>3} "
                  f"termLoss={_f(row['reward'], '.3e'):>11} "
                  f"{'DEGENERATE' if degenerate else 'nonzero'}")
        # Compare per (arm, budget) pair — the same label can calc the final
        # step in one budget and not in another (e.g. back_loaded at K=8 vs K=4).
        final_pairs = {(str(r["label"]), int(r["budget"]))
                       for r in terminal_rows if r.get("mask_final_calc")}
        zero_pairs = {(str(r["label"]), int(r["budget"])) for r in zero_arms}
        pred_miss = sorted(final_pairs - zero_pairs)
        obs_extra = sorted(zero_pairs - final_pairs)
        print(f"  (arm, budget) pairs whose mask calc's the FINAL step: "
              f"{sorted(final_pairs) or '(none)'}")
        print(f"  (arm, budget) pairs with reward ~ 0: "
              f"{sorted(zero_pairs) or '(none)'}")
        if final_pairs and zero_pairs == final_pairs:
            print("  MATCH: the degenerate pairs are exactly the final-step-calc "
                  "pairs. The terminal reward is structurally zero for them.")
        else:
            if pred_miss:
                print(f"  ANOMALY: final-step-calc pairs with NONZERO reward: "
                      f"{pred_miss} — the structural-zero prediction fails "
                      f"there")
            if obs_extra:
                print(f"  ANOMALY: zero-reward pairs that do NOT calc the "
                      f"final step: {obs_extra} — degeneracy has another "
                      f"source")

    # ------------------------------------------------------------------
    # Evidence sufficiency
    # ------------------------------------------------------------------
    print("")
    print("== evidence sufficiency ==")
    all_reward_rows = [row for _, rows in budget_arms for row in rows]
    thin: List[Dict[str, object]] = []
    if unusable:
        for budget, problems in unusable:
            print(f"  k{budget}: unusable — {'; '.join(problems)}")
    if excluded:
        for budget, reason in excluded:
            print(f"  k{budget}: excluded — {reason}")
    if not all_reward_rows:
        print("  (no budget produced all-arm reward data)")
    else:
        ns = [int(row["reward_n"]) for row in all_reward_rows]
        skips = [int(row["sentinel_skipped"] or 0)
                 for row in all_reward_rows]
        counts = [int(row["sentinel_count"] or 0)
                  for row in all_reward_rows]
        print(f"  reward n per arm: min {min(ns)}, max {max(ns)} "
              f"(--min-reward-n {args.min_reward_n})")
        if any(skips):
            pairs = ", ".join(f"k{row['budget']}/{row['label']}={s}"
                              for row, s in zip(all_reward_rows, skips))
            print(f"  sentinel_skipped: {{{pairs}}} — selected sentinels "
                  f"that produced no reward were dropped, shrinking n")
        else:
            print("  sentinel_skipped: 0 everywhere")
        if len(set(counts)) > 1:
            print(f"  sentinel_count differs across arms ({sorted(set(counts))}"
                  f") — unexpected with a shared session id")
        thin = [row for row in all_reward_rows
                if int(row["reward_n"]) < args.min_reward_n]
        if thin:
            thin_labels = sorted(f"k{row['budget']}/{row['label']}"
                                 for row in thin)
            print(f"  THIN EVIDENCE: {len(thin)} arm(s) below min n "
                  f"({args.min_reward_n}): {thin_labels}")

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    print("")
    print("== RECOMMENDATION ==")
    usable_ok = [t for t in budget_arms
                 if all(int(r["reward_n"]) >= args.min_reward_n
                        and not int(r["sentinel_skipped"] or 0)
                        for r in t[1])]
    if not budget_arms:
        print("  INSUFFICIENT EVIDENCE: no budget has all-arm reward data. "
              "Check that the sweep ran with --covr-sentinel-rate > 0 and "
              "that aggregate.covr_reward_telemetry landed in each "
              "results.json (forced mode never falls back to a full-baseline "
              "comparison — missing rewards show up as sentinel_skipped).")
    elif not usable_ok or thin:
        print("  EVIDENCE THIN: per-arm reward means rest on fewer than "
              f"{args.min_reward_n} observations (or selected sentinels were "
              "skipped), so no ranking conclusion is trustworthy. The "
              "numbers above are informational only. Re-run with "
              "SENTINEL_RATE=1.0 (and check that sentinel_skipped stays 0) "
              "before believing any proxy verdict.")
    elif any(row["mode"] == "terminal"
             for _, rows in budget_arms for row in rows):
        terminal_arms = [row for _, rows in budget_arms for row in rows
                         if row["mode"] == "terminal"]
        terminal_zero = [row for row in terminal_arms
                         if float(row["reward"]) <= args.abs_zero_tol]
        final_pairs = {(str(r["label"]), int(r["budget"]))
                       for r in terminal_arms if r.get("mask_final_calc")}
        zero_pairs = {(str(r["label"]), int(r["budget"]))
                      for r in terminal_zero}
        if final_pairs and zero_pairs == final_pairs:
            print(f"  TERMINAL REWARD IS STRUCTURALLY DEGENERATE: "
                  f"{len(terminal_zero)}/{len(terminal_arms)} terminal arms "
                  f"have mean fidelity loss ~ 0, and they are exactly the "
                  f"(arm, budget) pairs whose mask calc's the final step "
                  f"(distToEnd 0). For those arms the accelerated last step "
                  f"IS the full computation, so the shadow-full comparison "
                  f"is identity — MSE ~ 0 by construction. A bandit over "
                  f"these arms sees a tie at zero and can only pick "
                  f"arbitrarily; that is precisely the 92.1% single-arm "
                  f"convergence. The defect is in the REWARD (structure), "
                  f"not in the bandit algorithm. Terminal reward is "
                  f"unusable for mask arms; H-step is the reward to test.")
            if p_value is not None and p_value <= 0.05:
                print(f"  (The remaining non-degenerate arms do rank like FID "
                      f"at p={p_value:.4g}, but the zero ties still dominate "
                      f"bandit selection among the degenerate arms.)")
        elif final_pairs and not terminal_zero:
            print(f"  Structural hypothesis REJECTED here: (arm, budget) "
                  f"pairs calc the final step ({sorted(final_pairs)}) but "
                  f"no terminal loss is ~ 0 — the degeneracy prediction "
                  f"fails on this tree. Correlation verdict below applies "
                  f"as-is.")
        else:
            print(f"  Terminal rewards are NOT degenerate-only here"
                  + (f" (stratified p={p_value:.4g})" if p_value is not None
                     else "") + "; see the per-arm table.")
    else:
        assert p_value is not None
        # Rewards with no spread (all identical => all average-rank ties)
        # have zero discriminating power — that is not "disagreement", it is
        # a dead reward.
        spreads = [max(r) - min(r) for r in reward_lists]
        dead = all(s <= args.abs_zero_tol for s in spreads)
        if dead:
            print(f"  REWARD HAS NO DISCRIMINATING POWER: every arm's reward "
                  f"mean is identical (within {args.abs_zero_tol:g}) on this "
                  f"tree, so it cannot rank arms at all (Spearman p=1 by "
                  f"construction). For terminal mode, check the structural "
                  f"degeneration section; for hstep mode, this means the "
                  f"reward formula itself is not reacting to the arm.")
        elif p_value <= 0.05:
            print(f"  REWARD RANKS ARMS LIKE FID (stratified Spearman "
                  f"p={p_value:.4g}): the cheap reward is usable as a proxy "
                  f"for quality. A bandit trained on it should select the "
                  f"arm FID would pick — proceed to a real bandit run on "
                  f"this reward.")
        else:
            print(f"  REWARD DOES NOT RANK ARMS LIKE FID (stratified "
                  f"p={p_value:.4g}): the proxy and quality disagree on this "
                  f"tree. Do not let the bandit select arms on this reward "
                  f"until the reward formula is revisited.")
    print("  (All conclusions are stratified within budgets; FID magnitude "
          "differences across budgets do not enter the statistic.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
