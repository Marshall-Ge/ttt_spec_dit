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

Usage:
  python scripts/build_budget_manifest.py --num-steps 50 --refresh-count 8 \
      --output /tmp/budget_k8.json
"""

from __future__ import annotations

import argparse
import os
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


def _max_taylor_gap(mask: Tuple[bool, ...]) -> int:
    worst = run = 0
    for flag in mask:
        run = 0 if flag else run + 1
        worst = max(worst, run)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build an equal-FLOPs mask manifest at any refresh budget")
    ap.add_argument("--output", required=True, help="manifest json path")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--refresh-count", type=int, required=True,
                    help="calc steps per trajectory (identical across arms)")
    ap.add_argument("--method", default="teacache",
                    help="registered accelerator method for every arm")
    ap.add_argument("--mandatory-prefix", type=int, default=3,
                    help="leading steps forced to calc (mask[0] is required)")
    ap.add_argument("--max-taylor-gap", type=int, default=None,
                    help="reject arms exceeding this cache gap (default: "
                         "report only — the constraint is often infeasible "
                         "at aggressive budgets)")
    ap.add_argument("--baseline-arm", default="uniform",
                    help="arm id used as the bandit's conservative baseline")
    ap.add_argument("--version-key", default=None)
    args = ap.parse_args()

    ns, k = args.num_steps, args.refresh_count
    if not 1 <= k <= ns:
        ap.error(f"--refresh-count must be in [1, {ns}]")

    strategies: List[AccelerationStrategy] = []
    print(f"equal-FLOPs arms at calc={k}/{ns}, method={args.method}:")
    for name, steps in _layouts(ns, k, args.mandatory_prefix):
        calc = _fill_to(steps, k, ns)
        mask = tuple(i in set(calc) for i in range(ns))
        assert sum(mask) == k and mask[0], (name, sum(mask), mask[0])
        gap = _max_taylor_gap(mask)
        if args.max_taylor_gap is not None and gap > args.max_taylor_gap:
            print(f"  {name:<14} SKIPPED (max cache gap {gap} > "
                  f"{args.max_taylor_gap})")
            continue
        print(f"  {name:<14} max_gap={gap:>2}  calc at {calc}")
        strategies.append(AccelerationStrategy(
            strategy_id=name,
            method=args.method,
            params={"refresh_mask": list(mask), "num_steps": ns},
            modeled_flops=float(k),
            source=f"budget_manifest_k{k}",
        ))

    ids = [s.strategy_id for s in strategies]
    if not ids:
        print("[FATAL] every arm was rejected — relax --max-taylor-gap",
              file=sys.stderr)
        return 1
    baseline = args.baseline_arm if args.baseline_arm in ids else ids[0]
    manifest = StrategyManifest(
        version_key=args.version_key or f"budget-k{k}-n{ns}-{args.method}",
        num_steps=ns,
        baseline_strategy_id=baseline,
        strategies=tuple(strategies),
        source_groups=(f"synthetic_budget_k{k}",),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    manifest.save(args.output)
    print(f"\nbaseline arm: {baseline}")
    print(f"wrote {len(strategies)} arms -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
