#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a two-arm SpecA manifest from explicitly listed refresh steps.

The paired active-vs-uniform quality experiment needs masks that
``build_budget_manifest.py`` cannot express (its four layouts are generated,
not hand-picked). This script takes explicit step lists so the two arms differ
ONLY in where the risk-driven refreshes sit, holding the uniform skeleton,
refresh count, and max Taylor gap identical.

Every arm must satisfy:
  * mask[0] is True (SpecA requires the first step to refresh);
  * the longest Taylor gap <= --max-taylor-gap (numerical safety);
  * the refresh count == --refresh-count for both arms (equal FLOPs).

Usage:
  python scripts/build_paired_masks.py \
      --uniform-steps 0,2,5,10,15,20,25,30,35,40,45,47 \
      --learned-steps 0,4,5,6,10,15,20,25,30,35,40,45 \
      --num-steps 50 --max-taylor-gap 4 --version-key <RUNTIME_KEY> \
      --output /tmp/paired_k12.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from accelerators.covr_bandit import AccelerationStrategy, StrategyManifest


def _parse_steps(text: str) -> list[int]:
    try:
        steps = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    except ValueError:
        raise SystemExit(f"--steps must be comma-separated integers: {text!r}")
    return steps


def _validate(steps: Sequence[int], num_steps: int, refresh_count: int,
              max_taylor_gap: int) -> None:
    if not steps:
        raise SystemExit("step list must be non-empty")
    if steps[0] != 0:
        raise SystemExit(f"first step must refresh (got {steps[0]})")
    if steps[-1] >= num_steps:
        raise SystemExit(f"step {steps[-1]} exceeds num_steps={num_steps}")
    if len(steps) != refresh_count:
        raise SystemExit(
            f"refresh count {len(steps)} != --refresh-count {refresh_count}; "
            "arms must be equal-FLOPs")
    selected = set(steps)
    longest = run = 0
    for index in range(num_steps):
        run = 0 if index in selected else run + 1
        longest = max(longest, run)
    if longest > max_taylor_gap:
        raise SystemExit(
            f"longest Taylor gap {longest} exceeds --max-taylor-gap "
            f"{max_taylor_gap}; the mask is numerically unsafe for SpecA")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-arm equal-FLOPs SpecA manifest from explicit masks")
    parser.add_argument("--uniform-steps", required=True,
                        help="comma-separated refresh steps of the uniform arm")
    parser.add_argument("--learned-steps", required=True,
                        help="comma-separated refresh steps of the learned arm")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--refresh-count", type=int, required=True)
    parser.add_argument("--max-taylor-gap", type=int, required=True)
    parser.add_argument("--version-key", required=True,
                        help="COVR runtime version key (from the version probe)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    uniform_steps = _parse_steps(args.uniform_steps)
    learned_steps = _parse_steps(args.learned_steps)
    for name, steps in (("uniform", uniform_steps), ("learned", learned_steps)):
        _validate(steps, args.num_steps, args.refresh_count, args.max_taylor_gap)

    def mask(steps: Sequence[int]) -> list[bool]:
        selected = set(steps)
        return [index in selected for index in range(args.num_steps)]

    strategies = []
    for strategy_id, steps, source in (
            ("uniform_static", uniform_steps, "paired_uniform_control"),
            ("risk_learned", learned_steps, "paired_risk_learned")):
        strategies.append(AccelerationStrategy(
            strategy_id=strategy_id,
            method="speca",
            params={
                "refresh_mask": mask(steps),
                "refresh_count": args.refresh_count,
                "num_steps": args.num_steps,
            },
            modeled_flops=float(args.refresh_count),
            source=source,
        ))

    manifest = StrategyManifest(
        version_key=args.version_key,
        num_steps=args.num_steps,
        baseline_strategy_id="uniform_static",
        strategies=tuple(strategies),
        source_groups=("paired_masks",),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    manifest.save(args.output)
    print(f"uniform arm: {uniform_steps}")
    print(f"learned arm: {learned_steps}")
    print(f"wrote {len(strategies)} arms -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
