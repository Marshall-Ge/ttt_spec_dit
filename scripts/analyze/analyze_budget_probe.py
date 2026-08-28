#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-budget verdict for the hypothesis-A aggressive-budget probe.

``scripts/analyze_teacache_sweeps.py`` renders the arm-spread-vs-noise-floor
verdict for ONE budget. This script stacks those per-budget verdicts so the
actual question gets a single answer:

    Is there ANY calc budget at which equal-FLOPs mask arms separate — i.e. at
    which WHERE the calc steps sit matters more than noise?

It reuses ``analyze_teacache_sweeps``'s loaders and ``_spread`` so the floor
definition and the ``> 2x floor`` criterion stay identical to the framework the
already-falsified 13/50 experiment was judged by.

Expected layout (written by ``scripts/sweep_budget_probe.sh``)::

    <out>/k<K>/equalflops/arm_<id>/results.json
    <out>/k<K>/equalflops/noise_<seed>/results.json

Analysis only; always exits 0.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analyze.analyze_teacache_sweeps import _spread, analyze_equalflops_sweep


def _budget_dirs(root: str) -> List[Tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(root, "k*")):
        match = re.fullmatch(r"k(\d+)", os.path.basename(path))
        if match and os.path.isdir(path):
            out.append((int(match.group(1)), path))
    return sorted(out, reverse=True)   # loosest budget first


def _f(value: Optional[float], spec: str = ".2f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-budget arm-spread verdict (COVR hypothesis A)")
    ap.add_argument("probe_dir", help="root containing k<K>/ subdirs")
    args = ap.parse_args()

    budgets = _budget_dirs(args.probe_dir)
    if not budgets:
        print(f"no k<K>/ budget dirs under {args.probe_dir}")
        return 0

    print("== [A] Equal-FLOPs arm spread vs noise floor, by calc budget ==")
    print(f"  {'calc':>5} {'arms':>5} {'FID range':>16} {'spread':>7} "
          f"{'floor':>7} {'ratio':>6}  verdict")

    verdicts = []
    for budget, path in budgets:
        arm_rows, noise_rows, _ = analyze_equalflops_sweep(
            os.path.join(path, "equalflops"))
        arm_spread = _spread(arm_rows, "fid")
        floor = _spread(noise_rows, "fid")
        floor_measured = floor is not None
        if not floor_measured:
            floor = 2.0
        fids = [r["fid"] for r in arm_rows
                if r.get("fid") is not None and not math.isnan(r["fid"])]
        rng = (f"{min(fids):.2f}-{max(fids):.2f}" if fids else "n/a")
        ratio = (arm_spread / floor
                 if arm_spread is not None and floor > 0 else None)
        if arm_spread is None:
            tag = "inconclusive (<2 arms)"
        elif ratio is not None and ratio > 2.0:
            tag = "DIFFER"
        else:
            tag = "tie"
        verdicts.append((budget, tag, arm_spread, floor, ratio,
                         floor_measured, arm_rows))
        print(f"  {budget:>5} {len(arm_rows):>5} {rng:>16} "
              f"{_f(arm_spread):>7} {_f(floor):>7} {_f(ratio, '.2f'):>6}  {tag}"
              + ("" if floor_measured else "  [heuristic floor]"))

    print("")
    print("== RECOMMENDATION ==")
    if any(not v[5] for v in verdicts):
        print("  WARNING: some budgets have no noise replicas — their floor is "
              "the 2.0 FID heuristic, not measured. Add noise_* runs before "
              "trusting those rows.")

    differ = [v for v in verdicts if v[1] == "DIFFER"]
    ties = [v for v in verdicts if v[1] == "tie"]
    if differ:
        best = max(differ, key=lambda v: (v[4] or 0.0))
        rows = best[6]
        ranked = sorted(
            (r for r in rows if r.get("fid") is not None
             and not math.isnan(r["fid"])), key=lambda r: r["fid"])
        print(f"  Arms SEPARATE at calc={best[0]}/50 "
              f"(spread {_f(best[2])} = {_f(best[4], '.1f')}x floor).")
        if ranked:
            print(f"  Best arm there: {ranked[0]['label']} "
                  f"(FID {ranked[0]['fid']:.2f}); worst: "
                  f"{ranked[-1]['label']} (FID {ranked[-1]['fid']:.2f}).")
        print("  => STATIC PLACEMENT EFFECT: mask placement matters at this "
              "budget, but this is not evidence for an adaptive controller.")
        print("  Use paired FID/IS and held-out schedule comparisons to select "
              "an offline policy. Do not start a bandit from this arm spread; "
              "adaptive selection still requires a valid per-trajectory "
              "crossover signal and a multi-metric gate.")
    elif ties and all(v[1] == "tie" for v in verdicts):
        harshest = min(v[0] for v in verdicts)
        print(f"  Arms TIE at every budget tested, down to calc="
              f"{harshest}/50.")
        print("  => Hypothesis A is FALSIFIED over this range: equal-FLOPs "
              "mask placement on TeaCache is not a lever at any budget where "
              "quality is still usable. Close the same-method mask direction "
              "and move to heterogeneous arms (hypothesis B) or the TTT "
              "switch (hypothesis E).")
        print("  Before closing: confirm the harshest budget actually degraded "
              "quality (FID much worse than the calc=13 baseline). If FID "
              "barely moved, the budget was never binding and the probe did "
              "not test what it claims to.")
    else:
        print("  Inconclusive — need >=2 forced arms (and ideally 2 noise "
              "replicas) per budget. Re-run scripts/sweep_budget_probe.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
