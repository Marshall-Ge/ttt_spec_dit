#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the --costs JSON for simulate_conf_budget_controller.py.

Maps each budget rung to its controller arm (the per-budget best static
placement, spec section 2 — e.g. k6 is back_loaded, NOT uniform) and pulls
that run's ``aggregate.flops_accel_T`` plus fid/is_mean out of
``<run_root>/<arm>/results.json``.

FLOPs units only need to be CONSISTENT across rungs: the [P1-c] gate
consumes the ratio of closed-loop mean FLOPs to the top-rung null, so
torch's FLOPs-per-generation number is fine as-is (same batch layout on
every arm of a sweep).

Usage (budget-probe layout):
  python3 scripts/build_conf_costs.py output/covr_budget_probe \
      --arm 8=k8/equalflops/arm_pattern_uniform \
      --arm 6=k6/equalflops/arm_pattern_back_loaded \
      --arm 4=k4/equalflops/arm_pattern_uniform \
      -o costs.json
"""

from __future__ import annotations

import argparse
import json
import math
import os


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--arm", action="append", required=True,
                    metavar="BUDGET=ARM_PATH",
                    help="budget rung and its run dir relative to run_root; "
                         "repeat per rung")
    ap.add_argument("-o", "--output", default="costs.json")
    args = ap.parse_args(argv)

    costs = {}
    for spec in args.arm:
        budget, _, arm = spec.partition("=")
        if not budget.strip().isdigit() or not arm:
            ap.error(f"--arm expects BUDGET=ARM_PATH, got {spec!r}")
        path = os.path.join(args.run_root, arm, "results.json")
        with open(path) as fh:
            agg = json.load(fh)["aggregate"]
        flops = float(agg["flops_accel_T"])
        if not math.isfinite(flops) or flops <= 0:
            raise SystemExit(f"{path}: flops_accel_T={flops} is unusable")
        costs[str(int(budget))] = {
            "arm": arm,
            "flops_T": flops,
            "fid": agg.get("fid"),
            "is_mean": agg.get("is_mean"),
        }
        print(f"  budget {budget}: {arm}  flops_accel_T={flops:.4f} "
              f"fid={agg.get('fid')} is_mean={agg.get('is_mean')}")

    with open(args.output, "w") as fh:
        json.dump(costs, fh, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
