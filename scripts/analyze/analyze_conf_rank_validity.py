#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate [P1-a]: rank-validity of session-mean Inception entropy vs FID/IS.

Strengthens the original 4-arm [a] pass (budget probe, Spearman=1.000) to a
6-arm test on the K8 static-mask run (uniform/geometric/back/front + 2
random nulls). Verdict rule from the spec
(.claude/covr_quality_constrained_budget_20260820.md section 3):

  PASS           Spearman(mean_entropy, FID) = +1.0 AND
                 Spearman(mean_entropy, -IS) = +1.0
                 (n=6 arms: exact permutation p_min = 1/720)
  PASS-FID-ONLY  FID ranking perfect, and EVERY IS inversion happens inside
                 an arm pair that differs by less than the noise floor on
                 BOTH metrics (defaults ~2.4 FID / ~0.5 IS, the same-arm
                 replica floors). Rationale: the controller detects big
                 degradations; it does not order near-tied arms.
  FAIL           anything else — the conf signal is not a valid
                 session-level quality proxy on this run and P1 stops.

Inputs:
  conf_table.csv   written by scripts/extract_inception_conf.py
  --metrics-csv    CSV with columns arm,fid,is (arm labels must match the
                   conf table's arm column), OR
  --results-root   walk <root>/**/results.json and take aggregate.fid /
                   aggregate.is_mean with arm = run-dir path relative to
                   root — this matches extract_inception_conf.py's arm
                   labels by construction, no hand-made CSV needed.
  --arm-regex      restrict the gate to matching arm labels (e.g.
                   'arm_pattern_.*_k8$' pins the six offset-0 mask arms and
                   keeps reference/threshold/noise runs out of the gate).

Analysis only; always exits 0 — the verdict line is what gets recorded.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
import os
import re
from collections import defaultdict

import numpy as np


def _rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_pvalue(ent, target, rho_obs):
    """P(Spearman >= observed) under permutation of the entropy ranking.
    Exact enumeration for n <= 8 (includes identity, so p_min = 1/n!)."""
    n = len(ent)
    if n <= 8:
        perms = itertools.permutations(range(n))
        total = math.factorial(n)
    else:
        rng = np.random.default_rng(0)
        perms = (tuple(rng.permutation(n)) for _ in range(20000))
        total = 20000
    hits = sum(spearman(ent[list(k)], target) >= rho_obs - 1e-12 for k in perms)
    return hits / total, total


def metrics_from_results_root(root):
    out = {}
    for path in glob.glob(os.path.join(root, "**", "results.json"),
                          recursive=True):
        try:
            with open(path) as fh:
                agg = json.load(fh).get("aggregate", {})
        except (OSError, json.JSONDecodeError):
            continue
        fid, ism = agg.get("fid"), agg.get("is_mean")
        try:
            fid, ism = float(fid), float(ism)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(fid) and math.isfinite(ism)):
            continue
        out[os.path.relpath(os.path.dirname(path), root)] = (fid, ism)
    return out


def is_inversions(arms, ent, fid, ism):
    """Discordant (entropy, IS) arm pairs: entropy says i better (lower) but
    IS says i worse (lower). Returns [(arm_i, arm_j, d_fid, d_is)]."""
    bad = []
    for i, j in itertools.combinations(range(len(arms)), 2):
        lo, hi = (i, j) if ent[i] < ent[j] else (j, i)
        if ism[lo] < ism[hi]:
            bad.append((arms[lo], arms[hi],
                        abs(fid[lo] - fid[hi]), abs(ism[lo] - ism[hi])))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("conf_csv")
    ap.add_argument("--metrics-csv",
                    help="CSV with columns arm,fid,is")
    ap.add_argument("--results-root",
                    help="run root: read aggregate.fid/is_mean from every "
                         "**/results.json (arm = run dir relative to root)")
    ap.add_argument("--arm-regex", default=None,
                    help="only gate arms whose label matches this regex")
    ap.add_argument("--fid-floor", type=float, default=2.4,
                    help="same-arm replica floor for the PASS-FID-only "
                         "carve-out (default 2.4 FID)")
    ap.add_argument("--is-floor", type=float, default=0.5,
                    help="IS noise floor for the carve-out (default 0.5)")
    args = ap.parse_args(argv)
    if not args.metrics_csv and not args.results_root:
        ap.error("need --metrics-csv or --results-root")

    sums = defaultdict(list)
    with open(args.conf_csv) as fh:
        for row in csv.DictReader(fh):
            sums[row["arm"]].append(float(row["conf"]))

    metrics = {}
    if args.results_root:
        metrics.update(metrics_from_results_root(args.results_root))
    if args.metrics_csv:
        with open(args.metrics_csv) as fh:
            for row in csv.DictReader(fh):
                metrics[row["arm"]] = (float(row["fid"]), float(row["is"]))

    arms = sorted(set(sums) & set(metrics))
    if args.arm_regex:
        pat = re.compile(args.arm_regex)
        arms = [a for a in arms if pat.search(a)]
    if len(arms) < 4:
        print(f"only {len(arms)} matched arms — cannot run the ranking gate")
        print("  conf arms:   " + ", ".join(sorted(sums)))
        print("  metric arms: " + ", ".join(sorted(metrics)))
        return 0

    ent = np.array([np.mean(sums[a]) for a in arms])
    fid = np.array([metrics[a][0] for a in arms])
    ism = np.array([metrics[a][1] for a in arms])

    print(f"matched arms: {len(arms)}"
          + (f" (filtered by {args.arm_regex!r})" if args.arm_regex else ""))
    for k in np.argsort(ent):
        print(f"  {arms[k]:<44} mean_entropy={ent[k]:.4f} "
              f"FID={fid[k]:8.2f} IS={ism[k]:6.2f}")

    rho_fid = spearman(ent, fid)
    rho_is = spearman(ent, -ism)
    p_fid, total = perm_pvalue(ent, fid, rho_fid)
    p_is, _ = perm_pvalue(ent, -ism, rho_is)

    print(f"\nSpearman(mean_entropy, FID)  = {rho_fid:+.4f}  "
          f"p={p_fid:.5f} ({total} perms)")
    print(f"Spearman(mean_entropy, -IS)  = {rho_is:+.4f}  "
          f"p={p_is:.5f} ({total} perms)")

    fid_perfect = rho_fid >= 1.0 - 1e-12
    is_perfect = rho_is >= 1.0 - 1e-12

    if fid_perfect and is_perfect:
        verdict = "PASS"
    elif fid_perfect:
        bad = is_inversions(arms, ent, fid, ism)
        print(f"\nIS inversions ({len(bad)}; carve-out needs |dFID| < "
              f"{args.fid_floor} AND |dIS| < {args.is_floor} on every pair):")
        within = True
        for a, b, dfid, dis in bad:
            ok = dfid < args.fid_floor and dis < args.is_floor
            within &= ok
            print(f"  {a} <-> {b}: |dFID|={dfid:.2f} |dIS|={dis:.2f} "
                  f"{'within floor' if ok else 'EXCEEDS FLOOR'}")
        verdict = "PASS-FID-ONLY" if within else "FAIL"
    else:
        verdict = "FAIL"

    print(f"\nGATE [P1-a]: {verdict}")
    if verdict == "PASS":
        print("  both rankings perfect: lower entropy ranks lower FID and "
              "higher IS on every arm pair")
    elif verdict == "PASS-FID-ONLY":
        print("  FID ranking perfect; IS inversions confined to arm pairs "
              "inside the noise floor — record PASS-FID-only per spec: the "
              "controller detects large degradations, it does not order "
              "near-tied arms")
    else:
        print("  the conf signal is not a valid session-level quality proxy "
              "on this run — P1 stops (spec section 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
