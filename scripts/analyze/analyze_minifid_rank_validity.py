#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate [P1'-a]: mini-FID sentinel rank validity + discrimination power.

Spec (PRE-REGISTERED, criteria fixed before data):
.claude/covr_minifid_sentinel_20260821.md §2. Consumes the features npz
written by scripts/extract_inception_feats.py and each arm's true FID
(n=500, torch-fidelity) from its results.json.

  [P1'-a1] validity (n=500):  within every group (offset), the full-sample
           mini-FID must order every above-floor pair (|dFID_true| > floor)
           the same way as true FID. All groups, all pairs -> PASS.
  [P1'-a2] power (n=100):     R independent no-replacement subsamples per
           arm; every pair with |dFID_true| > gate-mult*floor must get
           direction accuracy >= min-acc in every group -> PASS.
           Pairs in (floor, gate-mult*floor] are reported, not adjudicated.
  GATE [P1'-a] = a1 AND a2. FAIL closes the direction (no re-tuning).

Pure numpy; the Frechet trace term uses the Gram-domain spectral identity
  nonzero-spec(Sf Sr) = spec(Xc Sr Xc^T / (n-1))   (n x n, symmetric PSD)
so a subsample draw costs one O(n^2) eigvalsh, not a 2048x2048 sqrtm. The
full Gram matrix G_all = X Sr X^T is precomputed once per (arm, real
group); a draw only slices it and re-centers in Gram space.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_conf_rank_validity import metrics_from_results_root  # noqa: E402


# ---------------------------------------------------------------------------
# mini-FID: Gram-domain implementation
# ---------------------------------------------------------------------------

def real_stats(R):
    """(m, d) real features -> (mu_r, S_r, tr_r), ddof=1."""
    R = np.asarray(R, dtype=np.float64)
    mu = R.mean(axis=0)
    Rc = R - mu
    S = Rc.T @ Rc / (len(R) - 1)
    return mu, S, float(np.trace(S))


class ArmMiniFid:
    """Precomputes everything needed for O(n^2)-per-draw mini-FID of row
    subsets of one arm's fake features against one fixed real side."""

    def __init__(self, X, mu_r, S_r, tr_r):
        self.X = np.asarray(X, dtype=np.float64)      # (N, d)
        self.mu_r, self.tr_r = mu_r, tr_r
        self.G = self.X @ S_r @ self.X.T              # (N, N)
        self.row_sq = (self.X * self.X).sum(axis=1)   # (N,)

    def fid(self, idx=None):
        idx = np.arange(len(self.X)) if idx is None else np.asarray(idx)
        n = len(idx)
        Xs = self.X[idx]
        mu_f = Xs.mean(axis=0)
        # tr(Sf) with ddof=1 via row norms
        tr_f = (self.row_sq[idx].sum() - n * (mu_f @ mu_f)) / (n - 1)
        # Gram-domain centering: H = C G C, C = I - 11^T/n
        G = self.G[np.ix_(idx, idx)]
        H = (G - G.mean(axis=0, keepdims=True)
               - G.mean(axis=1, keepdims=True) + G.mean())
        ev = np.linalg.eigvalsh(H / (n - 1))
        tr_sqrt = float(np.sqrt(np.clip(ev, 0.0, None)).sum())
        d_mu = mu_f - self.mu_r
        return float(d_mu @ d_mu + tr_f + self.tr_r - 2.0 * tr_sqrt)


def self_test():
    """Gram fast path == naive full-spectrum path on random low-d data."""
    rng = np.random.default_rng(0)
    d, m, n = 24, 60, 12
    R = rng.normal(0, 1, (m, d)) @ rng.normal(0, 0.4, (d, d))
    X = rng.normal(0.2, 1.1, (40, d))
    mu_r, S_r, tr_r = real_stats(R)
    fast = ArmMiniFid(X, mu_r, S_r, tr_r)
    for idx in (None, rng.choice(40, n, replace=False)):
        Xs = X if idx is None else X[idx]
        mu_f = Xs.mean(axis=0)
        Xc = Xs - mu_f
        S_f = Xc.T @ Xc / (len(Xs) - 1)
        ev = np.linalg.eigvals(S_f @ S_r).real
        want = float((mu_f - mu_r) @ (mu_f - mu_r) + np.trace(S_f) + tr_r
                     - 2.0 * np.sqrt(np.clip(ev, 0.0, None)).sum())
        got = fast.fid(idx)
        assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (got, want)
    return True


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def load_npz(path, arm_regex):
    z = np.load(path, allow_pickle=False)
    real_of = json.loads(str(z["real_of"]))
    arms = {}
    for key in z.files:
        if not key.startswith("fake::"):
            continue
        arm = key[len("fake::"):]
        if arm_regex and not re.search(arm_regex, arm):
            continue
        arms[arm] = z[key]
    reals = {k[len("real::"):]: z[k] for k in z.files
             if k.startswith("real::")}
    return arms, reals, real_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("feats_npz")
    ap.add_argument("--results-root", default=None,
                    help="root holding each arm's results.json (true FID)")
    ap.add_argument("--metrics-csv", default=None,
                    help="fallback: CSV with columns arm,fid")
    ap.add_argument("--arm-regex", default=None)
    ap.add_argument("--group-regex",
                    default=r"(?:^|[/_])rep[_-]?(\d+)(?:[/_]|$)",
                    help="capture group = offset id; no match -> group '0'")
    # pre-registered criteria (spec §2) — changing these breaks registration
    ap.add_argument("--floor", type=float, default=2.0)
    ap.add_argument("--gate-mult", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-acc", type=float, default=0.90)
    ap.add_argument("--self-test", action="store_true",
                    help="only run the fast-vs-naive FID numeric check")
    args = ap.parse_args(argv)

    self_test()
    print("mini-FID numeric self-test: OK (Gram fast path == naive)")
    if args.self_test:
        return 0

    arms, reals, real_of = load_npz(args.feats_npz, args.arm_regex)
    if not arms:
        print("no fake:: arms in npz (check --arm-regex)")
        return 1

    fid_true = {}
    if args.results_root:
        fid_true = {a: m[0] for a, m in
                    metrics_from_results_root(args.results_root).items()}
    elif args.metrics_csv:
        import csv
        with open(args.metrics_csv) as fh:
            for row in csv.DictReader(fh):
                fid_true[row["arm"]] = float(row["fid"])
    else:
        print("need --results-root or --metrics-csv for true FID")
        return 1

    groups = defaultdict(list)
    for arm in sorted(arms):
        if arm not in fid_true:
            print(f"  WARNING: no true FID for {arm} — skipped")
            continue
        if arm not in real_of:
            print(f"  WARNING: no real features for {arm} — skipped")
            continue
        m = re.search(args.group_regex, arm)
        groups[m.group(1) if m else "0"].append(arm)

    gate_floor = args.gate_mult * args.floor
    a1_fail, a2_fail = [], []
    a2_pairs_seen = 0
    rng = np.random.default_rng(args.seed)
    real_cache = {}

    for gname in sorted(groups):
        garms = groups[gname]
        print(f"\n== group {gname} ({len(garms)} arms) ==")
        mini500, mini100 = {}, {}
        for arm in garms:
            gid = real_of[arm]
            if gid not in real_cache:
                real_cache[gid] = real_stats(reals[gid])
            mu_r, S_r, tr_r = real_cache[gid]
            calc = ArmMiniFid(arms[arm], mu_r, S_r, tr_r)
            N = len(arms[arm])
            mini500[arm] = calc.fid()
            if args.n >= N:
                print(f"  WARNING: n={args.n} >= arm size {N}; "
                      f"power draws degenerate")
            draws = np.array([
                calc.fid(rng.choice(N, size=min(args.n, N), replace=False))
                for _ in range(args.draws)])
            mini100[arm] = draws
            print(f"  {arm:<44} FID={fid_true[arm]:7.2f}  "
                  f"mini{N}={mini500[arm]:7.2f}  "
                  f"mini{args.n}={draws.mean():7.2f}±{draws.std(ddof=1):.2f}")

        print(f"  -- pairs with |dFID_true| > floor {args.floor} --")
        for a, b in itertools.combinations(sorted(garms), 2):
            d_true = fid_true[a] - fid_true[b]
            if abs(d_true) <= args.floor:
                continue
            ok500 = (mini500[a] - mini500[b]) * d_true > 0
            acc = float(((mini100[a] - mini100[b]) * d_true > 0).mean())
            gated = abs(d_true) > gate_floor
            if not ok500:
                a1_fail.append((gname, a, b, d_true))
            if gated:
                a2_pairs_seen += 1
                if acc < args.min_acc:
                    a2_fail.append((gname, a, b, d_true, acc))
            print(f"    {os.path.basename(a)} vs {os.path.basename(b)}: "
                  f"dFID={d_true:+7.2f}  mini{'OK' if ok500 else 'XX'}  "
                  f"n{args.n} acc={acc * 100:5.1f}%"
                  f"{'  [gate]' if gated else '  (report-only)'}")

    n_groups = len(groups)
    a1_ok = not a1_fail and n_groups > 0
    a2_ok = not a2_fail and a2_pairs_seen > 0
    print(f"\nGATE [P1'-a1] validity (n=full): "
          f"{'PASS' if a1_ok else 'FAIL'} "
          f"({n_groups} groups; {len(a1_fail)} above-floor pair(s) "
          f"misordered)")
    for g, a, b, d in a1_fail:
        print(f"    [{g}] {a} vs {b} (dFID={d:+.2f}) misordered")
    print(f"GATE [P1'-a2] power (n={args.n}, R={args.draws}): "
          f"{'PASS' if a2_ok else 'FAIL'} "
          f"({a2_pairs_seen} gated pair(s) with |dFID|>{gate_floor}; "
          f"{len(a2_fail)} below min-acc {args.min_acc * 100:.0f}%)")
    for g, a, b, d, acc in a2_fail:
        print(f"    [{g}] {a} vs {b} (dFID={d:+.2f}) acc={acc * 100:.1f}%")
    final = a1_ok and a2_ok
    print(f"\nGATE [P1'-a]: {'PASS' if final else 'FAIL'} "
          f"(pre-registered: a1 AND a2; FAIL closes the direction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
