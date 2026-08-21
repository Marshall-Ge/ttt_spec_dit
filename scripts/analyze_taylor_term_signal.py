#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline gate for the "Taylor term-norm growth" candidate signal (P3).

Question
--------
At a skipped (Taylor) step with cache distance d, the per-order Taylor term
norms are observable for free from the cached derivatives (they only need
d^k rescaling of already-computed finite differences). Does this signal
predict

  [T0]  the current one-step transition defect (known target; one-hot
        timestep is the strongest prior baseline),
  [T1]  the defect ONE MORE skip later (d+1, same refresh episode) — the
        "what happens if we keep skipping" target,
  [X1]  an explosion indicator defect_{d+1} > tau (global p90),

better than one-hot timestep / one-hot timestep+distance baselines?

Data: the 2026-07-21 SpecA shadow-audit JSONL (37,808 events, 32 batch
trajectories, 1,210 unique (trajectory, step) contexts, distance 1-4).
All features are batch/context-level scalars copied across the 32 samples
of a batch; labels are per-sample. This limitation is inherited from the
recorder and is reported, not hidden.

Evaluation: 5-fold grouped CV (group = trajectory_id), ridge regression
with one-hot encodings, pure numpy. Analysis only; always exits 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_events(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            c = e["context"]
            rows.append({
                "traj": int(e["trajectory_id"]),
                "sample": str(e["sample_id"]),
                "step": int(c["step_idx"]),
                "dist": int(c["distance_since_refresh"]),
                "log_snr": float(c["log_snr"]),
                "tn": [float(x) for x in c["taylor_term_norms"]],
                "dis": float(c["order_2_4_disagreement"]),
                "attn_c": float(c["attn_curvature"]),
                "mlp_c": float(c["mlp_curvature"]),
                "defect": float(e["one_step_defect"]),
            })
    return rows


def group_contexts(rows):
    """Context-level rows: (traj, step) -> dict with shared scalars + defects."""
    ctx = {}
    for r in rows:
        key = (r["traj"], r["step"])
        if key not in ctx:
            ctx[key] = {
                "traj": r["traj"], "step": r["step"], "dist": r["dist"],
                "log_snr": r["log_snr"], "tn": np.array(r["tn"]),
                "dis": r["dis"], "attn_c": r["attn_c"], "mlp_c": r["mlp_c"],
                "defects": {}, "sample_order": [],
            }
        if r["sample"] not in ctx[key]["defects"]:
            ctx[key]["sample_order"].append(r["sample"])
        ctx[key]["defects"][r["sample"]] = r["defect"]
    return ctx


def build_episodes(ctx):
    """Refresh episodes per trajectory: consecutive steps with dist 1,2,3,..."""
    eps = []
    by_traj = defaultdict(list)
    for key, c in ctx.items():
        by_traj[c["traj"]].append(c)
    for traj, items in by_traj.items():
        items.sort(key=lambda c: c["step"])
        cur = []
        for c in items:
            if c["dist"] == 1:
                if cur:
                    eps.append(cur)
                cur = [c]
            else:
                if cur and c["dist"] == cur[-1]["dist"] + 1:
                    cur.append(c)
                else:  # broken chain (audit gap) — close episode
                    if cur:
                        eps.append(cur)
                    cur = []
        if cur:
            eps.append(cur)
    return eps


# ---------------------------------------------------------------------------
# ridge / metrics (pure numpy)
# ---------------------------------------------------------------------------

def ridge_fit(X, y, lam=1.0):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def one_hot(values, n_levels):
    out = np.zeros((len(values), n_levels))
    out[np.arange(len(values)), values] = 1.0
    return out


def auroc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort")
    # rank-based (ties averaged) via comparing all pairs is O(n^2); use rank sum
    ranks = order.astype(np.float64) + 1.0
    r_pos = ranks[labels == 1].sum()
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def top_decile_recall(scores, labels, frac=0.10):
    k = max(1, int(len(scores) * frac))
    idx = np.argsort(-scores)[:k]
    return float(labels[idx].mean()), float(labels.mean())


# ---------------------------------------------------------------------------
# experiment tables
# ---------------------------------------------------------------------------

def feature_matrix(entries, spec, num_steps):
    """entries: list of dicts with step, dist, tn (np array at distance d),
    prediction distance d_pred. spec: list of blocks."""
    cols = []
    for b in spec:
        if b == "bias":
            cols.append(np.ones(len(entries)))
        elif b == "step_onehot":
            steps = np.array([e["step"] for e in entries])
            cols.append(one_hot(steps, num_steps).T)
        elif b == "dist_onehot":
            dists = np.array([e["dist"] for e in entries])
            cols.append(one_hot(dists, 8).T)
        elif b == "tn_raw":
            vals = np.stack([e["tn"] for e in entries])
            cols.append(vals.T)
        elif b == "tn_pred":
            # rescale each cached order norm from distance d to d_pred:
            # term_k(d_pred) = term_k(d) * (d_pred/d)^k
            out = []
            for e in entries:
                d, dp = e["dist"], e["d_pred"]
                out.append([t * (dp / d) ** (k + 1)
                            for k, t in enumerate(e["tn"])])
            cols.append(np.stack(out).T)
        elif b == "scalars":
            vals = np.stack([[
                e["dis"], e["attn_c"], e["mlp_c"], e["log_snr"],
                math.log1p(e["dist"]),
            ] for e in entries])
            cols.append(vals.T)
        else:
            raise ValueError(b)
    return np.concatenate([c.T if c.ndim == 2 else c[:, None]
                           for c in cols], axis=1)


def run_cv(entries, y, spec, groups, num_steps, lam=1.0):
    X = feature_matrix(entries, spec, num_steps)
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    folds = np.array_split(ug, 5)
    preds = np.zeros(len(y))
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        w = ridge_fit(X[tr], y[tr], lam=lam)
        preds[te] = X[te] @ w
    mse = float(np.mean((y - preds) ** 2))
    mse_base = float(np.mean((y - y.mean()) ** 2))
    return {
        "mse": mse,
        "r2_oos": 1.0 - mse / mse_base if mse_base > 0 else float("nan"),
        "scores": preds,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("events_jsonl")
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args(argv)

    rows = load_events(args.events_jsonl)
    ctx = group_contexts(rows)
    num_steps = max(c["step"] for c in ctx.values()) + 1
    print(f"events={len(rows)} contexts={len(ctx)} "
          f"trajectories={len({c['traj'] for c in ctx.values()})} "
          f"num_steps={num_steps}")

    # -- 1. variance decomposition ------------------------------------------
    print("\n== [1] defect variance: between-context vs within-context ==")
    all_d = np.array([r["defect"] for r in rows])
    ctx_means, within_var, sizes = [], [], []
    for c in ctx.values():
        v = np.array(list(c["defects"].values()))
        ctx_means.append(v.mean())
        if len(v) > 1:
            within_var.append(v.var())
        sizes.append(len(v))
    ctx_means = np.array(ctx_means)
    print(f"  total var          {all_d.var():.4f}")
    print(f"  between-context    {ctx_means.var():.4f} "
          f"({ctx_means.var()/all_d.var()*100:.1f}% of total)")
    print(f"  mean within-context var {np.mean(within_var):.4f} "
          f"({np.mean(within_var)/all_d.var()*100:.1f}% of total)")
    print(f"  (context-level features can only explain the between part)")

    # -- 2. defect vs distance / step ---------------------------------------
    print("\n== [2] mean defect by cache distance and by step phase ==")
    by_dist = defaultdict(list)
    for r in rows:
        by_dist[r["dist"]].append(r["defect"])
    for d in sorted(by_dist):
        v = np.array(by_dist[d])
        print(f"  dist={d}: n={len(v):6d} mean={v.mean():.4f} "
              f"p50={np.median(v):.4f} p90={np.percentile(v,90):.4f}")
    by_phase = defaultdict(list)
    for r in rows:
        by_phase[r["step"] // 10].append(r["defect"])
    for p in sorted(by_phase):
        v = np.array(by_phase[p])
        print(f"  steps {p*10:2d}-{p*10+9:2d}: n={len(v):6d} "
              f"mean={v.mean():.4f}")

    # -- 3. episodes ---------------------------------------------------------
    eps = build_episodes(ctx)
    lens = defaultdict(int)
    for ep in eps:
        lens[len(ep)] += 1
    print(f"\nepisodes: {len(eps)}, length hist {dict(sorted(lens.items()))}")

    # paired (d -> d+1) sample-matched entries
    paired = []
    for ep in eps:
        for a, b in zip(ep, ep[1:]):
            common = [s for s in a["sample_order"] if s in b["defects"]]
            for s in common:
                paired.append({
                    "traj": a["traj"], "step": a["step"], "dist": a["dist"],
                    "d_pred": b["dist"], "tn": a["tn"], "dis": a["dis"],
                    "attn_c": a["attn_c"], "mlp_c": a["mlp_c"],
                    "log_snr": a["log_snr"],
                    "y_now": a["defects"][s], "y_next": b["defects"][s],
                })
    tau = float(np.percentile(all_d, 90))
    print(f"paired (d,d+1) per-sample rows: {len(paired)}; "
          f"explosion threshold tau=p90={tau:.4f}")

    # context-level paired (predict batch-mean next defect)
    paired_ctx = {}
    for p in paired:
        key = (p["traj"], p["step"])
        if key not in paired_ctx:
            paired_ctx[key] = {
                "traj": p["traj"], "step": p["step"], "dist": p["dist"],
                "d_pred": p["d_pred"], "tn": p["tn"], "dis": p["dis"],
                "attn_c": p["attn_c"], "mlp_c": p["mlp_c"],
                "log_snr": p["log_snr"], "ys_now": [], "ys_next": [],
            }
        paired_ctx[key]["ys_now"].append(p["y_now"])
        paired_ctx[key]["ys_next"].append(p["y_next"])

    # -- 4. regression gates -------------------------------------------------
    specs = [
        ("mean only", ["bias"]),
        ("onehot step", ["bias", "step_onehot"]),
        ("onehot step + dist", ["bias", "step_onehot", "dist_onehot"]),
        ("free feats (tn_pred+scalars)", ["bias", "tn_pred", "scalars"]),
        ("onehot step + free feats",
         ["bias", "step_onehot", "tn_pred", "scalars"]),
        ("onehot step+dist + free feats",
         ["bias", "step_onehot", "dist_onehot", "tn_pred", "scalars"]),
    ]
    groups_p = np.array([p["traj"] for p in paired])
    entries_p = [{k: p[k] for k in
                  ("step", "dist", "d_pred", "tn", "dis", "attn_c", "mlp_c",
                   "log_snr")} for p in paired]

    for target_name, key in [("T0 defect_d", "y_now"),
                             ("T1 defect_{d+1}", "y_next")]:
        y = np.array([p[key] for p in paired])
        print(f"\n== [4] regression, target {target_name} "
              f"(per-sample, grouped 5-fold) ==")
        print(f"  {'model':<32} {'MSE':>8} {'OOS R2':>8}")
        base = None
        for name, spec in specs:
            r = run_cv(entries_p, y, spec, groups_p, num_steps, args.lam)
            if base is None:
                base = r["mse"]
            print(f"  {name:<32} {r['mse']:8.4f} {r['r2_oos']:8.4f}")

    # context-mean version (what a batch-level controller would consume)
    entries_c = list(paired_ctx.values())
    groups_c = np.array([e["traj"] for e in entries_c])
    y_c = np.array([np.mean(e["ys_next"]) for e in entries_c])
    y_c_now = np.array([np.mean(e["ys_now"]) for e in entries_c])
    print(f"\n== [5] regression, target mean defect_{{d+1}} per context "
          f"(n={len(entries_c)}) ==")
    print(f"  {'model':<32} {'MSE':>8} {'OOS R2':>8}")
    for name, spec in specs:
        r = run_cv(entries_c, y_c, spec, groups_c, num_steps, args.lam)
        print(f"  {name:<32} {r['mse']:8.4f} {r['r2_oos']:8.4f}")
    # persistent-defect baseline: predict next by current context mean
    r_pers = run_cv(entries_c, y_c, ["bias", "tn_pred", "scalars"],
                    groups_c, num_steps, args.lam)
    # simple persistence R2 (no CV needed, uses observed y_now)
    mse_pers = float(np.mean((y_c - y_c_now) ** 2))
    mse_base_c = float(np.mean((y_c - y_c.mean()) ** 2))
    print(f"  {'persistence (defect_d)':<32} {mse_pers:8.4f} "
          f"{1.0 - mse_pers/mse_base_c:8.4f}  [NOT free: needs full label]")

    # -- 6. explosion classification ----------------------------------------
    y_exp = (np.array([p["y_next"] for p in paired]) > tau).astype(int)
    print(f"\n== [6] explosion gate: P(defect_{{d+1}} > tau), "
          f"base rate {y_exp.mean():.3f} ==")
    print(f"  {'model':<32} {'AUROC':>7} {'top10% prec':>11} "
          f"{'base':>7}")
    for name, spec in specs[1:]:
        r = run_cv(entries_p, y_exp.astype(float), spec, groups_p,
                   num_steps, args.lam)
        sc = r["scores"]
        prec, baserate = top_decile_recall(sc, y_exp)
        print(f"  {name:<32} {auroc(sc, y_exp):7.4f} {prec:11.3f} "
              f"{baserate:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
