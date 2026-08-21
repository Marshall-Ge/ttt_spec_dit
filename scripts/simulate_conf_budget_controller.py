#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline gates for the quality-constrained online budget controller (P1).

Spec: .claude/covr_quality_constrained_budget_20260820.md
Input: conf_table.csv written by scripts/extract_inception_conf.py
(columns arm,budget,image,class_name,conf; conf = InceptionV3 predictive
entropy in nats, smaller = better).

Modes (all pure numpy, analysis only):

  summary     per-arm conf stats + paired per-class headroom vs reference arm
  power       CUSUM false-alarm rate on stationary reference epochs and
              detection delay after a forced stream switch to each arm;
              prints the GATE [P1-b] verdict for the hardest adjacent
              switch (auto-picked from the budget column, or --gate-arm)
  simulate    active closed-loop controller trace on bootstrapped epoch
              streams; mean FLOPs/image vs the do-nothing null (top budget);
              prints the GATE [P1-c] FLOPs-leg verdict and (--mixture-out)
              the emitted-mixture plan for the GPU-side quality leg

Companion scripts:
  scripts/build_conf_costs.py       builds the --costs JSON from each rung
                                    arm's results.json (flops_accel_T)
  scripts/compute_mixture_fid.py    GPU box: assembles the mixture PNG set
                                    from the plan and adjudicates the
                                    [P1-c] quality leg via torch-fidelity

Gates (from the spec; PASS requires ALL of):
  [P1-b]  k8->k6 (hardest adjacent degradation) detected within <=2 epochs,
          stationary FA <= 5%/epoch
  [P1-c]  closed-loop mean FLOPs/image < null by >15% while the emitted
          image mixture stays non-inferior (mixture FID <= ref + 2.4,
          IS non-inferior; adjudicated by compute_mixture_fid.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_table(path):
    arms = defaultdict(list)          # arm -> [(image, class, conf)]
    by_img = defaultdict(dict)        # image -> {arm: conf}
    budget_of = {}                    # arm -> int budget (None if unknown)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            arm = row["arm"]
            conf = float(row["conf"])
            arms[arm].append((row["image"], row["class_name"], conf))
            by_img[row["image"]][arm] = conf
            b = (row.get("budget") or "").strip()
            budget_of[arm] = int(b) if b.isdigit() else None
    return arms, by_img, budget_of


def arm_conf(arms, arm):
    return np.array([c for _, _, c in arms[arm]])


def paired_gap(arms, by_img, arm_a, arm_b):
    """Per-image conf(a) - conf(b) for shared image names (negative = a
    better). Budget-probe arms share seeds/classes, so names pair 1:1."""
    ga, gb = [], []
    for img, d in by_img.items():
        if arm_a in d and arm_b in d:
            ga.append(d[arm_a])
            gb.append(d[arm_b])
    return np.array(ga) - np.array(gb)


# ---------------------------------------------------------------------------
# CUSUM on standardized -conf
# ---------------------------------------------------------------------------

class Cusum:
    """Upward CUSUM on standardized entropy (entropy larger = quality worse;
    alarm detects degradation). k=delta is the drift allowance in sigma."""

    def __init__(self, mu0, sigma0, delta=0.5, h=5.0):
        self.mu0, self.s0 = mu0, max(sigma0, 1e-9)
        self.k, self.h, self.s = delta, h, 0.0

    def update(self, x):
        z = (x - self.mu0) / self.s0   # z>0 = worse than calibration
        self.s = max(0.0, self.s + z - self.k)
        return self.s > self.h


def epoch_alarm(cusum, vals):
    return any(cusum.update(v) for v in vals)


def downstep_ok(z_epoch, eps, b):
    """Aggression rule: epoch-mean z lower bound >= -eps."""
    se = z_epoch.std(ddof=1) / math.sqrt(b) if b > 1 else 0.0
    return (z_epoch.mean() + 1.645 * se) >= -eps


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def mode_summary(arms, by_img, args):
    print(f"== per-arm conf (entropy, nats; smaller = better) ==")
    stats = {}
    for arm in sorted(arms):
        v = arm_conf(arms, arm)
        stats[arm] = (v.mean(), v.std(ddof=1))
        print(f"  {arm:<44} n={len(v):4d} mean={v.mean():.4f} "
              f"sd={v.std(ddof=1):.4f}")
    ref = args.reference
    if ref not in arms:
        print(f"\n(reference arm {ref!r} not in table; skip paired analysis)")
        return
    print(f"\n== paired gap vs reference {ref} (positive = worse than ref) ==")
    for arm in sorted(arms):
        if arm == ref:
            continue
        g = paired_gap(arms, by_img, arm, ref)
        if len(g) == 0:
            continue
        d = g.mean() / max(g.std(ddof=1) / math.sqrt(len(g)), 1e-9)
        print(f"  {arm:<44} n={len(g):4d} mean_gap={g.mean():+.4f} "
              f"t={d:+.1f}  Cohen_d={g.mean()/max(g.std(ddof=1),1e-9):+.3f}")
    # per-class headroom: is the k6 penalty uniform across classes?
    tgt = args.headroom_arm
    if tgt and tgt in arms:
        per_class = defaultdict(list)
        cls_of = {}
        for img_, cls_, _ in arms[ref]:
            cls_of[img_] = cls_
        for img, d in by_img.items():
            if ref in d and tgt in d and img in cls_of:
                per_class[cls_of[img]].append(d[tgt] - d[ref])
        gaps = np.array([np.mean(v) for v in per_class.values() if len(v) >= 3])
        print(f"\n== per-class headroom {tgt} vs {ref} "
              f"(classes with >=3 paired imgs) ==")
        print(f"  classes={len(gaps)} mean={gaps.mean():+.4f} "
              f"sd={gaps.std(ddof=1):.4f} min={gaps.min():+.4f} "
              f"max={gaps.max():+.4f}")
        for eps in (0.05, 0.10, 0.20):
            frac = float((gaps <= eps).mean())
            print(f"  classes with gap<= +{eps:.2f} nats: {frac*100:.1f}%")
        print("  (a large <=eps fraction means subpopulations tolerate the "
              "lower budget — P1/P2 headroom)")


def pick_gate_arm(arms, budget_of, reference):
    """The [P1-b] gate case is the hardest ADJACENT degradation: reference
    budget -> next lower rung, landing on the arm the controller would
    actually use there (per-budget best placement = lowest mean entropy,
    rank-validated by [P1-a])."""
    ref_b = budget_of.get(reference)
    if ref_b is None:
        return None
    lower = sorted({b for b in budget_of.values()
                    if b is not None and b < ref_b}, reverse=True)
    if not lower:
        return None
    cands = [a for a, b in budget_of.items() if b == lower[0]]
    return min(cands, key=lambda a: arm_conf(arms, a).mean())


def mode_power(arms, budget_of, args):
    rng = np.random.default_rng(args.seed)
    ref_v = arm_conf(arms, args.reference)
    mu0, s0 = ref_v.mean(), ref_v.std(ddof=1)

    def draw_epoch(vals, b):
        return rng.choice(vals, size=b, replace=True)

    def fa_rate(h, n_streams=100, epochs_per_stream=10):
        """Continuous-stream FA per epoch: after an alarm the controller
        acts and the detector restarts (as in the closed loop)."""
        fired_e = tot_e = 0
        for _ in range(n_streams):
            cusum = Cusum(mu0, s0, args.delta, h)
            for _ in range(epochs_per_stream):
                fired = epoch_alarm(cusum, draw_epoch(ref_v, args.epoch))
                fired_e += fired
                tot_e += 1
                if fired:
                    cusum = Cusum(mu0, s0, args.delta, h)
        return fired_e / max(tot_e, 1)

    # calibrate h: smallest threshold meeting the <=5%/epoch FA gate
    print(f"== [P1-b] CUSUM calibration (delta={args.delta}, "
          f"epoch={args.epoch}, gate FA<=5%/epoch) ==")
    h_star = fa_star = None
    for h in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
        fa = fa_rate(h)
        tag = ""
        if h_star is None and fa <= 0.05:
            h_star, fa_star, tag = h, fa, "  <- selected"
        print(f"  h={h:4.1f}: FA={fa*100:5.1f}%/epoch{tag}")
    fa_ok = h_star is not None
    if h_star is None:
        h_star = 10.0
        fa_star = fa_rate(h_star)
        print("  WARNING: no h meets the FA gate — signal gate fails "
              "as specified")
    args.h = h_star

    gate_arm = args.gate_arm or pick_gate_arm(arms, budget_of, args.reference)
    detect2 = {}
    print(f"== detection delay after switch (h={h_star}, warmup 1 epoch "
          f"at reference) ==")
    print(f"  {'arm':<44} {'delay<=2':>8} {'median':>7} {'censored':>9}")
    for arm in sorted(arms):
        if arm == args.reference:
            continue
        tgt_v = arm_conf(arms, arm)
        delays = []
        for _ in range(args.trials):
            cusum = Cusum(mu0, s0, args.delta, h_star)
            epoch_alarm(cusum, draw_epoch(ref_v, args.epoch))
            d = 20
            for i in range(20):
                if epoch_alarm(cusum, draw_epoch(tgt_v, args.epoch)):
                    d = i + 1
                    break
            delays.append(d)
        delays = np.array(delays)
        detect2[arm] = float(np.mean(delays <= 2))
        mark = "  <- gate arm" if arm == gate_arm else ""
        print(f"  {arm:<44} {detect2[arm]*100:7.0f}% "
              f"{np.median(delays):7.1f} {np.mean(delays==20)*100:8.0f}%"
              f"{mark}")

    # aggression-rule firing rate per arm (should fire on good arms, not bad)
    print(f"== downstep rule firing rate (eps={args.eps}) ==")
    for arm in sorted(arms):
        v = arm_conf(arms, arm)
        fires = 0
        for _ in range(args.trials):
            vals = draw_epoch(v, args.epoch)
            z = (mu0 - vals) / s0
            fires += downstep_ok(z, args.eps, args.epoch)
        print(f"  {arm:<44} {fires/args.trials*100:5.1f}%  "
              f"(want HIGH on >=ref-quality arms, LOW on degraded arms)")

    if gate_arm is None or gate_arm not in detect2:
        print("\nGATE [P1-b]: cannot adjudicate — no adjacent lower-budget "
              "arm identified (need budget column in the conf table, or "
              "pass --gate-arm)")
    else:
        det = detect2[gate_arm]
        ok = fa_ok and det >= args.detect_frac
        print(f"\nGATE [P1-b]: {'PASS' if ok else 'FAIL'} "
              f"(adjacent switch {args.reference} -> {gate_arm}: "
              f"detect<=2 epochs in {det*100:.0f}% of trials "
              f"[need >={args.detect_frac*100:.0f}%], "
              f"stationary FA={fa_star*100:.1f}%/epoch at h={h_star} "
              f"[need <=5%])")


def mode_simulate(arms, args):
    with open(args.costs) as fh:
        costs = json.load(fh)
    ladder = sorted((int(k) for k in costs), reverse=True)
    arm_of = {int(k): v["arm"] for k, v in costs.items()}
    flops_of = {int(k): float(v["flops_T"]) for k, v in costs.items()}
    rng = np.random.default_rng(args.seed)
    ref = ladder[0]
    ref_v = arm_conf(arms, arm_of[ref])

    null_flops = flops_of[ref]
    trials = args.trials
    tot_imgs = 0.0
    tot_flops = 0.0
    budget_hist = defaultdict(int)
    mixture = defaultdict(int)
    print(f"== closed loop: ladder {ladder}, null = budget {ref} "
          f"({null_flops:.3f} T/img), epoch={args.epoch} imgs, "
          f"{args.n_epochs} epochs ==")
    alarms = 0
    for _ in range(trials):
        k = ref
        calib = rng.choice(ref_v, size=args.epoch, replace=True)
        mu0, s0 = calib.mean(), calib.std(ddof=1)
        cusum = Cusum(mu0, s0, args.delta, args.h)
        z_hist = []
        for _ep in range(args.n_epochs):
            v = rng.choice(arm_conf(arms, arm_of[k]), size=args.epoch,
                           replace=True)
            fired = epoch_alarm(cusum, v)
            tot_imgs += args.epoch
            tot_flops += args.epoch * flops_of[k]
            budget_hist[k] += 1
            mixture[k] += args.epoch
            if fired:
                alarms += 1
                cusum = Cusum(mu0, s0, args.delta, args.h)
                z_hist.clear()
                k = ladder[0] if args.alarm_to_top else \
                    min((b for b in ladder if b > k), default=k)
            else:
                z = (mu0 - v) / s0
                z_hist.append(z.mean())
                if len(z_hist) >= args.stable:
                    recent = z_hist[-args.stable:]
                    se = np.std(recent, ddof=1) / math.sqrt(len(recent))
                    if np.mean(recent) + 1.645 * se >= -args.eps:
                        new_k = max((b for b in ladder if b < k), default=k)
                        if new_k != k:
                            # the "S clean epochs" clock restarts at the new
                            # rung — stale epochs from the higher budget must
                            # not chain into an immediate second downstep
                            z_hist.clear()
                            k = new_k
    mean_flops = tot_flops / max(tot_imgs, 1)
    saving = 1 - mean_flops / null_flops
    print(f"  mean FLOPs/img = {mean_flops:.3f} T vs null {null_flops:.3f} T "
          f"-> saving {saving*100:.1f}%")
    print(f"  alarms: {alarms} over {trials} trials x {args.n_epochs} epochs")
    print("  budget occupancy (epochs):",
          dict(sorted(budget_hist.items(), reverse=True)))
    total_mix = sum(mixture.values())
    print("  emitted image mixture (for the GPU-side mixture-FID check):")
    for k in sorted(mixture, reverse=True):
        print(f"    budget {k} ({arm_of[k]}): {mixture[k]} imgs "
              f"({mixture[k]/total_mix*100:.1f}%)  "
              f"-> sample from that arm's generated/ dir")

    if args.mixture_out:
        plan = {
            "conf_csv": args.conf_csv,
            "costs": costs,
            "reference_budget": ref,
            "reference_arm": arm_of[ref],
            "params": {"epoch": args.epoch, "n_epochs": args.n_epochs,
                       "trials": trials, "delta": args.delta, "h": args.h,
                       "eps": args.eps, "stable": args.stable,
                       "alarm_to_top": bool(args.alarm_to_top),
                       "seed": args.seed},
            "mean_flops_T": mean_flops,
            "null_flops_T": null_flops,
            "saving_pct": saving * 100,
            "mixture": {str(k): {"arm": arm_of[k],
                                 "images": int(mixture[k]),
                                 "frac": mixture[k] / total_mix}
                        for k in sorted(mixture, reverse=True)},
        }
        with open(args.mixture_out, "w") as fh:
            json.dump(plan, fh, indent=2)
        print(f"  mixture plan -> {args.mixture_out} "
              f"(feed to scripts/compute_mixture_fid.py on the GPU box)")

    print(f"\nGATE [P1-c] FLOPs leg: {'PASS' if saving > 0.15 else 'FAIL'} "
          f"(saving {saving*100:.1f}% vs null, need >15% to amortize the "
          f"InceptionV3 forwards + calibration epoch)")
    print("  quality leg (mixture FID <= ref+2.4, IS non-inferior) is "
          "adjudicated by scripts/compute_mixture_fid.py on the GPU box")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("conf_csv")
    ap.add_argument("mode", choices=["summary", "power", "simulate"])
    ap.add_argument("--reference",
                    default="k8/equalflops/arm_pattern_uniform")
    ap.add_argument("--headroom-arm", default=None,
                    help="compare per-class headroom of this arm vs reference")
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--h", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--stable", type=int, default=2)
    ap.add_argument("--costs", default="costs.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-arm", default=None,
                    help="[P1-b] verdict arm (default: best-conf arm on the "
                         "budget rung right below the reference)")
    ap.add_argument("--detect-frac", type=float, default=0.90,
                    help="[P1-b] fraction of trials that must detect the "
                         "adjacent switch within 2 epochs (spec says "
                         "'detected within <=2 epochs'; this operationalizes "
                         "it, default 0.90)")
    ap.add_argument("--alarm-to-top", action="store_true",
                    help="on alarm jump straight back to the top budget "
                         "instead of one rung up (spec offers both)")
    ap.add_argument("--mixture-out", default=None,
                    help="write the emitted-mixture plan JSON for "
                         "scripts/compute_mixture_fid.py")
    args = ap.parse_args(argv)

    arms, by_img, budget_of = load_table(args.conf_csv)
    if args.reference not in arms:
        cands = sorted(arms)
        print(f"reference {args.reference!r} not found; arms available:")
        for c in cands:
            print(f"  {c}")
        return 1
    if args.mode == "summary":
        mode_summary(arms, by_img, args)
    elif args.mode == "power":
        mode_power(arms, budget_of, args)
    else:
        mode_simulate(arms, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
