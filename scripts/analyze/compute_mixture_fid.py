#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate [P1-c] quality leg: mixture FID/IS from existing PNGs (GPU box).

Consumes the mixture plan written by
``simulate_conf_budget_controller.py <conf> simulate --mixture-out plan.json``,
assembles the controller's emitted image mixture out of the arms' existing
``generated/`` PNGs (no generation), and adjudicates the spec criterion
(.claude/covr_quality_constrained_budget_20260820.md, Gate [P1-c]):

    mixture FID <= reference-arm FID + fid-floor (default 2.4, the same-arm
    replica floor)  AND  mixture IS >= reference-arm IS - is-floor (0.5)

Faithfulness to the runs' own results.json numbers:
  * every image is preprocessed EXACTLY like eval/fid_is.py add(): lossless
    PNG -> uint8 -> PIL BICUBIC resize to 299x299 -> PNG. generated/ PNGs
    come from utils.save_image with the same uint8 quantization, so the
    assembled 299 set is bit-identical to the gen_299 set the run's own
    FID was computed on;
  * torch-fidelity is called with the same signature as
    run_dit._compute_generated_fid_is (input1=generated set, input2=real
    set, ISC on input1);
  * the verdict anchors on the reference arm RECOMPUTED THROUGH THIS SAME
    PIPELINE over the same shared image indices, which cancels residual
    preprocessing/version drift; the results.json values are printed as a
    cross-check and should agree closely (a large gap = drift warning).

Mixture assembly: every shared image index is assigned to exactly one
budget arm (largest-remainder allocation of the plan's fractions, seeded
shuffle), so the mixed set contains each of the N images exactly once —
the FID-visible content of a controller session, which emits every image
once at whatever budget was active when it streamed by. --replicates
re-draws the assignment to expose its sampling noise; the gate requires
EVERY replicate to sit inside the floors.

Usage (GPU box):
  python3 scripts/compute_mixture_fid.py plan.json \
      --run-root output/covr_budget_probe --workdir /tmp/p1c_mixture
(--real-dir defaults to <run_root>/<reference_arm>/real_299)

Reads PNGs; writes only under --workdir (kept for audit; delete manually).
--dry-run assembles the sets without computing metrics (no torch needed).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def load_png_names(run_root, arm):
    d = os.path.join(run_root, arm, "generated")
    if not os.path.isdir(d):
        raise SystemExit(f"missing generated dir: {d}")
    names = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not names:
        raise SystemExit(f"no PNGs in {d}")
    return d, names


def preprocess_into(src, dst):
    """Bit-identical to eval/fid_is.py add(): uint8 -> BICUBIC 299 -> PNG."""
    from PIL import Image
    img = Image.open(src).convert("RGB")
    if img.size != (299, 299):
        img = img.resize((299, 299), Image.BICUBIC)
    img.save(dst)


def allocate(budgets, fracs, n, rng):
    """Largest-remainder allocation of n images to budgets, then a seeded
    shuffle so the index->budget assignment is uniform at those counts."""
    quotas = {b: fracs[b] * n for b in budgets}
    counts = {b: int(math.floor(quotas[b])) for b in budgets}
    short = n - sum(counts.values())
    for b in sorted(budgets, key=lambda b: quotas[b] - counts[b],
                    reverse=True)[:short]:
        counts[b] += 1
    labels = [b for b in budgets for _ in range(counts[b])]
    rng.shuffle(labels)
    return labels, counts


def fid_is(gen_dir, real_dir, cuda):
    from torch_fidelity import calculate_metrics
    m = calculate_metrics(input1=gen_dir, input2=real_dir, cuda=cuda,
                          isc=True, fid=True, verbose=False,
                          samples_find_ext="png")
    return (float(m["frechet_inception_distance"]),
            float(m["inception_score_mean"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_json",
                    help="written by simulate_conf_budget_controller.py "
                         "simulate --mixture-out")
    ap.add_argument("--run-root", required=True,
                    help="root the plan's arm paths are relative to")
    ap.add_argument("--workdir", required=True,
                    help="output dir for the assembled 299x299 PNG sets")
    ap.add_argument("--real-dir", default=None,
                    help="real_299 dir (default: "
                         "<run_root>/<reference_arm>/real_299)")
    ap.add_argument("--fid-floor", type=float, default=2.4)
    ap.add_argument("--is-floor", type=float, default=0.5)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true",
                    help="run torch-fidelity on CPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the PNG sets only; skip torch-fidelity")
    args = ap.parse_args(argv)

    with open(args.plan_json) as fh:
        plan = json.load(fh)
    ref_arm = plan["reference_arm"]
    mixture = plan["mixture"]
    budgets = sorted(mixture, key=int, reverse=True)
    arms = {b: mixture[b]["arm"] for b in budgets}
    total = sum(mixture[b]["frac"] for b in budgets)
    fracs = {b: mixture[b]["frac"] / total for b in budgets}

    real_dir = args.real_dir or os.path.join(args.run_root, ref_arm,
                                             "real_299")
    if not args.dry_run and not os.path.isdir(real_dir):
        raise SystemExit(
            f"real_299 dir not found: {real_dir}\n"
            "pass --real-dir (any run of the same seed/dataset window has "
            "an equivalent real_299)")

    dirs, name_sets = {}, []
    for b in budgets:
        d, names = load_png_names(args.run_root, arms[b])
        dirs[b] = d
        name_sets.append(set(names))
    shared = sorted(set.intersection(*name_sets))
    if len(shared) < 2:
        raise SystemExit("fewer than 2 shared image names across arms — "
                         "are these runs from the same seed/window?")
    print(f"arms: " + ", ".join(f"k{b}={arms[b]} ({fracs[b]*100:.1f}%)"
                                for b in budgets))
    print(f"shared images across arms: {len(shared)}")

    os.makedirs(args.workdir, exist_ok=True)

    # reference arm through the same pipeline, over the same shared names
    ref_dir_299 = os.path.join(args.workdir, "reference")
    os.makedirs(ref_dir_299, exist_ok=True)
    ref_src = os.path.join(args.run_root, ref_arm, "generated")
    for nm in shared:
        dst = os.path.join(ref_dir_299, nm)
        if not os.path.exists(dst):
            preprocess_into(os.path.join(ref_src, nm), dst)

    mix_dirs = []
    for r in range(args.replicates):
        rng = np.random.default_rng(args.seed + r)
        labels, counts = allocate(budgets, fracs, len(shared), rng)
        d = os.path.join(args.workdir, f"mix_{r}")
        os.makedirs(d, exist_ok=True)
        for nm, b in zip(shared, labels):
            preprocess_into(os.path.join(dirs[b], nm),
                            os.path.join(d, nm))
        mix_dirs.append(d)
        print(f"  mix_{r}: " + ", ".join(f"k{b}:{counts[b]}"
                                         for b in budgets) + f" -> {d}")

    if args.dry_run:
        print("dry run: sets assembled, metrics skipped")
        return 0

    cuda = not args.cpu
    print("\ncomputing torch-fidelity FID/IS (input1=generated, "
          "input2=real; ISC on generated) ...")
    ref_fid, ref_is = fid_is(ref_dir_299, real_dir, cuda)
    rj = plan.get("costs", {}).get(str(plan.get("reference_budget")), {})
    print(f"  reference {ref_arm}: FID={ref_fid:.4f} IS={ref_is:.4f} "
          f"(results.json: FID={rj.get('fid')} IS={rj.get('is_mean')})")
    if rj.get("fid") is not None and abs(ref_fid - float(rj["fid"])) > 0.5:
        print("  WARNING: same-pipeline reference FID differs from "
              "results.json by >0.5 — preprocessing/version drift; the "
              "verdict below uses the same-pipeline anchor")

    fid_lim = ref_fid + args.fid_floor
    is_lim = ref_is - args.is_floor
    all_ok = True
    for r, d in enumerate(mix_dirs):
        m_fid, m_is = fid_is(d, real_dir, cuda)
        ok = m_fid <= fid_lim and m_is >= is_lim
        all_ok &= ok
        print(f"  mix_{r}: FID={m_fid:.4f} (limit {fid_lim:.4f}) "
              f"IS={m_is:.4f} (limit {is_lim:.4f}) "
              f"{'ok' if ok else 'VIOLATION'}")

    print(f"\nGATE [P1-c] quality leg: {'PASS' if all_ok else 'FAIL'} "
          f"(every mixture replicate must satisfy FID <= ref+"
          f"{args.fid_floor} AND IS >= ref-{args.is_floor}; "
          f"FLOPs leg is printed by the simulate mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
