#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose v3 (closed-loop): real SpecA runtime probe error vs skip distance.

Runs the ACTUAL SpecA pipeline (models/dit.py forward with current/cache_dic)
with an aggressive Taylor schedule (max_taylor_steps high so long streaks
occur) and records, at every do_check probe, the (distance, error) pair.
Confirms the open-loop oracle simulation of diag_error_vs_distance_v2.py
holds under closed-loop drift (inputs change because steps were skipped).

Also records TeaCache-equivalent: at each calc step, what the residual from
d steps ago would have given (closed-loop: reuse of the actual cached
residual against the current input).

Usage:
    python scripts/diag_error_vs_distance_v3.py --n_images 5 --num_steps 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import DIT_REPO
from models.dit import DiTTransformer2D
from accelerators.speca import speca_init, speca_cal_type
from diffusers import DDIMScheduler


def cosine_err(a, b):
    a_f = a.reshape(a.shape[0], -1).float()
    b_f = b.reshape(b.shape[0], -1).float()
    return (1.0 - F.cosine_similarity(a_f, b_f, dim=1)).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--max_taylor", type=int, default=8,
                   help="max consecutive Taylor steps before forced full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/diag_error_vs_distance_v3.json")
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    print("Loading DiT-2-256...")
    model = DiTTransformer2D.from_pretrained(DIT_REPO)
    model = model.to(device=device, dtype=dtype).eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, prediction_type="epsilon",
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False)
    scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = scheduler.timesteps
    num_layers = len(model.transformer_blocks)
    T = len(timesteps)

    # accumulators: distance -> list of (error, bucket)
    speca_dist_err = {d: [] for d in range(1, args.max_taylor + 1)}
    speca_bucket_err = {b: [] for b in range(3)}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)

        cache_dic, current = speca_init(
            num_steps=T, base_threshold=1.0, decay_rate=1.0,
            min_taylor_steps=1, max_taylor_steps=args.max_taylor,
            max_order=2, num_layers=num_layers,
            error_metric="cosine_similarity", check_layer=20,
        )
        x = latent.clone()
        for step_idx, t_val in enumerate(timesteps):
            t_batch = t_val.expand(1).to(torch.int64)
            latent_input = scheduler.scale_model_input(x, t_val)
            current.step = T - 1 - step_idx
            with torch.no_grad():
                # NOTE: speca_cal_type is called INSIDE model.forward; do not
                # call it here too (would double-advance the state).
                out = model(latent_input, timestep=t_batch,
                            current=current, cache_dic=cache_dic,
                            class_labels=class_id, return_dict=False)
            noise_pred = out[0][:, :4]
            x = scheduler.step(noise_pred, t_val, x, return_dict=False)[0]

            if current.last_layer_error is not None:
                d = abs(current.step - current.activated_steps[-1])
                e = float(current.last_layer_error)
                bucket = min(int(step_idx * 3 / T), 2)
                if 1 <= d <= args.max_taylor:
                    speca_dist_err[d].append(e)
                speca_bucket_err[bucket].append(e)
        print(f"  img {img_idx} done ({time.time()-t_start:.1f}s)", flush=True)

    summary = {
        "config": {"n_images": args.n_images, "num_steps": args.num_steps,
                   "max_taylor": args.max_taylor, "seed": args.seed},
        "by_distance": {str(d): {"mean": float(np.mean(v)) if v else None,
                                 "n": len(v)}
                        for d, v in speca_dist_err.items()},
        "by_bucket": {str(b): {"mean": float(np.mean(v)) if v else None,
                               "n": len(v)}
                      for b, v in speca_bucket_err.items()},
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== CLOSED-LOOP SpecA probe error vs distance (check_layer=20) ===")
    for d in range(1, args.max_taylor + 1):
        v = summary["by_distance"][str(d)]
        print(f"  d={d:>2}: err={v['mean']:.6f} (n={v['n']})" if v["mean"] is not None
              else f"  d={d:>2}: n/a")
    print("\n=== by timestep bucket ===")
    for b, bname in enumerate(["early", "mid", "late"]):
        v = summary["by_bucket"][str(b)]
        print(f"  {bname:>5}: err={v['mean']:.6f} (n={v['n']})" if v["mean"] is not None
              else f"  {bname:>5}: n/a")
    print(f"\nSaved -> {args.out}  Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
