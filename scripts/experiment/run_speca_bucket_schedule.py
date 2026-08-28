#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpecA time-bucket-aware scheduling vs uniform: quality-speed comparison.

Motivation (from diag_error_vs_distance_v2/v3):
  - SpecA Taylor error is concentrated in the LATE denoising bucket
    (late ~40x early), and grows super-linearly with skip distance.
  - Hypothesis: a time-bucket-aware max_taylor schedule (aggressive early,
    conservative late) should beat the uniform schedule at the SAME or
    better skip rate, because it allocates refresh where error is steep.

Schedules (max_taylor_steps per step_idx bucket, T=50):
  uniform : [4, 4, 4]
  sched-a : [8, 4, 2]     (aggressive early, mid default, conservative late)
  sched-b : [8, 4, 1]     (even more conservative late)
  sched-c : [6, 4, 2]

Metrics per config (same images/seeds):
  - latent MSE vs full-reference trajectory
  - skip rate (Taylor step fraction), wall time
  - terminal (final-latent) cosine error

Usage:
    python scripts/run_speca_bucket_schedule.py --n_images 20 --num_steps 50
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
from accelerators.speca import speca_init
from diffusers import DDIMScheduler


def run_speca(model, latent, timesteps, scheduler, class_labels,
              max_taylor_by_bucket, in_channels=4):
    """Run SpecA denoising with per-bucket max_taylor_steps.

    Returns (latent, stats) where stats has skip counts, MSE vs full is
    computed by caller.
    """
    T = len(timesteps)
    num_layers = len(model.transformer_blocks)
    cache_dic, current = speca_init(
        num_steps=T, base_threshold=0.01, decay_rate=0.01,
        min_taylor_steps=1, max_taylor_steps=max_taylor_by_bucket[1],
        max_order=2, num_layers=num_layers,
        error_metric="cosine_similarity", check_layer=20,
    )
    x = latent.clone()
    n_taylor = 0
    n_full = 0
    for step_idx, t_val in enumerate(timesteps):
        t_batch = t_val.expand(1).to(torch.int64)
        latent_input = scheduler.scale_model_input(x, t_val)
        current.step = T - 1 - step_idx
        # time-bucket-aware max_taylor: applied before forward (speca_cal_type
        # reads cache_dic.max_taylor_steps inside forward)
        bucket = min(int(step_idx * 3 / T), 2)
        cache_dic.max_taylor_steps = max_taylor_by_bucket[bucket]
        with torch.no_grad():
            out = model(latent_input, timestep=t_batch,
                        current=current, cache_dic=cache_dic,
                        class_labels=class_labels, return_dict=False)
        if current.type == "Taylor":
            n_taylor += 1
        else:
            n_full += 1
        noise_pred = out[0][:, :in_channels]
        x = scheduler.step(noise_pred, t_val, x, return_dict=False)[0]
    return x, {"n_taylor": n_taylor, "n_full": n_full}


def run_full(model, latent, timesteps, scheduler, class_labels, in_channels=4):
    x = latent.clone()
    for t_val in timesteps:
        t_batch = t_val.expand(1).to(torch.int64)
        latent_input = scheduler.scale_model_input(x, t_val)
        with torch.no_grad():
            out = model(latent_input, timestep=t_batch,
                        class_labels=class_labels, return_dict=False)
        noise_pred = out[0][:, :in_channels]
        x = scheduler.step(noise_pred, t_val, x, return_dict=False)[0]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/speca_bucket_schedule.json")
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
    T = len(timesteps)

    CONFIGS = {
        "uniform": (4, 4, 4),
        "sched_a": (8, 4, 2),
        "sched_b": (8, 4, 1),
        "sched_c": (6, 4, 2),
    }

    results = {name: {"mse": [], "cos": [], "skip": [], "wall": [],
                      "n_taylor": [], "n_full": []}
               for name in CONFIGS}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)

        # full reference
        ref = run_full(model, latent, timesteps, scheduler, class_id)

        for name, (e, m, l) in CONFIGS.items():
            t0 = time.time()
            x, stats = run_speca(model, latent, timesteps, scheduler, class_id,
                                 (e, m, l))
            wall = time.time() - t0
            mse = float(F.mse_loss(x.float(), ref.float()).item())
            cos = float((1.0 - F.cosine_similarity(
                x.float().reshape(1, -1), ref.float().reshape(1, -1),
                dim=1)).mean().item())
            results[name]["mse"].append(mse)
            results[name]["cos"].append(cos)
            results[name]["skip"].append(stats["n_taylor"] / T)
            results[name]["wall"].append(wall)
            results[name]["n_taylor"].append(stats["n_taylor"])
            results[name]["n_full"].append(stats["n_full"])
        print(f"  img {img_idx} done ({time.time()-t_start:.1f}s)", flush=True)

    summary = {}
    for name, r in results.items():
        summary[name] = {
            "schedule": CONFIGS[name],
            "mse_mean": float(np.mean(r["mse"])),
            "mse_std": float(np.std(r["mse"])),
            "cos_mean": float(np.mean(r["cos"])),
            "skip_mean": float(np.mean(r["skip"])),
            "skip_std": float(np.std(r["skip"])),
            "wall_mean": float(np.mean(r["wall"])),
            "n_taylor_mean": float(np.mean(r["n_taylor"])),
            "n_full_mean": float(np.mean(r["n_full"])),
        }

    out = {"config": {"n_images": args.n_images, "num_steps": args.num_steps,
                      "seed": args.seed}, "results": summary}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== SpecA time-bucket schedule comparison ===")
    print(f"{'config':>9} | {'sched(E,M,L)':>13} {'MSE':>10} {'cos':>8} "
          f"{'skip%':>7} {'wall_s':>8} {'full':>5} {'taylor':>7}")
    for name in CONFIGS:
        s = summary[name]
        print(f"{name:>9} | {str(CONFIGS[name]):>13} {s['mse_mean']:10.5f} "
              f"{s['cos_mean']:8.5f} {s['skip_mean']*100:7.1f} "
              f"{s['wall_mean']:8.3f} {s['n_full_mean']:5.1f} "
              f"{s['n_taylor_mean']:7.1f}")

    # efficiency: MSE at matched skip rate
    print("\nSaved ->", args.out, f" Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
