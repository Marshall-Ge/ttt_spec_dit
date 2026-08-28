#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictor-family error-vs-distance sweep on DiT-2-256.

Generalizes diag_error_vs_distance_v2: on an all-FULL trajectory, simulate
SEVEN predictor structures and measure their error vs skip distance d:

  0th-order family (block level):
    a. reuse_0th    : pred = block_out(t-d)                     [DeepCache-style]
    b. speca_1st    : 2*b(t-d) - b(t-2d)                        [SpecA 1st-order Taylor]
    c. speca_2nd    : 2.5*b(t-d) - 2*b(t-2d) + 0.5*b(t-3d)      [2nd-order Taylor]

  Residual family (TeaCache-style):
    d. tc_res       : x(t) + (b(t-d) - x(t-d))                  [pure residual reuse]
    e. tc_res_lin   : x(t) + r(t-d) + (r(t-d) - r(t-2d))        [residual + linear drift]
    f. tc_res_damp  : x(t) + r(t-d) + alpha_d*(r(t-d)-r(t-2d))  [Hermite-damped drift]
    g. tc_res_adapt : x(t) + r(t-d)*gamma_scale                  [gamma-scaled residual]

where b = post-28-block hidden, x = pos_embed output, r = b - x.

Also reports per-timestep-bucket breakdown for the key predictors.

Usage:
    python scripts/diag_predictor_family.py --n_images 6 --num_steps 50 --max_dist 12
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
from diffusers import DDIMScheduler


def run_full_trajectory(model, latent, timesteps, scheduler, class_labels,
                        in_channels=4):
    dtype = latent.dtype
    B = class_labels.shape[0]
    steps_data = []
    x = latent.clone()
    for t_val in timesteps:
        t_batch = t_val.expand(B).to(torch.int64)
        latent_input = scheduler.scale_model_input(x, t_val)
        with torch.no_grad():
            hidden = model.pos_embed(latent_input)
            block_in = hidden.clone()
            for block in model.transformer_blocks:
                norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                    block.norm1(hidden, timestep=t_batch,
                                class_labels=class_labels,
                                hidden_dtype=hidden.dtype)
                attn_out = block.attn1(norm_hidden)
                hidden = hidden + gate_msa.unsqueeze(1) * attn_out
                norm_ff = block.norm3(hidden)
                modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                ff_out = block.ff(modulated_ff)
                hidden = hidden + gate_mlp.unsqueeze(1) * ff_out
            block_out = hidden
            conditioning = model.transformer_blocks[0].norm1.emb(
                t_batch, class_labels, hidden_dtype=hidden.dtype)
            shift, scale = model.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_tail = model.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            hidden_tail = model.proj_out_2(hidden_tail)
            h = latent_input.shape[-2] // model.patch_size
            w = latent_input.shape[-1] // model.patch_size
            hidden_tail = hidden_tail.reshape(
                -1, h, w, model.patch_size, model.patch_size, model.out_channels)
            hidden_tail = torch.einsum("nhwpqc->nchpwq", hidden_tail)
            out = hidden_tail.reshape(
                -1, model.out_channels, h * model.patch_size, w * model.patch_size)
        noise_pred = out[:, :in_channels]
        x = scheduler.step(noise_pred, t_val, x, return_dict=False)[0]
        steps_data.append({"block_out": block_out, "block_in": block_in})
    return steps_data


def cosine_err(a, b):
    a_f = a.reshape(a.shape[0], -1).float()
    b_f = b.reshape(b.shape[0], -1).float()
    return (1.0 - F.cosine_similarity(a_f, b_f, dim=1)).mean().item()


def rel_l1_batch(a, b, eps=1e-6):
    return ((a - b).abs().mean() / (b.abs().mean() + eps)).float().item()


def hermit(alpha):
    """Cubic Hermite smoothstep: alpha in [0,1]."""
    return 3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=6)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--max_dist", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--damp_nmax", type=int, default=10,
                   help="streak length at which damped predictor is fully "
                        "conservative (alpha -> 1)")
    p.add_argument("--out", default="/tmp/diag_predictor_family.json")
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

    PREDICTORS = ["reuse_0th", "speca_1st", "speca_2nd",
                  "tc_res", "tc_res_lin", "tc_res_damp", "tc_res_adapt"]
    acc = {name: {d: [] for d in range(1, args.max_dist + 1)}
           for name in PREDICTORS}
    acc_rel = {name: {d: [] for d in range(1, args.max_dist + 1)}
               for name in PREDICTORS}
    acc_bucket = {name: {b: {d: [] for d in range(1, args.max_dist + 1)}
                         for b in range(3)} for name in PREDICTORS}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)
        steps_data = run_full_trajectory(model, latent, timesteps, scheduler, class_id)

        for t_idx in range(T):
            cur = steps_data[t_idx]
            x_t = cur["block_in"]
            b_t = cur["block_out"]
            bucket = min(int(t_idx * 3 / T), 2)
            for d in range(1, min(args.max_dist, t_idx) + 1):
                s1 = steps_data[t_idx - d]
                b1, x1 = s1["block_out"], s1["block_in"]
                r1 = b1 - x1

                # --- 0th-order family ---
                pred = b1
                acc["reuse_0th"][d].append(cosine_err(pred, b_t))
                acc_rel["reuse_0th"][d].append(rel_l1_batch(pred, b_t))

                # --- 1st / 2nd order Taylor on block output ---
                if t_idx - 2 * d >= 0:
                    b2 = steps_data[t_idx - 2 * d]["block_out"]
                else:
                    b2 = b1
                pred1 = 2.0 * b1 - b2
                acc["speca_1st"][d].append(cosine_err(pred1, b_t))
                acc_rel["speca_1st"][d].append(rel_l1_batch(pred1, b_t))

                if t_idx - 3 * d >= 0:
                    b3 = steps_data[t_idx - 3 * d]["block_out"]
                    pred2 = 2.5 * b1 - 2.0 * b2 + 0.5 * b3
                else:
                    pred2 = pred1
                acc["speca_2nd"][d].append(cosine_err(pred2, b_t))
                acc_rel["speca_2nd"][d].append(rel_l1_batch(pred2, b_t))

                # --- residual family ---
                pred = x_t + r1
                acc["tc_res"][d].append(cosine_err(pred, b_t))
                acc_rel["tc_res"][d].append(rel_l1_batch(pred, b_t))

                if t_idx - 2 * d >= 0:
                    r2 = (steps_data[t_idx - 2 * d]["block_out"]
                          - steps_data[t_idx - 2 * d]["block_in"])
                else:
                    r2 = r1
                drift = r1 - r2
                pred = x_t + r1 + drift
                acc["tc_res_lin"][d].append(cosine_err(pred, b_t))
                acc_rel["tc_res_lin"][d].append(rel_l1_batch(pred, b_t))

                alpha = hermit(min(d / args.damp_nmax, 1.0))
                pred = x_t + r1 + alpha * drift
                acc["tc_res_damp"][d].append(cosine_err(pred, b_t))
                acc_rel["tc_res_damp"][d].append(rel_l1_batch(pred, b_t))

                # gamma-scaled residual: optimize gamma per distance on the
                # fly (oracle scaling; shows the best a per-d scalar gain can do)
                num = (r1 * (b_t - x_t)).sum()
                den = (r1 * r1).sum() + 1e-6
                gamma = float((num / den).clamp(0.0, 2.0))
                pred = x_t + gamma * r1
                acc["tc_res_adapt"][d].append(cosine_err(pred, b_t))
                acc_rel["tc_res_adapt"][d].append(rel_l1_batch(pred, b_t))

                # bucket accumulation (cos only)
                for name in PREDICTORS:
                    acc_bucket[name][bucket][d].append(acc[name][d][-1])

        print(f"  img {img_idx}: T={T} done ({time.time()-t_start:.1f}s)", flush=True)

    def agg(acc_map):
        out = {}
        for name, by_d in acc_map.items():
            out[name] = {}
            for d in range(1, args.max_dist + 1):
                vals = by_d[d]
                out[name][str(d)] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "n": len(vals),
                }
        return out

    def agg3(acc_map):
        """Aggregate 3-level map: name -> bucket -> d -> [vals]."""
        out = {}
        for name, by_bucket in acc_map.items():
            out[name] = {}
            for bucket, by_d in by_bucket.items():
                out[name][str(bucket)] = {}
                for d in range(1, args.max_dist + 1):
                    vals = by_d[d]
                    out[name][str(bucket)][str(d)] = {
                        "mean": float(np.mean(vals)) if vals else None,
                        "n": len(vals),
                    }
        return out

    result = {
        "config": {"n_images": args.n_images, "num_steps": args.num_steps,
                   "max_dist": args.max_dist, "seed": args.seed,
                   "damp_nmax": args.damp_nmax},
        "cos": agg(acc),
        "rel_l1": agg(acc_rel),
        "bucket_cos": agg3(acc_bucket),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print("\n=== COSINE ERROR vs SKIP DISTANCE (predictor family) ===")
    hdr = f"{'d':>3} | " + " | ".join(f"{n:>11}" for n in PREDICTORS)
    print(hdr)
    for d in range(1, args.max_dist + 1):
        row = f"{d:>3} | "
        for n in PREDICTORS:
            v = result["cos"][n][str(d)]["mean"]
            row += f"{v:11.5f} | " if v is not None else f"{'n/a':>11} | "
        print(row)

    print("\n=== REL-L1 ERROR vs SKIP DISTANCE ===")
    print(f"{'d':>3} | " + " | ".join(f"{n:>11}" for n in PREDICTORS))
    for d in range(1, args.max_dist + 1):
        row = f"{d:>3} | "
        for n in PREDICTORS:
            v = result["rel_l1"][n][str(d)]["mean"]
            row += f"{v:11.4f} | " if v is not None else f"{'n/a':>11} | "
        print(row)

    print("\n=== BY BUCKET (cos, d=1 / d=5 / d=12) ===")
    for b, bname in enumerate(["early", "mid", "late"]):
        row = f"{bname:>5} | "
        for n in ["reuse_0th", "speca_1st", "speca_2nd", "tc_res",
                  "tc_res_lin", "tc_res_damp", "tc_res_adapt"]:
            vals = []
            for d in [1, 5, 12]:
                v = result["bucket_cos"][n][str(b)][str(d)]["mean"]
                vals.append(f"{v:.4f}" if v is not None else "n/a")
            row += f"{n}:{'/'.join(vals):>20} | "
        print(row)

    print(f"\nSaved -> {args.out}  Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
