#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose v2: per-layer + per-timestep-bucket error-vs-distance structure.

Extends diag_error_vs_distance.py:
  - per-layer aggregation (all 28 layers)
  - per-timestep-bucket aggregation (early/mid/late denoising thirds)
  - fixed rel_l1 (batch-normalized, not per-element)
  - SpecA block-level 1st-order Taylor on post-block output

Usage:
    python scripts/diag_error_vs_distance_v2.py --n_images 6 --num_steps 50 --max_dist 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            attn_outs, ff_outs = [], []
            for block in model.transformer_blocks:
                norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                    block.norm1(hidden, timestep=t_batch,
                                class_labels=class_labels,
                                hidden_dtype=hidden.dtype)
                attn_out = block.attn1(norm_hidden)
                attn_outs.append(attn_out)
                hidden = hidden + gate_msa.unsqueeze(1) * attn_out
                norm_ff = block.norm3(hidden)
                modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                ff_out = block.ff(modulated_ff)
                ff_outs.append(ff_out)
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
        steps_data.append({
            "attn": attn_outs, "ff": ff_outs,
            "block_out": block_out, "block_in": block_in,
        })
    return steps_data


def cosine_err(a, b):
    a_f = a.reshape(a.shape[0], -1).float()
    b_f = b.reshape(b.shape[0], -1).float()
    return (1.0 - F.cosine_similarity(a_f, b_f, dim=1)).mean().item()


def rel_l1_batch(a, b, eps=1e-6):
    """Batch-normalized relative L1: mean|a-b| / (mean|b| + eps)."""
    return ((a - b).abs().mean() / (b.abs().mean() + eps)).float().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=6)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--max_dist", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/diag_error_vs_distance_v2.json")
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

    # per-layer accumulators
    acc_layer = {name: {l: {d: [] for d in range(1, args.max_dist + 1)}
                        for l in range(num_layers)}
                 for name in ["speca_attn_cos", "speca_ff_cos",
                              "speca_block_cos", "tc_res_cos", "tc_res_rel_l1"]}
    # per-bucket accumulators (3 buckets of the denoising trajectory)
    acc_bucket = {name: {b: {d: [] for d in range(1, args.max_dist + 1)}
                         for b in range(3)}
                  for name in ["speca_attn_cos", "speca_block_cos",
                               "tc_res_cos", "tc_res_rel_l1"]}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)
        steps_data = run_full_trajectory(model, latent, timesteps, scheduler, class_id)

        for t_idx in range(T):
            cur = steps_data[t_idx]
            bucket = min(int(t_idx * 3 / T), 2)  # early(0)/mid(1)/late(2)
            for d in range(1, min(args.max_dist, t_idx) + 1):
                p1 = steps_data[t_idx - d]
                p2 = steps_data[t_idx - 2 * d] if t_idx - 2 * d >= 0 else p1

                for layer in range(num_layers):
                    a_prev, a_prev2 = p1["attn"][layer], p2["attn"][layer]
                    pred_a = 2.0 * a_prev - a_prev2
                    ea = cosine_err(pred_a, cur["attn"][layer])
                    acc_layer["speca_attn_cos"][layer][d].append(ea)

                    f_prev, f_prev2 = p1["ff"][layer], p2["ff"][layer]
                    pred_f = 2.0 * f_prev - f_prev2
                    acc_layer["speca_ff_cos"][layer][d].append(
                        cosine_err(pred_f, cur["ff"][layer]))

                # block-level SpecA Taylor
                pred_block = 2.0 * p1["block_out"] - p2["block_out"]
                eb = cosine_err(pred_block, cur["block_out"])
                for layer in range(num_layers):
                    acc_layer["speca_block_cos"][layer][d].append(eb)

                # TeaCache residual reuse
                res = p1["block_out"] - p1["block_in"]
                pred = cur["block_in"] + res
                ec = cosine_err(pred, cur["block_out"])
                el = rel_l1_batch(pred, cur["block_out"])
                for layer in range(num_layers):
                    acc_layer["tc_res_cos"][layer][d].append(ec)
                    acc_layer["tc_res_rel_l1"][layer][d].append(el)

                # bucket accumulators (average over layers for attn-level)
                acc_bucket["speca_attn_cos"][bucket][d].append(
                    np.mean([cosine_err(2.0 * p1["attn"][l] - p2["attn"][l],
                                        cur["attn"][l])
                             for l in range(num_layers)]))
                acc_bucket["speca_block_cos"][bucket][d].append(eb)
                acc_bucket["tc_res_cos"][bucket][d].append(ec)
                acc_bucket["tc_res_rel_l1"][bucket][d].append(el)

        print(f"  img {img_idx}: T={T} done ({time.time()-t_start:.1f}s)", flush=True)

    def agg(acc_map):
        out = {}
        for name, outer in acc_map.items():
            out[name] = {}
            for key, by_d in outer.items():
                out[name][str(key)] = {}
                for d in range(1, args.max_dist + 1):
                    vals = by_d[d]
                    out[name][str(key)][str(d)] = {
                        "mean": float(np.mean(vals)) if vals else None,
                        "n": len(vals),
                    }
        return out

    result = {
        "config": {"n_images": args.n_images, "num_steps": args.num_steps,
                   "max_dist": args.max_dist, "seed": args.seed,
                   "num_layers": num_layers},
        "per_layer": agg(acc_layer),
        "per_bucket": agg(acc_bucket),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    # ---- print: layer-averaged table ----
    print("\n=== ERROR vs DISTANCE (layer-averaged) ===")
    print(f"{'d':>3} | {'S attn cos':>11} {'S ff cos':>9} {'S block':>9} "
          f"{'TC res cos':>11} {'TC relL1':>9}")
    for d in range(1, args.max_dist + 1):
        def g(pl, name, d):
            vals = [pl[name][str(l)][str(d)]["mean"] for l in range(num_layers)]
            vals = [v for v in vals if v is not None]
            return f"{np.mean(vals):.5f}" if vals else "   n/a"
        pl = result["per_layer"]
        print(f"{d:>3} | {g(pl,'speca_attn_cos',d):>11} {g(pl,'speca_ff_cos',d):>9} "
              f"{g(pl,'speca_block_cos',d):>9} {g(pl,'tc_res_cos',d):>11} "
              f"{g(pl,'tc_res_rel_l1',d):>9}")

    # ---- print: per-layer slope summary (d=1 vs d=10 ratio) ----
    print("\n=== PER-LAYER: SpecA attn cos at d=1, d=5, d=10 ===")
    for l in [0, 3, 7, 10, 14, 17, 20, 23, 27]:
        v1 = result["per_layer"]["speca_attn_cos"][str(l)]["1"]["mean"]
        v5 = result["per_layer"]["speca_attn_cos"][str(l)]["5"]["mean"]
        v10 = result["per_layer"]["speca_attn_cos"][str(l)]["10"]["mean"]
        print(f"  layer {l:>2}: d1={v1:.5f} d5={v5:.5f} d10={v10:.5f} "
              f"ratio={v10/v1 if v1 else float('nan'):.1f}x")

    print("\n=== PER-BUCKET: SpecA block cos vs TC res cos ===")
    for b, bname in enumerate(["early", "mid", "late"]):
        sa = result["per_bucket"]["speca_block_cos"][str(b)]
        tc = result["per_bucket"]["tc_res_cos"][str(b)]
        print(f"  {bname:>5}: S d1={sa['1']['mean']:.5f} d5={sa['5']['mean']:.5f} "
              f"d10={sa['10']['mean']:.5f} | "
              f"TC d1={tc['1']['mean']:.5f} d5={tc['5']['mean']:.5f} "
              f"d10={tc['10']['mean']:.5f}")

    print(f"\nSaved -> {args.out}")
    print(f"Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
