#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose: cache error vs skip-distance structure on DiT-2-256.

Hypothesis (mechanism behind "COVR helps SpecA but not TeaCache"):
  - SpecA (Taylor extrapolation): error GROWS with skip distance d
    (1st-order truncation ~ d^2 * sup||y''||), so WHERE you refresh matters
    -> adaptive refresh decisions have real leverage.
  - TeaCache (residual reuse): error is FLAT in d (no extrapolation term),
    so WHERE you refresh barely matters -> adaptive decisions have no
    leverage; the only lever is the (discrete) gamma threshold.

Method: run an all-FULL 50-step trajectory as ground truth, collecting per-
layer submodule outputs (attn/ff) and post-block hidden states. Then, for
each step t and each skip distance d, simulate what the cached predictor
would have produced had the last full computation been d steps earlier,
and measure error vs the true output.

Predictors simulated:
  SpecA  : 1st-order Taylor from two anchors d and 2d steps ago
           pred = feat(t-d) + (feat(t-d) - feat(t-2d))
  TeaCache: residual reuse   pred = x(t) + (block_out(t-d) - x(t-d))
  TeaCache+lin (exploratory): residual + linear drift of the residual
           pred = x(t) + res(t-d) + (res(t-d) - res(t-2d))

Usage:
    python scripts/diag_error_vs_distance.py --n_images 4 --num_steps 50 --max_dist 10
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
    """All-FULL denoising trajectory; collect per-step per-layer features.

    Returns list of per-step dicts:
      attn      : list[28] attn1 outputs (B, seq, C)
      ff        : list[28] ff outputs
      block_out : post-28-block hidden (B, seq, C)
      block_in  : pos_embed output (B, seq, C)
    """
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
            # tail for scheduler step
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


def rel_l1(a, b, eps=1e-6):
    return ((a - b).abs() / (b.abs() + eps)).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--max_dist", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/diag_error_vs_distance.json")
    p.add_argument("--layers", type=str, default="0,7,14,20,27",
                   help="layers to aggregate (comma list; -1 = all)")
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
    if args.layers == "-1":
        layers = list(range(num_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]
        layers = [l for l in layers if 0 <= l < num_layers]

    # accumulators: d -> list of errors
    acc = {name: {d: [] for d in range(1, args.max_dist + 1)}
           for name in ["speca_attn_cos", "speca_ff_cos", "speca_block_cos",
                        "tc_res_cos", "tc_res_rel_l1",
                        "tc_lin_cos", "tc_lin_rel_l1"]}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)
        steps_data = run_full_trajectory(model, latent, timesteps, scheduler, class_id)
        T = len(steps_data)

        for t_idx in range(T):
            cur = steps_data[t_idx]
            for d in range(1, min(args.max_dist, t_idx) + 1):
                p1 = steps_data[t_idx - d]
                if t_idx - 2 * d >= 0:
                    p2 = steps_data[t_idx - 2 * d]
                else:
                    p2 = p1  # fall back to 0th order

                for layer in layers:
                    # ---- SpecA 1st-order Taylor ----
                    a_prev, a_prev2 = p1["attn"][layer], p2["attn"][layer]
                    pred_a = 2.0 * a_prev - a_prev2
                    acc["speca_attn_cos"][d].append(cosine_err(pred_a, cur["attn"][layer]))

                    f_prev, f_prev2 = p1["ff"][layer], p2["ff"][layer]
                    pred_f = 2.0 * f_prev - f_prev2
                    acc["speca_ff_cos"][d].append(cosine_err(pred_f, cur["ff"][layer]))

                # ---- SpecA block-level: 1st-order Taylor on post-block output ----
                pred_block = 2.0 * p1["block_out"] - p2["block_out"]
                acc["speca_block_cos"][d].append(
                    cosine_err(pred_block, cur["block_out"]))

                # ---- TeaCache residual reuse ----
                res = p1["block_out"] - p1["block_in"]
                pred = cur["block_in"] + res
                acc["tc_res_cos"][d].append(cosine_err(pred, cur["block_out"]))
                acc["tc_res_rel_l1"][d].append(rel_l1(pred, cur["block_out"]))

                # ---- TeaCache + linear residual drift (exploratory) ----
                if t_idx - 2 * d >= 0:
                    res2 = steps_data[t_idx - 2 * d]["block_out"] - \
                        steps_data[t_idx - 2 * d]["block_in"]
                    res_drift = res + (res - res2)
                else:
                    res_drift = res
                pred_lin = cur["block_in"] + res_drift
                acc["tc_lin_cos"][d].append(cosine_err(pred_lin, cur["block_out"]))
                acc["tc_lin_rel_l1"][d].append(rel_l1(pred_lin, cur["block_out"]))

        print(f"  img {img_idx}: T={T} steps done "
              f"({time.time()-t_start:.1f}s elapsed)", flush=True)

    # ---- aggregate ----
    summary = {}
    for name, by_d in acc.items():
        summary[name] = {}
        for d in range(1, args.max_dist + 1):
            vals = by_d[d]
            summary[name][d] = {
                "mean": float(np.mean(vals)) if vals else None,
                "n": len(vals),
            }

    result = {
        "config": {"n_images": args.n_images, "num_steps": args.num_steps,
                   "max_dist": args.max_dist, "seed": args.seed,
                   "layers": layers},
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    # ---- print table ----
    print("\n=== ERROR vs SKIP DISTANCE ===")
    print(f"{'d':>3} | {'SpecA attn cos':>14} {'SpecA ff cos':>13} "
          f"{'TC res cos':>11} {'TC res relL1':>12} "
          f"{'TC lin cos':>11} {'TC lin relL1':>12}")
    for d in range(1, args.max_dist + 1):
        def g(name):
            v = summary[name][d]["mean"]
            return f"{v:.5f}" if v is not None else "   n/a"
        print(f"{d:>3} | {g('speca_attn_cos'):>14} {g('speca_ff_cos'):>13} "
              f"{g('tc_res_cos'):>11} {g('tc_res_rel_l1'):>12} "
              f"{g('tc_lin_cos'):>11} {g('tc_lin_rel_l1'):>12}")
    print(f"\nSaved -> {args.out}")
    print(f"Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
