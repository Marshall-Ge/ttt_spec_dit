#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token-level long-tail analysis of cache prediction error on DiT-2-256.

Questions:
  1. Is per-token cache error long-tailed on DiT (single-modal, class-cond)?
     (WorldCache found long tails in world models; do we have them too?)
  2. Does token-level grouping (stable/reuse, linear/extrapolate, chaotic/damp)
     beat uniform predictors? (WorldCache CHTP's core claim, tested on DiT)
  3. What is the oracle gain of per-token predictor selection?

On an all-FULL trajectory, for each step t and skip distance d, compute the
per-token error of: reuse_0th, speca_1st (block-level), tc_res. Then:
  - long-tail: share of total error in top-1%/5%/10% tokens (vs uniform share)
  - grouping oracle: per-token argmin over predictors, then average error
  - chaotic-prioritized skipping: if we monitored only the top-k% worst
    tokens (by past error), how much earlier/later would we trigger full vs
    global-average monitoring? (approximated by the concentration of error)

Usage:
    python scripts/diag_token_longtail.py --n_images 4 --num_steps 50 --max_dist 8
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--max_dist", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/diag_token_longtail.json")
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

    # accumulators
    # error concentration: for each (d, bucket), fraction of total error in
    # top-k% tokens (per step, averaged)
    concentration = {d: {k: [] for k in [0.01, 0.05, 0.10, 0.25]}
                     for d in range(1, args.max_dist + 1)}
    # predictor family per-token errors (relative-L2 per token), aggregated
    pred_errors = {name: {d: [] for d in range(1, args.max_dist + 1)}
                   for name in ["reuse", "speca_1st", "tc_res", "oracle_group"]}
    # gini-ish: ratio mean/median of per-token error
    skew = {d: [] for d in range(1, args.max_dist + 1)}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)
        steps_data = run_full_trajectory(model, latent, timesteps, scheduler, class_id)

        for t_idx in range(T):
            cur = steps_data[t_idx]
            x_t = cur["block_in"]
            b_t = cur["block_out"]          # (1, 256, 1152)
            bucket = min(int(t_idx * 3 / T), 2)
            for d in range(1, min(args.max_dist, t_idx) + 1):
                s1 = steps_data[t_idx - d]
                b1, x1 = s1["block_out"], s1["block_in"]
                r1 = b1 - x1
                if t_idx - 2 * d >= 0:
                    b2 = steps_data[t_idx - 2 * d]["block_out"]
                else:
                    b2 = b1

                # per-token errors (cosine per token over feature dim)
                def tok_cos(pred, truth):
                    p = pred.float().squeeze(0)      # (seq, C)
                    q = truth.float().squeeze(0)
                    denom = p.norm(dim=1) * q.norm(dim=1) + 1e-6
                    return 1.0 - (p * q).sum(dim=1) / denom   # (seq,)

                e_reuse = tok_cos(b1, b_t)
                e_1st = tok_cos(2.0 * b1 - b2, b_t)
                e_res = tok_cos(x_t + r1, b_t)

                pred_errors["reuse"][d].append(e_reuse.detach().cpu())
                pred_errors["speca_1st"][d].append(e_1st.detach().cpu())
                pred_errors["tc_res"][d].append(e_res.detach().cpu())

                # oracle per-token grouping: argmin of the three
                stack = torch.stack([e_reuse, e_1st, e_res], dim=0)  # (3, seq)
                e_oracle = stack.min(dim=0).values
                pred_errors["oracle_group"][d].append(e_oracle.detach().cpu())

                # concentration of the reuse error
                e = e_reuse.detach().cpu().float()
                total = e.sum()
                if total > 0:
                    sorted_e = torch.sort(e, descending=True).values
                    for k in concentration[d]:
                        nk = max(int(k * e.numel()), 1)
                        concentration[d][k].append(
                            float(sorted_e[:nk].sum() / total))

                # skew: mean / median
                med = e.median()
                if med > 0:
                    skew[d].append(float(e.mean() / med))

        print(f"  img {img_idx} done ({time.time()-t_start:.1f}s)", flush=True)

    # ---- aggregate ----
    out = {"config": {"n_images": args.n_images, "num_steps": args.num_steps,
                      "max_dist": args.max_dist, "seed": args.seed},
           "concentration_topk_share": {str(d): {str(k): float(np.mean(v))
                                                 for k, v in byk.items()}
                                        for d, byk in concentration.items()},
           "skew_mean_median": {str(d): float(np.mean(v)) for d, v in skew.items()},
           "predictor_mean_cos": {},
           "oracle_gain": {}}
    for name, by_d in pred_errors.items():
        out["predictor_mean_cos"][name] = {}
        for d in range(1, args.max_dist + 1):
            vals = by_d[d]
            if vals:
                cat = torch.cat([v.reshape(-1) for v in vals])
                out["predictor_mean_cos"][name][str(d)] = float(cat.mean())
            else:
                out["predictor_mean_cos"][name][str(d)] = None

    for d in range(1, args.max_dist + 1):
        r = out["predictor_mean_cos"]["reuse"][str(d)]
        g = out["predictor_mean_cos"]["oracle_group"][str(d)]
        out["oracle_gain"][str(d)] = (r - g) if (r is not None and g is not None) else None

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== TOKEN-LEVEL ERROR CONCENTRATION (share of total error in top-k% tokens) ===")
    print(f"{'d':>3} | {'top1%':>7} {'top5%':>7} {'top10%':>7} {'top25%':>7} | skew(mean/med)")
    for d in range(1, args.max_dist + 1):
        c = out["concentration_topk_share"][str(d)]
        s = out["skew_mean_median"][str(d)]
        print(f"{d:>3} | {c['0.01']:7.3f} {c['0.05']:7.3f} {c['0.1']:7.3f} "
              f"{c['0.25']:7.3f} | {s:.2f}")

    print("\n=== PER-TOKEN MEAN COSINE ERROR (uniform vs oracle grouping) ===")
    print(f"{'d':>3} | {'reuse':>8} {'speca_1st':>10} {'tc_res':>8} {'oracle':>8} {'gain':>7}")
    for d in range(1, args.max_dist + 1):
        def g(name):
            v = out["predictor_mean_cos"][name][str(d)]
            return f"{v:.5f}" if v is not None else "  n/a"
        print(f"{d:>3} | {g('reuse'):>8} {g('speca_1st'):>10} {g('tc_res'):>8} "
              f"{g('oracle_group'):>8} {out['oracle_gain'][str(d)]:7.5f}")

    print(f"\nSaved -> {args.out}  Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
