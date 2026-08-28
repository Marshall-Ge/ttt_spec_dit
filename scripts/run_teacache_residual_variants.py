#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TeaCache residual-drift variants: closed-loop quality comparison.

Implements TeaCache with three residual application modes on skip steps:
  plain  : x + res                                  (vanilla TeaCache)
  linear : x + res + (res - res_prev)               (full drift extrapolation)
  damped : x + res + alpha_k*(res - res_prev)       (Hermite-damped drift;
           alpha_k = 3u^2-2u^3, u = min(streak/nmax, 1))

The TeaCache DECISION (accumulate vs threshold, gamma) is identical across
variants — only the skip-step predictor differs. Compares latent MSE vs a
full reference, skip rate, and wall time. Also sweeps gamma for the damped
variant to see whether "lever reconstruction" (distance-sensitive error)
gives adaptive decisions more room.

Usage:
    python scripts/run_teacache_residual_variants.py --n_images 12 --num_steps 50
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


def run_teacache_variant(model, latent, timesteps, scheduler, class_labels,
                         gamma, mode, damp_nmax=10, in_channels=4):
    """TeaCache-style loop with residual-application mode.

    Decision: first/last step calc; otherwise accumulate relative-L1 of the
    block-0 modulated input (mirrors teacache_decide without the poly4
    rescale — we use raw diff * 1.0 so gamma is interpretable).
    """
    T = len(timesteps)
    from accelerators.teacache import compute_modulated_input_dit
    from models.dit import _get_vfl_probe_layer  # noqa: F401 (unused)

    state = {
        "cnt": 0, "accumulated": 0.0, "prev_mod": None,
        "prev_res": None, "prev_res2": None, "streak": 0,
        "n_calc": 0, "n_skip": 0,
    }
    x = latent.clone()
    for step_idx, t_val in enumerate(timesteps):
        t_batch = t_val.expand(1).to(torch.int64)
        latent_input = scheduler.scale_model_input(x, t_val)
        with torch.no_grad():
            hidden = model.pos_embed(latent_input)
            mod = compute_modulated_input_dit(model, hidden, t_batch,
                                              class_labels)
            if state["cnt"] == 0 or state["cnt"] == T - 1:
                should_calc = True
            else:
                raw = ((mod - state["prev_mod"]).abs().mean()
                       / (state["prev_mod"].abs().mean() + 1e-8))
                state["accumulated"] += float(raw)
                should_calc = state["accumulated"] >= gamma
                if should_calc:
                    state["accumulated"] = 0.0
            state["prev_mod"] = mod.detach()

            if should_calc:
                # full block stack
                b_in = hidden
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
                b_out = hidden
                res = b_out - b_in
                state["prev_res2"] = state["prev_res"]
                state["prev_res"] = res.detach()
                state["streak"] = 0
                state["n_calc"] += 1
            else:
                # skip: apply residual variant
                res = state["prev_res"]
                if res is None:
                    res = torch.zeros_like(hidden)
                if mode == "plain":
                    hidden = hidden + res
                elif mode == "linear":
                    drift = res - state["prev_res2"] if state["prev_res2"] is not None \
                        else torch.zeros_like(res)
                    hidden = hidden + res + drift
                elif mode == "damped":
                    drift = res - state["prev_res2"] if state["prev_res2"] is not None \
                        else torch.zeros_like(res)
                    u = min(state["streak"] / damp_nmax, 1.0)
                    alpha = 3.0 * u * u - 2.0 * u * u * u
                    hidden = hidden + res + alpha * drift
                state["streak"] += 1
                state["n_skip"] += 1

            # tail
            conditioning = model.transformer_blocks[0].norm1.emb(
                t_batch, class_labels, hidden_dtype=hidden.dtype)
            shift, scale = model.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_t = model.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            hidden_t = model.proj_out_2(hidden_t)
            h = latent_input.shape[-2] // model.patch_size
            w = latent_input.shape[-1] // model.patch_size
            hidden_t = hidden_t.reshape(-1, h, w, model.patch_size,
                                        model.patch_size, model.out_channels)
            hidden_t = torch.einsum("nhwpqc->nchpwq", hidden_t)
            out = hidden_t.reshape(-1, model.out_channels,
                                   h * model.patch_size, w * model.patch_size)
        noise_pred = out[:, :in_channels]
        x = scheduler.step(noise_pred, t_val, x, return_dict=False)[0]
        state["cnt"] += 1
    return x, state


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
    p.add_argument("--n_images", type=int, default=12)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gammas", type=str, default="0.20,0.25,0.35",
                   help="gamma values to sweep (comma list)")
    p.add_argument("--out", default="/tmp/teacache_residual_variants.json")
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

    gammas = [float(g) for g in args.gammas.split(",")]
    MODES = ["plain", "linear", "damped"]
    results = {f"{mode}_g{g}": {"mse": [], "cos": [], "skip": [], "wall": []}
               for mode in MODES for g in gammas}

    t_start = time.time()
    for img_idx in range(args.n_images):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + img_idx)
        class_id = torch.randint(0, 1000, (1,), generator=gen).to(device)
        latent = torch.randn(1, 4, 32, 32, generator=gen).to(device=device, dtype=dtype)

        ref = run_full(model, latent, timesteps, scheduler, class_id)

        for mode in MODES:
            for g in gammas:
                t0 = time.time()
                x, state = run_teacache_variant(
                    model, latent, timesteps, scheduler, class_id, g, mode)
                wall = time.time() - t0
                mse = float(F.mse_loss(x.float(), ref.float()).item())
                cos = float((1.0 - F.cosine_similarity(
                    x.float().reshape(1, -1), ref.float().reshape(1, -1),
                    dim=1)).mean().item())
                key = f"{mode}_g{g}"
                results[key]["mse"].append(mse)
                results[key]["cos"].append(cos)
                results[key]["skip"].append(state["n_skip"] / T)
                results[key]["wall"].append(wall)
        print(f"  img {img_idx} done ({time.time()-t_start:.1f}s)", flush=True)

    summary = {}
    for key, r in results.items():
        summary[key] = {
            "mse_mean": float(np.mean(r["mse"])),
            "mse_std": float(np.std(r["mse"])),
            "cos_mean": float(np.mean(r["cos"])),
            "skip_mean": float(np.mean(r["skip"])),
            "wall_mean": float(np.mean(r["wall"])),
        }

    out = {"config": {"n_images": args.n_images, "num_steps": args.num_steps,
                      "seed": args.seed, "gammas": gammas},
           "results": summary}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== TeaCache residual variants (closed-loop) ===")
    print(f"{'config':>14} | {'MSE':>10} {'cos':>8} {'skip%':>7} {'wall_s':>8}")
    for mode in MODES:
        for g in gammas:
            s = summary[f"{mode}_g{g}"]
            print(f"{mode+' g='+str(g):>14} | {s['mse_mean']:10.5f} "
                  f"{s['cos_mean']:8.5f} {s['skip_mean']*100:7.1f} "
                  f"{s['wall_mean']:8.3f}")

    print("\nSaved ->", args.out, f" Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
