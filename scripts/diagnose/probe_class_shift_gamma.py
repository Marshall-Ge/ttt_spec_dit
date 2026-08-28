#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class-subset distribution-shift probe v2: static gamma under class shift.

Uses the OFFICIAL TeaCache state machine (teacache_init/decide/cache_residual/
apply_residual with poly4 rescale) so gamma semantics match main.py runs.

Two ImageNet class subsets with OPPOSITE cache sensitivity (from per-class
gamma calibration):
  - sensitive (gamma* = 0.15): 455, 38, 651, 808, 968
  - robust   (gamma* = 0.40): 449, 916, 927
Run the SAME static gamma=0.25 on both; report latent MSE vs full reference,
skip rate, wall time. Expected: sensitive degrades at gamma=0.25 (its optimum
is 0.15); robust wastes speed (its optimum is 0.40).

Usage:
    python scripts/probe_class_shift_gamma.py --n_per_class 40 --num_steps 50
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

from config import DIT_REPO, IMAGENET_DIR
from models.dit import DiTTransformer2D
from accelerators.teacache import (
    teacache_init, teacache_decide, teacache_cache_residual,
    teacache_apply_residual, compute_modulated_input_dit,
)
from diffusers import DDIMScheduler
from dataset.imagenet import ImageNetDataset

SENSITIVE_CLASSES = [455, 38, 651, 808, 968]
ROBUST_CLASSES = [449, 916, 927]


def _load_dit_coefficients():
    import json as _json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "dit_coef.json")
    with open(path) as f:
        return _json.load(f)["coefficients"]


def run_teacache_official(model, latent, timesteps, scheduler, class_labels,
                          gamma, coefficients):
    state = teacache_init(num_steps=len(timesteps), rel_l1_thresh=gamma,
                          coefficients=coefficients)
    x = latent.clone()
    for t_val in timesteps:
        t_batch = t_val.expand(1).to(torch.int64)
        latent_input = scheduler.scale_model_input(x, t_val)
        with torch.no_grad():
            hidden = model.pos_embed(latent_input)
            mod = compute_modulated_input_dit(model, hidden, t_batch,
                                              class_labels)
            should_calc, _ = teacache_decide(state, mod)
            if not should_calc:
                hidden = teacache_apply_residual(state, hidden)
            else:
                ori_hidden = hidden.clone()
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
                teacache_cache_residual(state, hidden, ori_hidden)
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
        noise_pred = out[:, :4]
        x = scheduler.step(noise_pred, t_val, x, return_dict=False)[0]
        teacache_step_count(state)
    n_calc = sum(1 for d in state["decisions"] if d == "calc")
    return x, n_calc


def teacache_step_count(state):
    state["cnt"] += 1
    if state["cnt"] == state["num_steps"]:
        state["cnt"] = 0


def run_full(model, latent, timesteps, scheduler, class_labels):
    x = latent.clone()
    for t_val in timesteps:
        t_batch = t_val.expand(1).to(torch.int64)
        latent_input = scheduler.scale_model_input(x, t_val)
        with torch.no_grad():
            out = model(latent_input, timestep=t_batch,
                        class_labels=class_labels, return_dict=False)
        x = scheduler.step(out[0][:, :4], t_val, x, return_dict=False)[0]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_class", type=int, default=40)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--gammas", type=str, default=None,
                   help="comma list; when set, sweep gamma per group "
                        "(e.g. '0.15,0.25' for sensitive)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/class_shift_gamma.json")
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
    coefficients = _load_dit_coefficients()

    ds = ImageNetDataset(imagenet_dir=IMAGENET_DIR, n_images=50000,
                         seed=args.seed)
    by_class = {}
    for item in ds.items:
        by_class.setdefault(item[2], []).append(item)
    print(f"classes in dataset: {len(by_class)}")

    # gamma per group: explicit sweep or single static gamma
    group_gammas = {}
    if args.gammas:
        vals = [float(g) for g in args.gammas.split(",")]
        group_gammas["sensitive"] = vals
        group_gammas["robust"] = vals
    else:
        group_gammas["sensitive"] = [args.gamma]
        group_gammas["robust"] = [args.gamma]

    results = {}
    t_start = time.time()
    for group, classes in [("sensitive", SENSITIVE_CLASSES),
                           ("robust", ROBUST_CLASSES)]:
        items = []
        for c in classes:
            items.extend(by_class.get(c, [])[:args.n_per_class])
        group_results = {}
        for gamma in group_gammas[group]:
            mses, skips, walls = [], [], []
            for i, (img_path, prompt, cls_id) in enumerate(items):
                gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
                latent = torch.randn(1, 4, 32, 32, generator=gen).to(
                    device=device, dtype=dtype)
                cl = torch.tensor([cls_id], device=device, dtype=torch.long)
                ref = run_full(model, latent, timesteps, scheduler, cl)
                t0 = time.time()
                x, n_calc = run_teacache_official(
                    model, latent, timesteps, scheduler, cl, gamma,
                    coefficients)
                walls.append(time.time() - t0)
                mses.append(float(F.mse_loss(x.float(), ref.float()).item()))
                skips.append(1.0 - n_calc / len(timesteps))
            group_results[str(gamma)] = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "skip_mean": float(np.mean(skips)),
                "wall_mean": float(np.mean(walls)),
                "n_images": len(items),
            }
            print(f"  {group} gamma={gamma}: mse={group_results[str(gamma)]['mse_mean']:.5f} "
                  f"skip={group_results[str(gamma)]['skip_mean']*100:.1f}% "
                  f"wall={group_results[str(gamma)]['wall_mean']:.3f}s",
                  flush=True)
        results[group] = {"classes": classes, "by_gamma": group_results}

    out = {"config": {"n_per_class": args.n_per_class,
                      "num_steps": args.num_steps, "gamma": args.gamma,
                      "gammas": args.gammas, "seed": args.seed},
           "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nSaved -> {args.out}  Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
