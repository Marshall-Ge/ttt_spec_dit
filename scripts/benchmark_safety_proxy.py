#!/usr/bin/env python3
"""Benchmark free noise-space signals as proxies for COVR safety defect.

Collects multiple candidate signals during Taylor steps (all zero-extra-forward)
and reports Spearman rank correlation vs the full shadow defect.

Signals:
  last_layer_error  — SpecA cosine-sim error at check_layer (baseline)
  taylor_chain      — consecutive Taylor steps since last full
  candidate_norm    — per-sample RMS norm of candidate noise prediction
  noise_delta       — ||noise_t - noise_{t-1}|| / ||noise_{t-1}|| (step-to-step change)
  step_progress     — step_idx / num_steps (progress through denoising)

Usage:
  python scripts/benchmark_safety_proxy.py --n-prompts 32
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DIT_REPO, IMAGENET_DIR
from dataset.imagenet import ImageNetDataset
from models.dit import DiTTransformer2D
from run_dit import (
    _covr_shadow_full,
    _covr_transition_components,
    _cache_scheduler_timestep_values,
)
from accelerators.speca import speca_init


def _spearmanr(x, y):
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    rx = np.argsort(np.argsort(x)).astype(float) + 1.0
    ry = np.argsort(np.argsort(y)).astype(float) + 1.0
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    if den == 0:
        return 0.0, 1.0
    return float(num / den), 1.0


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS norm: sqrt(mean(x^2)) -> (B,)."""
    return x.detach().float().flatten(1).square().mean(dim=1).sqrt()


def _collect_signals(
    transformer,
    scheduler,
    timesteps,
    latents,
    class_labels,
    guidance_scale,
    cache_dic,
    current,
    rng: np.random.Generator,
) -> Dict[str, List[float]]:
    """Run one denoising trajectory, collect all free signals + full defect."""

    in_channels = transformer.config.in_channels
    num_steps = len(timesteps)

    signals: Dict[str, List[float]] = {
        "last_layer_error": [],
        "taylor_chain": [],
        "candidate_norm": [],
        "noise_delta": [],
        "step_progress": [],
        "full_defect": [],
    }

    x_t = latents
    prev_candidate = None

    for step_idx in range(num_steps):
        t_tensor = timesteps[step_idx]
        t_batch = t_tensor.expand(x_t.shape[0])

        current.step = num_steps - 1 - step_idx
        latent_input = scheduler.scale_model_input(x_t, t_tensor)

        noise_pred = transformer.forward_with_cfg(
            latent_input, t_batch,
            current=current, cache_dic=cache_dic,
            class_labels=class_labels, cfg_scale=guidance_scale,
        )

        if current.type == 'full':
            x_t = scheduler.step(
                noise_pred[:, :in_channels], t_tensor, x_t,
                return_dict=False)[0]
            prev_candidate = None  # reset: noise_delta only for consecutive Taylor
            continue

        # ---- Taylor step ----
        candidate_noise = noise_pred[:, :in_channels]
        full_noise = _covr_shadow_full(
            transformer, latent_input, t_batch, class_labels,
            guidance_scale)
        full_noise_sliced = full_noise[:, :in_channels]

        # Compute full defect (ground truth)
        num, den = _covr_transition_components(
            candidate_noise, full_noise_sliced, x_t)
        defect = (num / (den + 1e-8)).mean().item()

        # Collect all free signals
        signals["full_defect"].append(defect)
        signals["step_progress"].append(step_idx / num_steps)
        signals["taylor_chain"].append(cache_dic.taylor_step_counter)
        signals["candidate_norm"].append(_rms_norm(candidate_noise).mean().item())

        # last_layer_error (may be stale if check_layer wasn't probed this step)
        signals["last_layer_error"].append(
            current.last_layer_error if current.last_layer_error is not None
            else float('nan'))

        # noise_delta: step-to-step change in candidate noise
        if prev_candidate is not None:
            delta = _rms_norm(candidate_noise - prev_candidate)
            delta_norm = (delta / (_rms_norm(prev_candidate) + 1e-8)).mean().item()
        else:
            delta_norm = float('nan')
        signals["noise_delta"].append(delta_norm)

        prev_candidate = candidate_noise.detach().clone()

        x_t = scheduler.step(
            full_noise_sliced, t_tensor, x_t, return_dict=False)[0]

    return signals


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark free noise-space signals as safety defect proxies")
    p.add_argument("--n-prompts", type=int, default=32,
                   help="Number of images to run (default: 32)")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"Device: {device}, dtype: {args.dtype}")
    print(f"Prompts: {args.n_prompts}, steps: {args.num_steps}")

    # Load model
    print("Loading DiT...")
    transformer = DiTTransformer2D.from_pretrained(
        DIT_REPO, subfolder="transformer")
    transformer = transformer.to(device=device, dtype=compute_dtype)
    transformer = transformer.to(device)
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad_(False)

    # Dataset
    ds = ImageNetDataset(
        IMAGENET_DIR, n_images=args.n_prompts, seed=args.seed)

    # Scheduler
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained(
        DIT_REPO, subfolder="scheduler")
    scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = scheduler.timesteps
    _cache_scheduler_timestep_values(scheduler)

    # Accumulator: lists per signal
    all_signals: Dict[str, List[float]] = {
        k: [] for k in
        ["last_layer_error", "taylor_chain", "candidate_norm",
         "noise_delta", "step_progress", "full_defect"]}

    n_batches = (args.n_prompts + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="batches"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, args.n_prompts)
        batch_items = ds.items[start:end]
        actual_bs = end - start

        class_labels = torch.tensor(
            [item[2] for item in batch_items], device=device, dtype=torch.long)
        seeds = [args.seed * 1000 + i for i in range(start, end)]

        latents = torch.randn(
            (actual_bs, transformer.config.in_channels,
             transformer.config.sample_size // 8,
             transformer.config.sample_size // 8),
            generator=torch.Generator(device=device).manual_seed(
                seeds[0]), device=device, dtype=compute_dtype)
        latents = torch.cat([latents, latents], dim=0)
        null_class = transformer.config.num_embeds_ada_norm
        null_labels = torch.full((actual_bs,), null_class,
                                 device=device, dtype=torch.long)
        class_labels = torch.cat([class_labels, null_labels], dim=0)

        cache_dic, current = speca_init(
            num_steps=args.num_steps,
            num_layers=len(transformer.transformer_blocks),
            base_threshold=0.01, decay_rate=0.01,
            min_taylor_steps=2, max_taylor_steps=5,
            max_order=2, error_metric="cosine_similarity",
            check_layer=20)

        batch_signals = _collect_signals(
            transformer, scheduler, timesteps, latents, class_labels,
            args.guidance_scale, cache_dic, current, rng)

        for k in all_signals:
            all_signals[k].extend(batch_signals[k])

    # Report
    defect_arr = np.array(all_signals["full_defect"])
    total = len(defect_arr)
    print(f"\n===== Signal vs Defect Correlation ({total} Taylor steps) =====")
    print(f"{'Signal':<22} {'valid':>6} {'Spearman r':>12} {'Pearson r':>12}")

    signal_names = [
        ("last_layer_error", "last_layer_error"),
        ("taylor_chain", "taylor_chain"),
        ("candidate_norm", "candidate_norm"),
        ("noise_delta", "noise_delta"),
        ("step_progress", "step_progress"),
    ]

    for label, key in signal_names:
        raw = np.array(all_signals[key])
        mask = ~np.isnan(raw)
        valid = raw[mask]
        if len(valid) < 10:
            print(f"{label:<22} {len(valid):>6}  {'(too few)':>12}")
            continue
        defect_valid = defect_arr[mask]
        sr, _ = _spearmanr(valid, defect_valid)
        pr = float(np.corrcoef(valid, defect_valid)[0, 1])
        flag = " ***" if abs(sr) > 0.9 else "  !!" if abs(sr) < 0.7 else ""
        print(f"{label:<22} {len(valid):>6}  {sr:>+12.4f}  {pr:>+12.4f}{flag}")

    print("\n*** |r| > 0.9 = strong proxy candidate")
    print("!! |r| < 0.7 = insufficient (individual)")

    # Quick multi-signal linear regression R²
    # Collect samples where all signals are valid
    all_keys = [k for _, k in signal_names]
    all_valid = np.ones(total, dtype=bool)
    for k in all_keys:
        all_valid &= ~np.isnan(np.array(all_signals[k]))
    n_valid = all_valid.sum()
    if n_valid >= 20:
        X = np.column_stack([
            np.array(all_signals[k])[all_valid] for k in all_keys
        ])
        y = defect_arr[all_valid]
        # Add bias column
        Xb = np.column_stack([X, np.ones(n_valid)])
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        y_pred = Xb @ coef
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"\nMulti-signal linear R²: {r2:.4f}  (n={n_valid})")
        if r2 > 0.5:
            print("  => promising! Linear combination may be usable.")
        else:
            print("  => linear combination still insufficient.")


if __name__ == "__main__":
    main()
