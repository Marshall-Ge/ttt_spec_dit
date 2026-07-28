#!/usr/bin/env python3
"""Benchmark SpecA check_layer error as a proxy for COVR safety defect.

On each Taylor step with a check_layer probe, collects:

  proxy  = current.last_layer_error  (cosine-sim error at check_layer, free)
  defect = ||candidate - full|| / ||full - x_t||  (full shadow, expensive)

Reports Spearman rank correlation to determine whether the already-computed
SpecA error signal can replace the expensive safety shadow.

Usage:
  python scripts/benchmark_safety_proxy.py --n-prompts 16
"""

import argparse
import os
import sys
from typing import List, Tuple

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
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float) + 1.0
    ry = np.argsort(np.argsort(y)).astype(float) + 1.0
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    if den == 0:
        return 0.0, 1.0
    return float(num / den), 1.0


def _collect_defect_pairs(
    transformer,
    scheduler,
    timesteps,
    latents,
    class_labels,
    guidance_scale,
    cache_dic,
    current,
    safety_rate: float = 1.0,
    rng: np.random.Generator = None,
) -> List[Tuple[float, float]]:
    """Run one full denoising trajectory, collect (speca_error, full_defect) pairs.

    On Taylor steps where check_layer was probed, records:
      - current.last_layer_error (the SpecA cosine-sim error at check_layer)
      - full defect = ||candidate - full|| / ||full - x_t||
    """
    if rng is None:
        rng = np.random.default_rng()
    in_channels = transformer.config.in_channels
    pairs: List[Tuple[float, float]] = []

    x_t = latents
    for step_idx in range(len(timesteps)):
        t_tensor = timesteps[step_idx]
        t_batch = t_tensor.expand(x_t.shape[0])

        current.step = len(timesteps) - 1 - step_idx
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
            continue

        # Taylor step: compute full shadow for ground-truth defect
        candidate_noise = noise_pred[:, :in_channels]
        full_noise = _covr_shadow_full(
            transformer, latent_input, t_batch, class_labels,
            guidance_scale)
        full_noise_sliced = full_noise[:, :in_channels]

        # Collect when check_layer was probed and error is fresh
        if (cache_dic.check and current.last_layer_error is not None
                and rng.random() < safety_rate):
            num, den = _covr_transition_components(
                candidate_noise, full_noise_sliced, x_t)
            defect = (num / (den + 1e-8)).mean().item()
            pairs.append((current.last_layer_error, defect))

        x_t = scheduler.step(
            full_noise_sliced, t_tensor, x_t, return_dict=False)[0]

    return pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark SpecA check_layer error as safety defect proxy")
    p.add_argument("--n-prompts", type=int, default=16,
                   help="Number of images to run (default: 16)")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--safety-rate", type=float, default=1.0,
                   help="Fraction of probed Taylor steps to sample (default: 1.0)")
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
    print(f"Safety sample rate: {args.safety_rate}")

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

    all_pairs: List[Tuple[float, float]] = []

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

        pairs = _collect_defect_pairs(
            transformer, scheduler, timesteps, latents, class_labels,
            args.guidance_scale, cache_dic, current,
            safety_rate=args.safety_rate, rng=rng)

        all_pairs.extend(pairs)

    # Report
    print(f"\n===== SpecA Error vs Safety Defect =====")
    print(f"Samples: {len(all_pairs)}")
    if len(all_pairs) < 10:
        print("(too few samples for correlation)")
        return

    proxy_vals = [p for p, _ in all_pairs]
    defect_vals = [d for _, d in all_pairs]
    r, _ = _spearmanr(proxy_vals, defect_vals)

    # Also try Pearson (linear correlation)
    proxy_arr = np.array(proxy_vals)
    defect_arr = np.array(defect_vals)
    pearson = float(np.corrcoef(proxy_arr, defect_arr)[0, 1])

    print(f"{'Spearman r':>16}: {r:+.4f}")
    print(f"{'Pearson r':>16}: {pearson:+.4f}")
    print(f"{'Proxy range':>16}: [{proxy_arr.min():.4f}, {proxy_arr.max():.4f}]")
    print(f"{'Defect range':>16}: [{defect_arr.min():.4f}, {defect_arr.max():.4f}]")

    if r > 0.9:
        print("\n*** r > 0.9 — SpecA error is a strong safety proxy!")
    elif r > 0.7:
        print("\n**  r > 0.7 — moderate correlation, may be usable with calibration")
    else:
        print("\n!! r < 0.7 — insufficient correlation")


if __name__ == "__main__":
    main()
