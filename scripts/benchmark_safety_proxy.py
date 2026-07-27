#!/usr/bin/env python3
"""Benchmark shallow DiT proxy vs full model for safety shadow defect estimation.

Runs full-model safety shadows on N trajectories and, for each sampled step,
computes:

  full_defect = ||candidate - full|| / ||full - x_t||
  proxy_K_defect = ||candidate_proxy - proxy_K|| / ||proxy_K - x_t||

where proxy_K runs only the first K transformer blocks (vanilla, no
SpecA/TeaCache) before the tail projection.  Reports Spearman rank
correlation per proxy depth.

Usage:
  python scripts/benchmark_safety_proxy.py --n-prompts 16 --proxy-depths 2,4,8
"""

import argparse
import copy
import os
import sys
from typing import Dict, List, Tuple

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
from accelerators.speca import speca_cal_type, speca_init


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


def _shallow_proxy_forward(
    transformer: DiTTransformer2D,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    class_labels: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    """Run only the first *num_blocks* blocks, then the normal tail.

    Returns the per-sample noise prediction tensor (same shape as full forward).
    """
    B, C, H, W = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # pos_embed
    height = H // transformer.patch_size
    width = W // transformer.patch_size
    hidden_states = transformer.pos_embed(hidden_states)

    # first num_blocks (vanilla — no SpecA/TeaCache)
    for layer_idx, block in enumerate(transformer.transformer_blocks[:num_blocks]):
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states, timestep=timestep, class_labels=class_labels,
            hidden_dtype=dtype,
        )
        attn_out = block.attn1(norm_hidden)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out

        norm_ff = block.norm3(hidden_states)
        modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = block.ff(modulated_ff)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_out

    # tail (same as full model)
    conditioning = transformer.transformer_blocks[0].norm1.emb(
        timestep, class_labels, hidden_dtype=dtype)
    shift, scale = transformer.proj_out_1(
        torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
    hidden_states = transformer.norm_out(
        hidden_states) * (1 + scale[:, None]) + shift[:, None]
    hidden_states = transformer.proj_out_2(hidden_states)

    # unpatchify
    hidden_states = hidden_states.reshape(
        shape=(-1, height, width, transformer.patch_size,
               transformer.patch_size, transformer.out_channels))
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(-1, transformer.out_channels, height * transformer.patch_size,
               width * transformer.patch_size))
    return output[:, :transformer.config.in_channels]


def _collect_defect_pairs(
    transformer,
    scheduler,
    timesteps,
    latents,
    class_labels,
    guidance_scale,
    cache_dic,
    current,
    proxy_depths,
    safety_rate: float = 0.25,
    rng: np.random.Generator = None,
) -> Dict[int, List[Tuple[float, float]]]:
    """Run one full denoising trajectory and collect proxy vs full defect pairs.

    On randomly sampled steps, runs:
      - full model forward (safety shadow)
      - shallow proxy forward at each depth
      - candidate (SpecA Taylor) output

    Returns: {depth: [(proxy_defect, full_defect), ...]}
    """
    if rng is None:
        rng = np.random.default_rng()
    in_channels = transformer.config.in_channels
    pairs: Dict[int, List[Tuple[float, float]]] = {d: [] for d in proxy_depths}

    x_t = latents
    for step_idx in range(len(timesteps)):
        t_val = scheduler._host_timestep_values[step_idx]
        t_tensor = timesteps[step_idx]
        t_batch = t_tensor.expand(x_t.shape[0])

        current.step = len(timesteps) - 1 - step_idx
        speca_cal_type(cache_dic, current)
        latent_input = scheduler.scale_model_input(x_t, t_tensor)

        if current.type == 'full' or rng.random() >= safety_rate:
            # Either a full step or not sampled — just run normally
            noise_pred = _covr_shadow_full(
                transformer, latent_input, t_batch, class_labels,
                guidance_scale)
            x_t = scheduler.step(
                noise_pred[:, :in_channels], t_tensor, x_t,
                return_dict=False)[0]
            continue

        # ---- sampled Taylor step: collect full + proxy defects ----
        # full shadow
        full_noise = _covr_shadow_full(
            transformer, latent_input, t_batch, class_labels,
            guidance_scale)
        full_noise = full_noise[:, :in_channels]

        # candidate (run through SpecA with Taylor)
        candidate_noise = transformer.forward_with_cfg(
            latent_input, t_tensor,
            current=current, cache_dic=cache_dic,
            class_labels=class_labels, cfg_scale=guidance_scale,
        )
        candidate_noise = candidate_noise[:, :in_channels]

        # proxy at each depth
        for depth in proxy_depths:
            proxy_noise = _shallow_proxy_forward(
                transformer, latent_input, t_batch, class_labels, depth)

            # defect = ||candidate - X|| / ||X - x_t||
            proxy_num, proxy_den = _covr_transition_components(
                candidate_noise, proxy_noise, x_t)
            full_num, full_den = _covr_transition_components(
                candidate_noise, full_noise, x_t)

            proxy_defect = (proxy_num / (proxy_den + 1e-8)).mean().item()
            full_defect = (full_num / (full_den + 1e-8)).mean().item()
            pairs[depth].append((proxy_defect, full_defect))

        # step with the full noise (safety uses full, not candidate)
        x_t = scheduler.step(
            full_noise, t_tensor, x_t, return_dict=False)[0]

    return pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark shallow proxy defect correlation")
    p.add_argument("--n-prompts", type=int, default=16,
                   help="Number of images to run (default: 16)")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--safety-rate", type=float, default=0.25,
                   help="Fraction of Taylor steps to sample (default: 0.25)")
    p.add_argument("--proxy-depths", type=str, default="2,4,6,8",
                   help="Comma-separated block counts for proxy")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    proxy_depths = [int(d.strip()) for d in args.proxy_depths.split(",")]
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"Device: {device}, dtype: {args.dtype}")
    print(f"Proxy depths: {proxy_depths}")
    print(f"Prompts: {args.n_prompts}, steps: {args.num_steps}")
    print(f"Safety sample rate: {args.safety_rate}")

    # Load model
    print("Loading DiT...")
    transformer = DiTTransformer2D.from_pretrained(
        DIT_REPO, subfolder="transformer", torch_dtype=compute_dtype)
    transformer = transformer.to(device)
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad_(False)

    # Dataset
    ds = ImageNetDataset(
        IMAGENET_DIR, split="val", num_samples=args.n_prompts,
        seed=args.seed, use_class_id_mapping=True)

    # Scheduler
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained(
        DIT_REPO, subfolder="scheduler")
    scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = scheduler.timesteps
    _cache_scheduler_timestep_values(scheduler)

    # Accumulator
    all_pairs: Dict[int, List[Tuple[float, float]]] = {
        d: [] for d in proxy_depths}

    n_batches = (args.n_prompts + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="batches"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, args.n_prompts)
        batch_items = ds.items[start:end]
        actual_bs = end - start

        class_labels = torch.tensor(
            [item[2] for item in batch_items], device=device, dtype=torch.long)
        seeds = [args.seed * 1000 + i for i in range(start, end)]

        # Init latents
        latents = torch.randn(
            (actual_bs, transformer.config.in_channels,
             transformer.config.sample_size // 8,
             transformer.config.sample_size // 8),
            generator=torch.Generator(device=device).manual_seed(
                seeds[0]), device=device, dtype=compute_dtype)
        # CFG doubling
        latents = torch.cat([latents, latents], dim=0)

        # SpecA init (adaptive)
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
            proxy_depths, safety_rate=args.safety_rate, rng=rng)

        for depth, samples in pairs.items():
            all_pairs[depth].extend(samples)

    # Report
    print("\n===== Proxy Defect Correlation =====")
    print(f"{'Depth':>6}  {'samples':>8}  {'Spearman r':>12}  {'p-value':>10}")
    print("-" * 44)
    for depth in proxy_depths:
        samples = all_pairs[depth]
        if len(samples) < 10:
            print(f"{depth:>6}  {len(samples):>8}  {'(too few)':>12}")
            continue
        proxy_vals = [p for p, _ in samples]
        full_vals = [f for _, f in samples]
        r, pval = _spearmanr(proxy_vals, full_vals)
        flag = " ***" if r > 0.9 else "  !!" if r < 0.7 else ""
        print(f"{depth:>6}  {len(samples):>8}  {r:>12.4f}  {pval:>10.2e}{flag}")

    print("\n*** r > 0.9 = strong proxy candidate")
    print("!! r < 0.7 = insufficient correlation")


if __name__ == "__main__":
    main()
