#!/usr/bin/env python3
"""Minimal unit test for the benchmark safety proxy SpecA cache lifecycle.

Usage:
  python tests/test_safety_proxy_cache.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import DIT_REPO
from models.dit import DiTTransformer2D
from run_dit import _covr_shadow_full, _cache_scheduler_timestep_values
from accelerators.speca import speca_init


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    print("Loading DiT...")
    transformer = DiTTransformer2D.from_pretrained(DIT_REPO, subfolder="transformer")
    transformer = transformer.to(device=device, dtype=dtype)
    transformer = transformer.to(device)
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    num_steps = 50
    num_layers = len(transformer.transformer_blocks)
    in_channels = transformer.config.in_channels
    null_class = transformer.config.num_embeds_ada_norm
    print(f"Layers: {num_layers}, in_channels: {in_channels}, null_class: {null_class}")

    # Scheduler
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained(DIT_REPO, subfolder="scheduler")
    scheduler.set_timesteps(num_steps, device=device)
    _cache_scheduler_timestep_values(scheduler)

    # Create fake input
    B = 2
    H = W = transformer.config.sample_size // 8
    latents = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
    latents = torch.cat([latents, latents], dim=0)  # CFG: 2B
    class_labels = torch.randint(0, null_class, (B,), device=device)
    null_labels = torch.full((B,), null_class, device=device, dtype=torch.long)
    class_labels = torch.cat([class_labels, null_labels], dim=0)  # CFG: 2B

    # SpecA init
    cache_dic, current = speca_init(
        num_steps=num_steps,
        num_layers=num_layers,
        base_threshold=0.01, decay_rate=0.01,
        min_taylor_steps=2, max_taylor_steps=5,
        max_order=2, error_metric="cosine_similarity",
        check_layer=20,
    )

    print(f"Initial: last_type={current.last_type}, type={current.type}, "
          f"activated_steps={current.activated_steps}")

    # ---- Step 0: must be 'full' ----
    current.step = num_steps - 1  # 49
    # speca_cal_type is called internally by forward_with_cfg — do NOT call it here
    print(f"Step 0 before forward: type={current.type}, "
          f"activated_steps={current.activated_steps}")

    t_tensor = scheduler.timesteps[0]
    t_batch = t_tensor.expand(latents.shape[0])
    latent_input = scheduler.scale_model_input(latents, t_tensor)

    print("  Running full forward with speca...")
    with torch.no_grad():
        noise_pred = transformer.forward_with_cfg(
            latent_input, t_batch,
            current=current, cache_dic=cache_dic,
            class_labels=class_labels, cfg_scale=4.5,
        )
    print(f"  After forward: type={current.type}")
    assert current.type == 'full', f"Expected 'full', got {current.type!r}"
    print(f"  noise_pred shape: {noise_pred.shape}")

    # Verify cache was populated for all layers
    print("  Checking cache[-1] contents...")
    for layer_idx in range(num_layers):
        layer_cache = cache_dic.cache[-1][layer_idx]
        assert 'attn' in layer_cache, f"Layer {layer_idx}: missing 'attn'"
        assert 'mlp' in layer_cache, f"Layer {layer_idx}: missing 'mlp'"
        assert len(layer_cache['attn']) > 0, f"Layer {layer_idx}: attn cache empty"
        assert len(layer_cache['mlp']) > 0, f"Layer {layer_idx}: mlp cache empty"
    print("  All layer caches populated ✓")

    # Update latents with scheduler step
    latents = scheduler.step(
        noise_pred[:, :in_channels], t_tensor, latents, return_dict=False,
    )[0]

    # ---- Step 1: Taylor ----
    current.step = num_steps - 2  # 48
    # speca_cal_type is called internally by forward_with_cfg — do NOT call it here
    print(f"Step 1 before forward: type={current.type}, "
          f"activated_steps={current.activated_steps}")

    t_tensor = scheduler.timesteps[1]
    t_batch = t_tensor.expand(latents.shape[0])
    latent_input = scheduler.scale_model_input(latents, t_tensor)

    # Run full shadow (control)
    print("  Running full shadow...")
    with torch.no_grad():
        full_noise = _covr_shadow_full(
            transformer, latent_input, t_batch, class_labels, 4.5)

    # Run candidate through speca (Taylor)
    print(f"  Running Taylor candidate...")
    with torch.no_grad():
        candidate_noise = transformer.forward_with_cfg(
            latent_input, t_batch,
            current=current, cache_dic=cache_dic,
            class_labels=class_labels, cfg_scale=4.5,
        )
    print(f"  After forward: type={current.type}")
    print(f"  candidate_noise shape: {candidate_noise.shape}")
    print("  Taylor step succeeded ✓")

    print("\n===== All checks passed =====")


if __name__ == "__main__":
    main()
