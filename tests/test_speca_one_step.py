#!/usr/bin/env python3
"""Minimal 1-step test: verify speca cache population during full forward.

Usage:
  python tests/test_speca_one_step.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import DIT_REPO
from models.dit import DiTTransformer2D
from run_dit import _cache_scheduler_timestep_values
from accelerators.speca import speca_init

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    # ---- load model ----
    transformer = DiTTransformer2D.from_pretrained(DIT_REPO, subfolder="transformer")
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)
    num_layers = len(transformer.transformer_blocks)
    in_channels = transformer.config.in_channels
    null_cls = transformer.config.num_embeds_ada_norm
    print(f"layers={num_layers}, in_channels={in_channels}, null_class={null_cls}")

    # ---- scheduler ----
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained(DIT_REPO, subfolder="scheduler")
    scheduler.set_timesteps(50, device=device)
    _cache_scheduler_timestep_values(scheduler)

    # ---- inputs (CFG doubled) ----
    B = 2
    H = W = transformer.config.sample_size // 8
    latents = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
    latents = torch.cat([latents, latents], dim=0)
    labels = torch.randint(0, null_cls, (B,), device=device)
    null_labels = torch.full((B,), null_cls, device=device, dtype=torch.long)
    labels = torch.cat([labels, null_labels], dim=0)

    # ---- speca init ----
    cache_dic, current = speca_init(
        num_steps=50, num_layers=num_layers,
        base_threshold=0.01, decay_rate=0.01,
        min_taylor_steps=2, max_taylor_steps=5,
        max_order=2, error_metric="cosine_similarity",
        check_layer=20)

    # ---- step 0: force 'full' ----
    current.step = 49
    # speca_cal_type is called internally by forward_with_cfg — do NOT call it here
    print(f"  step 0: current.step={current.step}, activated_steps={current.activated_steps}")

    t0 = scheduler.timesteps[0]
    t_batch = t0.expand(latents.shape[0])
    latent_input = scheduler.scale_model_input(latents, t0)

    # Check cache state BEFORE forward
    print(f"  cache[-1][0] before forward: {sorted(cache_dic.cache[-1][0].keys())}")

    print("  running forward (full, with speca)...")
    with torch.no_grad():
        out = transformer.forward_with_cfg(
            latent_input, t_batch,
            current=current, cache_dic=cache_dic,
            class_labels=labels, cfg_scale=4.5,
        )

    print(f"  after forward: type={current.type!r}")
    assert current.type == 'full', f"Expected full after forward, got {current.type!r}"

    # Check cache state AFTER forward
    missing = []
    for lidx in range(num_layers):
        lc = cache_dic.cache[-1][lidx]
        if 'attn' not in lc:
            missing.append(f"L{lidx}: attn")
        elif len(lc['attn']) == 0:
            missing.append(f"L{lidx}: attn empty")
        if 'mlp' not in lc:
            missing.append(f"L{lidx}: mlp")
        elif len(lc['mlp']) == 0:
            missing.append(f"L{lidx}: mlp empty")

    if missing:
        print(f"  FAIL: cache gaps: {missing}")
    else:
        print(f"  OK: all {num_layers} layers have attn+mlp in cache")
        print(f"  cache[-1][0]['attn'] len={len(cache_dic.cache[-1][0]['attn'])}")
        print(f"  out shape={out.shape}")

    print("done.")


if __name__ == "__main__":
    main()
