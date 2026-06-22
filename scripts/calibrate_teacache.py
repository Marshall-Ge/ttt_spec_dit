#!/usr/bin/env python3
"""
Universal TeaCache coefficient calibration script.

Supports any diffusion transformer model via a lightweight registry.
To add a new model, implement three functions:
  - load_fn()           → pipe
  - build_scheduler_fn(pipe, num_steps) → scheduler
  - prepare_cond_fn(pipe, cond) → cond_dict  (encode prompt / class label ONCE)
  - step_fn(pipe, scheduler, latents, t, cond_dict) → (noise_pred, modulated_input)

Usage:
  python scripts/calibrate_teacache.py --model dit
  python scripts/calibrate_teacache.py --model pixart --num_steps 20 --num_runs 20
  python scripts/calibrate_teacache.py --model dit --target_skip 0.4 --threshold 0.3

Output:
  <model>_coef.json — consumed automatically by TeaCacheAccelerator.install()
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from config import PIXART_REPO, DIT_REPO, HF_CACHE_DIR
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from diffusers import DiTPipeline, DDIMScheduler

DEVICE = 'cuda'
DTYPE = torch.float16

# ===========================================================================
# Model registry
# ===========================================================================

MODEL_REGISTRY = {}

def register(name, load_fn, build_scheduler_fn, prepare_cond_fn, step_fn,
             latent_channels=4):
    MODEL_REGISTRY[name] = dict(
        load_fn=load_fn, build_scheduler_fn=build_scheduler_fn,
        prepare_cond_fn=prepare_cond_fn, step_fn=step_fn,
        latent_channels=latent_channels,
    )


# ---- PixArt-α -----------------------------------------------------------

def _load_pixart():
    pipe = PixArtAlphaPipeline.from_pretrained(
        PIXART_REPO, cache_dir=HF_CACHE_DIR, torch_dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)
    pipe.transformer.eval(); pipe.vae.eval()
    return pipe

def _build_scheduler_pixart(pipe, num_steps):
    s = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    s.set_timesteps(num_steps, device=DEVICE)
    return s

def _prepare_cond_pixart(pipe, prompt):
    pe, am, _, _ = pipe.encode_prompt(
        prompt, do_classifier_free_guidance=False,
        num_images_per_prompt=1, device=DEVICE)
    return {'prompt_embeds': pe, 'attn_mask': am}

@torch.no_grad()
def _step_pixart(pipe, scheduler, latents, t, cond):
    """One denoising step. Returns (noise_pred, modulated_input)."""
    transformer = pipe.transformer
    latent_input = scheduler.scale_model_input(latents, t)
    current_t = t.expand(1).to(torch.int64)
    added = {"resolution": None, "aspect_ratio": None}

    # modulated_input: need block-0 hidden states + timestep_emb
    hidden_states = transformer.pos_embed(latent_input)
    timestep_emb, _ = transformer.adaln_single(
        current_t, added, batch_size=hidden_states.shape[0],
        hidden_dtype=hidden_states.dtype)

    block0 = transformer.transformer_blocks[0]
    B = hidden_states.shape[0]
    proj = block0.scale_shift_table[None] + timestep_emb.reshape(B, 6, -1)
    shift_msa, scale_msa, _, _, _, _ = proj.chunk(6, dim=1)
    modulated = block0.norm1(hidden_states) * (1 + scale_msa) + shift_msa

    # Full forward
    noise_pred = transformer(
        latent_input,
        encoder_hidden_states=cond['prompt_embeds'],
        encoder_attention_mask=cond['attn_mask'],
        timestep=current_t, added_cond_kwargs=added, return_dict=False,
    )[0]
    return noise_pred[:, :4], modulated


register('pixart', _load_pixart, _build_scheduler_pixart,
         _prepare_cond_pixart, _step_pixart)

# ---- DiT-2-256 -----------------------------------------------------------

def _load_dit():
    pipe = DiTPipeline.from_pretrained(
        DIT_REPO, cache_dir=HF_CACHE_DIR, torch_dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)
    pipe.transformer.eval(); pipe.vae.eval()
    return pipe

def _build_scheduler_dit(pipe, num_steps):
    s = DDIMScheduler.from_config(pipe.scheduler.config)
    s.set_timesteps(num_steps, device=DEVICE)
    return s

def _prepare_cond_dit(pipe, class_label):
    return {'class_labels': torch.tensor([class_label], device=DEVICE, dtype=torch.long)}

@torch.no_grad()
def _step_dit(pipe, scheduler, latents, t, cond):
    """One denoising step. Returns (noise_pred, modulated_input)."""
    transformer = pipe.transformer
    latent_input = scheduler.scale_model_input(latents, t)
    current_t = t.expand(1).to(torch.int64)
    class_labels = cond['class_labels']

    # modulated_input: block-0 AdaLayerNormZero first return
    hidden_states = transformer.pos_embed(latent_input)
    modulated = transformer.transformer_blocks[0].norm1(
        hidden_states, timestep=current_t, class_labels=class_labels,
        hidden_dtype=hidden_states.dtype)[0]

    # Full forward
    noise_pred = transformer(
        latent_input, timestep=current_t, class_labels=class_labels,
        return_dict=False)[0]
    return noise_pred[:, :4], modulated


register('dit', _load_dit, _build_scheduler_dit, _prepare_cond_dit, _step_dit)

# ===========================================================================
# Core calibration (model-agnostic)
# ===========================================================================

@torch.no_grad()
def collect_diffs(pipe, cfg, num_steps, seed, cond_dict):
    """Run one denoising trajectory, return raw relative-L1 diffs per step."""
    # Fresh scheduler per run (DPMSolverMultistepScheduler has internal state)
    scheduler = cfg['build_scheduler_fn'](pipe, num_steps)
    sample_size = pipe.transformer.config.sample_size
    shape = (1, cfg['latent_channels'], sample_size, sample_size)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = torch.randn(shape, device=DEVICE, dtype=DTYPE,
                          generator=generator) * scheduler.init_noise_sigma

    prev = None
    diffs = []

    for t in scheduler.timesteps:
        noise_pred, modulated = cfg['step_fn'](pipe, scheduler, latents, t, cond_dict)

        if prev is not None:
            d = ((modulated - prev).abs().mean() /
                 prev.abs().mean()).cpu().float().item()
            diffs.append(d)
        prev = modulated.detach()

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return diffs


def calibrate(model_name, num_steps=20, num_runs=10, target_skip=0.5,
              threshold=0.25, output_path=None):
    cfg = MODEL_REGISTRY.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    print("=" * 60)
    print(f"TeaCache Calibration — {model_name}")
    print("=" * 60)
    print(f"  Steps: {num_steps}  Runs: {num_runs}  "
          f"Target skip: {target_skip:.0%}  γ: {threshold}")

    # 1. Load
    print(f"\n[1] Loading {model_name}...")
    pipe = cfg['load_fn']()
    print(f"  Loaded. blocks={len(pipe.transformer.transformer_blocks)}")

    # 2. Collect
    print(f"\n[2] Collecting ({num_runs} runs × {num_steps} steps)...")
    all_diffs = []
    for i in tqdm(range(num_runs), desc='calibrating'):
        seed = 42 + i
        if model_name == 'dit':
            cond_dict = cfg['prepare_cond_fn'](pipe, (seed * 7) % 1000)
        else:
            prompts = ['a photo of a cat', 'a beautiful landscape',
                       'a portrait', 'a red car', 'a dog in the park',
                       'a sunset', 'a bowl of fruit', 'a city skyline',
                       'a flower garden', 'an astronaut']
            cond_dict = cfg['prepare_cond_fn'](pipe, prompts[i % len(prompts)])
        diffs = collect_diffs(pipe, cfg, num_steps, seed, cond_dict)
        all_diffs.extend(diffs)

    all_diffs = np.array(all_diffs)
    print(f"  Collected {len(all_diffs)} raw diffs")
    print(f"  Min={all_diffs.min():.6f}  Max={all_diffs.max():.6f}  "
          f"Mean={all_diffs.mean():.6f}  Median={np.median(all_diffs):.6f}")

    # 3. Fit polynomial
    middle = num_steps - 2
    n_calcs = middle * (1 - target_skip)
    steps_per_calc = middle / max(n_calcs, 1)
    target_rescaled = threshold / steps_per_calc

    print(f"\n[3] Fitting 4th-order poly (steps/calc={steps_per_calc:.1f}, "
          f"target_rescaled={target_rescaled:.4f})...")

    sorted_d = np.sort(all_diffs)
    n = len(sorted_d)
    max_r = target_rescaled * 3
    target_vals = np.linspace(0, 1, n) * max_r

    coeffs = np.polyfit(sorted_d, target_vals, deg=4)
    poly = np.poly1d(coeffs)
    print(f"  Coefficients: {[round(c, 8) for c in coeffs.tolist()]}")

    # Warn if poly oscillates negative outside training range
    test_d = np.linspace(0, sorted_d.max() * 1.2, 100)
    neg_n = int((poly(test_d) < 0).sum())
    if neg_n > 0:
        print(f"  ⚠  {neg_n}/100 test points negative — poly oscillates below "
              f"training range. Re-run with --num_steps matching inference steps.")

    # 4. Validate
    print(f"\n[4] Validation:")
    for pct in [10, 25, 50, 75, 90]:
        v = np.percentile(sorted_d, pct)
        print(f"  P{pct}: raw={v:.6f} → rescaled={max(0.0, poly(v)):.6f}")

    # 5. Save
    output_path = output_path or os.path.join(
        os.path.dirname(__file__), '..', f'{model_name}_coef.json')
    result = {
        'coefficients': [float(c) for c in coeffs.tolist()],
        'num_steps': num_steps, 'num_runs': num_runs,
        'target_skip': target_skip, 'threshold': threshold,
        'target_rescaled': float(target_rescaled), 'model': model_name,
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {output_path}")

    del pipe; torch.cuda.empty_cache()
    print("Done.\n")
    return coeffs, result


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description='Universal TeaCache coefficient calibration')
    p.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument('--num_steps', type=int, default=20)
    p.add_argument('--num_runs', type=int, default=10)
    p.add_argument('--target_skip', type=float, default=0.5)
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()
    calibrate(args.model, args.num_steps, args.num_runs,
              args.target_skip, args.threshold, args.output)


if __name__ == '__main__':
    main()
