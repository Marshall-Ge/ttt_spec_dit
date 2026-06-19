"""Quick IS fix verification: generate a few PixArt images and compute IS."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from phase2.config import PIXART_REPO, HF_CACHE_DIR
from phase2.eval_pipeline import (
    decode_latent, InceptionScoreComputer, MSCOCO_SAMPLE_PROMPTS,
)

print("=" * 70)
print("IS Fix Verification")
print("=" * 70)

device, dtype = "cuda", torch.float16

# 1. Load model
print("\n[1] Loading PixArt pipeline...")
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
pipe = PixArtAlphaPipeline.from_pretrained(
    PIXART_REPO, cache_dir=HF_CACHE_DIR, torch_dtype=dtype,
).to(device)
pipe.transformer.eval()
pipe.vae.eval()
print(f"  loaded. blocks={len(pipe.transformer.transformer_blocks)}")

# 2. Generate images
n_imgs = 6
prompts = MSCOCO_SAMPLE_PROMPTS[:n_imgs]
print(f"\n[2] Generating {n_imgs} images...")

latent_shape = (1, 4, pipe.transformer.config.sample_size, pipe.transformer.config.sample_size)
vae = pipe.vae
scaling_factor = vae.config.scaling_factor

images = []
for i, prompt in enumerate(prompts):
    seed = 100 + i
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(20, device=device)

    latents = torch.randn(latent_shape, device=device, dtype=dtype,
                          generator=generator) * scheduler.init_noise_sigma

    pe, am, _, _ = pipe.encode_prompt(
        prompt, do_classifier_free_guidance=False,
        num_images_per_prompt=1, device=device,
    )
    added = {"resolution": None, "aspect_ratio": None}

    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc=f"  img {i+1}/{n_imgs}", ncols=80, leave=False):
            latent_input = scheduler.scale_model_input(latents, t)
            current_t = t.expand(1).to(torch.int64)
            noise_pred = pipe.transformer(
                latent_input, encoder_hidden_states=pe,
                encoder_attention_mask=am, timestep=current_t,
                added_cond_kwargs=added, return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred[:, :4], t, latents, return_dict=False)[0]

    img = decode_latent(vae, latents, scaling_factor, dtype)
    images.append(img.squeeze(0))
    print(f"  [{i+1}] generated: shape={img.shape}, range=[{img.min().item():.4f}, {img.max().item():.4f}]")

# 3. Compute IS
print(f"\n[3] Computing Inception Score on {len(images)} images...")
is_computer = InceptionScoreComputer(device=device, splits=3)  # 3 splits for 6 imgs
for i, img in enumerate(images):
    print(f"  add image {i+1}: dtype={img.dtype}, shape={img.shape}, range=[{img.min().item():.4f}, {img.max().item():.4f}]")
    is_computer.add(img)

is_mean, is_std = is_computer.compute()
print(f"\n  IS = {is_mean:.4f} ± {is_std:.4f}")

if np.isnan(is_mean):
    print("\n  ❌ FAIL: IS is NaN!")
    sys.exit(1)
elif is_mean <= 1.0:
    print(f"\n  ⚠️  WARNING: IS={is_mean:.4f} ≤ 1.0 (unusually low, check images)")
else:
    print(f"\n  ✅ PASS: IS={is_mean:.4f} > 1.0 and non-NaN")

# 4. Quick sanity: edge case with 1 image
print(f"\n[4] Edge case: IS with 1 image...")
is2 = InceptionScoreComputer(device=device, splits=10)
is2.add(images[0])
m, s = is2.compute()
print(f"  IS(1 image) = {m:.4f} ± {s:.4f}")
if np.isnan(m):
    print("  ❌ FAIL: 1-image IS is NaN!")
    sys.exit(1)
else:
    print("  ✅ Edge case passed")

print(f"\n{'='*70}")
print("All IS checks passed!")
print(f"{'='*70}")
