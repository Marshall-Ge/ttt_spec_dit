#!/usr/bin/env python3
"""Session 2 — Flywheel: load LoRA + exploit-mode calibrator + SpecA inference.

Runs 200 images × 20 steps through SpecA with:
  1. LoRA checkpoint from Session 1 (smoothens trajectory at check_layer=20)
  2. VFL OnlineCalibrator in exploit mode (allows threshold below static default)
  3. The calibrator learns error distributions during inference, gradually
     lowering thresholds as it accumulates data per (layer, bucket) stratum.
  4. Fresh SpecA state per image + DDIM scheduler + CFG.

Outputs: skip ratio, FLOPs reduction, wall time → flywheel_session2/results.json
"""

import sys; sys.path.insert(0, '/root/ttt_spec_dit')
import torch, os, json, time
import numpy as np
from diffusers import DDIMScheduler

from config import DIT_REPO
from models.dit import DiTTransformer2D, set_vfl_buffer, set_vfl_calibrator, set_vfl_step_info
from accelerators.speca import speca_init
from verification_feedback_loop import (
    StratifiedReplayBuffer, OnlineCalibrator, VFLConfig, AsyncTrainer,
)
from dataset.imagenet import ImageNetDataset

device = 'cuda'
dtype = torch.float16
num_steps = 20
n_images = 200
seed = 42
guidance_scale = 4.5

torch.manual_seed(seed)
np.random.seed(seed)

# =====================================================================
# 1. Load model
# =====================================================================
print("=" * 60)
print("SESSION 2 — Flywheel (LoRA + exploit-mode calibrator)")
print("=" * 60)
print(f"\n[1] Loading DiT model...")
t = DiTTransformer2D.from_pretrained(DIT_REPO).to(device=device, dtype=dtype).eval()
null_class = t.config.num_embeds_ada_norm
print(f"  Model loaded (null_class={null_class})")

# =====================================================================
# 2. VFL setup with exploit mode
# =====================================================================
print(f"\n[2] Setting up VFL calibrator (exploit mode)...")
buf = StratifiedReplayBuffer(capacity_per_stratum=200)
cal = OnlineCalibrator(ema_window=100)
cal.set_exploit_mode(True)  # allows thresholds below static default
set_vfl_buffer(buf, 'dit-v1')
set_vfl_calibrator(cal)

# =====================================================================
# 3. Load LoRA from Session 1
# =====================================================================
print(f"\n[3] Loading LoRA from Session 1...")
lora_ckpt = './output/flywheel_session1/vfl/lora_candidate_v001.pt'
if os.path.exists(lora_ckpt):
    trainer = AsyncTrainer(t, buf, config=VFLConfig(), base_model_version='dit-v1',
                           output_dir='./output/flywheel_session2/vfl')
    trainer.load_checkpoint(lora_ckpt)
    print(f"  LoRA loaded from {lora_ckpt}")
else:
    print(f"  WARNING: checkpoint not found at {lora_ckpt}")
    print(f"  Running without LoRA (calibrator-only flywheel)")

# =====================================================================
# 4. Dataset
# =====================================================================
print(f"\n[4] Loading ImageNet dataset ({n_images} images)...")
ds = ImageNetDataset(imagenet_dir='/root/autodl-fs/data/imagenet', n_images=n_images, seed=seed)

# =====================================================================
# 5. Scheduler
# =====================================================================
scheduler = DDIMScheduler.from_config(
    os.path.join(DIT_REPO, "scheduler", "scheduler_config.json"))
scheduler.set_timesteps(num_steps, device=device)

# =====================================================================
# 6. Generation loop
# =====================================================================
print(f"\n[5] Generating {n_images} images with SpecA + LoRA + exploit calibrator...")
print(f"    num_steps={num_steps}, guidance_scale={guidance_scale}")

t_start = time.time()
total_full = 0
total_taylor = 0
wall_times = []

for idx in range(n_images):
    t0 = time.time()
    data = ds[idx]
    class_id = data[2] if len(data) > 2 else data[1]

    # CFG labels: [cond, null]
    cond_labels = torch.tensor([class_id], device=device, dtype=torch.long)
    null_labels = torch.full((1,), null_class, device=device, dtype=torch.long)
    class_labels = torch.cat([cond_labels, null_labels], dim=0)

    # Init latent with CFG doubling
    gen = torch.Generator(device=device).manual_seed(100000 + idx)
    latent = torch.randn((1, 4, 32, 32), device=device, dtype=dtype, generator=gen)
    latent = latent * scheduler.init_noise_sigma
    latent = torch.cat([latent, latent], dim=0)

    # ---- Fresh SpecA state per image ----
    cache_dic, current = speca_init(
        num_steps=num_steps, base_threshold=0.01, decay_rate=0.01,
        min_taylor_steps=1, max_taylor_steps=4, max_order=4,
        num_layers=28, error_metric='cosine_similarity', check_layer=20,
    )

    # ---- Denoising loop ----
    for si, t_step in enumerate(scheduler.timesteps):
        current.step = num_steps - 1 - si
        set_vfl_step_info(si, num_steps)

        latent_input = scheduler.scale_model_input(latent, t_step)
        current_t = t_step.expand(2).to(torch.int64)

        noise_pred = t.forward_with_cfg(
            latent_input, current_t,
            current=current, cache_dic=cache_dic,
            class_labels=class_labels, cfg_scale=guidance_scale,
        )

        # Learned-sigma: keep noise channels, discard variance channels
        if t.config.out_channels // 2 == t.config.in_channels:
            noise_pred = noise_pred[:, :t.config.in_channels]

        latent = scheduler.step(noise_pred, t_step, latent, return_dict=False)[0]

    wall_s = time.time() - t0
    wall_times.append(wall_s)

    full_cnt = cache_dic.full_count
    total_full += full_cnt
    total_taylor += num_steps - full_cnt

    if (idx + 1) % 25 == 0:
        sr = full_cnt / num_steps
        print(f"  [{idx+1}/{n_images}] full={full_cnt}/{num_steps} "
              f"({sr:.0%} skip, {wall_s:.2f}s)")

elapsed = time.time() - t_start
total_steps = total_full + total_taylor
skip_ratio = total_taylor / total_steps if total_steps > 0 else 0.0

# =====================================================================
# 7. Results
# =====================================================================
results = {
    'config': {
        'model': 'dit',
        'method': 'speca+vfl+flywheel',
        'n_prompts': n_images,
        'num_steps': num_steps,
        'guidance_scale': guidance_scale,
        'lora_ckpt': lora_ckpt,
        'exploit_mode': True,
    },
    'aggregate': {
        'n_images': n_images,
        'full_steps': total_full,
        'taylor_steps': total_taylor,
        'skip_ratio': skip_ratio,
        'flops_reduction_est': skip_ratio,  # Taylor steps ≈ skipped FLOPs
        'wall_s_mean': float(np.mean(wall_times)),
        'wall_s_std': float(np.std(wall_times)),
        'wall_s_total': elapsed,
        'speed_img_per_s': n_images / elapsed if elapsed > 0 else 0,
        'vfl_events': buf.total_samples,
        'vfl_calibrator_updates': cal.total_updates,
        'vfl_strata': cal.num_strata,
    }
}

os.makedirs('./output/flywheel_session2', exist_ok=True)
with open('./output/flywheel_session2/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print("SESSION 2 RESULTS")
print(f"{'=' * 60}")
print(f"  n_images:       {n_images}")
print(f"  full_steps:     {total_full}")
print(f"  taylor_steps:   {total_taylor}")
print(f"  skip_ratio:     {skip_ratio:.1%}")
print(f"  FLOPs reduction:{skip_ratio:.1%} (estimated from skip ratio)")
print(f"  wall_s_mean:    {elapsed/n_images:.2f}s/image")
print(f"  wall_s_total:   {elapsed:.1f}s")
print(f"  images/sec:     {n_images/elapsed:.1f}")
print(f"  VFL events:     {buf.total_samples}")
print(f"  VFL strata:     {cal.num_strata}")
print(f"  Results → ./output/flywheel_session2/results.json")
print(f"{'=' * 60}")
