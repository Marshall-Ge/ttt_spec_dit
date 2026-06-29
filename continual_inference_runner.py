# -*- coding: utf-8 -*-
"""Session-Level Test-Time Training (TTT) runner for DiT-2-256.

Phase 3 architectural pivot — Inter-Image Session-Level TTT (Continuous Stream
Adaptation). This script orchestrates a "session": a sequential stream of N
semantically-related images that share the same ImageNet class but use
independent Gaussian noise initialisations $z_0^{(i)}$. A microscopic,
persistent plugin ($\\phi$) is attached to the frozen DiT backbone ($\\Theta$,
$\\nabla_\\Theta = 0$). On each image:

  * calc steps distil the full 28-block backbone into $\\phi$ (one AdamW step
    per calc step, plugin-only);
  * skip steps bypass the 28 blocks and route the stale cache through $\\phi$.

As the session progresses, $\\phi$ learns the class-specific semantic manifold,
enabling the **Flywheel Effect**: later images can sustain extreme skip ratios
(>80%) without fidelity collapse.

==== Semantic-manifold adaptation note ====================================
DiT-2-256 is a class-conditional model with NO text encoder. The task spec's
"semantically consistent prompt manifold" is realised as a FIXED ImageNet
class label combined with N independent Gaussian noise seeds — the DiT
equivalent of "same subject, different variations". The plugin learns the
denoising-trajectory manifold of that class.
==========================================================================

==== Dynamic γ curriculum (Task 2, revised for Micro-Epoch) =============
The TeaCache threshold γ is scheduled per image to force exploration before
exploitation:
    Images 1-5   (extended burn-in): γ = 0.35  (5 imgs × ~5 calc × 3 me = 75 updates)
    Images 6-12  (transition):       γ = 0.55
    Images 13-20 (exploitation):     γ = 0.75  (extreme acceleration reliance)
==========================================================================

Usage::

    python continual_inference_runner.py \
        --num_steps 20 --session_class 207 --n_images 20 \
        --guidance_scale 4.5 --lr 1e-4 \
        --output_dir ./output/ttt_session

Outputs a CSV (Task 4 telemetry) tracking the Flywheel Effect per image.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    DIT_REPO, OUTPUT_DIR, DEFAULT_NUM_STEPS, DEFAULT_GUIDANCE_SCALE,
    DDIM_FLOP_MATCHED_STEPS, load_coefficients,
)
from utils import save_image

from models.dit import DiTGenerator
from models.ttt_plugin import (
    SessionAdaLNModulator, ttt_state_init, ttt_reset_for_image,
    ttt_avg_loss, ttt_skip_ratio, ttt_session_stats,
)
from accelerators.teacache import teacache_init, teacache_reset, teacache_stats


# ===========================================================================
# γ curriculum (Task 2)
# ===========================================================================

def gamma_schedule(image_index: int) -> float:
    """Dynamic γ threshold for TeaCache, per image index (1-based).

    - 1-5  (extended burn-in): 0.35  → 5×5×micro_epochs=75 updates w/ me=3
    - 6-12 (transition):       0.55
    - 13+  (exploitation):     0.75  → plugin sustains extreme skip ratios
    """
    if image_index <= 5:
        return 0.35
    elif image_index <= 12:
        return 0.55
    else:
        return 0.75


# ===========================================================================
# Baseline reference pre-cache (Task 4)
# ===========================================================================

def precompute_baselines(generator: DiTGenerator,
                        session_class: int,
                        seeds: List[int],
                        num_steps: int,
                        guidance_scale: float,
                        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run a vanilla full-compute generation for each (class, seed) pair.

    The TeaCache cache and the plugin are NOT involved — this is the
    fidelity reference against which the TTT session's Latent/Pixel MSE is
    measured. Results are kept in memory (CPU side) to avoid disk round-trips.

    Returns
    -------
    ref_latents : list of (4, 32, 32) float tensors (cond half, on CPU)
    ref_images  : list of (3, 256, 256) float tensors in [0,1] (on CPU)
    """
    print(f"\n[baseline] Pre-computing {len(seeds)} full-compute references "
          f"(class={session_class}, {num_steps} steps)...")
    ref_latents, ref_images = [], []
    t0 = time.time()
    for i, sd in enumerate(seeds):
        latent, image = generator.generate(
            session_class, sd,
            guidance_scale=guidance_scale,
            method="baseline",
        )
        # Move to CPU/fp32 for stable MSE comparison later.
        ref_latents.append(latent.detach().float().cpu())
        ref_images.append(image.detach().float().cpu())
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  baseline {i+1}/{len(seeds)} done "
                  f"({(time.time()-t0)/(i+1):.1f}s/img)")
    print(f"  [baseline] done in {time.time()-t0:.1f}s")
    return ref_latents, ref_images


# ===========================================================================
# Session loop
# ===========================================================================

def run_session(args) -> Dict:
    device = "cuda"
    dt = torch.float16

    n_images = args.n_images
    num_steps = args.num_steps
    session_class = args.session_class
    guidance_scale = args.guidance_scale

    # Independent Gaussian noise seeds for the semantic manifold.
    base_seed = args.seed
    seeds = [base_seed + 1000 * (i + 1) for i in range(n_images)]

    # ------------------------------------------------------------------
    # [0] Load DiT + freeze backbone
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Session-Level TTT — Inter-Image Continuous Stream Adaptation")
    print(f"  Model:        DiT-2-256 (frozen Θ)")
    print(f"  Session:      {n_images} images, class={session_class}")
    print(f"  Steps/image:  {num_steps}")
    print(f"  Guidance:     {guidance_scale}")
    print(f"  Plugin LR:    {args.lr}")
    print(f"  γ schedule:   burn-in 0.35 (1-5) → 0.55 (6-12) → 0.75 (13+)")
    print(f"  micro_epochs: {args.micro_epochs} (per calc step)")
    print("=" * 70)

    print("\n[0] Loading DiT-2-256...")
    generator = DiTGenerator(num_steps=num_steps, device=device, dtype=dt)
    generator.load()

    # ---- Freeze the backbone (strict ∇_Θ = 0). NOT done anywhere by default;
    #      the repo relies on @torch.no_grad(), which the TTT path must defeat.
    transformer = generator.transformer
    vae = generator.vae
    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    transformer.eval()
    vae.eval()
    n_bb = sum(p.numel() for p in transformer.parameters())
    n_bb_grad = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"  backbone frozen: {n_bb_grad}/{n_bb} params require grad")

    # ------------------------------------------------------------------
    # [1] Build plugin + persistent optimizer (ONCE, outside image loop)
    # ------------------------------------------------------------------
    print("\n[1] Building SessionAdaLNModulator...")
    # Detect hidden dim from the model (1152 for DiT-2-256).
    hidden_dim = transformer.config.attention_head_dim * \
        transformer.config.num_attention_heads
    # Plugin MUST be fp32: the calc-step hidden-state MSE can exceed fp16
    # max (65504), producing NaN during backward. The plugin is tiny (0.92M
    # params), so keeping it in fp32 adds negligible overhead.
    plugin = SessionAdaLNModulator(hidden_dim=hidden_dim, mid_dim=192).to(
        device=device, dtype=torch.float32)
    plugin.train()  # plugin learns; backbone stays .eval()
    n_plug = plugin.num_parameters()
    print(f"  plugin params: {n_plug:,}  ({n_plug/1e6:.3f}M, budget <1M)")

    # Load DiT TeaCache coefficients (step-count specific — see README pitfall 8).
    coef_path = os.path.join(os.path.dirname(__file__), "dit_coef.json")
    coefficients = load_coefficients(coef_path) if os.path.exists(coef_path) \
        else load_coefficients()

    ttt_state = ttt_state_init(num_steps=num_steps, plugin=plugin,
                               lr=args.lr,
                               micro_epochs=args.micro_epochs)

    # ------------------------------------------------------------------
    # [2] Pre-cache baseline references
    # ------------------------------------------------------------------
    ref_latents, ref_images = precompute_baselines(
        generator, session_class, seeds, num_steps, guidance_scale)

    # ------------------------------------------------------------------
    # [3] Session loop (k = 1 .. n_images)
    # ------------------------------------------------------------------
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"ttt_session_c{session_class}_s{num_steps}")
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    print(f"\n[3] Running {n_images}-image TTT session → {output_dir}")
    csv_rows: List[Dict] = []
    session_t0 = time.time()

    for k in range(1, n_images + 1):
        gamma = gamma_schedule(k)
        sd = seeds[k - 1]

        # (a) Rebuild TeaCache state for this image with the curriculum γ.
        #     The plugin weights + optimizer momentum PERSIST (never reset).
        teacache_state = teacache_init(
            num_steps=num_steps,
            rel_l1_thresh=gamma,
            coefficients=coefficients,
        )
        # (b) Reset per-image TTT telemetry only.
        ttt_reset_for_image(ttt_state)

        # (c) Generate with TTT (teacher/student dispatch inside).
        t_img0 = time.time()
        latent_k, image_k = generator.generate_ttt(
            session_class, sd,
            guidance_scale=guidance_scale,
            teacache_state=teacache_state,
            ttt_state=ttt_state,
        )
        img_wall = time.time() - t_img0

        # (d) Fidelity vs the pre-cached baseline.
        latent_k_cpu = latent_k.detach().float().cpu()
        image_k_cpu = image_k.detach().float().cpu()
        latent_mse = F.mse_loss(latent_k_cpu, ref_latents[k - 1]).item()
        pixel_mse = F.mse_loss(image_k_cpu, ref_images[k - 1]).item()

        # (e) Telemetry.
        skip_ratio = ttt_skip_ratio(ttt_state)
        avg_loss = ttt_avg_loss(ttt_state)
        st = teacache_stats(teacache_state)

        # Save image (keep all — session is small).
        save_image(image_k_cpu, os.path.join(gen_dir, f"{k:02d}_gamma{gamma}.png"))

        row = {
            "Image_Index": k,
            "Target_Gamma": gamma,
            "Actual_Skip_Ratio": round(skip_ratio, 2),
            "Average_Plugin_Loss": (round(avg_loss, 6)
                                    if avg_loss == avg_loss else "nan"),
            "Latent_MSE": latent_mse,
            "Pixel_MSE": pixel_mse,
            "Calc_Steps": st.get("total_calc", 0),
            "Skip_Steps": st.get("total_skip", 0),
            "Wall_s": round(img_wall, 2),
        }
        csv_rows.append(row)
        print(f"  [k={k:02d}] γ={gamma:.2f} skip={skip_ratio:5.1f}%  "
              f"loss={avg_loss if avg_loss==avg_loss else float('nan'):.5f}  "
              f"latent_mse={latent_mse:.5f} pixel_mse={pixel_mse:.4f}  "
              f"({img_wall:.1f}s)")

    session_wall = time.time() - session_t0
    print(f"\n  session wall: {session_wall:.1f}s "
          f"({session_wall/n_images:.1f}s/image)")

    # ------------------------------------------------------------------
    # [4] Write CSV (Task 4) + session summary
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "ttt_session_telemetry.csv")
    fieldnames = ["Image_Index", "Target_Gamma", "Actual_Skip_Ratio",
                  "Average_Plugin_Loss", "Latent_MSE", "Pixel_MSE",
                  "Calc_Steps", "Skip_Steps", "Wall_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"\n[4] CSV → {csv_path}")

    # Flywheel summary.
    stats = ttt_session_stats(ttt_state)
    print("\n" + "=" * 70)
    print("FLYWHEEL SUMMARY")
    print("=" * 70)
    print(f"  plugin params:        {stats['plugin_params']:,}")
    print(f"  total optimizer steps:{stats['trained_steps']}")
    print(f"  session loss first:   {stats['session_loss_first']:.6f}")
    print(f"  session loss mean:    {stats['session_loss_mean']:.6f}")
    print(f"  session loss last:    {stats['session_loss_last']:.6f}")

    # Skip-ratio trend across γ phases.
    phases = {"burn-in (1-5)": [], "transition (6-12)": [], "exploit (13+)": []}
    for r in csv_rows:
        k = r["Image_Index"]
        if k <= 5:
            phases["burn-in (1-5)"].append(r["Actual_Skip_Ratio"])
        elif k <= 12:
            phases["transition (6-12)"].append(r["Actual_Skip_Ratio"])
        else:
            phases["exploit (13+)"].append(r["Actual_Skip_Ratio"])
    print("  skip-ratio by phase:")
    for name, vals in phases.items():
        if vals:
            print(f"    {name:22s}: mean={np.mean(vals):5.1f}%  "
                  f"(range {min(vals):.1f}-{max(vals):.1f}%)")

    # Save a JSON summary too.
    import json
    summary = {
        "config": {
            "model": "dit", "session_class": session_class,
            "n_images": n_images, "num_steps": num_steps,
            "guidance_scale": guidance_scale, "lr": args.lr,
            "plugin_params": stats["plugin_params"],
            "micro_epochs": args.micro_epochs,
            "gamma_schedule": "1-5:0.35 / 6-12:0.55 / 13+:0.75",
        },
        "session_stats": stats,
        "per_image": [{k: v for k, v in r.items()} for r in csv_rows],
        "phase_skip_ratio": {
            name: {"mean": float(np.mean(v)), "min": float(min(v)),
                   "max": float(max(v))}
            for name, v in phases.items() if v
        },
    }
    with open(os.path.join(output_dir, "ttt_session_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {os.path.join(output_dir, 'ttt_session_summary.json')}")
    print("=" * 70)
    return summary


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Session-Level TTT runner for DiT-2-256 (Phase 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS,
                   help=f"Denoising steps per image (default {DEFAULT_NUM_STEPS}). "
                        "NOTE: TeaCache coefficients are step-count specific — "
                        "re-calibrate (scripts/calibrate_teacache.py) if changed.")
    p.add_argument("--session_class", type=int, default=207,
                   help="Fixed ImageNet class for the semantic manifold "
                        "(default 207 = 'golden retriever').")
    p.add_argument("--n_images", type=int, default=20,
                   help="Session length N (default 20).")
    p.add_argument("--guidance_scale", type=float,
                   default=DEFAULT_GUIDANCE_SCALE,
                   help=f"CFG scale (default {DEFAULT_GUIDANCE_SCALE}).")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Plugin AdamW learning rate (default 1e-4).")
    p.add_argument("--micro_epochs", type=int, default=3,
                   help="Per calc-step micro-epochs to squeeze z_true "
                        "(default 3). 1 = single-pass (fastest), "
                        "3-5 = better sample efficiency.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; per-image seeds = seed + 1000*(i+1).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (auto if omitted).")
    p.add_argument("--coef_path", type=str, default=None,
                   help="Path to DiT TeaCache coefficient JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(DIT_REPO):
        print(f"[ERROR] DIT_REPO not found: {DIT_REPO}")
        sys.exit(1)
    run_session(args)


if __name__ == "__main__":
    main()
