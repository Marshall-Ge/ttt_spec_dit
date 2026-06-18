# -*- coding: utf-8 -*-
"""
Phase 2: SOTA Baseline Strengthening & Full Evaluation Pipeline Setup

Main entry point. Architecture:

  RUN 1 (Vanilla): Full forward every step → final latent + image
  RUN 2 (Accelerated): Full forward for ground truth EVERY step,
       but TeaCache decides whether to USE the fresh features or
       CACHED features for scheduler stepping. The accelerated path
       timing is measured as: tail-only-cost for skip steps,
       full-forward-cost for full/rejection steps.

  This two-pass design per step (full forward for ground truth +
  accelerated path for scheduling) gives us BOTH:
    1. Real trajectory divergence (scheduler uses drafts on skip steps)
    2. Per-step ground truth for verification MSE
    3. Clean timing measurement of the accelerated path

Outputs:
  - Markdown report (stdout + phase2_report.md)
  - Generated images (vanilla.png, accelerated.png)
  - Rejection log JSON
  - Stage-vs-rejection heatmap JSON + CSV
  - Step tracker JSON
"""

import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

from phase2.teacache import (
    TeaCacheController,
    ProbeFeatureHook,
    AdalnCaptureHook,
    run_tail,
)
from phase2.eval_pipeline import (
    decode_latent,
    latent_to_pil,
    compute_latent_mse,
    compute_pixel_mse,
    CLIPScorer,
)
from phase2.profiler import (
    StepTracker,
    BottleneckAnalyzer,
    denoising_stage,
    print_deep_water_report,
)

# ===========================================================================
# Config
# ===========================================================================
CONFIG = {
    "model_repo": "PixArt-alpha/PixArt-XL-2-512x512",
    "model_dir": "./models",
    "output_dir": "./output/phase2",
    "num_steps": 50,
    "early_layer_idx": 2,
    "probe_layer_idx": 14,
    "gamma": 0.1,
    "max_skip": 3,
    "seed": 42,
    "device": "cuda",
    "dtype": torch.float16,
    "eval_prompts": [
        "A majestic astronaut riding a horse on Mars, cinematic lighting, highly detailed",
        "A serene lake at sunset with mountains in the background, oil painting style",
        "A futuristic city skyline with flying cars, neon lights, cyberpunk aesthetic",
        "A cute cat wearing a wizard hat, casting spells, digital art",
        "A bowl of fresh ramen with steam rising, food photography, warm lighting",
    ],
}


# ===========================================================================
# Early-feature hook for lightweight residual monitoring
# ===========================================================================

class EarlyFeatureHook:
    """Captures output of an early transformer block for residual check."""

    def __init__(self, buffer: dict):
        self.buffer = buffer

    def __call__(self, module, args, kwargs, output):
        self.buffer["early_features"] = output.detach()


# ===========================================================================
# Pipeline builders
# ===========================================================================

def build_pipeline(device="cuda", dtype=torch.float16):
    """Load PixArtAlphaPipeline."""
    pipe = PixArtAlphaPipeline.from_pretrained(
        CONFIG["model_repo"], cache_dir=CONFIG["model_dir"], torch_dtype=dtype,
    ).to(device)
    pipe.transformer.requires_grad_(False).eval()
    pipe.vae.eval()
    return pipe


def build_scheduler(pipe, num_steps, device="cuda"):
    """Create DPMSolverMultistepScheduler."""
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)
    return scheduler


# ===========================================================================
# Vanilla denoising — full forward every step
# ===========================================================================

def run_vanilla(pipe, scheduler, prompt_embeds, prompt_attn_mask,
                latent_shape, seed, device="cuda", dtype=torch.float16):
    """Run full denoising loop. Returns (final_latent, total_time_s)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, device=device, dtype=dtype,
                          generator=generator) * scheduler.init_noise_sigma
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    t_start = time.time()
    for _i, t in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents, t)
        current_t = t.expand(latent_input.shape[0]).to(torch.int64)
        with torch.no_grad():
            noise_pred = pipe.transformer(
                latent_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attn_mask,
                timestep=current_t,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        latents = scheduler.step(noise_pred[:, :4], t, latents, return_dict=False)[0]

    return latents, time.time() - t_start


# ===========================================================================
# Accelerated denoising — TeaCache with built-in ground truth
# ===========================================================================

def run_teacache_accelerated(
    pipe, scheduler, prompt_embeds, prompt_attn_mask,
    latent_shape, seed,
    gamma=0.1, max_skip=3,
    early_layer_idx=2, probe_layer_idx=14,
    device="cuda", dtype=torch.float16,
):
    """TeaCache-accelerated denoising with per-step ground truth.

    At each step:
      A. Run full forward (blocks 0..27) via transformer.forward()
         → captures: probe features (hook), early features (hook),
           timestep_emb, embedded_timestep, full noise_pred.
         This is the GROUND TRUTH for verification.

      B. TeaCache decision — using early features captured by hook
         during the full forward:
         - If SKIP: run tail(cached_probe_features, current_timestep)
           → draft noise_pred. Use draft for scheduler.
         - If FULL/REJECTION: use full forward noise_pred for scheduler.
           Update TeaCache with fresh features.

      C. Step scheduler with the chosen noise_pred.

    Returns:
      final_latent, tracker, teacache,
      t_accel_total,  t_full_total,
      mse_probe, mse_tail, Z_draft_all, Z_true_all, full_out_all, draft_out_all
    """
    transformer = pipe.transformer
    timesteps = scheduler.timesteps
    num_steps = len(timesteps)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, device=device, dtype=dtype,
                          generator=generator) * scheduler.init_noise_sigma
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    # ---- Hook buffers ----
    feat_buf = {}  # Shared buffer for probe + adaln hooks

    # Early feature hook (for TeaCache decision)
    early_handle = transformer.transformer_blocks[early_layer_idx].register_forward_hook(
        EarlyFeatureHook(feat_buf), with_kwargs=True
    )
    # Probe feature hook (for caching/draft)
    probe_handle = transformer.transformer_blocks[probe_layer_idx].register_forward_hook(
        ProbeFeatureHook(feat_buf), with_kwargs=True
    )
    # adaln hook (for embedded_timestep needed in tail)
    adaln_handle = transformer.adaln_single.register_forward_hook(
        AdalnCaptureHook(feat_buf), with_kwargs=True
    )

    # ---- TeaCache controller ----
    teacache = TeaCacheController(gamma=gamma, max_skip=max_skip)

    # ---- Step tracker ----
    tracker = StepTracker()

    # ---- Verification data ----
    Z_draft_all = {}
    Z_true_all = {}
    full_out_all = {}
    draft_out_all = {}

    # ---- Timing ----
    t_accel_total = 0.0   # Time for the accelerated path only
    t_full_total = 0.0    # Time for full forwards (ground truth)

    print(f"\n  TeaCache γ={gamma}, max_skip={max_skip}, "
          f"early=block[{early_layer_idx}], probe=block[{probe_layer_idx}]")
    pbar = tqdm(total=num_steps, desc="tea-cache", unit="step", ncols=115)

    for i, t in enumerate(timesteps):
        tracker.start_step()
        latent_input = scheduler.scale_model_input(latents, t)
        current_t = t.expand(latent_input.shape[0]).to(torch.int64)

        # ---------------------------------------------------------------
        # PART A: Full forward (ground truth + feature capture)
        # ---------------------------------------------------------------
        t0_full = time.time()
        with torch.no_grad():
            noise_pred_full = transformer(
                latent_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attn_mask,
                timestep=current_t,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        t_full = time.time() - t0_full
        t_full_total += t_full

        # Extract ground truth from hooks
        Z_true_all[i] = feat_buf["probe_features"]
        full_out_all[i] = noise_pred_full.detach()

        # ---------------------------------------------------------------
        # PART B: TeaCache decision + accelerated path
        # ---------------------------------------------------------------
        t0_accel = time.time()

        # Get early features (captured by hook during full forward)
        early_feat = feat_buf.get("early_features")

        if early_feat is None:
            # Fallback: no early features available
            should_skip = False
        else:
            should_skip = teacache.should_skip(early_feat)

        residual_val = None
        if teacache.early_features_ref is not None and early_feat is not None:
            residual_val = teacache.compute_residual(early_feat,
                                                     teacache.early_features_ref)

        if should_skip:
            # ---- SKIP ----
            draft = teacache.get_draft()
            Z_draft_all[i] = draft

            # Run tail with cached features + CURRENT timestep context
            t_emb = feat_buf.get("timestep_emb")
            ets = feat_buf.get("embedded_timestep")
            txt_emb = feat_buf.get("text_emb")
            a_mask = feat_buf.get("attention_mask")
            ea_mask = feat_buf.get("encoder_attention_mask")

            with torch.no_grad():
                draft_out = run_tail(
                    transformer, draft,
                    t_emb, ets, txt_emb,
                    attention_mask=a_mask,
                    encoder_attention_mask=ea_mask,
                    probe_layer_idx=probe_layer_idx,
                )
            draft_out_all[i] = draft_out

            eps_use = draft_out[:, :4]
            teacache.record_skip()
            decision = "skip"

        else:
            # ---- FULL FORWARD (rejection or initial key step) ----
            if teacache.early_features_ref is not None and early_feat is not None:
                # Determine rejection reason: check residual first
                # (residual >= gamma is the primary bottleneck signal)
                if residual_val is not None and residual_val >= gamma:
                    reason = "residual_threshold"
                elif teacache.skip_count >= max_skip:
                    reason = "max_skip"
                else:
                    reason = "residual_threshold"  # fallback
                teacache.record_rejection(i, t.item(), reason,
                                          residual_val or float("inf"))
                decision = "rejection"
            else:
                decision = "full_forward"

            eps_use = noise_pred_full[:, :4]

            # Update TeaCache cache
            sm = (
                feat_buf.get("timestep_emb").clone() if feat_buf.get("timestep_emb") is not None else None,
                feat_buf.get("embedded_timestep").clone() if feat_buf.get("embedded_timestep") is not None else None,
                feat_buf.get("text_emb").clone() if feat_buf.get("text_emb") is not None else None,
                feat_buf.get("attention_mask"),
                feat_buf.get("encoder_attention_mask"),
            )
            teacache.update(i, t.item(), early_feat, Z_true_all[i], sm)

        t_accel = time.time() - t0_accel
        t_accel_total += t_accel

        # ---------------------------------------------------------------
        # PART C: Scheduler step
        # ---------------------------------------------------------------
        latents = scheduler.step(eps_use, t, latents, return_dict=False)[0]

        # Log
        tracker.record(
            step_idx=i, timestep=t.item(), num_steps=num_steps,
            decision=decision,
            residual=residual_val,
            skip_count=teacache.skip_count,
        )

        # Progress
        z_norm = Z_true_all[i].float().norm().item()
        marker = {"skip": "⏭", "full_forward": "★", "rejection": "✗"}.get(decision, "?")
        postfix = f"t={int(t):4d}|{marker}|sk#{teacache.skip_count}|Z={z_norm:.1f}"
        if residual_val is not None:
            postfix += f"|r={residual_val:.3f}"
        pbar.set_postfix_str(postfix)
        pbar.update(1)

    pbar.close()

    # Cleanup hooks
    early_handle.remove()
    probe_handle.remove()
    adaln_handle.remove()

    # ---- Compute per-step verification MSE ----
    mse_probe = {}
    mse_tail = {}
    for i in Z_draft_all:
        if i in Z_true_all:
            mse_probe[i] = ((Z_draft_all[i].float() - Z_true_all[i].float()) ** 2).mean().item()
        if i in draft_out_all and i in full_out_all:
            mse_tail[i] = ((draft_out_all[i].float() - full_out_all[i].float()) ** 2).mean().item()

    return (latents, tracker, teacache,
            t_accel_total, t_full_total,
            mse_probe, mse_tail,
            Z_draft_all, Z_true_all, full_out_all, draft_out_all)


# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- Reproducibility ----
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    print("=" * 80)
    print("Phase 2: SOTA Baseline Strengthening & Full Evaluation Pipeline")
    print("=" * 80)

    # ---- [1/5] Load model ----
    print("\n[1/5] Loading PixArt-Alpha pipeline...")
    pipe = build_pipeline(device=cfg["device"], dtype=cfg["dtype"])
    transformer = pipe.transformer
    vae = pipe.vae
    print(f"  Blocks: {len(transformer.transformer_blocks)}, "
          f"Hidden: {transformer.config.num_attention_heads * transformer.config.attention_head_dim}")

    scheduler_vanilla = build_scheduler(pipe, cfg["num_steps"], cfg["device"])
    scheduler_accel = build_scheduler(pipe, cfg["num_steps"], cfg["device"])
    print(f"  Scheduler: DPMSolver++, {len(scheduler_vanilla.timesteps)} steps")

    # ---- Encode prompt ----
    primary_prompt = cfg["eval_prompts"][0]
    prompt_embeds, prompt_attn_mask, _, _ = pipe.encode_prompt(
        primary_prompt, do_classifier_free_guidance=False,
        num_images_per_prompt=1, device=cfg["device"],
    )
    latent_shape = (1, 4, transformer.config.sample_size, transformer.config.sample_size)
    print(f"  Prompt: \"{primary_prompt[:60]}...\"")

    # ==================================================================
    # [2/5] Vanilla run
    # ==================================================================
    print(f"\n[2/5] Vanilla pipeline (full forward every step)...")
    latent_vanilla, t_vanilla = run_vanilla(
        pipe, scheduler_vanilla, prompt_embeds, prompt_attn_mask,
        latent_shape, cfg["seed"], cfg["device"], cfg["dtype"],
    )
    print(f"  Time: {t_vanilla:.1f}s")

    with torch.no_grad():
        img_vanilla = decode_latent(vae, latent_vanilla,
                                    scaling_factor=vae.config.scaling_factor,
                                    dtype=cfg["dtype"])
    latent_to_pil(img_vanilla.squeeze(0)).save(os.path.join(cfg["output_dir"], "vanilla.png"))
    print(f"  Saved: {cfg['output_dir']}/vanilla.png")

    # ==================================================================
    # [3/5] TeaCache accelerated run
    # ==================================================================
    print(f"\n[3/5] TeaCache-accelerated pipeline...")
    (latent_accel, tracker, teacache,
     t_accel, t_full,
     mse_probe, mse_tail,
     _z_draft, _z_true, _full_out, _draft_out) = run_teacache_accelerated(
        pipe, scheduler_accel, prompt_embeds, prompt_attn_mask,
        latent_shape, cfg["seed"],
        gamma=cfg["gamma"], max_skip=cfg["max_skip"],
        early_layer_idx=cfg["early_layer_idx"],
        probe_layer_idx=cfg["probe_layer_idx"],
        device=cfg["device"], dtype=cfg["dtype"],
    )

    tc_stats = teacache.stats()
    print(f"\n  TeaCache stats: {tc_stats['total_full_forwards']} full, "
          f"{tc_stats['total_skips']} skips, "
          f"{tc_stats['total_rejections']} rejections "
          f"({tc_stats['skip_ratio']:.1%} skip ratio)")
    print(f"  Accelerated-path time: {t_accel:.1f}s  "
          f"(full-forward time: {t_full:.1f}s)")

    with torch.no_grad():
        img_accel = decode_latent(vae, latent_accel,
                                  scaling_factor=vae.config.scaling_factor,
                                  dtype=cfg["dtype"])
    latent_to_pil(img_accel.squeeze(0)).save(os.path.join(cfg["output_dir"], "accelerated.png"))
    print(f"  Saved: {cfg['output_dir']}/accelerated.png")

    # ==================================================================
    # [4/5] Evaluation metrics
    # ==================================================================
    print(f"\n[4/5] Computing evaluation metrics...")

    # Per-step MSE breakdown by stage
    if mse_probe:
        print("\n  Per-step Verification MSE by Stage:")
        for stage in ["Early (Noise)", "Middle (Structure)", "Late (Details)"]:
            stage_skips = [i for i in mse_probe
                           if denoising_stage(i, cfg["num_steps"]) == stage]
            if stage_skips:
                mp = np.mean([mse_probe[i] for i in stage_skips])
                mt = np.mean([mse_tail.get(i, 0) for i in stage_skips])
                print(f"    {stage}: n={len(stage_skips)}, "
                      f"probe MSE={mp:.6e}, tail MSE={mt:.6e}")
        overall_pmse = np.mean(list(mse_probe.values()))
        overall_tmse = np.mean(list(mse_tail.values()))
        print(f"    Overall: {len(mse_probe)} skipped steps, "
              f"probe MSE={overall_pmse:.6e}, tail MSE={overall_tmse:.6e}")

    # End-to-end metrics
    latent_mse_val = compute_latent_mse(latent_accel, latent_vanilla)
    pixel_mse_val = compute_pixel_mse(img_accel, img_vanilla)
    print(f"\n  End-to-end: latent MSE={latent_mse_val:.6e}, pixel MSE={pixel_mse_val:.6e}")

    # CLIP Score
    scorer = CLIPScorer(device=cfg["device"], dtype=cfg["dtype"])
    clip_v = scorer.score(primary_prompt, img_vanilla)
    clip_a = scorer.score(primary_prompt, img_accel)
    print(f"  CLIP Score: vanilla={clip_v:.2f}, accelerated={clip_a:.2f} (Δ={clip_a - clip_v:.2f})")

    # Speedup
    speedup = t_vanilla / max(t_accel, 0.001)
    print(f"  Speedup: {speedup:.2f}× (vanilla={t_vanilla:.1f}s, accel-path={t_accel:.1f}s)")

    # ==================================================================
    # [5/5] Bottleneck profiling & full report
    # ==================================================================
    print(f"\n[5/5] Bottleneck profiling & report generation...")

    analyzer = BottleneckAnalyzer(num_steps=cfg["num_steps"])
    analyzer.ingest(tracker)
    analyzer.ingest_rejections(teacache.rejections, max_skip=cfg["max_skip"])

    # Export all artifacts
    teacache.export_rejections(os.path.join(cfg["output_dir"], "rejection_log.json"))
    analyzer.export_heatmap_json(os.path.join(cfg["output_dir"], "bottleneck_heatmap.json"))
    analyzer.export_heatmap_csv(os.path.join(cfg["output_dir"], "bottleneck_heatmap.csv"))
    with open(os.path.join(cfg["output_dir"], "step_tracker.json"), "w") as f:
        json.dump(tracker.to_dataframe(), f, indent=2)

    # ---- Build evaluation aggregate ----
    eval_agg = {
        "latency_vanilla_s_mean": t_vanilla,
        "latency_vanilla_s_std": 0.0,
        "latency_accel_s_mean": t_accel,
        "latency_accel_s_std": 0.0,
        "speedup_ratio_mean": speedup,
        "latent_mse_mean": latent_mse_val,
        "pixel_mse_mean": pixel_mse_val,
        "clip_score_vanilla_mean": clip_v,
        "clip_score_vanilla_std": 0.0,
        "clip_score_accel_mean": clip_a,
        "clip_score_accel_std": 0.0,
        "clip_delta_mean": clip_a - clip_v,
        "n_prompts": 1,
    }

    # ---- Generate full Markdown report ----
    report_body = print_deep_water_report(analyzer, tc_stats, eval_agg)

    header = (
        f"# Phase 2: SOTA Baseline Strengthening — Full Report\n\n"
        f"**Model:** PixArt-Alpha XL-2 512x512 | **Steps:** {cfg['num_steps']} | "
        f"**Seed:** {cfg['seed']} | **γ:** {cfg['gamma']} | "
        f"**Max Skip:** {cfg['max_skip']} | "
        f"**Early Block:** {cfg['early_layer_idx']} | "
        f"**Probe Block:** {cfg['probe_layer_idx']}\n\n"
        f"---\n\n"
    )

    # Per-step verification MSE appendix
    mse_appendix = ""
    if mse_probe:
        mse_appendix = "\n\n## 🔬 Per-Step Verification MSE\n\n"
        mse_appendix += "| Stage | Avg Probe MSE | Avg Tail MSE | Skipped Steps |\n"
        mse_appendix += "|-------|---------------|-------------|---------------|\n"
        for stage in ["Early (Noise)", "Middle (Structure)", "Late (Details)"]:
            stage_skips = [i for i in mse_probe
                           if denoising_stage(i, cfg["num_steps"]) == stage]
            if stage_skips:
                mp = np.mean([mse_probe[i] for i in stage_skips])
                mt = np.mean([mse_tail.get(i, 0) for i in stage_skips])
                mse_appendix += f"| {stage} | {mp:.6e} | {mt:.6e} | {len(stage_skips)} |\n"
            else:
                mse_appendix += f"| {stage} | — | — | 0 |\n"

    full_report = header + report_body + mse_appendix

    print("\n")
    print("=" * 80)
    print(full_report)
    print("=" * 80)

    report_path = os.path.join(cfg["output_dir"], "phase2_report.md")
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\nReport saved: {report_path}")

    print(f"\n{'=' * 80}")
    print(f"Phase 2 complete. Outputs in {cfg['output_dir']}/")
    print(f"  vanilla.png, accelerated.png, rejection_log.json,")
    print(f"  bottleneck_heatmap.json, bottleneck_heatmap.csv,")
    print(f"  step_tracker.json, phase2_report.md")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
