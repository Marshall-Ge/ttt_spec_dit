# -*- coding: utf-8 -*-
"""
Phase 1: Open-loop Feature Probing + Key-step Caching + Baseline Forecaster + Large-batch Parallel Verification
Base model: PixArt-alpha/PixArt-XL-2-512x512 (pure Transformer Blocks + adaLN)

Design (no weight modification, fully decoupled via PyTorch Hooks):
  * Task 1.1: register_forward_hook on transformer_blocks[14], capturing
             Z_t (block output), timestep embedding, text embedding -> global feature_buffer.
             Additional hook on adaln_single to capture embedded_timestep (needed for tail modulation).
  * Task 1.2: 50-step denoising loop; every KEY_INTERVAL=4 steps is a key step,
             key steps run full DiT forward and store Z_ref(t) in cache.
  * Task 1.3: Non-key steps skip full forward; use a parameter-free linear extrapolation
             forecaster to predict Z_draft(t-1), Z_draft(t-2)... from two nearest key-step references.
  * Task 1.4: Concatenate consecutive non-key-step draft features along batch dim,
             feed them in one pass through the tail network (blocks[15:] + norm_out + proj_out)
             for parallel forward, compare with exact features and print MSE.
  * Deliverables: print layer 14 feature shape; 50-step open-loop generation; MSE vs Timestep curve (baseline).
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

# --------------------------------------------------------------------------- #
# Global config
# --------------------------------------------------------------------------- #
MODEL_REPO = "PixArt-alpha/PixArt-XL-2-512x512"
MODEL_DIR = "./models"          # model download dir (huggingface cache)
DATA_DIR = "./output"            # output / artifact dir
PROBE_LAYER = 14               # probing point: output of transformer_blocks[14]
NUM_STEPS = 50                 # denoising steps
KEY_INTERVAL = 4               # key-step interval
PROMPT = "A majestic astronaut riding a horse on Mars, cinematic lighting, highly detailed"
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16

os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Task 1.1: Feature probing via forward hook
# --------------------------------------------------------------------------- #
class FeatureProbeHook:
    """Forward hook attached to transformer_blocks[PROBE_LAYER].

    Captures the block output Z_t along with its inputs:
      - timestep            : adaLN timestep embedding [B, Hidden]
      - encoder_hidden_states: projected text embedding [B, Seq, Hidden] (for cross-attn)
      - attention_mask / encoder_attention_mask : bias-format masks
    All written to an external buffer dict; no modification to model source code.
    """

    def __init__(self, buffer: dict):
        self.buffer = buffer

    def __call__(self, module, args, kwargs, output):
        self.buffer["Z_t"] = output.detach()                       # [B, Seq, Hidden]
        self.buffer["timestep_emb"] = kwargs["timestep"].detach()  # [B, Hidden]
        self.buffer["text_emb"] = kwargs["encoder_hidden_states"].detach()
        self.buffer["attention_mask"] = kwargs.get("attention_mask", None)
        self.buffer["encoder_attention_mask"] = kwargs.get("encoder_attention_mask", None)


def adaln_capture_hook(buffer: dict):
    """Hook on adaln_single to capture its returned (timestep_emb, embedded_timestep)."""
    def _hook(module, args, kwargs, output):
        # output = (linear(silu(emb)), emb) -> use emb for tail scale_shift modulation
        buffer["embedded_timestep"] = output[1].detach()
    return _hook


# --------------------------------------------------------------------------- #
# Tail network (probe point onwards) parallel forward (core of Task 1.4)
# --------------------------------------------------------------------------- #
def run_tail(transformer, hidden_states, timestep_emb, embedded_timestep,
             encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
    """Forward from after the probe point (block 14 output): blocks[15:] -> norm_out -> proj_out -> unpatchify.

    hidden_states: probe-point features [B, Seq, Hidden], can take arbitrary batch.
    Returns full model output [B, out_channels, H, W] (consistent with transformer.forward).
    """
    p = transformer.config.patch_size
    out_channels = transformer.config.out_channels
    H = W = transformer.config.sample_size // p

    # 2'. Remaining Transformer Blocks (15 .. 27)
    for block in transformer.transformer_blocks[PROBE_LAYER + 1:]:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep_emb,
            cross_attention_kwargs=None,
            class_labels=None,
        )

    # 3'. Output (identical to official forward)
    shift, scale = (
        transformer.scale_shift_table[None] + embedded_timestep[:, None].to(transformer.scale_shift_table.device)
    ).chunk(2, dim=1)
    hidden_states = transformer.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
    hidden_states = transformer.proj_out(hidden_states)
    hidden_states = hidden_states.squeeze(1)

    hidden_states = hidden_states.reshape(
        shape=(-1, H, W, p, p, out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(shape=(-1, out_channels, H * p, W * p))
    return output


# --------------------------------------------------------------------------- #
# Task 1.3: Parameter-free open-loop forecaster (linear extrapolation)
# --------------------------------------------------------------------------- #
def forecast_draft(z_prev_ref, z_cur_ref, k, interval=KEY_INTERVAL):
    """Given two nearest key-step reference features Z_ref(t+interval) and Z_ref(t),
    linearly extrapolate the draft feature for the k-th subsequent step (k=1,2,3,...).

        v = (z_cur - z_prev) / interval          # per-step "velocity"
        z_draft(t+k) = z_cur + k * v

    Zero learnable parameters. When only one reference exists (start), velocity=0 -> hold constant.
    """
    if z_prev_ref is None:
        return z_cur_ref.clone()
    v = (z_cur_ref - z_prev_ref) / interval
    return z_cur_ref + k * v


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()

    print("=" * 70)
    print("[Phase 0/5] Loading model ...")
    print("=" * 70)

    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_REPO, cache_dir=MODEL_DIR, torch_dtype=DTYPE,
    ).to(DEVICE)
    transformer = pipe.transformer
    transformer.requires_grad_(False).eval()
    vae = pipe.vae.eval()

    print(f"\n[Arch] num transformer_blocks = {len(transformer.transformer_blocks)}")
    print(f"[Arch] probe layer = transformer_blocks[{PROBE_LAYER}]")
    print(f"[Arch] inner_dim = {transformer.config.num_attention_heads * transformer.config.attention_head_dim}")
    print(f"[Arch] out_channels = {transformer.config.out_channels} (learned-sigma, eps = output[:, :4])")

    # ---- Scheduler ----
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    print(f"\n[Scheduler] {scheduler.__class__.__name__}, num_inference_steps={NUM_STEPS}")
    print(f"[Scheduler] timesteps[0..4] = {timesteps[:5].tolist()} ... timesteps[-1] = {timesteps[-1].item()}")

    # ---- Text encoding (once) ----
    prompt_embeds, prompt_attn_mask, _, _ = pipe.encode_prompt(
        PROMPT, do_classifier_free_guidance=False, num_images_per_prompt=1, device=DEVICE
    )
    print(f"\n[Text] T5 prompt_embeds shape = {tuple(prompt_embeds.shape)} (caption_channels=4096)")

    # ---- Initial latent ----
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    latents = torch.randn(
        (1, 4, transformer.config.sample_size, transformer.config.sample_size),
        device=DEVICE, dtype=DTYPE, generator=generator,
    ) * scheduler.init_noise_sigma

    # ------------------------------------------------------------------ #
    # Register hooks
    # ------------------------------------------------------------------ #
    feature_buffer = {}
    probe_handle = transformer.transformer_blocks[PROBE_LAYER].register_forward_hook(
        FeatureProbeHook(feature_buffer), with_kwargs=True
    )
    adaln_handle = transformer.adaln_single.register_forward_hook(
        adaln_capture_hook(feature_buffer), with_kwargs=True
    )

    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}  # sample_size=64 -> no micro-conditioning

    # Cache containers
    Z_ref = {}                       # i(key-step index) -> probe-point features
    Z_true_all = {}                  # i -> probe-point exact features (ground truth)
    full_out_all = {}                # i -> full model output (8 channels, for tail MSE)
    step_meta = {}                   # i -> (timestep_emb, embedded_timestep, text_emb, masks)
    Z_draft = {}                     # i(non-key step) -> draft features

    print("\n" + "=" * 70)
    print(f"[Phase 1/5] 50-step open-loop denoising (key_interval={KEY_INTERVAL})")
    print("=" * 70)

    pbar = tqdm(total=NUM_STEPS, desc="denoising", unit="step", ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for i, t in enumerate(timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        current_timestep = t.expand(latent_model_input.shape[0]).to(torch.int64)

        # ---- Full DiT forward (ground truth, for baseline measurement) ----
        with torch.no_grad():
            noise_pred_full = transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attn_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]  # [B, 8, H, W]  (learned-sigma)

        # Hook captured this step's probe-point features and context
        Z_true_all[i] = feature_buffer["Z_t"]
        full_out_all[i] = noise_pred_full.detach()
        step_meta[i] = (
            feature_buffer["timestep_emb"],
            feature_buffer["embedded_timestep"],
            feature_buffer["text_emb"],
            feature_buffer["attention_mask"],
            feature_buffer["encoder_attention_mask"],
        )

        # ---- Task 1.2: Key-step caching ----
        is_key = (i % KEY_INTERVAL == 0)
        if is_key:
            Z_ref[i] = feature_buffer["Z_t"]
            # Non-key steps after this key step use this reference for extrapolation
            i_prev = i - KEY_INTERVAL
            z_prev = Z_ref.get(i_prev, None)
            for k in range(1, KEY_INTERVAL):
                j = i + k
                if j >= NUM_STEPS:
                    break
                Z_draft[j] = forecast_draft(z_prev, Z_ref[i], k, KEY_INTERVAL)
                # Note: draft_meta (the non-key step's own timestep/text emb) will
                # be read from step_meta[j] during verification -- all forward passes are done by then.

        # ---- Standard scheduler step (normal generation) ----
        eps_true = noise_pred_full[:, :4]  # learned-sigma: take first 4 channels as eps
        latents = scheduler.step(eps_true, t, latents, return_dict=False)[0]

        # Update progress bar with per-step telemetry
        z_norm = feature_buffer["Z_t"].float().norm().item()
        marker = "★" if is_key else " "
        pbar.set_postfix_str(f"t={int(t):4d} | key={marker} | |Z|={z_norm:.2f}")
        pbar.update(1)

    pbar.close()
    t_denoise = time.time()
    print(f"  denoising completed in {t_denoise - t_start:.1f}s")

    probe_handle.remove()
    adaln_handle.remove()

    # ------------------------------------------------------------------ #
    # Task 1.4: Large-batch parallel verification -- concat consecutive non-key
    #           step drafts after each key step, single tail forward pass
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[Phase 2/5] Task 1.4: Large-batch parallel verification")
    print("=" * 70)

    mse_probe = {}   # i -> probe-point MSE (Z_draft vs Z_true)
    mse_tail = {}    # i -> tail-output MSE (run_tail(draft) vs full_out_true)

    num_key_steps_with_tail = sum(1 for i in range(0, NUM_STEPS, KEY_INTERVAL)
                                  if i + 1 < min(i + KEY_INTERVAL, NUM_STEPS))
    vbar = tqdm(total=num_key_steps_with_tail, desc="verify ", unit="segment", ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for i in range(0, NUM_STEPS, KEY_INTERVAL):
        non_key = [j for j in range(i + 1, min(i + KEY_INTERVAL, NUM_STEPS))]
        if not non_key:
            continue

        # ---- Concatenate all draft features for this segment along batch dim ----
        draft_batch = torch.cat([Z_draft[j] for j in non_key], dim=0)          # [K, Seq, Hidden]
        t_emb_batch = torch.cat([step_meta[j][0] for j in non_key], dim=0)    # [K, Hidden]
        ets_batch = torch.cat([step_meta[j][1] for j in non_key], dim=0)      # [K, Hidden]
        text_emb = step_meta[non_key[0]][2]                                    # [1, Seq, Hidden] (shared)
        text_batch = text_emb.expand(len(non_key), -1, -1)
        attn_mask = step_meta[non_key[0]][3]
        enc_attn_mask = step_meta[non_key[0]][4]
        if attn_mask is not None:
            attn_mask = attn_mask.expand(len(non_key), -1, -1)
        if enc_attn_mask is not None:
            enc_attn_mask = enc_attn_mask.expand(len(non_key), -1, -1)

        # ---- Single parallel tail forward ----
        with torch.no_grad():
            draft_full = run_tail(
                transformer, draft_batch, t_emb_batch, ets_batch,
                text_batch, attention_mask=attn_mask, encoder_attention_mask=enc_attn_mask,
            )  # [K, 8, H, W]

        # ---- Compare with exact features ----
        true_full_batch = torch.cat([full_out_all[j] for j in non_key], dim=0)
        for idx, j in enumerate(non_key):
            mp = ((Z_draft[j].float() - Z_true_all[j].float()) ** 2).mean().item()
            mt = ((draft_full[idx:idx + 1].float() - true_full_batch[idx:idx + 1].float()) ** 2).mean().item()
            mse_probe[j] = mp
            mse_tail[j] = mt

        # Summarise this segment in the progress bar
        stage = stage_of(non_key[0], NUM_STEPS)
        worst_probe = max(mse_probe[j] for j in non_key)
        vbar.set_postfix_str(f"key_t={int(timesteps[i]):4d} | k=1..{len(non_key)} | max probe MSE={worst_probe:.4e} | {stage}")
        vbar.update(1)

    vbar.close()
    t_verify = time.time()
    print(f"  verification completed in {t_verify - t_denoise:.1f}s")

    # ------------------------------------------------------------------ #
    # Deliverable 1: Feature shape verification
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[Phase 3/5] Deliverable 1: Feature shape verification")
    print("=" * 70)
    z0 = Z_true_all[0]
    print(f"  transformer_blocks[{PROBE_LAYER}] output Z_t shape = {tuple(z0.shape)}")
    print(f"    -> [Batch, Sequence_Length, Hidden_Dim] = [{z0.shape[0]}, {z0.shape[1]}, {z0.shape[2]}]")
    print(f"  (Batch=1, Seq=32x32=1024 tokens, Hidden=16x72=1152)")

    # ------------------------------------------------------------------ #
    # Deliverable 2: MSE vs Timestep plot
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[Phase 4/5] Deliverable 2: MSE Error vs. Timestep baseline report")
    print("=" * 70)

    steps_sorted = sorted(mse_probe.keys())
    ts_axis = [int(timesteps[j]) for j in steps_sorted]
    probe_vals = [mse_probe[j] for j in steps_sorted]
    tail_vals = [mse_tail[j] for j in steps_sorted]

    # Stage summary
    for stage_name, lo, hi in [("Early (Noise)", 0, NUM_STEPS // 3),
                               ("Middle (Structure)", NUM_STEPS // 3, 2 * NUM_STEPS // 3),
                               ("Late (Details)", 2 * NUM_STEPS // 3, NUM_STEPS)]:
        idxs = [j for j in steps_sorted if lo <= j < hi]
        if idxs:
            mp = np.mean([mse_probe[j] for j in idxs])
            mt = np.mean([mse_tail[j] for j in idxs])
            print(f"  {stage_name}: avg probe MSE={mp:.6e}  avg tail MSE={mt:.6e}  (n={len(idxs)})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ts_axis, probe_vals, "o-", label="Probing-point MSE (layer 14 hidden)", lw=2)
    ax.plot(ts_axis, tail_vals, "s--", label="Tail-output MSE (blocks 15->out)", lw=2)
    ax.set_xlabel("Timestep (t, larger = more noise)")
    ax.set_ylabel("MSE (draft vs exact)")
    ax.set_title("Phase 1 Baseline: Open-loop Forecaster Error vs Timestep\n"
                 f"PixArt-XL-2-512x512 | {NUM_STEPS} steps | key_interval={KEY_INTERVAL} | No TTT")
    ax.invert_xaxis()  # left=late details, right=early noise, following denoising timeline
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    # Stage separator annotations
    for stage_name, lo, hi, xfrac in [("Late (Details)", 2 * NUM_STEPS // 3, NUM_STEPS, 0.18),
                                      ("Middle (Structure)", NUM_STEPS // 3, 2 * NUM_STEPS // 3, 0.5),
                                      ("Early (Noise)", 0, NUM_STEPS // 3, 0.82)]:
        ax.text(xfrac, 0.92, stage_name, transform=ax.transAxes,
                ha="center", fontsize=10, color="gray",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6))
    fig.tight_layout()
    curve_path = os.path.join(DATA_DIR, "mse_vs_timestep_baseline.png")
    fig.savefig(curve_path, dpi=130)
    print(f"\n  Plot saved: {curve_path}")

    # Save data
    log = {
        "prompt": PROMPT, "num_steps": NUM_STEPS, "key_interval": KEY_INTERVAL,
        "probe_layer": PROBE_LAYER,
        "records": [
            {"step": int(j), "timestep": int(timesteps[j]),
             "stage": stage_of(j, NUM_STEPS),
             "mse_probe": float(mse_probe[j]), "mse_tail": float(mse_tail[j])}
            for j in steps_sorted
        ],
    }
    log_path = os.path.join(DATA_DIR, "mse_baseline_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved: {log_path}")

    # ------------------------------------------------------------------ #
    # Bonus: VAE decode
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("[Phase 5/5] VAE decode (final image)")
    print("=" * 70)
    try:
        with torch.no_grad():
            latents_vae = (latents / vae.config.scaling_factor).to(DTYPE)
            image = vae.decode(latents_vae).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(image).save(os.path.join(DATA_DIR, "generated_baseline.png"))
        print(f"  Image saved: {os.path.join(DATA_DIR, 'generated_baseline.png')}")
    except Exception as e:
        print(f"  (VAE decode skipped: {e})")

    t_end = time.time()
    print(f"\n{'=' * 70}")
    print(f"Phase 1 complete in {t_end - t_start:.1f}s total.")
    print(f"  Model load : {t_denoise - t_start:.1f}s")
    print(f"  Denoising  : {t_verify - t_denoise:.1f}s")
    print(f"  Verification: {t_end - t_verify:.1f}s")
    print(f"This curve serves as the core baseline for Phase 2 TTT integration.")
    print(f"{'=' * 70}")


def stage_of(i, num_steps):
    if i < num_steps // 3:
        return "Early (Noise)"
    elif i < 2 * num_steps // 3:
        return "Middle (Structure)"
    return "Late (Details)"


if __name__ == "__main__":
    main()
