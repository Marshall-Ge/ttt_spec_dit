# -*- coding: utf-8 -*-
"""DiT-2-256 c2i entry point — explicit sampling loop, no monkeypatching.

Orchestrator: ``DiTGenerator`` manages VAE / scheduler / device / dtype /
prompt-encoding / metrics coordination. The transformer is the new
``DiTTransformer2D`` (explicit forward with SpecA branching).

Top-level entry: ``run_c2i(args)``.

Sampling methods (controlled by ``args.method``):
  - baseline (DDIM full steps)
  - teacache  (TeaCache residual reuse at loop level)
  - ddim      (DDIM step-skipping, fewer steps)
  - speca     (Speculative Taylor acceleration via cache_dic/current)
"""

import copy
import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler

from config import (
    DIT_REPO, IMAGENET_DIR, OUTPUT_DIR, load_coefficients,
)
from utils import (
    decode_latent, save_image, pil_to_tensor, ensure_real_299,
    get_vfl_checkpoint_dir, prune_checkpoints, latent_seed_for_index,
    _clean,
    _compute_generated_fid_is,
    _covr_bandit_sentinel_selection,
    _covr_forced_sentinel_selection,
    _covr_hash_index,
    _covr_hash_sample,
    _covr_log_snr,
    _covr_online_accounting,
    _covr_scalar,
    _covr_scheduler_config_json,
    _covr_sentinel_selection,
    _dataset_generation_window,
    _load_forced_covr_template,
    _record_profile_stage,
)

# Shared pure lifecycle helpers (version/resume/profiler/canonical-JSON).
# Re-exported so existing callers keep importing the historical names from
# ``run_dit`` (scripts/benchmark_safety_proxy.py, tests/*).
from run_dit_shared import (
    _GenerationProfiler,
    _covr_canonical_json,
    _covr_resume_metadata,
    _covr_scheduler_config_json,
    _covr_sentinel_selection,
)  # noqa: F401  (re-exported compatibility surface)

# COVR sampling-loop hooks (functions needing transformer/scheduler access).
from pipelines.hooks.covr_hook import (  # noqa: F401  (re-exported surface)
    _cache_scheduler_timestep_values,
    _covr_context,
    _covr_full_rollout,
    _covr_scheduler_alphas,
    _covr_scheduler_pair,
    _covr_shadow_full,
    _covr_teacache_terminal_skip,
    _covr_transition_components,
)

# DiTGenerator moved to pipelines/dit.py (see project-structure.md §2).
from pipelines.dit import DiTGenerator

from models.dit import (
    DiTTransformer2D, set_vfl_step_info, get_vfl_buffer, set_vfl_sample_id,
)
from feedback.vfl.lora_adapter import (
    set_lora_t_emb, clear_lora_t_emb,
    compute_timestep_emb_for_transformer,
    attach_lora_all_layers, count_lora_params,
    _swap_lora_weights, _load_state_into_wrappers,
)
from accelerators.teacache import (
    teacache_boundary_snapshot, teacache_init, teacache_step,
    teacache_reset, teacache_stats,
)
from accelerators.covr_viability import COVRViabilityRecorder
from accelerators.speca import SpecACache, SpecAState, speca_init
from accelerators.strategy_dispatch import apply_strategy
from accelerators.registry import get_adapter, is_registered
from accelerators.covr import (
    ActionAuditContext,
    ActionAuditEvent,
    ActionAuditRecorder,
    COVRAction,
    ddim_epsilon_transition_coefficients,
    transition_defect_batch,
)
from accelerators.covr_bandit import TemplateManifest
from accelerators.covr_runtime import (
    COVRMode,
    COVRRuntime,
    COVRTrajectoryAssignment,
    build_covr_runtime_config,
    flatten_prefix_features,
    load_strategy_manifest,
)
from accelerators.covr_viability import extract_prefix_features
from models.ttt_plugin import (
    SessionAdaLNModulator, ttt_state_init, ttt_reset_for_image,
    ttt_train_step, ttt_record_skip, ttt_session_stats,
)

from eval.fid_is import FIDISComputer
from eval.latency import LatencyMetric, FLOPsMetric


# ===========================================================================
# Valid metrics for c2i
# ===========================================================================

C2I_VALID_METRICS = {
    "coco":     {"fid", "is", "clip", "lpips", "mse", "latency", "flops", "speed"},
    "imagenet": {"fid", "is", "latency", "flops", "speed"},
}

# ---- DiT constants ----
DIT_IMAGE_SIZE = 256
DIT_LATENT_SIZE = 32
DIT_NULL_CLASS = 1000


# DiTGenerator — orchestrator (no monkeypatching)
# ===========================================================================

# ===========================================================================
# TTT plugin setup helper
# ===========================================================================

def _setup_ttt(generator: "DiTGenerator", args):
    """Create TTT plugin and state for the full c2i pipeline."""
    transformer = generator.transformer
    hidden_dim = (transformer.config.attention_head_dim *
                  transformer.config.num_attention_heads)
    plugin = SessionAdaLNModulator(hidden_dim=hidden_dim, mid_dim=192).to(
        device=generator.device, dtype=torch.float32)
    plugin.train()
    return ttt_state_init(num_steps=args.num_steps, plugin=plugin,
                          lr=args.ttt_lr,
                          micro_epochs=args.ttt_micro_epochs)


def _covr_generate_terminal_fallback(
        generator, prompts, seeds, guidance_scale: float,
        trajectory: COVRTrajectoryAssignment,
        profiler: _GenerationProfiler):
    """Run a profiled full baseline without collecting COVR feedback twice."""
    profiling_trajectory = COVRTrajectoryAssignment(
        trajectory_id=trajectory.trajectory_id,
        sample_count=trajectory.sample_count,
        strategy=None,
        bandit_assignment=None,
        sentinel_selected=False,
        sentinel_start_idx=None,
        sample_ids=(),
        session_id=trajectory.session_id,
        version=trajectory.version,
        profiler=profiler,
    )
    return generator.generate(
        prompts,
        seeds,
        guidance_scale=guidance_scale,
        method="baseline",
        covr_trajectory=profiling_trajectory,
    )


# ===========================================================================
# run_c2i — top-level evaluation entry point for DiT
# ===========================================================================

def run_c2i(args) -> Dict:
    """Run a DiT c2i evaluation (ImageNet only).

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    dict with keys: config, aggregate
    """
    dataset_name = args.dataset
    if dataset_name not in C2I_VALID_METRICS:
        raise ValueError(
            f"Unknown c2i dataset: {dataset_name}. "
            f"Valid: {list(C2I_VALID_METRICS.keys())}")

    valid_metrics = C2I_VALID_METRICS[dataset_name]
    requested = set(args.metrics)
    for m in sorted(requested - valid_metrics):
        print(f"  [WARN] '{m}' is not valid for c2i/{dataset_name} — skipping")
    selected = sorted(requested & valid_metrics)
    if not selected:
        print(f"  [ERROR] No valid metrics remain for c2i/{dataset_name}.")
        return {}

    # ---- Device / dtype ----
    if getattr(args, "debug", False):
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            print("[DEBUG MODE] MPS not available, falling back to cpu")
            device = "cpu"
        dt = torch.float32
        print("=" * 60)
        print("[DEBUG MODE] 1 layer, mps, fp32 — metrics meaningless")
        print("[DEBUG MODE] for smoke-test only, do NOT trust FID/IS/CLIP")
        print("=" * 60)
    else:
        device = "cuda"
        dt = torch.float16

    # Output dir
    dir_suffix = f"{args.method}_{args.num_steps}" if args.method == "ddim" else args.method
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"c2i_dit_{dataset_name}_{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    dataset_start_index = int(getattr(args, "dataset_start_index", 0))
    covr_bandit_state_path = None
    covr_resume = None
    covr_resume_sample_offset = 0
    covr_runtime_config = build_covr_runtime_config(args, output_dir)
    if getattr(args, "covr_template_bandit", False) or getattr(
            args, "covr_strategy_bandit", False):
        covr_bandit_state_path = getattr(args, "covr_bandit_state", None) or (
            covr_runtime_config.bandit_state_path
            if covr_runtime_config is not None
            else os.path.join(output_dir, "covr", "template_bandit_state.json"))
        covr_resume = _covr_resume_metadata(
            covr_runtime_config.resume_state_path
            if covr_runtime_config is not None else None)
        if covr_resume is not None:
            covr_resume_sample_offset = int(covr_resume["processed_samples"])
            requested_session = getattr(args, "covr_session_id", None)
            if (requested_session is not None and
                    requested_session != covr_resume["session_id"]):
                raise ValueError(
                    "--covr-session-id does not match the persisted bandit state")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = load_coefficients(args.coef_path) if args.coef_path else load_coefficients()

    print("=" * 70)
    print(f"DiT-2-256 C2I Evaluation — {dataset_name.upper()}")
    print(f"  Method:   {args.method}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  N:        {args.n_prompts}")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Metrics:  {selected}")
    print(f"  Output:   {output_dir}")
    if args.method == "teacache":
        print(f"  γ:        {args.thresh}")
    if args.method == "speca":
        print(f"  SpecA:    base_thresh={args.speca_base_threshold} "
              f"decay={args.speca_decay_rate} "
              f"taylor=[{args.speca_min_taylor_steps},{args.speca_max_taylor_steps}] "
              f"metric={args.speca_error_metric}")
    print("=" * 70)

    # ===================================================================
    # 1. Load dataset
    # ===================================================================
    print("\n[1] Loading dataset...")
    if dataset_name == "imagenet":
        from dataset.imagenet import ImageNetDataset
        ds = ImageNetDataset(
            imagenet_dir=getattr(args, "imagenet_dir", IMAGENET_DIR),
            n_images=dataset_start_index + args.n_prompts, seed=args.seed)
        generation_start_index, n = _dataset_generation_window(
            dataset_start_index, int(args.n_prompts),
            covr_resume_sample_offset, len(ds))
    elif dataset_name == "coco":
        from dataset.coco import COCO30KDataset
        ds = COCO30KDataset(
            coco_dir=getattr(args, "coco_dir", None) or args.coco_dir,
            n_images=args.n_prompts, seed=args.seed)
        generation_start_index = 0
        n = len(ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ===================================================================
    # 2. Load model
    # ===================================================================
    print("\n[2] Loading DiT-2-256 model...")
    generator = DiTGenerator(num_steps=args.num_steps, device=device, dtype=dt,
                            debug=getattr(args, "debug", False))
    generator.load()

    # ---- Debug: truncate to 1 transformer block ----
    if getattr(args, "debug", False):
        n_before = len(generator.transformer.transformer_blocks)
        generator.transformer.transformer_blocks = \
            generator.transformer.transformer_blocks[:1]
        print(f"[DEBUG MODE] truncated transformer_blocks: {n_before} -> 1")

    # ---- TTT: freeze backbone before anything touches it ----
    if args.ttt:
        transformer = generator.transformer
        vae = generator.vae
        for p in transformer.parameters():
            p.requires_grad_(False)
        for p in vae.parameters():
            p.requires_grad_(False)
        transformer.eval()
        vae.eval()

    # ---- VFL: Verification Feedback Loop (Phase 2 — true async) ----
    # 推理线程只写 buffer + 调用 set_vfl_sample_id; 后台 daemon 线程独立
    # 跑训练, 不感知推理循环。每轮推理开始前可选加载上一轮的 LoRA
    # checkpoint (跨 run 飞轮) — 加载的是 train_model 副本, 推理模型权重
    # 仍为本轮 base, FID 评估保持干净。
    vfl_buf = None
    vfl_cal = None
    vfl_worker = None
    inference_lora_wrappers = None
    if getattr(args, "vfl", False):
        from feedback.vfl import (
            StratifiedReplayBuffer, OnlineCalibrator,
            AsyncTrainingWorker, VFLConfig,
            find_latest_checkpoint,
        )
        from models.dit import set_vfl_buffer, set_vfl_calibrator

        vfl_cfg = VFLConfig()
        vfl_cfg.poll_interval_s = 5.0
        vfl_cfg.loRA_rank = 4
        vfl_cfg.time_conditioned_lora = not getattr(
            args, "vfl_no_time_lora", False)
        vfl_cfg.lambda_identity = getattr(args, "vfl_lambda_identity", 1.0)

        vfl_cal = OnlineCalibrator(ema_window=100)
        set_vfl_calibrator(vfl_cal)

        vfl_output_dir = getattr(args, "vfl_output_dir", None) or \
            get_vfl_checkpoint_dir(output_dir, args.method)
        vfl_no_train = getattr(args, "vfl_no_train", False)

        # ---- Unconditionally attach LoRA to inference model (B=0 → no-op) ----
        inference_lora_wrappers = attach_lora_all_layers(
            generator.transformer,
            rank=vfl_cfg.loRA_rank,
            alpha=1.0,
            time_conditioned=vfl_cfg.time_conditioned_lora,
        )
        n_lora_params = count_lora_params(inference_lora_wrappers)
        print(f"  Inference model LoRA attached: "
              f"{n_lora_params:,} params (B zero-init = no-op)")

        # ---- Load previous checkpoint as starting point (if any) ----
        prev_ckpt = find_latest_checkpoint(vfl_output_dir)
        if prev_ckpt is None:
            prev_ckpt = find_latest_checkpoint(
                getattr(args, "vfl_output_dir", None) or "")
        if prev_ckpt:
            try:
                _load_state_into_wrappers(
                    inference_lora_wrappers, prev_ckpt,
                    device=generator.device, dtype=dt)
                print(f"  Loaded LoRA from previous run: {prev_ckpt}")
            except Exception as e:
                print(f"  [WARN] failed to load LoRA checkpoint "
                      f"{prev_ckpt}: {e} — proceeding with zero-init LoRA")

        if vfl_no_train:
            # ---- Threshold-only mode (--vfl --vfl-no-train) ----
            # Only the calibrator is registered; the buffer is left None.
            # LoRA is already attached + previous weights loaded above.
            print(f"  VFL enabled (threshold-only, no training): "
                  f"calibrator-only hot path, no LoRA worker")
        else:
            # ---- Threshold+LoRA mode (--vfl) ----
            vfl_buf = StratifiedReplayBuffer(
                capacity_per_stratum=vfl_cfg.buffer_capacity_per_stratum)
            set_vfl_buffer(vfl_buf, model_version="dit-v1")

            def on_checkpoint_ready(version):
                print(f"[VFL] LoRA v{version} ready — "
                      f"will swap at next image boundary")

            vfl_worker = AsyncTrainingWorker(
                generator.transformer, vfl_buf, config=vfl_cfg,
                base_model_version="dit-v1", output_dir=vfl_output_dir,
                on_checkpoint_ready=on_checkpoint_ready,
            )

            vfl_worker.start()
            print(f"  VFL enabled (Phase 2 async): buffer + calibrator + "
                  f"background training thread (poll={vfl_cfg.poll_interval_s}s, "
                  f"min_strata={vfl_cfg.buffer_ready_min_strata}, "
                  f"min_anchors={vfl_cfg.buffer_ready_min_anchors})")

    # ---- Print model structure (once, after all modifications) ----
    print("\n" + "=" * 70)
    if inference_lora_wrappers is not None:
        print("MODEL STRUCTURE (inference model + LoRA)")
        print("=" * 70)
        print(generator.transformer)
        lora_p = count_lora_params(inference_lora_wrappers)
        print(f"\nLoRA params (inference): {lora_p:,}")
        print(f"LoRA layers: {len(inference_lora_wrappers)}")
        if vfl_worker is not None:
            vfl_worker._ensure_train_model()
            lora_train = sum(
                p.numel() for p in vfl_worker._train_model.parameters()
                if p.requires_grad)
            print(f"LoRA trainable params (train_model): {lora_train:,}")
            print("\nTrainable parameters:")
            for name, p in vfl_worker._train_model.named_parameters():
                if p.requires_grad:
                    print(f"  {name}: {list(p.shape)}")
    else:
        print("MODEL STRUCTURE")
        print("=" * 70)
        print(generator.transformer)
        total_params = sum(p.numel() for p in generator.transformer.parameters())
        trainable = sum(p.numel() for p in generator.transformer.parameters() if p.requires_grad)
        print(f"\nTotal params: {total_params:,}  |  Trainable: {trainable:,}")
        n_blocks = len(generator.transformer.transformer_blocks)
        print(f"transformer_blocks: {n_blocks}")
        if trainable > 0:
            print("\nTrainable parameters:")
            for name, p in generator.transformer.named_parameters():
                if p.requires_grad:
                    print(f"  {name}: {list(p.shape)}")
    print("=" * 70)

    # ===================================================================
    # 3. Setup metrics
    # ===================================================================
    print("\n[3] Setting up metrics...")
    metrics = {}
    need_fid = "fid" in selected
    need_is = "is" in selected
    need_clip = "clip" in selected
    need_lpips = "lpips" in selected
    need_mse = "mse" in selected
    need_flops = "flops" in selected
    need_latency = "latency" in selected or "speed" in selected
    need_fid_is = need_fid or need_is

    if need_fid_is:
        gen_299_dir = os.path.join(output_dir, "generated_299")
        metrics["fid_is"] = FIDISComputer(gen_dir=gen_299_dir)
    if need_clip:
        from eval.clip_score import CLIPScorer
        metrics["clip"] = CLIPScorer(device=device, dtype=dt)
    if need_lpips:
        from eval.lpips import LPIPSScorer
        metrics["lpips"] = LPIPSScorer(device=device)
    if need_mse:
        from eval.mse import MSEMetric
        metrics["mse"] = MSEMetric(which="pixel")
    if need_flops:
        metrics["flops"] = FLOPsMetric(generator)
        metrics["flops"].profile()  # MUST profile before any accelerator setup
    if need_latency:
        metrics["latency"] = LatencyMetric()

    # ===================================================================
    # 4. Setup accelerator state (no monkeypatching!)
    # ===================================================================
    teacache_state = None
    ttt_state = None
    speca_cache_dic = None
    speca_current = None
    speca_init_kwargs = None
    compute_controller = None
    speca_totals = {
        "full_steps": 0,
        "taylor_steps": 0,
        "probe_full_blocks": 0,
        "corrected_probe_blocks": 0,
        "recompute_full_blocks": 0,
    }
    ddim_steps = None

    if args.method == "teacache":
        teacache_state = teacache_init(
            num_steps=args.num_steps,
            rel_l1_thresh=args.thresh,
            coefficients=_load_dit_coefficients(args.coef_path) if args.coef_path
            else _load_dit_coefficients(),
            residual_mode=getattr(args, "teacache_residual_mode", "plain"),
            residual_decay_lambda=getattr(args, "teacache_residual_decay", 0.0),
        )
        print(f"  TeaCache ready (γ={args.thresh}, "
              f"residual_mode={getattr(args, 'teacache_residual_mode', 'plain')})")

        # Per-class γ lookup table (zero-overhead offline calibration)
        per_class_gamma_table = None
        if getattr(args, "per_class_gamma", None):
            with open(args.per_class_gamma) as f:
                pct = json.load(f)
            per_class_gamma_table = {
                int(k): float(v)
                for k, v in pct.get("per_class_gamma", {}).items()
            }
            default_gamma = float(pct.get("default_gamma", args.thresh))
            n_classes = len(per_class_gamma_table)
            print(f"  Per-class γ: {n_classes} classes calibrated "
                  f"(default γ={default_gamma})")
        else:
            per_class_gamma_table = None
    if args.ttt:
        ttt_state = _setup_ttt(generator, args)
        print(f"  TTT plugin ready (lr={args.ttt_lr}, "
              f"micro_epochs={args.ttt_micro_epochs})")
    elif args.method == "ddim":
        ddim_steps = args.num_steps
        print(f"  DDIM sampling ({args.num_steps} steps, no caching)")
    elif args.method == "speca":
        num_blocks = len(generator.transformer.transformer_blocks)
        default_check_layer = (
            0 if getattr(args, "debug", False) else min(20, num_blocks - 1))
        check_layer = getattr(args, "speca_check_layer", None)
        if check_layer is None:
            check_layer = default_check_layer
        speca_init_kwargs = {
            "num_steps": args.num_steps,
            "base_threshold": args.speca_base_threshold,
            "decay_rate": args.speca_decay_rate,
            "min_taylor_steps": args.speca_min_taylor_steps,
            "max_taylor_steps": args.speca_max_taylor_steps,
            "max_order": 4,
            "num_layers": num_blocks,
            "error_metric": args.speca_error_metric,
            "check_layer": check_layer,
            "suffix_recompute_blocks": getattr(
                args, "controller_suffix_blocks", 0),
            "suffix_recompute_budget": getattr(
                args, "controller_suffix_budget", 0),
        }
        if args.compute_controller == "probe_correct":
            from accelerators.compute_controller import ProbeCorrectController
            compute_controller = ProbeCorrectController(
                correction_policy=args.controller_correction_policy)
        # Structure-aware SpecA: time-bucket max_taylor schedule (e.g. "8,4,2")
        bucket_schedule = None
        if getattr(args, "speca_bucket_schedule", None):
            parsed = [int(x) for x in args.speca_bucket_schedule.split(",")]
            if len(parsed) != 3:
                raise ValueError(
                    "--speca-bucket-schedule must have exactly 3 values "
                    "(early, mid, late), e.g. '8,4,2'")
            bucket_schedule = parsed
            print(f"  SpecA bucket schedule: "
                  f"max_taylor(E,M,L)={bucket_schedule}")
        print(f"  SpecA ready (base_thresh={args.speca_base_threshold}, "
              f"check_layer={check_layer}, "
              f"controller={args.compute_controller})")
    elif args.method == "teacache":
        pass  # Already printed above with SFAC/per-class-gamma info
    else:
        print("  Baseline (full DDIM, no acceleration)")

    covr_recorder = None
    covr_bandit = None
    covr_forced_manifest = None
    covr_forced_template = None
    covr_forced_strategy = None
    covr_forced_manifest_strategy = None
    covr_version = None
    covr_session_id = None
    covr_safety_full_steps = 0
    covr_safety_wall_s = 0.0
    covr_sentinel_full_steps = 0
    covr_terminal_wall_s = 0.0
    covr_sentinel_count = 0
    covr_terminal_losses = []
    covr_h_step_numerators = []
    covr_h_step_denominators = []
    covr_sentinel_skipped = 0
    covr_trajectory_offset = 0
    covr_runtime = None
    if covr_runtime_config is not None:
        # ---- Runtime construction (version / resume / backend / recorder) ----
        scheduler_instance = generator.scheduler
        if getattr(args, "covr_template_bandit", False):
            # Legacy template bandit: full CLI validation + manifest gate.
            manifest = TemplateManifest.load(args.covr_template_manifest)
            run_identity = {
                "dataset": dataset_name,
                "dataset_start_index": dataset_start_index,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "safety_sample_rate": float(args.covr_safety_sample_rate),
                "sentinel_rate": float(args.covr_sentinel_rate),
                "sentinel_horizon": int(args.covr_sentinel_horizon),
                "prefix_steps": int(getattr(args, "covr_prefix_steps", 0)),
                "efficiency_lambda": float(getattr(args, "covr_efficiency_lambda", 0.0)),
            }
            covr_runtime = COVRRuntime.create(
                covr_runtime_config,
                scheduler=scheduler_instance,
                num_steps=int(args.num_steps),
                cfg_scale=float(args.guidance_scale),
                speca_init_kwargs=speca_init_kwargs,
                resume=covr_resume,
            )
            if manifest.version_key != covr_runtime.version.key:
                raise ValueError(
                    f"COVR template manifest version does not match the runtime: manifest={manifest.version_key}, runtime={covr_runtime.version.key}")
            covr_bandit_state_path = covr_runtime.config.bandit_state_path
            bandit_backend = covr_runtime.configure_experimental_bandit(
                manifest,
                epsilon=args.covr_bandit_epsilon,
                run_identity=run_identity,
                state_path=covr_bandit_state_path,
                resume_expected_samples=(
                    covr_resume_sample_offset
                    if covr_resume is not None else None),
            )
            covr_bandit = bandit_backend.bandit
            covr_trajectory_offset = covr_runtime.state.trajectory_offset
            print(f"  COVR template bandit: {args.covr_template_manifest} "
                  f"({len(manifest.templates)} templates, "
                  f"{manifest.common_refresh_count} refreshes)")
        elif getattr(args, "covr_strategy_bandit", False):
            # Method-agnostic strategy bandit (experimental).
            strat_path = (args.covr_strategy_manifest
                          if args.covr_strategy_manifest
                          else args.covr_template_manifest)
            if not strat_path:
                raise ValueError(
                    "--covr-strategy-bandit requires --covr-strategy-manifest")
            covr_runtime = COVRRuntime.create(
                covr_runtime_config,
                scheduler=scheduler_instance,
                num_steps=int(args.num_steps),
                cfg_scale=float(args.guidance_scale),
                speca_init_kwargs=speca_init_kwargs,
                resume=covr_resume,
            )
            strat_manifest = load_strategy_manifest(
                strat_path, covr_runtime.version.key,
                COVRMode.EXPERIMENTAL_BANDIT)
            covr_bandit_state_path = covr_runtime.config.bandit_state_path
            bandit_backend = covr_runtime.configure_experimental_bandit(
                strat_manifest,
                epsilon=args.covr_bandit_epsilon,
                run_identity={
                    "dataset": dataset_name,
                    "dataset_start_index": dataset_start_index,
                    "seed": int(args.seed),
                    "batch_size": int(args.batch_size),
                    "safety_sample_rate": float(args.covr_safety_sample_rate),
                    "sentinel_rate": float(args.covr_sentinel_rate),
                    "sentinel_horizon": int(args.covr_sentinel_horizon),
                    "prefix_steps": int(getattr(args, "covr_prefix_steps", 0)),
                    "efficiency_lambda": float(getattr(args, "covr_efficiency_lambda", 0.0)),
                },
                state_path=covr_bandit_state_path,
                resume_expected_samples=(
                    covr_resume_sample_offset
                    if covr_resume is not None else None),
            )
            covr_bandit = bandit_backend.bandit
            covr_trajectory_offset = covr_runtime.state.trajectory_offset
            print(f"  COVR strategy bandit: {strat_path} "
                  f"({len(strat_manifest.strategies)} strategies)")
        elif getattr(args, "covr_force_strategy_id", None):
            # Forced generic strategy: symmetric version gate in both modes.
            strat_path = (args.covr_strategy_manifest
                          if args.covr_strategy_manifest
                          else args.covr_template_manifest)
            if not strat_path:
                raise ValueError(
                    "--covr-force-strategy-id requires --covr-strategy-manifest")
            covr_runtime = COVRRuntime.create(
                covr_runtime_config,
                scheduler=scheduler_instance,
                num_steps=int(args.num_steps),
                cfg_scale=float(args.guidance_scale),
                speca_init_kwargs=speca_init_kwargs,
                resume=covr_resume,
            )
            strat_manifest = load_strategy_manifest(
                strat_path, covr_runtime.version.key, COVRMode.FORCED)
            strategy = strat_manifest.strategy_map.get(
                args.covr_force_strategy_id)
            if strategy is None:
                raise ValueError(
                    f"forced strategy {args.covr_force_strategy_id} "
                    f"not found in the manifest")
            covr_forced_manifest_strategy = strat_manifest
            covr_forced_strategy = strategy
            covr_runtime.configure_forced_backend(
                strategy=strategy, manifest=strat_manifest)
            print(
                f"  COVR forced strategy: {strategy.strategy_id} "
                f"(method={strategy.method})")
        else:
            # OBSERVE modes: shadow recorder and/or stage profiling.
            covr_runtime = COVRRuntime.create(
                covr_runtime_config,
                scheduler=scheduler_instance,
                num_steps=int(args.num_steps),
                cfg_scale=float(args.guidance_scale),
                speca_init_kwargs=speca_init_kwargs,
                resume=covr_resume,
            )

        # Expose runtime-owned pieces through the local compatibility names
        # used by the rest of run_c2i (loop wiring, aggregate, results config).
        covr_version = covr_runtime.version
        covr_session_id = covr_runtime.session_id
        covr_recorder = covr_runtime.recorder
        if covr_recorder is not None:
            print(f"  COVR shadow recorder: {covr_recorder.event_path}")

        if getattr(args, "covr_force_template_id", None):
            covr_forced_manifest, covr_forced_template = (
                _load_forced_covr_template(
                    args.covr_template_manifest,
                    args.covr_force_template_id,
                    covr_version.key,
                ))
            print(
                f"  COVR forced template: {covr_forced_template.template_id} "
                f"({covr_forced_template.refresh_count} refreshes)")
            covr_forced_strategy = covr_forced_template.to_strategy(
                args.num_steps)
            covr_forced_manifest_strategy = covr_forced_manifest
            covr_runtime.configure_forced_backend(
                strategy=covr_forced_strategy,
                manifest=covr_forced_manifest)

    # ===================================================================
    # 5. Generate images
    # ===================================================================
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    total_images = n
    viability_recorder = None
    viability_output = getattr(args, "covr_viability_output", None)
    _use_boundary_telemetry = bool(getattr(
        args, "covr_viability_boundary_telemetry", False))
    prefix_steps = int(getattr(args, "covr_viability_prefix_steps", 3))
    if viability_output:
        if covr_forced_strategy is None or covr_forced_manifest_strategy is None:
            raise ValueError(
                "COVR viability probe requires a forced strategy manifest")
        run_identity = {
            "model": "dit",
            "method": args.method,
            "num_steps": int(args.num_steps),
            "dataset_start_index": int(dataset_start_index),
            "latent_seed_offset": int(getattr(
                args, "latent_seed_offset", 0)),
            "session_id": str(getattr(args, "covr_session_id", "") or ""),
            "manifest_hash": covr_forced_manifest_strategy.manifest_hash,
            "boundary_telemetry": bool(_use_boundary_telemetry),
            "image_format": str(getattr(
                args, "covr_viability_image_format", None) or "png"),
        }
        viability_recorder = COVRViabilityRecorder(
            viability_output,
            prefix_steps=prefix_steps,
            run_identity=run_identity,
        )
    bs = args.batch_size
    print(f"\n[4] Generating {total_images} images ({args.method}, "
          f"{n} prompts in batches of ≤{bs})...")
    t_start = time.time()

    wall_times = []
    covr_safety_wall_times = []
    covr_terminal_wall_times = []
    covr_control_wall_times = []
    profile_stage_totals = defaultdict(float)
    profile_stage_counts = defaultdict(int)
    all_results = []
    global_idx = generation_start_index

    for batch_start in tqdm(range(0, n, bs), desc=f"c2i/{dataset_name}", ncols=80):
        # ---- VFL mid-run reload: pull latest LoRA from training daemon ----
        if vfl_worker is not None and inference_lora_wrappers is not None:
            new_state = vfl_worker.pull_latest_lora_state()
            if new_state is not None:
                try:
                    _swap_lora_weights(
                        inference_lora_wrappers, new_state,
                        device=generator.device, dtype=dt,
                    )
                    print(f"[VFL] swapped LoRA at image {batch_start} "
                          f"(v{vfl_worker._candidate_version})")
                except Exception as e:
                    print(f"[VFL] swap failed at image {batch_start}: {e} "
                          f"— keeping previous LoRA weights")

        batch_end = min(batch_start + bs, n)
        batch_indices = list(range(
            generation_start_index + batch_start,
            generation_start_index + batch_end))
        batch_absolute_indices = batch_indices
        actual_bs = len(batch_indices)
        covr_profiler = (
            _GenerationProfiler(
                generator.device,
                detailed=bool(getattr(args, "covr_profile_stages", False)),
            )
            if covr_runtime is not None else None)

        # Collect prompts (class labels) + seeds
        batch_inputs, batch_seeds = [], []
        for idx, absolute_idx in zip(batch_indices, batch_absolute_indices):
            data = ds[idx]
            gen_input = data[2] if len(data) > 2 else data[1]
            # DiT needs integer class labels
            if dataset_name == "imagenet":
                gen_input = gen_input  # already int from ImageNetDataset
            else:
                gen_input = data[1]  # text prompt for PixArt; DiT can't handle text
            batch_inputs.append(gen_input)
            # Per-image latent seed: the (image, latent draw) cell. --seed
            # decides which image absolute_idx names; --latent-seed-offset
            # selects an INDEPENDENT latent draw for the same image, so
            # replicate runs can be paired per-image across draws. Distinct
            # offsets give disjoint seed sets (stride 1e6 > any index < 50k),
            # and offset=0 stays bit-identical to the legacy
            # `100000 + absolute_idx` formula.
            batch_seeds.append(latent_seed_for_index(
                absolute_idx, int(getattr(args, "latent_seed_offset", 0))))

        # Reset accelerator state
        trajectory_id = covr_trajectory_offset + batch_start // bs
        covr_assignment = None
        covr_trajectory = None
        covr_sentinel_selected = False
        covr_sentinel_start_idx = None
        strategy_select_start = time.perf_counter()
        if covr_runtime is not None:
            covr_trajectory = covr_runtime.begin_trajectory(
                trajectory_id,
                sample_count=actual_bs,
                sample_ids=[
                    (f"{covr_session_id}:{trajectory_id}:{index}"
                     if covr_bandit is not None else str(index))
                    for index in batch_absolute_indices
                ],
                num_steps=args.num_steps,
                sentinel_rate=args.covr_sentinel_rate,
                sentinel_horizon=args.covr_sentinel_horizon,
                mandatory_prefix=int(getattr(
                    covr_forced_manifest_strategy, "mandatory_prefix", 0))
                if covr_forced_manifest_strategy is not None else 0,
            )
            covr_trajectory.profiler = covr_profiler
            covr_assignment = covr_trajectory.bandit_assignment
            covr_sentinel_selected = covr_trajectory.sentinel_selected
            covr_sentinel_start_idx = covr_trajectory.sentinel_start_idx
            # Deferred-commit contextual bandit: tell the denoise loop how many
            # forced-calc prefix steps to run before committing the arm.
            if (covr_runtime_config is not None
                    and covr_runtime_config.contextual):
                covr_trajectory.prefix_steps = int(
                    covr_runtime_config.prefix_steps)
        if covr_profiler is not None:
            covr_profiler.add_cpu(
                "strategy_selection", time.perf_counter() - strategy_select_start)

        reset_start = time.perf_counter()
        if args.method == "teacache" and teacache_state is not None:
            # ── γ selection (offline per-class table > default) ──
            if per_class_gamma_table is not None and batch_inputs:
                # Offline per-class table (zero-overhead fallback)
                batch_gammas = []
                for bi in batch_inputs:
                    if isinstance(bi, (int, float)) and int(bi) in per_class_gamma_table:
                        batch_gammas.append(per_class_gamma_table[int(bi)])
                if batch_gammas:
                    teacache_state["rel_l1_thresh"] = min(batch_gammas)
            teacache_reset(teacache_state)
        if args.ttt and ttt_state is not None:
            ttt_reset_for_image(ttt_state)
        reset_stage = (
            "speca_state_reset"
            if args.method == "speca" else "accelerator_state_reset")
        if covr_profiler is not None:
            covr_profiler.add_cpu(
                reset_stage, time.perf_counter() - reset_start)

        strategy_init_start = time.perf_counter()
        # Strategy dispatch: configure the accelerator from the
        # bandit/forced-template selection when one is active. Method-agnostic
        # via the adapter registry — apply_strategy() looks up the strategy's
        # AcceleratorAdapter and returns its state_keys, so adding a new
        # accelerator needs no branch here.
        _using_strategy = (
            covr_trajectory is not None
            and covr_trajectory.strategy is not None)
        if _using_strategy:
            dispatch_result = covr_runtime.apply_acceleration_strategy(
                covr_trajectory,
                speca_init_kwargs=(
                    {
                        **speca_init_kwargs,
                        "controller": compute_controller,
                        "trajectory_id": trajectory_id,
                    }
                    if speca_init_kwargs is not None else None
                ),
                teacache_init_kwargs={
                    "num_steps": args.num_steps,
                    "coefficients": _load_dit_coefficients(args.coef_path)
                    if args.coef_path else _load_dit_coefficients(),
                    **({"probe_prefix_steps": prefix_steps}
                       if _use_boundary_telemetry else {}),
                },
            )
            # Unpack whatever the selected adapter produced; untouched state
            # objects keep their prior (None) values.
            speca_cache_dic = dispatch_result.get("cache_dic", speca_cache_dic)
            speca_current = dispatch_result.get("current", speca_current)
            teacache_state = dispatch_result.get(
                "teacache_state", teacache_state)
            # SpecA controller lifecycle (controller is None for other methods).
            if "cache_dic" in dispatch_result and compute_controller is not None:
                compute_controller.begin_trajectory(trajectory_id)
        elif _using_strategy:
            strategy = (
                covr_forced_strategy
                if covr_forced_strategy is not None
                else covr_bandit.active_strategy
            )
            dispatch_result = apply_strategy(
                strategy,
                speca_init_kwargs=(
                    {
                        **speca_init_kwargs,
                        "controller": compute_controller,
                        "trajectory_id": trajectory_id,
                    }
                    if speca_init_kwargs is not None else None
                ),
                teacache_init_kwargs={
                    "num_steps": args.num_steps,
                    "coefficients": _load_dit_coefficients(args.coef_path)
                    if args.coef_path else _load_dit_coefficients(),
                    **({"probe_prefix_steps": prefix_steps}
                       if _use_boundary_telemetry else {}),
                },
            )
            # Unpack whatever the selected adapter produced; untouched state
            # objects keep their prior (None) values.
            speca_cache_dic = dispatch_result.get("cache_dic", speca_cache_dic)
            speca_current = dispatch_result.get("current", speca_current)
            teacache_state = dispatch_result.get(
                "teacache_state", teacache_state)
            # SpecA controller lifecycle (controller is None for other methods).
            if "cache_dic" in dispatch_result and compute_controller is not None:
                compute_controller.begin_trajectory(trajectory_id)
            if bucket_schedule is not None and speca_cache_dic is not None:
                speca_cache_dic.bucket_schedule = bucket_schedule
        elif speca_init_kwargs is not None:
            # No bandit, direct SpecA config (no refresh_mask = auto-decay)
            speca_cache_dic, speca_current = speca_init(
                **speca_init_kwargs,
                controller=compute_controller,
                trajectory_id=trajectory_id,
            )
            if compute_controller is not None:
                compute_controller.begin_trajectory(trajectory_id)
            if bucket_schedule is not None and speca_cache_dic is not None:
                speca_cache_dic.bucket_schedule = bucket_schedule
        if covr_profiler is not None:
            covr_profiler.add_cpu(
                "strategy_initialization",
                time.perf_counter() - strategy_init_start)
        if (speca_current is not None and
                getattr(args, "covr_profile_stages", False)):
            speca_current.profiler = covr_profiler

        # VFL (Phase 2): tag this batch's denoising trajectory with a unique
        # sample_id so curvature loss can group events from the same image.
        # We use batch_start as the id — every event recorded during the
        # upcoming generate() call (across all denoising steps and all
        # blocks) will share this id, which is exactly what the per-(sample,
        # layer) trajectory fitter needs. Set just before generate() so the
        # _denoise_loop's set_vfl_step_info calls layer on top of it.
        if vfl_buf is not None:
            set_vfl_sample_id(batch_start)

        if viability_recorder is not None:
            viability_recorder.begin_trajectory(
                global_idx=batch_absolute_indices[0],
                latent_seed=batch_seeds[0],
                strategy_id=covr_forced_strategy.strategy_id,
                manifest_hash=covr_forced_manifest_strategy.manifest_hash,
                refresh_mask=covr_forced_strategy.refresh_mask,
                batch_size=actual_bs,
                sample_id=str(batch_absolute_indices[0]),
            )

        # Generate
        covr_feedback_sink = (
            covr_trajectory.feedback_sink
            if covr_trajectory is not None else None)
        if covr_profiler is not None:
            generation_token = covr_profiler.start_gpu(
                "generation_online", required=True)
            generation_start = None
        else:
            if (torch.cuda.is_available()
                    and str(generator.device).startswith("cuda")):
                torch.cuda.synchronize(generator.device)
            generation_token = None
            generation_start = time.perf_counter()
        if args.ttt:
            latent, img = generator.generate_ttt(
                batch_inputs, batch_seeds,
                guidance_scale=args.guidance_scale,
                teacache_state=teacache_state,
                ttt_state=ttt_state,
            )
        else:
            latent, img = generator.generate(
                batch_inputs, batch_seeds,
                guidance_scale=args.guidance_scale,
                method=args.method,
                teacache_state=teacache_state,
                cache_dic=speca_cache_dic,
                current=speca_current,
                ddim_steps=ddim_steps,
                covr_trajectory=covr_trajectory,
                viability_recorder=viability_recorder,
                covr_runtime=(covr_runtime
                              if covr_runtime_config is not None
                              and covr_runtime_config.contextual else None),
            )
        if viability_recorder is not None:
            viability_recorder.end_trajectory()
        # Deferred-commit contextual bandit: commit_arm refreshed the bandit
        # assignment mid-trajectory; re-bind the outer reference so the results
        # dict and FLOPs attribution see the committed arm, not the pending one.
        if (covr_trajectory is not None
                and covr_runtime_config is not None
                and covr_runtime_config.contextual):
            covr_assignment = covr_trajectory.bandit_assignment
            # Efficiency-aware reward: write the measured per-trajectory FLOPs
            # ratio (full-trajectory decisions, prefix+suffix) into the feedback
            # sink. _bandit_feedback combines it with terminal fidelity as
            # combined = terminal_MSE + lambda * cost (lambda from config).
            if covr_feedback_sink is not None and teacache_state is not None:
                decisions = teacache_state.get("decisions", [])
                n_calc = sum(1 for d in decisions if d == "calc")
                n_skip = sum(1 for d in decisions if d == "skip")
                total_steps = n_calc + n_skip
                if total_steps > 0:
                    flops_metric = metrics.get("flops")
                    flops_full = getattr(flops_metric, "_flops_full", 0.0)
                    flops_skip = getattr(flops_metric, "_flops_skip", 0.0)
                    if flops_full > 0.0:
                        measured = (n_calc * flops_full
                                    + n_skip * flops_skip)
                        vanilla = total_steps * flops_full
                        cost_ratio = measured / vanilla if vanilla > 0 else 1.0
                    else:
                        # FLOPs not profiled for this run: approximate a skip as
                        # free so the cost term still favors aggressive arms.
                        cost_ratio = n_calc / total_steps
                    covr_feedback_sink["measured_cost_ratio"] = float(cost_ratio)

        # Pending safety observations are converted at the batch boundary
        # (tensors are still on the accelerator); the runtime materializes
        # them into the trajectory feedback and observes the bandit at close.
        pending_safety = (
            covr_feedback_sink.pop("pending_safety", ())
            if covr_feedback_sink is not None else ())
        if pending_safety:
            covr_feedback_sink["pending_safety_steps"] = tuple(
                step_idx for step_idx, _ in pending_safety)
            covr_feedback_sink["pending_safety_values"] = torch.stack(
                [values for _, values in pending_safety])
        if covr_profiler is not None:
            covr_profiler.stop_gpu(generation_token)
            covr_profiler.synchronize()
            generation_profile = covr_profiler.summary()

            # The runtime measures its own feedback materialization; the stage
            # profiler records it so wall accounting stays stage-consistent.
            feedback_start = time.perf_counter()
            feedback_materialize_s = time.perf_counter() - feedback_start
            covr_profiler.add_cpu(
                "feedback_materialization", feedback_materialize_s)
            generation_profile["feedback_materialization"] = (
                feedback_materialize_s)
        else:
            if (torch.cuda.is_available()
                    and str(generator.device).startswith("cuda")):
                torch.cuda.synchronize(generator.device)
            generation_profile = {
                "generation_online": time.perf_counter() - generation_start}

        fallback_wall_s = 0.0
        fallback_full_steps = 0
        fallback_profile = None
        if covr_assignment is not None:
            assert covr_trajectory is not None
            if (covr_sentinel_selected and covr_sentinel_start_idx is None
                    and covr_feedback_sink.get("terminal_fidelity_loss_tensor")
                    is None):
                # Terminal fidelity: prefer the cheap one-step computation
                # from _denoise_loop (1 extra forward pass) — available for
                # any refresh-mask accelerator (SpecA, forced-schedule
                # TeaCache). Fall back to the expensive full-baseline
                # comparison only when no cheap reward was produced (e.g. a
                # threshold-based TeaCache arm with no forced mask).
                fallback_profiler = _GenerationProfiler(
                    generator.device,
                    detailed=bool(getattr(
                        args, "covr_profile_stages", False)),
                )
                fallback_token = fallback_profiler.start_gpu(
                    "terminal_fallback_full", required=True)
                full_latent, _ = _covr_generate_terminal_fallback(
                    generator,
                    batch_inputs,
                    batch_seeds,
                    args.guidance_scale,
                    covr_trajectory,
                    fallback_profiler,
                )
                tf_loss_tensor = F.mse_loss(
                    latent.float(), full_latent.float())
                fallback_profiler.stop_gpu(fallback_token)
                fallback_profiler.synchronize()
                fallback_profile = fallback_profiler.summary()
                # The runtime materializes the fallback scalar through the
                # same path as the cheap one-step reward.
                covr_feedback_sink["terminal_fidelity_loss_tensor"] = (
                    tf_loss_tensor.detach().to("cpu"))
                fallback_wall_s = fallback_profile["terminal_fallback_full"]
                fallback_full_steps = args.num_steps

        # Runtime owns the trajectory end lifecycle: feedback materialization,
        # control timing buckets, sentinel/full-step counters, bandit
        # end_trajectory + per-trajectory state persist, forced telemetry.
        covr_outcome = (
            covr_runtime.end_trajectory(
                covr_trajectory,
                generation_profile=generation_profile,
                num_steps=args.num_steps,
                sentinel_rate=args.covr_sentinel_rate,
                sentinel_horizon=args.covr_sentinel_horizon,
                fallback_wall_s=fallback_wall_s,
                fallback_full_steps=fallback_full_steps,
                fallback_profile=fallback_profile,
            )
            if covr_runtime is not None and covr_trajectory is not None
            else None)
        if covr_profiler is not None:
            covr_profiler.reset()
        control_wall_s = (
            covr_outcome["control_wall_s"]
            if covr_outcome is not None else 0.0)
        wall_s = (
            covr_outcome["wall_s"]
            if covr_outcome is not None
            else generation_profile["generation_online"])
        if covr_feedback_sink is not None:
            covr_feedback_sink.pop("pending_safety_steps", None)
            covr_feedback_sink.pop("pending_safety_values", None)
            covr_feedback_sink.pop("h_step_components_tensor", None)
            covr_feedback_sink.pop("terminal_fidelity_loss_tensor", None)
            covr_feedback_sink.pop("safety_full_steps", None)
            covr_feedback_sink.pop("terminal_full_steps", None)

        # Keep the local aggregate counters synchronized from the runtime
        # state (the aggregate block below still reads these names).
        if covr_runtime is not None:
            _st = covr_runtime.state
            covr_safety_wall_times = list(_st.safety_wall_times)
            covr_terminal_wall_times = list(_st.terminal_wall_times)
            covr_control_wall_times = list(_st.control_wall_times)
            covr_safety_wall_s = float(_st.safety_wall_s)
            covr_safety_full_steps = int(_st.safety_full_steps)
            covr_terminal_wall_s = float(_st.terminal_wall_s)
            covr_sentinel_full_steps = int(_st.sentinel_full_steps)
            covr_sentinel_count = int(_st.sentinel_count)
            covr_terminal_losses = list(_st.terminal_losses)
            covr_h_step_numerators = list(_st.h_step_numerators)
            covr_h_step_denominators = list(_st.h_step_denominators)
            covr_sentinel_skipped = int(_st.sentinel_skipped)
            profile_stage_totals = dict(_st.profile_stage_totals)
            profile_stage_counts = dict(_st.profile_stage_counts)

        if speca_cache_dic is not None:
            speca_totals["full_steps"] += speca_cache_dic.full_count
            speca_totals["taylor_steps"] += speca_cache_dic.taylor_count
            speca_totals["probe_full_blocks"] += speca_cache_dic.probe_full_blocks
            speca_totals["corrected_probe_blocks"] += (
                speca_cache_dic.corrected_probe_blocks)
            speca_totals["recompute_full_blocks"] += (
                speca_cache_dic.recompute_full_blocks)
            if compute_controller is not None:
                compute_controller.end_trajectory()

        if covr_forced_manifest is not None and speca_cache_dic is not None:
            expected = covr_forced_manifest.common_refresh_count
            actual_full = speca_cache_dic.full_count
            actual_taylor = speca_cache_dic.taylor_count
            if actual_full != expected:
                print(
                    f"  [warn] forced template full-count mismatch: "
                    f"actual={actual_full} expected={expected} "
                    f"taylor={actual_taylor}",
                    flush=True,
                )

        if covr_forced_strategy is not None:
            # Forced single-arm sweep: same reward telemetry as the bandit
            # path, aggregated into results.json. Never fall back to a full
            # 50-step baseline comparison (that is the bandit-mode fallback
            # and would cost ~50x per missing reward) — a sentinel without a
            # cheap reward is skipped and counted instead. The runtime
            # materialized the reward telemetry; refresh the local aliases
            # from the runtime state (aggregate below still reads these).
            if covr_runtime is not None:
                _st = covr_runtime.state
                covr_terminal_losses = list(_st.terminal_losses)
                covr_h_step_numerators = list(_st.h_step_numerators)
                covr_h_step_denominators = list(_st.h_step_denominators)
                covr_sentinel_skipped = int(_st.sentinel_skipped)

        wall_times.append(wall_s)
        per_img_s = wall_s / actual_bs

        # Save and metric ingestion are excluded from generation img/s but
        # reported separately by the stage profiler.
        postprocess_start = time.perf_counter()
        img_limit = getattr(args, "img_save_limit", 50)
        for b, idx in enumerate(batch_indices):
            if global_idx - generation_start_index < img_limit:
                # Extract class name from dataset prompt
                cls_name = ds[idx][1].replace("a photo of a ", "").replace(" ", "_")
                image_format = getattr(
                    args, "covr_viability_image_format", None)
                extension = ".jpg" if image_format == "jpeg" else ".png"
                out_path = os.path.join(
                    gen_dir, f"{global_idx:06d}_{cls_name}{extension}")
                save_image(img[b:b+1], out_path)
            # Feed to FID/IS directly (resize from memory, no extra disk round-trip)
            if need_fid_is:
                tag = ds[idx][1].replace("a photo of a ", "").replace(" ", "_")
                metrics["fid_is"].add(img[b], tag=tag)
            global_idx += 1

        # Batch eval
        if need_clip:
            batch_prompts_text = [ds[idx][1] for idx in batch_indices]
            metrics["clip"].add_batch(img, prompts=batch_prompts_text)
        if need_lpips or need_mse:
            real_tensors = []
            for idx in batch_indices:
                img_path = ds[idx][0]
                try:
                    real_pil = Image.open(img_path).convert("RGB")
                    gen_size = (img.shape[-1], img.shape[-2])
                    real_pil = real_pil.resize(gen_size, Image.BICUBIC)
                    real_tensors.append(pil_to_tensor(real_pil).to(device))
                except Exception:
                    real_tensors.append(None)
            valid_indices = [i for i, rt in enumerate(real_tensors) if rt is not None]
            if valid_indices:
                valid_imgs = img[valid_indices]
                valid_refs = torch.stack([real_tensors[i] for i in valid_indices])
                if need_lpips:
                    metrics["lpips"].add_batch(valid_imgs, valid_refs)
                if need_mse:
                    metrics["mse"].add_batch(valid_imgs, valid_refs)
        if need_latency:
            metrics["latency"].add_pairs_batch(
                [per_img_s] * actual_bs, [per_img_s] * actual_bs)
        if need_flops:
            if args.ttt and teacache_state is not None:
                # TTT FLOPs: base TeaCache + plugin training overhead.
                # Plugin fwd~6M + bwd~12M + opt~1M ≈ 19M per micro-epoch.
                n_calc = sum(1 for d in teacache_state["decisions"] if d == "calc")
                n_skip = sum(1 for d in teacache_state["decisions"] if d == "skip")
                total = n_calc + n_skip
                pfe = 19e6  # plugin forward+backward+opt FLOPs per micro-epoch
                flops_full = metrics["flops"]._flops_full
                flops_skip = metrics["flops"]._flops_skip
                metrics["flops"]._total_vanilla += total * flops_full
                metrics["flops"]._total_accel += (
                    n_calc * (flops_full + args.ttt_micro_epochs * pfe)
                    + n_skip * flops_skip
                )
                metrics["flops"]._n += 1
            elif covr_runtime is not None and covr_trajectory is not None:
                covr_runtime.add_flops(
                    metrics["flops"], covr_trajectory,
                    num_layers=len(
                        generator.transformer.transformer_blocks),
                    method=args.method,
                    accelerator_states={
                        "cache_dic": speca_cache_dic,
                        "current": speca_current,
                        "teacache_state": teacache_state,
                    },
                    num_steps=args.num_steps,
                )
            elif is_registered(args.method):
                # Method-agnostic FLOPs accounting via the adapter registry.
                # Adapters no-op when their state object is absent, falling
                # through to the vanilla-step count below only if nothing was
                # accumulated.
                adapter = get_adapter(args.method)
                states = {
                    "cache_dic": speca_cache_dic,
                    "current": speca_current,
                    "teacache_state": teacache_state,
                }
                if any(states.get(k) is not None for k in adapter.state_keys):
                    adapter.add_flops(
                        metrics["flops"], states,
                        num_layers=len(
                            generator.transformer.transformer_blocks))
                else:
                    metrics["flops"].add_vanilla_steps(args.num_steps)
            else:
                metrics["flops"].add_vanilla_steps(args.num_steps)

        for idx, absolute_idx in zip(batch_indices, batch_absolute_indices):
            data = ds[idx]
            all_results.append({
                "idx": absolute_idx,
                "prompt": str(data[1])[:120],
                "wall_s": wall_s,
                "images": 1,
                **({
                    "covr_template_id": covr_assignment.template_id,
                    "covr_template_propensity": covr_assignment.propensity,
                    "covr_sentinel": covr_sentinel_selected,
                    "covr_sentinel_start_idx": covr_sentinel_start_idx,
                } if covr_assignment is not None else {}),
            })
        _record_profile_stage(
            profile_stage_totals,
            profile_stage_counts,
            "image_save_metrics",
            time.perf_counter() - postprocess_start,
        )

        # Phase 2: 推理循环内不再调用任何训练方法。后台线程独立轮询
        # buffer, 在数据足够时自行触发训练。这里只做 event / anchor
        # 收集 (在 _denoise_loop 内部完成), latency 完全不受训练影响。
        # -- COVR: done. --

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min "
          f"({elapsed/total_images:.2f} s/image, {len(wall_times)} batches)")

    # ---- VFL: 推理结束后处理 ----
    # 1. 优雅停止后台线程 (等待当前训练周期完成或 timeout);
    # 2. 报告本 run 的训练成果 (checkpoint / 周期数 / crash 计数)。
    # 注意: 不会强行 drain buffer — 残余样本留到下一轮加载 checkpoint
    # 后继续训练 (跨 run 飞轮)。如果想立即处理, 在 stop 前手动调用
    # worker._train_once() 即可 (但通常无必要)。
    if vfl_worker is not None:
        vfl_worker.stop(timeout=60.0)
        print(f"  [VFL] worker stopped. total updates: "
              f"{vfl_worker.total_updates}, "
              f"train_steps: {vfl_worker.total_train_steps}, "
              f"crashes: {vfl_worker.crash_count}")
        if vfl_worker.last_error:
            print(f"  [VFL] last error: {vfl_worker.last_error}")
        ckpt = vfl_worker.get_latest_checkpoint()
        if ckpt:
            print(f"  [VFL] latest checkpoint: {ckpt}")
        else:
            print("  [VFL] no checkpoint produced this run "
                  "(buffer may not have crossed readiness threshold)")
        # Retain only the 3 most recent checkpoints to bound disk usage.
        pruned = prune_checkpoints(vfl_output_dir, keep=3)
        if pruned:
            print(f"  [VFL] pruned {pruned} old checkpoint(s) "
                  f"in {vfl_output_dir}")

    # ===================================================================
    # 6. FID/IS
    # ===================================================================
    fid_is_results = {}
    if need_fid_is:
        real_299_dir = ensure_real_299(
            ds, output_dir, n, start_index=generation_start_index)
        metrics["fid_is"].real_dir = real_299_dir
        try:
            fid_is_results = _compute_generated_fid_is(metrics["fid_is"])
        finally:
            metrics["fid_is"].cleanup()

    # ===================================================================
    # 7. Aggregate
    # ===================================================================
    agg: Dict = {"n_images": total_images}

    if wall_times:
        agg["wall_s_mean"] = float(np.mean(wall_times))
        agg["wall_s_std"] = float(np.std(wall_times))
        agg["speed_img_per_s"] = float(n / np.sum(wall_times)) if wall_times else 0.0

    if need_clip:
        agg.update(metrics["clip"].compute())
    if need_lpips:
        agg.update(metrics["lpips"].compute())
    if need_mse:
        agg.update(metrics["mse"].compute())
    if need_latency:
        agg.update(metrics["latency"].compute())
    if need_flops:
        agg.update(metrics["flops"].compute())

    if covr_bandit is not None and covr_runtime is not None:
        agg.update(covr_runtime.online_accounting(
            wall_times=wall_times,
            n_images=n,
            candidate_flops_T=(
                agg.get("flops_accel_T") if need_flops else None),
            vanilla_flops_T=(
                agg.get("flops_vanilla_T") if need_flops else None),
            full_step_flops=(
                metrics["flops"]._flops_full if need_flops else None),
        ))

    if getattr(args, "covr_profile_stages", False):
        agg["generation_profile"] = (
            covr_runtime.generation_profile(batches=len(wall_times))
            if covr_runtime is not None
            else {
                "stage_total_s": {
                    stage: float(seconds)
                    for stage, seconds in profile_stage_totals.items()
                    if stage != "cuda_sync_calls"
                },
                "stage_mean_per_batch_s": {
                    stage: float(seconds / max(1, len(wall_times)))
                    for stage, seconds in profile_stage_totals.items()
                    if stage != "cuda_sync_calls"
                },
                "stage_observed_batches": {
                    stage: int(profile_stage_counts[stage])
                    for stage in profile_stage_totals
                    if stage != "cuda_sync_calls"
                },
                "cuda_sync_calls": int(profile_stage_totals.get(
                    "cuda_sync_calls", 0.0)),
                "batches": len(wall_times),
            })

    if need_fid_is:
        agg.update(fid_is_results)

    # TeaCache / TTT stats
    if args.method == "teacache" and teacache_state is not None:
        st = teacache_stats(teacache_state)
        agg["skip_ratio"] = st.get("skip_ratio", 0.0)
        agg["total_calc"] = st.get("total_calc", 0)
        agg["total_skip"] = st.get("total_skip", 0)
    if args.ttt and ttt_state is not None:
        ts = ttt_session_stats(ttt_state)
        agg["ttt_trained_steps"] = ts["trained_steps"]
        agg["ttt_loss_mean"] = ts["session_loss_mean"]
        agg["ttt_plugin_params"] = ts["plugin_params"]
    elif args.method == "speca":
        full_cnt = speca_totals["full_steps"]
        taylor_cnt = speca_totals["taylor_steps"]
        total = full_cnt + taylor_cnt
        agg["skip_ratio"] = taylor_cnt / total if total > 0 else 0.0
        agg["taylor_steps"] = taylor_cnt
        agg["full_steps"] = full_cnt
        agg["total_calc"] = full_cnt
        agg["total_skip"] = taylor_cnt
        agg["speca_check_layer"] = speca_init_kwargs["check_layer"]
        agg["speca_suffix_recompute_blocks"] = speca_init_kwargs[
            "suffix_recompute_blocks"]
        agg["speca_suffix_recompute_budget"] = speca_init_kwargs[
            "suffix_recompute_budget"]
        agg["speca_probe_full_blocks"] = speca_totals["probe_full_blocks"]
        agg["speca_corrected_probe_blocks"] = (
            speca_totals["corrected_probe_blocks"])
        agg["speca_recompute_full_blocks"] = (
            speca_totals["recompute_full_blocks"])
        num_layers = len(generator.transformer.transformer_blocks)
        agg["speca_full_block_equivalents"] = (
            full_cnt * num_layers
            + speca_totals["probe_full_blocks"]
            + speca_totals["recompute_full_blocks"]
        )
        if compute_controller is not None:
            agg["compute_controller"] = compute_controller.stats()

    # ---- VFL stats ----
    if vfl_buf is not None:
        vfl_stats = vfl_buf.stats()
        agg["vfl_events"] = vfl_stats["total_samples"]
        agg["vfl_strata"] = vfl_stats["num_strata_nonempty"]
    if vfl_cal is not None:
        agg["vfl_calibrator_updates"] = vfl_cal.total_updates
    if vfl_worker is not None:
        ws = vfl_worker.get_status()
        agg["vfl_train_steps"] = ws["train_step"]
        agg["vfl_total_updates"] = ws["total_updates"]
        agg["vfl_crash_count"] = ws["crash_count"]
        # Phase 2 全层挂 LoRA, attached_layers 不再有意义, 但保留
        # vfl_layers 字段向后兼容 (输出全 28 层)
        agg["vfl_layers"] = (sorted(vfl_worker._layer_wrappers.keys())
                            if vfl_worker._layer_wrappers else [])
        if ws.get("latest_checkpoint"):
            agg["vfl_latest_checkpoint"] = ws["latest_checkpoint"]

    # Runtime owns the run-level COVR aggregate payloads (shadow summary,
    # template-bandit summary, forced-template info, sentinel reward
    # telemetry). Local aliases mirror the runtime state so the prints below
    # keep their old shapes.
    if covr_runtime is not None:
        covr_agg = covr_runtime.aggregate(
            forced_template=covr_forced_template,
            forced_strategy=covr_forced_strategy,
            forced_manifest=(
                covr_forced_manifest
                if covr_forced_manifest is not None
                else covr_forced_manifest_strategy),
            speca_probe_full_blocks=speca_totals["probe_full_blocks"],
            state_path=covr_bandit_state_path,
            dataset_start_index=dataset_start_index,
            resume_sample_offset=covr_resume_sample_offset,
            generation_start_index=generation_start_index,
            target_samples=int(args.n_prompts),
            generated_samples_this_run=n,
        )
        agg.update(covr_agg)
        _st = covr_runtime.state
        covr_safety_full_steps = int(_st.safety_full_steps)
        covr_safety_wall_s = float(_st.safety_wall_s)
        covr_sentinel_count = int(_st.sentinel_count)
        covr_sentinel_full_steps = int(_st.sentinel_full_steps)
        covr_terminal_wall_s = float(_st.terminal_wall_s)

    covr_summary = None
    if covr_recorder is not None and covr_runtime is not None:
        covr_summary = covr_runtime.close()
        if covr_summary is not None:
            agg["covr_shadow"] = covr_summary
        print(
            f"  COVR action audits: {covr_summary['events']} batch-step contexts, "
            f"{covr_summary['samples']} sample labels")

    if viability_recorder is not None:
        viability_recorder.close()
        print(
            f"  COVR viability probe: {viability_recorder.record_count} "
            f"prefix records across {viability_recorder.trajectory_count} trajectories "
            f"-> {viability_recorder.output_path}")

    if covr_bandit is not None and covr_runtime is not None:
        bandit_summary = covr_agg["covr_template_bandit"]
        print(
            f"  COVR template bandit: {bandit_summary['completed_trajectories']} "
            f"trajectories, sentinel_full_steps={covr_sentinel_full_steps}")

    if "speed" in selected and all_results:
        unique_walls = list(dict.fromkeys(r["wall_s"] for r in all_results))
        agg["speed_img_per_s"] = float(n / np.sum(unique_walls)) if unique_walls else 0.0

    # COVR slice of the config payload; the runtime owns the exact keys.
    covr_config: Dict[str, Any] = {}
    if covr_runtime is not None:
        covr_config.update(covr_runtime.config_payload(
            covr_forced_template_id=(
                covr_forced_template.template_id
                if covr_forced_template is not None
                else (
                    covr_forced_strategy.strategy_id
                    if covr_forced_strategy is not None else None)),
            covr_session_id=covr_session_id,
            covr_version_key=(
                covr_version.key if covr_version is not None else None),
        ))
    else:
        covr_config.update({
            "covr_shadow": bool(covr_recorder is not None),
            "covr_template_bandit": bool(covr_bandit is not None),
            "covr_force_template_id": (
                covr_forced_template.template_id
                if covr_forced_template is not None else None),
            "covr_session_id": covr_session_id,
            "covr_version_key": (
                covr_version.key if covr_version is not None else None),
            "covr_profile_stages": bool(getattr(
                args, "covr_profile_stages", False)),
        })

    results = {
        "config": {
            "model": "dit",
            "task": "c2i",
            "dataset": dataset_name,
            # The seed drives BOTH the dataset shuffle (ImageNetDataset does
            # RandomState(seed).shuffle) and, through it, which source image
            # every global_idx names. Per-image analyses pair arms by
            # global_idx, so a run whose seed is unknown cannot be paired
            # against another run at all — record it.
            "seed": args.seed,
            # Which independent latent draw this run used for its images.
            # Pairing runs by global_idx is only sound when both consumed the
            # same seed AND the same offset; record it so downstream analysis
            # can verify two runs differ in draws without trusting dir names.
            "latent_seed_offset": int(getattr(args, "latent_seed_offset", 0)),
            "dataset_start_index": dataset_start_index,
            "resume_sample_offset": covr_resume_sample_offset,
            "generation_start_index": generation_start_index,
            "method": args.method,
            "n_prompts": n,
            "batch_size": args.batch_size,
            "total_images": total_images,
            "num_steps": args.num_steps,
            "rel_l1_thresh": args.thresh if args.method == "teacache" else None,
            "coefficients": coefficients if args.method == "teacache" else None,
            "speca_base_threshold": args.speca_base_threshold if args.method == "speca" else None,
            "speca_decay_rate": args.speca_decay_rate if args.method == "speca" else None,
            "speca_min_taylor_steps": args.speca_min_taylor_steps if args.method == "speca" else None,
            "speca_max_taylor_steps": args.speca_max_taylor_steps if args.method == "speca" else None,
            "speca_error_metric": args.speca_error_metric if args.method == "speca" else None,
            "guidance_scale": args.guidance_scale,
            "ttt": args.ttt,
            "ttt_lr": args.ttt_lr if args.ttt else None,
            "ttt_micro_epochs": args.ttt_micro_epochs if args.ttt else None,
        },
        "aggregate": agg,
    }
    results["config"].update(covr_config)

    # ===================================================================
    # 8. Save
    # ===================================================================
    print("\n[5] Saving results...")
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  Results → {results_path}")

    _print_summary(results, selected, args.method)
    return results


# ===========================================================================
# Helpers
# ===========================================================================

def _load_dit_coefficients(coef_path: Optional[str] = None):
    """Load DiT-specific TeaCache coefficients."""
    if coef_path:
        with open(coef_path) as f:
            data = json.load(f)
            return data.get("coefficients", load_coefficients())
    import os as _os
    dit_coef_path = _os.path.join(_os.path.dirname(__file__), "dit_coef.json")
    if _os.path.exists(dit_coef_path):
        with open(dit_coef_path) as f:
            data = json.load(f)
            return data.get("coefficients", load_coefficients())
    return load_coefficients()


def _print_summary(results: Dict, selected: List[str], method: str):
    """Print a short summary of results."""
    agg = results.get("aggregate", {})
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for k, v in sorted(agg.items()):
        if isinstance(v, (int, float)):
            print(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    if method in ("teacache", "speca"):
        print(f"  skip_ratio: {agg.get('skip_ratio', '?')}")
    print("=" * 50)
