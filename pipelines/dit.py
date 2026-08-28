# -*- coding: utf-8 -*-
"""DiT orchestrator — DiTGenerator (VAE / scheduler / denoise loops / TTT).

Moved here from run_dit.py by the layered-structure refactor (2026-08-27);
run_dit.py keeps the top-level ``run_c2i`` evaluation entry and imports this
class. See ``.claude/project-structure.md``.
"""

import copy
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from config import DIT_REPO, IMAGENET_DIR, OUTPUT_DIR, load_coefficients
from utils import (
    decode_latent, save_image, pil_to_tensor, ensure_real_299,
    latent_seed_for_index, prune_checkpoints,
)
from pipelines.hooks.covr_hook import (
    _cache_scheduler_timestep_values,
    _covr_context,
    _covr_full_rollout,
    _covr_hash_sample,
    _covr_scheduler_pair,
    _covr_shadow_full,
    _covr_teacache_terminal_skip,
    _covr_transition_components,
)
from models.dit import DiTTransformer2D
from accelerators.teacache import (
    teacache_init, teacache_reset, teacache_step,
    teacache_boundary_snapshot,
)
from accelerators.covr_viability import (
    COVRViabilityRecorder,
    extract_prefix_features,
    flatten_prefix_features,
)
from accelerators.speca import SpecACache, SpecAState, speca_init
from accelerators.strategy_dispatch import apply_strategy
from accelerators.registry import get_adapter, is_registered
from accelerators.covr_runtime import COVRRuntime, COVRTrajectoryAssignment
from accelerators.covr import (
    ActionAuditEvent,
    COVRAction,
    transition_defect_batch,
)
from models.ttt_plugin import (
    SessionAdaLNModulator,
    ttt_state_init,
    ttt_reset_for_image,
    ttt_train_step,
    ttt_record_skip,
    ttt_session_stats,
)
from models.dit import (
    set_vfl_step_info,
    set_vfl_sample_id,
    get_vfl_buffer,
)
from feedback.vfl.lora_adapter import (
    attach_lora_all_layers,
    freeze_backbone,
    count_lora_params,
    load_lora_checkpoint,
    find_latest_checkpoint,
    compute_timestep_emb_for_transformer,
    set_lora_t_emb,
    clear_lora_t_emb,
)

class DiTGenerator:
    """Orchestrator for DiT-2-256 class-conditional generation.

    Manages VAE / scheduler / device / dtype / prompt-encoding /
    metrics coordination. The transformer is the new ``DiTTransformer2D``.

    Parameters
    ----------
    num_steps : int
    device : str
    dtype : torch.dtype
    """

    def __init__(self, num_steps: int = 20, device: str = "cuda",
                 dtype: torch.dtype = torch.float16, debug: bool = False):
        self.num_steps = num_steps
        self.device = device
        self._dtype = dtype
        self._debug = debug
        self._transformer: Optional[DiTTransformer2D] = None
        self._vae = None
        self._scheduler = None
        self._latent_shape = None
        self._id2label = None
        self.null_class = DIT_NULL_CLASS

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        """Load DiT transformer weights + VAE."""
        if self._transformer is not None:
            return

        if self._debug:
            # Debug mode: tiny model (hidden_dim=96), random weights,
            # no pretrained files needed. Structure identical to full DiT.
            self._transformer = DiTTransformer2D(
                num_attention_heads=4,
                attention_head_dim=24,
                in_channels=4,
                out_channels=8,
                num_layers=1,
                sample_size=32,
                patch_size=2,
                num_embeds_ada_norm=1001,
            )
            self._transformer.to(device=self.device, dtype=self._dtype)
            self._transformer.eval()

            # VAE: construct from known SD VAE config (random weights).
            from diffusers import AutoencoderKL
            self._vae = AutoencoderKL.from_config({
                "_class_name": "AutoencoderKL",
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": [
                    "DownEncoderBlock2D", "DownEncoderBlock2D",
                    "DownEncoderBlock2D", "DownEncoderBlock2D"],
                "up_block_types": [
                    "UpDecoderBlock2D", "UpDecoderBlock2D",
                    "UpDecoderBlock2D", "UpDecoderBlock2D"],
                "block_out_channels": [64, 128, 256, 256],
                "latent_channels": 4,
                "layers_per_block": 1,
                "sample_size": 256,
                "scaling_factor": 0.18215,
            }).to(device=self.device, dtype=self._dtype)
            self._vae.eval()
        else:
            # Transformer
            self._transformer = DiTTransformer2D.from_pretrained(DIT_REPO)
            self._transformer.to(device=self.device, dtype=self._dtype)
            self._transformer.eval()

            # VAE
            from diffusers import AutoencoderKL
            vae_path = os.path.join(DIT_REPO, "vae")
            self._vae = AutoencoderKL.from_pretrained(
                vae_path, local_files_only=True,
            ).to(device=self.device, dtype=self._dtype)
            self._vae.eval()

        # Derived props
        self._latent_shape = (1, 4, DIT_LATENT_SIZE, DIT_LATENT_SIZE)
        self.null_class = self._transformer.config.num_embeds_ada_norm

        # id2label
        model_index_path = os.path.join(DIT_REPO, "model_index.json")
        if not self._debug and os.path.exists(model_index_path):
            with open(model_index_path) as f:
                self._id2label = json.load(f).get("id2label", {})
        else:
            self._id2label = {}

        self._build_scheduler()
        n_blocks = len(self._transformer.transformer_blocks)
        print(f"  [DiT] loaded. blocks={n_blocks}")

    def unload(self):
        """Free GPU memory."""
        del self._transformer
        del self._vae
        self._transformer = None
        self._vae = None
        torch.cuda.empty_cache()

    # ---- Properties (required by FLOPsMetric / eval code) ----

    @property
    def transformer(self) -> DiTTransformer2D:
        if self._transformer is None:
            raise RuntimeError("DiTGenerator not loaded. Call .load() first.")
        return self._transformer

    @property
    def vae(self):
        if self._vae is None:
            raise RuntimeError("DiTGenerator not loaded. Call .load() first.")
        return self._vae

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def latent_shape(self):
        return self._latent_shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def id2label(self):
        return self._id2label

    def _build_scheduler(self):
        """Build DDIM scheduler from default config."""
        if self._debug:
            sched = DDIMScheduler(
                num_train_timesteps=1000,
                prediction_type="epsilon",
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
            )
            sched.set_timesteps(self.num_steps, device=self.device)
            _cache_scheduler_timestep_values(sched)
            self._scheduler = sched
            return
        cfg = DDIMScheduler.load_config(
            os.path.join(DIT_REPO, "scheduler", "scheduler_config.json"))
        sched = DDIMScheduler.from_config(cfg)
        sched.set_timesteps(self.num_steps, device=self.device)
        self._scheduler = sched

    def rebuild_scheduler(self):
        self._build_scheduler()

    # ------------------------------------------------------------------
    # Prompt encoding (class label → tensor)
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: Union[int, str, List]) -> torch.Tensor:
        """Convert prompt(s) to class-label tensor(s).

        Parameters
        ----------
        prompt : int, str, or list of int/str
            Single label → tensor of shape (1,).
            List of labels → tensor of shape (B,).

        Returns
        -------
        torch.LongTensor of shape (B,) on the correct device.
        """
        if isinstance(prompt, list):
            labels = [self._encode_single(p) for p in prompt]
            return torch.cat(labels, dim=0)
        return self._encode_single(prompt)

    def _encode_single(self, prompt: Union[int, str]) -> torch.Tensor:
        """Convert a single prompt to a class-label tensor of shape (1,)."""
        if isinstance(prompt, int):
            return torch.tensor([prompt], device=self.device, dtype=torch.long)
        try:
            return torch.tensor([int(prompt)], device=self.device, dtype=torch.long)
        except (ValueError, TypeError):
            pass

        if self._id2label:
            for idx, name in self._id2label.items():
                if prompt.lower() in name.lower():
                    return torch.tensor([int(idx)], device=self.device, dtype=torch.long)

        raise ValueError(
            f"Could not convert prompt '{prompt}' to a class label. "
            f"Provide an int (0-{self.null_class - 1}) or a valid ImageNet class name.")

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: Union[int, str, List],
                 seed: Union[int, List[int]],
                 guidance_scale: float = 4.0,
                 method: str = "baseline",
                 teacache_state: Optional[dict] = None,
                 cache_dic: Optional[SpecACache] = None,
                 current: Optional[SpecAState] = None,
                 ddim_steps: Optional[int] = None,
                 covr_trajectory: Optional[
                     COVRTrajectoryAssignment] = None,
                 viability_recorder: Optional[
                     COVRViabilityRecorder] = None,
                 covr_runtime: Optional[COVRRuntime] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate image(s).

        Parameters
        ----------
        method : str
            "baseline" | "teacache" | "ddim" | "speca"
        teacache_state : dict, optional
            TeaCache state (from ``teacache_init``). Used when method="teacache".
        cache_dic, current : dict, optional
            SpecA state dicts. Used when method="speca".
        ddim_steps : int, optional
            Override step count for DDIM (default: self.num_steps).
        """
        covr_profiler = (
            covr_trajectory.profiler if covr_trajectory is not None else None)
        self._build_scheduler()

        # Normalise to lists
        if isinstance(prompt, list):
            prompts = prompt
            seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        else:
            prompts = [prompt]
            seeds = [seed] if isinstance(seed, int) else seed
        B = len(prompts)

        cond_labels = self.encode_prompt(prompts)  # (B,)

        # CFG: double once — cat([cond, null]) → cond at index 0
        if guidance_scale > 1.0:
            null_labels = torch.full((B,), self.null_class,
                                     device=self.device, dtype=torch.long)
            class_labels = torch.cat([cond_labels, null_labels], dim=0)
        else:
            class_labels = cond_labels

        # Build scheduler (DDIM uses potentially different step count)
        if method == "ddim" and ddim_steps is not None:
            sched = DDIMScheduler.from_config(self._scheduler.config)
            sched.set_timesteps(ddim_steps, device=self.device)
            _cache_scheduler_timestep_values(sched)
            num_steps = ddim_steps
        else:
            sched = self._scheduler
            num_steps = self.num_steps

        # Run denoising loop
        denoise_token = (
            covr_profiler.start_gpu("denoise_loop")
            if covr_profiler is not None else None)
        latent = self._denoise_loop(
            class_labels, seeds, guidance_scale,
            sched, method=method,
            teacache_state=teacache_state,
            cache_dic=cache_dic, current=current,
            covr_trajectory=covr_trajectory,
            viability_recorder=viability_recorder,
            covr_runtime=covr_runtime,
        )
        if covr_profiler is not None:
            covr_profiler.stop_gpu(denoise_token)

        # Unchunk: cond half (index 0 — cond comes first)
        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[0]

        decode_token = (
            covr_profiler.start_gpu("vae_decode")
            if covr_profiler is not None else None)
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = decode_latent(self.vae, latent, scaling_factor, self._dtype)
        if covr_profiler is not None:
            covr_profiler.stop_gpu(decode_token)
        return latent, image

    # ------------------------------------------------------------------
    # Denoising loop — explicit, all methods visible
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _denoise_loop(self, class_labels: torch.Tensor,
                       seed: Union[int, List[int]],
                       guidance_scale: float,
                       scheduler,
                       method: str,
                       teacache_state: Optional[dict],
                       cache_dic: Optional[SpecACache],
                       current: Optional[SpecAState],
                       covr_trajectory: Optional[
                           COVRTrajectoryAssignment] = None,
                       viability_recorder: Optional[
                           COVRViabilityRecorder] = None,
                       covr_runtime: Optional[COVRRuntime] = None,
                       ) -> torch.Tensor:
        """Single denoising loop with method dispatch.

        TeaCache logic lives here (loop-level), NOT inside the model.
        SpecA logic lives inside the model (via current/cache_dic).
        """
        transformer = self.transformer
        base_bs = class_labels.shape[0] // 2 if guidance_scale > 1.0 else class_labels.shape[0]
        if covr_trajectory is None:
            covr_recorder = None
            covr_trajectory_id = 0
            covr_sample_ids = ()
            covr_safety_sample_rate = 0.0
            covr_safety_chain_threshold = 0
            covr_sentinel_start_idx = None
            covr_sentinel_horizon = 0
            covr_feedback_sink = None
            covr_sentinel_selected = False
            covr_terminal_reward_active = False
            covr_profiler = None
            covr_session_id = ""
        else:
            covr_recorder = covr_trajectory.recorder
            covr_trajectory_id = covr_trajectory.trajectory_id
            covr_sample_ids = covr_trajectory.sample_ids
            covr_safety_sample_rate = covr_trajectory.safety_sample_rate
            covr_safety_chain_threshold = covr_trajectory.safety_chain_threshold
            covr_sentinel_start_idx = covr_trajectory.sentinel_start_idx
            covr_sentinel_horizon = covr_trajectory.sentinel_horizon
            covr_feedback_sink = covr_trajectory.feedback_sink
            covr_sentinel_selected = covr_trajectory.sentinel_selected
            covr_terminal_reward_active = (
                covr_trajectory.terminal_reward_active)
            covr_profiler = covr_trajectory.profiler
            covr_session_id = covr_trajectory.session_id

        if covr_recorder is not None and method != "speca":
            raise ValueError("COVR shadow auditing requires method='speca'")
        if covr_trajectory is not None and len(covr_sample_ids) != base_bs:
            raise ValueError("COVR sample IDs must match the unguided batch size")
        if covr_sentinel_start_idx is not None:
            if covr_feedback_sink is None or covr_sentinel_horizon <= 0:
                raise ValueError("H-step sentinel requires a feedback sink and horizon")
        # Deferred-commit contextual bandit: covr_runtime.commit_arm selects the
        # arm after ``prefix_steps`` forced-calc denoise steps. prefix_steps==0
        # means non-contextual (the arm was selected in begin_trajectory).
        covr_prefix_steps = (
            int(covr_trajectory.prefix_steps) if covr_trajectory is not None else 0)
        covr_contextual_active = (
            covr_prefix_steps > 0 and covr_runtime is not None
            and covr_trajectory is not None)
        covr_contextual_committed = False
        sentinel_start_latent = None
        sentinel_reference_latent = None

        # Init latents
        if isinstance(seed, list):
            assert len(seed) == base_bs
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seed]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seed)

        if isinstance(generators, list):
            noises = []
            for g in generators:
                shape_one = (1,) + self._latent_shape[1:]
                noises.append(torch.randn(shape_one, device=self.device,
                                           dtype=self._dtype, generator=g))
            latents = torch.cat(noises, dim=0) * scheduler.init_noise_sigma
        else:
            shape = (base_bs,) + self._latent_shape[1:]
            latents = torch.randn(shape, device=self.device, dtype=self._dtype,
                                  generator=generators) * scheduler.init_noise_sigma

        # CFG doubling
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        timesteps = scheduler.timesteps
        timestep_values = getattr(scheduler, "_host_timestep_values", None)
        if timestep_values is None:
            _cache_scheduler_timestep_values(scheduler)
            timestep_values = getattr(scheduler, "_host_timestep_values")
        if covr_sentinel_start_idx is not None and (
                covr_sentinel_start_idx < 0 or
                covr_sentinel_start_idx + covr_sentinel_horizon > len(timesteps)):
            raise ValueError("H-step sentinel exceeds the denoising trajectory")
        for step_idx, (t, timestep_value) in enumerate(
                zip(timesteps, timestep_values)):
            # VFL: track current step + real timestep for event recording hooks
            set_vfl_step_info(
                step_idx, len(timesteps), timestep_actual=int(timestep_value))
            # Time-conditioned LoRA: cache t_emb once per step so the 168
            # LoRALinear forwards (28 blocks × 6 Linears) inside the upcoming
            # transformer call all read the same value without recomputing.
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            _t_emb = compute_timestep_emb_for_transformer(
                transformer, current_t,
                class_labels=class_labels,
                hidden_dtype=latents.dtype,
            )
            if _t_emb is not None:
                set_lora_t_emb(_t_emb)
            if covr_sentinel_start_idx == step_idx:
                assert covr_feedback_sink is not None
                sentinel_start_latent = latents.detach().clone()
                sentinel_token = (
                    covr_profiler.start_gpu(
                        "delayed_sentinel_shadow_full", required=True)
                    if covr_profiler is not None else None)
                sentinel_reference_latent = _covr_full_rollout(
                    transformer, scheduler, timesteps, step_idx,
                    covr_sentinel_horizon, latents, class_labels,
                    guidance_scale, transformer.config.in_channels)
                if covr_profiler is not None:
                    covr_profiler.stop_gpu(sentinel_token)
                covr_feedback_sink["terminal_full_steps"] = (
                    covr_feedback_sink.get("terminal_full_steps", 0)
                    + covr_sentinel_horizon)
            latent_input = scheduler.scale_model_input(latents, t)
            terminal_teacache_skip = None
            if (method == "teacache" and teacache_state is not None
                    and covr_sentinel_selected
                    and covr_sentinel_start_idx is None
                    and step_idx == len(timesteps) - 1
                    and covr_terminal_reward_active
                    and covr_feedback_sink is not None):
                refresh_mask = teacache_state.get("refresh_mask")
                cnt = int(teacache_state["cnt"])
                terminal_will_calc = (
                    refresh_mask is None or bool(refresh_mask[cnt]))
                if terminal_will_calc:
                    terminal_token = (
                        covr_profiler.start_gpu(
                            "terminal_fidelity_shadow_full", required=True)
                        if covr_profiler is not None else None)
                    terminal_teacache_skip = _covr_teacache_terminal_skip(
                        transformer, latent_input, current_t, class_labels,
                        guidance_scale, teacache_state)
                    if covr_profiler is not None:
                        covr_profiler.stop_gpu(terminal_token)

            # --------------- method dispatch ---------------
            if method == "teacache":
                if guidance_scale > 1.0:
                    noise_pred = transformer.forward_with_cfg(
                        latent_input, current_t,
                        current=None, cache_dic=None,
                        teacache_state=teacache_state,
                        class_labels=class_labels, cfg_scale=guidance_scale,
                    )
                else:
                    noise_pred = transformer(
                        latent_input, timestep=current_t,
                        teacache_state=teacache_state,
                        class_labels=class_labels, return_dict=False,
                    )[0]
                if teacache_state is not None:
                    teacache_step(teacache_state)

            elif method == "speca":
                # SpecA: state threaded through model
                if current is not None:
                    current.step = len(timesteps) - 1 - step_idx
                    # Time-bucket-aware max_taylor schedule (structure-aware
                    # design): aggressive early, conservative late. Applied
                    # before forward because speca_cal_type reads
                    # cache_dic.max_taylor_steps inside the model.
                    _sched = getattr(cache_dic, "bucket_schedule", None)
                    if _sched is not None:
                        _bucket = min(int(step_idx * 3 / len(timesteps)), 2)
                        cache_dic.max_taylor_steps = _sched[_bucket]
                speca_token = (
                    covr_profiler.start_gpu("speca_forward")
                    if covr_profiler is not None else None)
                if guidance_scale > 1.0:
                    noise_pred = transformer.forward_with_cfg(
                        latent_input, current_t,
                        current=current, cache_dic=cache_dic,
                        class_labels=class_labels, cfg_scale=guidance_scale,
                    )
                else:
                    noise_pred = transformer(
                        latent_input, timestep=current_t,
                        current=current, cache_dic=cache_dic,
                        class_labels=class_labels, return_dict=False,
                    )[0]
                if covr_profiler is not None:
                    speca_stage = (
                        "speca_full" if current is not None and
                        current.type == "full" else "speca_taylor")
                    covr_profiler.stop_gpu(speca_token, speca_stage)

            else:
                # baseline / ddim: vanilla forward (no cache state)
                if guidance_scale > 1.0:
                    noise_pred = transformer.forward_with_cfg(
                        latent_input, current_t,
                        current=None, cache_dic=None,
                        class_labels=class_labels, cfg_scale=guidance_scale,
                    )
                else:
                    noise_pred = transformer(
                        latent_input, timestep=current_t,
                        class_labels=class_labels, return_dict=False,
                    )[0]

            # Learned-sigma: keep noise channels, discard variance channels
            in_channels = int(getattr(transformer.config, "in_channels"))
            out_channels = int(getattr(transformer.config, "out_channels"))
            if out_channels // 2 == in_channels:
                noise_pred = noise_pred[:, :in_channels]

            # ---- Deferred-commit contextual bandit: buffer prefix features
            # and commit the arm once the forced-calc prefix completes. The
            # prefix strategy (installed in begin_trajectory) forces every step
            # to recompute, so the residual and previous_modulated_input are
            # fresh when the dynamic threshold path resumes at the next step.
            if (covr_contextual_active and not covr_contextual_committed
                    and teacache_state is not None
                    and step_idx < covr_prefix_steps):
                prefix_feats = extract_prefix_features(
                    latent_input[:1], noise_pred[:1], batch_size=1)
                covr_trajectory.prefix_feature_buffer.append(prefix_feats)
                if step_idx == covr_prefix_steps - 1:
                    context_vector = flatten_prefix_features(
                        covr_trajectory.prefix_feature_buffer)
                    commit_token = (
                        covr_profiler.start_gpu("strategy_selection")
                        if covr_profiler is not None else None)
                    selection = covr_runtime.commit_arm(
                        covr_trajectory, context_vector)
                    if covr_profiler is not None:
                        covr_profiler.stop_gpu(commit_token)
                    chosen_strategy = selection.strategy
                    if chosen_strategy is None:
                        raise RuntimeError(
                            "contextual commit did not resolve a strategy")
                    gamma = chosen_strategy.params.get("rel_l1_thresh")
                    if gamma is None:
                        raise RuntimeError(
                            "contextual commit selected a strategy without "
                            "rel_l1_thresh; only TeaCache threshold arms are "
                            "supported in deferred-commit mode")
                    # Drop the forced-calc mask; the chosen threshold drives the
                    # dynamic accumulate-vs-threshold path from step K onward.
                    # The mask branch already zeroed the accumulator each step
                    # and kept previous_modulated_input current, so the dynamic
                    # branch resumes cleanly (teacache.py:316 reads rel_l1_thresh
                    # fresh each call).
                    teacache_state["refresh_mask"] = None
                    teacache_state["rel_l1_thresh"] = float(gamma)
                    covr_contextual_committed = True

            # Terminal fidelity for bandit reward at the last denoising step.
            # A TeaCache arm whose last step recomputes is paired with the
            # isolated forced-skip candidate captured above; otherwise one
            # full shadow forward supplies the counterfactual. This avoids the
            # degenerate full-vs-full zero reward of dynamic threshold arms.
            # Other accelerators retain the candidate-vs-full shadow path.
            # Whether this method exposes that reward is the adapter's call.
            # (``covr_terminal_reward_active`` is the loop-level gate: in
            # bandit mode it duplicates the old local adapter check, and in
            # forced mode it enables the reward without a bandit.)
            _terminal_reward_active = bool(covr_terminal_reward_active) or (
                is_registered(method)
                and get_adapter(method).terminal_reward_active({
                    "cache_dic": cache_dic,
                    "current": current,
                    "teacache_state": teacache_state,
                })
            )
            if (covr_sentinel_selected and covr_sentinel_start_idx is None
                    and step_idx == len(timesteps) - 1
                    and _terminal_reward_active
                    and covr_feedback_sink is not None):
                terminal_token = None
                terminal_approx = noise_pred
                terminal_full_steps = 0
                if terminal_teacache_skip is not None:
                    terminal_approx = terminal_teacache_skip
                    t_out = int(getattr(transformer.config, "out_channels"))
                    if t_out // 2 == in_channels:
                        terminal_approx = terminal_approx[:, :in_channels]
                    terminal_full = noise_pred
                else:
                    terminal_token = (
                        covr_profiler.start_gpu(
                            "terminal_fidelity_shadow_full", required=True)
                        if covr_profiler is not None else None)
                    terminal_full = _covr_shadow_full(
                        transformer, latent_input, current_t,
                        class_labels, guidance_scale)
                    t_out = int(getattr(transformer.config, "out_channels"))
                    if t_out // 2 == in_channels:
                        terminal_full = terminal_full[:, :in_channels]
                    terminal_full_steps = 1
                x_prev_approx, x_prev_full = _covr_scheduler_pair(
                    scheduler, terminal_approx, terminal_full, t, latents)
                covr_feedback_sink["terminal_fidelity_loss_tensor"] = F.mse_loss(
                    x_prev_approx[:base_bs].float(),
                    x_prev_full[:base_bs].float(),
                )
                covr_feedback_sink["terminal_full_steps"] = (
                    covr_feedback_sink.get("terminal_full_steps", 0)
                    + terminal_full_steps)
                if covr_profiler is not None and terminal_token is not None:
                    covr_profiler.stop_gpu(terminal_token)

            if (covr_recorder is not None and covr_recorder.enabled
                    and method == "speca" and current is not None
                    and cache_dic is not None and current.type == "Taylor"):
                context = _covr_context(
                    covr_recorder, scheduler, timesteps, step_idx,
                    timestep_value, current)
                full_noise_pred = _covr_shadow_full(
                    transformer, latent_input, current_t,
                    class_labels, guidance_scale)
                if out_channels // 2 == in_channels:
                    full_noise_pred = full_noise_pred[:, :in_channels]
                x_prev_approx, x_prev_full = _covr_scheduler_pair(
                    scheduler, noise_pred, full_noise_pred, t, latents)
                transition = transition_defect_batch(
                    x_prev_approx[:base_bs],
                    x_prev_full[:base_bs],
                    latents[:base_bs],
                )
                local_probe_error = (
                    current.last_layer_error if cache_dic.check else None)
                event = ActionAuditEvent(
                    session_id=covr_recorder.session_id,
                    trajectory_id=covr_trajectory_id,
                    sample_ids=tuple(covr_sample_ids),
                    class_ids=tuple(
                        int(label.item()) for label in class_labels[:base_bs]),
                    version_key=covr_recorder.version.key,
                    context=context,
                    committed_action=COVRAction.ACCEPT,
                    committed_propensity=1.0,
                    audit_action=COVRAction.REFRESH,
                    audit_propensity=1.0,
                    policy="shadow_static_speca",
                    incremental_cost=1.0,
                    one_step_transition=transition,
                    local_probe_error=local_probe_error,
                )
                covr_recorder.record(event)

            # Safety shadow gate: skip when Taylor chain is short.
            # Benchmark shows defect correlates with chain length (r=+0.63);
            # first N steps after each full step have low defect → skip shadow.
            _skip_safety_shadow = (
                covr_safety_chain_threshold > 0
                and cache_dic is not None
                and cache_dic.taylor_step_counter <= covr_safety_chain_threshold
            )

            if (covr_trajectory is not None
                    and covr_safety_sample_rate > 0.0
                    and method == "speca"
                    and current is not None and cache_dic is not None
                    and current.type == "Taylor"
                    and not _skip_safety_shadow
                    and _covr_hash_sample(
                        covr_session_id, covr_trajectory_id, step_idx,
                        covr_safety_sample_rate, "safety")):
                if covr_feedback_sink is None:
                    raise ValueError("COVR safety sampling requires a feedback sink")
                safety_token = (
                    covr_profiler.start_gpu("safety_shadow_full", required=True)
                    if covr_profiler is not None else None)
                full_noise_pred = _covr_shadow_full(
                    transformer, latent_input, current_t,
                    class_labels, guidance_scale)
                covr_feedback_sink["safety_full_steps"] = (
                    covr_feedback_sink.get("safety_full_steps", 0) + 1)
                if out_channels // 2 == in_channels:
                    full_noise_pred = full_noise_pred[:, :in_channels]
                x_prev_approx, x_prev_full = _covr_scheduler_pair(
                    scheduler, noise_pred, full_noise_pred, t, latents)
                numerator, denominator = _covr_transition_components(
                    x_prev_approx[:base_bs], x_prev_full[:base_bs],
                    latents[:base_bs])
                covr_feedback_sink.setdefault("pending_safety", []).append((
                    step_idx, torch.stack((numerator, denominator))))
                if covr_profiler is not None:
                    covr_profiler.stop_gpu(safety_token)

            # ---- VFL: anchor sample collection (low frequency) ----
            if step_idx % 5 == 0:
                _vfl_buf = get_vfl_buffer()
                if _vfl_buf is not None and len(_vfl_buf._anchor_samples) < 50:
                    _vfl_buf.add_anchor_from_tensors(
                        prompt=int(class_labels[0].item()) if class_labels is not None else 0,
                        latent=latent_input[0:1].detach(),
                        timestep=current_t[:1],
                        target=noise_pred[0:1].detach(),
                        model="dit",
                    )

            if (viability_recorder is not None
                    and step_idx < viability_recorder.prefix_steps):
                boundary = None
                if teacache_state is not None:
                    boundary = teacache_boundary_snapshot(teacache_state)
                viability_recorder.record_step(
                    step_idx=step_idx,
                    timestep=int(timestep_value),
                    latent_input=latent_input,
                    noise_pred=noise_pred,
                    boundary=boundary,
                )
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if (sentinel_reference_latent is not None and
                    covr_sentinel_start_idx is not None and
                    step_idx == covr_sentinel_start_idx + covr_sentinel_horizon - 1):
                assert covr_feedback_sink is not None
                assert sentinel_start_latent is not None
                h_step_token = (
                    covr_profiler.start_gpu(
                        "delayed_sentinel_feedback", required=True)
                    if covr_profiler is not None else None)
                numerator, denominator = _covr_transition_components(
                    latents[:base_bs], sentinel_reference_latent[:base_bs],
                    sentinel_start_latent[:base_bs])
                covr_feedback_sink["h_step_components_tensor"] = torch.stack((
                    numerator.mean(), denominator.mean()))
                if covr_profiler is not None:
                    covr_profiler.stop_gpu(h_step_token)

        # Clear the LoRA t_emb cache so the next image starts clean.
        clear_lora_t_emb()
        # Deferred-commit contextual bandit: every contextual trajectory must
        # commit exactly once during the prefix. If the loop exited without
        # committing (e.g. num_steps < prefix_steps, or the hook was skipped),
        # the bandit's pending arm would never resolve — fail loudly rather
        # than silently degrading to all-calc.
        if covr_contextual_active and not covr_contextual_committed:
            raise RuntimeError(
                "contextual trajectory exited the denoise loop before the "
                f"deferred commit (prefix_steps={covr_prefix_steps}, "
                f"num_steps={len(timesteps)})")
        return latents

    # ==================================================================
    # Session-TTT generation path (Phase 3)
    # ==================================================================
    #
    # These methods are deliberately SEPARATE from generate / _denoise_loop:
    #   * they are NOT under ``@torch.no_grad()`` — the plugin (φ) must build
    #     a graph so its calc-step prediction can be backpropped;
    #   * they call ``ttt_train_step`` after each calc step (one AdamW update
    #     on φ only — Θ is frozen via requires_grad_(False) at the runner);
    #   * they DO NOT disturb the existing 20-combo benchmark path.
    # All non-plugin work is wrapped in ``torch.no_grad()`` so only φ's graph
    # survives — keeping memory bounded.

    def generate_ttt(self, prompt: Union[int, str, List],
                     seed: Union[int, List[int]],
                     guidance_scale: float = 4.0,
                     teacache_state: Optional[dict] = None,
                     ttt_state: Optional[dict] = None,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one image in Session-TTT mode.

        Mirrors ``generate`` but: (1) no ``@torch.no_grad()`` decorator; (2)
        threads ``ttt_state`` so calc steps distil the backbone into φ and skip
        steps route the cache through φ; (3) the caller owns the TeaCache step
        counter and the γ threshold (set ``teacache_state['rel_l1_thresh']``).

        Returns ``(latent, image)`` — same contract as ``generate``. The latent
        retains the CFG-doubled batch if guidance is on; the caller takes the
        cond half for metric comparison.
        """
        self._build_scheduler()

        if isinstance(prompt, list):
            prompts = prompt
            seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        else:
            prompts = [prompt]
            seeds = [seed] if isinstance(seed, int) else seed
        B = len(prompts)

        cond_labels = self.encode_prompt(prompts)

        if guidance_scale > 1.0:
            null_labels = torch.full((B,), self.null_class,
                                     device=self.device, dtype=torch.long)
            class_labels = torch.cat([cond_labels, null_labels], dim=0)
        else:
            class_labels = cond_labels

        sched = self._scheduler
        latent = self._denoise_loop_ttt(
            class_labels, seeds, guidance_scale, sched,
            teacache_state=teacache_state, ttt_state=ttt_state,
        )

        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[0]

        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        # VAE decode never needs gradients.
        with torch.no_grad():
            image = decode_latent(self.vae, latent, scaling_factor, self._dtype)
        return latent, image

    def _denoise_loop_ttt(self, class_labels: torch.Tensor,
                          seed: Union[int, List[int]],
                          guidance_scale: float,
                          scheduler,
                          teacache_state: Optional[dict],
                          ttt_state: Optional[dict],
                          ) -> torch.Tensor:
        """Session-TTT denoise loop — Teacher/Student dispatch per step.

        Loop-level responsibilities (mirrors the ``method == "teacache"``
        branch of ``_denoise_loop`` but with TTT training):

          * init latents + CFG doubling — under no_grad (no graph needed);
          * per step: call ``forward_with_cfg_ttt`` (the model decides
            teacher/student internally based on TeaCache's calc/skip);
          * calc step → ``ttt_train_step`` (one AdamW step on φ);
          * skip step  → ``ttt_record_skip`` (telemetry only);
          * advance the TeaCache step counter (loop owns the counter, as in
            the vanilla TeaCache path);
          * learned-sigma split + scheduler.step.

        The ambient autograd context is ENABLED here (no decorator). Inside
        ``DiTTransformer2D._forward_ttt`` the 28-block teacher and the tail run
        under ``torch.no_grad()``; ONLY the plugin forward builds a graph. This
        keeps peak memory roughly proportional to one block's activation set.
        """
        transformer = self.transformer
        base_bs = (class_labels.shape[0] // 2
                   if guidance_scale > 1.0 else class_labels.shape[0])

        # Latent init — no graph needed.
        with torch.no_grad():
            if isinstance(seed, list):
                assert len(seed) == base_bs
                generators = [torch.Generator(device=self.device).manual_seed(s)
                              for s in seed]
            else:
                generators = torch.Generator(device=self.device).manual_seed(seed)

            if isinstance(generators, list):
                noises = []
                for g in generators:
                    shape_one = (1,) + self._latent_shape[1:]
                    noises.append(torch.randn(shape_one, device=self.device,
                                              dtype=self._dtype, generator=g))
                latents = torch.cat(noises, dim=0) * scheduler.init_noise_sigma
            else:
                shape = (base_bs,) + self._latent_shape[1:]
                latents = torch.randn(shape, device=self.device, dtype=self._dtype,
                                      generator=generators) * scheduler.init_noise_sigma

            if guidance_scale > 1.0:
                latents = torch.cat([latents, latents], dim=0)

        timesteps = scheduler.timesteps
        for step_idx, t in enumerate(timesteps):
            # VFL: track current step + real timestep for event recording hooks
            set_vfl_step_info(step_idx, len(timesteps), timestep_actual=int(t))
            # Time-conditioned LoRA: cache t_emb once per step so the 168
            # LoRALinear forwards (28 blocks × 6 Linears) inside the upcoming
            # transformer call all read the same value without recomputing.
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            _t_emb = compute_timestep_emb_for_transformer(
                transformer, current_t,
                class_labels=class_labels,
                hidden_dtype=latents.dtype,
            )
            if _t_emb is not None:
                set_lora_t_emb(_t_emb)
            latent_input = scheduler.scale_model_input(latents, t)

            # --- TTT forward (decides teacher/student inside) ---
            noise_pred = transformer.forward_with_cfg_ttt(
                latent_input, current_t,
                teacache_state=teacache_state, ttt_state=ttt_state,
                class_labels=class_labels, cfg_scale=guidance_scale,
            )

            # --- TTT training dispatch (loop owns this, mirroring
            #     teacache_step in the vanilla path) ---
            from accelerators.teacache import teacache_step
            last_decision = (teacache_state["decisions"][-1]
                             if teacache_state and teacache_state["decisions"]
                             else "calc")
            if last_decision == "calc":
                ttt_train_step(ttt_state)
            else:
                ttt_record_skip(ttt_state)
            teacache_step(teacache_state)

            # --- learned-sigma split + scheduler step (no graph) ---
            with torch.no_grad():
                if transformer.config.out_channels // 2 == transformer.config.in_channels:
                    noise_pred = noise_pred[:, :transformer.config.in_channels]
                latents = scheduler.step(
                    noise_pred.detach(), t, latents, return_dict=False)[0]

        clear_lora_t_emb()
        return latents


