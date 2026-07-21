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
import json
import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler

from config import (
    DIT_REPO, IMAGENET_DIR, OUTPUT_DIR,
    DEFAULT_REL_L1_THRESH, DEFAULT_NUM_STEPS,
    DDIM_FLOP_MATCHED_STEPS, load_coefficients,
)
from utils import CudaTimer, decode_latent, save_image, pil_to_tensor, ensure_real_299, get_vfl_checkpoint_dir, prune_checkpoints

from models.dit import (
    DiTTransformer2D, set_vfl_step_info, get_vfl_buffer, set_vfl_sample_id,
)
from verification_feedback_loop.lora_adapter import (
    set_lora_t_emb, clear_lora_t_emb,
    compute_timestep_emb_for_transformer,
    attach_lora_all_layers, count_lora_params,
    _swap_lora_weights, _load_state_into_wrappers,
)
from accelerators.teacache import (
    teacache_init, teacache_decide, teacache_cache_residual,
    teacache_apply_residual, teacache_step, teacache_reset,
    teacache_stats, compute_modulated_input_dit,
)
from accelerators.speca import SpecACache, SpecAState, speca_init
from accelerators.covr import (
    COVRAction,
    COVRContext,
    COVRVersion,
    CounterfactualEvent,
    ShadowAuditRecorder,
    summarize_taylor_cache,
    transition_defects,
)
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


def _covr_log_snr(scheduler, timestep) -> float:
    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        return 0.0
    index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    alpha = float(alphas_cumprod[index].detach().float().item())
    alpha = min(max(alpha, 1e-8), 1.0 - 1e-8)
    return float(np.log(alpha / (1.0 - alpha)))


def _covr_step_size(timesteps, step_idx: int) -> float:
    if step_idx + 1 >= len(timesteps):
        return 0.0
    current = float(timesteps[step_idx])
    following = float(timesteps[step_idx + 1])
    return abs(current - following) / max(abs(current), 1.0)


def _covr_scheduler_pair(scheduler, noise_approx, noise_full, timestep, latents):
    approx_scheduler = copy.deepcopy(scheduler)
    full_scheduler = copy.deepcopy(scheduler)
    x_prev_approx = approx_scheduler.step(
        noise_approx.detach(), timestep, latents.detach(), return_dict=False)[0]
    x_prev_full = full_scheduler.step(
        noise_full.detach(), timestep, latents.detach(), return_dict=False)[0]
    return x_prev_approx, x_prev_full


def _covr_shadow_full(transformer, latent_input, timestep, class_labels,
                      guidance_scale):
    if guidance_scale > 1.0:
        return transformer.forward_with_cfg(
            latent_input, timestep,
            current=None, cache_dic=None, teacache_state=None,
            class_labels=class_labels, cfg_scale=guidance_scale,
        )
    return transformer(
        latent_input, timestep=timestep,
        current=None, cache_dic=None, teacache_state=None,
        class_labels=class_labels, return_dict=False,
    )[0]


def _covr_context(recorder, scheduler, timesteps, step_idx, timestep,
                  current, cache_dic, cfg_disagreement):
    if current.activated_steps:
        distance = abs(current.step - current.activated_steps[-1])
    else:
        distance = 0
    cache_features = summarize_taylor_cache(cache_dic, distance)
    if recorder.max_events is None:
        remaining_budget = 1.0
    else:
        remaining_budget = max(
            recorder.max_events - recorder.count, 0) / recorder.max_events
    return COVRContext(
        step_idx=step_idx,
        num_steps=len(timesteps),
        timestep=float(timestep),
        log_snr=_covr_log_snr(scheduler, timestep),
        scheduler_step_size=_covr_step_size(timesteps, step_idx),
        distance_since_refresh=distance,
        taylor_term_norms=cache_features["taylor_term_norms"],
        order_2_4_disagreement=cache_features["order_2_4_disagreement"],
        attn_curvature=cache_features["attn_curvature"],
        mlp_curvature=cache_features["mlp_curvature"],
        cfg_draft_disagreement=cfg_disagreement,
        previous_defect=recorder.previous_defect,
        remaining_budget=remaining_budget,
    )


# ===========================================================================
# DiTGenerator — orchestrator (no monkeypatching)
# ===========================================================================

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
                 covr_recorder: Optional[ShadowAuditRecorder] = None,
                 covr_trajectory_id: int = 0,
                 covr_sample_ids: Optional[List[str]] = None,
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
            num_steps = ddim_steps
        else:
            sched = self._scheduler
            num_steps = self.num_steps

        # Run denoising loop
        latent = self._denoise_loop(
            class_labels, seeds, guidance_scale,
            sched, method=method,
            teacache_state=teacache_state,
            cache_dic=cache_dic, current=current,
            covr_recorder=covr_recorder,
            covr_trajectory_id=covr_trajectory_id,
            covr_sample_ids=covr_sample_ids,
        )

        # Unchunk: cond half (index 0 — cond comes first)
        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[0]

        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = decode_latent(self.vae, latent, scaling_factor, self._dtype)
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
                       covr_recorder: Optional[ShadowAuditRecorder] = None,
                       covr_trajectory_id: int = 0,
                       covr_sample_ids: Optional[List[str]] = None,
                       ) -> torch.Tensor:
        """Single denoising loop with method dispatch.

        TeaCache logic lives here (loop-level), NOT inside the model.
        SpecA logic lives inside the model (via current/cache_dic).
        """
        transformer = self.transformer
        base_bs = class_labels.shape[0] // 2 if guidance_scale > 1.0 else class_labels.shape[0]
        if covr_recorder is not None and method != "speca":
            raise ValueError("COVR shadow auditing requires method='speca'")
        if covr_sample_ids is None:
            covr_sample_ids = [
                f"{covr_trajectory_id}:{index}" for index in range(base_bs)]
        if len(covr_sample_ids) != base_bs:
            raise ValueError("COVR sample IDs must match the unguided batch size")

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
            cfg_draft_disagreement = 0.0

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
                if guidance_scale > 1.0:
                    if covr_recorder is not None and covr_recorder.enabled:
                        noise_pred = transformer.forward_with_cfg(
                            latent_input, current_t,
                            current=current, cache_dic=cache_dic,
                            class_labels=class_labels, cfg_scale=guidance_scale,
                            track_cfg_disagreement=True,
                        )
                        cfg_draft_disagreement = transformer.last_cfg_disagreement
                    else:
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

            if (covr_recorder is not None and covr_recorder.enabled
                    and method == "speca" and current is not None
                    and cache_dic is not None and current.type == "Taylor"):
                context = _covr_context(
                    covr_recorder, scheduler, timesteps, step_idx, t,
                    current, cache_dic, cfg_draft_disagreement)
                full_noise_pred = _covr_shadow_full(
                    transformer, latent_input, current_t,
                    class_labels, guidance_scale)
                if out_channels // 2 == in_channels:
                    full_noise_pred = full_noise_pred[:, :in_channels]
                x_prev_approx, x_prev_full = _covr_scheduler_pair(
                    scheduler, noise_pred, full_noise_pred, t, latents)
                defects = transition_defects(
                    x_prev_approx[:base_bs], x_prev_full[:base_bs], latents[:base_bs])
                local_probe_error = (
                    current.last_layer_error if cache_dic.check else None)
                for sample_index, defect in enumerate(defects):
                    event = CounterfactualEvent(
                        session_id=covr_recorder.session_id,
                        trajectory_id=covr_trajectory_id,
                        sample_id=covr_sample_ids[sample_index],
                        version_key=covr_recorder.version.key,
                        context=context,
                        action=COVRAction.REFRESH,
                        propensity=1.0,
                        policy="shadow_static_speca",
                        incremental_cost=1.0,
                        one_step_defect=defect,
                        local_probe_error=local_probe_error,
                        accepted_approximation=True,
                        metadata={
                            "committed_action": COVRAction.ACCEPT.value,
                            "class_id": int(class_labels[sample_index].item()),
                        },
                    )
                    if not covr_recorder.record(event):
                        break

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

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Clear the LoRA t_emb cache so the next image starts clean.
        clear_lora_t_emb()
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
            n_images=args.n_prompts, seed=args.seed)
    elif dataset_name == "coco":
        from dataset.coco import COCO30KDataset
        ds = COCO30KDataset(
            coco_dir=getattr(args, "coco_dir", None) or args.coco_dir,
            n_images=args.n_prompts, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    n = len(ds)

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
        from verification_feedback_loop import (
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
        )
        print(f"  TeaCache ready (γ={args.thresh})")
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
        print(f"  SpecA ready (base_thresh={args.speca_base_threshold}, "
              f"check_layer={check_layer}, "
              f"controller={args.compute_controller})")
    else:
        print("  Baseline (full DDIM, no acceleration)")

    covr_recorder = None
    if getattr(args, "covr_shadow", False):
        covr_session_id = args.covr_session_id or (
            f"{time.strftime('%Y%m%d-%H%M%S')}-seed{args.seed}")
        scheduler_instance = generator.scheduler
        scheduler_config = json.dumps(
            dict(getattr(scheduler_instance, "config", {})),
            sort_keys=True, default=str)
        speca_config = json.dumps(speca_init_kwargs, sort_keys=True, default=str)
        covr_version = COVRVersion(
            model="dit",
            base_model_version=(
                args.covr_base_model_version or os.path.expanduser(DIT_REPO)),
            scheduler=scheduler_instance.__class__.__name__,
            scheduler_config=scheduler_config,
            num_steps=args.num_steps,
            cfg_scale=args.guidance_scale,
            speca_config=speca_config,
        )
        covr_output_dir = args.covr_output_dir or os.path.join(output_dir, "covr")
        covr_recorder = ShadowAuditRecorder(
            output_dir=covr_output_dir,
            session_id=covr_session_id,
            version=covr_version,
            max_events=args.covr_max_events,
        )
        print(f"  COVR shadow recorder: {covr_recorder.event_path}")

    # ===================================================================
    # 5. Generate images
    # ===================================================================
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    total_images = n
    bs = args.batch_size
    print(f"\n[4] Generating {total_images} images ({args.method}, "
          f"{n} prompts in batches of ≤{bs})...")
    t_start = time.time()

    wall_times = []
    all_results = []
    global_idx = 0

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
        batch_indices = list(range(batch_start, batch_end))
        actual_bs = len(batch_indices)

        # Collect prompts (class labels) + seeds
        batch_inputs, batch_seeds = [], []
        for idx in batch_indices:
            data = ds[idx]
            gen_input = data[2] if len(data) > 2 else data[1]
            # DiT needs integer class labels
            if dataset_name == "imagenet":
                gen_input = gen_input  # already int from ImageNetDataset
            else:
                gen_input = data[1]  # text prompt for PixArt; DiT can't handle text
            batch_inputs.append(gen_input)
            batch_seeds.append(100000 + idx)

        # Reset accelerator state
        trajectory_id = batch_start // bs
        if args.method == "teacache" and teacache_state is not None:
            teacache_reset(teacache_state)
        if args.ttt and ttt_state is not None:
            ttt_reset_for_image(ttt_state)
        if speca_init_kwargs is not None:
            speca_cache_dic, speca_current = speca_init(
                **speca_init_kwargs,
                controller=compute_controller,
                trajectory_id=trajectory_id,
            )
            if compute_controller is not None:
                compute_controller.begin_trajectory(trajectory_id)

        # VFL (Phase 2): tag this batch's denoising trajectory with a unique
        # sample_id so curvature loss can group events from the same image.
        # We use batch_start as the id — every event recorded during the
        # upcoming generate() call (across all denoising steps and all
        # blocks) will share this id, which is exactly what the per-(sample,
        # layer) trajectory fitter needs. Set just before generate() so the
        # _denoise_loop's set_vfl_step_info calls layer on top of it.
        if vfl_buf is not None:
            set_vfl_sample_id(batch_start)

        # Generate
        t0 = time.time()
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
                covr_recorder=covr_recorder,
                covr_trajectory_id=trajectory_id,
                covr_sample_ids=[str(index) for index in batch_indices],
            )
        wall_s = time.time() - t0

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
        wall_times.append(wall_s)
        per_img_s = wall_s / actual_bs

        # Save
        img_limit = getattr(args, "img_save_limit", 50)
        for b, idx in enumerate(batch_indices):
            if global_idx < img_limit:
                # Extract class name from dataset prompt
                cls_name = ds[idx][1].replace("a photo of a ", "").replace(" ", "_")
                out_path = os.path.join(gen_dir, f"{global_idx:06d}_{cls_name}.png")
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
            elif args.method == "teacache" and teacache_state is not None:
                metrics["flops"].add_generation(
                    SimpleNamespace(decisions=teacache_state["decisions"]))
            elif args.method == "speca" and speca_cache_dic is not None:
                metrics["flops"].add_speca_generation(
                    full_steps=speca_cache_dic.full_count,
                    taylor_steps=speca_cache_dic.taylor_count,
                    probe_full_blocks=(
                        speca_cache_dic.probe_full_blocks
                        + speca_cache_dic.recompute_full_blocks),
                    num_layers=len(generator.transformer.transformer_blocks),
                )
            else:
                metrics["flops"].add_vanilla_steps(args.num_steps)

        for idx in batch_indices:
            data = ds[idx]
            all_results.append({
                "idx": idx,
                "prompt": str(data[1])[:120],
                "wall_s": wall_s,
                "images": 1,
            })

        # Phase 2: 推理循环内不再调用任何训练方法。后台线程独立轮询
        # buffer, 在数据足够时自行触发训练。这里只做 event / anchor
        # 收集 (在 _denoise_loop 内部完成), latency 完全不受训练影响。

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
        real_299_dir = ensure_real_299(ds, output_dir, n)
        metrics["fid_is"].real_dir = real_299_dir
        fid_is_results = metrics["fid_is"].compute()
        metrics["fid_is"].cleanup()  # remove temp generated_299, keep only generated/ + real_299/

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

    covr_summary = None
    if covr_recorder is not None:
        covr_summary = covr_recorder.close()
        agg["covr_shadow"] = covr_summary

    if "speed" in selected and all_results:
        unique_walls = list(dict.fromkeys(r["wall_s"] for r in all_results))
        agg["speed_img_per_s"] = float(n / np.sum(unique_walls)) if unique_walls else 0.0

    results = {
        "config": {
            "model": "dit",
            "task": "c2i",
            "dataset": dataset_name,
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
            "covr_shadow": bool(covr_recorder is not None),
            "covr_session_id": (
                covr_recorder.session_id if covr_recorder is not None else None),
            "covr_version_key": (
                covr_recorder.version.key if covr_recorder is not None else None),
            "ttt": args.ttt,
            "ttt_lr": args.ttt_lr if args.ttt else None,
            "ttt_micro_epochs": args.ttt_micro_epochs if args.ttt else None,
        },
        "aggregate": agg,
    }

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


def _clean(obj, _seen=None):
    """Recursively make dicts JSON-safe."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        return {k: _clean(v, _seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v, _seen) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item()
    return obj


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
