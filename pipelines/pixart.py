# -*- coding: utf-8 -*-
"""PixArt orchestrator — PixArtGenerator (VAE / scheduler / T5 encoding / loops).

Moved here from run_pixart.py by the layered-structure refactor (2026-08-27);
run_pixart.py keeps the top-level ``run_t2i`` / ``run_c2i`` evaluation entries
and imports this class. See ``.claude/project-structure.md``.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

from config import PIXART_REPO, HF_CACHE_DIR
from utils import CudaTimer, decode_latent, save_image
from models.pixart import PixArtTransformer2D, set_vfl_step_info
from feedback.vfl.lora_adapter import (
    set_lora_t_emb, clear_lora_t_emb,
    compute_timestep_emb_for_transformer,
)
from accelerators.teacache import (
    teacache_init, teacache_decide, teacache_cache_residual,
    teacache_apply_residual, teacache_step, teacache_reset,
    teacache_stats, compute_modulated_input,
)
from accelerators.speca import SpecACache, SpecAState, speca_init

class PixArtGenerator:
    """Orchestrator for PixArt-α text-to-image generation.

    Manages VAE / scheduler / device / dtype / T5 prompt-encoding /
    metrics coordination. The transformer is the new ``PixArtTransformer2D``.

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
        self._transformer: Optional[PixArtTransformer2D] = None
        self._vae = None
        self._tokenizer = None
        self._text_encoder = None
        self._scheduler = None
        self._latent_shape = None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        """Load PixArt transformer + VAE + T5 from diffusers pipeline."""
        if self._transformer is not None:
            return

        if self._debug:
            # Debug mode: tiny model (hidden_dim=96), random weights,
            # no pretrained files needed. Structure identical to full PixArt.
            self._transformer = PixArtTransformer2D(
                num_attention_heads=4,
                attention_head_dim=24,
                in_channels=4,
                out_channels=8,
                num_layers=1,
                sample_size=64,
                patch_size=2,
                cross_attention_dim=96,
                use_additional_conditions=False,
                caption_channels=4096,
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

            # T5: load tokenizer + text encoder from HuggingFace (small download).
            # If offline, fall back to a stub that produces zero embeddings.
            from transformers import T5Tokenizer, T5EncoderModel
            try:
                self._tokenizer = T5Tokenizer.from_pretrained(
                    "google-t5/t5-base", legacy=False)
                self._text_encoder = T5EncoderModel.from_pretrained(
                    "google-t5/t5-base").to(device=self.device, dtype=self._dtype)
                self._text_encoder.eval()
            except Exception:
                print("[DEBUG MODE] T5 unavailable, using stub text encoder")
                self._tokenizer = None
                self._text_encoder = None
        else:
            model, vae, tokenizer, text_encoder = PixArtTransformer2D.from_pretrained(
                PIXART_REPO, cache_dir=HF_CACHE_DIR, dtype=self._dtype)
            self._transformer = model.to(device=self.device, dtype=self._dtype)
            self._transformer.eval()
            self._vae = vae.to(device=self.device, dtype=self._dtype)
            self._vae.eval()
            self._tokenizer = tokenizer
            self._text_encoder = text_encoder.to(device=self.device, dtype=self._dtype)
            self._text_encoder.eval()

        sample_size = self._transformer.config.sample_size
        self._latent_shape = (1, 4, sample_size, sample_size)
        self._build_scheduler()
        n_blocks = len(self._transformer.transformer_blocks)
        print(f"  [PixArt] loaded. blocks={n_blocks}")

    def unload(self):
        """Free GPU memory."""
        del self._transformer, self._vae, self._text_encoder
        self._transformer = None
        self._vae = None
        self._text_encoder = None
        torch.cuda.empty_cache()

    # ---- Properties (required by FLOPsMetric / eval code) ----

    @property
    def transformer(self) -> PixArtTransformer2D:
        if self._transformer is None:
            raise RuntimeError("PixArtGenerator not loaded. Call .load() first.")
        return self._transformer

    @property
    def vae(self):
        if self._vae is None:
            raise RuntimeError("PixArtGenerator not loaded. Call .load() first.")
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

    def _build_scheduler(self):
        """Build DPM-Solver++ scheduler."""
        if self._debug:
            sched = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                prediction_type="epsilon",
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                algorithm_type="dpmsolver++",
            )
            sched.set_timesteps(self.num_steps, device=self.device)
            self._scheduler = sched
            return
        sched = DPMSolverMultistepScheduler.from_pretrained(
            PIXART_REPO, subfolder="scheduler", cache_dir=HF_CACHE_DIR,
            local_files_only=True)
        sched.set_timesteps(self.num_steps, device=self.device)
        self._scheduler = sched

    def rebuild_scheduler(self):
        self._build_scheduler()

    # ------------------------------------------------------------------
    # Prompt encoding (T5 text encoder)
    # ------------------------------------------------------------------

    def encode_prompt(self, prompts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one or more text prompts. Returns (prompt_embeds, attention_mask).

        Each prompt is independently encoded, then padded to the max sequence
        length across the batch.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Stub path when T5 is unavailable (debug mode without network)
        # Produce pre-projection embeddings (caption_channels dim);
        # caption_projection inside the transformer will project to hidden_dim.
        if self._tokenizer is None or self._text_encoder is None:
            B = len(prompts)
            cap_ch = self._transformer.config.caption_channels
            embeds = torch.zeros(B, 120, cap_ch, device=self.device, dtype=self._dtype)
            masks = torch.ones(B, 120, device=self.device, dtype=torch.long)
            return embeds, masks

        # Use the pipeline's encode_prompt method via a temporary fixture
        from diffusers import PixArtAlphaPipeline
        # We just need the tokenizer + text_encoder; use pipeline.encode_prompt logic
        all_embeds, all_masks = [], []
        for p in prompts:
            text_inputs = self._tokenizer(
                p or "", padding="max_length", max_length=300,
                truncation=True, return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            text_attention_mask = text_inputs.attention_mask.to(self.device)
            with torch.no_grad():
                prompt_embeds = self._text_encoder(
                    text_input_ids, attention_mask=text_attention_mask,
                )
                # T5EncoderModel returns last_hidden_state
                prompt_embeds = prompt_embeds[0] if isinstance(prompt_embeds, tuple) else prompt_embeds.last_hidden_state
            # Truncate/pad to max 120 tokens
            prompt_embeds = prompt_embeds[:, :120]
            text_attention_mask = text_attention_mask[:, :120]
            all_embeds.append(prompt_embeds)
            all_masks.append(text_attention_mask)

        # Pad to max sequence length
        max_len = max(e.shape[1] for e in all_embeds)
        padded_embeds, padded_masks = [], []
        for emb, mask in zip(all_embeds, all_masks):
            pad_len = max_len - emb.shape[1]
            if pad_len > 0:
                emb = F.pad(emb, (0, 0, 0, pad_len))
                mask = F.pad(mask, (0, pad_len))
            padded_embeds.append(emb)
            padded_masks.append(mask)

        return torch.cat(padded_embeds, dim=0), torch.cat(padded_masks, dim=0)

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[str]],
                 seed: Union[int, List[int]],
                 guidance_scale: float = 4.5,
                 method: str = "baseline",
                 teacache_state: Optional[dict] = None,
                 cache_dic: Optional[SpecACache] = None,
                 current: Optional[SpecAState] = None,
                 ddim_steps: Optional[int] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate image(s).

        Parameters
        ----------
        method : str
            "baseline" | "teacache" | "ddim" | "speca"
        teacache_state : dict, optional
            TeaCache state (from ``teacache_init``).
        cache_dic, current : dict, optional
            SpecA state dicts.
        ddim_steps : int, optional
            Override step count for DDIM.
        """
        self._build_scheduler()

        if isinstance(prompt, str):
            prompts = [prompt]
            seeds = [seed] if isinstance(seed, int) else seed
        else:
            prompts = prompt
            seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        B = len(prompts)

        cond_emb, cond_mask = self.encode_prompt(prompts)

        # CFG: [uncond, cond] → cond at index 1
        if guidance_scale > 1.0:
            uncond_emb, uncond_mask = self.encode_prompt([""] * B)
            max_len = max(cond_emb.shape[1], uncond_emb.shape[1])
            if cond_emb.shape[1] < max_len:
                pad = max_len - cond_emb.shape[1]
                cond_emb = F.pad(cond_emb, (0, 0, 0, pad))
                cond_mask = F.pad(cond_mask, (0, pad))
            if uncond_emb.shape[1] < max_len:
                pad = max_len - uncond_emb.shape[1]
                uncond_emb = F.pad(uncond_emb, (0, 0, 0, pad))
                uncond_mask = F.pad(uncond_mask, (0, pad))
            emb = torch.cat([uncond_emb, cond_emb], dim=0)
            mask = torch.cat([uncond_mask, cond_mask], dim=0) if cond_mask is not None else None
        else:
            emb, mask = cond_emb, cond_mask

        # Scheduler selection
        if method == "ddim" and ddim_steps is not None:
            sched = DDIMScheduler.from_config(
                self._scheduler.config, clip_sample=False)
            sched.set_timesteps(ddim_steps, device=self.device)
            num_steps = ddim_steps
        else:
            sched = self._scheduler
            num_steps = self.num_steps

        latent = self._denoise_loop(
            emb, mask, seeds, guidance_scale,
            sched, method=method,
            teacache_state=teacache_state,
            cache_dic=cache_dic, current=current,
        )

        # Unchunk: cond half (index 1 — cond follows uncond)
        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[1]

        image = decode_latent(self.vae, latent,
                              self.vae.config.scaling_factor, self._dtype)
        return latent, image

    @torch.no_grad()
    def generate_timed(self, prompt: Union[str, List[str]],
                       seed: Union[int, List[int]],
                       guidance_scale: float = 4.5,
                       method: str = "baseline",
                       teacache_state: Optional[dict] = None,
                       cache_dic: Optional[SpecACache] = None,
                       current: Optional[SpecAState] = None,
                       ddim_steps: Optional[int] = None,
                       ) -> Tuple[torch.Tensor, float]:
        """Generate with CUDA-event timing. Returns (latent, time_s)."""
        self._build_scheduler()

        if isinstance(prompt, str):
            prompts = [prompt]
            seeds = [seed] if isinstance(seed, int) else seed
        else:
            prompts = prompt
            seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        B = len(prompts)

        cond_emb, cond_mask = self.encode_prompt(prompts)
        if guidance_scale > 1.0:
            uncond_emb, uncond_mask = self.encode_prompt([""] * B)
            max_len = max(cond_emb.shape[1], uncond_emb.shape[1])
            if cond_emb.shape[1] < max_len:
                pad = max_len - cond_emb.shape[1]
                cond_emb = F.pad(cond_emb, (0, 0, 0, pad))
                cond_mask = F.pad(cond_mask, (0, pad))
            if uncond_emb.shape[1] < max_len:
                pad = max_len - uncond_emb.shape[1]
                uncond_emb = F.pad(uncond_emb, (0, 0, 0, pad))
                uncond_mask = F.pad(uncond_mask, (0, pad))
            emb = torch.cat([uncond_emb, cond_emb], dim=0)
            mask = torch.cat([uncond_mask, cond_mask], dim=0) if cond_mask is not None else None
        else:
            emb, mask = cond_emb, cond_mask

        transformer = self.transformer
        base_bs = emb.shape[0] // 2 if guidance_scale > 1.0 else emb.shape[0]

        if isinstance(seeds, list):
            assert len(seeds) == base_bs
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seeds)
        latents = self._init_latents(base_bs, generators)

        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        # DDIM scheduler?
        if method == "ddim" and ddim_steps is not None:
            sched = DDIMScheduler.from_config(
                self._scheduler.config, clip_sample=False)
            sched.set_timesteps(ddim_steps, device=self.device)
        else:
            sched = self._scheduler

        added = {"resolution": None, "aspect_ratio": None}
        timer = CudaTimer(self.device)

        for step_idx, t in enumerate(sched.timesteps):
            # VFL: track current step + real timestep for event recording hooks
            set_vfl_step_info(step_idx, len(sched.timesteps), timestep_actual=int(t))
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            # Time-conditioned LoRA: cache t_emb once per step so the 168
            # LoRALinear forwards inside the transformer all read the same
            # value without recomputing.
            _t_emb = compute_timestep_emb_for_transformer(
                transformer, current_t, hidden_dtype=latents.dtype,
            )
            if _t_emb is not None:
                set_lora_t_emb(_t_emb)
            latent_input = sched.scale_model_input(latents, t)

            if current is not None:
                current.step = len(timesteps) - 1 - step_idx
                _sched = getattr(cache_dic, "bucket_schedule", None)
                if _sched is not None:
                    _bucket = min(int(step_idx * 3 / len(timesteps)), 2)
                    cache_dic.max_taylor_steps = _sched[_bucket]

            with timer:
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=emb,
                    timestep=current_t,
                    current=current, cache_dic=cache_dic,
                    teacache_state=teacache_state,
                    encoder_attention_mask=mask,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
                if method == "teacache" and teacache_state is not None:
                    from accelerators.teacache import teacache_step
                    teacache_step(teacache_state)

            # Learned-sigma: keep noise channels, discard variance channels
            if transformer.config.out_channels // 2 == transformer.config.in_channels:
                noise_pred = noise_pred[:, :transformer.config.in_channels]
            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]

        clear_lora_t_emb()
        if guidance_scale > 1.0:
            latents = latents.chunk(2, dim=0)[1]

        return latents, timer.total_ms / 1000.0

    # ------------------------------------------------------------------
    # Denoising loop — explicit, all methods visible
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _denoise_loop(self, prompt_embeds: torch.Tensor,
                       attn_mask: Optional[torch.Tensor],
                       seed: Union[int, List[int]],
                       guidance_scale: float,
                       scheduler,
                       method: str,
                       teacache_state: Optional[dict],
                       cache_dic: Optional[SpecACache],
                       current: Optional[SpecAState],
                       ) -> torch.Tensor:
        """Single denoising loop with method dispatch."""
        transformer = self.transformer
        base_bs = prompt_embeds.shape[0] // 2 if guidance_scale > 1.0 else prompt_embeds.shape[0]

        # Init latents
        if isinstance(seed, list):
            assert len(seed) == base_bs
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seed]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seed)
        latents = self._init_latents(base_bs, generators)

        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        added = {"resolution": None, "aspect_ratio": None}
        timesteps = scheduler.timesteps

        for step_idx, t in enumerate(timesteps):
            # VFL: track current step + real timestep for event recording hooks
            set_vfl_step_info(step_idx, len(timesteps), timestep_actual=int(t))
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            # Time-conditioned LoRA: cache t_emb once per step.
            _t_emb = compute_timestep_emb_for_transformer(
                transformer, current_t, hidden_dtype=latents.dtype,
            )
            if _t_emb is not None:
                set_lora_t_emb(_t_emb)
            latent_input = scheduler.scale_model_input(latents, t)

            if current is not None:
                current.step = len(timesteps) - 1 - step_idx
                _sched = getattr(cache_dic, "bucket_schedule", None)
                if _sched is not None:
                    _bucket = min(int(step_idx * 3 / len(timesteps)), 2)
                    cache_dic.max_taylor_steps = _sched[_bucket]

            # --------------- method dispatch ---------------
            if method == "teacache" and teacache_state is not None:
                # TeaCache: pass state into model; model handles check internally
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=prompt_embeds,
                    timestep=current_t,
                    teacache_state=teacache_state,
                    encoder_attention_mask=attn_mask,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
                teacache_step(teacache_state)
            elif method == "speca":
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=prompt_embeds,
                    timestep=current_t,
                    current=current, cache_dic=cache_dic,
                    encoder_attention_mask=attn_mask,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
            else:
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=prompt_embeds,
                    timestep=current_t,
                    current=current, cache_dic=cache_dic,
                    encoder_attention_mask=attn_mask,
                    added_cond_kwargs=added, return_dict=False,
                )[0]

            # Learned-sigma: keep noise channels, discard variance channels
            if transformer.config.out_channels // 2 == transformer.config.in_channels:
                noise_pred = noise_pred[:, :transformer.config.in_channels]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        clear_lora_t_emb()
        return latents

    # ------------------------------------------------------------------
    # Latent initialisation
    # ------------------------------------------------------------------

    def _init_latents(self, base_batch: int,
                       generators: Union[torch.Generator, List[torch.Generator]],
                       ) -> torch.Tensor:
        """Create initial noise of shape (base_batch, C, H, W)."""
        if isinstance(generators, list):
            assert len(generators) == base_batch
            noises = []
            for g in generators:
                shape_one = (1,) + self._latent_shape[1:]
                noises.append(
                    torch.randn(shape_one, device=self.device, dtype=self._dtype,
                                generator=g))
            return torch.cat(noises, dim=0) * self.scheduler.init_noise_sigma
        else:
            shape = (base_batch,) + self._latent_shape[1:]
            return torch.randn(shape, device=self.device, dtype=self._dtype,
                               generator=generators) * self.scheduler.init_noise_sigma


# ===========================================================================
# run_t2i — top-level t2i evaluation entry point
# ===========================================================================

