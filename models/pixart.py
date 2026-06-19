# -*- coding: utf-8 -*-
"""PixArt-α generator: loading, prompt encoding, vanilla generation."""

import torch
from typing import Tuple

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

from config import PIXART_REPO, HF_CACHE_DIR
from utils import CudaTimer, decode_latent
from .base import DiffusionGenerator


class PixArtGenerator(DiffusionGenerator):
    """PixArt-α text-to-image generator.

    Parameters
    ----------
    num_steps : int
        Number of denoising steps.
    device : str
    dtype : torch.dtype
    """

    def __init__(self, num_steps: int = 20, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_steps = num_steps
        self.device = device
        self.dtype = dtype
        self._pipe = None
        self._scheduler = None
        self._latent_shape = None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        """Load PixArtAlphaPipeline from local HF cache."""
        if self._pipe is not None:
            return
        self._pipe = PixArtAlphaPipeline.from_pretrained(
            PIXART_REPO, cache_dir=HF_CACHE_DIR, torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self._pipe.transformer.eval()
        self._pipe.vae.eval()
        sample_size = self._pipe.transformer.config.sample_size
        self._latent_shape = (1, 4, sample_size, sample_size)
        self._build_scheduler()
        print(f"  [PixArt] loaded. blocks={len(self._pipe.transformer.transformer_blocks)}")

    def unload(self):
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    @property
    def transformer(self):
        if self._pipe is None:
            raise RuntimeError("PixArtGenerator not loaded. Call .load() first.")
        return self._pipe.transformer

    @property
    def vae(self):
        if self._pipe is None:
            raise RuntimeError("PixArtGenerator not loaded. Call .load() first.")
        return self._pipe.vae

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def latent_shape(self):
        return self._latent_shape

    def _build_scheduler(self):
        sched = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config)
        sched.set_timesteps(self.num_steps, device=self.device)
        self._scheduler = sched

    def rebuild_scheduler(self):
        """Rebuild scheduler (call after changing num_steps)."""
        self._build_scheduler()

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str):
        """Encode a text prompt. Returns (prompt_embeds, attention_mask)."""
        pe, am, _, _ = self._pipe.encode_prompt(
            prompt, do_classifier_free_guidance=False,
            num_images_per_prompt=1, device=self.device,
        )
        return pe, am

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: str, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vanilla DPMSolver++ generation.

        Returns (latent, image_tensor), image in [0,1].
        """
        self._build_scheduler()
        prompt_embeds, attn_mask = self.encode_prompt(prompt)
        latent = self._denoise_vanilla(prompt_embeds, attn_mask, seed)
        image = decode_latent(self.vae, latent, self.vae.config.scaling_factor, self.dtype)
        return latent, image

    @torch.no_grad()
    def generate_teacache(self, prompt: str, seed: int, teacache) -> Tuple[torch.Tensor, torch.Tensor]:
        """TeaCache-accelerated generation.

        Parameters
        ----------
        teacache : PixArtTeaCache
            Configured TeaCache controller.

        Returns (latent, image_tensor), image in [0,1].
        """
        from .teacache import install_teacache, uninstall_teacache

        self._build_scheduler()
        prompt_embeds, attn_mask = self.encode_prompt(prompt)
        transformer = self.transformer

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(self._latent_shape, device=self.device, dtype=self.dtype,
                              generator=generator) * self.scheduler.init_noise_sigma
        added = {"resolution": None, "aspect_ratio": None}

        original_forward = install_teacache(transformer, teacache)
        try:
            for t in self.scheduler.timesteps:
                latent_input = self.scheduler.scale_model_input(latents, t)
                current_t = t.expand(1).to(torch.int64)
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attn_mask, timestep=current_t,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
                latents = self.scheduler.step(noise_pred[:, :4], t, latents, return_dict=False)[0]
        finally:
            uninstall_teacache(transformer, original_forward)

        image = decode_latent(self.vae, latents, self.vae.config.scaling_factor, self.dtype)
        return latents, image

    # ------------------------------------------------------------------
    # Internal: vanilla denoising loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _denoise_vanilla(self, prompt_embeds, attn_mask, seed: int) -> torch.Tensor:
        """Run vanilla denoising loop, returns final latent."""
        transformer = self.transformer
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(self._latent_shape, device=self.device, dtype=self.dtype,
                              generator=generator) * self.scheduler.init_noise_sigma
        added = {"resolution": None, "aspect_ratio": None}
        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            current_t = t.expand(1).to(torch.int64)
            noise_pred = transformer(
                latent_input, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attn_mask, timestep=current_t,
                added_cond_kwargs=added, return_dict=False,
            )[0]
            latents = self.scheduler.step(noise_pred[:, :4], t, latents, return_dict=False)[0]
        return latents

    # ------------------------------------------------------------------
    # CUDA-timed generation (returns latent + model-only time)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_timed(self, prompt: str, seed: int) -> Tuple[torch.Tensor, float]:
        """Vanilla generation with CUDA-event timing. Returns (latent, time_s)."""
        self._build_scheduler()
        prompt_embeds, attn_mask = self.encode_prompt(prompt)
        transformer = self.transformer
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(self._latent_shape, device=self.device, dtype=self.dtype,
                              generator=generator) * self.scheduler.init_noise_sigma
        added = {"resolution": None, "aspect_ratio": None}
        timer = CudaTimer(self.device)
        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            current_t = t.expand(1).to(torch.int64)
            with timer:
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attn_mask, timestep=current_t,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
            latents = self.scheduler.step(noise_pred[:, :4], t, latents, return_dict=False)[0]
        return latents, timer.total_ms / 1000.0

    @torch.no_grad()
    def generate_teacache_timed(self, prompt: str, seed: int, teacache) -> Tuple[torch.Tensor, float]:
        """TeaCache generation with CUDA-event timing. Returns (latent, time_s)."""
        from .teacache import install_teacache, uninstall_teacache

        self._build_scheduler()
        prompt_embeds, attn_mask = self.encode_prompt(prompt)
        transformer = self.transformer
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(self._latent_shape, device=self.device, dtype=self.dtype,
                              generator=generator) * self.scheduler.init_noise_sigma
        added = {"resolution": None, "aspect_ratio": None}
        timer = CudaTimer(self.device)

        original_forward = install_teacache(transformer, teacache)
        try:
            for t in self.scheduler.timesteps:
                latent_input = self.scheduler.scale_model_input(latents, t)
                current_t = t.expand(1).to(torch.int64)
                with timer:
                    noise_pred = transformer(
                        latent_input, encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=attn_mask, timestep=current_t,
                        added_cond_kwargs=added, return_dict=False,
                    )[0]
                latents = self.scheduler.step(noise_pred[:, :4], t, latents, return_dict=False)[0]
        finally:
            uninstall_teacache(transformer, original_forward)

        return latents, timer.total_ms / 1000.0
