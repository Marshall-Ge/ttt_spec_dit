# -*- coding: utf-8 -*-
"""PixArt-α generator: loading, prompt encoding, vanilla generation."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler, DDIMScheduler

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

    def _build_ddim_scheduler(self, num_steps: int):
        """Build a DDIM scheduler for the same noise schedule as the default
        DPMSolver++ scheduler. ``clip_sample`` is disabled to match the
        modern PixArt pipeline (the DPM path never clips)."""
        sched = DDIMScheduler.from_config(
            self._pipe.scheduler.config,
            clip_sample=False,
        )
        sched.set_timesteps(num_steps, device=self.device)
        return sched

    def rebuild_scheduler(self):
        """Rebuild scheduler (call after changing num_steps)."""
        self._build_scheduler()

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def encode_prompt(self, prompts: Union[str, List[str]]):
        """Encode one or more text prompts. Returns (prompt_embeds, attention_mask).

        Each prompt is independently encoded with ``num_images_per_prompt=1``,
        then padded to the maximum sequence length across the batch.
        Empty string → null (unconditional) embedding.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        all_embeds = []
        all_masks = []
        for p in prompts:
            pe, am, _, _ = self._pipe.encode_prompt(
                p or "", do_classifier_free_guidance=False,
                num_images_per_prompt=1, device=self.device,
            )
            all_embeds.append(pe)     # (1, seq_i, dim)
            all_masks.append(am)      # (1, seq_i)

        # Pad to max sequence length
        max_len = max(e.shape[1] for e in all_embeds)
        padded_embeds, padded_masks = [], []
        for emb, mask in zip(all_embeds, all_masks):
            pad_len = max_len - emb.shape[1]
            if pad_len > 0:
                emb = F.pad(emb, (0, 0, 0, pad_len))     # pad seq dim
                mask = F.pad(mask, (0, pad_len))           # pad seq dim with 0
            padded_embeds.append(emb)
            padded_masks.append(mask)

        return torch.cat(padded_embeds, dim=0), torch.cat(padded_masks, dim=0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[str]],
                 seed: Union[int, List[int]],
                 guidance_scale: float = 4.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vanilla DPMSolver++ generation with CFG.

        Parameters
        ----------
        prompt : str or list[str]
            One prompt (single image) or a list of *different* prompts (batch).
        seed : int or list[int]
            One seed or one seed per prompt.  Must match prompt cardinality.
        guidance_scale : float
            CFG scale (>1.0 = enabled).  Default 4.5.

        Returns (latent, image_tensor), latent shape (B, 4, H, W),
        image in [0,1].
        """
        self._build_scheduler()

        # --- Normalise to lists ---
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
            # Pad to same max length so cat along dim=0 is legal
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

        latent = self._denoise_vanilla(emb, mask, seeds, guidance_scale=guidance_scale)

        # Unchunk: take cond half (index 1 — cond follows uncond after cat)
        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[1]

        image = decode_latent(self.vae, latent, self.vae.config.scaling_factor, self.dtype)
        return latent, image

    # ------------------------------------------------------------------
    # Internal: vanilla denoising loop
    # ------------------------------------------------------------------

    def _init_latents(self, base_batch: int,
                       generators: Union[torch.Generator, List[torch.Generator]]) -> torch.Tensor:
        """Create initial noise of shape (base_batch, C, H, W).

        When *generators* is a list, each batch element gets independent noise
        from its own :class:`torch.Generator` (different seeds).
        """
        if isinstance(generators, list):
            assert len(generators) == base_batch
            noises = []
            for g in generators:
                shape_one = (1,) + self._latent_shape[1:]
                noises.append(
                    torch.randn(shape_one, device=self.device, dtype=self.dtype,
                                generator=g))
            return torch.cat(noises, dim=0) * self.scheduler.init_noise_sigma
        else:
            shape = (base_batch,) + self._latent_shape[1:]
            return torch.randn(shape, device=self.device, dtype=self.dtype,
                              generator=generators) * self.scheduler.init_noise_sigma

    @torch.no_grad()
    def _denoise_vanilla(self, prompt_embeds, attn_mask,
                         seed: Union[int, List[int]],
                         guidance_scale: float = 1.0) -> torch.Tensor:
        """Run vanilla denoising loop, returns final latent.

        The *base* batch size is inferred from ``prompt_embeds``:
        if CFG doubles the batch, we revert to pre-doubling size.

        When *seed* is a list, each batch element gets independent noise.
        """
        transformer = self.transformer
        base_bs = prompt_embeds.shape[0] // 2 if guidance_scale > 1.0 else prompt_embeds.shape[0]

        if isinstance(seed, list):
            assert len(seed) == base_bs, f"seed list length {len(seed)} != base batch {base_bs}"
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seed]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seed)
        latents = self._init_latents(base_bs, generators)

        # CFG doubling — once before the loop; loop is completely CFG-agnostic
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        added = {"resolution": None, "aspect_ratio": None}
        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            noise_pred = transformer(
                latent_input, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attn_mask, timestep=current_t,
                added_cond_kwargs=added, return_dict=False,
            )[0]
            # Learned-sigma check (replaces hard-coded [:, :4])
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # ------------------------------------------------------------------
    # DDIM sampling (step-skipping baseline; no block caching)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_ddim(self, prompt: Union[str, List[str]],
                      seed: Union[int, List[int]],
                      num_steps: int = None,
                      guidance_scale: float = 4.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDIM deterministic sampling with CFG.

        Each step is a *full* 28-block forward (no block caching) — only the
        number of sampling steps is reduced. This is the classic
        step-compression baseline to contrast with TeaCache's block-skipping.

        Parameters
        ----------
        prompt : str or list[str]
            One prompt (single image) or a list of *different* prompts (batch).
        seed : int or list[int]
            One seed or one seed per prompt.  Must match prompt cardinality.
        num_steps : int, optional
            Override the scheduler step count. Defaults to ``self.num_steps``.
        guidance_scale : float
            CFG scale (>1.0 = enabled). Default 4.5.

        Returns (latent, image_tensor), image in [0,1].
        """
        n = num_steps if num_steps is not None else self.num_steps

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

        latent = self._denoise_ddim(emb, mask, seeds, n, guidance_scale=guidance_scale)

        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[1]

        image = decode_latent(self.vae, latent, self.vae.config.scaling_factor, self.dtype)
        return latent, image

    @torch.no_grad()
    def _denoise_ddim(self, prompt_embeds, attn_mask,
                      seed: Union[int, List[int]],
                      num_steps: int, guidance_scale: float = 1.0) -> torch.Tensor:
        """Run the DDIM denoising loop with a temporary DDIM scheduler."""
        transformer = self.transformer
        scheduler = self._build_ddim_scheduler(num_steps)
        base_bs = prompt_embeds.shape[0] // 2 if guidance_scale > 1.0 else prompt_embeds.shape[0]

        if isinstance(seed, list):
            assert len(seed) == base_bs
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seed]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seed)

        if isinstance(generators, list):
            noises = []
            for g in generators:
                shape_one = (1,) + self._latent_shape[1:]
                noises.append(torch.randn(shape_one, device=self.device, dtype=self.dtype,
                                          generator=g))
            latents = torch.cat(noises, dim=0) * scheduler.init_noise_sigma
        else:
            shape = (base_bs,) + self._latent_shape[1:]
            latents = torch.randn(shape, device=self.device, dtype=self.dtype,
                                  generator=generators) * scheduler.init_noise_sigma

        # CFG doubling
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        added = {"resolution": None, "aspect_ratio": None}
        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            noise_pred = transformer(
                latent_input, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attn_mask, timestep=current_t,
                added_cond_kwargs=added, return_dict=False,
            )[0]
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # ------------------------------------------------------------------
    # CUDA-timed generation (returns latent + model-only time)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_timed(self, prompt: Union[str, List[str]],
                       seed: Union[int, List[int]],
                       guidance_scale: float = 4.5) -> Tuple[torch.Tensor, float]:
        """Vanilla generation with CUDA-event timing. Returns (latent, time_s)."""
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

        # CFG doubling
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        added = {"resolution": None, "aspect_ratio": None}
        timer = CudaTimer(self.device)
        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            with timer:
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=emb,
                    encoder_attention_mask=mask, timestep=current_t,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Unchunk
        if guidance_scale > 1.0:
            latents = latents.chunk(2, dim=0)[1]

        return latents, timer.total_ms / 1000.0

    @torch.no_grad()
    def generate_ddim_timed(self, prompt: Union[str, List[str]],
                            seed: Union[int, List[int]],
                            num_steps: int = None,
                            guidance_scale: float = 4.5) -> Tuple[torch.Tensor, float]:
        """DDIM generation with CUDA-event timing. Returns (latent, time_s)."""
        n = num_steps if num_steps is not None else self.num_steps

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
        scheduler = self._build_ddim_scheduler(n)
        base_bs = emb.shape[0] // 2 if guidance_scale > 1.0 else emb.shape[0]

        if isinstance(seeds, list):
            assert len(seeds) == base_bs
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seeds)

        if isinstance(generators, list):
            noises = []
            for g in generators:
                shape_one = (1,) + self._latent_shape[1:]
                noises.append(torch.randn(shape_one, device=self.device, dtype=self.dtype,
                                          generator=g))
            latents = torch.cat(noises, dim=0) * scheduler.init_noise_sigma
        else:
            shape = (base_bs,) + self._latent_shape[1:]
            latents = torch.randn(shape, device=self.device, dtype=self.dtype,
                                  generator=generators) * scheduler.init_noise_sigma

        # CFG doubling
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        added = {"resolution": None, "aspect_ratio": None}
        timer = CudaTimer(self.device)
        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            with timer:
                noise_pred = transformer(
                    latent_input, encoder_hidden_states=emb,
                    encoder_attention_mask=mask, timestep=current_t,
                    added_cond_kwargs=added, return_dict=False,
                )[0]
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Unchunk
        if guidance_scale > 1.0:
            latents = latents.chunk(2, dim=0)[1]

        return latents, timer.total_ms / 1000.0
