# -*- coding: utf-8 -*-
"""DiT-2-256 generator: loading, class-label encoding, vanilla generation."""

import os
import torch
from typing import List, Tuple, Union

from diffusers import DiTPipeline, DDIMScheduler

from config import DIT_REPO, HF_CACHE_DIR
from utils import CudaTimer, decode_latent
from .base import DiffusionGenerator


class DiTGenerator(DiffusionGenerator):
    """DiT-2-256 class-conditional image generator.

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
        self._id2label = None
        self.null_class = 1000  # overridden in load() from model config

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        """Load DiTPipeline from local model directory."""
        if self._pipe is not None:
            return
        self._pipe = DiTPipeline.from_pretrained(
            DIT_REPO, cache_dir=HF_CACHE_DIR, torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self._pipe.transformer.eval()
        self._pipe.vae.eval()
        sample_size = self._pipe.transformer.config.sample_size
        self._latent_shape = (1, 4, sample_size, sample_size)
        self.null_class = self._pipe.transformer.config.num_embeds_ada_norm  # typically 1000
        # Load id2label from model_index.json (not exposed as pipe attribute)
        import json
        model_index_path = os.path.join(DIT_REPO, "model_index.json")
        if os.path.exists(model_index_path):
            with open(model_index_path) as f:
                self._id2label = json.load(f).get("id2label", {})
        else:
            self._id2label = {}
        self._build_scheduler()
        print(f"  [DiT] loaded. blocks={len(self._pipe.transformer.transformer_blocks)}")

    def unload(self):
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    @property
    def transformer(self):
        if self._pipe is None:
            raise RuntimeError("DiTGenerator not loaded. Call .load() first.")
        return self._pipe.transformer

    @property
    def vae(self):
        if self._pipe is None:
            raise RuntimeError("DiTGenerator not loaded. Call .load() first.")
        return self._pipe.vae

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def latent_shape(self):
        return self._latent_shape

    @property
    def id2label(self):
        return self._id2label

    def _build_scheduler(self):
        """Build DDIM scheduler from config."""
        sched = DDIMScheduler.from_config(
            self._pipe.scheduler.config)
        sched.set_timesteps(self.num_steps, device=self.device)
        self._scheduler = sched

    def rebuild_scheduler(self):
        """Rebuild scheduler (call after changing num_steps)."""
        self._build_scheduler()

    # ------------------------------------------------------------------
    # Prompt encoding (class label → tensor)
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt):
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

    def _encode_single(self, prompt):
        """Convert a single prompt to a class-label tensor of shape (1,)."""
        if isinstance(prompt, int):
            return torch.tensor([prompt], device=self.device, dtype=torch.long)

        # Try parsing as integer string
        try:
            return torch.tensor([int(prompt)], device=self.device, dtype=torch.long)
        except (ValueError, TypeError):
            pass

        # Reverse lookup by class name
        if self._id2label is not None:
            for idx, name in self._id2label.items():
                if prompt.lower() in name.lower():
                    return torch.tensor([int(idx)], device=self.device, dtype=torch.long)

        raise ValueError(
            f"Could not convert prompt '{prompt}' to a class label. "
            f"Provide an int (0–999) or a valid ImageNet class name.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt, seed: Union[int, List[int]],
                 guidance_scale: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vanilla DDIM generation with CFG.

        Parameters
        ----------
        prompt : int, str, or list of int/str
            Single class label or list of *different* class labels (batch).
        seed : int or list[int]
            One seed or one seed per prompt.  Must match prompt cardinality.
        guidance_scale : float
            CFG scale (>1.0 = enabled).  Default 4.0.

        Returns (latent, image_tensor), latent shape (B, 4, H, W),
        image in [0,1].

        The transformer.forward may be monkeypatched by an accelerator;
        this method is agnostic to the actual forward implementation.
        """
        self._build_scheduler()

        # --- Normalise ---
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
            null_labels = torch.full((B,), self.null_class, device=self.device, dtype=torch.long)
            class_labels = torch.cat([cond_labels, null_labels], dim=0)
        else:
            class_labels = cond_labels

        latent = self._denoise_vanilla(class_labels, seeds, guidance_scale=guidance_scale)

        # Unchunk: cond half (index 0 — cond comes first in cat)
        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[0]

        # DiT uses standard SD VAE with scaling_factor=0.18215
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = decode_latent(self.vae, latent, scaling_factor, self.dtype)
        return latent, image

    @torch.no_grad()
    def generate_timed(self, prompt, seed: Union[int, List[int]],
                       guidance_scale: float = 4.0) -> Tuple[torch.Tensor, float]:
        """Vanilla generation with CUDA-event timing. Returns (latent, time_s)."""
        self._build_scheduler()

        if isinstance(prompt, list):
            prompts = prompt
            seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        else:
            prompts = [prompt]
            seeds = [seed] if isinstance(seed, int) else seed
        B = len(prompts)

        cond_labels = self.encode_prompt(prompts)  # (B,)

        if guidance_scale > 1.0:
            null_labels = torch.full((B,), self.null_class, device=self.device, dtype=torch.long)
            class_labels = torch.cat([cond_labels, null_labels], dim=0)
        else:
            class_labels = cond_labels

        transformer = self.transformer
        base_bs = class_labels.shape[0] // 2 if guidance_scale > 1.0 else class_labels.shape[0]

        if isinstance(seeds, list):
            assert len(seeds) == base_bs
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seeds)
        latents = self._init_latents(base_bs, generators)

        # CFG doubling
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        timer = CudaTimer(self.device)
        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            with timer:
                noise_pred = transformer(
                    latent_input, timestep=current_t,
                    class_labels=class_labels, return_dict=False,
                )[0]
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Unchunk
        if guidance_scale > 1.0:
            latents = latents.chunk(2, dim=0)[0]

        return latents, timer.total_ms / 1000.0

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
    def _denoise_vanilla(self, class_labels: torch.Tensor,
                         seed: Union[int, List[int]],
                         guidance_scale: float = 1.0) -> torch.Tensor:
        """Run vanilla denoising loop, returns final latent.

        The *base* batch size is inferred from ``class_labels``:
        if CFG doubles the batch, we revert to pre-doubling size.

        When *seed* is a list, each batch element gets independent noise.
        """
        transformer = self.transformer
        base_bs = class_labels.shape[0] // 2 if guidance_scale > 1.0 else class_labels.shape[0]

        if isinstance(seed, list):
            assert len(seed) == base_bs, f"seed list length {len(seed)} != base batch {base_bs}"
            generators = [torch.Generator(device=self.device).manual_seed(s) for s in seed]
        else:
            generators = torch.Generator(device=self.device).manual_seed(seed)
        latents = self._init_latents(base_bs, generators)

        # CFG doubling — once before the loop; loop is completely CFG-agnostic
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)

        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            noise_pred = transformer(
                latent_input, timestep=current_t,
                class_labels=class_labels, return_dict=False,
            )[0]
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # ------------------------------------------------------------------
    # DDIM sampling (reduced steps; no block caching)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_ddim(self, prompt, seed: Union[int, List[int]],
                      num_steps: int = None,
                      guidance_scale: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDIM deterministic sampling with reduced steps and CFG.

        Each step is a full 28-block forward (no block caching) — only the
        number of sampling steps is reduced.

        Parameters
        ----------
        prompt : int, str, or list of int/str
            Single class label or list of *different* class labels (batch).
        seed : int or list[int]
            One seed or one seed per prompt.  Must match prompt cardinality.
        num_steps : int, optional
            Override the scheduler step count. Defaults to ``self.num_steps``.
        guidance_scale : float
            CFG scale (>1.0 = enabled). Default 4.0 for DiT.

        Returns (latent, image_tensor), image in [0,1].
        """
        n = num_steps if num_steps is not None else self.num_steps

        if isinstance(prompt, list):
            prompts = prompt
            seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        else:
            prompts = [prompt]
            seeds = [seed] if isinstance(seed, int) else seed
        B = len(prompts)

        cond_labels = self.encode_prompt(prompts)  # (B,)

        if guidance_scale > 1.0:
            null_labels = torch.full((B,), self.null_class, device=self.device, dtype=torch.long)
            class_labels = torch.cat([cond_labels, null_labels], dim=0)
        else:
            class_labels = cond_labels

        latent = self._denoise_ddim(class_labels, seeds, n, guidance_scale=guidance_scale)

        if guidance_scale > 1.0:
            latent = latent.chunk(2, dim=0)[0]

        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = decode_latent(self.vae, latent, scaling_factor, self.dtype)
        return latent, image

    @torch.no_grad()
    def _denoise_ddim(self, class_labels: torch.Tensor,
                      seed: Union[int, List[int]],
                      num_steps: int, guidance_scale: float = 1.0) -> torch.Tensor:
        """Run the DDIM denoising loop with a fresh scheduler."""
        transformer = self.transformer
        scheduler = DDIMScheduler.from_config(
            self._pipe.scheduler.config, clip_sample=False,
        )
        scheduler.set_timesteps(num_steps, device=self.device)
        base_bs = class_labels.shape[0] // 2 if guidance_scale > 1.0 else class_labels.shape[0]

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

        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)
            noise_pred = transformer(
                latent_input, timestep=current_t,
                class_labels=class_labels, return_dict=False,
            )[0]
            if transformer.config.out_channels // 2 == self._latent_shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents
