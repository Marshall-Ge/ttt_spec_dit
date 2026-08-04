# -*- coding: utf-8 -*-
"""PixArt-α t2i + c2i entry point — explicit sampling loop, no monkeypatching.

Orchestrator: ``PixArtGenerator`` manages VAE / scheduler / device / dtype /
T5 prompt-encoding / metrics coordination. The transformer is the new
``PixArtTransformer2D`` (explicit forward with SpecA branching).

Top-level entries:
  - ``run_t2i(args)``  (drawbench / geneval)
  - ``run_c2i(args)``  (coco / imagenet)

Sampling methods (controlled by ``args.method``):
  - baseline (DPM-Solver++ full steps)
  - teacache  (TeaCache residual reuse at loop level)
  - ddim      (DDIM step-skipping, fewer steps)
  - speca     (Speculative Taylor acceleration via cache_dic/current)
"""

import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

from config import (
    PIXART_REPO, HF_CACHE_DIR, COCO_DIR, IMAGENET_DIR, OUTPUT_DIR,
    DEFAULT_REL_L1_THRESH, DEFAULT_NUM_STEPS, load_coefficients,
)
from utils import (
    CudaTimer, decode_latent, save_image, pil_to_tensor, ensure_real_299,
    latent_seed_for_index,
)

from models.pixart import PixArtTransformer2D, set_vfl_step_info
from verification_feedback_loop.lora_adapter import (
    set_lora_t_emb, clear_lora_t_emb,
    compute_timestep_emb_for_transformer,
    attach_lora_all_layers, count_lora_params,
    _swap_lora_weights, _load_state_into_wrappers,
)
from accelerators.teacache import (
    teacache_init, teacache_decide, teacache_cache_residual,
    teacache_apply_residual, teacache_step, teacache_reset,
    teacache_stats, compute_modulated_input,
)
from accelerators.speca import SpecACache, SpecAState, speca_init

from eval.latency import LatencyMetric, FLOPsMetric


# ===========================================================================
# Valid metrics
# ===========================================================================

T2I_VALID_METRICS = {
    "drawbench": {"imagereward", "latency", "flops", "speed"},
    "geneval":   {"geneval", "latency", "flops", "speed"},
}
C2I_VALID_METRICS = {
    "coco":     {"fid", "is", "clip", "lpips", "mse", "latency", "flops", "speed"},
    "imagenet": {"fid", "is", "latency", "flops", "speed"},
}


# ===========================================================================
# PixArtGenerator — orchestrator (no monkeypatching)
# ===========================================================================

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

def run_t2i(args) -> Dict:
    """Run a PixArt t2i evaluation (drawbench / geneval)."""
    dataset_name = args.dataset
    if dataset_name not in T2I_VALID_METRICS:
        raise ValueError(
            f"Unknown t2i dataset: {dataset_name}. "
            f"Valid: {list(T2I_VALID_METRICS.keys())}")

    valid_metrics = T2I_VALID_METRICS[dataset_name]
    requested = set(args.metrics)
    for m in sorted(requested - valid_metrics):
        print(f"  [WARN] '{m}' is not valid for t2i/{dataset_name} — skipping")
    selected = sorted(requested & valid_metrics)
    if not selected:
        print(f"  [ERROR] No valid metrics remain for t2i/{dataset_name}.")
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

    dir_suffix = f"{args.method}_{args.num_steps}" if args.method == "ddim" else args.method
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"t2i_pixart_{dataset_name}_{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = load_coefficients(args.coef_path) if args.coef_path else load_coefficients()

    print("=" * 70)
    print(f"PixArt-α T2I Evaluation — {dataset_name.upper()}")
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

    # 1. Dataset
    print("\n[1] Loading dataset...")
    if dataset_name == "drawbench":
        from dataset.drawbench import DrawBenchDataset
        ds = DrawBenchDataset(n_prompts=args.n_prompts, base_seed=args.seed)
        items = [(ds[i][0], ds[i][1], None) for i in range(len(ds))]
    elif dataset_name == "geneval":
        from dataset.geneval import GenEvalDataset
        n_p = args.n_prompts if args.n_prompts else 553
        ds = GenEvalDataset(n_prompts=n_p, base_seed=args.seed)
        items = [ds[i] for i in range(len(ds))]
    n = len(items)

    # 2. Model
    print("\n[2] Loading PixArt-α model...")
    generator = PixArtGenerator(num_steps=args.num_steps, device=device, dtype=dt,
                                debug=getattr(args, "debug", False))
    generator.load()

    # ---- Debug: truncate to 1 transformer block ----
    if getattr(args, "debug", False):
        n_before = len(generator.transformer.transformer_blocks)
        generator.transformer.transformer_blocks = \
            generator.transformer.transformer_blocks[:1]
        print(f"[DEBUG MODE] truncated transformer_blocks: {n_before} -> 1")

    # ---- Print model structure (once, after all modifications) ----
    print("\n" + "=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(generator.transformer)
    total_params = sum(p.numel() for p in generator.transformer.parameters())
    trainable = sum(p.numel() for p in generator.transformer.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}  |  Trainable: {trainable:,}")
    n_blocks = len(generator.transformer.transformer_blocks)
    print(f"transformer_blocks: {n_blocks}")
    print("=" * 70)

    # ---- VFL: mid-run reload (interface-symmetric with run_dit.py) ----
    # PixArt VFL is not yet end-to-end wired (D1 known gap).  This block
    # provides the same init + swap scaffolding so that when --vfl is passed
    # for PixArt it will at least attach LoRA and attempt swaps, but the
    # data path (buffer events) is not yet connected.
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
        from models.pixart import set_vfl_buffer, set_vfl_calibrator

        vfl_cfg = VFLConfig()
        vfl_cfg.loRA_rank = 4
        vfl_cfg.time_conditioned_lora = not getattr(
            args, "vfl_no_time_lora", False)
        vfl_cfg.lambda_identity = getattr(args, "vfl_lambda_identity", 1.0)

        vfl_cal = OnlineCalibrator(ema_window=100)
        set_vfl_calibrator(vfl_cal)

        vfl_output_dir = getattr(args, "vfl_output_dir", None) or \
            os.path.join(output_dir, "vfl_checkpoints")
        vfl_no_train = getattr(args, "vfl_no_train", False)

        inference_lora_wrappers = attach_lora_all_layers(
            generator.transformer,
            rank=vfl_cfg.loRA_rank,
            alpha=1.0,
            time_conditioned=vfl_cfg.time_conditioned_lora,
        )
        n_lora_params = count_lora_params(inference_lora_wrappers)
        print(f"  Inference model LoRA attached: "
              f"{n_lora_params:,} params (B zero-init = no-op)")

        prev_ckpt = find_latest_checkpoint(vfl_output_dir)
        if prev_ckpt:
            try:
                _load_state_into_wrappers(
                    inference_lora_wrappers, prev_ckpt,
                    device=generator.device, dtype=dt)
                print(f"  Loaded LoRA from previous run: {prev_ckpt}")
            except Exception as e:
                print(f"  [WARN] failed to load LoRA checkpoint "
                      f"{prev_ckpt}: {e} — proceeding with zero-init LoRA")

        if not vfl_no_train:
            vfl_buf = StratifiedReplayBuffer(
                capacity_per_stratum=vfl_cfg.buffer_capacity_per_stratum)
            set_vfl_buffer(vfl_buf, model_version="pixart-v1")

            def on_checkpoint_ready(version):
                print(f"[VFL] LoRA v{version} ready — "
                      f"will swap at next image boundary")

            vfl_worker = AsyncTrainingWorker(
                generator.transformer, vfl_buf, config=vfl_cfg,
                base_model_version="pixart-v1", output_dir=vfl_output_dir,
                on_checkpoint_ready=on_checkpoint_ready,
            )
            vfl_worker.start()
            print(f"  VFL enabled (Phase 2 async, PixArt): "
                  f"buffer + calibrator + training thread")
        else:
            print(f"  VFL enabled (threshold-only, no training, PixArt)")

    # 3. Metrics
    print("\n[3] Setting up metrics...")
    metrics = {}
    need_imagereward = "imagereward" in selected
    need_geneval = "geneval" in selected
    need_flops = "flops" in selected
    need_latency = "latency" in selected or "speed" in selected

    if need_imagereward:
        from eval.image_reward import ImageRewardScorer
        metrics["imagereward"] = ImageRewardScorer(device=device)
    if need_geneval:
        from eval.gen_eval import GenEvalScorer
        metrics["geneval"] = GenEvalScorer(device=device)
    if need_flops:
        metrics["flops"] = FLOPsMetric(generator)
        metrics["flops"].profile()
    if need_latency:
        metrics["latency"] = LatencyMetric()

    # 4. Accelerator state
    teacache_state = None
    speca_cache_dic = None
    speca_current = None
    ddim_steps = None

    if args.method == "teacache":
        teacache_state = teacache_init(
            num_steps=args.num_steps,
            rel_l1_thresh=args.thresh,
            coefficients=coefficients,
        )
        print(f"  TeaCache ready (γ={args.thresh})")
    elif args.method == "ddim":
        ddim_steps = args.num_steps
        print(f"  DDIM sampling ({args.num_steps} steps, no caching)")
    elif args.method == "speca":
        num_blocks = len(generator.transformer.transformer_blocks)
        check_layer = 0 if getattr(args, "debug", False) else min(24, num_blocks - 1)
        speca_cache_dic, speca_current = speca_init(
            num_steps=args.num_steps,
            base_threshold=args.speca_base_threshold,
            decay_rate=args.speca_decay_rate,
            min_taylor_steps=args.speca_min_taylor_steps,
            max_taylor_steps=args.speca_max_taylor_steps,
            max_order=4,
            num_layers=len(generator.transformer.transformer_blocks),
            error_metric=args.speca_error_metric,
            check_layer=check_layer,
        )
        print(f"  SpecA ready (base_thresh={args.speca_base_threshold}, "
              f"check_layer={check_layer})")
    else:
        print("  Baseline (DPM-Solver++, no acceleration)")

    # 5. Generate
    total_images = n
    bs = args.batch_size
    print(f"\n[4] Generating {total_images} images ({args.method}, "
          f"{n} prompts in batches of ≤{bs})...")
    t_start = time.time()

    all_results = []
    for batch_start in tqdm(range(0, n, bs), desc=f"t2i/{dataset_name}", ncols=80):
        # ---- VFL mid-run reload ----
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
        batch_items = items[batch_start:batch_end]
        batch_prompts = [it[0] for it in batch_items]
        batch_seeds = [it[1] for it in batch_items]
        actual_bs = len(batch_prompts)

        # Reset accelerator state
        if args.method == "teacache" and teacache_state is not None:
            teacache_reset(teacache_state)
        elif args.method == "speca":
            speca_cache_dic, speca_current = speca_init(
                num_steps=args.num_steps,
                base_threshold=args.speca_base_threshold,
                decay_rate=args.speca_decay_rate,
                min_taylor_steps=args.speca_min_taylor_steps,
                max_taylor_steps=args.speca_max_taylor_steps,
                max_order=4,
                num_layers=len(generator.transformer.transformer_blocks),
                error_metric=args.speca_error_metric,
                check_layer=0 if getattr(args, "debug", False)
                else min(24, len(generator.transformer.transformer_blocks) - 1),
            )


        t0 = time.time()
        if args.method == "ddim":
            latent, img = generator.generate(
                batch_prompts, batch_seeds,
                guidance_scale=args.guidance_scale,
                method="ddim", ddim_steps=ddim_steps,
                teacache_state=teacache_state,
                cache_dic=speca_cache_dic, current=speca_current,
            )
        else:
            latent, img = generator.generate(
                batch_prompts, batch_seeds,
                guidance_scale=args.guidance_scale,
                method=args.method,
                teacache_state=teacache_state,
                cache_dic=speca_cache_dic, current=speca_current,
                ddim_steps=ddim_steps,
            )
        wall_s = time.time() - t0
        per_img_s = wall_s / actual_bs

        # Batch eval
        if need_imagereward:
            metrics["imagereward"].add_batch(img, prompts=batch_prompts)
        if need_geneval:
            metrics["geneval"].add_batch(img, prompts=batch_prompts)
        if need_latency:
            metrics["latency"].add_pairs_batch(
                [per_img_s] * actual_bs, [per_img_s] * actual_bs)

        for b, (prompt, seed, tag) in enumerate(batch_items):
            result = {"prompt": prompt, "seed": seed, "wall_s": wall_s,
                       "images": 1}
            if tag is not None:
                result["tag"] = tag
            all_results.append(result)

        if need_flops:
            if args.method == "teacache" and teacache_state is not None:
                metrics["flops"].add_generation(
                    SimpleNamespace(decisions=teacache_state["decisions"]))
            elif args.method == "speca" and speca_cache_dic is not None:
                full_cnt = speca_cache_dic.full_count
                taylor_cnt = args.num_steps - full_cnt
                metrics["flops"].add_generation(
                    SimpleNamespace(
                        decisions=["calc"] * full_cnt + ["skip"] * taylor_cnt))
            else:
                metrics["flops"].add_vanilla_steps(args.num_steps)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min "
          f"({elapsed/total_images:.2f} s/image)")

    # 6. Aggregate
    agg = _aggregate_results(all_results, dataset_name)

    if need_imagereward:
        agg.update(metrics["imagereward"].compute())
    if need_geneval:
        geneval_agg = metrics["geneval"].compute()
        agg["geneval_overall"] = geneval_agg.get("geneval_overall", float("nan"))
        for k, v in geneval_agg.items():
            agg[k] = v
    if need_latency:
        agg.update(metrics["latency"].compute())
    if need_flops:
        agg.update(metrics["flops"].compute())

    # Skip ratio
    if args.method == "teacache" and teacache_state is not None:
        st = teacache_stats(teacache_state)
        agg["skip_ratio"] = st.get("skip_ratio", 0.0)
        agg["total_calc"] = st.get("total_calc", 0)
        agg["total_skip"] = st.get("total_skip", 0)
    elif args.method == "speca" and speca_cache_dic is not None:
        full_cnt = speca_cache_dic.full_count
        taylor_cnt = args.num_steps - full_cnt
        total = full_cnt + taylor_cnt
        agg["skip_ratio"] = taylor_cnt / total if total > 0 else 0.0
        agg["taylor_steps"] = taylor_cnt
        agg["full_steps"] = full_cnt
        agg["total_calc"] = full_cnt
        agg["total_skip"] = taylor_cnt

    if "speed" in selected and all_results:
        unique_walls = list(dict.fromkeys(r["wall_s"] for r in all_results))
        agg["speed_img_per_s"] = float(n / np.sum(unique_walls)) if unique_walls else 0.0

    results = {
        "config": {
            "model": "pixart",
            "task": "t2i",
            "dataset": dataset_name,
            "seed": args.seed,
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
        },
        "aggregate": agg,
        "per_prompt": all_results,
    }

    # 7. Save
    print("\n[5] Saving results...")
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  Results → {results_path}")

    _print_summary(results, selected, args.method)
    return results


# ===========================================================================
# run_c2i — top-level c2i evaluation entry point for PixArt
# ===========================================================================

def run_c2i(args) -> Dict:
    """Run a PixArt c2i evaluation (coco / imagenet)."""
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

    dir_suffix = f"{args.method}_{args.num_steps}" if args.method == "ddim" else args.method
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"c2i_pixart_{dataset_name}_{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = load_coefficients(args.coef_path) if args.coef_path else load_coefficients()

    print("=" * 70)
    print(f"PixArt-α C2I Evaluation — {dataset_name.upper()}")
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

    # 1. Dataset
    print("\n[1] Loading dataset...")
    if dataset_name == "coco":
        from dataset.coco import COCO30KDataset
        ds = COCO30KDataset(
            coco_dir=getattr(args, "coco_dir", COCO_DIR),
            n_images=args.n_prompts, seed=args.seed)
    elif dataset_name == "imagenet":
        from dataset.imagenet import ImageNetDataset
        ds = ImageNetDataset(
            imagenet_dir=getattr(args, "imagenet_dir", IMAGENET_DIR),
            n_images=args.n_prompts, seed=args.seed)
    n = len(ds)

    # 2. Model
    print("\n[2] Loading PixArt-α model...")
    generator = PixArtGenerator(num_steps=args.num_steps, device=device, dtype=dt,
                                debug=getattr(args, "debug", False))
    generator.load()

    # ---- Debug: truncate to 1 transformer block ----
    if getattr(args, "debug", False):
        n_before = len(generator.transformer.transformer_blocks)
        generator.transformer.transformer_blocks = \
            generator.transformer.transformer_blocks[:1]
        print(f"[DEBUG MODE] truncated transformer_blocks: {n_before} -> 1")

    # ---- Print model structure (once, after all modifications) ----
    print("\n" + "=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(generator.transformer)
    total_params = sum(p.numel() for p in generator.transformer.parameters())
    trainable = sum(p.numel() for p in generator.transformer.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}  |  Trainable: {trainable:,}")
    n_blocks = len(generator.transformer.transformer_blocks)
    print(f"transformer_blocks: {n_blocks}")
    print("=" * 70)

    # ---- VFL: mid-run reload (interface-symmetric with run_dit.py) ----
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
        from models.pixart import set_vfl_buffer, set_vfl_calibrator

        vfl_cfg = VFLConfig()
        vfl_cfg.loRA_rank = 4
        vfl_cfg.time_conditioned_lora = not getattr(
            args, "vfl_no_time_lora", False)
        vfl_cfg.lambda_identity = getattr(args, "vfl_lambda_identity", 1.0)

        vfl_cal = OnlineCalibrator(ema_window=100)
        set_vfl_calibrator(vfl_cal)

        vfl_output_dir = getattr(args, "vfl_output_dir", None) or \
            os.path.join(output_dir, "vfl_checkpoints")
        vfl_no_train = getattr(args, "vfl_no_train", False)

        inference_lora_wrappers = attach_lora_all_layers(
            generator.transformer,
            rank=vfl_cfg.loRA_rank,
            alpha=1.0,
            time_conditioned=vfl_cfg.time_conditioned_lora,
        )
        n_lora_params = count_lora_params(inference_lora_wrappers)
        print(f"  Inference model LoRA attached: "
              f"{n_lora_params:,} params (B zero-init = no-op)")

        prev_ckpt = find_latest_checkpoint(vfl_output_dir)
        if prev_ckpt:
            try:
                _load_state_into_wrappers(
                    inference_lora_wrappers, prev_ckpt,
                    device=generator.device, dtype=dt)
                print(f"  Loaded LoRA from previous run: {prev_ckpt}")
            except Exception as e:
                print(f"  [WARN] failed to load LoRA checkpoint "
                      f"{prev_ckpt}: {e} — proceeding with zero-init LoRA")

        if not vfl_no_train:
            vfl_buf = StratifiedReplayBuffer(
                capacity_per_stratum=vfl_cfg.buffer_capacity_per_stratum)
            set_vfl_buffer(vfl_buf, model_version="pixart-v1")

            def on_checkpoint_ready(version):
                print(f"[VFL] LoRA v{version} ready — "
                      f"will swap at next image boundary")

            vfl_worker = AsyncTrainingWorker(
                generator.transformer, vfl_buf, config=vfl_cfg,
                base_model_version="pixart-v1", output_dir=vfl_output_dir,
                on_checkpoint_ready=on_checkpoint_ready,
            )
            vfl_worker.start()
            print(f"  VFL enabled (Phase 2 async, PixArt c2i): "
                  f"buffer + calibrator + training thread")
        else:
            print(f"  VFL enabled (threshold-only, no training, PixArt c2i)")

    # 3. Metrics
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
        from eval.fid_is import FIDISComputer
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
        metrics["flops"].profile()
    if need_latency:
        metrics["latency"] = LatencyMetric()

    # 4. Accelerator state
    teacache_state = None
    speca_cache_dic = None
    speca_current = None
    ddim_steps = None

    if args.method == "teacache":
        teacache_state = teacache_init(
            num_steps=args.num_steps,
            rel_l1_thresh=args.thresh,
            coefficients=coefficients,
        )
        print(f"  TeaCache ready (γ={args.thresh})")
    elif args.method == "ddim":
        ddim_steps = args.num_steps
        print(f"  DDIM sampling ({args.num_steps} steps, no caching)")
    elif args.method == "speca":
        num_blocks = len(generator.transformer.transformer_blocks)
        check_layer = 0 if getattr(args, "debug", False) else min(24, num_blocks - 1)
        speca_cache_dic, speca_current = speca_init(
            num_steps=args.num_steps,
            base_threshold=args.speca_base_threshold,
            decay_rate=args.speca_decay_rate,
            min_taylor_steps=args.speca_min_taylor_steps,
            max_taylor_steps=args.speca_max_taylor_steps,
            max_order=4,
            num_layers=len(generator.transformer.transformer_blocks),
            error_metric=args.speca_error_metric,
            check_layer=check_layer,
        )
        print(f"  SpecA ready (base_thresh={args.speca_base_threshold}, "
              f"check_layer={check_layer})")
    else:
        print("  Baseline (DPM-Solver++, no acceleration)")

    # 5. Generate
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
        # ---- VFL mid-run reload ----
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

        batch_inputs, batch_seeds = [], []
        for idx in batch_indices:
            data = ds[idx]
            if dataset_name == "imagenet":
                # ImageNet: (image_path, prompt_text, class_id)
                gen_input = data[1]  # "a photo of a XXXX" text prompt
            else:
                gen_input = data[1]  # caption text
            batch_inputs.append(gen_input)
            # Same-image replicate runs need distinct latent draws per image;
            # latent_seed_for_index keeps offset=0 bit-identical to the
            # legacy `100000 + idx` formula.
            batch_seeds.append(latent_seed_for_index(
                idx, int(getattr(args, "latent_seed_offset", 0))))

        # Reset accelerator state
        if args.method == "teacache" and teacache_state is not None:
            teacache_reset(teacache_state)
        elif args.method == "speca":
            speca_cache_dic, speca_current = speca_init(
                num_steps=args.num_steps,
                base_threshold=args.speca_base_threshold,
                decay_rate=args.speca_decay_rate,
                min_taylor_steps=args.speca_min_taylor_steps,
                max_taylor_steps=args.speca_max_taylor_steps,
                max_order=4,
                num_layers=len(generator.transformer.transformer_blocks),
                error_metric=args.speca_error_metric,
                check_layer=0 if getattr(args, "debug", False)
                else min(24, len(generator.transformer.transformer_blocks) - 1),
            )


        t0 = time.time()
        if args.method == "ddim":
            latent, img = generator.generate(
                batch_inputs, batch_seeds,
                guidance_scale=args.guidance_scale,
                method="ddim", ddim_steps=ddim_steps,
                teacache_state=teacache_state,
                cache_dic=speca_cache_dic, current=speca_current,
            )
        else:
            latent, img = generator.generate(
                batch_inputs, batch_seeds,
                guidance_scale=args.guidance_scale,
                method=args.method,
                teacache_state=teacache_state,
                cache_dic=speca_cache_dic, current=speca_current,
                ddim_steps=ddim_steps,
            )
        wall_s = time.time() - t0
        wall_times.append(wall_s)
        per_img_s = wall_s / actual_bs

        # Save
        img_limit = getattr(args, "img_save_limit", 50)
        for b, idx in enumerate(batch_indices):
            if global_idx < img_limit:
                # Truncate prompt to first 3 words, sanitize for filename
                raw = ds[idx][1]
                tag = "_".join(raw.replace(",", "").split()[:3])
                tag = "".join(c for c in tag if c.isalnum() or c == "_")[:40]
                out_path = os.path.join(gen_dir, f"{global_idx:06d}_{tag}.png")
                save_image(img[b:b+1], out_path)
            if need_fid_is:
                raw = ds[idx][1]
                tag = "_".join(raw.replace(",", "").split()[:3])
                tag = "".join(c for c in tag if c.isalnum() or c == "_")[:40]
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
            if args.method == "teacache" and teacache_state is not None:
                metrics["flops"].add_generation(
                    SimpleNamespace(decisions=teacache_state["decisions"]))
            elif args.method == "speca" and speca_cache_dic is not None:
                full_cnt = speca_cache_dic.full_count
                taylor_cnt = args.num_steps - full_cnt
                metrics["flops"].add_generation(
                    SimpleNamespace(
                        decisions=["calc"] * full_cnt + ["skip"] * taylor_cnt))
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

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min "
          f"({elapsed/total_images:.2f} s/image, {len(wall_times)} batches)")

    # 6. FID/IS
    fid_is_results = {}
    if need_fid_is:
        real_299_dir = ensure_real_299(ds, output_dir, n)
        metrics["fid_is"].real_dir = real_299_dir
        fid_is_results = metrics["fid_is"].compute()
        metrics["fid_is"].cleanup()  # remove temp generated_299, keep only generated/ + real_299/

    # 7. Aggregate
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

    if args.method == "teacache" and teacache_state is not None:
        st = teacache_stats(teacache_state)
        agg["skip_ratio"] = st.get("skip_ratio", 0.0)
        agg["total_calc"] = st.get("total_calc", 0)
        agg["total_skip"] = st.get("total_skip", 0)
    elif args.method == "speca" and speca_cache_dic is not None:
        full_cnt = speca_cache_dic.full_count
        taylor_cnt = args.num_steps - full_cnt
        total = full_cnt + taylor_cnt
        agg["skip_ratio"] = taylor_cnt / total if total > 0 else 0.0
        agg["taylor_steps"] = taylor_cnt
        agg["full_steps"] = full_cnt
        agg["total_calc"] = full_cnt
        agg["total_skip"] = taylor_cnt

    if "speed" in selected and all_results:
        unique_walls = list(dict.fromkeys(r["wall_s"] for r in all_results))
        agg["speed_img_per_s"] = float(n / np.sum(unique_walls)) if unique_walls else 0.0

    results = {
        "config": {
            "model": "pixart",
            "task": "c2i",
            "dataset": dataset_name,
            "seed": args.seed,
            "latent_seed_offset": int(getattr(args, "latent_seed_offset", 0)),
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
        },
        "aggregate": agg,
    }

    # 8. Save
    print("\n[5] Saving results...")
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  Results → {results_path}")

    _print_summary(results, selected, args.method)
    return results


# ===========================================================================
# Shared helpers
# ===========================================================================

def _aggregate_results(per_prompt: List[Dict], dataset_name: str) -> Dict:
    """Compute mean/std over per-prompt numeric fields (t2i specific)."""
    if not per_prompt:
        return {"n_prompts": 0}
    numeric_keys = ["wall_s", "imagereward", "geneval"]
    agg = {"n_prompts": len(per_prompt)}
    for k in numeric_keys:
        vals = [r[k] for r in per_prompt if k in r and r[k] is not None
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    return agg



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
