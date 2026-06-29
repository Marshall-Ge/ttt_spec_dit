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
from utils import CudaTimer, decode_latent, save_image, pil_to_tensor, ensure_real_299

from models.dit import DiTTransformer2D
from accelerators.teacache import (
    teacache_init, teacache_decide, teacache_cache_residual,
    teacache_apply_residual, teacache_step, teacache_reset,
    teacache_stats, compute_modulated_input_dit,
)
from accelerators.speca import speca_init
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
                 dtype: torch.dtype = torch.float16):
        self.num_steps = num_steps
        self.device = device
        self._dtype = dtype
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
        if os.path.exists(model_index_path):
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
                 cache_dic: Optional[dict] = None,
                 current: Optional[dict] = None,
                 ddim_steps: Optional[int] = None,
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
                       cache_dic: Optional[dict],
                       current: Optional[dict],
                       ) -> torch.Tensor:
        """Single denoising loop with method dispatch.

        TeaCache logic lives here (loop-level), NOT inside the model.
        SpecA logic lives inside the model (via current/cache_dic).
        """
        transformer = self.transformer
        base_bs = class_labels.shape[0] // 2 if guidance_scale > 1.0 else class_labels.shape[0]

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
            latent_input = scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)

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
                    current['step'] = len(timesteps) - 1 - step_idx
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
            if transformer.config.out_channels // 2 == transformer.config.in_channels:
                noise_pred = noise_pred[:, :transformer.config.in_channels]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

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
            latent_input = scheduler.scale_model_input(latents, t)
            current_t = t.expand(latents.shape[0]).to(torch.int64)

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
    generator = DiTGenerator(num_steps=args.num_steps, device=device, dtype=dt)
    generator.load()

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
        check_layer = 20  # DiT gate_mlp U-shape blind spot
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
        print("  Baseline (full DDIM, no acceleration)")

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
        if args.method == "teacache" and teacache_state is not None:
            teacache_reset(teacache_state)
        if args.ttt and ttt_state is not None:
            ttt_reset_for_image(ttt_state)
        if args.method == "speca":
            speca_cache_dic, speca_current = speca_init(
                num_steps=args.num_steps,
                base_threshold=args.speca_base_threshold,
                decay_rate=args.speca_decay_rate,
                min_taylor_steps=args.speca_min_taylor_steps,
                max_taylor_steps=args.speca_max_taylor_steps,
                max_order=4,
                num_layers=len(generator.transformer.transformer_blocks),
                error_metric=args.speca_error_metric,
                check_layer=20,
            )

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
            )
        wall_s = time.time() - t0
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
            elif args.method == "speca":
                full_cnt = speca_cache_dic.get('full_count', 0)
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
    elif args.method == "speca" and speca_cache_dic is not None:
        full_cnt = speca_cache_dic.get('full_count', 0)
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
