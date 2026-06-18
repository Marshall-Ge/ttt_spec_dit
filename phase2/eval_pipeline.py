# -*- coding: utf-8 -*-
"""
Task 2.2: End-to-End Evaluation Pipeline

Bulletproof evaluation metrics to prove that our acceleration method does not
sacrifice image quality.

Metrics:
  1. Latent MSE — MSE between accelerated and vanilla final latents Z_0
  2. Pixel MSE — MSE between accelerated and vanilla decoded RGB images
  3. CLIP Score — Semantic alignment between prompt and generated image
  4. FID — Fréchet Inception Distance over a mini-dataset of prompts

All metrics are computed deterministically (seed 42) and tensors are
explicitly detached to prevent memory leaks.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from diffusers import PixArtAlphaPipeline


# ---------------------------------------------------------------------------
# VAE Decode helper
# ---------------------------------------------------------------------------

def decode_latent(vae, latents: torch.Tensor, scaling_factor: float = 0.18215,
                  dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Decode VAE latent to RGB image tensor.

    Parameters
    ----------
    vae : AutoencoderKL
        Pre-loaded VAE in eval mode.
    latents : torch.Tensor
        Shape [B, C, H, W] — final denoised latent Z_0.
    scaling_factor : float
        VAE config scaling_factor (PixArt default: 0.18215).

    Returns
    -------
    image : torch.Tensor
        Shape [B, 3, H_pix, W_pix], values in [0, 1].
    """
    latents_input = latents / scaling_factor
    if latents_input.dtype != dtype:
        latents_input = latents_input.to(dtype)
    with torch.no_grad():
        image = vae.decode(latents_input).sample
    # Clamp to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def latent_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a single [3, H, W] tensor in [0,1] to PIL Image."""
    arr = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Latent & Pixel MSE
# ---------------------------------------------------------------------------

def compute_latent_mse(latent_accel: torch.Tensor,
                       latent_vanilla: torch.Tensor) -> float:
    """Mean squared error between two final latents Z_0.

    Parameters
    ----------
    latent_accel : torch.Tensor — accelerated pipeline output
    latent_vanilla : torch.Tensor — vanilla (no-skip) output

    Returns
    -------
    mse : float
    """
    diff = (latent_accel.float().detach().cpu() -
            latent_vanilla.float().detach().cpu()) ** 2
    return diff.mean().item()


def compute_pixel_mse(image_accel: torch.Tensor,
                      image_vanilla: torch.Tensor) -> float:
    """Mean squared error between two decoded RGB images.

    Parameters
    ----------
    image_accel : torch.Tensor [1, 3, H, W] or [3, H, W] — accelerated image
    image_vanilla : torch.Tensor — vanilla image

    Returns
    -------
    mse : float
    """
    diff = (image_accel.float().detach().cpu() -
            image_vanilla.float().detach().cpu()) ** 2
    return diff.mean().item()


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------

class CLIPScorer:
    """Compute CLIP Score (semantic alignment) using openai/clip-vit-base-patch32.

    Usage:
        scorer = CLIPScorer(device="cuda")
        score = scorer.score(prompt, image_tensor)  # image_tensor: [1, 3, H, W] in [0, 1]
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None
        self._load_failed = False

    def _lazy_load(self):
        if self._model is not None:
            return
        if self._load_failed:
            return
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        try:
            self._model = CLIPModel.from_pretrained(
                model_name, cache_dir="./models", local_files_only=False,
            ).to(self.device, dtype=self.dtype).eval()
            self._processor = CLIPProcessor.from_pretrained(
                model_name, cache_dir="./models", local_files_only=False,
            )
        except Exception as e:
            print(f"  [WARN] CLIP model not available (offline?): {e}")
            print(f"  [WARN] CLIP scores will be reported as NaN.")
            self._load_failed = True

    @torch.no_grad()
    def score(self, prompt: str, image: torch.Tensor) -> float:
        """Compute CLIP cosine similarity between prompt and image.

        Parameters
        ----------
        prompt : str
        image : torch.Tensor [3, H, W] or [1, 3, H, W], values in [0, 1]

        Returns
        -------
        score : float — cosine similarity * 100 (CLIPScore convention)
        """
        self._lazy_load()
        if self._load_failed or self._model is None:
            return float("nan")

        # Prepare image for CLIP processor (PIL or tensor in [0,1])
        if image.dim() == 4:
            image = image.squeeze(0)
        pil_image = latent_to_pil(image)

        inputs = self._processor(
            text=[prompt], images=[pil_image],
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        img_emb = outputs.image_embeds  # [1, dim]
        txt_emb = outputs.text_embeds   # [1, dim]

        # Cosine similarity
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cosine = (img_emb * txt_emb).sum(dim=-1)
        return (cosine.item() * 100.0)


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------

class FIDComputer:
    """Compute FID over a mini-dataset of generated images.

    Usage:
        fid = FIDComputer(device="cuda")
        fid.update_real(real_images)   # [N, 3, H, W] uint8 [0,255]
        fid.update_fake(fake_images)   # [N, 3, H, W] uint8 [0,255]
        score = fid.compute()
    """

    def __init__(self, device: str = "cuda", feature_dim: int = 2048):
        self.device = device
        self.feature_dim = feature_dim
        self._fid = None

    def _lazy_load(self):
        if self._fid is not None:
            return
        from torchmetrics.image.fid import FrechetInceptionDistance
        self._fid = FrechetInceptionDistance(
            feature=self.feature_dim, normalize=True
        ).to(self.device)

    def update_real(self, images: torch.Tensor):
        """Add real images. images: [N, 3, H, W] uint8 [0, 255]."""
        self._lazy_load()
        self._fid.update(images.to(self.device), real=True)

    def update_fake(self, images: torch.Tensor):
        """Add generated images. images: [N, 3, H, W] uint8 [0, 255]."""
        self._lazy_load()
        self._fid.update(images.to(self.device), real=False)

    def compute(self) -> float:
        """Return FID score (lower = better)."""
        self._lazy_load()
        val = self._fid.compute()
        self._fid.reset()
        return float(val.detach().cpu())


# ---------------------------------------------------------------------------
# Multi-prompt dataset helpers
# ---------------------------------------------------------------------------

# A small set of diverse prompts for mini-benchmark evaluation.
MSCOCO_SAMPLE_PROMPTS = [
    "A majestic astronaut riding a horse on Mars, cinematic lighting, highly detailed",
    "A serene lake at sunset with mountains in the background, oil painting style",
    "A futuristic city skyline with flying cars, neon lights, cyberpunk aesthetic",
    "A cute cat wearing a wizard hat, casting spells, digital art",
    "A bowl of fresh ramen with steam rising, food photography, warm lighting",
    "An ancient castle on a cliff overlooking the ocean, watercolor painting",
    "A robot playing chess with an old man in a park, photorealistic",
    "A field of sunflowers under a blue sky, Van Gogh style",
    "A spaceship interior with holographic displays, sci-fi concept art",
    "A cozy cabin in the snowy woods with smoke coming from the chimney, winter scene",
]


def load_prompt_dataset(n_prompts: int = 10) -> List[str]:
    """Load a mini prompt dataset for evaluation."""
    return MSCOCO_SAMPLE_PROMPTS[:n_prompts]


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Orchestrates the end-to-end evaluation of an accelerated pipeline
    against the vanilla baseline.

    Parameters
    ----------
    pipe_accel : callable
        Function that takes (prompt, seed) and returns (latent_z0, image_tensor).
    pipe_vanilla : callable
        Same interface for the vanilla (no-skip) pipeline.
    device : str
    dtype : torch.dtype
    """

    def __init__(self,
                 pipe_accel,
                 pipe_vanilla,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.pipe_accel = pipe_accel
        self.pipe_vanilla = pipe_vanilla
        self.device = device
        self.dtype = dtype
        self.clip_scorer = CLIPScorer(device=device, dtype=dtype)
        self.results: List[Dict] = []

    def evaluate_prompt(self, prompt: str, seed: int = 42) -> Dict:
        """Run both pipelines for one prompt, compute all metrics.

        Returns dict with all per-prompt metrics.
        """
        # Run vanilla (full forward at every step)
        t0 = time.time()
        latent_v, image_v = self.pipe_vanilla(prompt, seed)
        t_vanilla = time.time() - t0

        # Run accelerated
        t0 = time.time()
        latent_a, image_a = self.pipe_accel(prompt, seed)
        t_accel = time.time() - t0

        # Metrics
        latent_mse = compute_latent_mse(latent_a, latent_v)
        pixel_mse = compute_pixel_mse(image_a, image_v)

        # CLIP Score (use accelerated image — measures quality, not fidelity)
        clip_accel = self.clip_scorer.score(prompt, image_a)
        clip_vanilla = self.clip_scorer.score(prompt, image_v)

        result = {
            "prompt": prompt,
            "seed": seed,
            "latency_vanilla_s": t_vanilla,
            "latency_accel_s": t_accel,
            "speedup_ratio": t_vanilla / t_accel if t_accel > 0 else float("inf"),
            "latent_mse": latent_mse,
            "pixel_mse": pixel_mse,
            "clip_score_vanilla": clip_vanilla,
            "clip_score_accel": clip_accel,
            "clip_delta": clip_accel - clip_vanilla,
        }
        self.results.append(result)
        return result

    def evaluate_batch(self, prompts: List[str], seed: int = 42) -> List[Dict]:
        """Evaluate all prompts and return aggregated results."""
        for prompt in prompts:
            self.evaluate_prompt(prompt, seed=seed)
        return self.results

    def aggregate(self) -> Dict:
        """Compute aggregate statistics across all evaluated prompts."""
        if not self.results:
            return {}
        keys = ["latency_vanilla_s", "latency_accel_s", "speedup_ratio",
                "latent_mse", "pixel_mse",
                "clip_score_vanilla", "clip_score_accel", "clip_delta"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in self.results]
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"] = np.std(vals)
        agg["n_prompts"] = len(self.results)
        return agg
