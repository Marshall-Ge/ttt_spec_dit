# -*- coding: utf-8 -*-
"""CUDA timer, VAE decode, tensor↔PIL, image I/O helpers."""

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# CUDA-event timer
# ---------------------------------------------------------------------------

class CudaTimer:
    """Accurate GPU-side timer using CUDA events."""

    def __init__(self, device="cuda"):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0.0

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *a):
        self.end.record()
        torch.cuda.synchronize()
        self.total_ms += self.start.elapsed_time(self.end)


# ---------------------------------------------------------------------------
# VAE decode
# ---------------------------------------------------------------------------

def decode_latent(vae, latents: torch.Tensor, scaling_factor: float = 0.18215,
                  dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Decode VAE latent [B, C, H, W] → RGB image tensor [B, 3, Hp, Wp] in [0,1]."""
    latents_input = latents / scaling_factor
    if latents_input.dtype != dtype:
        latents_input = latents_input.to(dtype)
    with torch.no_grad():
        image = vae.decode(latents_input).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


# ---------------------------------------------------------------------------
# Tensor ↔ PIL
# ---------------------------------------------------------------------------

def latent_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """[3, H, W] (or [1,3,H,W]) in [0,1] → PIL Image."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    arr = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """PIL Image → [3, H, W] float tensor in [0,1]."""
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image(tensor: torch.Tensor, path: str):
    """Save [3,H,W] or [1,3,H,W] float [0,1] tensor as PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_real_image(path: str, size: int = 299) -> torch.Tensor:
    """Load a real image JPEG/PNG → [3, size, size] tensor in [0,1]."""
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((size, size), Image.BICUBIC)
    return pil_to_tensor(pil)
