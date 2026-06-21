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
    """[3,H,W] or [B,3,H,W] in [0,1] → PIL Image. Batched → first image."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
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
    """Save [3,H,W] or [B,3,H,W] float [0,1] tensor as PNG. Batched → saves first image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_real_image(path: str, size: int = 299) -> torch.Tensor:
    """Load a real image JPEG/PNG → [3, size, size] tensor in [0,1]."""
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((size, size), Image.BICUBIC)
    return pil_to_tensor(pil)


# ---------------------------------------------------------------------------
# FID real-image preprocessing (one-time, shared across runs)
# ---------------------------------------------------------------------------

def ensure_real_299(ds, output_dir: str, n: int) -> str:
    """Ensure real images at 299×299 exist for FID.

    Pre-processes all dataset images to 299×299 once into a flat directory,
    then creates a lightweight subset via symlinks for the current run's
    shuffle order and sample count.

    Returns path to the subset directory (symlinks into the pre-processed cache).
    """
    import os as _os
    from tqdm import tqdm as _tqdm

    # Determine cache path from dataset root and set name
    val_dir = getattr(ds, 'val_dir', None)
    if val_dir is None:
        val_dir = _os.path.dirname(ds[0][0]) if len(ds) > 0 else "/tmp"
    cache_dir = _os.path.join(_os.path.dirname(val_dir) or val_dir, "val_299_cache")
    _os.makedirs(cache_dir, exist_ok=True)

    # Pre-process all dataset images once (including class name in filename)
    existing = set(_os.listdir(cache_dir))
    total_items = len(ds.items) if hasattr(ds, 'items') else len(ds)
    need_preprocess = sum(1 for i in range(total_items)
                          if not any(f.startswith(f"{i:06d}_") for f in existing))

    if need_preprocess > 0:
        print(f"  [FID] Pre-processing {need_preprocess} real images to 299×299 "
              f"(one-time, cached in {cache_dir})...")
        for idx in _tqdm(range(total_items), desc="preprocess real 299", ncols=80):
            # Get class name from dataset prompt
            if hasattr(ds, '__getitem__'):
                _, prompt, _ = ds[idx]
                cls_name = prompt.replace("a photo of a ", "").replace(" ", "_")
            else:
                cls_name = "unknown"
            fname = f"{idx:06d}_{cls_name}.png"
            out_path = _os.path.join(cache_dir, fname)
            if _os.path.exists(out_path):
                continue
            img_path = ds[idx][0] if hasattr(ds, '__getitem__') else ds.items[idx][0]
            if not _os.path.exists(img_path):
                continue
            pil_img = Image.open(img_path).convert("RGB")
            pil_img = pil_img.resize((299, 299), Image.BICUBIC)
            pil_img.save(out_path)

    # Create run-specific subset via symlinks (match cache filename format)
    subset_dir = _os.path.join(output_dir, "real_299")
    _os.makedirs(subset_dir, exist_ok=True)

    # Clean previous symlinks
    for f in _os.listdir(subset_dir):
        p = _os.path.join(subset_dir, f)
        if _os.path.islink(p) or f.endswith('.png'):
            _os.remove(p)

    for idx in range(n):
        _, prompt, _ = ds[idx] if hasattr(ds, '__getitem__') else (None, "unknown", None)
        cls_name = prompt.replace("a photo of a ", "").replace(" ", "_")
        fname = f"{idx:06d}_{cls_name}.png"
        src = _os.path.join(cache_dir, fname)
        dst = _os.path.join(subset_dir, fname)
        if _os.path.exists(src):
            _os.symlink(src, dst)

    print(f"  [FID] real_299 ready: {n} symlinks → {cache_dir}")
    return subset_dir
