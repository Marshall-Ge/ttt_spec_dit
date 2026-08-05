# -*- coding: utf-8 -*-
"""CUDA timer, VAE decode, tensor↔PIL, image I/O helpers."""

import hashlib
import numpy as np
import re
import time
import torch
from PIL import Image
from typing import Tuple

# Old, seed-dependent cache naming: "<6-digit dataset index>_<class name>.png".
_re_old_cache_name = re.compile(r"^\d{6}_.*\.png$")


# ---------------------------------------------------------------------------
# CUDA-event timer
# ---------------------------------------------------------------------------

class CudaTimer:
    """Accurate GPU-side timer using CUDA events, with MPS/CPU fallback."""

    def __init__(self, device="cuda"):
        self._device = device
        self.total_ms = 0.0
        self._use_cuda_events = (device == "cuda" and torch.cuda.is_available())
        if self._use_cuda_events:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = 0.0

    def __enter__(self):
        if self._use_cuda_events:
            self.start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self._use_cuda_events:
            self.end.record()
            torch.cuda.synchronize()
            self.total_ms += self.start.elapsed_time(self.end)
        else:
            elapsed = (time.perf_counter() - self._t0) * 1000.0
            self.total_ms += elapsed


# ---------------------------------------------------------------------------
# Per-image latent seeds
# ---------------------------------------------------------------------------

# Stride between latent draws of the same image. absolute_idx is bounded by
# 50,000 (ImageNet val size; COCO/drawbench/geneval are smaller), so 1_000_000
# leaves the per-offset ranges disjoint and the seeds stay far below
# torch.Generator's int64 range even for large offsets.
_LATENT_SEED_BASE = 100000
_LATENT_SEED_STRIDE = 1_000_000


def latent_seed_for_index(absolute_idx: int, latent_seed_offset: int = 0) -> int:
    """Deterministic per-image latent seed for (image, latent draw) cells.

    ``--seed`` selects which image ``absolute_idx`` names (dataset shuffle);
    this selects which independent latent draw a run uses for that image, so
    two runs over the SAME images can pair per-image metrics across latent
    draws. Distinct offsets give disjoint, reproducible seed sets for the same
    image indices.

    INVARIANT: offset=0 must stay bit-identical to the legacy formula
    ``100000 + absolute_idx`` — every existing run and cached comparison was
    produced with it.
    """
    if latent_seed_offset < 0:
        raise ValueError("latent_seed_offset must be non-negative")
    return _LATENT_SEED_BASE + latent_seed_offset * _LATENT_SEED_STRIDE + absolute_idx


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
    """Save a [3,H,W] or [B,3,H,W] float [0,1] tensor by path extension."""
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

def _real_299_cache_name(img_path: str) -> str:
    """Cache filename for one source image, keyed by its PATH.

    Keying by dataset index is WRONG and silently corrupts FID: dataset order
    depends on ``--seed`` (``ImageNetDataset`` does ``RandomState(seed).shuffle``),
    so index 000000 names a different source image at every seed. A cache built
    at one seed then answers "already preprocessed" for another seed while
    holding none of that seed's files. Path keying makes an entry mean the same
    image for every seed, so the cache is shared instead of mutually
    invalidating.

    The stem keeps the source basename for human inspection; the hash makes it
    collision-free across class subdirectories that reuse filenames.
    """
    digest = hashlib.sha1(img_path.encode("utf-8")).hexdigest()[:16]
    stem = _os_basename_stem(img_path)
    return f"{stem}_{digest}.png"


def _os_basename_stem(path: str) -> str:
    import os as _os
    base = _os.path.basename(path)
    stem = _os.path.splitext(base)[0]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:64]


def _real_299_items(ds, start_index: int, n: int) -> Tuple[list, int]:
    """Return (all source paths, total_items) with the slice bounds checked."""
    total_items = len(ds.items) if hasattr(ds, "items") else len(ds)
    if start_index < 0 or n < 0 or start_index + n > total_items:
        raise ValueError("real-image slice exceeds the loaded dataset prefix")
    paths = []
    for idx in range(total_items):
        item = ds.items[idx] if hasattr(ds, "items") else ds[idx]
        paths.append(item[0])
    return paths, total_items


def ensure_real_299(ds, output_dir: str, n: int, start_index: int = 0) -> str:
    """Ensure a deterministic real-image slice at 299×299 exists for FID.

    Pre-processes every dataset image to 299×299 once into a shared cache keyed
    by SOURCE PATH (see ``_real_299_cache_name``), then links the requested
    absolute slice ``[start_index, start_index + n)`` of the dataset's
    deterministic order into ``<output_dir>/real_299``.

    Raises RuntimeError if fewer than ``n`` images could be linked. A short
    real set does not fail loudly inside torch-fidelity — 1 image yields a
    degenerate covariance ("Array must not contain infs or NaNs" -> FID NaN)
    and 2 images yield a meaningless finite number, either of which quietly
    destroys a whole sweep. Fail here instead.

    Returns path to the subset directory (symlinks into the shared cache).
    """
    import os as _os
    from tqdm import tqdm as _tqdm

    # Determine cache path from dataset root and set name
    val_dir = getattr(ds, 'val_dir', None)
    if val_dir is None:
        val_dir = _os.path.dirname(ds[0][0]) if len(ds) > 0 else "/tmp"
    cache_dir = _os.path.join(_os.path.dirname(val_dir) or val_dir, "val_299_cache")
    _os.makedirs(cache_dir, exist_ok=True)

    all_paths, total_items = _real_299_items(ds, start_index, n)
    cached = set(_os.listdir(cache_dir))
    stale = sum(1 for f in cached if _re_old_cache_name.match(f))
    if stale:
        print(f"  [FID] note: {stale} cache entries use the old index-keyed "
              f"naming and are now unused (safe to delete: "
              f"find {cache_dir} -name '[0-9][0-9][0-9][0-9][0-9][0-9]_*.png' "
              f"-delete)")
    todo = [(idx, p) for idx, p in enumerate(all_paths)
            if _real_299_cache_name(p) not in cached]

    if todo:
        print(f"  [FID] Pre-processing {len(todo)} real images to 299×299 "
              f"(one-time, cached in {cache_dir})...")
        for _, img_path in _tqdm(todo, desc="preprocess real 299", ncols=80):
            if not _os.path.exists(img_path):
                continue
            out_path = _os.path.join(cache_dir, _real_299_cache_name(img_path))
            if _os.path.exists(out_path):
                continue
            pil_img = Image.open(img_path).convert("RGB")
            pil_img = pil_img.resize((299, 299), Image.BICUBIC)
            pil_img.save(out_path)

    # Create run-specific subset via symlinks
    subset_dir = _os.path.join(output_dir, "real_299")
    _os.makedirs(subset_dir, exist_ok=True)

    # Clean previous symlinks
    for f in _os.listdir(subset_dir):
        p = _os.path.join(subset_dir, f)
        if _os.path.islink(p) or f.endswith('.png'):
            _os.remove(p)

    linked = 0
    missing = []
    for idx in range(start_index, start_index + n):
        img_path = all_paths[idx]
        src = _os.path.join(cache_dir, _real_299_cache_name(img_path))
        if not _os.path.exists(src):
            missing.append(img_path)
            continue
        _os.symlink(src, _os.path.join(subset_dir, f"{idx:06d}_"
                                       + _real_299_cache_name(img_path)))
        linked += 1

    if linked < n:
        raise RuntimeError(
            f"real_299 incomplete: linked {linked}/{n} images into "
            f"{subset_dir} (cache: {cache_dir}). FID computed from a short "
            f"real set is invalid, so this is fatal rather than a warning. "
            f"First missing source: {missing[0] if missing else 'n/a'}")

    print(f"  [FID] real_299 ready: {linked} symlinks → {cache_dir}")
    return subset_dir


# ---------------------------------------------------------------------------
# VFL checkpoint management
# ---------------------------------------------------------------------------

def get_vfl_checkpoint_dir(output_dir: str, method: str) -> str:
    """Return the method-scoped VFL checkpoint directory.

    Layout: ``{output_dir}/checkpoints/{method}/`` — segregating by method
    avoids teacache / speca / baseline LoRA weights overwriting each other.
    """
    import os
    return os.path.join(output_dir, "checkpoints", method)


def prune_checkpoints(checkpoint_dir: str, keep: int = 3) -> int:
    """Retain only the ``keep`` most recent ``.pt`` checkpoints.

    Sorts by mtime (descending); older files beyond the cutoff are deleted.
    Returns the number of files removed. Silently no-ops if the directory
    does not exist or contains ≤ ``keep`` files.
    """
    import os
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return 0
    files = []
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            files.append((os.path.getmtime(path), path))
        except OSError:
            continue
    if len(files) <= keep:
        return 0
    files.sort(key=lambda x: x[0], reverse=True)  # newest first
    deleted = 0
    for _, path in files[keep:]:
        try:
            os.remove(path)
            deleted += 1
        except OSError:
            continue
    return deleted
