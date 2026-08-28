# -*- coding: utf-8 -*-
"""Image I/O, tensor↔PIL conversion, VAE decode helpers (pure functions)."""

import numpy as np
import torch
from PIL import Image


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


def ensure_real_299(ds, output_dir: str, n: int,
                   start_index: int = 0) -> str:
    """Ensure real images at 299×299 exist for FID.

    Pre-processes dataset images to 299×299 once into a flat cache keyed by
    SOURCE PATH (not dataset index), so different ``--seed`` shuffle orders
    share the same cache. The run-specific subset links them under names
    prefixed by the ABSOLUTE dataset index (``start_index + i``), which keeps
    resume/dataset-window runs stable.

    Raises RuntimeError when the linked subset is incomplete (missing source
    images must be fatal — a silent skip would produce NaN FID).
    """
    import os as _os
    from tqdm import tqdm as _tqdm

    val_dir = getattr(ds, 'val_dir', None)
    if val_dir is None:
        val_dir = _os.path.dirname(ds[0][0]) if len(ds) > 0 else "/tmp"
    cache_dir = _os.path.join(_os.path.dirname(val_dir) or val_dir, "val_299_cache")
    _os.makedirs(cache_dir, exist_ok=True)

    # 1) Pre-process each source image once, keyed by its path stem.
    existing = set(_os.listdir(cache_dir))
    for i in _tqdm(range(len(ds)), desc="preprocess real 299", ncols=80):
        item = ds[i]
        img_path = item[0] if isinstance(item, (tuple, list)) else item
        if not _os.path.exists(img_path):
            continue
        stem = _os.path.splitext(_os.path.basename(img_path))[0]
        cache_name = f"src_{stem}.png"
        out_path = _os.path.join(cache_dir, cache_name)
        if cache_name in existing or _os.path.exists(out_path):
            continue
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = pil_img.resize((299, 299), Image.BICUBIC)
        pil_img.save(out_path)

    # 2) Link the run-specific subset with absolute-index names.
    subset_dir = _os.path.join(output_dir, "real_299")
    _os.makedirs(subset_dir, exist_ok=True)
    for f in _os.listdir(subset_dir):
        p = _os.path.join(subset_dir, f)
        if _os.path.islink(p) or f.endswith('.png'):
            _os.remove(p)

    linked = 0
    for i in range(n):
        item = ds[i]
        img_path = item[0] if isinstance(item, (tuple, list)) else item
        if not _os.path.exists(img_path):
            continue
        stem = _os.path.splitext(_os.path.basename(img_path))[0]
        cls_name = str(item[1]).replace("a photo of a ", "").replace(" ", "_") \
            if isinstance(item, (tuple, list)) and len(item) > 1 else "unknown"
        abs_idx = start_index + i
        link_name = f"{abs_idx:06d}_source_{abs_idx}_{cls_name}.png"
        src = _os.path.join(cache_dir, f"src_{stem}.png")
        dst = _os.path.join(subset_dir, link_name)
        if _os.path.exists(src):
            _os.symlink(src, dst)
            linked += 1

    if linked != n:
        raise RuntimeError(
            f"real_299 incomplete: linked {linked}/{n} "
            f"(missing source images in {val_dir})")

    print(f"  [FID] real_299 ready: {n} symlinks → {cache_dir}")
    return subset_dir
