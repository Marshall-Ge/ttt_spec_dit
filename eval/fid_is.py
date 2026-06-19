# -*- coding: utf-8 -*-
"""FID + IS via torch-fidelity.calculate_metrics.

Standard protocol:
  1. Collect generated images as PNGs in a flat directory (299×299).
  2. Collect real reference images as PNGs in a flat directory (299×299).
  3. Call torch-fidelity's calculate_metrics to compute FID + IS.
"""

import os
from typing import Dict, Optional

import torch
from PIL import Image

from .base import Metric


class FIDISComputer(Metric):
    """FID + Inception Score via torch-fidelity.

    Parameters
    ----------
    real_dir : str
        Path to a flat directory of 299×299 real reference PNGs.
    gen_dir : str
        Path where generated images will be saved (299×299 PNGs).
    device : str
    """

    def __init__(self, real_dir: str = None, gen_dir: str = None,
                 device: str = "cuda"):
        self.real_dir = real_dir
        self.gen_dir = gen_dir
        self.device = device
        self._counter = 0
        self._load_failed = False
        self._did_init = False

    def _init_dirs(self):
        """Ensure gen_dir exists and has a clean counter."""
        if self._did_init:
            return
        if self.gen_dir is None:
            raise ValueError("FIDISComputer.gen_dir must be set before adding images.")
        os.makedirs(self.gen_dir, exist_ok=True)
        self._did_init = True

    def add(self, image: torch.Tensor, prompt: str = None, reference: torch.Tensor = None):
        """Save one generated image as 299×299 PNG to gen_dir.

        image: [3,H,W] or [1,3,H,W] in [0,1].
        """
        self._init_dirs()
        if image.dim() == 4:
            image = image.squeeze(0)
        # Resize to 299×299 if needed
        if image.shape[-1] != 299 or image.shape[-2] != 299:
            pil = Image.fromarray(
                (image.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            )
            pil = pil.resize((299, 299), Image.BICUBIC)
            out_path = os.path.join(self.gen_dir, f"{self._counter:06d}.png")
            pil.save(out_path)
        else:
            from utils import save_image
            out_path = os.path.join(self.gen_dir, f"{self._counter:06d}.png")
            save_image(image, out_path)
        self._counter += 1

    def compute(self) -> Dict[str, float]:
        """Run torch-fidelity calculate_metrics.

        Returns dict with keys: fid, is_mean, is_std.
        """
        if self._load_failed:
            return {"fid": float("nan"), "is_mean": float("nan"), "is_std": float("nan")}

        if not self.real_dir or not os.path.isdir(self.real_dir):
            print(f"  [WARN] FID: real_dir not set or missing; FID/IS -> NaN")
            return {"fid": float("nan"), "is_mean": float("nan"), "is_std": float("nan")}

        if not self.gen_dir or not os.path.isdir(self.gen_dir):
            print(f"  [WARN] FID: gen_dir not set or missing; FID/IS -> NaN")
            return {"fid": float("nan"), "is_mean": float("nan"), "is_std": float("nan")}

        gen_files = [f for f in os.listdir(self.gen_dir) if f.endswith('.png')]
        if not gen_files:
            print(f"  [WARN] FID: no generated images in {self.gen_dir}; FID/IS -> NaN")
            return {"fid": float("nan"), "is_mean": float("nan"), "is_std": float("nan")}

        try:
            from torch_fidelity import calculate_metrics
            print(f"\n  [FID/IS] Computing via torch-fidelity...")
            print(f"    Real: {self.real_dir}")
            print(f"    Gen:  {self.gen_dir} ({len(gen_files)} images)")

            metrics = calculate_metrics(
                input1=self.real_dir,
                input2=self.gen_dir,
                cuda=True,
                isc=True,
                fid=True,
                verbose=False,
                samples_find_ext='png',
            )

            fid_val = float(metrics.get('frechet_inception_distance', float('nan')))
            is_mean = float(metrics.get('inception_score_mean', float('nan')))
            is_std = float(metrics.get('inception_score_std', float('nan')))

            print(f"    FID = {fid_val:.4f}")
            print(f"    IS  = {is_mean:.4f} ± {is_std:.4f}")

            return {"fid": fid_val, "is_mean": is_mean, "is_std": is_std}

        except Exception as e:
            print(f"  [WARN] torch-fidelity compute failed ({e}); FID/IS -> NaN")
            self._load_failed = True
            return {"fid": float("nan"), "is_mean": float("nan"), "is_std": float("nan")}

    def reset(self):
        """Clear generated images and reset counter."""
        self._counter = 0
        if self.gen_dir and os.path.isdir(self.gen_dir):
            for f in os.listdir(self.gen_dir):
                if f.endswith('.png'):
                    os.remove(os.path.join(self.gen_dir, f))
        self._did_init = False
