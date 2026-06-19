# -*- coding: utf-8 -*-
"""LPIPS — learned perceptual image patch similarity."""

import torch

from .base import Metric


class LPIPSScorer(Metric):
    """Learned Perceptual Image Patch Similarity (lower = more similar).

    Gracefully degrades to NaN if VGG weights are unavailable offline.

    Parameters
    ----------
    device : str
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._load_failed = False
        self._scores: list = []

    def _lazy_load(self):
        if self._model is not None or self._load_failed:
            return
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self._model = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg").to(self.device).eval()
            print(f"  [LPIPS] loaded (vgg)")
        except Exception as e:
            print(f"  [WARN] LPIPS unavailable ({e}); LPIPS -> NaN")
            self._load_failed = True

    @torch.no_grad()
    def score(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        """img_a, img_b: [1,3,H,W] in [0,1]. Returns LPIPS distance."""
        self._lazy_load()
        if self._load_failed or self._model is None:
            return float("nan")
        a = (img_a * 2 - 1).to(self.device).float()   # LPIPS expects [-1,1]
        b = (img_b * 2 - 1).to(self.device).float()
        return float(self._model(a, b).item())

    # Metric interface
    def add(self, image: torch.Tensor, prompt: str = None, reference: torch.Tensor = None):
        s = self.score(image, reference)
        self._scores.append(s)

    def compute(self) -> dict:
        if not self._scores:
            return {"lpips_mean": float("nan"), "lpips_std": float("nan")}
        import numpy as np
        vals = [s for s in self._scores if not (isinstance(s, float) and np.isnan(s))]
        if not vals:
            return {"lpips_mean": float("nan"), "lpips_std": float("nan")}
        return {
            "lpips_mean": float(np.mean(vals)),
            "lpips_std": float(np.std(vals)),
        }

    def reset(self):
        self._scores.clear()
