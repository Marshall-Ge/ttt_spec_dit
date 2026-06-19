# -*- coding: utf-8 -*-
"""ImageReward — human preference score for text-to-image generation.

ImageReward (THUDM/ImageReward, NeurIPS 2023) is a BLIP-based model fine-tuned
on 137k human comparisons. Higher score = better human preference alignment.

Reference: https://github.com/THUDM/ImageReward

Requires:  pip install image-reward
"""

import torch
import numpy as np
from .base import Metric


class ImageRewardScorer(Metric):
    """Human preference score for text-to-image.

    Loads the official ImageReward-v1.0 model via the ``image-reward`` package.

    Parameters
    ----------
    device : str
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._available = None   # None = not yet checked; True/False after first attempt
        self._scores: list = []

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _lazy_load(self):
        if self._available is not None:
            return

        try:
            import ImageReward as RM
            self._model = RM.load("ImageReward-v1.0", device=self.device)
            self._available = True
            print(f"  [ImageReward] loaded (official package, v1.0)")
        except Exception as e:
            self._available = False
            print(f"  [ImageReward] FAILED — {e}")
            print(f"  [ImageReward] Install:  pip install image-reward")
            print(f"  [ImageReward] All scores will be NaN until resolved.")

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, prompt: str, image: torch.Tensor) -> float:
        """Return ImageReward score for one (prompt, image) pair.

        image: [3,H,W] or [1,3,H,W] in [0,1].
        """
        self._lazy_load()
        if not self._available or self._model is None:
            return float("nan")

        from utils import latent_to_pil
        if image.dim() == 4:
            image = image.squeeze(0)
        pil = latent_to_pil(image)
        s = self._model.score(prompt, pil)
        return float(s) if isinstance(s, (int, float)) else float(s[0])

    # ------------------------------------------------------------------
    # Metric interface
    # ------------------------------------------------------------------

    def add(self, image: torch.Tensor, prompt: str = None,
            reference: torch.Tensor = None):
        self._scores.append(self.score(prompt, image))

    def compute(self) -> dict:
        if not self._scores:
            return {"image_reward_mean": float("nan"),
                    "image_reward_std": float("nan")}
        vals = [s for s in self._scores
                if not (isinstance(s, float) and np.isnan(s))]
        if not vals:
            return {"image_reward_mean": float("nan"),
                    "image_reward_std": float("nan")}
        return {
            "image_reward_mean": float(np.mean(vals)),
            "image_reward_std": float(np.std(vals)),
        }

    def reset(self):
        self._scores.clear()
