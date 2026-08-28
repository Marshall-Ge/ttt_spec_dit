# -*- coding: utf-8 -*-
"""Latent MSE + Pixel MSE metrics."""

import torch

from .base import Metric


def compute_latent_mse(latent_accel: torch.Tensor,
                       latent_vanilla: torch.Tensor) -> float:
    """Mean squared error between two latent tensors."""
    diff = (latent_accel.float().detach().cpu() -
            latent_vanilla.float().detach().cpu()) ** 2
    return diff.mean().item()


def compute_pixel_mse(image_accel: torch.Tensor,
                      image_vanilla: torch.Tensor) -> float:
    """Mean squared error between two image tensors (in [0,1])."""
    diff = (image_accel.float().detach().cpu() -
            image_vanilla.float().detach().cpu()) ** 2
    return diff.mean().item()


class MSEMetric(Metric):
    """Collects latent MSE and pixel MSE over multiple pairs.

    Parameters
    ----------
    which : str
        "latent" or "pixel".
    """

    def __init__(self, which: str = "pixel"):
        self.which = which
        self._compute_fn = compute_latent_mse if which == "latent" else compute_pixel_mse
        self._scores: list = []

    def add(self, image: torch.Tensor, prompt: str = None, reference: torch.Tensor = None):
        if reference is None:
            return
        s = self._compute_fn(image, reference)
        self._scores.append(s)

    def add_batch(self, images: torch.Tensor, references: torch.Tensor):
        """Add MSE scores for a batch of (image, reference) pairs.

        Parameters
        ----------
        images : torch.Tensor
            (B, 3, H, W) in [0,1].
        references : torch.Tensor
            (B, 3, H, W) in [0,1].
        """
        if images.shape[0] != references.shape[0]:
            return
        diff = (images.float().detach().cpu() - references.float().detach().cpu()) ** 2
        per_image_mse = diff.flatten(1).mean(1)  # (B,)
        self._scores.extend(per_image_mse.tolist())

    def compute(self) -> dict:
        if not self._scores:
            key = f"{self.which}_mse_mean"
            return {key: float("nan"), f"{self.which}_mse_std": float("nan")}
        import numpy as np
        vals = [s for s in self._scores if not (isinstance(s, float) and np.isnan(s))]
        if not vals:
            key = f"{self.which}_mse_mean"
            return {key: float("nan"), f"{self.which}_mse_std": float("nan")}
        return {
            f"{self.which}_mse_mean": float(np.mean(vals)),
            f"{self.which}_mse_std": float(np.std(vals)),
        }

    def reset(self):
        self._scores.clear()
