# -*- coding: utf-8 -*-
"""CLIP Score — image-text cosine similarity."""

import torch
import numpy as np

from config import CLIP_PATH
from utils import latent_to_pil
from .base import Metric


class CLIPScorer(Metric):
    """CLIP Score (image-text cosine similarity × 100).

    Parameters
    ----------
    clip_path : str
        Local directory with the CLIP model + processor.
    device : str
    dtype : torch.dtype
    """

    def __init__(self, clip_path: str = None, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.clip_path = clip_path or CLIP_PATH
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None
        self._load_failed = False
        self._scores: list = []

    def _lazy_load(self):
        if self._model is not None or self._load_failed:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(
                self.clip_path, local_files_only=True).to(
                self.device, dtype=self.dtype).eval()
            self._processor = CLIPProcessor.from_pretrained(
                self.clip_path, local_files_only=True)
            print(f"  [CLIP] loaded from {self.clip_path}")
        except Exception as e:
            print(f"  [WARN] CLIP load failed ({e}); CLIP scores -> NaN")
            self._load_failed = True

    @torch.no_grad()
    def score(self, prompt: str, image: torch.Tensor) -> float:
        """Cosine similarity (image, text) × 100. image: [3,H,W] or [1,3,H,W] in [0,1]."""
        self._lazy_load()
        if self._load_failed or self._model is None:
            return float("nan")
        if image.dim() == 4:
            image = image[0]
        pil_image = latent_to_pil(image)
        inputs = self._processor(text=[prompt], images=[pil_image],
                                 return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cosine = (img_emb * txt_emb).sum(dim=-1)
        return float(cosine.item() * 100.0)

    # Metric interface
    def add(self, image: torch.Tensor, prompt: str = None, reference: torch.Tensor = None):
        s = self.score(prompt, image)
        self._scores.append(s)

    @torch.no_grad()
    def add_batch(self, images: torch.Tensor, prompts: list):
        """Add scores for a batch of (image, prompt) pairs using batched CLIP forward.

        Parameters
        ----------
        images : torch.Tensor
            (B, 3, H, W) in [0,1].
        prompts : list of str
            Length B text prompts.
        """
        self._lazy_load()
        if self._load_failed or self._model is None:
            for _ in range(images.shape[0]):
                self._scores.append(float("nan"))
            return

        from utils import latent_to_pil
        pil_list = []
        for i in range(images.shape[0]):
            img = images[i]
            if img.dim() == 3:
                pil_list.append(latent_to_pil(img))
            else:
                pil_list.append(latent_to_pil(img[0] if img.dim() == 4 else img))

        inputs = self._processor(text=prompts, images=pil_list,
                                 return_tensors="pt", padding=True,
                                 truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        img_emb = outputs.image_embeds  # (B, D)
        txt_emb = outputs.text_embeds   # (B, D)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cosines = (img_emb * txt_emb).sum(dim=-1)  # (B,)
        for c in cosines:
            self._scores.append(float(c.item() * 100.0))

    def compute(self) -> dict:
        if not self._scores:
            return {"clip_score_mean": float("nan"), "clip_score_std": float("nan")}
        vals = [s for s in self._scores if not (isinstance(s, float) and np.isnan(s))]
        if not vals:
            return {"clip_score_mean": float("nan"), "clip_score_std": float("nan")}
        return {
            "clip_score_mean": float(np.mean(vals)),
            "clip_score_std": float(np.std(vals)),
        }

    def reset(self):
        self._scores.clear()
