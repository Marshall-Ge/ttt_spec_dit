# -*- coding: utf-8 -*-
"""COCO 2014 validation dataset: (image_path, caption) pairs."""

import os
import numpy as np
from torch.utils.data import Dataset

from config import COCO_DIR


class COCO30KDataset(Dataset):
    """Load COCO 2014 val captions, returning (image_path, caption) pairs.

    Parameters
    ----------
    coco_dir : str
        Root COCO directory (contains val2014/ and annotations/).
    n_images : int
        Number of image-caption pairs to sample.
    seed : int
        Random seed for deterministic shuffle.
    """

    def __init__(self, coco_dir: str = None, n_images: int = 30000,
                 seed: int = 42):
        import json as _json
        coco_dir = coco_dir or COCO_DIR
        anno_path = os.path.join(coco_dir, "annotations", "captions_val2014.json")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(
                f"Captions not found at {anno_path}. "
                f"Run scripts/download_coco.sh first."
            )
        with open(anno_path) as f:
            data = _json.load(f)

        # Group captions by image_id, pick first caption per image
        img_to_caption = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_caption:
                img_to_caption[img_id] = ann["caption"]

        # Build list of (image_path, caption)
        img_dir = os.path.join(coco_dir, "val2014")
        items = []
        for img_info in data["images"]:
            img_id = img_info["id"]
            if img_id in img_to_caption:
                fname = img_info["file_name"]
                img_path = os.path.join(img_dir, fname)
                if os.path.exists(img_path):
                    items.append((img_path, img_to_caption[img_id]))

        # Deterministic shuffle + subsample
        rng = np.random.RandomState(seed)
        rng.shuffle(items)
        self.items = items[:n_images]
        print(f"  [COCO] {len(self.items)} image-caption pairs loaded "
              f"(requested {n_images})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        return img_path, caption


# Alias
COCODataset = COCO30KDataset
