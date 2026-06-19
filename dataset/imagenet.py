# -*- coding: utf-8 -*-
"""ImageNet 2012 validation dataset.

Iterates class subdirectories under val/, produces prompts like
"a photo of a {class_name}", embeds ImageNet 1K class names.
"""

import os
from typing import List

import numpy as np

from config import IMAGENET_DIR

# ImageNet 1K class names (synset words, ordered by class index 0-999)
# A compact subset — if the full mapping is needed, load from a
# class_index.json or similar mapping file.
# For now we derive class names from the directory names.
# ImageNet dirs are named like "n01440764" — we map to plain english via
# a canonical mapping. If no mapping is available, we use a template.

IMAGENET_CLASS_NAMES = None  # Set to a list of 1000 strings if available


def _load_class_names(imagenet_dir: str) -> List[str]:
    """Try to load or derive class names for ImageNet folders."""
    val_dir = os.path.join(imagenet_dir, "val")
    if not os.path.exists(val_dir):
        val_dir = imagenet_dir  # maybe the path IS the val dir

    # Look for a class name mapping file
    mapping_paths = [
        os.path.join(imagenet_dir, "imagenet_class_index.json"),
        os.path.join(imagenet_dir, "class_index.json"),
        os.path.join(imagenet_dir, "synset_words.txt"),
    ]

    for mp in mapping_paths:
        if os.path.exists(mp):
            if mp.endswith(".json"):
                import json
                with open(mp) as f:
                    data = json.load(f)
                # Typical format: {"0": ["n01440764", "tench"], ...}
                if isinstance(data, dict):
                    names = []
                    for k in sorted(data.keys(), key=lambda x: int(x)):
                        names.append(data[k][1] if isinstance(data[k], list) else data[k])
                    if len(names) == 1000:
                        return names
            elif mp.endswith(".txt"):
                with open(mp) as f:
                    # Format: "n01440764 tench, Tinca tinca"
                    names = []
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            names.append(parts[1].split(",")[0].strip())
                    if len(names) == 1000:
                        return names

    # Fallback: derive class name from synset directory name
    subdirs = sorted(os.listdir(val_dir))
    if len(subdirs) == 1000:
        print("  [ImageNet] No class-name mapping found; using synset IDs as class names.")
        return subdirs  # Use "n01440764" style names

    raise FileNotFoundError(
        f"Could not find 1000 class subdirectories in {val_dir}. "
        f"Expected ImageNet val directory with 1000 class subdirectories."
    )


class ImageNetDataset:
    """ImageNet 2012 validation dataset.

    Parameters
    ----------
    imagenet_dir : str
        Root ImageNet directory (contains val/ with 1000 class subdirs).
    n_images : int
        Number of images to use (max 50000).
    seed : int
        Random seed for shuffle.
    class_names : list of str, optional
        Mapping from class index to human-readable name.
    """

    def __init__(self, imagenet_dir: str = None, n_images: int = 50000,
                 seed: int = 42, class_names: List[str] = None):
        imagenet_dir = imagenet_dir or IMAGENET_DIR
        val_dir = os.path.join(imagenet_dir, "val")
        if not os.path.exists(val_dir):
            val_dir = imagenet_dir
        self.val_dir = val_dir
        self.n_images = n_images

        # Load class names
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = _load_class_names(imagenet_dir)

        # Build (image_path, class_idx, prompt) list
        self.items = []
        class_dirs = sorted(os.listdir(val_dir))
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(val_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            class_name = (self.class_names[class_idx]
                          if class_idx < len(self.class_names)
                          else class_dir)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                    img_path = os.path.join(class_path, fname)
                    prompt = f"a photo of a {class_name}"
                    self.items.append((img_path, prompt, class_idx))

        # Deterministic shuffle + subsample
        rng = np.random.RandomState(seed)
        rng.shuffle(self.items)
        self.items = self.items[:n_images]
        print(f"  [ImageNet] {len(self.items)} image-prompt pairs loaded "
              f"(requested {n_images})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, prompt, class_idx = self.items[idx]
        return img_path, prompt

    def image_dir_for_fid(self) -> str:
        """Pre-resize all real images to 299×299 for FID computation.

        Returns the path to a flat directory of 299×299 PNGs.
        """
        # We don't pre-resize here; the eval/fid_is.py module handles that.
        # Return the val directory — the FIDISComputer will handle resizing.
        return self.val_dir
