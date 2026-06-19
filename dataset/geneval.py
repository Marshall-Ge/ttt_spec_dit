# -*- coding: utf-8 -*-
"""GenEval 553 compositional prompt dataset.

Loads prompts from the official geneval metadata JSONL.
Each item is a (prompt, tag) pair where tag is one of:
  single_object, two_object, counting, colors, position, color_attr
"""

import json
import os
import numpy as np

from .base import PromptDataset


class GenEvalDataset(PromptDataset):
    """GenEval compositional T2I evaluation dataset.

    Parameters
    ----------
    n_prompts : int or None
        Number of prompts to use (max 553). None = all.
    base_seed : int
        Base seed offset for per-prompt seeds.
    metadata_path : str or None
        Path to geneval_metadata.jsonl. Defaults to eval/geneval_metadata.jsonl.
    """

    def __init__(self, n_prompts: int = None, base_seed: int = 42,
                 metadata_path: str = None):
        if metadata_path is None:
            metadata_path = os.path.join(
                os.path.dirname(__file__), "..", "eval", "geneval_metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"GenEval metadata not found at {metadata_path}")

        self.items = []
        with open(metadata_path) as f:
            for line in f:
                item = json.loads(line)
                self.items.append((item["prompt"], item["tag"]))

        # Deterministic subsample
        if n_prompts is not None and len(self.items) > n_prompts:
            rng = np.random.RandomState(42)
            idxs = rng.choice(len(self.items), size=n_prompts, replace=False)
            self.items = [self.items[i] for i in idxs]

        self.base_seed = base_seed
        print(f"  [GenEval] {len(self.items)} prompts loaded "
              f"(requested {n_prompts or 'all'})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        prompt, tag = self.items[idx]
        seed = self.base_seed + idx
        return prompt, seed, tag
