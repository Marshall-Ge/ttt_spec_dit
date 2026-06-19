# -*- coding: utf-8 -*-
"""DrawBench 200 prompt dataset."""

import os
from typing import List

from .base import PromptDataset


# Default path: phase2/drawbench200.txt (kept alongside old code for reference)
DEFAULT_DRAWBENCH_PATH = os.path.join(os.path.dirname(__file__), "..", "drawbench200.txt")


def load_drawbench_prompts(path: str = None, n: int = 200) -> List[str]:
    """Load DrawBench prompts from a text file (one per line).

    Parameters
    ----------
    path : str, optional
        Path to the prompts file. Defaults to phase2/drawbench200.txt.
    n : int
        Maximum number of prompts to return.
    """
    path = path or DEFAULT_DRAWBENCH_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DrawBench prompts not found at {path}. "
            f"Download from Cache4Diffusion: "
            f"https://raw.githubusercontent.com/Shenyi-Z/Cache4Diffusion/"
            f"main/assets/prompts/DrawBench200.txt"
        )
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:n]


class DrawBenchDataset(PromptDataset):
    """DrawBench prompt dataset with deterministic seeds.

    Parameters
    ----------
    n_prompts : int
        Number of prompts (max 200).
    base_seed : int
        Base seed offset for per-prompt seeds.
    path : str, optional
        Path to prompts file.
    """

    def __init__(self, n_prompts: int = 200, base_seed: int = 42,
                 path: str = None):
        self.prompts = load_drawbench_prompts(path=path, n=n_prompts)
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        seed = self.base_seed + idx
        return prompt, seed
