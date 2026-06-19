# -*- coding: utf-8 -*-
"""Abstract base classes for prompt and real-image datasets."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple


class PromptDataset(ABC):
    """Yields (prompt, seed) pairs for generation."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[str, int]:
        ...


class RealImageDataset(ABC):
    """Yields real reference images for FID computation."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def image_dir(self) -> str:
        """Path to flat directory of 299×299 PNGs."""
        ...
