# -*- coding: utf-8 -*-
"""Abstract Metric interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class Metric(ABC):
    """Abstract evaluation metric.

    All metrics are optional and degrade gracefully to NaN when
    dependencies are unavailable.
    """

    @abstractmethod
    def add(self, image: torch.Tensor, prompt: str = None,
            reference: torch.Tensor = None):
        """Accumulate one image (or image pair)."""
        ...

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Compute and return metric(s). Returns a dict like {"fid": ..., "is_mean": ...}."""
        ...

    @abstractmethod
    def reset(self):
        """Clear accumulated state."""
        ...
