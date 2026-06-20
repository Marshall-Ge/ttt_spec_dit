# -*- coding: utf-8 -*-
"""Abstract base classes for diffusion models and acceleration methods."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch


class DiffusionGenerator(ABC):
    """Abstract text-to-image generator.

    Subclass for each model (PixArt, FLUX, etc.).
    """

    @abstractmethod
    def load(self):
        """Load model weights and move to device."""
        ...

    @abstractmethod
    def generate(self, prompt: Union[str, List[str]],
                 seed: Union[int, List[int]],
                 guidance_scale: float = 4.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate image(s) from prompt(s).

        When *prompt* is a list, each entry is a different prompt; *seed*
        must then be a matching list so every element gets independent
        initial noise.  The batch dimension is ``len(prompts)``.

        Returns (latent, image_tensor), image in [0,1].
        """
        ...

    @abstractmethod
    def unload(self):
        """Free GPU memory."""
        ...


class Accelerator(ABC):
    """Abstract acceleration wrapper.

    Subclass: TeaCache, no-op Vanilla, etc.
    """

    @abstractmethod
    def install(self, generator: "DiffusionGenerator"):
        """Install acceleration onto a generator."""
        ...

    @abstractmethod
    def uninstall(self):
        """Remove acceleration, restore original behaviour."""
        ...

    @property
    @abstractmethod
    def stats(self) -> dict:
        """Return current acceleration statistics."""
        ...
