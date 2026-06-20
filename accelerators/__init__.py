# -*- coding: utf-8 -*-
"""Accelerators for diffusion model inference.

Each accelerator implements the Accelerator interface (models/base.py)
and auto-detects the generator type to install the correct forward
replacement.
"""

from .teacache import TeaCacheAccelerator
from .speca import SpecAAccelerator
