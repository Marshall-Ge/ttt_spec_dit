# -*- coding: utf-8 -*-
"""Explicit diffusion transformer models.

Forward signatures are explicit about the SpecA cache state::

    DiTTransformer2D.forward(x, t, current, cache_dic, class_labels)
    PixArtTransformer2D.forward(x, encoder_hidden_states, t, current, cache_dic, ...)
"""

from .dit import DiTTransformer2D
from .pixart import PixArtTransformer2D
