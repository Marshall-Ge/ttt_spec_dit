# -*- coding: utf-8 -*-
"""Accelerators for diffusion model inference — pure functions, no classes.

  - speca:   per-block per-submodule Taylor cache; ``cache_dic``/``current``
             dicts are owned by the caller and threaded through the model.
  - teacache: whole-step residual cache; integrated at the top-level sampling
             loop (the model itself is agnostic to it).
"""

from .speca import (
    speca_init,
    speca_cal_type,
    derivative_approximation,
    taylor_formula,
    taylor_cache_init,
    cache_step_dit,
    cache_step_pixart,
    compute_error_gate,
)
from .teacache import (
    teacache_init,
    teacache_decide,
    teacache_cache_residual,
    teacache_apply_residual,
    teacache_step,
    teacache_reset,
)
