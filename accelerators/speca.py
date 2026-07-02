# -*- coding: utf-8 -*-
"""SpecA (Speculative Acceleration) — pure functions, no monkeypatching.

Cache4Diffusion mechanics:
  1. ``speca_cal_type()`` decides 'full' or 'Taylor' for each denoising step
  2. Full step: compute all blocks, cache per-block per-submodule
     features + finite-difference derivatives
  3. Taylor step: predict each submodule's output via a Taylor series
     (cheap — skips attention/MLP computation)
  4. Last-block error check: if Taylor-vs-full error exceeds threshold,
     next step reverts to full
  5. 4 hyperparams: base_threshold, decay_rate, min_taylor_steps, max_taylor_steps

``SpecACache`` and ``SpecAState`` are plain classes owned and threaded by the
caller (the top-level sampling loop in ``run_dit.py`` / ``run_pixart.py``).

Module-name convention (keys into ``cache.cache[-1][layer]``):
  - DiT  (ada_norm_zero):   'attn', 'mlp'         (2 submodules)
  - PixArt (ada_norm_single): 'attn1', 'attn2', 'ff'  (3 submodules)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


# Precomputed 1 / n! for Taylor coefficients (up to order 6)
_INV_FACTORIAL = [1.0, 1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0]


# ===========================================================================
# SpecAState — per-step mutable state
# ===========================================================================

class SpecAState:
    """Per-step mutable state for SpecA acceleration.

    Owned by the top-level sampling loop; threaded explicitly through
    ``forward()`` and mutated during the denoising loop.  Replaces the
    old flat ``current`` dict with named, documented attributes.

    Attributes set by the denoising loop (before ``forward()``):
        * ``step`` — current step index (counts DOWN from num_steps-1 to 0).

    Attributes set by ``speca_cal_type()``:
        * ``type`` — ``'full'`` or ``'Taylor'`` for this step.
        * ``last_type`` — the type of the *previous* step.
        * ``last_layer_error`` — error from the check-layer probe (or ``None``
          immediately after a full step, signalling "no error yet").

    Attributes set by the model during ``forward()``:
        * ``layer`` — current block index (0..27).
        * ``module`` — current submodule name (``'attn'``/``'mlp'`` etc.).
    """

    def __init__(self, num_steps: int):
        # Written by the denoising loop.
        self.step: int = 0

        # Written by speca_cal_type.
        self.type: str = 'full'          # safe default; overwritten each forward
        self.last_type: str = 'None'
        self.last_layer_error: Optional[float] = 0.0

        # Written by the model during ``forward()``.
        self.layer: int = 0
        self.module: Optional[str] = None

        # Immutable after init.
        self.num_steps: int = num_steps
        self.activated_steps: List[int] = [num_steps - 1]  # descending order


# ===========================================================================
# SpecACache — per-image acceleration cache
# ===========================================================================

class SpecACache:
    """Persistent cache for SpecA acceleration.

    Owned by the top-level sampling loop; threaded explicitly through
    ``forward()`` and mutated during the denoising loop.  Replaces the
    old flat ``cache_dic`` dict with named, documented attributes.

    The nested ``cache`` dict indexes by ``(step_index, layer_index,
    module_name)`` and stores lists of Taylor-factor tensors.
    ``cache[-1]`` is the *active* slot — the most recent full step's
    cache, used for prediction during Taylor steps.
    """

    def __init__(self, num_steps: int, num_layers: int,
                 base_threshold: float, decay_rate: float,
                 min_taylor_steps: int, max_taylor_steps: int,
                 max_order: int = 4,
                 error_metric: str = 'cosine_similarity',
                 check_layer: int = 27):
        # ---- Nested cache ----
        cache: Dict[int, Dict[int, Dict[str, list]]] = {}
        # Active slot (most recent full step).
        cache[-1] = {}
        for j in range(num_layers):
            cache[-1][j] = {}
        # Per-step slots.
        for i in range(num_steps):
            cache[i] = {}
            for j in range(num_layers):
                cache[i][j] = {}
        self.cache: Dict[int, Dict[int, Dict[str, list]]] = cache

        # ---- Configuration (immutable after init) ----
        self.max_order: int = max_order
        self.first_enhance: int = 3
        self.base_threshold: float = base_threshold
        self.decay_rate: float = decay_rate
        self.min_taylor_steps: int = min_taylor_steps
        self.max_taylor_steps: int = max_taylor_steps
        self.error_metric: str = error_metric
        self.check_layer: int = check_layer

        # ---- Runtime counters (mutated by speca_cal_type) ----
        self.cache_counter: int = 0
        self.taylor_step_counter: int = 0
        self.check: bool = False
        self.full_count: int = 0


# ===========================================================================
# Taylor utilities (from speca-dit/taylor_utils/__init__.py)
# ===========================================================================

def derivative_approximation(cache_dic: SpecACache, current: SpecAState,
                             feature: torch.Tensor) -> None:
    """Compute finite-difference derivative approximation.

    Reads the previous Taylor factor list for ``(current layer, module)``
    from ``cache_dic.cache[-1]`` and overwrites it with an updated list,
    where entry ``k`` is the k-th finite-difference derivative (entry 0 is the
    feature itself).
    """
    difference_distance = (
        current.activated_steps[-1] - current.activated_steps[-2]
    )

    updated_taylor_factors = [feature]  # order 0

    prev_cache = cache_dic.cache[-1][current.layer][current.module]
    still_enhancing = (
        current.step
        < (current.num_steps - cache_dic.first_enhance + 1)
    )

    for i in range(cache_dic.max_order):
        if i < len(prev_cache) and still_enhancing:
            updated_taylor_factors.append(
                (updated_taylor_factors[i] - prev_cache[i]) / difference_distance
            )
        else:
            break

    cache_dic.cache[-1][current.layer][current.module] = \
        updated_taylor_factors


def taylor_cache_init(cache_dic: SpecACache, current: SpecAState) -> None:
    """Allocate (reset) a per-module cache slot on the very first step."""
    if current.step == (current.num_steps - 1):
        cache_dic.cache[-1][current.layer][current.module] = []


def taylor_formula(module_list: list, distance: int) -> torch.Tensor:
    """Evaluate Taylor series prediction.

    :param module_list: ``[feat_order0, feat_order1, ...]``
    :param distance:    steps since last activation
    """
    if not module_list:
        return 0
    out = 0
    for i, feat in enumerate(module_list):
        coeff = _INV_FACTORIAL[i] * (distance ** i)
        out = out + coeff * feat
    return out


# ---- cache_step (block-level Taylor application) ----

def cache_step_dit(x: torch.Tensor,
                   attn_list: list,
                   mlp_list: list,
                   gate_msa: torch.Tensor,
                   gate_mlp: torch.Tensor,
                   distance: int) -> torch.Tensor:
    """DiT Taylor prediction (2 submodules): attn + mlp.

    gate_msa/gate_mlp: (B, C) from adaLN-Zero, unsqueezed to (B, 1, C).
    """
    pred_attn = taylor_formula(attn_list, distance)
    pred_mlp = taylor_formula(mlp_list, distance)
    x = x + gate_msa.unsqueeze(1) * pred_attn
    x = x + gate_mlp.unsqueeze(1) * pred_mlp
    return x


def cache_step_pixart(x: torch.Tensor,
                      attn1_list: list,
                      attn2_list: list,
                      ff_list: list,
                      gate_msa: torch.Tensor,
                      gate_mlp: torch.Tensor,
                      distance: int) -> torch.Tensor:
    """PixArt Taylor prediction (3 submodules): attn1 + attn2 + ff.

    gate_msa/gate_mlp: (B, 1, C) already broadcast from chunk.
    attn2 has NO gate (cross-attention).
    """
    pred_attn1 = taylor_formula(attn1_list, distance)
    pred_attn2 = taylor_formula(attn2_list, distance)
    pred_ff = taylor_formula(ff_list, distance)
    x = x + gate_msa * pred_attn1
    x = x + pred_attn2   # no gate for cross-attention
    x = x + gate_mlp * pred_ff
    return x


# ===========================================================================
# Error calculation — unified metrics from speca-dit/models.py
# ===========================================================================

def calculate_l1_error(x: torch.Tensor, full_x: torch.Tensor) -> float:
    """L1 error (mean absolute error)."""
    return torch.abs(x - full_x).mean().item()


def calculate_l2_error(x: torch.Tensor, full_x: torch.Tensor) -> float:
    """L2 error (root mean square error)."""
    return torch.sqrt(torch.mean((x - full_x) ** 2)).item()


def calculate_relative_l1_error(x: torch.Tensor, full_x: torch.Tensor,
                                eps: float = 1e-10) -> float:
    """Relative L1 error between Taylor prediction and full computation."""
    error = torch.abs(x - full_x) / (torch.abs(full_x) + eps)
    return error.mean().item()


def calculate_relative_l2_error(x: torch.Tensor, full_x: torch.Tensor,
                                eps: float = 1e-10) -> float:
    """Relative L2 error (root-mean-squared relative error)."""
    error = torch.abs(x - full_x) / (torch.abs(full_x) + eps)
    return torch.sqrt(torch.mean(error ** 2)).item()


def calculate_cosine_similarity_error(x: torch.Tensor, full_x: torch.Tensor,
                                      eps: float = 1e-10) -> float:
    """Cosine similarity error (1 - cosine_similarity)."""
    x_flat = x.reshape(x.size(0), -1)
    full_x_flat = full_x.reshape(full_x.size(0), -1)
    cosine_sim = torch.nn.functional.cosine_similarity(
        x_flat, full_x_flat, dim=1, eps=eps)
    return (1.0 - cosine_sim.mean()).item()


def calculate_all_errors(x: torch.Tensor, full_x: torch.Tensor,
                         eps: float = 1e-10) -> Dict[str, float]:
    """Compute all error metrics. Returns a dict with keys:
    l1, l2, relative_l1, relative_l2, cosine_similarity.
    """
    return {
        'l1': calculate_l1_error(x, full_x),
        'l2': calculate_l2_error(x, full_x),
        'relative_l1': calculate_relative_l1_error(x, full_x, eps),
        'relative_l2': calculate_relative_l2_error(x, full_x, eps),
        'cosine_similarity': calculate_cosine_similarity_error(x, full_x, eps),
    }


# ---- unified dispatcher ----

_VALID_ERROR_METRICS = {
    'l1', 'l2', 'relative_l1', 'relative_l2', 'cosine_similarity', 'all',
}

_SINGLE_METRIC_FNS = {
    'l1': calculate_l1_error,
    'l2': calculate_l2_error,
    'relative_l1': calculate_relative_l1_error,
    'relative_l2': calculate_relative_l2_error,
    'cosine_similarity': calculate_cosine_similarity_error,
}


def compute_error_gate(x: torch.Tensor,
                       full_x: torch.Tensor,
                       metric: str = 'relative_l1',
                       eps: float = 1e-10) -> Tuple[float, Optional[Dict[str, float]]]:
    """Compute error and return (gate_value, full_error_dict).

    - ``metric == 'all'``:  returns ``(relative_l1, {all five metrics})``
    - single-metric mode:   returns ``(metric_value, None)``

    ``gate_value`` is always a float suitable for threshold comparison.
    """
    if metric not in _VALID_ERROR_METRICS:
        raise ValueError(
            f"Unknown error_metric: {metric!r}. "
            f"Valid choices: {sorted(_VALID_ERROR_METRICS)}")

    if metric == 'all':
        errors = calculate_all_errors(x, full_x, eps)
        return errors['relative_l1'], errors

    fn = _SINGLE_METRIC_FNS[metric]
    _METRICS_NEEDING_EPS = {'relative_l1', 'relative_l2', 'cosine_similarity'}
    if metric in _METRICS_NEEDING_EPS:
        return fn(x, full_x, eps), None
    return fn(x, full_x), None


# ===========================================================================
# Cache initialisation
# ===========================================================================

def speca_init(
    num_steps: int,
    base_threshold: float,
    decay_rate: float,
    min_taylor_steps: int,
    max_taylor_steps: int,
    max_order: int = 4,
    num_layers: int = 28,
    error_metric: str = 'cosine_similarity',
    check_layer: int = 27,
) -> Tuple[SpecACache, SpecAState]:
    """Allocate the SpecA cache and current-state objects.

    Returns ``(SpecACache, SpecAState)``. Both are owned by the caller
    and threaded explicitly through the model's ``forward``.

    Parameters
    ----------
    num_layers : int
        Number of transformer blocks (28 for both PixArt and DiT).
    check_layer : int
        Layer index at which the Taylor-vs-full error probe runs
        (DiT: 20, PixArt: 24; overrides the config default of 27).
    """
    cache_dic = SpecACache(
        num_steps=num_steps,
        num_layers=num_layers,
        base_threshold=base_threshold,
        decay_rate=decay_rate,
        min_taylor_steps=min_taylor_steps,
        max_taylor_steps=max_taylor_steps,
        max_order=max_order,
        error_metric=error_metric,
        check_layer=check_layer,
    )
    current = SpecAState(num_steps=num_steps)
    return cache_dic, current


# ===========================================================================
# Step-type decision
# ===========================================================================

def speca_cal_type(cache_dic: SpecACache, current: SpecAState,
                   calibrator=None) -> None:
    """Decide whether the current step is 'full' or 'Taylor'.

    Side-effects: updates ``current.type`` / ``current.last_type`` and
    the counters in ``cache_dic``.

    Parameters
    ----------
    calibrator : OnlineCalibrator, optional
        VFL M2 online calibrator. When provided and ready for the current bucket
        at ``cache_dic.check_layer``, its ``get_threshold()`` overrides the
        static ``base_threshold * (decay_rate ** progress)`` formula.
    """
    min_taylor_steps = cache_dic.min_taylor_steps
    max_taylor_steps = cache_dic.max_taylor_steps

    if current.last_type == 'full':
        # a full step just happened → next step is Taylor (at least try)
        current.type = 'Taylor'
        cache_dic.taylor_step_counter = 1
        cache_dic.check = False
        current.last_layer_error = None
    else:
        first_steps = (
            current.step
            > (current.num_steps - cache_dic.first_enhance - 1)
        )
        reached_max_taylor = (
            cache_dic.taylor_step_counter >= max_taylor_steps
        )

        # ---- Compute threshold: online calibrator > static formula ----
        progress = (current.num_steps - current.step) / current.num_steps
        base_threshold = cache_dic.base_threshold
        decay_rate = cache_dic.decay_rate
        threshold = base_threshold * (decay_rate ** progress)
        threshold = max(threshold, 0.01)

        if calibrator is not None:
            check_layer = cache_dic.check_layer
            # Map step_idx to bucket: step counts DOWN in SpecA
            step_idx = current.num_steps - 1 - current.step
            timestep_bucket = int(step_idx * 3 / current.num_steps) if current.num_steps > 0 else 0
            timestep_bucket = min(timestep_bucket, 2)
            # Use the TTL-cached lookup — speca_cal_type is called once per
            # denoising step (≈50k queries over a 1000-image run) and the
            # EMA threshold moves <1e-4 between adjacent steps, so caching
            # for a few steps is a free 5-10x speedup of the hot path.
            # ``current.step`` is monotonic within a single denoising
            # trajectory (counts down from num_steps-1 to 0), which is all
            # that matters for the TTL freshness check.
            online_thresh = calibrator.get_threshold_cached(
                check_layer, timestep_bucket,
                current_step=current.step,
                default=threshold)
            threshold = online_thresh

        if cache_dic.taylor_step_counter >= min_taylor_steps:
            cache_dic.check = True
        else:
            cache_dic.check = False

        error_too_large = (
            current.last_layer_error is not None
            and current.last_layer_error > threshold
        )

        if first_steps:
            current.type = 'full'
            cache_dic.taylor_step_counter = 0
            cache_dic.full_count += 1
        elif reached_max_taylor:
            current.type = 'full'
            cache_dic.taylor_step_counter = 0
            cache_dic.full_count += 1
        elif error_too_large and cache_dic.check:
            current.type = 'full'
            cache_dic.taylor_step_counter = 0
            cache_dic.full_count += 1
        elif cache_dic.taylor_step_counter < min_taylor_steps:
            current.type = 'Taylor'
            cache_dic.taylor_step_counter += 1
        else:
            current.type = 'Taylor'
            cache_dic.taylor_step_counter += 1

    current.last_type = current.type

    if current.type == 'full':
        cache_dic.cache_counter = 0
        current.activated_steps.append(current.step)
    else:
        cache_dic.cache_counter += 1
