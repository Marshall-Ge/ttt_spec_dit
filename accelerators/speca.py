# -*- coding: utf-8 -*-
"""
SpecA (Speculative Acceleration) for PixArt-Alpha and DiT-2-256.

Core mechanics (Cache4Diffusion):
  1. cal_type() decides 'full' or 'Taylor' for each denoising step
  2. Full step: compute all blocks, cache per-block per-submodule
     features + finite-difference derivatives
  3. Taylor step: predict each submodule's output via Taylor series
     (cheap — skips attention/MLP computation)
  4. Last block error check: if Taylor-vs-full error exceeds
     threshold, next step reverts to full
  5. 4 hyperparams: base_threshold, decay_rate, min_taylor_steps, max_taylor_steps

Model-type dispatch:
  - PixArt-α: 3 submodules (attn1, attn2, ff), ada_norm_single
  - DiT-2-256: 2 submodules (attn1, ff), ada_norm_zero
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Precomputed 1 / n! for Taylor coefficients (up to order 6)
_INV_FACTORIAL = [1.0, 1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0]


# ===========================================================================
# Taylor utilities (from speca-dit/taylor_utils/__init__.py)
# ===========================================================================

def derivative_approximation(cache_dic: Dict, current: Dict,
                              feature: torch.Tensor) -> None:
    """Compute finite-difference derivative approximation."""
    difference_distance = (
        current['activated_steps'][-1] - current['activated_steps'][-2]
    )

    updated_taylor_factors = [feature]  # order 0

    prev_cache = cache_dic['cache'][-1][current['layer']][current['module']]
    still_enhancing = (
        current['step']
        < (current['num_steps'] - cache_dic['first_enhance'] + 1)
    )

    for i in range(cache_dic['max_order']):
        if i < len(prev_cache) and still_enhancing:
            updated_taylor_factors.append(
                (updated_taylor_factors[i] - prev_cache[i]) / difference_distance
            )
        else:
            break

    cache_dic['cache'][-1][current['layer']][current['module']] = \
        updated_taylor_factors


def taylor_cache_init(cache_dic: Dict, current: Dict) -> None:
    """Allocate a per-module cache slot on the last full step."""
    if current['step'] == (current['num_steps'] - 1):
        cache_dic['cache'][-1][current['layer']][current['module']] = []


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


# ---- cache_step (trivial tensor ops, no compile needed) ----

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

def speca_cache_init(
    num_steps: int,
    base_threshold: float,
    decay_rate: float,
    min_taylor_steps: int,
    max_taylor_steps: int,
    max_order: int = 4,
    num_layers: int = 28,
    error_metric: str = 'cosine_similarity',
    check_layer: int = 27,
) -> Tuple[Dict, Dict]:
    """Allocate the SpecA cache dictionary and current-state dict.

    Parameters
    ----------
    num_layers : int
        Number of transformer blocks (28 for both PixArt and DiT).
    """
    cache_dic: Dict = {}
    cache: Dict = {}
    cache[-1] = {}

    for j in range(num_layers):
        cache[-1][j] = {}

    for i in range(num_steps):
        cache[i] = {}
        for j in range(num_layers):
            cache[i][j] = {}

    cache_dic['cache'] = cache
    cache_dic['flops'] = 0.0
    cache_dic['max_order'] = max_order
    cache_dic['test_FLOPs'] = False
    cache_dic['first_enhance'] = 3
    cache_dic['cache_counter'] = 0
    cache_dic['taylor_step_counter'] = 0
    cache_dic['check'] = False
    cache_dic['base_threshold'] = base_threshold
    cache_dic['decay_rate'] = decay_rate
    cache_dic['min_taylor_steps'] = min_taylor_steps
    cache_dic['max_taylor_steps'] = max_taylor_steps
    cache_dic['error_metric'] = error_metric
    cache_dic['check_layer'] = check_layer

    current: Dict = {}
    current['last_layer_error'] = 0.0
    current['num_steps'] = num_steps
    current['activated_steps'] = [num_steps - 1]  # start from last step
    current['last_type'] = 'None'
    current['step'] = 0
    current['module'] = None
    current['layer'] = 0

    return cache_dic, current


# ===========================================================================
# Step-type decision
# ===========================================================================

def speca_cal_type(cache_dic: Dict, current: Dict) -> None:
    """Decide whether the current step is 'full' or 'Taylor'."""
    min_taylor_steps = cache_dic['min_taylor_steps']
    max_taylor_steps = cache_dic['max_taylor_steps']

    if 'full_count' not in cache_dic:
        cache_dic['full_count'] = 0

    if current['last_type'] == 'full':
        current['type'] = 'Taylor'
        cache_dic['taylor_step_counter'] = 1
        cache_dic['check'] = False
        current['last_layer_error'] = None
    else:
        first_steps = (
            current['step']
            > (current['num_steps'] - cache_dic['first_enhance'] - 1)
        )
        reached_max_taylor = (
            cache_dic['taylor_step_counter'] >= max_taylor_steps
        )
        progress = (current['num_steps'] - current['step']) / current['num_steps']
        base_threshold = cache_dic['base_threshold']
        decay_rate = cache_dic['decay_rate']
        threshold = base_threshold * (decay_rate ** progress)
        threshold = max(threshold, 0.01)

        if cache_dic['taylor_step_counter'] >= min_taylor_steps:
            cache_dic['check'] = True
        else:
            cache_dic['check'] = False

        error_too_large = (
            current.get('last_layer_error') is not None
            and current.get('last_layer_error') > threshold
        )

        if first_steps:
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1
        elif reached_max_taylor:
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1
        elif error_too_large and cache_dic['check']:
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1
        elif cache_dic['taylor_step_counter'] < min_taylor_steps:
            current['type'] = 'Taylor'
            cache_dic['taylor_step_counter'] += 1
        else:
            current['type'] = 'Taylor'
            cache_dic['taylor_step_counter'] += 1

    current['last_type'] = current['type']

    if current['type'] == 'full':
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
    else:
        cache_dic['cache_counter'] += 1


# ===========================================================================
#  SpecA Controller
# ===========================================================================

class SpecAController:
    """Stateful SpecA controller holding cache_dic, current, decisions.

    Parameters
    ----------
    num_steps : int
    base_threshold : float
    decay_rate : float
    min_taylor_steps : int
    max_taylor_steps : int
    max_order : int
    """

    def __init__(
        self,
        num_steps: int = 20,
        base_threshold: float = 0.01,
        decay_rate: float = 0.01,
        min_taylor_steps: int = 1,
        max_taylor_steps: int = 2,
        max_order: int = 4,
        error_metric: str = 'cosine_similarity',
        check_layer: int = 27,
    ):
        self.num_steps = num_steps
        self.base_threshold = base_threshold
        self.decay_rate = decay_rate
        self.min_taylor_steps = min_taylor_steps
        self.max_taylor_steps = max_taylor_steps
        self.max_order = max_order
        self.error_metric = error_metric
        self.check_layer = check_layer

        self.cache_dic, self.current = speca_cache_init(
            num_steps, base_threshold, decay_rate,
            min_taylor_steps, max_taylor_steps, max_order,
            error_metric=error_metric,
            check_layer=check_layer,
        )
        self.cnt: int = 0
        self.decisions: List[str] = []
        self._type_history: List[str] = []

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the step counter (call once per denoising step)."""
        self.cnt += 1

    def record_decision(self, step_type: str) -> None:
        """Record one step's decision ('full' or 'Taylor')."""
        self._type_history.append(step_type)
        self.decisions.append('calc' if step_type == 'full' else 'skip')

    def reset(self) -> None:
        """Reset state for a new generation."""
        self.cache_dic, self.current = speca_cache_init(
            self.num_steps, self.base_threshold, self.decay_rate,
            self.min_taylor_steps, self.max_taylor_steps, self.max_order,
            error_metric=self.error_metric,
            check_layer=self.check_layer,
        )
        self.cnt = 0
        self.decisions = []
        self._type_history = []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        n_calc = sum(1 for d in self.decisions if d == 'calc')
        n_skip = sum(1 for d in self.decisions if d == 'skip')
        total = n_calc + n_skip
        return {
            'total_full': n_calc,
            'total_taylor': n_skip,
            'total_steps': total,
            'skip_ratio': n_skip / total if total > 0 else 0.0,
            'base_threshold': self.base_threshold,
            'decay_rate': self.decay_rate,
            'min_taylor_steps': self.min_taylor_steps,
            'max_taylor_steps': self.max_taylor_steps,
            'max_order': self.max_order,
            'error_metric': self.error_metric,
        }


# ===========================================================================
#  Patched forward — PixArt-α (3 submodules)
# ===========================================================================

def make_speca_forward_pixart(controller: SpecAController):
    """Build a PixArt forward function with SpecA acceleration."""

    def speca_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional "
                "conditions for `adaln_single`."
            )

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        cache_dic = controller.cache_dic
        current = controller.current

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep_emb, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # ===============================================================
        # SpecA decision + block loop
        # ===============================================================
        current['step'] = current['num_steps'] - 1 - controller.cnt
        speca_cal_type(cache_dic, current)
        controller.record_decision(current['type'])

        for layer_idx, block in enumerate(self.transformer_blocks):
            current['layer'] = layer_idx

            B = hidden_states.shape[0]
            proj = block.scale_shift_table[None] + timestep_emb.reshape(B, 6, -1)
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = proj.chunk(6, dim=1)

            cross_kw = {} if cross_attention_kwargs is None else cross_attention_kwargs

            # -----------------------------------------------------------
            #  FULL step
            # -----------------------------------------------------------
            if current['type'] == 'full':
                # --- Submodule 1: attn1 (self-attention, modulated) ---
                current['module'] = 'attn1'
                taylor_cache_init(cache_dic, current)
                norm_hidden = block.norm1(hidden_states)
                modulated = norm_hidden * (1 + scale_msa) + shift_msa
                attn1_out = block.attn1(
                    modulated, attention_mask=attention_mask, **cross_kw,
                )
                derivative_approximation(cache_dic, current, attn1_out)
                hidden_states = hidden_states + gate_msa * attn1_out

                # --- Submodule 2: attn2 (cross-attention, NO norm2/modulation/gate) ---
                # PixArt ada_norm_single: attn2 takes raw hidden_states directly
                current['module'] = 'attn2'
                taylor_cache_init(cache_dic, current)
                attn2_out = block.attn2(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_kw,
                )
                derivative_approximation(cache_dic, current, attn2_out)
                hidden_states = hidden_states + attn2_out  # no gate

                # --- Submodule 3: ff (feed-forward, modulated, reuses norm2) ---
                current['module'] = 'ff'
                taylor_cache_init(cache_dic, current)
                norm_hidden = block.norm2(hidden_states)
                modulated = norm_hidden * (1 + scale_mlp) + shift_mlp
                ff_out = block.ff(modulated)
                derivative_approximation(cache_dic, current, ff_out)
                hidden_states = hidden_states + gate_mlp * ff_out

            # -----------------------------------------------------------
            #  TAYLOR step
            # -----------------------------------------------------------
            elif current['type'] == 'Taylor':
                check_layer = cache_dic.get('check_layer', 27)
                if layer_idx == check_layer and cache_dic['check']:
                    full_hidden = hidden_states.clone()

                distance = current['step'] - current['activated_steps'][-1]

                hidden_states = cache_step_pixart(
                    hidden_states,
                    cache_dic['cache'][-1][layer_idx]['attn1'],
                    cache_dic['cache'][-1][layer_idx]['attn2'],
                    cache_dic['cache'][-1][layer_idx]['ff'],
                    gate_msa, gate_mlp, distance,
                )

                # Error check at configured layer
                if layer_idx == check_layer and cache_dic['check']:
                    norm_hidden = block.norm1(full_hidden)
                    modulated = norm_hidden * (1 + scale_msa) + shift_msa
                    attn1_full = block.attn1(
                        modulated, attention_mask=attention_mask, **cross_kw,
                    )
                    full_hidden = full_hidden + gate_msa * attn1_full

                    # PixArt ada_norm_single: attn2 takes raw hidden_states directly (no norm2)
                    attn2_full = block.attn2(
                        full_hidden,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_kw,
                    )
                    full_hidden = full_hidden + attn2_full

                    norm_hidden = block.norm2(full_hidden)
                    modulated = norm_hidden * (1 + scale_mlp) + shift_mlp
                    ff_full = block.ff(modulated)
                    full_hidden = full_hidden + gate_mlp * ff_full

                    gate_value, _ = compute_error_gate(
                        hidden_states, full_hidden,
                        metric=cache_dic['error_metric'],
                    )
                    current['last_layer_error'] = gate_value

        controller.step()
        # ===============================================================

        # 3. Output (norm_out + proj_out + unpatchify) — always runs
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = (
            hidden_states * (1 + scale.to(hidden_states.device))
            + shift.to(hidden_states.device)
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size,
                   self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   height * self.config.patch_size,
                   width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)

    return speca_forward


# ===========================================================================
#  Patched forward — DiT-2-256 (2 submodules, ada_norm_zero)
# ===========================================================================

def make_speca_forward_dit(controller: SpecAController):
    """Build a DiT forward function with SpecA acceleration.

    DiT block has 2 submodules (no cross-attention):
      1. attn1: norm1(h, t, class_labels) → attn1(modulated) → *gate_msa + h
      2. ff:    norm3(h) → modulate(shift_mlp, scale_mlp) → ff(modulated) → *gate_mlp + h
    """

    def speca_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[Dict] = None,
        return_dict: bool = True,
    ):
        cache_dic = controller.cache_dic
        current = controller.current

        # 1. Input
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        # ===============================================================
        # SpecA decision + block loop
        # ===============================================================
        current['step'] = current['num_steps'] - 1 - controller.cnt
        speca_cal_type(cache_dic, current)
        controller.record_decision(current['type'])

        for layer_idx, block in enumerate(self.transformer_blocks):
            current['layer'] = layer_idx

            # ===============================================================
            #  FULL step
            # ===============================================================
            if current['type'] == 'full':
                # --- Submodule 1: attn1 (self-attention, modulated via norm1) ---
                current['module'] = 'attn1'
                taylor_cache_init(cache_dic, current)
                norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                    hidden_states, timestep=timestep, class_labels=class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
                # norm_hidden is already modulated: norm(h) * (1+scale_msa) + shift_msa
                attn1_out = block.attn1(
                    norm_hidden, attention_mask=None,
                    **({} if cross_attention_kwargs is None else cross_attention_kwargs),
                )
                derivative_approximation(cache_dic, current, attn1_out)
                hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn1_out

                # --- Submodule 2: ff (feed-forward, modulated via norm3 + shift_mlp/scale_mlp) ---
                # gate_msa, shift_mlp, scale_mlp, gate_mlp already from norm1 above
                current['module'] = 'ff'
                taylor_cache_init(cache_dic, current)
                norm_hidden = block.norm3(hidden_states)
                modulated = norm_hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                ff_out = block.ff(modulated)
                derivative_approximation(cache_dic, current, ff_out)
                hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_out

            # ===============================================================
            #  TAYLOR step
            # ===============================================================
            elif current['type'] == 'Taylor':
                # Compute modulation params for Taylor (needed for gates + ff modulation)
                _norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                    hidden_states, timestep=timestep, class_labels=class_labels,
                    hidden_dtype=hidden_states.dtype,
                )

                check_layer = cache_dic.get('check_layer', 27)
                if layer_idx == check_layer and cache_dic['check']:
                    full_hidden = hidden_states.clone()

                distance = current['step'] - current['activated_steps'][-1]

                hidden_states = cache_step_dit(
                    hidden_states,
                    cache_dic['cache'][-1][layer_idx]['attn1'],
                    cache_dic['cache'][-1][layer_idx]['ff'],
                    gate_msa, gate_mlp, distance,
                )

                # Error check at configured layer
                if layer_idx == check_layer and cache_dic['check']:
                    # Compute full attn1 on saved full_hidden
                    norm_hidden, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = block.norm1(
                        full_hidden, timestep=timestep, class_labels=class_labels,
                        hidden_dtype=full_hidden.dtype,
                    )
                    attn1_full = block.attn1(
                        norm_hidden, attention_mask=None,
                        **({} if cross_attention_kwargs is None else cross_attention_kwargs),
                    )
                    full_hidden = full_hidden + _gate_msa.unsqueeze(1) * attn1_full

                    # Compute full ff
                    norm_hidden = block.norm3(full_hidden)
                    modulated = norm_hidden * (1 + _scale_mlp[:, None]) + _shift_mlp[:, None]
                    ff_full = block.ff(modulated)
                    full_hidden = full_hidden + _gate_mlp.unsqueeze(1) * ff_full

                    gate_value, _ = compute_error_gate(
                        hidden_states, full_hidden,
                        metric=cache_dic['error_metric'],
                    )
                    current['last_layer_error'] = gate_value

        controller.step()
        # ===============================================================

        # 3. Output (norm_out + proj_out_1/2 + unpatchify) — always runs
        from torch.nn.functional import silu
        conditioning = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size,
                   self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   height * self.patch_size, width * self.patch_size)
        )

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)

    return speca_forward


# ===========================================================================
#  Install / uninstall
# ===========================================================================

def install_speca(transformer, controller: SpecAController, model_type: str = "pixart"):
    """Monkeypatch ``transformer.forward`` with the SpecA-augmented version.

    Returns the original forward so it can be restored.
    """
    if model_type == "dit":
        forward_fn = make_speca_forward_dit(controller)
    else:
        forward_fn = make_speca_forward_pixart(controller)
    original_forward = transformer.forward
    transformer.forward = forward_fn.__get__(transformer, type(transformer))
    return original_forward


def uninstall_speca(transformer, original_forward):
    """Restore the original ``transformer.forward``."""
    transformer.forward = original_forward


# ===========================================================================
#  SpecAAccelerator (auto-detects model type)
# ===========================================================================

class SpecAAccelerator:
    """Accelerator wrapper for SpecA.

    Auto-detects generator type (PixArt vs DiT) and installs the
    correct forward replacement.

    Parameters
    ----------
    num_steps : int
    base_threshold : float
    decay_rate : float
    min_taylor_steps : int
    max_taylor_steps : int
    max_order : int
    error_metric : str
        Error metric for gate/threshold comparison.
        One of: l1, l2, relative_l1, relative_l2, cosine_similarity, all.
        When 'all', relative_l1 is used as the gate value.
    """

    def __init__(
        self,
        num_steps: int = 20,
        base_threshold: float = 0.01,
        decay_rate: float = 0.01,
        min_taylor_steps: int = 1,
        max_taylor_steps: int = 2,
        max_order: int = 4,
        error_metric: str = 'cosine_similarity',
        check_layer: int = None,
    ):
        self.num_steps = num_steps
        self.base_threshold = base_threshold
        self.decay_rate = decay_rate
        self.min_taylor_steps = min_taylor_steps
        self.max_taylor_steps = max_taylor_steps
        self.max_order = max_order
        self.error_metric = error_metric
        self._check_layer_arg = check_layer  # resolved in install()
        self._check_layer = check_layer
        self._controller: Optional[SpecAController] = None
        self._original_forward = None
        self._generator = None
        self._model_type = None

    @property
    def controller(self) -> SpecAController:
        """Lazy-create the SpecA controller."""
        if self._controller is None:
            self._controller = SpecAController(
                num_steps=self.num_steps,
                base_threshold=self.base_threshold,
                decay_rate=self.decay_rate,
                min_taylor_steps=self.min_taylor_steps,
                max_taylor_steps=self.max_taylor_steps,
                max_order=self.max_order,
                error_metric=self.error_metric,
                check_layer=self._check_layer or 27,
            )
        return self._controller

    @property
    def speca(self) -> SpecAController:
        """Alias for FLOPs tracking (``.decisions`` required)."""
        return self.controller

    def _detect_model_type(self, generator) -> str:
        """Detect the model type from the generator instance."""
        from models.pixart import PixArtGenerator
        from models.dit import DiTGenerator
        if isinstance(generator, DiTGenerator):
            return "dit"
        elif isinstance(generator, PixArtGenerator):
            return "pixart"
        else:
            raise TypeError(
                f"Unknown generator type: {type(generator).__name__}. "
                f"Expected PixArtGenerator or DiTGenerator.")

    def install(self, generator) -> None:
        """Install SpecA onto a generator (PixArt or DiT).

        Auto-detects the optimal check_layer:
        - DiT: layer 20 (gate_mlp ≈ 2.3, catches FF errors that layer 27 masks)
        - PixArt: layer 27 (standard last-block behavior)
        """
        self._model_type = self._detect_model_type(generator)
        self._generator = generator

        # Auto-detect check_layer if not explicitly set
        if self._check_layer_arg is None:
            if self._model_type == 'dit':
                self._check_layer = 20  # hi-gate layer for DiT (gate_mlp≈2.3)
            else:
                self._check_layer = 24  # PixArt: highest gate_mlp (1.29 vs 0.81 at L27)

        self._original_forward = install_speca(
            generator.transformer, self.controller, model_type=self._model_type)

    def uninstall(self) -> None:
        """Restore original transformer.forward."""
        if self._generator is not None and self._original_forward is not None:
            uninstall_speca(self._generator.transformer, self._original_forward)
            self._original_forward = None

    @property
    def stats(self) -> dict:
        if self._controller is None:
            return {}
        return self._controller.stats()

    def reset(self) -> None:
        """Reset SpecA state for a new generation."""
        if self._controller is not None:
            self._controller.reset()
