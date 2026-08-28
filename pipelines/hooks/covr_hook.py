# -*- coding: utf-8 -*-
"""COVR sampling-loop hooks — functions that need transformer/scheduler access.

Pure helpers that only touch torch tensors / models (no run-dit logic) live
here so the orchestrator (run_dit.py / pipelines/) only calls these hooks.
Deterministic hashing / accounting pure functions live in
``utils/serialization.py``.

Import convention: orchestrators do
    from pipelines.hooks.covr_hook import _covr_shadow_full, ...
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity with old module)

from accelerators.covr import (
    ActionAuditContext,
    ddim_epsilon_transition_coefficients,
)
from utils.serialization import _covr_log_snr, _covr_scalar


def _cache_scheduler_timestep_values(scheduler) -> None:
    """Cache the host timestep values on the scheduler object."""
    timesteps = scheduler.timesteps
    if torch.is_tensor(timesteps):
        values = timesteps.detach().to("cpu").tolist()
    else:
        values = timesteps
    setattr(scheduler, "_host_timestep_values", tuple(
        int(timestep) for timestep in values))


def _covr_scheduler_alphas(scheduler, timesteps, step_idx: int, timestep):
    """(alpha_t, alpha_prev) pair for a DDIM epsilon-prediction scheduler."""
    config = getattr(scheduler, "config", None)
    prediction_type = getattr(config, "prediction_type", None)
    if prediction_type is None and hasattr(config, "get"):
        prediction_type = config.get("prediction_type")
    if prediction_type != "epsilon":
        raise ValueError(
            "COVR action audits require a DDIM epsilon-prediction scheduler")

    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        raise ValueError("COVR action audits require scheduler alphas_cumprod")
    timestep_index = int(_covr_scalar(timestep))
    alpha_t = _covr_scalar(alphas_cumprod[timestep_index])

    if step_idx + 1 < len(timesteps):
        previous_index = int(_covr_scalar(timesteps[step_idx + 1]))
        alpha_prev = _covr_scalar(alphas_cumprod[previous_index])
    else:
        final_alpha = getattr(scheduler, "final_alpha_cumprod", None)
        if final_alpha is None:
            raise ValueError(
                "COVR action audits require scheduler final_alpha_cumprod")
        alpha_prev = _covr_scalar(final_alpha)
    return alpha_t, alpha_prev


def _covr_scheduler_pair(scheduler, noise_approx, noise_full, timestep, latents):
    """Scheduler-step both noise predictions → (x_prev_approx, x_prev_full)."""
    approx_scheduler = copy.deepcopy(scheduler)
    full_scheduler = copy.deepcopy(scheduler)
    x_prev_approx = approx_scheduler.step(
        noise_approx.detach(), timestep, latents.detach(), return_dict=False)[0]
    x_prev_full = full_scheduler.step(
        noise_full.detach(), timestep, latents.detach(), return_dict=False)[0]
    return x_prev_approx, x_prev_full


def _covr_transition_components(x_prev_approx, x_prev_full, x_t):
    """Transition-defect numerators/denominators (sqrt-MSE per sample)."""
    approx = x_prev_approx.detach().float().flatten(1)
    full = x_prev_full.detach().float().flatten(1)
    current = x_t.detach().float().flatten(1)
    numerator = (approx - full).square().mean(dim=1).sqrt()
    denominator = (full - current).square().mean(dim=1).sqrt()
    return numerator, denominator


def _covr_shadow_full(transformer, latent_input, timestep, class_labels,
                      guidance_scale):
    """One unaccelerated (full) forward as counterfactual reference."""
    if guidance_scale > 1.0:
        return transformer.forward_with_cfg(
            latent_input, timestep,
            current=None, cache_dic=None, teacache_state=None,
            class_labels=class_labels, cfg_scale=guidance_scale,
        )
    return transformer(
        latent_input, timestep=timestep,
        current=None, cache_dic=None, teacache_state=None,
        class_labels=class_labels, return_dict=False,
    )[0]


def _covr_teacache_terminal_skip(
        transformer, latent_input, timestep, class_labels, guidance_scale,
        teacache_state):
    """Evaluate one isolated forced-skip candidate from the stale residual."""
    shadow_state = dict(teacache_state)
    for key, value in teacache_state.items():
        if isinstance(value, list):
            shadow_state[key] = list(value)
    boundary_probe = teacache_state.get("boundary_probe")
    if boundary_probe is not None:
        shadow_state["boundary_probe"] = dict(boundary_probe)
        shadow_state["boundary_probe"]["rows"] = list(
            boundary_probe.get("rows", ()))

    num_steps = int(shadow_state["num_steps"])
    cnt = int(shadow_state["cnt"])
    refresh_mask = [True] * num_steps
    refresh_mask[cnt] = False
    shadow_state["refresh_mask"] = tuple(refresh_mask)

    if guidance_scale > 1.0:
        return transformer.forward_with_cfg(
            latent_input, timestep,
            current=None, cache_dic=None, teacache_state=shadow_state,
            class_labels=class_labels, cfg_scale=guidance_scale,
        )
    return transformer(
        latent_input, timestep=timestep,
        current=None, cache_dic=None, teacache_state=shadow_state,
        class_labels=class_labels, return_dict=False,
    )[0]


def _covr_full_rollout(transformer, scheduler, timesteps, start_idx: int,
                       horizon: int, latents, class_labels, guidance_scale,
                       in_channels: int):
    """Unaccelerated H-step rollout from ``start_idx`` (sentinel reference)."""
    if horizon <= 0 or start_idx < 0 or start_idx + horizon > len(timesteps):
        raise ValueError("invalid COVR sentinel rollout interval")
    branch_scheduler = copy.deepcopy(scheduler)
    branch_latents = latents.detach().clone()
    for branch_idx in range(start_idx, start_idx + horizon):
        timestep = timesteps[branch_idx]
        timestep_batch = timestep.expand(branch_latents.shape[0]).to(torch.int64)
        latent_input = branch_scheduler.scale_model_input(
            branch_latents, timestep)
        noise_pred = _covr_shadow_full(
            transformer, latent_input, timestep_batch, class_labels,
            guidance_scale)
        noise_pred = noise_pred[:, :in_channels]
        branch_latents = branch_scheduler.step(
            noise_pred, timestep, branch_latents, return_dict=False)[0]
    return branch_latents


def _covr_context(recorder, scheduler, timesteps, step_idx, timestep,
                  current):
    """Build an ActionAuditContext for one batch-step."""
    if current.activated_steps:
        distance = abs(current.step - current.activated_steps[-1])
    else:
        distance = 0
    alpha_t, alpha_prev = _covr_scheduler_alphas(
        scheduler, timesteps, step_idx, timestep)
    latent_coefficient, model_output_coefficient = (
        ddim_epsilon_transition_coefficients(alpha_t, alpha_prev))
    return ActionAuditContext(
        step_idx=step_idx,
        num_steps=len(timesteps),
        timestep=_covr_scalar(timestep),
        log_snr=_covr_log_snr(scheduler, timestep),
        alpha_t=alpha_t,
        alpha_prev=alpha_prev,
        latent_coefficient=latent_coefficient,
        model_output_coefficient=model_output_coefficient,
        distance_since_refresh=distance,
        previous_defect_mean=(
            recorder.previous_defect if recorder is not None else 0.0),
    )


__all__ = [
    "_cache_scheduler_timestep_values",
    "_covr_context",
    "_covr_full_rollout",
    "_covr_scheduler_alphas",
    "_covr_scheduler_pair",
    "_covr_shadow_full",
    "_covr_teacache_terminal_skip",
    "_covr_transition_components",
]
