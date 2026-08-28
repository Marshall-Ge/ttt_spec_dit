# -*- coding: utf-8 -*-
"""Serialization + deterministic hashing + COVR accounting pure functions.

No torch model / scheduler dependencies here — only stdlib, numpy and torch
tensor helpers. Functions that need a transformer/scheduler live in
``pipelines/hooks/covr_hook.py``.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch


# ===========================================================================
# JSON-safe cleaning
# ===========================================================================

def _clean(obj, _seen=None):
    """Recursively make dicts JSON-safe."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        return {k: _clean(v, _seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v, _seen) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item()
    return obj


def _covr_canonical_json(value) -> str:
    """Canonical JSON serialization (sorted keys, compact separators)."""
    def normalize(item):
        if isinstance(item, Mapping):
            return {
                str(key): normalize(item[key])
                for key in sorted(item, key=lambda key: str(key))
            }
        if isinstance(item, (list, tuple)):
            return [normalize(entry) for entry in item]
        if isinstance(item, (set, frozenset)):
            entries = [normalize(entry) for entry in item]
            return sorted(
                entries,
                key=lambda entry: json.dumps(
                    entry, sort_keys=True, separators=(",", ":")),
            )
        if isinstance(item, np.generic):
            return item.item()
        if torch.is_tensor(item):
            return item.detach().cpu().tolist()
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"))


def _covr_scheduler_config_json(config) -> str:
    identity = {
        key: value for key, value in dict(config).items()
        if key != "_use_default_values"
    }
    return _covr_canonical_json(identity)


def _covr_version_key(version) -> str:
    """Short hash identifying one COVR runtime configuration."""
    return version.key


# ===========================================================================
# Deterministic hashing (session/trajectory-level sampling)
# ===========================================================================

def _covr_scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().item())
    return float(value)


def _covr_log_snr(scheduler, timestep) -> float:
    """Log-SNR at a timestep (requires scheduler.alphas_cumprod)."""
    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        return 0.0
    index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    alpha = float(alphas_cumprod[index].detach().float().item())
    alpha = min(max(alpha, 1e-8), 1.0 - 1e-8)
    return float(np.log(alpha / (1.0 - alpha)))


def _covr_hash_sample(session_id: str, trajectory_id: int,
                      step_idx: int, rate: float, purpose: str) -> bool:
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    payload = f"{purpose}:{session_id}:{trajectory_id}:{step_idx}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64) < rate


def _covr_hash_index(session_id: str, trajectory_id: int,
                     purpose: str, size: int) -> int:
    if size <= 0:
        raise ValueError("hash index size must be positive")
    payload = f"{purpose}:{session_id}:{trajectory_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % size


def _covr_sentinel_selection(session_id: str, trajectory_id: int,
                             rate: float, horizon: int, num_steps: int,
                             mandatory_prefix: int = 0) -> Tuple[bool, Optional[int]]:
    """Determine whether a trajectory is a sentinel and, if so, where its
    H-step reference rollout starts.

    Shared by the bandit and forced-strategy paths so both make the same
    deterministic decision: hashed by (session_id, trajectory_id) with the
    "delayed_sentinel" purpose string, so a trajectory is selected (or not)
    identically under both modes.

    Returns ``(selected, start_idx)``.
    """
    selected = _covr_hash_sample(
        session_id, trajectory_id, -1, rate, "delayed_sentinel")
    start_idx = None
    if selected and 0 < horizon < num_steps:
        max_start = num_steps - horizon
        min_start = min(int(mandatory_prefix), max_start)
        start_idx = min_start + _covr_hash_index(
            session_id, trajectory_id, "sentinel_start",
            max_start - min_start + 1)
    return selected, start_idx


def _covr_bandit_sentinel_selection(covr_bandit, trajectory_id: int,
                                    rate: float, horizon: int,
                                    num_steps: int) -> Tuple[bool, Optional[int]]:
    """Bandit-path wrapper over ``_covr_sentinel_selection`` (identity logic)."""
    return _covr_sentinel_selection(
        covr_bandit.session_id, trajectory_id, rate, horizon, num_steps,
        int(getattr(covr_bandit.manifest, "mandatory_prefix", 0)))


def _covr_forced_sentinel_selection(covr_session_id: Optional[str],
                                    covr_forced_manifest, trajectory_id: int,
                                    rate: float, horizon: int,
                                    num_steps: int) -> Tuple[bool, Optional[int]]:
    """Forced-strategy wrapper over ``_covr_sentinel_selection`` (identity
    logic; ``covr_session_id`` is None when no COVR mode is active)."""
    return _covr_sentinel_selection(
        str(covr_session_id) if covr_session_id is not None else "",
        trajectory_id, rate, horizon, num_steps,
        int(getattr(covr_forced_manifest, "mandatory_prefix", 0)))


# ===========================================================================
# COVR accounting (pure arithmetic)
# ===========================================================================

def _covr_online_accounting(
        online_wall_times: List[float], safety_wall_times: List[float],
        n_images: int, safety_full_steps: int,
        candidate_flops_T: Optional[float] = None,
        vanilla_flops_T: Optional[float] = None,
        full_step_flops: Optional[float] = None,
        terminal_wall_times: Optional[List[float]] = None,
        terminal_full_steps: int = 0,
        control_wall_times: Optional[List[float]] = None) -> Dict[str, float]:
    if terminal_wall_times is None:
        terminal_wall_times = [0.0] * len(online_wall_times)
    if control_wall_times is None:
        control_wall_times = [0.0] * len(online_wall_times)
    if (len(online_wall_times) != len(safety_wall_times) or
            len(online_wall_times) != len(terminal_wall_times) or
            len(online_wall_times) != len(control_wall_times)):
        raise ValueError("online and feedback wall-time samples must align")
    if not online_wall_times:
        return {}

    candidate_wall_times = [
        max(0.0, online - safety - terminal - control)
        for online, safety, terminal, control in zip(
            online_wall_times, safety_wall_times, terminal_wall_times,
            control_wall_times)
    ]
    online_total = float(sum(online_wall_times))
    safety_total = float(sum(safety_wall_times))
    terminal_total = float(sum(terminal_wall_times))
    control_total = float(sum(control_wall_times))
    candidate_total = float(sum(candidate_wall_times))
    result = {
        "wall_s_candidate_mean": float(np.mean(candidate_wall_times)),
        "wall_s_candidate_std": float(np.std(candidate_wall_times)),
        "wall_s_candidate_total": candidate_total,
        "wall_s_safety_mean": float(np.mean(safety_wall_times)),
        "wall_s_safety_total": safety_total,
        "wall_s_terminal_mean": float(np.mean(terminal_wall_times)),
        "wall_s_terminal_total": terminal_total,
        "wall_s_control_mean": float(np.mean(control_wall_times)),
        "wall_s_control_total": control_total,
        "wall_s_online_mean": float(np.mean(online_wall_times)),
        "wall_s_online_std": float(np.std(online_wall_times)),
        "wall_s_online_total": online_total,
        "speed_candidate_img_per_s": (
            float(n_images / candidate_total) if candidate_total > 0 else 0.0),
        "speed_online_img_per_s": (
            float(n_images / online_total) if online_total > 0 else 0.0),
        "safety_full_steps_mean_per_trajectory": (
            float(safety_full_steps / len(online_wall_times))),
        "terminal_full_steps_mean_per_trajectory": (
            float(terminal_full_steps / len(online_wall_times))),
    }

    if (candidate_flops_T is not None and vanilla_flops_T is not None and
            full_step_flops is not None):
        safety_flops_T = (
            safety_full_steps / len(online_wall_times) * full_step_flops / 1e12)
        terminal_flops_T = (
            terminal_full_steps / len(online_wall_times) * full_step_flops / 1e12)
        online_flops_T = candidate_flops_T + safety_flops_T + terminal_flops_T
        result.update({
            "flops_candidate_T": float(candidate_flops_T),
            "flops_safety_T": float(safety_flops_T),
            "flops_terminal_T": float(terminal_flops_T),
            "flops_online_T": float(online_flops_T),
            "flops_reduction_candidate": (
                1.0 - candidate_flops_T / vanilla_flops_T
                if vanilla_flops_T > 0 else 0.0),
            "flops_reduction_online": (
                1.0 - online_flops_T / vanilla_flops_T
                if vanilla_flops_T > 0 else 0.0),
            "speedup_flops_candidate": (
                vanilla_flops_T / candidate_flops_T
                if candidate_flops_T > 0 else float("nan")),
            "speedup_flops_online": (
                vanilla_flops_T / online_flops_T
                if online_flops_T > 0 else float("nan")),
        })
    return result


def _compute_generated_fid_is(metric) -> Dict[str, float]:
    """Compute FID with real/gen dirs swapped (generated-as-reference)."""
    real_dir, gen_dir = metric.real_dir, metric.gen_dir
    try:
        metric.real_dir, metric.gen_dir = gen_dir, real_dir
        return metric.compute()
    finally:
        metric.real_dir, metric.gen_dir = real_dir, gen_dir


def _dataset_generation_window(dataset_start_index: int, target_samples: int,
                               processed_samples: int,
                               loaded_samples: int) -> Tuple[int, int]:
    if dataset_start_index < 0 or target_samples <= 0 or processed_samples < 0:
        raise ValueError("invalid deterministic dataset window")
    if processed_samples >= target_samples:
        raise ValueError("resume state has consumed the requested dataset slice")
    if loaded_samples < dataset_start_index + target_samples:
        raise ValueError("requested dataset slice exceeds the available dataset")
    return dataset_start_index + processed_samples, target_samples - processed_samples


def _load_forced_covr_template(path: str, template_id: str,
                               version_key: str):
    """Load a forced COVR template manifest and validate version."""
    from accelerators.covr_bandit import TemplateManifest
    manifest = TemplateManifest.load(path)
    if manifest.version_key != version_key:
        raise ValueError(
            "COVR template manifest version does not match the runtime: "
            f"manifest={manifest.version_key}, runtime={version_key}")
    for template in manifest.templates:
        if template.template_id == template_id:
            return manifest, template
    raise ValueError(f"COVR template ID not found in manifest: {template_id}")


def _covr_resume_metadata(path: Optional[str]) -> Optional[Dict[str, object]]:
    """Resume metadata from a persisted bandit state file."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assignments = payload.get("assignments", [])
    if any("sample_count" not in assignment for assignment in assignments):
        raise ValueError(
            "COVR bandit state predates sample-count tracking; start a new "
            "state or provide a separately managed dataset shard")
    sample_counts = [int(assignment["sample_count"]) for assignment in assignments]
    if any(count <= 0 for count in sample_counts):
        raise ValueError("COVR bandit state contains invalid sample counts")
    return {
        "session_id": str(payload.get("session_id", "")),
        "processed_samples": sum(sample_counts),
        "run_identity": dict(payload.get("run_identity", {})),
    }

