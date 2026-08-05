# -*- coding: utf-8 -*-
"""COVR-v2 viability probe recording.

This module is deliberately outside the accelerator decision path.  It records
small, causal scalar summaries from a verified shared prefix so an offline
analysis can test whether a per-image policy has anything learnable to use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


SCHEMA_VERSION = 1


def _mask_hash(refresh_mask: Sequence[bool]) -> str:
    payload = json.dumps([bool(v) for v in refresh_mask], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _scalar(value: torch.Tensor) -> float:
    result = float(value.detach().float().cpu().item())
    if not math.isfinite(result):
        raise ValueError("viability probe observed a non-finite feature")
    return result


def _tensor_stats(value: torch.Tensor, prefix: str) -> Dict[str, float]:
    value = value.detach().float()
    flat = value.reshape(value.shape[0], -1)
    return {
        f"{prefix}_mean": _scalar(value.mean()),
        f"{prefix}_std": _scalar(value.std(unbiased=False)),
        f"{prefix}_l1": _scalar(flat.abs().mean()),
        f"{prefix}_l2": _scalar(flat.square().mean().sqrt()),
    }


def extract_prefix_features(
    latent_input: torch.Tensor,
    noise_pred: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    """Return per-image-safe scalar features for the first ``batch_size`` rows."""
    if not isinstance(latent_input, torch.Tensor) or not isinstance(noise_pred, torch.Tensor):
        raise TypeError("viability features require torch tensors")
    if batch_size != 1:
        raise ValueError("viability probe requires batch_size=1")
    if latent_input.ndim < 2 or noise_pred.ndim < 2:
        raise ValueError("viability tensors must have a batch dimension")
    if latent_input.shape[0] < batch_size or noise_pred.shape[0] < batch_size:
        raise ValueError("viability tensors are smaller than batch_size")

    latent = latent_input[:batch_size]
    noise = noise_pred[:batch_size]
    delta = noise - latent
    features = {}
    features.update(_tensor_stats(latent, "latent"))
    features.update(_tensor_stats(noise, "noise"))
    features.update(_tensor_stats(delta, "delta"))
    latent_flat = latent.detach().float().reshape(batch_size, -1)
    noise_flat = noise.detach().float().reshape(batch_size, -1)
    features["latent_noise_cosine"] = _scalar(
        torch.nn.functional.cosine_similarity(latent_flat, noise_flat, dim=1).mean()
    )
    features["delta_relative_l1"] = _scalar(
        delta.detach().float().abs().mean()
        / latent.detach().float().abs().mean().clamp_min(1e-8)
    )
    return features


class COVRViabilityRecorder:
    """Write causal prefix observations as one JSON object per step.

    The recorder is opt-in: callers construct it only when an output path is
    supplied.  A trajectory must be opened and closed in order, which catches
    accidental cross-image or cross-arm feature mixing.
    """

    def __init__(
        self,
        output_path: str,
        prefix_steps: int = 3,
        *,
        run_identity: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not output_path:
            raise ValueError("output_path must be non-empty")
        if prefix_steps <= 0:
            raise ValueError("prefix_steps must be positive")
        self.output_path = os.path.abspath(output_path)
        self.prefix_steps = int(prefix_steps)
        self.run_identity = dict(run_identity or {})
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")
        self._active: Optional[Dict[str, Any]] = None
        self._next_step = 0
        self.record_count = 0
        self.trajectory_count = 0
        self._closed = False

    @property
    def active(self) -> bool:
        return self._active is not None

    def begin_trajectory(
        self,
        *,
        global_idx: int,
        latent_seed: int,
        strategy_id: str,
        manifest_hash: str,
        refresh_mask: Sequence[bool],
        batch_size: int = 1,
        sample_id: Optional[str] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("viability recorder is closed")
        if self._active is not None:
            raise RuntimeError("previous viability trajectory is still active")
        if batch_size != 1:
            raise ValueError("viability probe requires batch_size=1")
        mask = tuple(refresh_mask)
        if len(mask) < self.prefix_steps:
            raise ValueError("refresh mask is shorter than viability prefix")
        if any(type(value) is not bool for value in mask):
            raise ValueError("refresh mask values must be booleans")
        if not all(mask[: self.prefix_steps]):
            raise ValueError("refresh mask does not verify the shared prefix")
        self._active = {
            "global_idx": int(global_idx),
            "latent_seed": int(latent_seed),
            "strategy_id": str(strategy_id),
            "manifest_hash": str(manifest_hash),
            "refresh_mask_hash": _mask_hash(mask),
            "prefix_steps": self.prefix_steps,
            "prefix_verified": True,
            "sample_id": None if sample_id is None else str(sample_id),
            "batch_size": int(batch_size),
        }
        self._next_step = 0
        self.trajectory_count += 1

    def record_step(
        self,
        *,
        step_idx: int,
        timestep: int,
        latent_input: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> None:
        if self._active is None:
            raise RuntimeError("begin_trajectory must precede record_step")
        step_idx = int(step_idx)
        if step_idx != self._next_step:
            raise ValueError(
                f"viability prefix step must be sequential: expected {self._next_step}, got {step_idx}"
            )
        if step_idx >= self.prefix_steps:
            raise ValueError("cannot record a step outside the configured prefix")
        features = extract_prefix_features(
            latent_input, noise_pred, self._active["batch_size"]
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "feature_source": "causal_prefix",
            **self.run_identity,
            **self._active,
            "step_idx": step_idx,
            "timestep": int(timestep),
            "features": features,
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()
        self.record_count += 1
        self._next_step += 1

    def end_trajectory(self) -> None:
        if self._active is None:
            raise RuntimeError("no active viability trajectory")
        if self._next_step != self.prefix_steps:
            raise ValueError(
                f"incomplete viability prefix: {self._next_step}/{self.prefix_steps} steps"
            )
        self._active = None
        self._next_step = 0

    def close(self) -> None:
        if self._closed:
            return
        if self._active is not None:
            raise RuntimeError("cannot close viability recorder with active trajectory")
        self._file.flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> "COVRViabilityRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "COVRViabilityRecorder",
    "SCHEMA_VERSION",
    "extract_prefix_features",
]
