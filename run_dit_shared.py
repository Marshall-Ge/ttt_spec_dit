# -*- coding: utf-8 -*-
"""Pure lifecycle helpers shared between run_dit and the COVR runtime.

These helpers are intentionally free of model/runner dependencies so the
runtime boundary (``accelerators/covr_runtime.py``) can construct versions,
resume metadata, and profilers without importing the DiT orchestrator.

``run_dit`` imports the names from here and re-exports them so existing
callers (``scripts/benchmark_safety_proxy.py``, ``tests/*``) keep importing
the historical ``_covr_*`` / ``_GenerationProfiler`` names from ``run_dit``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch


def _load_covr_version_cls():
    """Import COVRVersion lazily to break the package import cycle.

    ``run_dit_shared`` is imported by the COVR runtime while the
    ``accelerators`` package is still initializing (``accelerators.covr`` is
    itself loading), so a module-level ``from accelerators.covr import
    COVRVersion`` would resolve against the half-initialized package. The
    version class is only needed at call time, never at import time.
    """
    from accelerators.covr import COVRVersion
    return COVRVersion


def _covr_canonical_json(value) -> str:
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


def _covr_resume_metadata(path: Optional[str]) -> Optional[Dict[str, object]]:
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

    Returns ``(selected, start_idx)`` where:
      * ``selected`` is ``_covr_hash_sample(session_id, trajectory_id, -1,
        rate, "delayed_sentinel")`` (strict 0.0/1.0 short-circuits, same as
        the hash helper);
      * ``start_idx`` is None when ``selected`` is False, ``horizon <= 0``,
        or ``horizon >= num_steps`` (no interior H-step window);
      * otherwise ``start_idx`` is a deterministic random index inside
        ``[min(mandatory_prefix, max_start), max_start]`` with
        ``max_start = num_steps - horizon`` (the bandit's existing formula).
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


class _GenerationProfiler:
    def __init__(self, device, detailed: bool = False):
        self.device = torch.device(device)
        self.detailed = detailed
        self._use_cuda = (
            self.device.type == "cuda" and torch.cuda.is_available())
        self._gpu_intervals = defaultdict(list)
        self._cpu_seconds = defaultdict(float)
        self._synchronized = not self._use_cuda

    def start_gpu(self, stage: str, required: bool = False):
        if not (required or self.detailed):
            return None
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            return stage, start, None
        return stage, None, time.perf_counter()

    def stop_gpu(self, token, stage: Optional[str] = None) -> None:
        if token is None:
            return
        initial_stage, start_event, start_time = token
        stage = stage or initial_stage
        if self._use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._gpu_intervals[stage].append((start_event, end_event))
            self._synchronized = False
        else:
            self._cpu_seconds[stage] += time.perf_counter() - start_time

    def add_cpu(self, stage: str, seconds: float) -> None:
        self._cpu_seconds[stage] += float(seconds)

    def synchronize(self) -> float:
        if not self._use_cuda or self._synchronized:
            return 0.0
        start = time.perf_counter()
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        self._cpu_seconds["cuda_sync_wait"] += elapsed
        self._cpu_seconds["cuda_sync_calls"] += 1.0
        self._synchronized = True
        return elapsed

    def reset(self) -> None:
        self._gpu_intervals.clear()
        self._cpu_seconds.clear()
        self._synchronized = not self._use_cuda

    def seconds(self, stage: str) -> float:
        if self._use_cuda and not self._synchronized:
            raise RuntimeError("GPU timings require one batch-boundary synchronize")
        total = self._cpu_seconds.get(stage, 0.0)
        for start, end in self._gpu_intervals.get(stage, ()):
            total += start.elapsed_time(end) / 1000.0
        return float(total)

    def summary(self) -> Dict[str, float]:
        stages = set(self._cpu_seconds) | set(self._gpu_intervals)
        return {stage: self.seconds(stage) for stage in sorted(stages)}
