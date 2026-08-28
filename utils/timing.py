# -*- coding: utf-8 -*-
"""Timing utilities: CUDA-event timer, generation stage profiler."""

import time
from collections import defaultdict
from typing import Any, Dict

import torch


class CudaTimer:
    """Accurate GPU-side timer using CUDA events."""

    def __init__(self, device="cuda"):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0.0

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *a):
        self.end.record()
        torch.cuda.synchronize()
        self.total_ms += self.start.elapsed_time(self.end)


class _GenerationProfiler:
    """Per-stage wall-time profiler (GPU events + CPU fallback).

    Owned by the sampling loop; used by COVR for control-overhead accounting.
    """

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

    def stop_gpu(self, token, stage: str = None) -> None:
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


def _record_profile_stage(totals: Dict[str, float], counts: Dict[str, int],
                          stage: str, elapsed_s: float) -> None:
    """Accumulate a profiler stage in either dict or defaultdict mappings."""
    totals[stage] = totals.get(stage, 0.0) + float(elapsed_s)
    counts[stage] = counts.get(stage, 0) + 1
