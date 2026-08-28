# -*- coding: utf-8 -*-
"""M7: Eval Gate — Canary 发布闸门.

发布 candidate LoRA adapter 前, 必须依次通过两道闸门:

1. **质量回归测试**: 固定 held-out prompt set 上跑标准生成质量指标
   (FID/CLIP-score), 与当前线上 baseline 比较, 退化超过 ``quality_epsilon``
   直接拒绝。

2. **效果验证**: 同一组 prompt 下, 确认 verification reject 率确实下降
   超过 ``reject_delta`` (否则说明训练没有实际收益)。

两者都通过 → 标记为 ``canary_ready``。
任一步失败 → 丢弃 candidate, 记录失败原因, 保留 buffer 继续累积。

维护最近 N 个已知良好 checkpoint, 支持一键回滚。

注意: 本模块不实现真正的灰度发布 (那是 infra 层的事),
但提供 gate 决策的完整逻辑和结果记录。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class GateStatus(Enum):
    PENDING = "pending"
    QUALITY_PASSED = "quality_passed"
    EFFECT_PASSED = "effect_passed"
    CANARY_READY = "canary_ready"
    REJECTED_QUALITY = "rejected_quality"
    REJECTED_EFFECT = "rejected_effect"
    ROLLED_BACK = "rolled_back"


@dataclass
class GateResult:
    """Single gate evaluation result."""
    status: GateStatus
    candidate_path: str
    candidate_version: str
    base_model_version: str

    # Quality metrics
    baseline_fid: Optional[float] = None
    candidate_fid: Optional[float] = None
    fid_delta: Optional[float] = None

    # Effect metrics
    baseline_reject_rate: Optional[float] = None
    candidate_reject_rate: Optional[float] = None
    reject_delta: Optional[float] = None

    # Metadata
    reason: str = ""
    evaluated_at: str = ""
    heldout_prompt_count: int = 0


@dataclass
class GoodCheckpoint:
    """A known-good checkpoint record."""
    path: str
    version: str
    base_model_version: str
    fid: float
    reject_rate: float
    saved_at: str


class EvalGate:
    """Canary release gate for LoRA adapter checkpoints.

    Usage::

        gate = EvalGate(output_dir="./output/vfl_checkpoints",
                        max_good_checkpoints=5)

        # After async trainer produces a candidate:
        result = gate.evaluate(
            candidate_path="/path/to/lora_candidate_v003.pt",
            baseline_fid=250.0,
            candidate_fid=252.0,
            baseline_reject_rate=0.45,
            candidate_reject_rate=0.38,
            base_model_version="dit-v1.0",
            heldout_prompt_count=50,
        )

        if result.status == GateStatus.CANARY_READY:
            gate.promote(result)
    """

    def __init__(self,
                 output_dir: str = "./output/vfl_checkpoints",
                 quality_epsilon: float = 5.0,
                 reject_delta: float = 0.05,
                 max_good_checkpoints: int = 5):
        """
        Parameters
        ----------
        output_dir : str
            Directory for storing gate state and good checkpoints.
        quality_epsilon : float
            Maximum allowed FID degradation (absolute).
        reject_delta : float
            Minimum required reject-rate reduction (absolute, e.g. 0.05 = 5pp).
        max_good_checkpoints : int
            Maximum number of known-good checkpoints to retain.
        """
        self.output_dir = output_dir
        self.quality_epsilon = quality_epsilon
        self.reject_delta = reject_delta
        self.max_good_checkpoints = max_good_checkpoints

        self._good_checkpoints: List[GoodCheckpoint] = []
        self._history: List[GateResult] = []

        os.makedirs(output_dir, exist_ok=True)
        self._state_path = os.path.join(output_dir, "eval_gate_state.json")
        self._load_state()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self,
                 candidate_path: str,
                 baseline_fid: float,
                 candidate_fid: float,
                 baseline_reject_rate: float,
                 candidate_reject_rate: float,
                 base_model_version: str = "unknown",
                 candidate_version: str = "unknown",
                 heldout_prompt_count: int = 0,
                 ) -> GateResult:
        """Evaluate a candidate checkpoint against quality and effect gates.

        Parameters
        ----------
        candidate_path : str
            Path to the LoRA checkpoint file.
        baseline_fid : float
            Current baseline FID (without adapter).
        candidate_fid : float
            FID with the candidate adapter loaded.
        baseline_reject_rate : float
            Current reject rate (0.0–1.0) without adapter.
        candidate_reject_rate : float
            Reject rate (0.0–1.0) with the candidate adapter.
        base_model_version : str
        candidate_version : str
        heldout_prompt_count : int

        Returns
        -------
        GateResult
        """
        import time
        result = GateResult(
            status=GateStatus.PENDING,
            candidate_path=candidate_path,
            candidate_version=candidate_version,
            base_model_version=base_model_version,
            baseline_fid=baseline_fid,
            candidate_fid=candidate_fid,
            fid_delta=candidate_fid - baseline_fid,
            baseline_reject_rate=baseline_reject_rate,
            candidate_reject_rate=candidate_reject_rate,
            reject_delta=baseline_reject_rate - candidate_reject_rate,
            evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            heldout_prompt_count=heldout_prompt_count,
        )

        # ---- Gate 1: Quality regression ----
        fid_delta = candidate_fid - baseline_fid
        if fid_delta > self.quality_epsilon:
            result.status = GateStatus.REJECTED_QUALITY
            result.reason = (
                f"FID degraded by {fid_delta:.2f} > ε={self.quality_epsilon} "
                f"(baseline={baseline_fid:.2f}, candidate={candidate_fid:.2f})")
            self._history.append(result)
            self._save_state()
            return result

        result.status = GateStatus.QUALITY_PASSED

        # ---- Gate 2: Effect verification ----
        reject_reduction = baseline_reject_rate - candidate_reject_rate
        if reject_reduction < self.reject_delta:
            result.status = GateStatus.REJECTED_EFFECT
            result.reason = (
                f"Reject rate reduction {reject_reduction:.3f} < δ={self.reject_delta} "
                f"(baseline={baseline_reject_rate:.3f}, candidate={candidate_reject_rate:.3f})")
            self._history.append(result)
            self._save_state()
            return result

        result.status = GateStatus.EFFECT_PASSED

        # ---- Both passed → canary ready ----
        result.status = GateStatus.CANARY_READY
        result.reason = (
            f"Quality OK (ΔFID={fid_delta:.2f} ≤ ε={self.quality_epsilon}), "
            f"Effect OK (Δreject={reject_reduction:.3f} ≥ δ={self.reject_delta})")
        self._history.append(result)
        self._save_state()
        return result

    # ------------------------------------------------------------------
    # Promote / Rollback
    # ------------------------------------------------------------------

    def promote(self, result: GateResult):
        """Promote a CANARY_READY candidate to a known-good checkpoint.

        Copies the checkpoint to the good-checkpoints directory and
        registers it for potential rollback.
        """
        if result.status != GateStatus.CANARY_READY:
            raise ValueError(f"Cannot promote candidate with status {result.status}")

        # Copy to good-checkpoints directory
        good_dir = os.path.join(self.output_dir, "good_checkpoints")
        os.makedirs(good_dir, exist_ok=True)

        dst = os.path.join(
            good_dir,
            f"lora_good_{result.candidate_version}.pt",
        )
        if os.path.exists(result.candidate_path):
            shutil.copy2(result.candidate_path, dst)

        good = GoodCheckpoint(
            path=dst,
            version=result.candidate_version,
            base_model_version=result.base_model_version,
            fid=result.candidate_fid or 0.0,
            reject_rate=result.candidate_reject_rate or 0.0,
            saved_at=result.evaluated_at,
        )
        self._good_checkpoints.append(good)

        # Prune old checkpoints
        while len(self._good_checkpoints) > self.max_good_checkpoints:
            old = self._good_checkpoints.pop(0)
            if os.path.exists(old.path):
                os.remove(old.path)

        self._save_state()

    def rollback(self) -> Optional[str]:
        """Roll back to the most recent known-good checkpoint.

        Returns the path to the good checkpoint, or None if none available.
        """
        if not self._good_checkpoints:
            return None

        # Mark the current one as rolled back
        current = self._good_checkpoints[-1]
        result = GateResult(
            status=GateStatus.ROLLED_BACK,
            candidate_path=current.path,
            candidate_version=current.version,
            base_model_version=current.base_model_version,
            reason=f"Rolled back to {current.version}",
        )
        self._history.append(result)
        self._save_state()

        return current.path

    def get_latest_good(self) -> Optional[GoodCheckpoint]:
        """Return the most recent known-good checkpoint."""
        if self._good_checkpoints:
            return self._good_checkpoints[-1]
        return None

    # ------------------------------------------------------------------
    # Config adaptation
    # ------------------------------------------------------------------

    def suggest_lambda_adjustment(self, last_result: GateResult) -> float:
        """Suggest lambda_curvature adjustment based on gate result.

        Returns a multiplier (e.g. 0.5 to halve, 2.0 to double).
        """
        if last_result.status == GateStatus.REJECTED_QUALITY:
            # Quality degraded → reduce lambda
            return 0.5
        elif last_result.status == GateStatus.REJECTED_EFFECT:
            # No effect → increase lambda (if quality is fine)
            if last_result.fid_delta is not None and last_result.fid_delta < 0:
                # Actually quality improved — still increase
                return 2.0
            return 1.5
        elif last_result.status == GateStatus.CANARY_READY:
            return 1.0  # keep current
        return 1.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load gate state from disk."""
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            self._good_checkpoints = [
                GoodCheckpoint(**g) for g in data.get("good_checkpoints", [])
            ]
            self._history = [
                GateResult(**{**r, "status": GateStatus(r["status"])})
                for r in data.get("history", [])
            ]
        except Exception:
            pass  # Corrupted state → start fresh

    def _save_state(self):
        """Persist gate state to disk."""
        data = {
            "good_checkpoints": [
                {
                    "path": g.path,
                    "version": g.version,
                    "base_model_version": g.base_model_version,
                    "fid": g.fid,
                    "reject_rate": g.reject_rate,
                    "saved_at": g.saved_at,
                }
                for g in self._good_checkpoints
            ],
            "history": [
                {
                    "status": r.status.value,
                    "candidate_path": r.candidate_path,
                    "candidate_version": r.candidate_version,
                    "base_model_version": r.base_model_version,
                    "baseline_fid": r.baseline_fid,
                    "candidate_fid": r.candidate_fid,
                    "fid_delta": r.fid_delta,
                    "baseline_reject_rate": r.baseline_reject_rate,
                    "candidate_reject_rate": r.candidate_reject_rate,
                    "reject_delta": r.reject_delta,
                    "reason": r.reason,
                    "evaluated_at": r.evaluated_at,
                    "heldout_prompt_count": r.heldout_prompt_count,
                }
                for r in self._history[-50:]  # Keep last 50
            ],
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def good_checkpoint_count(self) -> int:
        return len(self._good_checkpoints)

    @property
    def history_count(self) -> int:
        return len(self._history)
