# -*- coding: utf-8 -*-
"""Model-agnostic compute decisions for verification-guided acceleration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ComputeAction(str, Enum):
    APPROXIMATE = "approximate"
    VERIFY = "verify"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class ComputeOpportunity:
    model: str
    method: str
    trajectory_id: int
    step_idx: int
    num_steps: int
    layer_idx: int
    timestep_bucket: int
    approximation_distance: int
    estimated_cost: float = 1.0
    verification_requested: bool = False


@dataclass(frozen=True)
class VerificationResult:
    error_value: float
    threshold: float
    accepted: bool


class ComputeController(ABC):
    @abstractmethod
    def begin_trajectory(self, trajectory_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def decide(self, opportunity: ComputeOpportunity) -> ComputeAction:
        raise NotImplementedError

    @abstractmethod
    def observe(self, opportunity: ComputeOpportunity,
                result: VerificationResult) -> bool:
        """Record a verification and return whether to use its full output."""
        raise NotImplementedError

    @abstractmethod
    def end_trajectory(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> Dict[str, object]:
        raise NotImplementedError


class ProbeCorrectController(ComputeController):
    def __init__(self, correction_policy: str = "reject"):
        if correction_policy not in {"reject", "always"}:
            raise ValueError(
                "correction_policy must be 'reject' or 'always'")
        self.correction_policy = correction_policy
        self._active_trajectory: Optional[int] = None
        self._stats = {
            "trajectories": 0,
            "opportunities": 0,
            "verifications": 0,
            "accepted_verifications": 0,
            "rejected_verifications": 0,
            "corrected_verifications": 0,
        }

    def begin_trajectory(self, trajectory_id: int) -> None:
        if self._active_trajectory is not None:
            raise RuntimeError("end_trajectory() must precede the next trajectory")
        self._active_trajectory = trajectory_id
        self._stats["trajectories"] += 1

    def decide(self, opportunity: ComputeOpportunity) -> ComputeAction:
        self._require_active(opportunity)
        self._stats["opportunities"] += 1
        if opportunity.verification_requested:
            return ComputeAction.VERIFY
        return ComputeAction.APPROXIMATE

    def observe(self, opportunity: ComputeOpportunity,
                result: VerificationResult) -> bool:
        self._require_active(opportunity)
        self._stats["verifications"] += 1
        if result.accepted:
            self._stats["accepted_verifications"] += 1
        else:
            self._stats["rejected_verifications"] += 1

        use_verified = self.correction_policy == "always" or not result.accepted
        if use_verified:
            self._stats["corrected_verifications"] += 1
        return use_verified

    def end_trajectory(self) -> None:
        if self._active_trajectory is None:
            raise RuntimeError("begin_trajectory() must precede end_trajectory()")
        self._active_trajectory = None

    def stats(self) -> Dict[str, object]:
        result = dict(self._stats)
        result["controller"] = "probe_correct"
        result["correction_policy"] = self.correction_policy
        result["active_trajectory"] = self._active_trajectory
        return result

    def _require_active(self, opportunity: ComputeOpportunity) -> None:
        if self._active_trajectory is None:
            raise RuntimeError("begin_trajectory() must precede controller use")
        if opportunity.trajectory_id != self._active_trajectory:
            raise ValueError(
                f"opportunity trajectory {opportunity.trajectory_id} does not "
                f"match active trajectory {self._active_trajectory}")
