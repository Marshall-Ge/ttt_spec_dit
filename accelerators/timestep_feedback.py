# -*- coding: utf-8 -*-
"""Session-level timestep feedback for budgeted speculative diffusion.

This module deliberately does not select per-image strategy arms. It keeps one
online risk statistic per timestep and uses observed refresh defects to adjust
future trajectories' refresh decisions. Labels are observed only when a full
refresh is taken, so updates use clipped inverse-propensity weighting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


TIMESTEP_FEEDBACK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TimestepDecision:
    """One controller decision and the propensity needed for later IPW."""

    step_idx: int
    refresh: bool
    propensity: float
    risk_mean: float
    risk_ucb: float
    price: float
    remaining_budget: float
    reason: str

    def __post_init__(self) -> None:
        if self.step_idx < 0:
            raise ValueError("step_idx must be non-negative")
        if not 0.0 <= self.propensity <= 1.0:
            raise ValueError("propensity must be in [0, 1]")
        if self.refresh and self.propensity <= 0.0:
            raise ValueError("refresh decisions require positive propensity")
        if not all(math.isfinite(float(value)) for value in (
                self.risk_mean, self.risk_ucb, self.price,
                self.remaining_budget)):
            raise ValueError("decision values must be finite")


class _WeightedStats:
    def __init__(self) -> None:
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self.observations = 0

    @property
    def effective_count(self) -> float:
        if self.weight_square_sum <= 0.0:
            return 0.0
        return self.weight_sum ** 2 / self.weight_square_sum

    @property
    def variance(self) -> float:
        if self.weight_sum <= 0.0:
            return 0.0
        return max(self.m2 / self.weight_sum, 0.0)

    def update(self, value: float, weight: float) -> None:
        total = self.weight_sum + weight
        delta = value - self.mean
        self.mean += weight * delta / total
        self.m2 += weight * delta * (value - self.mean)
        self.weight_sum = total
        self.weight_square_sum += weight * weight
        self.observations += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "weight_sum": self.weight_sum,
            "weight_square_sum": self.weight_square_sum,
            "mean": self.mean,
            "m2": self.m2,
            "observations": self.observations,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_WeightedStats":
        stats = cls()
        for key in ("weight_sum", "weight_square_sum", "mean", "m2"):
            value = float(state[key])
            if not math.isfinite(value) or value < 0.0 and key != "mean":
                raise ValueError(f"invalid persisted timestep statistic: {key}")
            setattr(stats, key, value)
        if not math.isfinite(stats.mean):
            raise ValueError("invalid persisted timestep statistic: mean")
        stats.observations = int(state.get("observations", 0))
        if stats.observations < 0:
            raise ValueError("persisted observations must be non-negative")
        return stats


class TimestepFeedbackController:
    """Online timestep risk estimator with hard budget and exploration.

    ``observe`` is intended to be called only for refreshes. The controller's
    learned target is explicitly the one-step transition defect; no terminal
    quality claim is implied by this class.
    """

    def __init__(
            self,
            num_steps: int,
            budget_refreshes: int,
            version_key: str,
            *,
            p_min: float = 0.02,
            ucb_beta: float = 1.0,
            prior_mean: float = 0.0,
            prior_std: float = 0.1,
            initial_price: float = 0.1,
            price_learning_rate: float = 0.05,
            max_ipw_weight: float = 20.0,
            seed: int = 0,
            mandatory_steps: Sequence[int] = (),
            ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0 <= budget_refreshes <= num_steps:
            raise ValueError("budget_refreshes must be in [0, num_steps]")
        if not version_key:
            raise ValueError("version_key must be non-empty")
        if not 0.0 < p_min <= 1.0:
            raise ValueError("p_min must be in (0, 1]")
        if ucb_beta < 0.0 or prior_std < 0.0 or initial_price < 0.0:
            raise ValueError("UCB/prior/price parameters must be non-negative")
        if price_learning_rate <= 0.0 or max_ipw_weight < 1.0:
            raise ValueError("invalid price or IPW parameters")
        if not all(math.isfinite(float(value)) for value in (
                ucb_beta, prior_mean, prior_std, initial_price,
                price_learning_rate, max_ipw_weight)):
            raise ValueError("controller parameters must be finite")
        mandatory = tuple(sorted(set(int(step) for step in mandatory_steps)))
        if any(step < 0 or step >= num_steps for step in mandatory):
            raise ValueError("mandatory timestep is out of range")
        if len(mandatory) > budget_refreshes:
            raise ValueError("mandatory steps exceed refresh budget")

        self.num_steps = int(num_steps)
        self.budget_refreshes = int(budget_refreshes)
        self.version_key = str(version_key)
        self.p_min = float(p_min)
        self.ucb_beta = float(ucb_beta)
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)
        self.price = float(initial_price)
        self.price_learning_rate = float(price_learning_rate)
        self.max_ipw_weight = float(max_ipw_weight)
        self.mandatory_steps = mandatory
        self._rng = random.Random(seed)
        self._stats = [_WeightedStats() for _ in range(self.num_steps)]
        self._trajectory_steps = 0
        self._trajectory_refreshes = 0
        self.trajectories = 0

    @property
    def target_refresh_rate(self) -> float:
        return self.budget_refreshes / self.num_steps

    @property
    def trajectory_refreshes(self) -> int:
        return self._trajectory_refreshes

    def begin_trajectory(self) -> None:
        self._trajectory_steps = 0
        self._trajectory_refreshes = 0

    def risk(self, step_idx: int) -> tuple[float, float]:
        stats = self._get_stats(step_idx)
        effective_count = stats.effective_count
        if effective_count <= 0.0:
            return self.prior_mean, self.prior_mean + self.ucb_beta * self.prior_std
        uncertainty = self.ucb_beta * math.sqrt(
            (stats.variance + self.prior_std ** 2) /
            (effective_count + 1.0))
        return stats.mean, stats.mean + uncertainty

    def decide(
            self,
            step_idx: int,
            remaining_budget: float,
            *,
            refresh_cost: float = 1.0,
            ) -> TimestepDecision:
        if not 0 <= step_idx < self.num_steps:
            raise ValueError("step_idx is out of range")
        if not math.isfinite(float(remaining_budget)) or remaining_budget < 0.0:
            raise ValueError("remaining_budget must be finite and non-negative")
        if not math.isfinite(float(refresh_cost)) or refresh_cost <= 0.0:
            raise ValueError("refresh_cost must be finite and positive")

        risk_mean, risk_ucb = self.risk(step_idx)
        self._trajectory_steps += 1
        feasible = (
            remaining_budget + 1e-12 >= refresh_cost
            and self._trajectory_refreshes < self.budget_refreshes)
        if not feasible:
            return TimestepDecision(
                step_idx, False, 0.0, risk_mean, risk_ucb, self.price,
                float(remaining_budget), "hard_budget")

        if step_idx in self.mandatory_steps:
            refresh, propensity, reason = True, 1.0, "mandatory"
        elif risk_ucb > self.price:
            refresh, propensity, reason = True, 1.0, "risk_ucb"
        else:
            refresh = self._rng.random() < self.p_min
            propensity = self.p_min
            reason = "explore" if refresh else "accept"
        if refresh:
            self._trajectory_refreshes += 1
        return TimestepDecision(
            step_idx, refresh, propensity, risk_mean, risk_ucb, self.price,
            float(remaining_budget), reason)

    def observe(
            self,
            step_idx: int,
            defect: float,
            propensity: float,
            ) -> float:
        """Observe one refresh label and return the applied clipped IPW weight."""
        self._get_stats(step_idx)
        if not math.isfinite(float(defect)) or defect < 0.0:
            raise ValueError("defect must be finite and non-negative")
        if not 0.0 < float(propensity) <= 1.0:
            raise ValueError("propensity must be in (0, 1]")
        weight = min(1.0 / float(propensity), self.max_ipw_weight)
        self._stats[step_idx].update(float(defect), weight)
        return weight

    def end_trajectory(self, *, observed_cost: Optional[float] = None) -> float:
        """Update the session budget price and return its new value."""
        if observed_cost is None:
            observed_cost = self._trajectory_refreshes / self.num_steps
        if not math.isfinite(float(observed_cost)) or observed_cost < 0.0:
            raise ValueError("observed_cost must be finite and non-negative")
        self.price = max(
            0.0,
            self.price + self.price_learning_rate * (
                float(observed_cost) - self.target_refresh_rate))
        self.trajectories += 1
        return self.price

    def recommended_refresh_mask(
            self,
            *,
            budget_refreshes: Optional[int] = None,
            include_first: bool = True,
            ) -> tuple[bool, ...]:
        """Build a fixed mask for the next trajectory from timestep risk.

        This is session-level schedule adaptation, not per-image contextual
        selection. With no observations it falls back to an even schedule so a
        fresh learner does not invent a late-step bias from tied priors.
        """
        budget = self.budget_refreshes if budget_refreshes is None else int(
            budget_refreshes)
        if not 0 <= budget <= self.num_steps:
            raise ValueError("budget_refreshes must be in [0, num_steps]")
        mandatory = set(self.mandatory_steps)
        if include_first:
            mandatory.add(0)
        if len(mandatory) > budget:
            raise ValueError("mandatory steps exceed refresh budget")
        selected = set(mandatory)
        remaining = budget - len(selected)
        if remaining > 0:
            observed = sum(stats.observations for stats in self._stats)
            if observed == 0:
                denominator = max(budget - 1, 1)
                candidates = [
                    int(round(index * (self.num_steps - 1) / denominator))
                    for index in range(budget)
                ]
                ranked = candidates + list(range(self.num_steps))
            else:
                ranked = sorted(
                    range(self.num_steps),
                    key=lambda step: (self.risk(step)[1], -step),
                    reverse=True,
                )
            for step in ranked:
                if step not in selected:
                    selected.add(step)
                    remaining -= 1
                    if remaining == 0:
                        break
        return tuple(step in selected for step in range(self.num_steps))

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": TIMESTEP_FEEDBACK_SCHEMA_VERSION,
            "version_key": self.version_key,
            "num_steps": self.num_steps,
            "budget_refreshes": self.budget_refreshes,
            "p_min": self.p_min,
            "ucb_beta": self.ucb_beta,
            "price": self.price,
            "trajectories": self.trajectories,
            "timesteps": [
                {
                    "step_idx": index,
                    "observations": stats.observations,
                    "effective_count": stats.effective_count,
                    "mean": stats.mean,
                    "variance": stats.variance,
                }
                for index, stats in enumerate(self._stats)
            ],
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TIMESTEP_FEEDBACK_SCHEMA_VERSION,
            "version_key": self.version_key,
            "num_steps": self.num_steps,
            "budget_refreshes": self.budget_refreshes,
            "p_min": self.p_min,
            "ucb_beta": self.ucb_beta,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "price": self.price,
            "price_learning_rate": self.price_learning_rate,
            "max_ipw_weight": self.max_ipw_weight,
            "mandatory_steps": list(self.mandatory_steps),
            "trajectories": self.trajectories,
            "stats": [stats.state_dict() for stats in self._stats],
        }

    @classmethod
    def from_state_dict(
            cls,
            state: Mapping[str, Any],
            *,
            expected_version_key: Optional[str] = None,
            seed: int = 0,
            ) -> "TimestepFeedbackController":
        if int(state.get("schema_version", -1)) != TIMESTEP_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported timestep feedback schema")
        version_key = str(state["version_key"])
        if expected_version_key is not None and version_key != expected_version_key:
            raise ValueError(
                "timestep feedback version does not match runtime: "
                f"state={version_key}, runtime={expected_version_key}")
        controller = cls(
            int(state["num_steps"]),
            int(state["budget_refreshes"]),
            version_key,
            p_min=float(state["p_min"]),
            ucb_beta=float(state["ucb_beta"]),
            prior_mean=float(state["prior_mean"]),
            prior_std=float(state["prior_std"]),
            initial_price=float(state["price"]),
            price_learning_rate=float(state["price_learning_rate"]),
            max_ipw_weight=float(state["max_ipw_weight"]),
            mandatory_steps=tuple(state.get("mandatory_steps", ())),
            seed=seed,
        )
        stats = state.get("stats", ())
        if len(stats) != controller.num_steps:
            raise ValueError("persisted timestep statistic count does not match num_steps")
        controller._stats = [_WeightedStats.from_state_dict(item) for item in stats]
        controller.trajectories = int(state.get("trajectories", 0))
        if controller.trajectories < 0:
            raise ValueError("persisted trajectories must be non-negative")
        return controller

    def _get_stats(self, step_idx: int) -> _WeightedStats:
        if not 0 <= int(step_idx) < self.num_steps:
            raise ValueError("step_idx is out of range")
        return self._stats[int(step_idx)]
