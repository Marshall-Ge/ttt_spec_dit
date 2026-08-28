# -*- coding: utf-8 -*-
"""Counterfactual online verification for budgeted SpecA inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 2
FEATURE_NAMES = (
    "bias",
    "progress",
    "log_snr",
    "scheduler_step_size",
    "distance_since_refresh",
    "taylor_term_1",
    "taylor_term_2",
    "taylor_term_3",
    "taylor_term_4",
    "order_2_4_disagreement",
    "attn_curvature",
    "mlp_curvature",
    "cfg_draft_disagreement",
    "previous_defect",
    "remaining_budget",
)


class COVRAction(str, Enum):
    ACCEPT = "accept"
    REFRESH = "refresh"


@dataclass(frozen=True)
class COVRVersion:
    model: str
    base_model_version: str
    scheduler: str
    scheduler_config: str
    num_steps: int
    cfg_scale: float
    speca_config: str

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class COVRContext:
    step_idx: int
    num_steps: int
    timestep: float
    log_snr: float = 0.0
    scheduler_step_size: float = 0.0
    distance_since_refresh: int = 0
    taylor_term_norms: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    order_2_4_disagreement: float = 0.0
    attn_curvature: float = 0.0
    mlp_curvature: float = 0.0
    cfg_draft_disagreement: float = 0.0
    previous_defect: float = 0.0
    remaining_budget: float = 1.0

    def features(self) -> np.ndarray:
        progress = self.step_idx / max(self.num_steps - 1, 1)
        distance = self.distance_since_refresh / max(self.num_steps, 1)
        values = (
            1.0,
            progress,
            _finite(self.log_snr),
            _finite(self.scheduler_step_size),
            distance,
            *(_finite(v) for v in self.taylor_term_norms),
            _finite(self.order_2_4_disagreement),
            _finite(self.attn_curvature),
            _finite(self.mlp_curvature),
            _finite(self.cfg_draft_disagreement),
            _finite(self.previous_defect),
            min(max(_finite(self.remaining_budget), 0.0), 1.0),
        )
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "COVRContext":
        data = dict(value)
        data["taylor_term_norms"] = tuple(data.get("taylor_term_norms", (0.0,) * 4))
        return cls(**data)


@dataclass(frozen=True)
class COVRDecision:
    action: COVRAction
    propensity: float
    refresh_probability: float
    reason: str
    risk_hat: float
    risk_ucb: float
    incremental_cost: float


@dataclass(frozen=True)
class CounterfactualEvent:
    session_id: str
    trajectory_id: int
    sample_id: str
    version_key: str
    context: COVRContext
    action: COVRAction
    propensity: float
    policy: str
    incremental_cost: float
    one_step_defect: Optional[float] = None
    local_probe_error: Optional[float] = None
    h_step_defect: Optional[float] = None
    terminal_intervention_gain: Optional[float] = None
    accepted_approximation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["schema_version"] = SCHEMA_VERSION
        _assert_scalar_tree(data)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterfactualEvent":
        data = dict(value)
        schema_version = data.pop("schema_version", SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported COVR event schema: {schema_version}")
        data["context"] = COVRContext.from_dict(data["context"])
        data["action"] = COVRAction(data["action"])
        return cls(**data)


def transition_defects(
    x_prev_approx: Any,
    x_prev_full: Any,
    x_t: Any,
    eps: float = 1e-8,
) -> Tuple[float, ...]:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.is_tensor(x_prev_approx):
        approx = x_prev_approx.detach().float().reshape(x_prev_approx.shape[0], -1)
        full = x_prev_full.detach().float().reshape(x_prev_full.shape[0], -1)
        current = x_t.detach().float().reshape(x_t.shape[0], -1)
        numerator = (approx - full).norm(dim=1)
        denominator = (full - current).norm(dim=1).clamp_min(eps)
        return tuple(float(v) for v in (numerator / denominator).cpu().tolist())

    approx_np = _as_batch_array(x_prev_approx)
    full_np = _as_batch_array(x_prev_full)
    current_np = _as_batch_array(x_t)
    numerator = np.linalg.norm(approx_np - full_np, axis=1)
    denominator = np.maximum(np.linalg.norm(full_np - current_np, axis=1), eps)
    return tuple(float(v) for v in numerator / denominator)


def normalized_transition_defect(
    x_prev_approx: Any,
    x_prev_full: Any,
    x_t: Any,
    eps: float = 1e-8,
) -> float:
    defects = transition_defects(x_prev_approx, x_prev_full, x_t, eps=eps)
    return float(sum(defects) / len(defects))


class BudgetLedger:
    def __init__(self, capacity: float):
        if capacity < 0:
            raise ValueError("budget capacity must be non-negative")
        self.capacity = float(capacity)
        self.spent = 0.0
        self.by_category: Dict[str, float] = {}

    @property
    def remaining(self) -> float:
        return max(self.capacity - self.spent, 0.0)

    @property
    def remaining_fraction(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.remaining / self.capacity

    def can_spend(self, cost: float) -> bool:
        return cost >= 0 and self.spent + cost <= self.capacity + 1e-12

    def spend(self, cost: float, category: str) -> None:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if not self.can_spend(cost):
            raise RuntimeError(
                f"budget exceeded: spent={self.spent}, cost={cost}, "
                f"capacity={self.capacity}")
        self.spent += float(cost)
        self.by_category[category] = self.by_category.get(category, 0.0) + float(cost)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "spent": self.spent,
            "by_category": dict(self.by_category),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BudgetLedger":
        ledger = cls(float(state["capacity"]))
        ledger.spent = float(state["spent"])
        ledger.by_category = {
            str(k): float(v) for k, v in state.get("by_category", {}).items()
        }
        if ledger.spent > ledger.capacity + 1e-12:
            raise ValueError("persisted budget exceeds capacity")
        return ledger


class OnlineRidgeUCB:
    def __init__(self, feature_dim: int, alpha: float = 1.0, beta: float = 1.0):
        if feature_dim <= 0 or alpha <= 0 or beta < 0:
            raise ValueError("invalid ridge/UCB parameters")
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.precision = np.eye(feature_dim, dtype=np.float64) * alpha
        self.target = np.zeros(feature_dim, dtype=np.float64)
        self.observations = 0
        self.weight_sum = 0.0

    def predict(self, features: Sequence[float]) -> Tuple[float, float]:
        phi = self._validate_features(features)
        theta = np.linalg.solve(self.precision, self.target)
        variance = max(float(phi @ np.linalg.solve(self.precision, phi)), 0.0)
        risk_hat = float(phi @ theta)
        risk_ucb = risk_hat + self.beta * math.sqrt(variance)
        return risk_hat, risk_ucb

    def observe(self, features: Sequence[float], target: float,
                sample_weight: float = 1.0) -> None:
        phi = self._validate_features(features)
        if sample_weight <= 0 or not math.isfinite(sample_weight):
            raise ValueError("sample_weight must be finite and positive")
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        self.precision += sample_weight * np.outer(phi, phi)
        self.target += sample_weight * phi * float(target)
        self.observations += 1
        self.weight_sum += float(sample_weight)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "beta": self.beta,
            "precision": self.precision.tolist(),
            "target": self.target.tolist(),
            "observations": self.observations,
            "weight_sum": self.weight_sum,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "OnlineRidgeUCB":
        model = cls(
            int(state["feature_dim"]),
            alpha=float(state["alpha"]),
            beta=float(state["beta"]),
        )
        model.precision = np.asarray(state["precision"], dtype=np.float64)
        model.target = np.asarray(state["target"], dtype=np.float64)
        if model.precision.shape != (model.feature_dim, model.feature_dim):
            raise ValueError("invalid persisted precision shape")
        if model.target.shape != (model.feature_dim,):
            raise ValueError("invalid persisted target shape")
        model.observations = int(state.get("observations", 0))
        model.weight_sum = float(state.get("weight_sum", 0.0))
        return model

    def _validate_features(self, features: Sequence[float]) -> np.ndarray:
        phi = np.asarray(features, dtype=np.float64)
        if phi.shape != (self.feature_dim,):
            raise ValueError(
                f"expected {self.feature_dim} features, got shape {phi.shape}")
        if not np.isfinite(phi).all():
            raise ValueError("features must be finite")
        return phi


class PrimalDualBudget:
    def __init__(self, target_cost: float, learning_rate: float = 0.01,
                 initial_price: float = 0.0):
        if target_cost < 0 or learning_rate <= 0 or initial_price < 0:
            raise ValueError("invalid primal-dual parameters")
        self.target_cost = float(target_cost)
        self.learning_rate = float(learning_rate)
        self.price = float(initial_price)
        self.updates = 0

    def update(self, observed_cost: float) -> float:
        self.price = max(
            0.0,
            self.price + self.learning_rate * (observed_cost - self.target_cost),
        )
        self.updates += 1
        return self.price

    def state_dict(self) -> Dict[str, Any]:
        return {
            "target_cost": self.target_cost,
            "learning_rate": self.learning_rate,
            "price": self.price,
            "updates": self.updates,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PrimalDualBudget":
        dual = cls(
            target_cost=float(state["target_cost"]),
            learning_rate=float(state["learning_rate"]),
            initial_price=float(state["price"]),
        )
        dual.updates = int(state.get("updates", 0))
        return dual


class COVRPolicy:
    def __init__(
        self,
        risk_model: OnlineRidgeUCB,
        budget: BudgetLedger,
        dual: PrimalDualBudget,
        p_min: float = 0.02,
        ipw_clip: float = 20.0,
        stabilized_numerator: Optional[float] = None,
        seed: int = 0,
    ):
        if not 0 < p_min <= 1 or ipw_clip < 1:
            raise ValueError("invalid exploration/IPW settings")
        if stabilized_numerator is not None and not 0 < stabilized_numerator <= 1:
            raise ValueError("stabilized_numerator must be in (0, 1]")
        self.risk_model = risk_model
        self.budget = budget
        self.dual = dual
        self.p_min = float(p_min)
        self.ipw_clip = float(ipw_clip)
        self.stabilized_numerator = float(
            p_min if stabilized_numerator is None else stabilized_numerator)
        self.rng = random.Random(seed)
        self.stats = {
            "decisions": 0,
            "accepts": 0,
            "refreshes": 0,
            "exploration_refreshes": 0,
            "budget_rejections": 0,
            "observations": 0,
        }

    def decide(self, context: COVRContext, incremental_cost: float,
               force_refresh: bool = False) -> COVRDecision:
        if incremental_cost < 0:
            raise ValueError("incremental_cost must be non-negative")
        risk_hat, risk_ucb = self.risk_model.predict(context.features())
        self.stats["decisions"] += 1

        if not self.budget.can_spend(incremental_cost):
            self.stats["accepts"] += 1
            self.stats["budget_rejections"] += 1
            return COVRDecision(
                COVRAction.ACCEPT, 1.0, 0.0, "budget_exhausted",
                risk_hat, risk_ucb, incremental_cost)

        policy_refresh = force_refresh or (
            risk_ucb > self.dual.price * incremental_cost)
        refresh_probability = 1.0 if policy_refresh else self.p_min
        refresh = policy_refresh or self.rng.random() < self.p_min
        if refresh:
            self.stats["refreshes"] += 1
            reason = "forced" if force_refresh else (
                "risk_ucb" if policy_refresh else "exploration")
            if reason == "exploration":
                self.stats["exploration_refreshes"] += 1
            propensity = refresh_probability
            action = COVRAction.REFRESH
        else:
            self.stats["accepts"] += 1
            reason = "policy_accept"
            propensity = 1.0 - refresh_probability
            action = COVRAction.ACCEPT

        return COVRDecision(
            action, propensity, refresh_probability, reason,
            risk_hat, risk_ucb, incremental_cost)

    def commit(self, decision: COVRDecision) -> None:
        cost = decision.incremental_cost if decision.action == COVRAction.REFRESH else 0.0
        if cost:
            category = (
                "exploration_refresh"
                if decision.reason == "exploration"
                else "policy_refresh"
            )
            self.budget.spend(cost, category)
        self.dual.update(cost)

    def observe(self, context: COVRContext, decision: COVRDecision,
                defect: float) -> float:
        if decision.action != COVRAction.REFRESH:
            raise ValueError("counterfactual labels are only revealed on refresh")
        if not 0 < decision.propensity <= 1:
            raise ValueError("refresh propensity must be in (0, 1]")
        weight = min(
            self.stabilized_numerator / decision.propensity,
            self.ipw_clip,
        )
        self.risk_model.observe(context.features(), defect, sample_weight=weight)
        self.stats["observations"] += 1
        return weight

    def state_dict(self) -> Dict[str, Any]:
        return {
            "risk_model": self.risk_model.state_dict(),
            "budget": self.budget.state_dict(),
            "dual": self.dual.state_dict(),
            "p_min": self.p_min,
            "ipw_clip": self.ipw_clip,
            "stabilized_numerator": self.stabilized_numerator,
            "stats": dict(self.stats),
        }


class ShadowAuditRecorder:
    def __init__(self, output_dir: str, session_id: str, version: COVRVersion,
                 max_events: Optional[int] = None):
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.version = version
        self.max_events = max_events
        self.event_path = self.output_dir / f"events_{session_id}.jsonl"
        self.summary_path = self.output_dir / f"summary_{session_id}.json"
        self._handle = self.event_path.open("a", encoding="utf-8")
        self.count = 0
        self.defect_sum = 0.0
        self.previous_defect = 0.0
        self.closed = False

    @property
    def enabled(self) -> bool:
        return not self.closed and (
            self.max_events is None or self.count < self.max_events)

    def record(self, event: CounterfactualEvent) -> bool:
        if self.closed:
            raise RuntimeError("recorder is closed")
        if not self.enabled:
            return False
        if event.session_id != self.session_id:
            raise ValueError("event session does not match recorder")
        if event.version_key != self.version.key:
            raise ValueError("event version does not match recorder")
        self._handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        self._handle.flush()
        self.count += 1
        if event.one_step_defect is not None:
            self.previous_defect = float(event.one_step_defect)
            self.defect_sum += self.previous_defect
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "version": asdict(self.version),
            "version_key": self.version.key,
            "events": self.count,
            "mean_one_step_defect": self.defect_sum / max(self.count, 1),
            "event_path": str(self.event_path),
        }

    def close(self) -> Dict[str, Any]:
        if not self.closed:
            self._handle.close()
            _atomic_json_dump(self.summary_path, self.summary())
            self.closed = True
        return self.summary()

    def __enter__(self) -> "ShadowAuditRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def save_policy_state(path: str, version: COVRVersion,
                      policy: COVRPolicy, session_id: str) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "version": asdict(version),
        "version_key": version.key,
        "session_id": session_id,
        "feature_names": FEATURE_NAMES,
        "policy": policy.state_dict(),
    }
    _atomic_json_dump(Path(path), payload)


def load_policy_state(path: str, expected_version: COVRVersion,
                      seed: int = 0) -> COVRPolicy:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("incompatible COVR policy schema")
    if payload.get("version_key") != expected_version.key:
        raise ValueError("COVR policy version does not match current inference config")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("COVR feature schema does not match current code")
    state = payload["policy"]
    policy = COVRPolicy(
        risk_model=OnlineRidgeUCB.from_state_dict(state["risk_model"]),
        budget=BudgetLedger.from_state_dict(state["budget"]),
        dual=PrimalDualBudget.from_state_dict(state["dual"]),
        p_min=float(state["p_min"]),
        ipw_clip=float(state["ipw_clip"]),
        stabilized_numerator=float(
            state.get("stabilized_numerator", state["p_min"])),
        seed=seed,
    )
    policy.stats.update({k: int(v) for k, v in state.get("stats", {}).items()})
    return policy


def read_events(paths: Iterable[str]) -> List[CounterfactualEvent]:
    events: List[CounterfactualEvent] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(CounterfactualEvent.from_dict(json.loads(line)))
    return events


def summarize_taylor_cache(cache_dic: Any, distance: int) -> Dict[str, Any]:
    factorials = (1.0, 1.0, 2.0, 6.0, 24.0)
    term_values: List[List[float]] = [[], [], [], []]
    disagreement: List[float] = []
    attn_curvature: List[float] = []
    mlp_curvature: List[float] = []
    active_cache = getattr(cache_dic, "cache", {}).get(-1, {})

    for modules in active_cache.values():
        for module_name, factors in modules.items():
            if not factors:
                continue
            term_norms = []
            for order in range(1, 5):
                if order < len(factors):
                    factor = factors[order]
                    value = _tensor_abs_mean(factor)
                    value *= abs(distance) ** order / factorials[order]
                else:
                    value = 0.0
                term_values[order - 1].append(value)
                term_norms.append(value)
            base_norm = max(_tensor_abs_mean(factors[0]), 1e-8)
            disagreement.append((term_norms[2] + term_norms[3]) / base_norm)
            curvature = term_norms[1] / base_norm
            if "attn" in module_name:
                attn_curvature.append(curvature)
            elif module_name in {"mlp", "ff"}:
                mlp_curvature.append(curvature)

    return {
        "taylor_term_norms": tuple(_mean(values) for values in term_values),
        "order_2_4_disagreement": _mean(disagreement),
        "attn_curvature": _mean(attn_curvature),
        "mlp_curvature": _mean(mlp_curvature),
    }


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _as_batch_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _tensor_abs_mean(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().abs().mean().item())
    return float(np.abs(np.asarray(value)).mean())


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _assert_scalar_tree(value: Any, path: str = "event") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_scalar_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_scalar_tree(child, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    raise TypeError(f"{path} contains non-scalar value {type(value).__name__}")


def _atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class ActionAuditContext:
    step_idx: int
    num_steps: int
    timestep: float
    log_snr: float
    alpha_t: float
    alpha_prev: float
    latent_coefficient: float
    model_output_coefficient: float
    distance_since_refresh: int
    previous_defect_mean: float = 0.0

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0 <= self.step_idx < self.num_steps:
            raise ValueError("step_idx must be within the denoising trajectory")
        if self.distance_since_refresh < 0:
            raise ValueError("distance_since_refresh must be non-negative")
        values = (
            self.timestep,
            self.log_snr,
            self.alpha_t,
            self.alpha_prev,
            self.latent_coefficient,
            self.model_output_coefficient,
            self.previous_defect_mean,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("action audit context values must be finite")
        if not 0 < self.alpha_t <= 1 or not 0 < self.alpha_prev <= 1:
            raise ValueError("scheduler alpha values must be in (0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionAuditContext":
        return cls(**dict(value))


@dataclass(frozen=True)
class TransitionDefectBatch:
    numerators: Tuple[float, ...]
    denominators: Tuple[float, ...]
    ratios: Tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.numerators)
        if size == 0 or len(self.denominators) != size or len(self.ratios) != size:
            raise ValueError("transition metric arrays must have equal non-zero lengths")
        for numerator, denominator, ratio in zip(
                self.numerators, self.denominators, self.ratios):
            values = (float(numerator), float(denominator), float(ratio))
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError("transition metrics must be finite and non-negative")
            if denominator >= 1e-8 and not math.isclose(
                    ratio, numerator / denominator, rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError("transition ratio is inconsistent with numerator/denominator")

    @property
    def sample_count(self) -> int:
        return len(self.ratios)

    @property
    def mean_numerator(self) -> float:
        return _mean(self.numerators)

    @property
    def mean_denominator(self) -> float:
        return _mean(self.denominators)

    @property
    def mean_ratio(self) -> float:
        return _mean(self.ratios)

    @property
    def cvar90_ratio(self) -> float:
        values = np.asarray(self.ratios, dtype=np.float64)
        threshold = float(np.quantile(values, 0.9))
        return float(values[values >= threshold].mean())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionDefectBatch":
        return cls(
            numerators=tuple(float(item) for item in value["numerators"]),
            denominators=tuple(float(item) for item in value["denominators"]),
            ratios=tuple(float(item) for item in value["ratios"]),
        )


@dataclass(frozen=True)
class ActionAuditEvent:
    session_id: str
    trajectory_id: int
    sample_ids: Tuple[str, ...]
    class_ids: Tuple[int, ...]
    version_key: str
    context: ActionAuditContext
    committed_action: COVRAction
    committed_propensity: float
    audit_action: COVRAction
    audit_propensity: float
    policy: str
    incremental_cost: float
    one_step_transition: TransitionDefectBatch
    local_probe_error: Optional[float] = None
    h_step_transition: Optional[TransitionDefectBatch] = None
    terminal_quality_gains: Optional[Tuple[float, ...]] = None
    terminal_fidelity_gains: Optional[Tuple[float, ...]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(self.class_ids) != sample_count:
            raise ValueError("sample_ids and class_ids must have equal non-zero lengths")
        if self.one_step_transition.sample_count != sample_count:
            raise ValueError("one-step metrics must match the event sample count")
        if self.h_step_transition is not None \
                and self.h_step_transition.sample_count != sample_count:
            raise ValueError("H-step metrics must match the event sample count")
        for gains in (self.terminal_quality_gains, self.terminal_fidelity_gains):
            if gains is not None and len(gains) != sample_count:
                raise ValueError("terminal gains must match the event sample count")
            if gains is not None and not all(math.isfinite(float(item)) for item in gains):
                raise ValueError("terminal gains must be finite")
        for propensity in (self.committed_propensity, self.audit_propensity):
            if not 0 < float(propensity) <= 1:
                raise ValueError("propensities must be in (0, 1]")
        if not math.isfinite(float(self.incremental_cost)) or self.incremental_cost < 0:
            raise ValueError("incremental_cost must be finite and non-negative")
        if self.local_probe_error is not None \
                and not math.isfinite(float(self.local_probe_error)):
            raise ValueError("local_probe_error must be finite")
        _assert_scalar_tree(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AUDIT_SCHEMA_VERSION
        payload["event_type"] = "batch_step_audit"
        payload["committed_action"] = self.committed_action.value
        payload["audit_action"] = self.audit_action.value
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionAuditEvent":
        data = dict(value)
        schema_version = data.pop("schema_version", None)
        event_type = data.pop("event_type", None)
        if schema_version != AUDIT_SCHEMA_VERSION or event_type != "batch_step_audit":
            raise ValueError("unsupported action audit event schema")
        data["sample_ids"] = tuple(str(item) for item in data["sample_ids"])
        data["class_ids"] = tuple(int(item) for item in data["class_ids"])
        data["context"] = ActionAuditContext.from_dict(data["context"])
        data["committed_action"] = COVRAction(data["committed_action"])
        data["audit_action"] = COVRAction(data["audit_action"])
        data["one_step_transition"] = TransitionDefectBatch.from_dict(
            data["one_step_transition"])
        if data.get("h_step_transition") is not None:
            data["h_step_transition"] = TransitionDefectBatch.from_dict(
                data["h_step_transition"])
        for key in ("terminal_quality_gains", "terminal_fidelity_gains"):
            if data.get(key) is not None:
                data[key] = tuple(float(item) for item in data[key])
        return cls(**data)


def transition_defect_batch(
    x_prev_approx: Any,
    x_prev_full: Any,
    x_t: Any,
    eps: float = 1e-8,
) -> TransitionDefectBatch:
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    try:
        import torch
    except ImportError:
        torch = None

    tensor_inputs = (x_prev_approx, x_prev_full, x_t)
    if torch is not None and any(torch.is_tensor(value) for value in tensor_inputs):
        if not all(torch.is_tensor(value) for value in tensor_inputs):
            raise TypeError("transition inputs must all be tensors or arrays")
        shapes = tuple(value.shape for value in tensor_inputs)
        if len(set(shapes)) != 1:
            raise ValueError("transition inputs must have identical shapes")
        approx = x_prev_approx.detach().float().reshape(x_prev_approx.shape[0], -1)
        full = x_prev_full.detach().float().reshape(x_prev_full.shape[0], -1)
        current = x_t.detach().float().reshape(x_t.shape[0], -1)
        numerator = (approx - full).square().mean(dim=1).sqrt()
        denominator = (full - current).square().mean(dim=1).sqrt()
        ratio = numerator / denominator.clamp_min(eps)
        return TransitionDefectBatch(
            numerators=tuple(float(item) for item in numerator.cpu().tolist()),
            denominators=tuple(float(item) for item in denominator.cpu().tolist()),
            ratios=tuple(float(item) for item in ratio.cpu().tolist()),
        )

    approx = _as_batch_array(x_prev_approx)
    full = _as_batch_array(x_prev_full)
    current = _as_batch_array(x_t)
    shapes = (approx.shape, full.shape, current.shape)
    if len(set(shapes)) != 1:
        raise ValueError("transition inputs must have identical shapes")
    numerator = np.sqrt(np.square(approx - full).mean(axis=1))
    denominator = np.sqrt(np.square(full - current).mean(axis=1))
    ratio = numerator / np.maximum(denominator, eps)
    return TransitionDefectBatch(
        numerators=tuple(float(item) for item in numerator.tolist()),
        denominators=tuple(float(item) for item in denominator.tolist()),
        ratios=tuple(float(item) for item in ratio.tolist()),
    )


def ddim_epsilon_transition_coefficients(
    alpha_t: float,
    alpha_prev: float,
) -> Tuple[float, float]:
    alpha_t = float(alpha_t)
    alpha_prev = float(alpha_prev)
    if not 0 < alpha_t <= 1 or not 0 < alpha_prev <= 1:
        raise ValueError("DDIM alpha values must be in (0, 1]")
    latent_coefficient = math.sqrt(alpha_prev / alpha_t)
    model_output_coefficient = (
        math.sqrt(max(1.0 - alpha_prev, 0.0))
        - latent_coefficient * math.sqrt(max(1.0 - alpha_t, 0.0))
    )
    return latent_coefficient, model_output_coefficient


class ActionAuditRecorder:
    def __init__(self, output_dir: str, session_id: str, version: COVRVersion,
                 max_events: Optional[int] = None):
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.version = version
        self.max_events = max_events
        self.event_path = self.output_dir / f"events_{session_id}.jsonl"
        self.summary_path = self.output_dir / f"summary_{session_id}.json"
        self._handle = self.event_path.open("a", encoding="utf-8")
        self.count = 0
        self.sample_count = 0
        self.numerator_sum = 0.0
        self.denominator_sum = 0.0
        self.ratio_sum = 0.0
        self.previous_defect = 0.0
        self.closed = False

    @property
    def enabled(self) -> bool:
        return not self.closed and (
            self.max_events is None or self.count < self.max_events)

    def record(self, event: ActionAuditEvent) -> bool:
        if self.closed:
            raise RuntimeError("recorder is closed")
        if not self.enabled:
            return False
        if event.session_id != self.session_id:
            raise ValueError("event session does not match recorder")
        if event.version_key != self.version.key:
            raise ValueError("event version does not match recorder")
        self._handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        self._handle.flush()
        transition = event.one_step_transition
        self.count += 1
        self.sample_count += transition.sample_count
        self.numerator_sum += sum(transition.numerators)
        self.denominator_sum += sum(transition.denominators)
        self.ratio_sum += sum(transition.ratios)
        self.previous_defect = transition.mean_ratio
        return True

    def summary(self) -> Dict[str, Any]:
        denominator = max(self.sample_count, 1)
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "event_type": "batch_step_audit",
            "session_id": self.session_id,
            "version": asdict(self.version),
            "version_key": self.version.key,
            "events": self.count,
            "samples": self.sample_count,
            "mean_numerator": self.numerator_sum / denominator,
            "mean_denominator": self.denominator_sum / denominator,
            "mean_one_step_defect": self.ratio_sum / denominator,
            "event_path": str(self.event_path),
        }

    def close(self) -> Dict[str, Any]:
        if not self.closed:
            self._handle.close()
            self.closed = True
            _atomic_json_dump(self.summary_path, self.summary())
        return self.summary()

    def __enter__(self) -> "ActionAuditRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_action_audits(paths: Iterable[str]) -> List[ActionAuditEvent]:
    events: List[ActionAuditEvent] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(ActionAuditEvent.from_dict(json.loads(line)))
    return events
