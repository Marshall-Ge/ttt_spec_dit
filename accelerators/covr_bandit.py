# -*- coding: utf-8 -*-
"""Equal-FLOPs trajectory template selection for SpecA.

EXPERIMENTAL: ``ConservativeTemplateBandit`` (and the strategy-bandit
variant built on ``StrategyManifest``) is an adaptive policy backend used
only through the optional COVR runtime boundary. Its selection/prior/state
semantics are frozen for the runtime extraction; algorithmic changes are
out of scope for the runtime plugin and must be validated separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


BANDIT_SCHEMA_VERSION = 1
# Contextual (LinUCB, deferred-commit) bandit state is not interchangeable with
# the non-contextual state: it carries per-arm (A, b) matrices and per-assignment
# context vectors. Resume across the two must fail loudly, never silently migrate.
CONTEXTUAL_BANDIT_SCHEMA_VERSION = 2
STRATEGY_SCHEMA_VERSION = 2
_LOG_EPSILON = 1e-12


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(handle.name, path)
    except Exception:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class TimestepSafetyPrior:
    step_idx: int
    log_numerator_mean: float
    log_numerator_std: float
    log_denominator_mean: float
    log_denominator_std: float
    sample_count: int

    def __post_init__(self) -> None:
        values = (
            self.log_numerator_mean,
            self.log_numerator_std,
            self.log_denominator_mean,
            self.log_denominator_std,
        )
        if self.step_idx < 0:
            raise ValueError("safety-prior step_idx must be non-negative")
        if self.sample_count <= 0:
            raise ValueError("safety-prior sample_count must be positive")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("safety-prior moments must be finite")
        if self.log_numerator_std < 0 or self.log_denominator_std < 0:
            raise ValueError("safety-prior standard deviations must be non-negative")


@dataclass(frozen=True)
class RefreshTemplate:
    template_id: str
    refresh_mask: Tuple[bool, ...]
    modeled_full_block_equivalents: int
    source: str = ""

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("template_id must be non-empty")
        if not self.refresh_mask:
            raise ValueError("refresh_mask must be non-empty")
        if any(type(value) is not bool for value in self.refresh_mask):
            raise ValueError("refresh_mask values must be booleans")
        if self.modeled_full_block_equivalents <= 0:
            raise ValueError("modeled FLOPs must be positive")

    @property
    def refresh_count(self) -> int:
        return sum(self.refresh_mask)

    @property
    def mask_hash(self) -> str:
        encoded = "".join("1" if value else "0" for value in self.refresh_mask)
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "refresh_mask": list(self.refresh_mask),
            "modeled_full_block_equivalents": self.modeled_full_block_equivalents,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefreshTemplate":
        return cls(
            template_id=str(value["template_id"]),
            refresh_mask=tuple(value["refresh_mask"]),
            modeled_full_block_equivalents=int(
                value["modeled_full_block_equivalents"]),
            source=str(value.get("source", "")),
        )

    def to_strategy(self, num_steps: int) -> "AccelerationStrategy":
        """Convert to a method-agnostic AccelerationStrategy."""
        return AccelerationStrategy(
            strategy_id=self.template_id,
            method="speca",
            params={
                "refresh_mask": list(self.refresh_mask),
                "refresh_count": self.refresh_count,
                "num_steps": num_steps,
            },
            modeled_flops=float(self.modeled_full_block_equivalents),
            source=self.source,
        )


@dataclass(frozen=True)
class AccelerationStrategy:
    """Method-agnostic acceleration strategy descriptor.

    This is the core abstraction for the COVR bandit — a single
    ``AccelerationStrategy`` represents one arm (e.g. one TeaCache
    threshold or one SpecA refresh mask). The bandit selects among
    strategies, and ``apply_strategy()`` (in ``strategy_dispatch.py``)
    translates the selection into the concrete accelerator configuration.

    Parameters
    ----------
    strategy_id : str
        Unique identifier across the manifest.
    method : str
        Accelerator method, e.g. ``"speca"`` or ``"teacache"``.
    params : Dict[str, Any]
        Method-specific parameters (e.g. ``{"refresh_mask": [...]}``
        for SpecA, ``{"rel_l1_thresh": 0.25}`` for TeaCache).
    modeled_flops : float
        Modeled FLOPs for this strategy (used for cost-aware bandit
        decisions and FLOPs accounting).
    source : str
        Optional provenance tag.
    """
    strategy_id: str
    method: str
    params: Dict[str, Any]
    modeled_flops: float
    source: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        # Method validity is delegated to the adapter registry — a method is
        # supported iff an AcceleratorAdapter is registered for it. This is
        # the pluggable seam: new accelerators register an adapter and become
        # valid strategies without editing this file.
        from .registry import is_registered, registered_methods
        if not is_registered(self.method):
            raise ValueError(
                f"unsupported acceleration method: {self.method} "
                f"(registered: {registered_methods()})")
        if self.modeled_flops <= 0 and not math.isclose(self.modeled_flops, 0.0):
            raise ValueError("modeled_flops must be non-negative")

    @property
    def template_id(self) -> str:
        """Alias for strategy_id — enables duck-type compatibility with bandit internals."""
        return self.strategy_id

    @property
    def refresh_mask(self) -> Optional[Tuple[bool, ...]]:
        """Refresh mask (per-step calc/skip schedule), if this strategy has one.

        Both SpecA and forced-schedule TeaCache strategies carry a
        ``refresh_mask`` in their params — it is the method-agnostic
        "which timesteps recompute vs reuse cache" plan. Returns ``None``
        when the strategy has no mask (e.g. a threshold-based TeaCache arm).
        """
        raw = self.params.get("refresh_mask")
        if raw is None:
            return None
        return tuple(raw)

    @property
    def refresh_count(self) -> int:
        if self.refresh_mask is not None:
            return sum(self.refresh_mask)
        return 0

    def to_refresh_template(self) -> RefreshTemplate:
        """Backward compat: convert SpecA strategy to a RefreshTemplate."""
        if self.method != "speca":
            raise RuntimeError(
                f"cannot convert {self.method} strategy to RefreshTemplate")
        mask = self.refresh_mask
        if mask is None:
            raise RuntimeError("SpecA strategy missing refresh_mask")
        return RefreshTemplate(
            template_id=self.strategy_id,
            refresh_mask=mask,
            modeled_full_block_equivalents=int(self.modeled_flops),
            source=self.source,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "method": self.method,
            "params": dict(self.params),
            "modeled_flops": self.modeled_flops,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AccelerationStrategy":
        return cls(
            strategy_id=str(value["strategy_id"]),
            method=str(value["method"]),
            params=dict(value.get("params", {})),
            modeled_flops=float(value["modeled_flops"]),
            source=str(value.get("source", "")),
        )

    @classmethod
    def from_refresh_template(
        cls, template: RefreshTemplate, num_steps: int
    ) -> "AccelerationStrategy":
        return template.to_strategy(num_steps)


@dataclass(frozen=True)
class StrategyManifest:
    """A simpler manifest for method-agnostic acceleration strategies.

    Unlike ``TemplateManifest`` (which is SpecA-specific and validates
    mandatory_prefix, max_taylor_gap, per-step safety priors, etc.),
    ``StrategyManifest`` holds only the essential structure needed by
    the bandit: a list of strategies, a baseline, and a version key.

    For SpecA use-cases, prefer ``TemplateManifest`` (which is also
    accepted by the bandit).  For TeaCache (or future methods) use
    this manifest.
    """
    version_key: str
    num_steps: int
    baseline_strategy_id: str
    strategies: Tuple[AccelerationStrategy, ...]
    schema_version: int = STRATEGY_SCHEMA_VERSION
    source_groups: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.version_key:
            raise ValueError("version_key must be non-empty")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not self.strategies:
            raise ValueError("manifest must contain at least one strategy")
        ids = [s.strategy_id for s in self.strategies]
        if len(set(ids)) != len(ids):
            raise ValueError("strategy IDs must be unique")
        if self.baseline_strategy_id not in ids:
            raise ValueError("baseline strategy is missing")

    @property
    def strategy_map(self) -> Dict[str, AccelerationStrategy]:
        return {s.strategy_id: s for s in self.strategies}

    # -- Duck-type properties so StrategyManifest is compatible with
    # -- ConservativeTemplateBandit / TimestepSafetyTable internals.
    @property
    def baseline_template_id(self) -> str:
        return self.baseline_strategy_id

    @property
    def template_map(self) -> Dict[str, AccelerationStrategy]:
        return self.strategy_map

    @property
    def templates(self) -> Tuple[AccelerationStrategy, ...]:
        return self.strategies

    @property
    def prior_map(self) -> Dict[int, object]:
        return {}

    @property
    def timestep_priors(self) -> Tuple:
        return ()

    @property
    def safety_numerator_ucb_limit(self) -> float:
        return float("inf")

    @property
    def safety_denominator_lcb_floor(self) -> float:
        return 0.0

    @property
    def common_refresh_count(self) -> int:
        return 0

    @property
    def common_full_block_equivalents(self) -> int:
        return 0

    @property
    def num_layers(self) -> int:
        return 0

    @property
    def mandatory_prefix(self) -> int:
        return 0

    @property
    def max_taylor_gap(self) -> int:
        return 0

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version_key": self.version_key,
            "num_steps": self.num_steps,
            "baseline_strategy_id": self.baseline_strategy_id,
            "strategies": [s.to_dict() for s in self.strategies],
            "source_groups": list(self.source_groups),
        }

    def save(self, path: str) -> None:
        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyManifest":
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            version_key=str(value["version_key"]),
            num_steps=int(value["num_steps"]),
            baseline_strategy_id=str(value["baseline_strategy_id"]),
            strategies=tuple(
                AccelerationStrategy.from_dict(s)
                for s in value["strategies"]
            ),
            source_groups=tuple(
                str(g) for g in value.get("source_groups", [])),
        )

    @classmethod
    def load(cls, path: str) -> "StrategyManifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class TemplateManifest:
    version_key: str
    num_steps: int
    num_layers: int
    mandatory_prefix: int
    max_taylor_gap: int
    baseline_template_id: str
    templates: Tuple[RefreshTemplate, ...]
    timestep_priors: Tuple[TimestepSafetyPrior, ...] = ()
    safety_numerator_ucb_limit: float = 1.0
    safety_denominator_lcb_floor: float = 1e-8
    schema_version: int = BANDIT_SCHEMA_VERSION
    source_groups: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != BANDIT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported template schema version: {self.schema_version}")
        if not self.version_key:
            raise ValueError("version_key must be non-empty")
        if self.num_steps <= 0 or self.num_layers <= 0:
            raise ValueError("num_steps and num_layers must be positive")
        if not 1 <= self.mandatory_prefix <= self.num_steps:
            raise ValueError("mandatory_prefix must be within the trajectory")
        if self.max_taylor_gap <= 0:
            raise ValueError("max_taylor_gap must be positive")
        if not self.templates:
            raise ValueError("manifest must contain at least one template")
        if not math.isfinite(self.safety_denominator_lcb_floor):
            raise ValueError("denominator safety floor must be finite")
        if self.safety_denominator_lcb_floor < 0:
            raise ValueError("denominator safety floor must be non-negative")
        if (not math.isfinite(self.safety_numerator_ucb_limit)
                or self.safety_numerator_ucb_limit < 0):
            raise ValueError("numerator safety limit must be finite and non-negative")

        ids = [template.template_id for template in self.templates]
        hashes = [template.mask_hash for template in self.templates]
        if len(set(ids)) != len(ids):
            raise ValueError("template IDs must be unique")
        if len(set(hashes)) != len(hashes):
            raise ValueError("template masks must be unique")
        if self.baseline_template_id not in ids:
            raise ValueError("baseline template is missing")

        refresh_counts = {template.refresh_count for template in self.templates}
        modeled_costs = {
            template.modeled_full_block_equivalents
            for template in self.templates
        }
        if len(refresh_counts) != 1 or len(modeled_costs) != 1:
            raise ValueError("all templates must have exactly equal FLOPs")

        expected_cost = next(iter(refresh_counts)) * self.num_layers
        if modeled_costs != {expected_cost}:
            raise ValueError(
                "modeled block FLOPs must equal refresh_count * num_layers")

        for template in self.templates:
            if len(template.refresh_mask) != self.num_steps:
                raise ValueError("template mask length must match num_steps")
            if not all(template.refresh_mask[:self.mandatory_prefix]):
                raise ValueError("template omits a mandatory prefix refresh")
            longest_gap = 0
            gap = 0
            for refresh in template.refresh_mask:
                if refresh:
                    gap = 0
                else:
                    gap += 1
                    longest_gap = max(longest_gap, gap)
            if longest_gap > self.max_taylor_gap:
                raise ValueError("template exceeds max_taylor_gap")

        prior_steps = [prior.step_idx for prior in self.timestep_priors]
        if len(set(prior_steps)) != len(prior_steps):
            raise ValueError("timestep safety priors must have unique steps")
        if any(step >= self.num_steps for step in prior_steps):
            raise ValueError("timestep safety prior is outside the trajectory")

    @property
    def common_refresh_count(self) -> int:
        return self.templates[0].refresh_count

    @property
    def common_full_block_equivalents(self) -> int:
        return self.templates[0].modeled_full_block_equivalents

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def template_map(self) -> Dict[str, RefreshTemplate]:
        return {template.template_id: template for template in self.templates}

    @property
    def prior_map(self) -> Dict[int, TimestepSafetyPrior]:
        return {prior.step_idx: prior for prior in self.timestep_priors}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version_key": self.version_key,
            "num_steps": self.num_steps,
            "num_layers": self.num_layers,
            "mandatory_prefix": self.mandatory_prefix,
            "max_taylor_gap": self.max_taylor_gap,
            "baseline_template_id": self.baseline_template_id,
            "templates": [template.to_dict() for template in self.templates],
            "timestep_priors": [asdict(prior) for prior in self.timestep_priors],
            "safety_numerator_ucb_limit": self.safety_numerator_ucb_limit,
            "safety_denominator_lcb_floor": self.safety_denominator_lcb_floor,
            "source_groups": list(self.source_groups),
        }

    def save(self, path: str) -> None:
        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemplateManifest":
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            version_key=str(value["version_key"]),
            num_steps=int(value["num_steps"]),
            num_layers=int(value["num_layers"]),
            mandatory_prefix=int(value["mandatory_prefix"]),
            max_taylor_gap=int(value["max_taylor_gap"]),
            baseline_template_id=str(value["baseline_template_id"]),
            templates=tuple(
                RefreshTemplate.from_dict(template)
                for template in value["templates"]
            ),
            timestep_priors=tuple(
                TimestepSafetyPrior(**prior)
                for prior in value.get("timestep_priors", [])
            ),
            safety_numerator_ucb_limit=float(
                value.get("safety_numerator_ucb_limit", 1.0)),
            safety_denominator_lcb_floor=float(
                value.get("safety_denominator_lcb_floor", 1e-8)),
            source_groups=tuple(str(group) for group in value.get("source_groups", [])),
        )

    @classmethod
    def load(cls, path: str) -> "TemplateManifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class TemplateAssignment:
    session_id: str
    trajectory_id: int
    prequential_index: int
    # template_id is Optional only for a deferred-commit "pending" assignment
    # (contextual mode, before commit_arm selects the arm). A committed
    # assignment always carries a real strategy id.
    template_id: Optional[str]
    propensity: float
    manifest_hash: str
    sample_count: int = 0
    # Per-trajectory context vector used by the contextual bandit. None on
    # non-contextual assignments and on pending (pre-commit) assignments.
    context: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.trajectory_id < 0 or self.prequential_index < 0:
            raise ValueError("trajectory indices must be non-negative")
        if not 0 < self.propensity <= 1:
            raise ValueError("assignment propensity must be in (0, 1]")
        if self.sample_count < 0:
            raise ValueError("assignment sample_count must be non-negative")

    @property
    def strategy_id(self) -> str:
        """Method-agnostic alias for template_id."""
        return self.template_id


@dataclass(frozen=True)
class TemplateFeedback:
    trajectory_id: int
    template_id: str
    sentinel_propensity: float
    horizon: int
    h_step_numerator: Optional[float] = None
    h_step_denominator: Optional[float] = None
    terminal_fidelity_loss: Optional[float] = None
    terminal_quality_loss: Optional[float] = None
    # Normalized measured cost (∈[0,1], fraction of vanilla FLOPs). Efficiency-
    # aware reward only; None on legacy / quality-only feedback. Not part of the
    # all-None guard (kept separate so v1 feedback still validates).
    terminal_efficiency_loss: Optional[float] = None
    # Pre-combined loss the bandit actually optimizes (terminal_fidelity +
    # lambda*cost). Set by the runtime in efficiency-aware mode; bandit_loss
    # prefers it so the legacy path (None) falls back to fidelity/quality/h-step.
    combined_loss: Optional[float] = None

    def __post_init__(self) -> None:
        if self.trajectory_id < 0 or self.horizon <= 0:
            raise ValueError("feedback indices must be positive")
        if not 0 < self.sentinel_propensity <= 1:
            raise ValueError("sentinel propensity must be in (0, 1]")
        values = (
            self.h_step_numerator,
            self.h_step_denominator,
            self.terminal_fidelity_loss,
            self.terminal_quality_loss,
        )
        if all(value is None for value in values):
            raise ValueError("feedback must contain a delayed sentinel label")
        if any(value is not None and (not math.isfinite(value) or value < 0)
               for value in values):
            raise ValueError("sentinel labels must be finite and non-negative")
        for extra in (self.terminal_efficiency_loss, self.combined_loss):
            if extra is not None and (not math.isfinite(extra) or extra < 0):
                raise ValueError(
                    "efficiency/combined labels must be finite and non-negative")
        if (self.h_step_numerator is None) != (self.h_step_denominator is None):
            raise ValueError("H-step numerator and denominator must be paired")

    @property
    def strategy_id(self) -> str:
        """Method-agnostic alias for template_id."""
        return self.template_id

    @property
    def bandit_loss(self) -> float:
        if self.combined_loss is not None:
            return self.combined_loss
        if self.terminal_quality_loss is not None:
            return self.terminal_quality_loss
        if self.terminal_fidelity_loss is not None:
            return self.terminal_fidelity_loss
        assert self.h_step_numerator is not None
        return self.h_step_numerator


@dataclass
class _RunningLogMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("safety observations must be finite and non-negative")
        logged = math.log(max(value, _LOG_EPSILON))
        self.count += 1
        delta = logged - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (logged - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.m2 / (self.count - 1), 0.0))

    def state_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "_RunningLogMoments":
        result = cls(
            count=int(value["count"]),
            mean=float(value["mean"]),
            m2=float(value["m2"]),
        )
        if result.count < 0 or result.m2 < 0:
            raise ValueError("invalid persisted moments")
        return result


@dataclass
class _SafetyMoments:
    numerator: _RunningLogMoments = field(default_factory=_RunningLogMoments)
    denominator: _RunningLogMoments = field(default_factory=_RunningLogMoments)

    def update(self, numerator: float, denominator: float) -> None:
        self.numerator.update(numerator)
        self.denominator.update(denominator)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "numerator": self.numerator.state_dict(),
            "denominator": self.denominator.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "_SafetyMoments":
        return cls(
            numerator=_RunningLogMoments.from_state_dict(value["numerator"]),
            denominator=_RunningLogMoments.from_state_dict(value["denominator"]),
        )


class TimestepSafetyTable:
    def __init__(self, manifest: TemplateManifest, confidence_z: float = 2.0,
                 min_template_samples: int = 2):
        if confidence_z < 0 or min_template_samples <= 0:
            raise ValueError("invalid safety-table configuration")
        self.manifest = manifest
        self.confidence_z = float(confidence_z)
        self.min_template_samples = int(min_template_samples)
        self._by_template_step: Dict[Tuple[str, int], _SafetyMoments] = {}
        self._by_step: Dict[int, _SafetyMoments] = {}

    def observe(self, template_id: str, step_idx: int,
                numerator: float, denominator: float) -> None:
        if template_id not in self.manifest.template_map:
            raise ValueError("unknown template ID")
        if not 0 <= step_idx < self.manifest.num_steps:
            raise ValueError("safety observation step is outside the trajectory")
        self._by_template_step.setdefault(
            (template_id, step_idx), _SafetyMoments()).update(
                numerator, denominator)
        self._by_step.setdefault(step_idx, _SafetyMoments()).update(
            numerator, denominator)

    def _moments(self, template_id: str, step_idx: int) -> Tuple[float, float, float, float, int]:
        exact = self._by_template_step.get((template_id, step_idx))
        if exact is not None and exact.numerator.count >= self.min_template_samples:
            return (
                exact.numerator.mean, exact.numerator.std,
                exact.denominator.mean, exact.denominator.std,
                exact.numerator.count,
            )
        pooled = self._by_step.get(step_idx)
        if pooled is not None and pooled.numerator.count >= self.min_template_samples:
            return (
                pooled.numerator.mean, pooled.numerator.std,
                pooled.denominator.mean, pooled.denominator.std,
                pooled.numerator.count,
            )
        prior = self.manifest.prior_map.get(step_idx)
        if prior is None:
            return math.inf, 0.0, -math.inf, 0.0, 1
        return (
            prior.log_numerator_mean, prior.log_numerator_std,
            prior.log_denominator_mean, prior.log_denominator_std,
            prior.sample_count,
        )

    def bounds(self, template_id: str, step_idx: int) -> Tuple[float, float]:
        num_mean, num_std, den_mean, den_std, count = self._moments(
            template_id, step_idx)
        if not math.isfinite(num_mean) or not math.isfinite(den_mean):
            return math.inf, 0.0
        scale = self.confidence_z / math.sqrt(max(count, 1))
        numerator_ucb = math.exp(num_mean + scale * num_std)
        denominator_lcb = math.exp(den_mean - scale * den_std)
        return numerator_ucb, denominator_lcb

    def is_safe(self, template: RefreshTemplate) -> bool:
        for step_idx, refresh in enumerate(template.refresh_mask):
            if refresh:
                continue
            numerator_ucb, denominator_lcb = self.bounds(
                template.template_id, step_idx)
            if numerator_ucb > self.manifest.safety_numerator_ucb_limit:
                return False
            if denominator_lcb < self.manifest.safety_denominator_lcb_floor:
                return False
        return True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "by_template_step": {
                f"{template_id}:{step_idx}": moments.state_dict()
                for (template_id, step_idx), moments in self._by_template_step.items()
            },
            "by_step": {
                str(step_idx): moments.state_dict()
                for step_idx, moments in self._by_step.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        self._by_template_step = {}
        for key, moments in value.get("by_template_step", {}).items():
            template_id, step_text = key.rsplit(":", 1)
            if template_id not in self.manifest.template_map:
                raise ValueError("persisted safety state references an unknown template")
            self._by_template_step[(template_id, int(step_text))] = (
                _SafetyMoments.from_state_dict(moments))
        self._by_step = {
            int(step): _SafetyMoments.from_state_dict(moments)
            for step, moments in value.get("by_step", {}).items()
        }


@dataclass
class _ArmLossStats:
    count: int
    mean: float
    m2: float = 0.0

    def update(self, loss: float) -> None:
        logged = math.log1p(loss)
        self.count += 1
        delta = logged - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (logged - self.mean)

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "_ArmLossStats":
        result = cls(
            count=int(value["count"]), mean=float(value["mean"]),
            m2=float(value.get("m2", 0.0)),
        )
        if result.count <= 0 or result.m2 < 0 or not math.isfinite(result.mean):
            raise ValueError("invalid persisted arm statistics")
        return result


class ConservativeTemplateBandit:
    def __init__(self, manifest: TemplateManifest, session_id: str,
                 epsilon: float = 0.1, seed: int = 0,
                 baseline_prior_count: int = 8,
                 alternative_prior_penalty: float = 0.0,
                 run_identity: Optional[Mapping[str, Any]] = None):
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if baseline_prior_count <= 0 or alternative_prior_penalty < 0:
            raise ValueError("invalid bandit prior configuration")
        self.manifest = manifest
        self.session_id = session_id
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.run_identity = dict(run_identity or {})
        self.safety = TimestepSafetyTable(manifest)
        self._rng = np.random.default_rng(seed)

        # Internal list of AccelerationStrategy objects (method-agnostic).
        # Converted from templates for backward compat; for StrategyManifest
        # the from_strategies classmethod sets this directly.
        self._strategies: list[AccelerationStrategy] = [
            t.to_strategy(manifest.num_steps) for t in manifest.templates
        ]

        self._arm_stats = {
            strategy.strategy_id: _ArmLossStats(
                count=(baseline_prior_count
                       if strategy.strategy_id == manifest.baseline_template_id
                       else 1),
                mean=(0.0
                      if strategy.strategy_id == manifest.baseline_template_id
                      else alternative_prior_penalty),
            )
            for strategy in self._strategies
        }
        self._active: Optional[TemplateAssignment] = None
        self._pending_safety: list[Tuple[int, float, float]] = []
        self._completed_trajectories: set[int] = set()
        self.assignments: list[TemplateAssignment] = []
        self.feedback: list[TemplateFeedback] = []

    @classmethod
    def from_strategies(
        cls, manifest: StrategyManifest, session_id: str,
        epsilon: float = 0.1, seed: int = 0,
        baseline_prior_count: int = 8,
        alternative_prior_penalty: float = 0.0,
        run_identity: Optional[Mapping[str, Any]] = None,
    ) -> "ConservativeTemplateBandit":
        """Create a bandit from a method-agnostic StrategyManifest.

        Safety table is created but never queried — Theorem A (no
        step-level safety) means observation is skipped at the runner
        level for non-SpecA methods.
        """
        instance = cls.__new__(cls)
        instance.manifest = manifest
        instance.session_id = session_id
        instance.epsilon = float(epsilon)
        instance.seed = int(seed)
        instance.run_identity = dict(run_identity or {})
        instance.safety = TimestepSafetyTable(manifest)
        instance._rng = np.random.default_rng(seed)
        instance._strategies = list(manifest.strategies)
        instance._arm_stats = {
            s.strategy_id: _ArmLossStats(
                count=(baseline_prior_count
                       if s.strategy_id == manifest.baseline_strategy_id
                       else 1),
                mean=(0.0
                      if s.strategy_id == manifest.baseline_strategy_id
                      else alternative_prior_penalty),
            )
            for s in manifest.strategies
        }
        instance._active = None
        instance._pending_safety = []
        instance._completed_trajectories = set()
        instance.assignments = []
        instance.feedback = []
        return instance

    def _eligible_templates(self) -> list:
        """Return arms eligible for selection.

        For TemplateManifest (SpecA), this applies the safety gate.
        For StrategyManifest (method-agnostic, e.g. TeaCache),
        all strategies are eligible (no step-level safety).
        """
        if isinstance(self.manifest, StrategyManifest):
            return list(self._strategies)
        baseline_id = self.manifest.baseline_template_id
        return [
            template for template in self.manifest.templates
            if template.template_id == baseline_id or self.safety.is_safe(template)
        ]

    def begin_trajectory(self, trajectory_id: int,
                         sample_count: int = 0) -> TemplateAssignment:
        if self._active is not None:
            raise RuntimeError("the previous trajectory is still active")
        if trajectory_id in self._completed_trajectories:
            raise ValueError("trajectory has already completed")
        if trajectory_id < 0:
            raise ValueError("trajectory_id must be non-negative")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")

        eligible = self._eligible_templates()
        greedy = min(
            eligible,
            key=lambda template: (
                self._arm_stats[template.template_id].mean,
                template.template_id != self.manifest.baseline_template_id,
                template.template_id,
            ),
        )
        probabilities = np.full(len(eligible), self.epsilon / len(eligible))
        greedy_index = eligible.index(greedy)
        probabilities[greedy_index] += 1.0 - self.epsilon
        selected_index = int(self._rng.choice(len(eligible), p=probabilities))
        selected = eligible[selected_index]
        assignment = TemplateAssignment(
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            prequential_index=len(self._completed_trajectories),
            template_id=selected.template_id,
            propensity=float(probabilities[selected_index]),
            manifest_hash=self.manifest.manifest_hash,
            sample_count=int(sample_count),
        )
        self._active = assignment
        self._pending_safety = []
        self.assignments.append(assignment)
        return assignment

    def observe_one_step(self, trajectory_id: int, step_idx: int,
                         numerators: Sequence[float],
                         denominators: Sequence[float]) -> None:
        if self._active is None or self._active.trajectory_id != trajectory_id:
            raise RuntimeError("one-step observation does not match the active trajectory")
        if len(numerators) == 0 or len(numerators) != len(denominators):
            raise ValueError("one-step labels must have equal non-zero cardinality")
        numerator = float(np.mean(np.asarray(numerators, dtype=np.float64)))
        denominator = float(np.mean(np.asarray(denominators, dtype=np.float64)))
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise ValueError("one-step labels must be finite")
        self._pending_safety.append((int(step_idx), numerator, denominator))

    def end_trajectory(self, trajectory_id: int,
                       feedback: Optional[TemplateFeedback] = None) -> None:
        if self._active is None or self._active.trajectory_id != trajectory_id:
            raise RuntimeError("trajectory close does not match the active assignment")
        if feedback is not None:
            if feedback.trajectory_id != trajectory_id:
                raise ValueError("feedback trajectory does not match assignment")
            if feedback.template_id != self._active.template_id:
                raise ValueError("feedback template does not match assignment")

        for step_idx, numerator, denominator in self._pending_safety:
            self.safety.observe(
                self._active.template_id, step_idx, numerator, denominator)
        if feedback is not None:
            self._arm_stats[self._active.template_id].update(feedback.bandit_loss)
            self.feedback.append(feedback)

        self._completed_trajectories.add(trajectory_id)
        self._active = None
        self._pending_safety = []

    @property
    def active_template(self) -> RefreshTemplate:
        """Return the active arm as a RefreshTemplate (SpecA-only compat).

        Raises ``RuntimeError`` if the active strategy is not SpecA.
        """
        return self.active_strategy.to_refresh_template()

    @property
    def is_contextual(self) -> bool:
        """Whether this bandit defers arm selection to commit_arm."""
        return False

    @property
    def _schema_version(self) -> int:
        """Persisted-state schema version; the contextual bandit overrides this."""
        return BANDIT_SCHEMA_VERSION

    def commit_arm(self, trajectory_id: int,
                   context: Sequence[float]) -> TemplateAssignment:
        """Select the arm for the active trajectory given a context vector.

        Only the contextual (LinUCB, deferred-commit) bandit implements this.
        The non-contextual bandit selects the arm in ``begin_trajectory`` and
        raises here so a missing contextual flag fails loudly rather than
        silently double-selecting.
        """
        raise NotImplementedError(
            "commit_arm is only supported by the contextual bandit")

    @property
    def active_strategy(self) -> AccelerationStrategy:
        """Return the active arm as a method-agnostic AccelerationStrategy."""
        if self._active is None:
            raise RuntimeError("no trajectory is active")
        strategy_id = self._active.template_id
        if strategy_id is None:
            raise RuntimeError(
                "active arm is pending commit; no strategy is selected yet")
        if isinstance(self.manifest, StrategyManifest):
            return self.manifest.strategy_map[strategy_id]
        # TemplateManifest: look up from the internal _strategies list
        for strategy in self._strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        raise KeyError(f"active strategy {strategy_id} not found in _strategies")

    def summary(self) -> Dict[str, Any]:
        if isinstance(self.manifest, StrategyManifest):
            arm_ids = [s.strategy_id for s in self.manifest.strategies]
        else:
            arm_ids = [t.template_id for t in self.manifest.templates]
        assignment_counts = {
            arm_id: sum(
                assignment.template_id == arm_id
                for assignment in self.assignments)
            for arm_id in arm_ids
        }
        summary = {
            "session_id": self.session_id,
            "manifest_hash": self.manifest.manifest_hash,
            "assignments": len(self.assignments),
            "processed_samples": sum(
                assignment.sample_count for assignment in self.assignments),
            "completed_trajectories": len(self._completed_trajectories),
            "delayed_feedback": len(self.feedback),
            "assignment_counts": assignment_counts,
            "arm_log1p_loss_mean": {
                arm_id: stats.mean
                for arm_id, stats in self._arm_stats.items()
            },
        }
        if not isinstance(self.manifest, StrategyManifest):
            summary.update({
                "common_refresh_count": self.manifest.common_refresh_count,
                "common_full_block_equivalents": (
                    self.manifest.common_full_block_equivalents),
            })
        return summary

    def state_dict(self) -> Dict[str, Any]:
        if self._active is not None:
            raise RuntimeError("cannot persist bandit state during an active trajectory")
        return {
            "schema_version": self._schema_version,
            "session_id": self.session_id,
            "version_key": self.manifest.version_key,
            "manifest_hash": self.manifest.manifest_hash,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "run_identity": self.run_identity,
            "rng_state": self._rng.bit_generator.state,
            "arm_stats": {
                template_id: stats.state_dict()
                for template_id, stats in self._arm_stats.items()
            },
            "safety": self.safety.state_dict(),
            "completed_trajectories": sorted(self._completed_trajectories),
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "feedback": [asdict(item) for item in self.feedback],
        }

    def save_state(self, path: str) -> None:
        _atomic_json_write(Path(path), self.state_dict())

    def load_state(self, path: str) -> None:
        if self._active is not None:
            raise RuntimeError("cannot load state during an active trajectory")
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        identity = (
            int(value.get("schema_version", 0)),
            str(value.get("session_id", "")),
            str(value.get("version_key", "")),
            str(value.get("manifest_hash", "")),
        )
        expected = (
            self._schema_version,
            self.session_id,
            self.manifest.version_key,
            self.manifest.manifest_hash,
        )
        if identity != expected:
            raise ValueError("bandit state identity does not match this session")
        if not math.isclose(float(value["epsilon"]), self.epsilon):
            raise ValueError("persisted epsilon does not match runtime configuration")
        if dict(value.get("run_identity", {})) != self.run_identity:
            raise ValueError("bandit state run identity does not match runtime configuration")

        arm_stats = {
            template_id: _ArmLossStats.from_state_dict(stats)
            for template_id, stats in value["arm_stats"].items()
        }
        if set(arm_stats) != set(self.manifest.template_map):
            raise ValueError("persisted arm set does not match the manifest")
        self._arm_stats = arm_stats
        self.safety.load_state_dict(value.get("safety", {}))
        self._completed_trajectories = {
            int(item) for item in value.get("completed_trajectories", [])
        }
        self.assignments = [
            TemplateAssignment(**assignment)
            for assignment in value.get("assignments", [])
        ]
        self.feedback = [
            TemplateFeedback(**feedback)
            for feedback in value.get("feedback", [])
        ]
        self._rng.bit_generator.state = value["rng_state"]


class ContextualLinUCBBandit(ConservativeTemplateBandit):
    """Deferred-commit LinUCB bandit over a ``StrategyManifest``.

    The non-contextual bandit selects the arm in ``begin_trajectory`` and so
    collapses onto the globally-best arm (which, for c2i terminal fidelity, *is*
    the baseline — gain ≈ 0 by construction). This bandit defers selection to
    ``commit_arm``: ``begin_trajectory`` opens a *pending* trajectory (no arm),
    the loop runs a mandatory calc prefix and extracts a context vector, then
    ``commit_arm`` picks the arm via a per-arm linear loss model. Selection is
    epsilon-greedy on top of LinUCB so the logged propensity stays non-degenerate
    (IPS-estimable). ``end_trajectory`` updates the chosen arm's ``(A, b)`` with
    the observed ``(context, loss)``.

    The bandit MINIMIZES loss, so exploration uses a lower-confidence bound:
    ``score_a(x) = theta_a.x - alpha * sqrt(x^T A_a^-1 x)`` and the greedy arm is
    ``argmin_a score_a(x)`` (optimistic-low-loss).
    """

    def __init__(self, manifest: StrategyManifest, session_id: str,
                 context_dim: int, epsilon: float = 0.1, seed: int = 0,
                 alpha: float = 1.0, baseline_prior_count: int = 8,
                 alternative_prior_penalty: float = 0.0,
                 run_identity: Optional[Mapping[str, Any]] = None):
        super().__init__(
            manifest, session_id, epsilon=epsilon, seed=seed,
            baseline_prior_count=baseline_prior_count,
            alternative_prior_penalty=alternative_prior_penalty,
            run_identity=run_identity)
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if alpha < 0:
            raise ValueError("linucb alpha must be non-negative")
        self._context_dim = int(context_dim)
        self._alpha = float(alpha)
        self._linucb = self._fresh_linucb(self._context_dim)

    @classmethod
    def from_strategies(
        cls, manifest: StrategyManifest, session_id: str, context_dim: int, *,
        epsilon: float = 0.1, seed: int = 0, alpha: float = 1.0,
        baseline_prior_count: int = 8, alternative_prior_penalty: float = 0.0,
        run_identity: Optional[Mapping[str, Any]] = None,
    ) -> "ContextualLinUCBBandit":
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if baseline_prior_count <= 0 or alternative_prior_penalty < 0:
            raise ValueError("invalid bandit prior configuration")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if alpha < 0:
            raise ValueError("linucb alpha must be non-negative")
        instance = cls.__new__(cls)
        instance.manifest = manifest
        instance.session_id = session_id
        instance.epsilon = float(epsilon)
        instance.seed = int(seed)
        instance.run_identity = dict(run_identity or {})
        instance.safety = TimestepSafetyTable(manifest)
        instance._rng = np.random.default_rng(seed)
        instance._strategies = list(manifest.strategies)
        instance._arm_stats = {
            s.strategy_id: _ArmLossStats(
                count=(baseline_prior_count
                       if s.strategy_id == manifest.baseline_strategy_id else 1),
                mean=(0.0 if s.strategy_id == manifest.baseline_strategy_id
                      else alternative_prior_penalty))
            for s in manifest.strategies
        }
        instance._active = None
        instance._pending_safety = []
        instance._completed_trajectories = set()
        instance.assignments = []
        instance.feedback = []
        instance._context_dim = int(context_dim)
        instance._alpha = float(alpha)
        instance._linucb = instance._fresh_linucb(instance._context_dim)
        return instance

    def _fresh_linucb(self, context_dim: int) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            s.strategy_id: {"A": np.eye(context_dim),
                            "b": np.zeros(context_dim)}
            for s in self._strategies
        }

    @property
    def is_contextual(self) -> bool:
        return True

    @property
    def _schema_version(self) -> int:
        return CONTEXTUAL_BANDIT_SCHEMA_VERSION

    @property
    def context_dim(self) -> int:
        return self._context_dim

    def begin_trajectory(self, trajectory_id: int,
                         sample_count: int = 0) -> TemplateAssignment:
        """Open a pending trajectory; the arm is chosen later by ``commit_arm``."""
        if self._active is not None:
            raise RuntimeError("the previous trajectory is still active")
        if trajectory_id in self._completed_trajectories:
            raise ValueError("trajectory has already completed")
        if trajectory_id < 0:
            raise ValueError("trajectory_id must be non-negative")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        pending = TemplateAssignment(
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            prequential_index=len(self._completed_trajectories),
            template_id=None,
            propensity=1.0,  # placeholder; the real propensity is set at commit
            manifest_hash=self.manifest.manifest_hash,
            sample_count=int(sample_count),
        )
        self._active = pending
        self._pending_safety = []
        # Deliberately NOT appended to self.assignments: commit_arm records the
        # real (committed) assignment so the analyzer sees one arm per trajectory.
        return pending

    def _linucb_argmin(self, context: np.ndarray) -> str:
        """Greedy arm = argmin of the optimistic-low-loss LCB score."""
        best_arm: Optional[str] = None
        best_score = math.inf
        for strategy in self._strategies:
            arm = strategy.strategy_id
            matrices = self._linucb[arm]
            theta = np.linalg.solve(matrices["A"], matrices["b"])
            mean = float(theta @ context)
            a_inv_x = np.linalg.solve(matrices["A"], context)
            variance = float(context @ a_inv_x)
            bonus = math.sqrt(max(0.0, variance))
            score = mean - self._alpha * bonus
            # Deterministic tie-break: manifest order (first wins on strict <).
            if score < best_score:
                best_score = score
                best_arm = arm
        assert best_arm is not None
        return best_arm

    def commit_arm(self, trajectory_id: int,
                   context: Sequence[float]) -> TemplateAssignment:
        if self._active is None or self._active.trajectory_id != trajectory_id:
            raise RuntimeError("commit_arm does not match the active trajectory")
        if self._active.template_id is not None:
            raise RuntimeError("trajectory arm has already been committed")
        ctx = np.asarray(context, dtype=np.float64)
        if ctx.shape != (self._context_dim,):
            raise ValueError(
                f"context shape {ctx.shape} does not match context_dim "
                f"{self._context_dim}")
        if not np.all(np.isfinite(ctx)):
            raise ValueError("context must be finite")

        arms = [s.strategy_id for s in self._strategies]
        greedy = self._linucb_argmin(ctx)
        # epsilon-greedy on top of LinUCB: keeps propensity non-degenerate so the
        # deployed contextual policy is IPS-estimable offline.
        n_arms = len(arms)
        probabilities = np.full(n_arms, self.epsilon / n_arms)
        probabilities[arms.index(greedy)] += 1.0 - self.epsilon
        selected_index = int(self._rng.choice(n_arms, p=probabilities))
        selected = arms[selected_index]

        assignment = TemplateAssignment(
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            prequential_index=self._active.prequential_index,
            template_id=selected,
            propensity=float(probabilities[selected_index]),
            manifest_hash=self.manifest.manifest_hash,
            sample_count=self._active.sample_count,
            context=[float(v) for v in ctx.tolist()],
        )
        self._active = assignment
        self.assignments.append(assignment)
        return assignment

    def end_trajectory(self, trajectory_id: int,
                       feedback: Optional[TemplateFeedback] = None) -> None:
        if self._active is None or self._active.trajectory_id != trajectory_id:
            raise RuntimeError("trajectory close does not match the active assignment")
        if self._active.template_id is None:
            raise RuntimeError("cannot end a contextual trajectory before commit_arm")
        if feedback is not None:
            if feedback.trajectory_id != trajectory_id:
                raise ValueError("feedback trajectory does not match assignment")
            if feedback.template_id != self._active.template_id:
                raise ValueError("feedback template does not match assignment")

        for step_idx, numerator, denominator in self._pending_safety:
            self.safety.observe(
                self._active.template_id, step_idx, numerator, denominator)

        if feedback is not None and self._active.context is not None:
            ctx = np.asarray(self._active.context, dtype=np.float64)
            arm = self._active.template_id
            self._linucb[arm]["A"] = self._linucb[arm]["A"] + np.outer(ctx, ctx)
            self._linucb[arm]["b"] = (
                self._linucb[arm]["b"] + float(feedback.bandit_loss) * ctx)
            self._arm_stats[arm].update(feedback.bandit_loss)
            self.feedback.append(feedback)

        self._completed_trajectories.add(trajectory_id)
        self._active = None
        self._pending_safety = []

    def theta(self, arm_id: str) -> np.ndarray:
        """Current linear loss-model estimate for an arm (analyzer diagnostic)."""
        matrices = self._linucb[arm_id]
        return np.linalg.solve(matrices["A"], matrices["b"])

    def summary(self) -> Dict[str, Any]:
        base = super().summary()
        theta_norm = {
            strategy.strategy_id: float(np.linalg.norm(self.theta(strategy.strategy_id)))
            for strategy in self._strategies
        }
        base.update({
            "contextual": True,
            "context_dim": self._context_dim,
            "linucb_alpha": self._alpha,
            "linucb_theta_norm": theta_norm,
        })
        return base

    def state_dict(self) -> Dict[str, Any]:
        payload = super().state_dict()
        payload["context_dim"] = self._context_dim
        payload["linucb_alpha"] = self._alpha
        payload["linucb"] = {
            arm: {"A": matrices["A"].tolist(),
                  "b": matrices["b"].tolist()}
            for arm, matrices in self._linucb.items()
        }
        return payload

    def load_state(self, path: str) -> None:
        super().load_state(path)  # identity gate (contextual schema version) + base restore
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        linucb_payload = value.get("linucb", {})
        restored: Dict[str, Dict[str, np.ndarray]] = {}
        for strategy in self._strategies:
            arm = strategy.strategy_id
            entry = linucb_payload.get(arm)
            if entry is None:
                # Arm present in manifest but absent in persisted state (e.g.
                # manifest grew): fall back to the uninformative prior.
                restored[arm] = {"A": np.eye(self._context_dim),
                                 "b": np.zeros(self._context_dim)}
                continue
            a_matrix = np.asarray(entry["A"], dtype=np.float64)
            b_vector = np.asarray(entry["b"], dtype=np.float64)
            if a_matrix.shape != (self._context_dim, self._context_dim):
                raise ValueError(
                    f"persisted A for {arm} has shape {a_matrix.shape}")
            if b_vector.shape != (self._context_dim,):
                raise ValueError(
                    f"persisted b for {arm} has shape {b_vector.shape}")
            restored[arm] = {"A": a_matrix, "b": b_vector}
        if set(restored) != {s.strategy_id for s in self._strategies}:
            raise ValueError("persisted linucb arm set does not match the manifest")
        self._linucb = restored
