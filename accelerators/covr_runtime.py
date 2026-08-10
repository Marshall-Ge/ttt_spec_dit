# -*- coding: utf-8 -*-
"""Optional runtime boundary for COVR strategy control.

The runtime owns COVR lifecycle and serialization concerns. Accelerator compute
remains explicit in the model and is configured through the adapter registry.
The adaptive backend is experimental; disabled runs do not construct a runtime.

Phase 1 supports only the DiT main denoise loop (no TTT); PixArt and TTT fail
early in ``validate_covr_capabilities``. VFL is independent and unchanged.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from config import DIT_REPO

from .covr import ActionAuditRecorder, COVRVersion
from .covr_bandit import (
    AccelerationStrategy,
    ConservativeTemplateBandit,
    StrategyManifest,
    TemplateAssignment,
    TemplateFeedback,
    TemplateManifest,
)
from .registry import get_adapter, is_registered
from .strategy_dispatch import apply_strategy
from run_dit_shared import (
    _covr_canonical_json,
    _covr_resume_metadata,
    _covr_scheduler_config_json,
    _GenerationProfiler,
)

# COVR control stages whose wall time is counted as control overhead. The
# denoise loop and the runtime share these exact names (see §13 of CLAUDE.md).
_COVR_CONTROL_STAGES = (
    "strategy_selection",
    "accelerator_state_reset",
    "speca_state_reset",
    "strategy_initialization",
    "feedback_materialization",
    "bandit_update",
    "bandit_state_persist",
)


class COVRMode(str, Enum):
    """Enabled COVR runtime modes.

    Disabled is represented by the absence of a runtime/config object so the
    ordinary inference path does not pay for a null plugin on the hot path.
    """

    OBSERVE = "observe"
    FORCED = "forced"
    EXPERIMENTAL_BANDIT = "experimental_bandit"


@dataclass(frozen=True)
class COVRCapabilities:
    """Capabilities validated before entering the denoise loop."""

    model: str
    method: str
    main_denoise_loop: bool
    shadow_audit: bool
    speca_safety: bool
    terminal_reward: bool


@dataclass(frozen=True)
class COVRRuntimeConfig:
    """Immutable COVR request resolved from the existing CLI surface."""

    mode: COVRMode
    capabilities: COVRCapabilities
    output_dir: str
    bandit_state_path: str
    resume_state_path: Optional[str]
    manifest_path: Optional[str]
    forced_strategy_id: Optional[str]
    uses_legacy_template: bool
    shadow: bool
    profile_stages: bool
    session_id: Optional[str]
    base_model_version: str
    max_events: Optional[int]
    epsilon: float
    safety_sample_rate: float
    safety_chain_threshold: int
    sentinel_rate: float
    sentinel_horizon: int
    seed: int
    bandit_prior_penalty: float = 0.0


@dataclass(frozen=True)
class BackendSelection:
    """Policy output for one trajectory."""

    strategy: AccelerationStrategy
    assignment: Optional[TemplateAssignment]


class COVRPolicyBackend(Protocol):
    """Common lifecycle used by forced and experimental policy backends."""

    @property
    def active_strategy(self) -> Optional[AccelerationStrategy]:
        ...

    def begin_trajectory(
            self, trajectory_id: int, sample_count: int) -> BackendSelection:
        ...

    def observe_one_step(
            self, trajectory_id: int, step_idx: int,
            numerators: Sequence[float], denominators: Sequence[float]) -> None:
        ...

    def end_trajectory(
            self, trajectory_id: int,
            feedback: Optional[TemplateFeedback]) -> None:
        ...

    def persist(self) -> None:
        ...

    def summary(self) -> Dict[str, Any]:
        ...


class ForcedStrategyBackend:
    """Static strategy backend with no learning or persistent state."""

    def __init__(self, strategy: AccelerationStrategy,
                 manifest: Optional[Any] = None):
        self._strategy = strategy
        self.manifest = manifest

    @property
    def active_strategy(self) -> AccelerationStrategy:
        return self._strategy

    def begin_trajectory(
            self, trajectory_id: int, sample_count: int) -> BackendSelection:
        del trajectory_id, sample_count
        return BackendSelection(strategy=self._strategy, assignment=None)

    def observe_one_step(
            self, trajectory_id: int, step_idx: int,
            numerators: Sequence[float], denominators: Sequence[float]) -> None:
        del trajectory_id, step_idx, numerators, denominators

    def end_trajectory(
            self, trajectory_id: int,
            feedback: Optional[TemplateFeedback]) -> None:
        del trajectory_id, feedback

    def persist(self) -> None:
        return None

    def summary(self) -> Dict[str, Any]:
        return {}


class ExperimentalBanditBackend:
    """Compatibility wrapper around the existing experimental bandit."""

    def __init__(self, bandit: ConservativeTemplateBandit, state_path: str):
        self.bandit = bandit
        self.state_path = state_path

    @property
    def active_strategy(self) -> Optional[AccelerationStrategy]:
        try:
            return self.bandit.active_strategy
        except RuntimeError:
            return None

    def begin_trajectory(
            self, trajectory_id: int, sample_count: int) -> BackendSelection:
        assignment = self.bandit.begin_trajectory(
            trajectory_id, sample_count=sample_count)
        return BackendSelection(
            strategy=self.bandit.active_strategy,
            assignment=assignment,
        )

    def observe_one_step(
            self, trajectory_id: int, step_idx: int,
            numerators: Sequence[float], denominators: Sequence[float]) -> None:
        self.bandit.observe_one_step(
            trajectory_id, step_idx, numerators, denominators)

    def end_trajectory(
            self, trajectory_id: int,
            feedback: Optional[TemplateFeedback]) -> None:
        self.bandit.end_trajectory(trajectory_id, feedback)

    def persist(self) -> None:
        self.bandit.save_state(self.state_path)

    def summary(self) -> Dict[str, Any]:
        return self.bandit.summary()


@dataclass
class COVRTrajectoryFeedback:
    """Mutable denoise feedback collected for one trajectory."""

    pending_safety: List[Tuple[int, Any]] = field(default_factory=list)
    terminal_fidelity_loss_tensor: Any = None
    h_step_components_tensor: Any = None
    safety_full_steps: int = 0
    terminal_full_steps: int = 0

    def record_safety(self, step_idx: int, values: Any) -> None:
        self.pending_safety.append((int(step_idx), values))
        self.safety_full_steps += 1

    def record_terminal_loss(self, value: Any) -> None:
        self.terminal_fidelity_loss_tensor = value
        self.terminal_full_steps += 1

    def record_h_step(self, value: Any, full_steps: int) -> None:
        self.h_step_components_tensor = value
        self.terminal_full_steps += int(full_steps)


@dataclass
class COVRTrajectoryAssignment:
    """Assignment and live feedback for one trajectory."""

    trajectory_id: int
    sample_count: int
    strategy: Optional[AccelerationStrategy]
    bandit_assignment: Optional[TemplateAssignment]
    sentinel_selected: bool
    sentinel_start_idx: Optional[int]
    sample_ids: Tuple[str, ...]
    recorder: Optional[ActionAuditRecorder] = None
    session_id: str = ""
    safety_sample_rate: float = 0.0
    safety_chain_threshold: int = 0
    sentinel_horizon: int = 0
    terminal_reward_active: bool = False
    version: Optional[COVRVersion] = None
    template_id: Optional[str] = None
    feedback_sink: Dict[str, Any] = field(default_factory=dict)
    feedback: COVRTrajectoryFeedback = field(
        default_factory=COVRTrajectoryFeedback)
    accelerator_states: Dict[str, Any] = field(default_factory=dict)
    profiler: Any = None


@dataclass
class COVRRunState:
    """Mutable counters and timings owned by one COVR runtime."""

    trajectory_offset: int = 0
    safety_full_steps: int = 0
    sentinel_full_steps: int = 0
    sentinel_count: int = 0
    sentinel_skipped: int = 0
    safety_wall_s: float = 0.0
    terminal_wall_s: float = 0.0
    safety_wall_times: List[float] = field(default_factory=list)
    terminal_wall_times: List[float] = field(default_factory=list)
    control_wall_times: List[float] = field(default_factory=list)
    terminal_losses: List[float] = field(default_factory=list)
    h_step_numerators: List[float] = field(default_factory=list)
    h_step_denominators: List[float] = field(default_factory=list)
    profile_stage_totals: Dict[str, float] = field(default_factory=dict)
    profile_stage_counts: Dict[str, int] = field(default_factory=dict)


class COVRRuntime:
    """Thin lifecycle facade composed from focused COVR state objects.

    The runtime owns initialization (version/resume/backend/recorder),
    trajectory begin/end (sentinel selection, assignment, feedback
    materialization, bandit update/persist) and aggregate/config payload
    construction. The sampling loop keeps the tensor computation and the
    terminal/safety feedback sources; feedback arrives through
    ``COVRTrajectoryAssignment.feedback``.
    """

    def __init__(
            self, config: COVRRuntimeConfig, version: COVRVersion,
            backend: Optional[COVRPolicyBackend] = None,
            recorder: Optional[ActionAuditRecorder] = None,
            state: Optional[COVRRunState] = None,
            session_id: Optional[str] = None):
        self.config = config
        self.version = version
        self.backend = backend
        self.recorder = recorder
        self.state = state or COVRRunState()
        self.session_id = str(
            session_id or getattr(config, "session_id", None) or "")
        self._closed = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
            cls, config: COVRRuntimeConfig, *, scheduler=None,
            num_steps: Optional[int] = None, cfg_scale: float = 0.0,
            speca_init_kwargs: Optional[Dict[str, Any]] = None,
            resume: Optional[Mapping[str, Any]] = None,
            backend: Optional[COVRPolicyBackend] = None,
            state: Optional[COVRRunState] = None) -> "COVRRuntime":
        """Build a runtime from an immutable config.

        ``scheduler`` is only needed to construct ``COVRVersion`` (the
        scheduler class name + config JSON); ``cfg_scale`` is the guidance
        scale recorded in the version. ``speca_init_kwargs`` feeds the
        version's ``speca_config`` payload (may be None for non-SpecA runs).
        ``resume`` is the ``_covr_resume_metadata`` payload of the persisted
        bandit state (None when starting fresh).
        """
        if not isinstance(config, COVRRuntimeConfig):
            raise TypeError("COVRRuntime.create requires a COVRRuntimeConfig")

        session_id = (
            config.session_id
            or (str(resume["session_id"]) if resume else None)
            or f"{time.strftime('%Y%m%d-%H%M%S')}-seed{config.seed}"
        )

        if scheduler is not None:
            scheduler_config = _covr_scheduler_config_json(
                getattr(scheduler, "config", {}))
            scheduler_name = scheduler.__class__.__name__
            version_num_steps = int(
                num_steps if num_steps is not None
                else getattr(scheduler, "num_steps", 0))
        else:
            scheduler_config = ""
            scheduler_name = ""
            version_num_steps = int(num_steps or 0)

        version = COVRVersion(
            model=config.capabilities.model,
            base_model_version=config.base_model_version,
            scheduler=scheduler_name,
            scheduler_config=scheduler_config,
            num_steps=version_num_steps,
            cfg_scale=float(cfg_scale),
            speca_config=(
                _covr_canonical_json(speca_init_kwargs)
                if speca_init_kwargs is not None else ""),
        )

        runtime = cls(
            config=config, version=version, backend=backend,
            recorder=None, state=state, session_id=session_id)
        # Recorder ownership stays with the runtime.
        if config.shadow:
            runtime._ensure_recorder(session_id)
        return runtime

    def _ensure_recorder(self, session_id: str) -> ActionAuditRecorder:
        if self.recorder is None:
            self.recorder = ActionAuditRecorder(
                output_dir=self.config.output_dir,
                session_id=session_id,
                version=self.version,
                max_events=self.config.max_events,
            )
        return self.recorder

    # ------------------------------------------------------------------
    # Backend construction helpers (used by run_dit init)
    # ------------------------------------------------------------------

    def new_forced_backend(
            self, strategy: AccelerationStrategy,
            manifest: Optional[Any] = None) -> ForcedStrategyBackend:
        return ForcedStrategyBackend(strategy=strategy, manifest=manifest)

    def new_bandit_backend(
            self, manifest: Any, session_id: str, epsilon: float, seed: int,
            run_identity: Mapping[str, Any],
            state_path: str) -> ExperimentalBanditBackend:
        if isinstance(manifest, StrategyManifest):
            bandit = ConservativeTemplateBandit.from_strategies(
                manifest, session_id=session_id, epsilon=epsilon, seed=seed,
                run_identity=run_identity,
                alternative_prior_penalty=self.config.bandit_prior_penalty)
        else:
            bandit = ConservativeTemplateBandit(
                manifest, session_id=session_id, epsilon=epsilon, seed=seed,
                run_identity=run_identity,
                alternative_prior_penalty=self.config.bandit_prior_penalty)
        return ExperimentalBanditBackend(bandit=bandit, state_path=state_path)

    def configure_forced_backend(
            self, strategy: AccelerationStrategy,
            manifest: Optional[Any] = None) -> ForcedStrategyBackend:
        backend = self.new_forced_backend(strategy, manifest)
        self.backend = backend
        return backend

    def configure_experimental_bandit(
            self, manifest: Any, *, epsilon: float,
            run_identity: Mapping[str, Any], state_path: str,
            resume_expected_samples: Optional[int] = None
            ) -> ExperimentalBanditBackend:
        backend = self.new_bandit_backend(
            manifest, session_id=self.session_id, epsilon=epsilon,
            seed=self.config.seed, run_identity=run_identity,
            state_path=state_path)
        if resume_expected_samples is not None:
            backend.bandit.load_state(state_path)
            processed = int(backend.bandit.summary()["processed_samples"])
            if processed != int(resume_expected_samples):
                raise ValueError(
                    "COVR resume sample count changed while loading state")
            self.state.trajectory_offset = len(backend.bandit.assignments)
        self.backend = backend
        return backend

    # ------------------------------------------------------------------
    # Trajectory lifecycle
    # ------------------------------------------------------------------

    def begin_trajectory(
            self, trajectory_id: int, sample_count: int, *,
            sample_ids: Optional[Sequence[str]] = None,
            num_steps: Optional[int] = None,
            sentinel_rate: Optional[float] = None,
            sentinel_horizon: Optional[int] = None,
            mandatory_prefix: int = 0) -> COVRTrajectoryAssignment:
        """Select a strategy (if any) and open a trajectory context.

        Sentinel selection uses the same deterministic hash as the legacy
        ``_covr_sentinel_selection`` helpers, driven by the backend's session
        id (bandit) or the runtime session id (forced).
        """
        from run_dit_shared import _covr_sentinel_selection
        if self.backend is None:
            return COVRTrajectoryAssignment(
                trajectory_id=trajectory_id,
                sample_count=int(sample_count),
                strategy=None,
                bandit_assignment=None,
                sentinel_selected=False,
                sentinel_start_idx=None,
                sample_ids=tuple(sample_ids or ()),
                recorder=self.recorder,
                session_id=self.session_id,
                sentinel_horizon=int(sentinel_horizon or 0),
                version=self.version,
            )

        sentinel_rate = (
            self.config.sentinel_rate if sentinel_rate is None else sentinel_rate)
        sentinel_horizon = (
            self.config.sentinel_horizon
            if sentinel_horizon is None else sentinel_horizon)
        num_steps = num_steps or 0

        # Legacy-template forced mode carries a manifest whose
        # mandatory_prefix participates in sentinel selection.
        if isinstance(self.backend, ForcedStrategyBackend):
            session = self.session_id
        else:
            session = str(self.backend.bandit.session_id
                          if isinstance(self.backend, ExperimentalBanditBackend)
                          else self.config.session_id or "")
        sentinel_selected, sentinel_start_idx = _covr_sentinel_selection(
            session, trajectory_id, sentinel_rate, sentinel_horizon,
            num_steps, mandatory_prefix)

        selection = self.backend.begin_trajectory(
            trajectory_id, sample_count=int(sample_count))
        assignment = COVRTrajectoryAssignment(
            trajectory_id=trajectory_id,
            sample_count=int(sample_count),
            strategy=selection.strategy,
            bandit_assignment=selection.assignment,
            sentinel_selected=sentinel_selected,
            sentinel_start_idx=sentinel_start_idx,
            sample_ids=tuple(sample_ids or ()),
            recorder=self.recorder,
            session_id=session,
            safety_sample_rate=(
                float(self.config.safety_sample_rate)
                if isinstance(self.backend, ExperimentalBanditBackend)
                else 0.0),
            safety_chain_threshold=(
                int(self.config.safety_chain_threshold)
                if isinstance(self.backend, ExperimentalBanditBackend)
                else 0),
            sentinel_horizon=int(sentinel_horizon),
            version=self.version,
            template_id=(
                selection.assignment.template_id
                if selection.assignment is not None
                else selection.strategy.strategy_id
                if selection.strategy is not None else None),
        )
        return assignment

    def apply_acceleration_strategy(
            self, trajectory: COVRTrajectoryAssignment,
            *, speca_init_kwargs: Optional[Dict[str, Any]] = None,
            teacache_init_kwargs: Optional[Dict[str, Any]] = None,
            adapter_init_kwargs: Optional[Mapping[str, Dict[str, Any]]] = None,
            **legacy_init_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if trajectory.strategy is None:
            return {}
        states = apply_strategy(
            trajectory.strategy,
            speca_init_kwargs=speca_init_kwargs,
            teacache_init_kwargs=teacache_init_kwargs,
            adapter_init_kwargs=adapter_init_kwargs,
            **legacy_init_kwargs,
        )
        trajectory.accelerator_states = states
        adapter = get_adapter(trajectory.strategy.method)
        trajectory.terminal_reward_active = (
            isinstance(self.backend, ForcedStrategyBackend)
            or adapter.terminal_reward_active(states))
        return states

    def add_flops(
            self, metric: Any, trajectory: COVRTrajectoryAssignment,
            num_layers: int, *, method: Optional[str] = None,
            accelerator_states: Optional[Mapping[str, Any]] = None,
            num_steps: Optional[int] = None) -> None:
        """Fold one trajectory's accelerator decisions into ``flops_metric``.

        Selected strategies use their injected state. Observe-only runs use the
        runner's existing accelerator state, so shadow/profile modes preserve
        normal SpecA or TeaCache accounting. Runs without an active registered
        accelerator fall back to vanilla full steps.
        """
        effective_method = (
            trajectory.strategy.method
            if trajectory.strategy is not None else method)
        states = (
            trajectory.accelerator_states
            if trajectory.strategy is not None
            else dict(accelerator_states or {}))
        if effective_method and is_registered(effective_method):
            adapter = get_adapter(effective_method)
            if any(states.get(key) is not None for key in adapter.state_keys):
                adapter.add_flops(metric, states, num_layers=num_layers)
                return
        if num_steps is None:
            metric.add_vanilla_steps()
        else:
            metric.add_vanilla_steps(num_steps)

    # ------------------------------------------------------------------
    # Aggregate / config payloads (serializer slice)
    # ------------------------------------------------------------------

    def aggregate(self, *, wall_times: Optional[Sequence[float]] = None,
                  forced_template: Optional[Any] = None,
                  forced_strategy: Optional[AccelerationStrategy] = None,
                  forced_manifest: Optional[Any] = None,
                  speca_probe_full_blocks: Optional[int] = None,
                  state_path: Optional[str] = None,
                  dataset_start_index: int = 0,
                  resume_sample_offset: int = 0,
                  generation_start_index: int = 0,
                  target_samples: int = 0,
                  generated_samples_this_run: int = 0,
                  ) -> Dict[str, Any]:
        """Build the old COVR aggregate payloads from run state + config.

        Mirrors run_dit's final aggregate block exactly: ``covr_template_bandit``
        (summary + identity + accounting flags), ``covr_forced_template``
        (forced single-arm template info), and ``covr_reward_telemetry``
        (sentinel reward stats shared by bandit and forced modes). Keys that
        need run-dit-only values (forced template/manifest objects, speca
        probe counters) are passed in; everything else comes from the config
        and ``COVRRunState``.
        """
        state = self.state
        payload: Dict[str, Any] = {}
        if isinstance(self.backend, ExperimentalBanditBackend):
            summary = self.backend.summary()
            summary.update({
                "state_path": state_path or self.config.bandit_state_path,
                "dataset_start_index": dataset_start_index,
                "resume_sample_offset": resume_sample_offset,
                "generation_start_index": generation_start_index,
                "target_samples": int(target_samples),
                "generated_samples_this_run": int(generated_samples_this_run),
                "safety_full_steps": state.safety_full_steps,
                "safety_wall_s": state.safety_wall_s,
                "sentinel_count": state.sentinel_count,
                "sentinel_horizon": self.config.sentinel_horizon,
                "sentinel_full_steps": state.sentinel_full_steps,
                "sentinel_wall_s": state.terminal_wall_s,
                "terminal_feedback_wall_s": state.terminal_wall_s,
                "candidate_flops_exclude_safety": True,
                "candidate_flops_exclude_sentinel": True,
                "online_flops_include_safety": True,
                "online_flops_include_terminal": True,
                "online_flops_exclude_sentinel": True,
                "online_wall_exclude_sentinel": True,
            })
            payload["covr_template_bandit"] = summary
        if forced_template is not None:
            payload["covr_forced_template"] = {
                "template_id": forced_template.template_id,
                "manifest_hash": (
                    forced_manifest.manifest_hash
                    if forced_manifest is not None else ""),
                "refresh_count": forced_template.refresh_count,
                "modeled_full_block_equivalents": (
                    forced_template.modeled_full_block_equivalents),
                "probe_full_blocks": int(speca_probe_full_blocks or 0),
            }
        elif forced_strategy is not None:
            payload["covr_forced_template"] = {
                "template_id": forced_strategy.strategy_id,
                "manifest_hash": (
                    forced_manifest.manifest_hash
                    if forced_manifest is not None else ""),
                "refresh_count": forced_strategy.refresh_count,
                "modeled_full_block_equivalents": (
                    forced_strategy.modeled_flops),
                "probe_full_blocks": int(speca_probe_full_blocks or 0),
            }
        # Sentinel reward telemetry (bandit and forced modes).
        reward = {
            "sentinel_count": state.sentinel_count,
            "sentinel_full_steps": state.sentinel_full_steps,
            "sentinel_wall_s": state.terminal_wall_s,
            "sentinel_horizon": self.config.sentinel_horizon,
            "sentinel_rate": self.config.sentinel_rate,
            "sentinel_skipped": state.sentinel_skipped,
        }
        if state.terminal_losses:
            reward["terminal_fidelity_loss_mean"] = float(
                np.mean(state.terminal_losses))
            reward["terminal_fidelity_loss_std"] = float(
                np.std(state.terminal_losses))
            reward["terminal_fidelity_loss_n"] = len(state.terminal_losses)
        if state.h_step_numerators:
            reward["h_step_numerator_mean"] = float(
                np.mean(state.h_step_numerators))
            reward["h_step_numerator_n"] = len(state.h_step_numerators)
        if state.h_step_denominators:
            reward["h_step_denominator_mean"] = float(
                np.mean(state.h_step_denominators))
            reward["h_step_denominator_n"] = len(state.h_step_denominators)
        if isinstance(self.backend, ForcedStrategyBackend):
            strategy = self.backend.active_strategy
            if strategy is not None:
                reward["forced_strategy_id"] = strategy.strategy_id
        payload["covr_reward_telemetry"] = reward
        return payload

    def generation_profile(self, batches: int) -> Dict[str, Any]:
        """The ``generation_profile`` payload from the run state buckets."""
        totals = {
            stage: float(seconds)
            for stage, seconds in self.state.profile_stage_totals.items()
            if stage != "cuda_sync_calls"
        }
        return {
            "stage_total_s": totals,
            "stage_mean_per_batch_s": {
                stage: float(seconds / max(1, batches))
                for stage, seconds in totals.items()
            },
            "stage_observed_batches": {
                stage: int(self.state.profile_stage_counts.get(stage, 0))
                for stage in totals
            },
            "cuda_sync_calls": int(self.state.profile_stage_totals.get(
                "cuda_sync_calls", 0.0)),
            "batches": batches,
        }

    def online_accounting(
            self, *, wall_times: Optional[Sequence[float]] = None,
            n_images: int,
            candidate_flops_T: Optional[float] = None,
            vanilla_flops_T: Optional[float] = None,
            full_step_flops: Optional[float] = None) -> Dict[str, float]:
        """Exact ``_covr_online_accounting`` from run state (bandit mode).

        ``wall_times`` is the online wall list the loop appends (fallback
        profile stages are folded into the candidate bucket the same way the
        old inline call did). Returns {} for an empty run, matching the old
        helper. ``flops_online_T`` / ``speed_online_img_per_s`` fall back to
        the candidate-side values when the inputs are unavailable, which is
        the post-slice-2 accounting contract.
        """
        state = self.state
        if not wall_times:
            return {}
        online = [float(value) for value in wall_times]
        safety = list(state.safety_wall_times)
        terminal = list(state.terminal_wall_times)
        control = list(state.control_wall_times)
        if len(online) != len(safety) or len(online) != len(terminal) or (
                len(online) != len(control)):
            raise ValueError(
                "online and feedback wall-time samples must align")
        candidate_wall_times = [
            max(0.0, o - s - t - c)
            for o, s, t, c in zip(online, safety, terminal, control)]
        online_total = float(sum(online))
        safety_total = float(sum(safety))
        terminal_total = float(sum(terminal))
        control_total = float(sum(control))
        candidate_total = float(sum(candidate_wall_times))
        result = {
            "wall_s_candidate_mean": float(np.mean(candidate_wall_times)),
            "wall_s_candidate_std": float(np.std(candidate_wall_times)),
            "wall_s_candidate_total": candidate_total,
            "wall_s_safety_mean": float(np.mean(safety)),
            "wall_s_safety_total": safety_total,
            "wall_s_terminal_mean": float(np.mean(terminal)),
            "wall_s_terminal_total": terminal_total,
            "wall_s_control_mean": float(np.mean(control)),
            "wall_s_control_total": control_total,
            "wall_s_online_mean": float(np.mean(online)),
            "wall_s_online_std": float(np.std(online)),
            "wall_s_online_total": online_total,
            "speed_candidate_img_per_s": (
                float(n_images / candidate_total)
                if candidate_total > 0 else 0.0),
            "speed_online_img_per_s": (
                float(n_images / online_total)
                if online_total > 0 else 0.0),
            "safety_full_steps_mean_per_trajectory": (
                float(state.safety_full_steps / len(online))),
            "terminal_full_steps_mean_per_trajectory": (
                float(state.sentinel_full_steps / len(online))),
        }
        if (candidate_flops_T is not None and vanilla_flops_T is not None
                and full_step_flops is not None):
            safety_flops_T = (
                state.safety_full_steps / len(online) * full_step_flops / 1e12)
            terminal_flops_T = (
                state.sentinel_full_steps / len(online)
                * full_step_flops / 1e12)
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
        else:
            result["flops_online_T"] = (
                float(candidate_flops_T)
                if candidate_flops_T is not None else 0.0)
            result["speed_online_img_per_s"] = result[
                "speed_candidate_img_per_s"]
        return result

    def config_payload(
            self, *, covr_forced_template_id: Optional[str] = None,
            covr_session_id: Optional[str] = None,
            covr_version_key: Optional[str] = None) -> Dict[str, Any]:
        """The COVR slice of ``results["config"]`` (exact old keys)."""
        return {
            "covr_shadow": bool(self.recorder is not None),
            "covr_template_bandit": bool(
                isinstance(self.backend, ExperimentalBanditBackend)),
            "covr_force_template_id": covr_forced_template_id,
            "covr_session_id": covr_session_id,
            "covr_version_key": covr_version_key,
            "covr_profile_stages": bool(self.config.profile_stages),
            "covr_bandit_prior_penalty": self.config.bandit_prior_penalty,
        }

    # ------------------------------------------------------------------
    # End-of-trajectory lifecycle
    # ------------------------------------------------------------------

    def _materialize_feedback(
            self, trajectory: COVRTrajectoryAssignment,
            sink: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert loop-produced feedback into the trajectory feedback object.

        The denoise loop keeps tensor compute in run_dit and writes raw tensors
        into ``sink``; this method pops them, moves them to host scalars and
        materializes the per-trajectory ``COVRTrajectoryFeedback`` that the
        backends consume. Returns the materialized scalars as a plain dict so
        the loop can keep printing/telemetry in its own namespace.
        """
        feedback = trajectory.feedback
        # Pending safety: one-step labels per sampled Taylor step. The loop
        # already converted them to CPU tensors at the batch boundary; the
        # bandit consumes them through observe_one_step at end_trajectory.
        pending_values = sink.pop("pending_safety_values", None)
        pending_steps = sink.pop("pending_safety_steps", ())
        if pending_values is not None:
            safety_values = pending_values.detach().to("cpu").tolist()
            for step_idx, (numerators, denominators) in zip(
                    pending_steps, safety_values):
                feedback.record_safety(
                    step_idx, ([numerators], [denominators]))
        # H-step: mean components over the unguided half of the batch.
        h_step_components = sink.pop("h_step_components_tensor", None)
        h_step_numerator = h_step_denominator = None
        if h_step_components is not None:
            h_step_numerator, h_step_denominator = (
                h_step_components.detach().to("cpu").tolist())
        # Terminal fidelity: one extra forward pass at the final step. The
        # loop may substitute a full-baseline fallback scalar (bandit
        # terminal sentinels without a cheap reward); both sources resolve
        # to the same float here.
        terminal_loss = sink.pop("terminal_fidelity_loss_tensor", None)
        terminal_fidelity_loss = None
        if terminal_loss is not None:
            terminal_fidelity_loss = float(terminal_loss.detach().to("cpu"))
        if terminal_fidelity_loss is None:
            scalar_loss = sink.pop("terminal_fidelity_loss", None)
            if scalar_loss is not None:
                terminal_fidelity_loss = float(scalar_loss)
        return {
            "terminal_fidelity_loss": terminal_fidelity_loss,
            "h_step_numerator": h_step_numerator,
            "h_step_denominator": h_step_denominator,
            "safety_full_steps": int(sink.get("safety_full_steps", 0)),
            "terminal_full_steps": int(sink.get("terminal_full_steps", 0)),
        }

    def _profile_bucket_summary(
            self, profile: Mapping[str, float]) -> Dict[str, float]:
        """Control timing buckets summed from a generation profile."""
        return {
            stage: float(profile.get(stage, 0.0))
            for stage in _COVR_CONTROL_STAGES
        }

    def _accumulate_profile(self, profile: Mapping[str, float]) -> None:
        """Fold one trajectory's profile into the run state."""
        totals = self.state.profile_stage_totals
        counts = self.state.profile_stage_counts
        for stage, seconds in profile.items():
            totals[stage] = totals.get(stage, 0.0) + float(seconds)
            counts[stage] = counts.get(stage, 0) + 1

    def _bandit_feedback(
            self, trajectory: COVRTrajectoryAssignment,
            materialized: Mapping[str, Any],
            sentinel_propensity: float, horizon: int) -> Optional[TemplateFeedback]:
        """Build the TemplateFeedback schema for the current sentinel."""
        if self.backend is None or not isinstance(
                self.backend, ExperimentalBanditBackend):
            return None
        if trajectory.bandit_assignment is None:
            raise ValueError("bandit trajectory has no assignment")
        if not trajectory.sentinel_selected:
            # Preserve the historical bandit semantics: close the trajectory
            # without adding an unlabeled sample to delayed-loss statistics.
            return None
        if trajectory.sentinel_start_idx is None:
            # Terminal fidelity: prefer the cheap one-step reward; the caller
            # falls back to a full baseline only when absent.
            if materialized["terminal_fidelity_loss"] is None:
                raise RuntimeError(
                    "terminal sentinel did not produce feedback")
            return TemplateFeedback(
                trajectory_id=trajectory.trajectory_id,
                template_id=trajectory.bandit_assignment.template_id,
                sentinel_propensity=float(sentinel_propensity),
                horizon=int(horizon),
                terminal_fidelity_loss=float(
                    materialized["terminal_fidelity_loss"]),
            )
        if (materialized["h_step_numerator"] is None
                or materialized["h_step_denominator"] is None):
            raise RuntimeError(
                "H-step sentinel did not produce feedback")
        return TemplateFeedback(
            trajectory_id=trajectory.trajectory_id,
            template_id=trajectory.bandit_assignment.template_id,
            sentinel_propensity=float(sentinel_propensity),
            horizon=int(horizon),
            h_step_numerator=float(materialized["h_step_numerator"]),
            h_step_denominator=float(materialized["h_step_denominator"]),
        )

    def end_trajectory(
            self, trajectory: COVRTrajectoryAssignment,
            sink: Optional[Mapping[str, Any]] = None,
            *,
            generation_profile: Mapping[str, float],
            num_steps: int, sentinel_rate: float, sentinel_horizon: int,
            fallback_wall_s: float = 0.0, fallback_full_steps: int = 0,
            fallback_profile: Optional[Mapping[str, float]] = None
            ) -> Dict[str, Any]:
        """Materialize feedback, close the trajectory and persist run state.

        Owns the per-trajectory end lifecycle that the sampling loop used to
        inline: feedback materialization, control timing buckets, sentinel /
        full-step counters, bandit ``end_trajectory`` + per-trajectory state
        persist, and the forced-mode reward telemetry. Tensor compute stays in
        the loop; only host scalars cross this boundary.

        ``generation_profile`` is the batch profile (with the
        ``feedback_materialization`` stage already patched in by the caller).
        ``fallback_wall_s`` / ``fallback_full_steps`` / ``fallback_profile``
        describe the full-baseline fallback generation that the loop ran
        (bandit terminal sentinels without a cheap reward).

        Returns the per-trajectory outcome dict consumed by the loop:
        ``control_wall_s`` (excluding the fallback), ``wall_s`` (total
        trajectory wall time), ``feedback`` (materialized scalars) and the
        bandit/forced telemetry slices.
        """
        state = self.state
        sink = trajectory.feedback_sink if sink is None else sink
        materialized = self._materialize_feedback(trajectory, sink)
        sentinel_wall_s = (
            generation_profile.get("delayed_sentinel_shadow_full", 0.0)
            + generation_profile.get("delayed_sentinel_feedback", 0.0))
        safety_wall_s = generation_profile.get("safety_shadow_full", 0.0)
        terminal_wall_s = (
            sentinel_wall_s
            + generation_profile.get("terminal_fidelity_shadow_full", 0.0)
            + float(fallback_wall_s))
        safety_full_steps = int(materialized["safety_full_steps"])
        terminal_full_steps = (
            int(materialized["terminal_full_steps"]) + int(fallback_full_steps))

        state.safety_wall_times.append(safety_wall_s)
        state.terminal_wall_times.append(terminal_wall_s)
        state.safety_wall_s += safety_wall_s
        state.safety_full_steps += safety_full_steps
        state.terminal_wall_s += terminal_wall_s
        state.sentinel_full_steps += terminal_full_steps
        if terminal_full_steps:
            state.sentinel_count += 1

        self._accumulate_profile(generation_profile)
        if fallback_profile is not None:
            self._accumulate_profile(fallback_profile)
        # Control stages before the bandit section (the loop's profiler owns
        # strategy_selection / accelerator_state_reset / speca_state_reset /
        # strategy_initialization; the materialization stage was patched in).
        control_wall_s = sum(
            generation_profile.get(stage, 0.0)
            for stage in (
                "strategy_selection", "accelerator_state_reset",
                "speca_state_reset", "strategy_initialization",
                "feedback_materialization"))
        wall_s = (
            float(generation_profile.get("generation_online", 0.0))
            + float(fallback_wall_s) + control_wall_s)

        # ---- Bandit: close trajectory + per-trajectory state persist ----
        bandit_telemetry: Dict[str, Any] = {}
        if self.backend is not None and isinstance(
                self.backend, ExperimentalBanditBackend):
            feedback = self._bandit_feedback(
                trajectory, materialized,
                sentinel_propensity=(
                    sentinel_rate if trajectory.sentinel_selected
                    else 1.0),
                horizon=(
                    sentinel_horizon if trajectory.sentinel_start_idx is not None
                    else num_steps),
            )
            update_start = time.perf_counter()
            # One-step safety observations flow through the bandit close
            # (they flush into the safety table inside end_trajectory).
            for step_idx, (numerators, denominators) in trajectory.feedback.pending_safety:
                self.backend.observe_one_step(
                    trajectory.trajectory_id, step_idx,
                    numerators, denominators)
            self.backend.end_trajectory(trajectory.trajectory_id, feedback)
            update_s = time.perf_counter() - update_start
            state.profile_stage_totals["bandit_update"] = (
                state.profile_stage_totals.get("bandit_update", 0.0) + update_s)
            state.profile_stage_counts["bandit_update"] = (
                state.profile_stage_counts.get("bandit_update", 0) + 1)
            persist_start = time.perf_counter()
            self.backend.persist()
            persist_s = time.perf_counter() - persist_start
            state.profile_stage_totals["bandit_state_persist"] = (
                state.profile_stage_totals.get("bandit_state_persist", 0.0)
                + persist_s)
            state.profile_stage_counts["bandit_state_persist"] = (
                state.profile_stage_counts.get("bandit_state_persist", 0) + 1)
            control_s = update_s + persist_s
            wall_s += control_s
            state.control_wall_times.append(control_wall_s + control_s)
            bandit_telemetry = {
                "template_id": trajectory.bandit_assignment.template_id,
                "feedback": feedback,
                "update_s": update_s,
                "persist_s": persist_s,
            }
        elif trajectory.sentinel_selected:
            # ---- Forced single-arm sweep: same reward telemetry as the
            # bandit path, aggregated into results.json. Never fall back to a
            # full-baseline comparison (that is the bandit-mode fallback and
            # would cost ~50x per missing reward) — a sentinel without a
            # cheap reward is skipped and counted instead.
            if trajectory.sentinel_start_idx is None:
                if materialized["terminal_fidelity_loss"] is not None:
                    state.terminal_losses.append(
                        float(materialized["terminal_fidelity_loss"]))
                else:
                    state.sentinel_skipped += 1
            else:
                if (materialized["h_step_numerator"] is not None
                        and materialized["h_step_denominator"] is not None):
                    state.h_step_numerators.append(
                        float(materialized["h_step_numerator"]))
                    state.h_step_denominators.append(
                        float(materialized["h_step_denominator"]))
                else:
                    state.sentinel_skipped += 1
            state.control_wall_times.append(control_wall_s)

        return {
            "control_wall_s": control_wall_s,
            "wall_s": wall_s,
            "feedback": materialized,
            "bandit_telemetry": bandit_telemetry,
        }

    def close(self) -> Optional[Dict[str, Any]]:
        """Close the recorder and persist the backend state.

        Returns the shadow recorder summary (``{events, samples, ...}``) or
        None when no recorder exists — the old ``covr_summary =
        covr_recorder.close()`` contract. Per-trajectory persistence is owned
        by ``end_trajectory``; this is the run-level close the aggregate
        block relies on.
        """
        if self._closed:
            return None
        summary = None
        if self.recorder is not None:
            summary = self.recorder.close()
        if self.backend is not None:
            self.backend.persist()
        self._closed = True
        return summary


def covr_requested(args: Any) -> bool:
    """Return whether any existing CLI flag enables the COVR runtime."""
    return bool(
        getattr(args, "covr_shadow", False)
        or getattr(args, "covr_profile_stages", False)
        or getattr(args, "covr_template_bandit", False)
        or getattr(args, "covr_force_template_id", None)
        or getattr(args, "covr_strategy_bandit", False)
        or getattr(args, "covr_force_strategy_id", None)
        or getattr(args, "covr_strategy_manifest", None)
    )


def validate_covr_capabilities(args: Any) -> None:
    """Reject phase-one combinations before runner dispatch."""
    if not covr_requested(args):
        return
    if getattr(args, "model", None) != "dit":
        raise ValueError(
            "COVR runtime phase 1 supports only --model dit")
    if getattr(args, "ttt", False):
        raise ValueError(
            "COVR runtime phase 1 does not support --ttt")
    method = getattr(args, "method", None)
    if getattr(args, "covr_shadow", False) and method != "speca":
        raise ValueError("COVR shadow audit requires --method speca")
    if (getattr(args, "covr_template_bandit", False)
            or getattr(args, "covr_force_template_id", None)) and method != "speca":
        raise ValueError(
            "legacy COVR template modes require --method speca")
    if (getattr(args, "covr_strategy_bandit", False)
            or getattr(args, "covr_template_bandit", False)):
        if int(getattr(args, "batch_size", 1)) != 1:
            raise ValueError(
                "COVR strategy bandit requires --batch_size 1 so each "
                "trajectory is one image")
        prior_penalty = float(
            getattr(args, "covr_bandit_prior_penalty", 0.0))
        if not math.isfinite(prior_penalty) or prior_penalty < 0.0:
            raise ValueError(
                "--covr-bandit-prior-penalty must be finite and non-negative")


def _resolve_mode(args: Any) -> COVRMode:
    if (getattr(args, "covr_strategy_bandit", False)
            or getattr(args, "covr_template_bandit", False)):
        return COVRMode.EXPERIMENTAL_BANDIT
    if (getattr(args, "covr_force_strategy_id", None) is not None
            or getattr(args, "covr_force_template_id", None) is not None):
        return COVRMode.FORCED
    return COVRMode.OBSERVE


def build_covr_runtime_config(
        args: Any, output_dir: str) -> Optional[COVRRuntimeConfig]:
    """Resolve immutable runtime configuration without filesystem writes."""
    if not covr_requested(args):
        return None
    validate_covr_capabilities(args)

    mode = _resolve_mode(args)
    strategy_manifest = getattr(args, "covr_strategy_manifest", None)
    template_manifest = getattr(args, "covr_template_manifest", None)
    manifest_path = strategy_manifest or template_manifest
    force_strategy = getattr(args, "covr_force_strategy_id", None)
    force_template = getattr(args, "covr_force_template_id", None)
    uses_legacy = strategy_manifest is None and template_manifest is not None
    covr_output_dir = (
        getattr(args, "covr_output_dir", None)
        or os.path.join(output_dir, "covr")
    )
    state_path = (
        getattr(args, "covr_bandit_state", None)
        or os.path.join(covr_output_dir, "template_bandit_state.json")
    )
    method = str(getattr(args, "method", ""))
    adapter = get_adapter(method) if is_registered(method) else None
    capabilities = COVRCapabilities(
        model="dit",
        method=method,
        main_denoise_loop=True,
        shadow_audit=(method == "speca"),
        speca_safety=(method == "speca"),
        terminal_reward=bool(adapter),
    )
    return COVRRuntimeConfig(
        mode=mode,
        capabilities=capabilities,
        output_dir=covr_output_dir,
        bandit_state_path=state_path,
        resume_state_path=getattr(args, "covr_bandit_state", None),
        manifest_path=manifest_path,
        forced_strategy_id=force_strategy or force_template,
        uses_legacy_template=uses_legacy,
        shadow=bool(getattr(args, "covr_shadow", False)),
        profile_stages=bool(getattr(args, "covr_profile_stages", False)),
        session_id=getattr(args, "covr_session_id", None),
        base_model_version=str(
            getattr(args, "covr_base_model_version", None)
            or os.path.expanduser(DIT_REPO)),
        max_events=getattr(args, "covr_max_events", None),
        epsilon=float(getattr(args, "covr_bandit_epsilon", 0.1)),
        safety_sample_rate=float(
            getattr(args, "covr_safety_sample_rate", 0.0)),
        safety_chain_threshold=int(
            getattr(args, "covr_safety_chain_threshold", 0)),
        sentinel_rate=float(getattr(args, "covr_sentinel_rate", 0.0)),
        sentinel_horizon=int(getattr(args, "covr_sentinel_horizon", 0)),
        seed=int(getattr(args, "seed", 0)),
        bandit_prior_penalty=float(
            getattr(args, "covr_bandit_prior_penalty", 0.0)),
    )


def load_strategy_manifest(
        path: str, expected_version_key: str,
        mode: COVRMode) -> StrategyManifest:
    """Load a generic manifest with the same identity gate in both modes."""
    if mode not in (COVRMode.FORCED, COVRMode.EXPERIMENTAL_BANDIT):
        raise ValueError("strategy manifests require forced or bandit mode")
    manifest = StrategyManifest.load(path)
    if manifest.version_key != expected_version_key:
        raise ValueError(
            "strategy manifest version does not match runtime: "
            f"manifest={manifest.version_key}, runtime={expected_version_key}")
    return manifest
