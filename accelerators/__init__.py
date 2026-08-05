# -*- coding: utf-8 -*-
"""Accelerators for diffusion model inference — pure functions, no classes.

  - speca:   per-block per-submodule Taylor cache; ``SpecACache``/``SpecAState``
             objects are owned by the caller and threaded through the model.
  - teacache: whole-step residual cache; integrated at the top-level sampling
             loop (the model itself is agnostic to it).
"""

from .compute_controller import (
    ComputeAction,
    ComputeController,
    ComputeOpportunity,
    ProbeCorrectController,
    VerificationResult,
)
from .covr import (
    AUDIT_SCHEMA_VERSION,
    FEATURE_NAMES,
    ActionAuditContext,
    ActionAuditEvent,
    ActionAuditRecorder,
    BudgetLedger,
    COVRAction,
    COVRContext,
    COVRDecision,
    COVRPolicy,
    COVRVersion,
    CounterfactualEvent,
    OnlineRidgeUCB,
    PrimalDualBudget,
    ShadowAuditRecorder,
    TransitionDefectBatch,
    ddim_epsilon_transition_coefficients,
    load_policy_state,
    normalized_transition_defect,
    read_action_audits,
    read_events,
    save_policy_state,
    summarize_taylor_cache,
    transition_defect_batch,
    transition_defects,
)
from .covr_bandit import (
    AccelerationStrategy,
    BANDIT_SCHEMA_VERSION,
    ConservativeTemplateBandit,
    RefreshTemplate,
    StrategyManifest,
    TemplateAssignment,
    TemplateFeedback,
    TemplateManifest,
    TimestepSafetyPrior,
    TimestepSafetyTable,
)
from .covr_runtime import (
    COVRCapabilities,
    COVRMode,
    COVRPolicyBackend,
    COVRRunState,
    COVRRuntime,
    COVRRuntimeConfig,
    COVRTrajectoryAssignment,
    COVRTrajectoryFeedback,
    ExperimentalBanditBackend,
    ForcedStrategyBackend,
    build_covr_runtime_config,
    covr_requested,
    load_strategy_manifest,
    validate_covr_capabilities,
)
from .registry import (
    AcceleratorAdapter,
    SpecAAdapter,
    TeaCacheAdapter,
    get_adapter,
    is_registered,
    register_adapter,
    registered_methods,
)
from .speca import (
    SpecACache,
    SpecAState,
    speca_init,
    speca_cal_type,
    speca_controller_action,
    speca_controller_observe,
    derivative_approximation,
    taylor_formula,
    taylor_cache_init,
    cache_step_dit,
    cache_step_pixart,
    compute_error_gate,
)
from .strategy_dispatch import (
    apply_strategy,
    strategy_from_refresh_template,
)
from .teacache import (
    teacache_init,
    teacache_decide,
    teacache_cache_residual,
    teacache_apply_residual,
    teacache_step,
    teacache_reset,
)
