# -*- coding: utf-8 -*-
"""Accelerator adapter registry — the pluggable seam for COVR.

COVR (the per-trajectory strategy bandit) is *method-agnostic*: it selects
among ``AccelerationStrategy`` arms and never needs to know whether an arm
is SpecA, TeaCache, or something added later. Everything method-specific
that COVR touches is exactly three concerns — the **init/reward/flops
trio**:

  1. **init**  — configure the concrete accelerator state from a strategy
                 (``AccelerationStrategy`` -> ``teacache_state`` / SpecA
                 ``cache_dic``+``current``).
  2. **reward**— given the live state objects, say whether the cheap
                 terminal-fidelity reward the bandit learns from is
                 available for this method.
  3. **flops** — fold this generation's decisions into ``FLOPsMetric``
                 with the method's own accounting.

An ``AcceleratorAdapter`` bundles those three. Adapters register into a
process-global table; call sites look one up by ``method`` instead of
branching on ``if method == "speca" / elif "teacache"``. Adding a new
accelerator to COVR is then "write an adapter + ``register_adapter(...)``"
— no edits to ``covr_bandit`` / ``strategy_dispatch`` / ``run_dit``.

The *forward pass* stays method-branched inside the model
(``models/dit.py`` threads ``teacache_state`` / ``cache_dic`` / ``current``
through ``forward``) — that is deliberately out of scope here: the registry
governs the strategy layer, not the compute graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

# ``AccelerationStrategy`` is only needed for type hints; import lazily inside
# methods to keep this module importable without a covr_bandit dependency
# cycle (covr_bandit imports *us* for its method whitelist).


# ===========================================================================
# Adapter interface — the init / reward / flops trio
# ===========================================================================


class AcceleratorAdapter(ABC):
    """Method-specific glue for one accelerator, seen by COVR.

    Subclasses set the class attributes below and implement the trio.

    Attributes
    ----------
    method : str
        The ``AccelerationStrategy.method`` string this adapter handles
        (e.g. ``"speca"``, ``"teacache"``). Must be unique in the registry.
    init_kwargs_key : str
        Keyword name under which ``apply_strategy`` receives this method's
        base init kwargs (e.g. ``"teacache_init_kwargs"``). Keeps
        ``apply_strategy`` generic — it forwards ``**kwargs`` and each
        adapter picks out its own bag.
    state_keys : tuple of str
        Keys the adapter writes into the state dict returned by
        :meth:`init_state` (e.g. ``("teacache_state",)`` or
        ``("cache_dic", "current")``). Lets callers unpack results without
        knowing the method.
    """

    method: str = ""
    init_kwargs_key: str = ""
    state_keys: tuple = ()

    # -- 1. init -----------------------------------------------------------
    @abstractmethod
    def init_state(
        self,
        strategy: "Any",
        base_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Configure accelerator state from ``strategy``.

        Returns a dict whose keys are :attr:`state_keys`. The dict is the
        method-agnostic hand-off to the sampling loop.
        """

    # -- 2. reward ---------------------------------------------------------
    @abstractmethod
    def terminal_reward_active(self, states: Mapping[str, Any]) -> bool:
        """Whether the bandit's terminal-fidelity reward applies here.

        ``states`` is the union of live state objects the loop holds
        (``{"cache_dic", "current", "teacache_state", ...}``). Return True
        only when this method's accelerator is actually engaged, so a
        sentinel trajectory computes the last-step full-vs-cache reward.
        """

    # -- 3. flops ----------------------------------------------------------
    @abstractmethod
    def add_flops(
        self,
        flops_metric: Any,
        states: Mapping[str, Any],
        *,
        num_layers: int,
    ) -> None:
        """Fold this generation's decisions into ``flops_metric``.

        Uses the method's own accounting (per-step decisions for TeaCache,
        full-block-equivalents for SpecA). ``num_layers`` is the transformer
        block count (needed by SpecA; TeaCache ignores it).
        """


# ===========================================================================
# Built-in adapters
# ===========================================================================


class SpecAAdapter(AcceleratorAdapter):
    """SpecA — per-block Taylor cache; state = ``cache_dic`` + ``current``."""

    method = "speca"
    init_kwargs_key = "speca_init_kwargs"
    state_keys = ("cache_dic", "current")

    def init_state(self, strategy, base_kwargs=None):
        from .speca import speca_init

        kwargs = dict(base_kwargs or {})
        refresh_mask = strategy.refresh_mask
        if refresh_mask is not None:
            # Forced-schedule COVR arm: the mask drives calc/skip verbatim.
            kwargs["refresh_mask"] = refresh_mask
        cache_dic, current = speca_init(**kwargs)
        return {"cache_dic": cache_dic, "current": current}

    def terminal_reward_active(self, states):
        return states.get("current") is not None

    def add_flops(self, flops_metric, states, *, num_layers):
        cache_dic = states.get("cache_dic")
        if cache_dic is None:
            return
        flops_metric.add_speca_generation(
            full_steps=cache_dic.full_count,
            taylor_steps=cache_dic.taylor_count,
            probe_full_blocks=(
                cache_dic.probe_full_blocks
                + cache_dic.recompute_full_blocks),
            num_layers=num_layers,
        )


class TeaCacheAdapter(AcceleratorAdapter):
    """TeaCache — whole-step residual cache; state = ``teacache_state``."""

    method = "teacache"
    init_kwargs_key = "teacache_init_kwargs"
    state_keys = ("teacache_state",)

    def init_state(self, strategy, base_kwargs=None):
        from .teacache import teacache_init

        kwargs = dict(base_kwargs or {})
        refresh_mask = strategy.refresh_mask
        if refresh_mask is not None:
            # Forced-schedule (equal-FLOPs COVR arm): the mask drives the
            # per-step calc/skip decision, bypassing the dynamic threshold.
            kwargs["refresh_mask"] = refresh_mask
        else:
            rel_l1_thresh = strategy.params.get("rel_l1_thresh")
            if rel_l1_thresh is not None:
                kwargs["rel_l1_thresh"] = rel_l1_thresh
        return {"teacache_state": teacache_init(**kwargs)}

    def terminal_reward_active(self, states):
        return states.get("teacache_state") is not None

    def add_flops(self, flops_metric, states, *, num_layers):
        teacache_state = states.get("teacache_state")
        if teacache_state is None:
            return
        flops_metric.add_generation(
            SimpleNamespace(decisions=teacache_state["decisions"]))


# ===========================================================================
# Registry
# ===========================================================================

_REGISTRY: Dict[str, AcceleratorAdapter] = {}


def register_adapter(adapter: AcceleratorAdapter) -> None:
    """Register ``adapter`` under its ``method`` (idempotent overwrite)."""
    if not getattr(adapter, "method", ""):
        raise ValueError("adapter.method must be a non-empty string")
    _REGISTRY[adapter.method] = adapter


def get_adapter(method: str) -> AcceleratorAdapter:
    """Return the adapter for ``method`` or raise ``KeyError``-style error."""
    try:
        return _REGISTRY[method]
    except KeyError:
        raise ValueError(
            f"unsupported acceleration method: {method} "
            f"(registered: {registered_methods()})")


def is_registered(method: str) -> bool:
    """Whether ``method`` has a registered adapter."""
    return method in _REGISTRY


def registered_methods() -> List[str]:
    """Sorted list of registered method names (for error messages/tests)."""
    return sorted(_REGISTRY)


# Register the built-in accelerators at import time.
register_adapter(SpecAAdapter())
register_adapter(TeaCacheAdapter())
