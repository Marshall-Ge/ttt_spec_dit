# -*- coding: utf-8 -*-
"""Method-agnostic accelerator strategy dispatcher.

Given an ``AccelerationStrategy`` (e.g. "speca" with a refresh_mask, or
"teacache" with a rel_l1_thresh), ``apply_strategy`` looks up the method's
``AcceleratorAdapter`` in the registry and calls its ``init_state`` with the
strategy's parameters, returning the resulting state objects.

The method-specific logic lives in the adapters (``accelerators/registry.py``),
so a new accelerator becomes dispatchable by registering an adapter — no edit
here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .covr_bandit import AccelerationStrategy, RefreshTemplate
from .registry import get_adapter


def apply_strategy(
    strategy: AccelerationStrategy,
    speca_init_kwargs: Optional[Dict[str, Any]] = None,
    teacache_init_kwargs: Optional[Dict[str, Any]] = None,
    **method_init_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Configure an accelerator from an AccelerationStrategy.

    Parameters
    ----------
    strategy : AccelerationStrategy
        The selected strategy (from a bandit or forced).
    speca_init_kwargs : dict, optional
        Base keyword arguments for SpecA init (used when
        ``strategy.method == "speca"``). The ``refresh_mask`` kwarg is
        injected by the adapter from the strategy.
    teacache_init_kwargs : dict, optional
        Base keyword arguments for TeaCache init (used when
        ``strategy.method == "teacache"``). When the strategy carries a
        ``refresh_mask`` (equal-FLOPs COVR arm) it is injected as
        ``refresh_mask``; otherwise ``rel_l1_thresh`` is injected from the
        strategy params.
    **method_init_kwargs
        Base init kwargs for any other registered method, keyed by that
        adapter's ``init_kwargs_key`` (e.g. ``foo_init_kwargs={...}``).

    Returns
    -------
    dict
        Method-specific state dictionary. Guaranteed keys are the selected
        adapter's ``state_keys``:

        * For ``"speca"``:   ``{"cache_dic": SpecACache, "current": SpecAState}``
        * For ``"teacache"``: ``{"teacache_state": dict}``
    """
    adapter = get_adapter(strategy.method)

    # Gather every method's base-kwargs bag by its adapter key, so this
    # function forwards the right one without branching on the method.
    kwargs_bags: Dict[str, Any] = dict(method_init_kwargs)
    kwargs_bags["speca_init_kwargs"] = speca_init_kwargs
    kwargs_bags["teacache_init_kwargs"] = teacache_init_kwargs

    base_kwargs = kwargs_bags.get(adapter.init_kwargs_key)
    return adapter.init_state(strategy, base_kwargs)


def strategy_from_refresh_template(
    template: RefreshTemplate,
    num_steps: int,
) -> AccelerationStrategy:
    """Convert a SpecA RefreshTemplate to a method-agnostic strategy.

    Convenience function; equivalent to ``template.to_strategy(num_steps)``.
    """
    return template.to_strategy(num_steps)
