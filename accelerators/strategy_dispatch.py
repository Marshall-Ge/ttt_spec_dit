# -*- coding: utf-8 -*-
"""Method-agnostic accelerator strategy dispatcher.

Given an ``AccelerationStrategy`` (e.g. "speca" with a refresh_mask, or
"teacache" with a rel_l1_thresh), ``apply_strategy`` calls the correct
accelerator's init function with the strategy's parameters and returns
the resulting state objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from .covr_bandit import AccelerationStrategy, RefreshTemplate
from .speca import SpecACache, SpecAState, speca_init
from .teacache import teacache_init


def apply_strategy(
    strategy: AccelerationStrategy,
    speca_init_kwargs: Optional[Dict[str, Any]] = None,
    teacache_init_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Configure an accelerator from an AccelerationStrategy.

    Parameters
    ----------
    strategy : AccelerationStrategy
        The selected strategy (from a bandit or forced).
    speca_init_kwargs : dict, optional
        Base keyword arguments for ``speca_init`` (used when
        ``strategy.method == "speca"``).  The ``refresh_mask`` and
        ``controller`` kwargs are injected from the strategy.
    teacache_init_kwargs : dict, optional
        Base keyword arguments for ``teacache_init`` (used when
        ``strategy.method == "teacache"``).  The ``rel_l1_thresh``
        kwarg is injected from the strategy.

    Returns
    -------
    dict
        Method-specific state dictionary.  Guaranteed keys:

        * For ``"speca"``:  ``{"cache_dic": SpecACache,
          "current": SpecAState}``
        * For ``"teacache"``: ``{"teacache_state": dict}``
    """
    method = strategy.method

    if method == "speca":
        kwargs = dict(speca_init_kwargs or {})
        refresh_mask = strategy.refresh_mask
        if refresh_mask is not None:
            kwargs["refresh_mask"] = refresh_mask
        cache_dic, current = speca_init(**kwargs)
        return {"cache_dic": cache_dic, "current": current}

    elif method == "teacache":
        kwargs = dict(teacache_init_kwargs or {})
        rel_l1_thresh = strategy.params.get("rel_l1_thresh")
        if rel_l1_thresh is not None:
            kwargs["rel_l1_thresh"] = rel_l1_thresh
        teacache_state = teacache_init(**kwargs)
        return {"teacache_state": teacache_state}

    else:
        raise ValueError(f"unsupported acceleration method: {method}")


def strategy_from_refresh_template(
    template: RefreshTemplate,
    num_steps: int,
) -> AccelerationStrategy:
    """Convert a SpecA RefreshTemplate to a method-agnostic strategy.

    Convenience function; equivalent to ``template.to_strategy(num_steps)``.
    """
    return template.to_strategy(num_steps)
