# -*- coding: utf-8 -*-
"""Compatibility shim — lifecycle helpers moved to ``utils``.

Historical import sites (run_dit, scripts, tests) keep importing the
``_covr_*`` / ``_GenerationProfiler`` names from here. New code should import
from ``utils.serialization`` / ``utils.timing`` directly (see
``.claude/project-structure.md``).
"""

from utils.serialization import (
    _covr_bandit_sentinel_selection,
    _covr_canonical_json,
    _covr_forced_sentinel_selection,
    _covr_hash_index,
    _covr_hash_sample,
    _covr_resume_metadata,
    _covr_scheduler_config_json,
    _covr_sentinel_selection,
    _covr_version_key,
)
from utils.timing import _GenerationProfiler


def _load_covr_version_cls():
    """Import COVRVersion lazily to break the package import cycle."""
    from accelerators.covr import COVRVersion
    return COVRVersion


__all__ = [
    "_GenerationProfiler",
    "_covr_bandit_sentinel_selection",
    "_covr_canonical_json",
    "_covr_forced_sentinel_selection",
    "_covr_hash_index",
    "_covr_hash_sample",
    "_covr_resume_metadata",
    "_covr_scheduler_config_json",
    "_covr_sentinel_selection",
    "_covr_version_key",
    "_load_covr_version_cls",
]
