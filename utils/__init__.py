# -*- coding: utf-8 -*-
"""通用工具包 — 纯函数,不依赖项目内其他模块。

子模块:
  - io.py          图像 I/O、tensor↔PIL、VAE decode、FID 真实图预处理
  - timing.py      CudaTimer、_GenerationProfiler、_record_profile_stage
  - serialization.py JSON 安全化、canonical JSON、确定性哈希、COVR 记账纯函数

规范:新增纯计算逻辑时,优先放入本包对应子模块(见 .claude/project-structure.md §5)。
"""

from .common import (
    get_vfl_checkpoint_dir,
    latent_seed_for_index,
    prune_checkpoints,
)
from .io import (
    decode_latent,
    ensure_real_299,
    latent_to_pil,
    load_real_image,
    pil_to_tensor,
    save_image,
)
from .serialization import (
    _clean,
    _compute_generated_fid_is,
    _covr_bandit_sentinel_selection,
    _covr_canonical_json,
    _covr_forced_sentinel_selection,
    _covr_hash_index,
    _covr_hash_sample,
    _covr_log_snr,
    _covr_online_accounting,
    _covr_resume_metadata,
    _covr_scalar,
    _covr_scheduler_config_json,
    _covr_sentinel_selection,
    _covr_version_key,
    _dataset_generation_window,
    _load_forced_covr_template,
)
from .timing import (
    CudaTimer,
    _GenerationProfiler,
    _record_profile_stage,
)

__all__ = [
    # common
    "get_vfl_checkpoint_dir", "latent_seed_for_index", "prune_checkpoints",
    # io
    "decode_latent", "ensure_real_299", "latent_to_pil", "load_real_image",
    "pil_to_tensor", "save_image",
    # serialization
    "_clean", "_compute_generated_fid_is", "_covr_bandit_sentinel_selection",
    "_covr_canonical_json", "_covr_forced_sentinel_selection",
    "_covr_hash_index", "_covr_hash_sample", "_covr_log_snr",
    "_covr_online_accounting", "_covr_resume_metadata", "_covr_scalar",
    "_covr_scheduler_config_json", "_covr_sentinel_selection",
    "_covr_version_key", "_dataset_generation_window",
    "_load_forced_covr_template",
    # timing
    "CudaTimer", "_GenerationProfiler", "_record_profile_stage",
]
