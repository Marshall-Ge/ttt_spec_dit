# -*- coding: utf-8 -*-
"""VFL 专用配置 — 所有阈值、比例、调度参数。"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class VFLConfig:
    """VFL 系统全局配置单例。

    Attributes
    ----------
    enabled : bool
        全局开关。False 时所有 VFL 操作为 no-op (保持原有行为不变)。
    accept_sample_rate : float
        Accept 事件的随机记录比例 (仅用于 L1 校准和 L2 normal 配额)。
    buffer_capacity_per_stratum : int
        每个 (layer_id, timestep_bucket) stratum 的 ring buffer 最大容量。
    batch_ratio : dict
        训练时三类样本的采样比例。
    trigger_min_samples : int
        Buffer 新增样本数 >= N 触发训练。
    trigger_min_interval_s : float
        距上次训练 >= T 秒触发训练。
    loRA_rank : int
        LoRA adapter 的秩。
    loRA_alpha : int
        LoRA scaling factor。
    top_k_layers : int
        挂载 LoRA 的高拒绝率 layer 数量。
    lambda_curvature : float
        Curvature loss 权重初始值。
    lambda_curvature_min : float
        Curvature loss 权重下限。
    lambda_curvature_max : float
        Curvature loss 权重上限。
    ema_window_short : int
        版本切换后的短 EMA 窗口步数。
    ema_window_long : int
        正常运行时的长 EMA 窗口步数。
    quality_epsilon : float
        FID 退化容忍上限。
    reject_delta : float
        Reject 率下降最低要求。
    trainer_steps_per_trigger : int
        每次触发训练的梯度步数。
    """

    # ---- 全局开关 ----
    enabled: bool = True

    # ---- M1: 采集 ----
    accept_sample_rate: float = 0.02  # 2% accept 采样率

    # ---- M2: 在线校准 ----
    ema_window_long: int = 500     # 正常运行 EMA 窗口
    ema_window_short: int = 50     # 版本切换后短窗口
    # ---- M3: 回放缓冲区 ----
    buffer_capacity_per_stratum: int = 1000
    batch_ratio: Dict[str, float] = field(default_factory=lambda: {
        "hard_negative": 0.5,
        "normal": 0.3,
        "anchor": 0.2,
    })

    # ---- M4: LoRA ----
    loRA_rank: int = 8
    loRA_alpha: int = 16
    top_k_layers: int = 3

    # ---- M5: Curvature Loss ----
    lambda_curvature: float = 1e-4
    lambda_curvature_min: float = 1e-6
    lambda_curvature_max: float = 1e-2
    curvature_order: int = 2

    # ---- M6: 异步训练 ----
    trigger_min_samples: int = 200
    trigger_min_interval_s: float = 300.0
    trainer_steps_per_trigger: int = 50

    # ---- M6 (Phase 2): real async worker ----
    # Polling interval between buffer-readiness checks in the background
    # training thread. Small enough to feel responsive, large enough to avoid
    # busy-spinning while the inference thread fills the buffer.
    poll_interval_s: float = 5.0
    # Buffer-readiness thresholds for ``AsyncTrainingWorker._buffer_ready``:
    #   * ``buffer_ready_min_strata`` — how many (layer, bucket) strata must
    #     each hold at least ``buffer_ready_min_per_stratum`` events before
    #     we attempt a training cycle. Replaces the old raw-count trigger
    #     (``trigger_min_samples``) so that sparse first-batch data doesn't
    #     drive a bad gradient update.
    #   * ``buffer_ready_min_anchors`` — minimum anchor samples for the
    #     diffusion anchor loss to have signal (otherwise L_anchor is 0
    #     and we waste a cycle).
    buffer_ready_min_strata: int = 2
    buffer_ready_min_per_stratum: int = 5
    buffer_ready_min_anchors: int = 10

    # ---- M7: Eval Gate ----
    quality_epsilon: float = 5.0   # FID 退化容忍上限
    reject_delta: float = 0.05     # Reject 率下降最低要求 (5pp)


# 全局默认配置实例
DEFAULT_VFL_CONFIG = VFLConfig()
