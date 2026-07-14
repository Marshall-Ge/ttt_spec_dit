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
    # Time-conditioned LoRA (default True). When True, each LoRA adapter is
    # augmented with a small t_proj MLP that modulates ΔW by γ(t_emb), letting
    # the rank=4 bottleneck adapt across the early/mid/late denoising phases.
    # Set False via ``--vfl-no-time-lora`` to fall back to vanilla LoRA.
    time_conditioned_lora: bool = True

    # ---- M5: Curvature Loss ----
    lambda_curvature: float = 1e-2
    lambda_curvature_min: float = 1e-6
    lambda_curvature_max: float = 1e-1
    curvature_order: int = 2
    # Anchor loss weight — temporarily disabled (0.0). The self-distillation
    # target (noise_pred from base forward) yields loss=0 at LoRA no-op
    # starting point, providing no training signal. Re-enable after
    # redesigning anchor target (candidates: DDIM-inferred x̂_0 or
    # next-step noise_pred temporal consistency).
    lambda_anchor: float = 0.0

    # ---- M5 (v3): Homing + Identity loss ----
    # Weight for L_identity = MSE(lora_hidden, true_feature), same computation
    # as L_homing but independently weighted. L_total = (1 + lambda_identity)
    # * MSE(lora_hidden, true_feature) + lambda_anchor * L_anchor.
    # For SpecA check events, true_feature = base_block(block_input_hidden),
    # so MSE(lora_hidden, true_feature) = ||LoRA(block_input_hidden)||^2,
    # which is both L_homing (push LoRA toward true_feature) and L_identity
    # (penalize LoRA residual magnitude).
    lambda_identity: float = 1.0

    # ---- M6: 异步训练 ----
    trigger_min_samples: int = 200
    trigger_min_interval_s: float = 300.0
    trainer_steps_per_trigger: int = 50
    max_checkpoints: int = 5  # 只保留最近 N 个 checkpoint

    # ---- M6 (Phase 2): real async worker ----
    # Polling interval between buffer-readiness checks in the background
    # training thread. Small enough to feel responsive, large enough to avoid
    # busy-spinning while the inference thread fills the buffer.
    poll_interval_s: float = 5.0
    # Buffer-readiness thresholds for ``AsyncTrainingWorker._buffer_ready``:
    #   * ``buffer_ready_min_strata`` — how many (layer, bucket) strata must
    #     be non-empty (i.e. cover different denoising phases) before training.
    #   * ``buffer_ready_min_total_samples`` — minimum total events across ALL
    #     strata combined.  This is the primary volume gate — it prevents the
    #     small-buffer overfitting that the old per-stratum minimum (≥5 events
    #     per stratum) allowed when only 2 strata × 5 = 10 events were present.
    #   * ``buffer_ready_min_anchors`` — minimum anchor samples for the
    #     diffusion anchor loss to have signal (otherwise L_anchor is 0 and
    #     we waste a cycle).
    buffer_ready_min_strata: int = 2
    buffer_ready_min_anchors: int = 10
    buffer_ready_min_total_samples: int = 200

    # Maximum events carrying block_input_hidden in the replay buffer.
    # Each event stores one (B, L, D) tensor ~590KB for DiT, so 500
    # events ≈ 300MB — much more manageable than the ~99MB/event
    # SpecA cache snapshot that this replaces.
    max_block_input_events: int = 500

    # Maximum curvature events processed per gradient step (limits GPU memory).
    # Subsamples randomly each step — all events get used across the M steps.
    max_events_per_step: int = 2

    # ---- M7: Eval Gate ----
    quality_epsilon: float = 5.0   # FID 退化容忍上限
    reject_delta: float = 0.05     # Reject 率下降最低要求 (5pp)


# 全局默认配置实例
DEFAULT_VFL_CONFIG = VFLConfig()
