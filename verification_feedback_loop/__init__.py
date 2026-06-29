# -*- coding: utf-8 -*-
"""Verification Feedback Loop (VFL) — 验证负反馈闭环系统.

三层架构 (风险递增、收益递增):
  L1 — online_calibration  : 递归统计量在线重估计阈值/校准函数 (无梯度, 常驻)
  L2 — replay_buffer       : 分层回放缓冲区, 为 L3 提供 stratified 训练数据
  L3 — lora_adapter        : 低秩 adapter 微调 (有梯度, 异步, 需 canary 发布)

与 Phase 3 TTT 的区别:
  TTT 改进缓存**内容** (hidden state correction), VFL 改进缓存**决策**
  (threshold / rescale / backbone 轨迹平滑度)。两者互不依赖。
"""

from verification_feedback_loop.verification_hook import (
    VerificationEvent,
    record_event,
    make_timestep_bucket,
    NUM_TIMESTEP_BUCKETS,
)
from verification_feedback_loop.replay_buffer import (
    StratifiedReplayBuffer,
    AnchorSample,
)
from verification_feedback_loop.online_calibration import (
    OnlineCalibrator,
)
from verification_feedback_loop.lora_adapter import (
    LoRALinear,
    attach_lora,
    detach_lora,
    get_lora_params,
    freeze_backbone,
    select_top_k_layers,
    save_lora_checkpoint,
    load_lora_checkpoint,
)
from verification_feedback_loop.curvature_loss import (
    trajectory_curvature_loss,
    trajectory_curvature_loss_from_buffer,
    compute_training_loss,
)
from verification_feedback_loop.async_trainer import (
    AsyncTrainer,
)
from verification_feedback_loop.eval_gate import (
    EvalGate,
    GateStatus,
    GateResult,
)
from verification_feedback_loop.version_registry import (
    VersionRegistry,
    AdapterStatus,
    AdapterRecord,
)
from verification_feedback_loop.config import VFLConfig

__all__ = [
    # M1
    "VerificationEvent",
    "record_event",
    "make_timestep_bucket",
    "NUM_TIMESTEP_BUCKETS",
    # M2
    "OnlineCalibrator",
    # M3
    "StratifiedReplayBuffer",
    "AnchorSample",
    # M4
    "LoRALinear",
    "attach_lora",
    "detach_lora",
    "get_lora_params",
    "freeze_backbone",
    "select_top_k_layers",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    # M5
    "trajectory_curvature_loss",
    "trajectory_curvature_loss_from_buffer",
    "compute_training_loss",
    # M6
    "AsyncTrainer",
    # M7
    "EvalGate",
    "GateStatus",
    "GateResult",
    # M8
    "VersionRegistry",
    "AdapterStatus",
    "AdapterRecord",
    # Config
    "VFLConfig",
]
