# -*- coding: utf-8 -*-
"""M1: Verification Hook — 验证拦截层.

在 TeaCache/SpecA 已有的 accept/reject 判断之外, 额外记录 verification 事件。
纯旁路操作, 不改变任何推理决策。

事件类型:
  - reject 事件: 全部记录 (高价值监督样本)
  - accept 事件: 低采样率随机记录 (用于 L1 校准 + L2 normal 配额)

bucket 策略:
  timestep_bucket ∈ {0, 1, 2} — early / mid / late 三段
  layer_id ∈ [0, 27]          — 28 transformer blocks
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import torch

# Bucket 划分
NUM_TIMESTEP_BUCKETS = 3


@dataclass
class VerificationEvent:
    """单次 verification 事件的完整快照。

    所有 tensor 字段都必须是 detached 的 CPU tensor, 确保:
      1. 不保留 autograd 图 (O(1) 内存)
      2. 不受推理线程的 GPU 内存管理影响
    """

    layer_id: int                      # 0-27
    timestep: int                      # 原始 timestep 值 (如 981, 947, ...)
    timestep_bucket: int               # 0=early, 1=mid, 2=late
    predicted_feature: torch.Tensor    # Taylor / draft / cached 输出, detached CPU
    true_feature: torch.Tensor         # 强制重算的真实输出, detached CPU
    error_value: float                 # error metric 值 (cosine similarity / rel L1 / ...)
    decision: str                      # "accept" | "reject"
    model: str                         # "dit" | "pixart"
    base_model_version: str            # 从 checkpoint 路径提取的版本标识
    step_idx: int                      # 去噪步序号 (0 = first, N-1 = last)
    module: str = ""                   # 子模块名 ("attn"/"mlp" for DiT, "attn1"/"attn2"/"ff" for PixArt)

    def to_dict(self) -> dict:
        """序列化为纯 Python 对象 (用于日志/存储)。不保留 tensor。"""
        return {
            "layer_id": self.layer_id,
            "timestep": self.timestep,
            "timestep_bucket": self.timestep_bucket,
            "error_value": self.error_value,
            "decision": self.decision,
            "model": self.model,
            "base_model_version": self.base_model_version,
            "step_idx": self.step_idx,
            "module": self.module,
            # tensor shapes for reference
            "predicted_shape": tuple(self.predicted_feature.shape),
            "true_shape": tuple(self.true_feature.shape),
        }


def make_timestep_bucket(timestep: int, num_steps: int) -> int:
    """将原始 timestep 值映射到 0/1/2 三桶。

    Bucket 0: early   — 前 1/3 去噪轨迹 (高噪声, 大结构)
    Bucket 1: mid     — 中 1/3 去噪轨迹
    Bucket 2: late    — 后 1/3 去噪轨迹 (低噪声, 细节)

    使用 timestep 的绝对值 (而非 step_idx) 因为不同 scheduler 的
    timestep 序列不同, 但它们的相对位置意义相近。

    简化实现: 直接用 bucket = step_idx * 3 / num_steps。
    调用方传入的是去噪步序号 (0..N-1)。
    """
    if num_steps <= 0:
        return 0
    bucket = int(timestep * NUM_TIMESTEP_BUCKETS / num_steps)
    return min(bucket, NUM_TIMESTEP_BUCKETS - 1)


def record_event(event: VerificationEvent,
                 buffer: Optional["StratifiedReplayBuffer"] = None,
                 accept_sample_rate: float = 0.02) -> bool:
    """记录 verification 事件到缓冲区。

    O(1) 写入 — 不做任何 GPU 同步, 不阻塞推理主线程。
    所有 tensor 已在调用前 detach + cpu。

    Parameters
    ----------
    event : VerificationEvent
        要记录的事件。
    buffer : StratifiedReplayBuffer or None
        目标缓冲区。None 时静默丢弃 (no-op, 用于 VFL 未启用场景)。
    accept_sample_rate : float
        Accept 事件的随机记录概率 (0.0 - 1.0)。

    Returns
    -------
    recorded : bool
        True 如果事件被实际写入 buffer。
    """
    if buffer is None:
        return False

    if event.decision == "reject":
        # 全部记录
        buffer.add(event, kind="hard_negative")
        return True
    else:
        # 低采样率随机记录
        if random.random() < accept_sample_rate:
            buffer.add(event, kind="normal")
            return True
    return False


# ===========================================================================
# Hook helpers — 从模型 forward 中提取 VerificationEvent
# ===========================================================================


def make_speca_event(
    layer_id: int,
    timestep_val: int,
    step_idx: int,
    num_steps: int,
    predicted_hidden: torch.Tensor,
    full_hidden: torch.Tensor,
    error_value: float,
    error_metric: str,
    model: str,
    base_model_version: str,
    module: str = "",
) -> VerificationEvent:
    """从 SpecA check_layer 的比较结果构造 VerificationEvent。

    decision: error_value > threshold → "reject", else "accept"
    但这里不判断 threshold — 由调用方传入 decision (已在 speca_cal_type 中决定)。
    简化: 只要触发了 do_check (即达到了 min_taylor_steps), 就是 "reject" 候选。
    实际 decision 由 record_event 调用方根据 cache_dic['check'] 和 step_type 判断。
    """
    bucket = make_timestep_bucket(step_idx, num_steps)
    return VerificationEvent(
        layer_id=layer_id,
        timestep=timestep_val,
        timestep_bucket=bucket,
        predicted_feature=predicted_hidden.detach().float().cpu(),
        true_feature=full_hidden.detach().float().cpu(),
        error_value=error_value,
        decision="reject",  # check_layer 触发 = 潜在 rejection
        model=model,
        base_model_version=base_model_version,
        step_idx=step_idx,
        module=module,
    )


def make_teacache_probe_event(
    layer_id: int,
    timestep_val: int,
    step_idx: int,
    num_steps: int,
    predicted_hidden: torch.Tensor,
    true_hidden: torch.Tensor,
    model: str,
    base_model_version: str,
) -> VerificationEvent:
    """从 TeaCache calc 步的 per-layer probe 构造 VerificationEvent。

    TeaCache 的 calc = 计算了完整 block stack, 同时我们也模拟了 skip 路径
    (用 cached residual) 得到 predicted。这等价于一个 reject 事件:
      预测 = 如果用缓存会得到什么
      真实 = 完整计算的结果
    """
    bucket = make_timestep_bucket(step_idx, num_steps)
    # 用 relative L1 作为 error metric (与 TeaCache 保持一致)
    with torch.no_grad():
        eps = 1e-10
        error = (predicted_hidden - true_hidden).abs() / (true_hidden.abs() + eps)
        error_val = float(error.mean().item())

    return VerificationEvent(
        layer_id=layer_id,
        timestep=timestep_val,
        timestep_bucket=bucket,
        predicted_feature=predicted_hidden.detach().float().cpu(),
        true_feature=true_hidden.detach().float().cpu(),
        error_value=error_val,
        decision="reject",  # calc 步骤本身说明 TeaCache 认为不能 skip
        model=model,
        base_model_version=base_model_version,
        step_idx=step_idx,
        module="residual",   # TeaCache 是全 block 级别的 residual
    )
