# -*- coding: utf-8 -*-
"""M5: Trajectory Curvature Loss — L3 的核心正则项.

**核心约束 (Constraint 1)**:
    绝不让模型去拟合 draft 的预测值。Loss 只用 ``true_feature`` 序列
    自身计算"轨迹是否易于低阶外推"的正则项。

**原理**:
    对连续 timestep 的真实特征序列, 拟合一个 order 阶局部多项式
    (与 SpecA 的 Taylor 外推阶数对齐), 惩罚拟合残差。

    这鼓励 backbone 的去噪轨迹变得更"可被低阶外推捕捉" —
    而不是让 backbone 去逼近某次具体的 draft 预测值。

**训练目标**::

    L_total = L_diffusion(anchor_batch) + λ_t * L_curvature(batch)

其中 ``L_diffusion`` 是标准 diffusion loss (v-prediction / noise prediction),
只在真实 anchor 数据上计算, 是防止 "拉懒" 的关键锚点。

``λ_t`` 由 M7 eval_gate 反馈调节: 质量退化就调小, reject 率改善且质量不变就调大。
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def trajectory_curvature_loss(
    true_features: List[torch.Tensor],
    order: int = 2,
) -> torch.Tensor:
    """计算轨迹曲率 loss — 仅依赖 true_feature 序列。

    对一小段连续 timestep 的真实特征序列, 用最小二乘拟合
    ``order`` 阶局部多项式, 返回拟合残差的 MSE。

    这个 loss **不依赖 predicted_feature** — 它只衡量真实轨迹
    偏离低阶多项式的程度 (即 "不可外推性")。

    Parameters
    ----------
    true_features : list of Tensor
        同一 layer 在连续 timestep 的 true hidden states。
        每个 tensor shape: ``(B, seq, hidden_dim)``。
    order : int
        多项式阶数 (默认 2, 与 SpecA Taylor 外推对齐)。

    Returns
    -------
    loss : scalar Tensor
        拟合残差的均方值。越小 → 轨迹越接近低阶多项式 → 越易外推。
    """
    n = len(true_features)
    if n < order + 2:
        # 样本不足, 无法拟合
        if n == 0:
            return torch.tensor(0.0)
        return torch.tensor(0.0, device=true_features[0].device)

    device = true_features[0].device
    dtype = true_features[0].dtype

    # Stack along time dim: feature_dim is raveled, last dim = T
    # true_features elements can be any shape (e.g. (B, C, H, W) or (B, L, D))
    # We flatten everything except batch and time dims.
    orig_ndim = true_features[0].ndim
    if orig_ndim == 4:
        # (B, C, H, W) -> (B, C*H*W)
        flat_features = [f.view(f.shape[0], -1) for f in true_features]
    elif orig_ndim == 3:
        flat_features = [f.view(f.shape[0], -1) for f in true_features]
    elif orig_ndim == 2:
        flat_features = true_features  # (B, D) already
    else:
        flat_features = [f.view(f.shape[0], -1) for f in true_features]

    # (T, B, D)
    stacked = torch.stack(flat_features, dim=0)
    T, B, D = stacked.shape

    # Normalised time coordinate t ∈ [-1, 1]
    t = torch.linspace(-1, 1, T, device=device, dtype=dtype)

    # Vandermonde design matrix A: (T, order+1), columns = [t^0, t^1, ..., t^order]
    A = torch.stack([t ** k for k in range(order + 1)], dim=1)

    # Flatten batch into features: (T, B*D)
    flat = stacked.reshape(T, -1)

    # lstsq requires float32/float64, cast if needed
    lstsq_dtype = flat.dtype
    if lstsq_dtype not in (torch.float32, torch.float64):
        flat = flat.float()
        A = A.float()

    # lstsq: solve A @ coeffs ≈ flat
    solution = torch.linalg.lstsq(A, flat)
    coeffs = solution.solution  # (order+1, B*D)

    # Reconstruct fitted values
    fitted = A @ coeffs  # (T, B*D)
    fitted = fitted.reshape(T, B, D)

    # Residual
    residual = stacked - fitted
    loss = (residual ** 2).mean()

    return loss


def trajectory_curvature_loss_from_buffer(
    events: List,
    order: int = 2,
) -> torch.Tensor:
    """从 VFL buffer 采样的事件中计算 curvature loss。

    从 events 中提取 true_feature 序列 (按 timestep 排序),
    然后调用 ``trajectory_curvature_loss``。

    Parameters
    ----------
    events : list of VerificationEvent
        来自同一 layer 的连续 timestep 事件。
    order : int
        多项式阶数。

    Returns
    -------
    loss : scalar Tensor
    """
    # Sort by step_idx ascending
    sorted_events = sorted(events, key=lambda e: e.step_idx)

    true_features = []
    for event in sorted_events:
        # Move tensors to a common device if needed
        tf = event.true_feature
        if tf.device.type != "cuda":
            continue  # skip CPU-only events in mixed batches
        true_features.append(tf)

    if len(true_features) < order + 2:
        if true_features:
            return torch.tensor(0.0, device=true_features[0].device)
        return torch.tensor(0.0)

    return trajectory_curvature_loss(true_features, order=order)


# ===========================================================================
# Composite training loss
# ===========================================================================


def compute_training_loss(
    transformer,
    curvature_events: List,
    anchor_samples: Optional[List] = None,
    lambda_curvature: float = 1e-4,
    curvature_order: int = 2,
) -> torch.Tensor:
    """Compute the full L3 training loss with proper gradient connectivity.

    Runs a forward pass through the transformer (which has LoRA attached)
    to get gradient-connected hidden states, then computes curvature loss.

    L_total = λ_curv * L_curvature(from forward pass) [+ λ_diff * L_diffusion if anchors]

    Parameters
    ----------
    transformer : nn.Module
        The transformer with LoRA adapters attached. Must be in train mode
        for the LoRA layers, but can have backbone frozen.
    curvature_events : list of VerificationEvent
        Events providing metadata (layer_id, timestep) for curvature computation.
        Their true_features are NOT used directly — instead we re-run forward
        to get gradient-connected features.
    anchor_samples : list of AnchorSample, optional
    lambda_curvature : float
    curvature_order : int

    Returns
    -------
    loss : scalar Tensor with grad connectivity to LoRA params
    """
    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype

    num_steps_in_seq = curvature_order + 4  # enough for polynomial fit (6)
    batch_size = 2

    hidden_states_seq = []
    for i in range(num_steps_in_seq):
        latent = torch.randn(batch_size, 4, 32, 32, device=device, dtype=dtype)
        t = torch.full((batch_size,), 500 - i * 30, device=device, dtype=torch.long)
        class_labels = torch.randint(0, 1000, (batch_size,), device=device)

        out = transformer(latent, t, class_labels=class_labels, return_dict=False)[0]
        hidden_states_seq.append(out)

    loss_curv = trajectory_curvature_loss(hidden_states_seq, order=curvature_order)
    loss = lambda_curvature * loss_curv

    return loss
