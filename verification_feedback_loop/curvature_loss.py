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

**训练目标 (M5 v2 — buffer-driven closed loop)**::

    L_total = L_supervised + λ_curv * L_curvature + λ_anchor * L_anchor

其中:
  * ``L_supervised`` — 对每个 verification event, 重跑 forward (带 LoRA),
    用 hook 捕获目标层输出, MSE 逼近 ``event.true_feature`` (真实计算结果)。
  * ``L_curvature``  — 对同 (sample, layer) 的时序 hidden 序列拟合多项式,
    惩罚残差, 鼓励轨迹光滑可外推。
  * ``L_anchor``     — 标准 diffusion loss, 在真实 anchor 样本上计算,
    防止 LoRA 坍缩 (只监督 noise 通道, 与 DiT learned-sigma CFG 一致)。
"""

from collections import defaultdict
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
# Internal helpers for the buffer-driven training loss
# ===========================================================================


def _run_transformer_forward(transformer,
                             latent: torch.Tensor,
                             timestep: torch.Tensor,
                             class_labels: Optional[torch.Tensor] = None,
                             encoder_hidden_states: Optional[torch.Tensor] = None):
    """Dispatch a vanilla forward call to DiT or PixArt based on inputs.

    Both forwards run with current=None, cache_dic=None, teacache_state=None
    so they take the vanilla path (full 28-block stack). LoRA-modified
    submodules still apply because LoRA is attached to the block params.
    """
    if encoder_hidden_states is not None:
        # PixArt signature: forward(hidden_states, encoder_hidden_states, timestep, ...)
        return transformer(
            latent,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )
    # DiT signature: forward(hidden_states, timestep, class_labels=None, ...)
    return transformer(
        latent,
        timestep=timestep,
        class_labels=class_labels,
        return_dict=False,
    )


def _resolve_hook_layer(event, num_layers: int) -> int:
    """Map an event to the layer index whose output the hook should capture.

    SpecA events record a single block's output → hook that block directly.
    TeaCache events record the full block-stack output (event.layer_id is just
    a probe label, typically _VFL_PROBE_LAYER) → hook the last block.
    """
    if getattr(event, "module", "") == "residual":
        return num_layers - 1
    return event.layer_id


# ===========================================================================
# Composite training loss (buffer-driven, gradient-connected)
# ===========================================================================


def compute_training_loss(
    transformer,
    curvature_events: List,
    anchor_samples: Optional[List] = None,
    lambda_curvature: float = 1e-4,
    curvature_order: int = 2,
    lambda_anchor: float = 1.0,
    in_channels: int = 4,
):
    """Compute the full L3 training loss with proper gradient connectivity.

    Replaces the previous random-noise version: now drives LoRA training
    from real buffer events with three loss terms — supervised MSE on
    event.true_feature, curvature on per-(sample, layer) trajectories, and
    a standard diffusion anchor on real anchor samples.

    L_total = L_supervised
            + λ_curv * L_curvature
            + λ_anchor * L_anchor

    Parameters
    ----------
    transformer : nn.Module
        The transformer with LoRA adapters attached. Must be in train mode
        for the LoRA layers, but can have backbone frozen.
    curvature_events : list of VerificationEvent
        Events with ``latent_input`` / ``class_labels`` / ``encoder_hidden_states``
        populated. Their ``true_feature`` is the supervised target.
    anchor_samples : list of AnchorSample, optional
        Real data anchors for the standard diffusion loss term.
    lambda_curvature : float
        Weight for the curvature term.
    curvature_order : int
        Polynomial order for the curvature fit.
    lambda_anchor : float
        Weight for the anchor diffusion loss term.
    in_channels : int
        Number of noise channels (DiT learned-sigma: 4 noise + 4 variance).
        Anchor loss only supervises the first ``in_channels`` channels.

    Returns
    -------
    loss : scalar Tensor with grad connectivity to LoRA params
    """
    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype

    # ----------------------------------------------------------------------
    # 0. Short-circuit: nothing to learn from.
    # ----------------------------------------------------------------------
    has_events = bool(curvature_events) and any(
        getattr(e, "latent_input", None) is not None for e in curvature_events
    )
    has_anchors = bool(anchor_samples)
    if not has_events and not has_anchors:
        return torch.tensor(0.0, device=device, dtype=dtype,
                            requires_grad=True)

    # ----------------------------------------------------------------------
    # 1. Register forward hooks on the layers we need to capture.
    # ----------------------------------------------------------------------
    num_layers = len(transformer.transformer_blocks)
    target_layers = set()
    for e in curvature_events:
        if getattr(e, "latent_input", None) is None:
            continue
        target_layers.add(_resolve_hook_layer(e, num_layers))

    captured: dict = {}
    hooks: list = []

    def _make_hook(lid):
        def _hook(_module, _inp, out):
            captured[lid] = out
        return _hook

    for lid in target_layers:
        block = transformer.transformer_blocks[lid]
        hooks.append(block.register_forward_hook(_make_hook(lid)))

    # ----------------------------------------------------------------------
    # 2. For each event, re-run forward (with LoRA) and capture target output.
    # ----------------------------------------------------------------------
    supervised_losses: List[torch.Tensor] = []
    # (sample_id, hook_layer) → list of (step_idx, lora_hidden)
    curvature_by_layer: dict = defaultdict(list)

    try:
        for event in curvature_events:
            if getattr(event, "latent_input", None) is None:
                continue

            latent = event.latent_input.to(device=device, dtype=dtype)
            # Prefer the real diffusion timestep (e.g. 981) over the legacy
            # `timestep` field which historically held step_idx. adaLN
            # modulation depends on the actual t — using step_idx would
            # force LoRA to compensate for a wrong modulation signal.
            t_val = getattr(event, "timestep_actual", 0) or event.timestep
            timestep = torch.tensor(
                [t_val], device=device, dtype=torch.long,
            ).expand(latent.shape[0])

            cl = (event.class_labels.to(device=device, dtype=dtype)
                  if event.class_labels is not None else None)
            enc = (event.encoder_hidden_states.to(device=device, dtype=dtype)
                   if event.encoder_hidden_states is not None else None)

            captured.clear()
            _run_transformer_forward(
                transformer, latent, timestep,
                class_labels=cl, encoder_hidden_states=enc,
            )

            hook_layer = _resolve_hook_layer(event, num_layers)
            if hook_layer not in captured:
                continue  # hook didn't fire (shouldn't happen, but be safe)

            lora_hidden = captured[hook_layer]
            target = event.true_feature.to(device=device, dtype=dtype)
            if target.shape != lora_hidden.shape:
                # Shape mismatch (e.g. CFG-doubled vs single) — skip safely.
                continue

            supervised_losses.append(F.mse_loss(lora_hidden, target))

            curvature_by_layer[(event.sample_id, hook_layer)].append(
                (event.step_idx, lora_hidden))

        # ----------------------------------------------------------------------
        # 3. Curvature loss: fit polynomial per (sample, layer) trajectory.
        # ----------------------------------------------------------------------
        curv_losses: List[torch.Tensor] = []
        for seq in curvature_by_layer.values():
            if len(seq) < curvature_order + 2:
                continue
            seq.sort(key=lambda x: x[0])
            hiddens = [h for _, h in seq]
            curv_losses.append(trajectory_curvature_loss(
                hiddens, order=curvature_order))

        # ----------------------------------------------------------------------
        # 4. Anchor diffusion loss on real samples (prevents collapse).
        # ----------------------------------------------------------------------
        anchor_losses: List[torch.Tensor] = []
        for anchor in (anchor_samples or []):
            if getattr(anchor, "latent", None) is None:
                continue
            a_latent = anchor.latent.to(device=device, dtype=dtype)
            a_t = anchor.timestep.to(device=device, dtype=torch.long)
            if a_t.numel() == 1:
                a_t = a_t.expand(a_latent.shape[0])
            a_target = anchor.target.to(device=device, dtype=dtype)

            # AnchorSample.prompt is class_labels (DiT) or encoder_hidden_states (PixArt)
            prompt = anchor.prompt
            a_cl = None
            a_enc = None
            if isinstance(prompt, torch.Tensor):
                if prompt.ndim == 1:
                    a_cl = prompt.to(device=device, dtype=dtype)
                elif prompt.ndim == 3:
                    a_enc = prompt.to(device=device, dtype=dtype)

            out = _run_transformer_forward(
                transformer, a_latent, a_t,
                class_labels=a_cl, encoder_hidden_states=a_enc,
            )
            model_out = out[0] if isinstance(out, tuple) else out.sample
            # Only supervise the noise channels (learned-sigma safe).
            ch = min(in_channels, model_out.shape[1],
                     a_target.shape[1])
            anchor_losses.append(F.mse_loss(model_out[:, :ch], a_target[:, :ch]))
    finally:
        # ----------------------------------------------------------------------
        # 5. Always remove hooks, even on exception.
        # ----------------------------------------------------------------------
        for h in hooks:
            h.remove()

    # ----------------------------------------------------------------------
    # 6. Weighted sum. Empty terms contribute zero.
    # ----------------------------------------------------------------------
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    loss_sup = (sum(supervised_losses) / len(supervised_losses)
                if supervised_losses else zero)
    loss_curv = (sum(curv_losses) / len(curv_losses)
                 if curv_losses else zero)
    loss_anchor = (sum(anchor_losses) / len(anchor_losses)
                   if anchor_losses else zero)

    loss = loss_sup + lambda_curvature * loss_curv + lambda_anchor * loss_anchor

    # If everything was empty (e.g. all events lacked latent_input), still
    # return a grad-connected zero so .backward() doesn't blow up.
    if not loss.requires_grad:
        loss = loss + 0.0 * sum(
            p.sum() for p in transformer.parameters() if p.requires_grad
        )
    return loss
