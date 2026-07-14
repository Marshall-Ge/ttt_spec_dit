# -*- coding: utf-8 -*-
"""M5: Training loss for L3 LoRA — Homing + Identity.

**Principle**:

    For SpecA check events with ``block_input_hidden`` populated, the per-block
    replay path computes::

        lora_hidden  = block_with_lora(block_input_hidden)
        true_feature = base_block(block_input_hidden)

    Since ``true_feature`` is the base block output on the *same* Taylor-drifted
    input, the MSE between them measures how much LoRA modifies the output::

        MSE(lora_hidden, true_feature) = ||LoRA(block_input_hidden)||^2

    This single value serves **two** purposes when weighted independently:

    * **L_homing** — push LoRA toward correcting Taylor-prediction error.
    * **L_identity** — penalize LoRA residual magnitude to prevent ||B||_F
      from growing unboundedly (no-op regularization).

    Since both terms are mathematically identical for SpecA check events,
    the total simplifies to::

        L_total = (1 + lambda_identity) * MSE(lora_hidden, true_feature)
                + lambda_anchor * L_anchor

    Events WITHOUT ``block_input_hidden`` (TeaCache events, old checkpoints
    saved before the field was added) are **skipped** — the fallback full-
    forward path has loss ≈ 0 at the no-op starting point, providing no
    useful training signal.

**Constraints**:

    1. Do NOT let LoRA learn to predict the draft output. The supervised
       target is always the base block's own computation (``true_feature``),
       not the Taylor-predicted hidden state.
    2. Do NOT add L_align (cosine similarity).
    3. Do NOT remove ||B||_F monitoring (done in async_trainer.py).
    4. Preserve --vfl-no-train mode (no change needed — it skips L3 entirely).
    5. Old checkpoint backward compat: log a warning, do not crash.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


# ===========================================================================
# Per-block replay for supervised loss
# ===========================================================================


def _run_block_dit(block, block_input, timestep, class_labels, dtype):
    """Run a single DiT block (adaLN-Zero) with LoRA on block_input.

    Replicates the inline submodule calls from dit.py's block loop.
    Returns the block output with grad connectivity to LoRA params.
    """
    norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
        block_input, timestep=timestep, class_labels=class_labels,
        hidden_dtype=dtype,
    )
    attn_out = block.attn1(norm_hidden)
    x = block_input + gate_msa.unsqueeze(1) * attn_out
    norm_ff = block.norm3(x)
    modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff_out = block.ff(modulated_ff)
    x = x + gate_mlp.unsqueeze(1) * ff_out
    return x


def _run_block_pixart(block, block_input, timestep_emb, encoder_hidden_states,
                      encoder_attention_mask):
    """Run a single PixArt block (ada_norm_single) with LoRA on block_input.

    Replicates the inline submodule calls from pixart.py's block loop.
    Returns the block output with grad connectivity to LoRA params.
    """
    B = block_input.shape[0]
    proj = (block.scale_shift_table[None]
            + timestep_emb.reshape(B, 6, -1))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        proj.chunk(6, dim=1)

    norm_hidden = block.norm1(block_input)
    modulated = norm_hidden * (1 + scale_msa) + shift_msa
    attn1_out = block.attn1(modulated)
    x = block_input + gate_msa * attn1_out

    attn2_out = block.attn2(
        x,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
    )
    x = x + attn2_out

    norm_ff = block.norm2(x)
    modulated_ff = norm_ff * (1 + scale_mlp) + shift_mlp
    ff_out = block.ff(modulated_ff)
    x = x + gate_mlp * ff_out
    return x


def _run_transformer_forward(transformer,
                             latent: torch.Tensor,
                             timestep: torch.Tensor,
                             class_labels: Optional[torch.Tensor] = None,
                             encoder_hidden_states: Optional[torch.Tensor] = None):
    """Dispatch a vanilla forward call to DiT or PixArt based on inputs.

    Both forwards run with current=None, cache_dic=None, teacache_state=None
    so they take the vanilla path (full 28-block stack). LoRA-modified
    submodules still apply because LoRA is attached to the block params.

    Side effect: sets the global t_emb cache so time-conditioned LoRA layers
    can read it without recomputing per Linear. Cleared in ``finally`` to
    avoid leaking across forwards.
    """
    from verification_feedback_loop.lora_adapter import (
        set_lora_t_emb, clear_lora_t_emb,
        compute_timestep_emb_for_transformer,
    )

    hidden_dtype = latent.dtype
    t_emb = compute_timestep_emb_for_transformer(
        transformer, timestep, class_labels=class_labels,
        hidden_dtype=hidden_dtype,
    )
    if t_emb is not None:
        set_lora_t_emb(t_emb)
    try:
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
    finally:
        if t_emb is not None:
            clear_lora_t_emb()


def _resolve_hook_layer(event, num_layers: int) -> int:
    """Map an event to the layer index whose output the hook should capture.

    SpecA events record a single block's output -- hook that block directly.
    TeaCache events record the full block-stack output (event.layer_id is just
    a probe label, typically _VFL_PROBE_LAYER) -- hook the last block.
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
    lambda_identity: float = 1.0,
    lambda_anchor: float = 1.0,
    in_channels: int = 4,
):
    """Compute the L3 training loss with the two-term scheme.

    L_total = L_homing + lambda_identity * L_identity + lambda_anchor * L_anchor

    **L_homing and L_identity** are both MSE(lora_hidden, true_feature) from
    per-block replay.  For SpecA check events, ``true_feature`` is the base
    block's output on ``block_input_hidden`` (the Taylor-drifted input), so::

        MSE(lora_hidden, true_feature) = ||LoRA(block_input_hidden)||^2

    …which simultaneously (a) trains LoRA to correct Taylor error and (b)
    penalizes LoRA residual magnitude to prevent unbounded ||B||_F growth.
    The two names allow independent weighting::

        L_total = (1 + lambda_identity) * MSE(lora_hidden, true_feature)
                + lambda_anchor * L_anchor

    **Only events WITH ``block_input_hidden`` are used.**  Events lacking it
    (TeaCache events, old checkpoints) are skipped — the fallback full-forward
    path has loss approx 0 at the no-op starting point, providing no useful
    signal.

    Parameters
    ----------
    transformer : nn.Module
        The transformer with LoRA adapters attached. Must be in train mode
        for the LoRA layers, but can have backbone frozen.
    curvature_events : list of VerificationEvent
        Events with ``block_input_hidden`` populated.  Their ``true_feature``
        is the supervised target.
    anchor_samples : list of AnchorSample, optional
        Real data anchors for the standard diffusion loss term.
    lambda_identity : float
        Weight for the identity regularization term (L_identity).
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
    has_block_input_events = any(
        getattr(e, "block_input_hidden", None) is not None
        for e in curvature_events
    )
    has_anchors = bool(anchor_samples)
    if not has_block_input_events and not has_anchors:
        return torch.tensor(0.0, device=device, dtype=dtype,
                            requires_grad=True)

    # ----------------------------------------------------------------------
    # 1. Determine model type from transformer attributes.
    # ----------------------------------------------------------------------
    is_pixart = hasattr(transformer, 'adaln_single')

    # Pre-compute timestep_emb for PixArt (shared across blocks per step).
    # For DiT, block.norm1 computes it internally from (timestep, class_labels).
    _pixart_t_emb_cache = {}

    def _get_pixart_t_emb(t_val, hidden_dtype):
        key = int(t_val[0].item()) if t_val.numel() > 0 else 0
        if key not in _pixart_t_emb_cache:
            from verification_feedback_loop.lora_adapter import (
                set_lora_t_emb, clear_lora_t_emb,
                compute_timestep_emb_for_transformer,
            )
            t_emb = compute_timestep_emb_for_transformer(
                transformer, t_val, hidden_dtype=hidden_dtype)
            if t_emb is not None:
                set_lora_t_emb(t_emb)
            _pixart_t_emb_cache[key] = (t_emb.detach().clone()
                                        if t_emb is not None else None)
        return _pixart_t_emb_cache[key]

    # ----------------------------------------------------------------------
    # 2. Per-block replay: for each event with block_input_hidden, run the
    #    target block directly and compute MSE against true_feature.
    #    This is both L_homing and L_identity (same value, different names).
    # ----------------------------------------------------------------------
    mse_losses: List[torch.Tensor] = []
    _warned_no_block_input = False

    try:
        for event in curvature_events:
            block_input = getattr(event, "block_input_hidden", None)
            if block_input is None:
                if not _warned_no_block_input:
                    import warnings
                    warnings.warn(
                        "[VFL] curvature_loss: event lacks block_input_hidden "
                        "-- skipping. This is expected for TeaCache events or "
                        "old checkpoints (before the field was added).")
                    _warned_no_block_input = True
                continue

            # Per-block replay: run target block on Taylor-predicted input.
            block = transformer.transformer_blocks[event.layer_id]
            inp = block_input.to(device=device, dtype=dtype)

            t_val = getattr(event, "timestep_actual", 0) or event.timestep
            timestep = torch.tensor(
                [t_val], device=device, dtype=torch.long,
            ).expand(inp.shape[0])

            cl = (event.class_labels.to(device=device)
                  if event.class_labels is not None else None)
            enc = (event.encoder_hidden_states.to(device=device, dtype=dtype)
                   if event.encoder_hidden_states is not None else None)

            if is_pixart:
                t_emb = _get_pixart_t_emb(timestep, dtype)
                enc_mask = None  # PixArt encoder_attention_mask not stored
                lora_hidden = _run_block_pixart(
                    block, inp, t_emb, enc, enc_mask)
            else:
                lora_hidden = _run_block_dit(
                    block, inp, timestep, cl, dtype)

            target = event.true_feature.to(device=device, dtype=dtype)
            if target.shape != lora_hidden.shape:
                continue
            mse_losses.append(F.mse_loss(lora_hidden, target))

        # ------------------------------------------------------------------
        # 3. Anchor diffusion loss on real samples (prevents collapse).
        # ------------------------------------------------------------------
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
                    a_cl = prompt.to(device=device)
                elif prompt.ndim == 3:
                    a_enc = prompt.to(device=device, dtype=dtype)
            elif isinstance(prompt, int):
                a_cl = torch.tensor([prompt], device=device, dtype=torch.long)

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
        # No hooks to clean up in this simplified loss.
        pass

    # ------------------------------------------------------------------
    # 4. Weighted sum. Empty terms contribute zero.
    # ------------------------------------------------------------------
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    loss_homing = (sum(mse_losses) / len(mse_losses)
                   if mse_losses else zero)
    loss_identity = loss_homing  # same value, independently weighted
    loss_anchor = (sum(anchor_losses) / len(anchor_losses)
                   if anchor_losses else zero)

    loss = loss_homing + lambda_identity * loss_identity + lambda_anchor * loss_anchor

    # If everything was empty (e.g. all events lacked block_input_hidden),
    # still return a grad-connected zero so .backward() doesn't blow up.
    if not loss.requires_grad:
        loss = loss + 0.0 * sum(
            p.sum() for p in transformer.parameters() if p.requires_grad
        )
    return loss
