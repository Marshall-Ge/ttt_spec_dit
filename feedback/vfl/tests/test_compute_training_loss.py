# -*- coding: utf-8 -*-
"""Unit test for the buffer-driven compute_training_loss (M5 v3: Homing + Identity).

Verifies:
  1. Loss is finite and non-zero when events carry block_input_hidden (per-block replay).
  2. Events WITHOUT block_input_hidden are skipped (no crash, no NaN).
  3. Loss handles empty events / empty anchors gracefully.
  4. Anchor-only path still produces a grad-connected loss.
  5. Old checkpoints / TeaCache events (no block_input_hidden) are skipped cleanly.
  6. Optimizer step changes parameters through the new loss.

Run:
    python feedback.vfl/tests/test_compute_training_loss.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from feedback.vfl.curvature_loss import compute_training_loss
from feedback.vfl.replay_buffer import AnchorSample
from feedback.vfl.verification_hook import (
    make_speca_event,
    make_teacache_probe_event,
)


# ===========================================================================
# Tiny stub transformer -- mimics the structure DiT/PixArt forward hooks need
# ===========================================================================


class _StubTimestepEmbedder(nn.Module):
    """Mimics the timestep embedding module (norm1.emb) used by DiT blocks.

    Accepts timestep and optional class_labels/hidden_dtype.
    Returns a dummy embedding tensor for the stub.
    """

    def forward(self, timestep, class_labels=None, hidden_dtype=None):
        # Return a dummy embedding matching the stub dim
        return torch.randn(timestep.shape[0], 16, device=timestep.device)


class _StubNorm1(nn.Module):
    """Minimal mock of AdaLayerNormZero: forward(x, timestep, class_labels, hidden_dtype).

    Returns norm_hidden ~ x, and scalar gates (~0.5 for gating, ~0.0 for shift/scale).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.emb = _StubTimestepEmbedder()

    def forward(self, x, timestep=None, class_labels=None, hidden_dtype=None):
        b, seq, d = x.shape
        # gate_msa, gate_mlp ~ 0.5; shift_mlp, scale_mlp ~ 0
        gate = torch.full((b, 1), 0.5, device=x.device, dtype=x.dtype)
        zero = torch.zeros((b, 1), device=x.device, dtype=x.dtype)
        return x, gate, zero, zero, gate


class _StubAttn1(nn.Module):
    """Minimal mock of attention: just passes through."""

    def forward(self, x):
        return x


class _StubFF(nn.Module):
    """Minimal mock of feed-forward block."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class _StubBlock(nn.Module):
    """A single transformer block replicating the submodule structure
    of a real DiT BasicTransformerBlock: norm1, attn1, norm3, ff.

    Output shape (B, seq, dim) so the curvature loss code path is exercised
    the same way as the real DiT block.
    """

    def __init__(self, dim: int = 16):
        super().__init__()
        self.norm1 = _StubNorm1(dim)
        self.attn1 = _StubAttn1()
        self.norm3 = nn.Identity()
        self.ff = _StubFF(dim)

    def forward(self, x):
        # Simplified whole-block forward (used by the stub transformer, not
        # by _run_block_dit which calls submodules directly).
        return x


class _StubTransformer(nn.Module):
    """Mimics DiTTransformer2D forward signature just enough:
    forward(latent, timestep, class_labels=None, return_dict=False)
    pos_embed: (B, C, H, W) -> (B, seq, dim)
    """

    def __init__(self, num_layers: int = 4, dim: int = 16, seq: int = 4,
                 in_channels: int = 4, out_channels: int = 8,
                 latent_size: int = 4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_StubBlock(dim) for _ in range(num_layers)])
        # Flatten (B, C, H, W) -> (B, C*H*W) and project to (B, seq*dim)
        self.pos_embed = nn.Linear(in_channels * latent_size * latent_size,
                                    dim * seq)
        self.head = nn.Linear(dim, out_channels)
        self.dim = dim
        self.seq = seq
        self.out_channels = out_channels
        self.latent_size = latent_size

    def forward(self, hidden_states, timestep=None, class_labels=None,
                return_dict=True, **kwargs):
        b = hidden_states.shape[0]
        x = hidden_states.flatten(1)                 # (B, C*H*W)
        x = self.pos_embed(x)                         # (B, dim*seq)
        x = x.view(b, self.seq, self.dim)             # (B, seq, dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)                             # (B, dim)
        out = self.head(x)                            # (B, out_channels)
        out = out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
        if not return_dict:
            return (out,)
        return out


# ===========================================================================
# Helpers to fabricate events / anchors
# ===========================================================================


def _make_event(layer_id: int, step_idx: int, num_steps: int,
                latent_input: torch.Tensor, class_labels: torch.Tensor,
                true_feature: torch.Tensor, sample_id: int = 0,
                module: str = "block",
                block_input_hidden: torch.Tensor = None):
    """Build a VerificationEvent with replay context via the public factory."""
    if block_input_hidden is None:
        # Use latent_input-derived fake (B, seq, dim) to simulate block input
        batch = latent_input.shape[0]
        block_input_hidden = torch.randn(batch, 4, 16)
    return make_speca_event(
        layer_id=layer_id, timestep_val=step_idx * 20,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=true_feature,
        full_hidden=true_feature,
        error_value=0.05,
        error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module=module,
        latent_input=latent_input,
        class_labels=class_labels,
        sample_id=sample_id,
        block_input_hidden=block_input_hidden,
    )


# ===========================================================================
# Tests
# ===========================================================================


def test_loss_finite_nonzero_and_backward():
    """Core path: events with block_input_hidden produce a finite, non-zero loss
    that backprops into the transformer's parameters."""
    print("=" * 60)
    print("Test 1: loss is finite / non-zero / backward-able")
    print("=" * 60)

    torch.manual_seed(0)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4,
                                    in_channels=4, out_channels=8)
    transformer.train()

    events = []
    for step_idx in range(5):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        block_input = torch.randn(2, 4, 16)  # Taylor-drifted input
        events.append(_make_event(
            layer_id=2, step_idx=step_idx, num_steps=50,
            latent_input=latent, class_labels=cl,
            true_feature=true_feat, sample_id=0,
            block_input_hidden=block_input,
        ))

    loss = compute_training_loss(
        transformer,
        curvature_events=events,
        anchor_samples=None,
        lambda_identity=1.0,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() > 0, f"loss should be > 0, got {loss.item()}"
    assert loss.requires_grad, "loss should be grad-connected"

    loss.backward()
    has_grad = any(p.grad is not None for p in transformer.parameters())
    assert has_grad, "no parameter received gradient"

    print(f"  loss = {loss.item():.6f}  y finite, > 0, backward-able")


def test_skips_events_without_block_input_hidden():
    """Events without block_input_hidden (TeaCache events, old checkpoints)
    should be skipped gracefully (not crash, not produce NaN)."""
    print("=" * 60)
    print("Test 2: events without block_input_hidden are skipped")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    # All events lack block_input_hidden -- should return grad-connected 0
    events = []
    for step_idx in range(5):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        e = make_speca_event(
            layer_id=2, timestep_val=step_idx * 20,
            step_idx=step_idx, num_steps=50,
            predicted_hidden=true_feat, full_hidden=true_feat,
            error_value=0.05, error_metric="cosine_similarity",
            model="dit", base_model_version="stub-v1",
            module="block",
            latent_input=latent, class_labels=cl,
            sample_id=0,
            # block_input_hidden NOT passed
        )
        events.append(e)

    loss = compute_training_loss(
        transformer,
        curvature_events=events,
        anchor_samples=None,
        lambda_identity=1.0,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    # With no block_input_hidden events and no anchors, loss should be 0
    assert loss.item() == 0.0, f"expected 0 loss, got {loss.item()}"
    assert loss.requires_grad, "loss should be grad-connected even when zero"
    loss.backward()
    print(f"  loss = {loss.item():.6f}  y events without block_input_hidden skipped cleanly")


def test_loss_without_block_input_plus_anchors():
    """Mix: events without block_input_hidden + anchors should produce
    anchor-only loss (events are skipped)."""
    print("=" * 60)
    print("Test 3: events without block_input_hidden + anchors")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4,
                                    in_channels=4, out_channels=8)
    transformer.train()

    # Events without block_input_hidden
    events = []
    for step_idx in range(3):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        e = make_speca_event(
            layer_id=2, timestep_val=step_idx * 20,
            step_idx=step_idx, num_steps=50,
            predicted_hidden=true_feat, full_hidden=true_feat,
            error_value=0.05, error_metric="cosine_similarity",
            model="dit", base_model_version="stub-v1",
            latent_input=latent, class_labels=cl,
            sample_id=0,
        )
        events.append(e)

    anchor = AnchorSample(
        prompt=torch.tensor([0, 1]),
        latent=torch.randn(2, 4, 4, 4),
        timestep=torch.tensor([500, 500]),
        target=torch.randn(2, 8, 2, 2),
        model="dit",
        base_model_version="stub-v1",
    )

    loss = compute_training_loss(
        transformer,
        curvature_events=events,
        anchor_samples=[anchor],
        lambda_identity=1.0,
        lambda_anchor=1.0,
        in_channels=4,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() > 0, f"anchor loss should be > 0, got {loss.item()}"
    assert loss.requires_grad, "loss should be grad-connected"
    loss.backward()
    has_grad = any(p.grad is not None for p in transformer.parameters())
    assert has_grad, "no parameter received gradient from anchor loss"

    print(f"  loss = {loss.item():.6f}  y anchor-only path works with skipped events")


def test_loss_empty_events_and_anchors():
    """Edge case: no usable events, no anchors -> grad-connected zero."""
    print("=" * 60)
    print("Test 4: empty events + anchors -> grad-connected zero")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    loss = compute_training_loss(
        transformer,
        curvature_events=[],
        anchor_samples=None,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.requires_grad, "empty loss should still be grad-connected"
    loss.backward()
    print(f"  loss = {loss.item():.6f}  y grad-connected zero, backward OK")


def test_anchor_only_path():
    """Anchor-only path produces a finite, grad-connected loss."""
    print("=" * 60)
    print("Test 5: anchor-only loss path")
    print("=" * 60)

    torch.manual_seed(1)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4,
                                    in_channels=4, out_channels=8)
    transformer.train()

    anchor = AnchorSample(
        prompt=torch.tensor([0, 1]),
        latent=torch.randn(2, 4, 4, 4),
        timestep=torch.tensor([500, 500]),
        target=torch.randn(2, 8, 2, 2),
        model="dit",
        base_model_version="stub-v1",
    )

    loss = compute_training_loss(
        transformer,
        curvature_events=[],
        anchor_samples=[anchor],
        lambda_identity=1.0,
        lambda_anchor=1.0,
        in_channels=4,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() > 0, f"anchor loss should be > 0, got {loss.item()}"
    assert loss.requires_grad, "anchor loss should be grad-connected"
    loss.backward()
    has_grad = any(p.grad is not None for p in transformer.parameters())
    assert has_grad, "no parameter received gradient from anchor loss"

    print(f"  loss = {loss.item():.6f}  y anchor-only path works")


def test_teacache_event_no_block_input_skipped():
    """TeaCache events (no block_input_hidden) should be skipped, returning
    zero loss when no anchors are provided."""
    print("=" * 60)
    print("Test 6: TeaCache event skipped (no block_input_hidden)")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    latent = torch.randn(2, 4, 4, 4)
    cl = torch.tensor([0, 1])
    true_feat = torch.randn(2, 4, 16)

    e = make_teacache_probe_event(
        layer_id=2, timestep_val=500,
        step_idx=10, num_steps=50,
        predicted_hidden=true_feat, true_hidden=true_feat,
        model="dit", base_model_version="stub-v1",
        latent_input=latent, class_labels=cl,
        sample_id=0,
    )

    loss = compute_training_loss(
        transformer,
        curvature_events=[e],
        anchor_samples=None,
        lambda_identity=1.0,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() == 0.0, (
        f"TeaCache events without block_input_hidden should give 0 loss, "
        f"got {loss.item()}"
    )
    print(f"  loss = {loss.item():.6f}  y TeaCache event skipped")


def test_lora_weights_change_after_step():
    """Sanity: one optimizer step on the new loss changes params."""
    print("=" * 60)
    print("Test 7: optimizer step changes parameters")
    print("=" * 60)

    torch.manual_seed(3)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    events = []
    for step_idx in range(5):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        block_input = torch.randn(2, 4, 16)
        events.append(_make_event(
            layer_id=2, step_idx=step_idx, num_steps=50,
            latent_input=latent, class_labels=cl,
            true_feature=true_feat, sample_id=0,
            block_input_hidden=block_input,
        ))

    params = [p for p in transformer.parameters() if p.requires_grad]
    before = torch.cat([p.detach().flatten() for p in params]).clone()

    opt = torch.optim.AdamW(params, lr=1e-2)
    opt.zero_grad()
    loss = compute_training_loss(
        transformer, curvature_events=events,
        lambda_identity=1.0,
    )
    loss.backward()
    opt.step()

    after = torch.cat([p.detach().flatten() for p in params]).clone()
    delta = (after - before).abs().max().item()
    assert delta > 0, "optimizer step did not change any parameter"

    print(f"  max |Dw| = {delta:.6f}  y params updated")


def test_lambda_identity_weighting():
    """Verify lambda_identity correctly scales the loss.
    With lambda_identity=0, only L_homing contributes (= MSE loss).
    With lambda_identity=1, total should be 2x the lambda_identity=0 case
    (since L_homing = L_identity = MSE)."""
    print("=" * 60)
    print("Test 8: lambda_identity weighting")
    print("=" * 60)

    torch.manual_seed(42)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    events = []
    for step_idx in range(3):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        block_input = torch.randn(2, 4, 16)
        events.append(_make_event(
            layer_id=1, step_idx=step_idx, num_steps=50,
            latent_input=latent, class_labels=cl,
            true_feature=true_feat, sample_id=0,
            block_input_hidden=block_input,
        ))

    loss_id0 = compute_training_loss(
        transformer, curvature_events=events,
        lambda_identity=0.0,  # only L_homing
    )
    loss_id1 = compute_training_loss(
        transformer, curvature_events=events,
        lambda_identity=1.0,  # L_homing + L_identity
    )

    # lambda_identity=1.0 should give 2x the lambda_identity=0.0 case
    # (within floating-point tolerance)
    ratio = loss_id1.item() / loss_id0.item()
    assert abs(ratio - 2.0) < 1e-5, (
        f"expected ratio ~2.0, got {ratio:.6f} "
        f"(id0={loss_id0.item():.6f}, id1={loss_id1.item():.6f})"
    )
    print(f"  lambda_id=0: {loss_id0.item():.6f}, "
          f"lambda_id=1: {loss_id1.item():.6f}, "
          f"ratio={ratio:.6f}  y 2x as expected")


# ===========================================================================
# Runner
# ===========================================================================


def main():
    print()
    test_loss_finite_nonzero_and_backward()
    test_skips_events_without_block_input_hidden()
    test_loss_without_block_input_plus_anchors()
    test_loss_empty_events_and_anchors()
    test_anchor_only_path()
    test_teacache_event_no_block_input_skipped()
    test_lora_weights_change_after_step()
    test_lambda_identity_weighting()
    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
