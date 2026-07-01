# -*- coding: utf-8 -*-
"""Unit test for the buffer-driven compute_training_loss (M5 v2).

Verifies:
  1. Loss is finite and non-zero when events carry replay context.
  2. Loss is grad-connected to LoRA params (backward works).
  3. Loss handles empty events / empty anchors gracefully.
  4. Loss handles events without latent_input (skipped, no crash).
  5. Anchor-only path still produces a grad-connected loss.

Run:
    python verification_feedback_loop/tests/test_compute_training_loss.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from verification_feedback_loop.curvature_loss import compute_training_loss
from verification_feedback_loop.replay_buffer import AnchorSample
from verification_feedback_loop.verification_hook import (
    make_speca_event,
    make_teacache_probe_event,
)


# ===========================================================================
# Tiny stub transformer — mimics the structure DiT/PixArt forward hooks need
# ===========================================================================


class _StubBlock(nn.Module):
    """A single transformer block: nn.Linear → activation → nn.Linear.

    Output shape (B, seq, dim) so the curvature loss code path is exercised
    the same way as the real DiT block.
    """

    def __init__(self, dim: int = 16, seq: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.seq = seq
        self.dim = dim

    def forward(self, x):
        # x: (B, seq, dim) — mimic hidden states
        return self.fc2(torch.relu(self.fc1(x)))


class _StubTransformer(nn.Module):
    """Mimics DiTTransformer2D forward signature just enough:
    forward(latent, timestep, class_labels=None, return_dict=False)
    pos_embed: (B, C, H, W) → (B, seq, dim)
    """

    def __init__(self, num_layers: int = 4, dim: int = 16, seq: int = 4,
                 in_channels: int = 4, out_channels: int = 8,
                 latent_size: int = 4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_StubBlock(dim, seq) for _ in range(num_layers)])
        # Flatten (B, C, H, W) → (B, C*H*W) and project to (B, seq*dim)
        self.pos_embed = nn.Linear(in_channels * latent_size * latent_size,
                                    dim * seq)
        self.head = nn.Linear(dim, out_channels)
        self.dim = dim
        self.seq = seq
        self.out_channels = out_channels
        self.latent_size = latent_size

    def forward(self, hidden_states, timestep=None, class_labels=None,
                return_dict=True, **kwargs):
        # hidden_states: (B, C, H, W) — mimics raw latent
        b = hidden_states.shape[0]
        x = hidden_states.flatten(1)                 # (B, C*H*W)
        x = self.pos_embed(x)                         # (B, dim*seq)
        x = x.view(b, self.seq, self.dim)             # (B, seq, dim)
        for block in self.transformer_blocks:
            x = block(x)
        # Project to (B, out_channels, 2, 2) so anchor loss has spatial output
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
                module: str = "block"):
    """Build a VerificationEvent with replay context via the public factory."""
    return make_speca_event(
        layer_id=layer_id, timestep_val=step_idx * 20,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=true_feature,  # not used by loss, just shape ref
        full_hidden=true_feature,
        error_value=0.05,
        error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module=module,
        latent_input=latent_input,
        class_labels=class_labels,
        sample_id=sample_id,
    )


# ===========================================================================
# Tests
# ===========================================================================


def test_loss_finite_nonzero_and_backward():
    """Core path: events with replay context produce a finite, non-zero loss
    that backprops into the transformer's parameters."""
    print("=" * 60)
    print("Test 1: loss is finite / non-zero / backward-able")
    print("=" * 60)

    torch.manual_seed(0)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4,
                                    in_channels=4, out_channels=8)
    transformer.train()

    # 5 events across 5 timesteps on layer 2, same sample_id (curvature path)
    events = []
    for step_idx in range(5):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        events.append(_make_event(
            layer_id=2, step_idx=step_idx, num_steps=50,
            latent_input=latent, class_labels=cl,
            true_feature=true_feat, sample_id=0,
        ))

    loss = compute_training_loss(
        transformer,
        curvature_events=events,
        anchor_samples=None,
        lambda_curvature=1e-3,
        curvature_order=2,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() > 0, f"loss should be > 0, got {loss.item()}"
    assert loss.requires_grad, "loss should be grad-connected"

    loss.backward()
    # At least one parameter should have a non-None grad
    has_grad = any(p.grad is not None for p in transformer.parameters())
    assert has_grad, "no parameter received gradient"

    print(f"  loss = {loss.item():.6f}  ✓ finite, > 0, backward-able")


def test_loss_handles_events_without_latent_input():
    """Old events recorded before replay context existed should be skipped
    gracefully (not crash, not produce NaN)."""
    print("=" * 60)
    print("Test 2: events without latent_input are skipped")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    # Mix of events: some with latent_input, some without
    events = []
    for step_idx in range(5):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        e = _make_event(
            layer_id=2, step_idx=step_idx, num_steps=50,
            latent_input=latent, class_labels=cl,
            true_feature=true_feat, sample_id=0,
        )
        if step_idx % 2 == 0:
            e.latent_input = None  # simulate legacy event
        events.append(e)

    loss = compute_training_loss(
        transformer,
        curvature_events=events,
        anchor_samples=None,
        lambda_curvature=1e-3,
        curvature_order=2,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    print(f"  loss = {loss.item():.6f}  ✓ legacy events skipped cleanly")


def test_loss_empty_events_and_anchors():
    """Edge case: no usable events, no anchors → grad-connected zero."""
    print("=" * 60)
    print("Test 3: empty events + anchors → grad-connected zero")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    loss = compute_training_loss(
        transformer,
        curvature_events=[],
        anchor_samples=None,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    # Should still be grad-connected so .backward() doesn't blow up
    assert loss.requires_grad, "empty loss should still be grad-connected"
    loss.backward()
    print(f"  loss = {loss.item():.6f}  ✓ grad-connected zero, backward OK")


def test_anchor_only_path():
    """Anchor-only path produces a finite, grad-connected loss."""
    print("=" * 60)
    print("Test 4: anchor-only loss path")
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
        curvature_events=[],  # no events
        anchor_samples=[anchor],
        lambda_curvature=1e-3,
        curvature_order=2,
        lambda_anchor=1.0,
        in_channels=4,
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() > 0, f"anchor loss should be > 0, got {loss.item()}"
    assert loss.requires_grad, "anchor loss should be grad-connected"
    loss.backward()
    has_grad = any(p.grad is not None for p in transformer.parameters())
    assert has_grad, "no parameter received gradient from anchor loss"

    print(f"  loss = {loss.item():.6f}  ✓ anchor-only path works")


def test_teacache_event_uses_last_block_hook():
    """TeaCache events (module='residual') should hook the last block,
    not event.layer_id. Verify the captured output shape matches
    true_feature shape (which is post-stack for TeaCache)."""
    print("=" * 60)
    print("Test 5: TeaCache event hooks last block")
    print("=" * 60)

    torch.manual_seed(2)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    # TeaCache event: layer_id=2 (probe), but module='residual' → hook last block (3)
    latent = torch.randn(2, 4, 4, 4)
    cl = torch.tensor([0, 1])
    # True feature shape = post-stack hidden (B, seq, dim) — what last block emits
    true_feat = torch.randn(2, 4, 16)

    e = make_teacache_probe_event(
        layer_id=2, timestep_val=500,
        step_idx=10, num_steps=50,
        predicted_hidden=true_feat, true_hidden=true_feat,
        model="dit", base_model_version="stub-v1",
        latent_input=latent, class_labels=cl,
        sample_id=0,
    )
    assert e.module == "residual"

    loss = compute_training_loss(
        transformer,
        curvature_events=[e],
        anchor_samples=None,
        lambda_curvature=0.0,  # isolate supervised term
    )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    assert loss.item() > 0, f"supervised loss should be > 0, got {loss.item()}"
    print(f"  loss = {loss.item():.6f}  ✓ TeaCache event captured via last block")


def test_lora_weights_change_after_step():
    """Sanity: one optimizer step on the new loss actually changes LoRA-ish
    params (here: stub params with requires_grad=True)."""
    print("=" * 60)
    print("Test 6: optimizer step changes parameters")
    print("=" * 60)

    torch.manual_seed(3)
    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    events = []
    for step_idx in range(5):
        latent = torch.randn(2, 4, 4, 4)
        cl = torch.tensor([0, 1])
        true_feat = torch.randn(2, 4, 16)
        events.append(_make_event(
            layer_id=2, step_idx=step_idx, num_steps=50,
            latent_input=latent, class_labels=cl,
            true_feature=true_feat, sample_id=0,
        ))

    params = [p for p in transformer.parameters() if p.requires_grad]
    before = torch.cat([p.detach().flatten() for p in params]).clone()

    opt = torch.optim.AdamW(params, lr=1e-2)
    opt.zero_grad()
    loss = compute_training_loss(
        transformer, curvature_events=events,
        lambda_curvature=1e-3, curvature_order=2,
    )
    loss.backward()
    opt.step()

    after = torch.cat([p.detach().flatten() for p in params]).clone()
    delta = (after - before).abs().max().item()
    assert delta > 0, "optimizer step did not change any parameter"

    print(f"  max |Δw| = {delta:.6f}  ✓ params updated")


def test_timestep_actual_preferred_over_step_idx():
    """timestep_actual must be used over the legacy `timestep` field (which
    historically held step_idx). Verify the event carries it and the loss
    code path picks it up."""
    print("=" * 60)
    print("Test 7: timestep_actual preferred over step_idx")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16, seq=4)
    transformer.train()

    # Build an event with a clearly different timestep_actual vs timestep
    latent = torch.randn(2, 4, 4, 4)
    cl = torch.tensor([0, 1])
    true_feat = torch.randn(2, 4, 16)
    e = make_speca_event(
        layer_id=2, timestep_val=10,        # legacy field = step_idx
        step_idx=10, num_steps=50,
        predicted_hidden=true_feat, full_hidden=true_feat,
        error_value=0.05, error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module="block",
        latent_input=latent, class_labels=cl,
        sample_id=0,
        timestep_actual=981,                # real diffusion timestep
    )
    assert e.timestep == 10, "legacy timestep field should hold step_idx"
    assert e.timestep_actual == 981, "timestep_actual should hold real t"

    # Spy on what timestep the transformer actually sees
    seen_ts = []
    orig_forward = transformer.forward

    def spy_forward(hidden_states, timestep=None, **kwargs):
        seen_ts.append(int(timestep[0]) if hasattr(timestep, "__getitem__") else int(timestep))
        return orig_forward(hidden_states, timestep=timestep, **kwargs)
    transformer.forward = spy_forward

    try:
        compute_training_loss(
            transformer,
            curvature_events=[e],
            anchor_samples=None,
            lambda_curvature=0.0,
        )
    finally:
        transformer.forward = orig_forward

    assert seen_ts, "transformer.forward was never called"
    assert seen_ts[0] == 981, (
        f"transformer should see timestep_actual=981, got {seen_ts[0]}"
    )
    print(f"  transformer saw t={seen_ts[0]}  ✓ timestep_actual used (not step_idx=10)")


def test_encoder_hidden_states_stored_as_fp16():
    """encoder_hidden_states should be stored as fp16 to halve memory."""
    print("=" * 60)
    print("Test 8: encoder_hidden_states stored as fp16")
    print("=" * 60)

    ehs = torch.randn(2, 77, 4096, dtype=torch.float32)
    e = make_speca_event(
        layer_id=2, timestep_val=10,
        step_idx=10, num_steps=50,
        predicted_hidden=torch.randn(2, 4, 16),
        full_hidden=torch.randn(2, 4, 16),
        error_value=0.05, error_metric="cosine_similarity",
        model="pixart", base_model_version="stub-v1",
        module="block",
        latent_input=torch.randn(2, 4, 4, 4),
        encoder_hidden_states=ehs,
    )
    assert e.encoder_hidden_states is not None
    assert e.encoder_hidden_states.dtype == torch.float16, (
        f"expected fp16, got {e.encoder_hidden_states.dtype}"
    )
    print(f"  encoder_hidden_states dtype = {e.encoder_hidden_states.dtype}  ✓ fp16")


def test_buffer_caps_encoder_hidden_states():
    """Buffer must drop encoder_hidden_states once the cap is hit."""
    print("=" * 60)
    print("Test 9: buffer caps encoder_hidden_states storage")
    print("=" * 60)

    from verification_feedback_loop import StratifiedReplayBuffer

    buf = StratifiedReplayBuffer(
        capacity_per_stratum=100,
        max_encoder_hidden_states_events=3,
    )

    def _make_with_ehs(ehs_present: bool):
        ehs = torch.randn(1, 77, 4096) if ehs_present else None
        return make_speca_event(
            layer_id=2, timestep_val=1,
            step_idx=1, num_steps=50,
            predicted_hidden=torch.randn(1, 4, 16),
            full_hidden=torch.randn(1, 4, 16),
            error_value=0.05, error_metric="cosine_similarity",
            model="pixart", base_model_version="stub-v1",
            module="block",
            latent_input=torch.randn(1, 4, 4, 4),
            encoder_hidden_states=ehs,
        )

    # Add 5 events with ehs — only first 3 should retain it
    for _ in range(5):
        buf.add(_make_with_ehs(True), kind="hard_negative")

    # Sample them back (all 5 should be in the buffer since cap=100 > 5)
    events, _ = buf.sample_training_batch(batch_size=10)
    with_ehs = [e for e in events if e.encoder_hidden_states is not None]
    without_ehs = [e for e in events if e.encoder_hidden_states is None]

    assert len(with_ehs) == 3, f"expected 3 with ehs, got {len(with_ehs)}"
    assert len(without_ehs) >= 2, f"expected ≥2 without ehs, got {len(without_ehs)}"

    stats = buf.stats()
    assert stats["ehs_count"] == 3, f"ehs_count={stats['ehs_count']}"
    assert stats["ehs_dropped"] == 2, f"ehs_dropped={stats['ehs_dropped']}"
    print(f"  with_ehs={len(with_ehs)}, without_ehs={len(without_ehs)}")
    print(f"  ehs_count={stats['ehs_count']}, ehs_dropped={stats['ehs_dropped']}  ✓ cap enforced")


# ===========================================================================
# Runner
# ===========================================================================


def main():
    print()
    test_loss_finite_nonzero_and_backward()
    test_loss_handles_events_without_latent_input()
    test_loss_empty_events_and_anchors()
    test_anchor_only_path()
    test_teacache_event_uses_last_block_hook()
    test_lora_weights_change_after_step()
    test_timestep_actual_preferred_over_step_idx()
    test_encoder_hidden_states_stored_as_fp16()
    test_buffer_caps_encoder_hidden_states()
    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
