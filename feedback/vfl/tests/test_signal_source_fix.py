# -*- coding: utf-8 -*-
"""Unit tests for the VFL signal-source fix (block_input_hidden per-block replay).

Verifies:
  1. block_input_hidden is stored in VerificationEvent and survives round-trip
  2. Per-block replay produces different output than vanilla (loss > 0 at no-op)
  3. Gradient flows to LoRA B matrices through per-block replay
  4. Backward compat: events without block_input_hidden fall back to vanilla

Run:
    python feedback.vfl/tests/test_signal_source_fix.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerators.speca import SpecACache, SpecAState, speca_init
from feedback.vfl.curvature_loss import (
    compute_training_loss,
    _run_block_dit,
    _run_block_pixart,
)
from feedback.vfl.verification_hook import (
    VerificationEvent,
    make_speca_event,
)
from feedback.vfl.lora_adapter import (
    attach_lora_all_layers,
    LoRALinear,
)


# ===========================================================================
# Tiny stub transformer — supports SpecA forward path
# ===========================================================================


class _StubAttn(nn.Module):
    """Minimal attention-like module with LoRA-target submodules."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        return self.to_out[0](q + k + v)


class _StubFF(nn.Module):
    """Minimal feed-forward module with LoRA-target submodules."""

    def __init__(self, dim: int = 16):
        super().__init__()
        _proj = nn.Module()
        _proj.proj = nn.Linear(dim, dim)
        self.net = nn.ModuleList([_proj, nn.Identity(), nn.Linear(dim, dim)])

    def forward(self, x):
        return self.net[2](torch.relu(self.net[0].proj(x)))


class _StubAdaLNZero(nn.Module):
    """Minimal adaLN-Zero stub: returns (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.emb = nn.Linear(1, 6 * dim)
        self.dim = dim

    def forward(self, x, timestep=None, class_labels=None, hidden_dtype=None):
        B, L, D = x.shape
        t = timestep.float().reshape(-1, 1) if timestep is not None else torch.zeros(B, 1, device=x.device)
        emb = self.emb(t)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            emb.chunk(6, dim=1)
        x_norm = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp


class _StubSpecABlock(nn.Module):
    """A single transformer block matching real DiT block structure.

    Has norm1 (AdaLayerNormZero), attn1 (callable), norm3 (LayerNorm),
    ff (callable). Submodules have LoRA-target names.
    """

    def __init__(self, dim: int = 16, seq: int = 4):
        super().__init__()
        self.norm1 = _StubAdaLNZero(dim)
        self.attn1 = _StubAttn(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _StubFF(dim)
        self.dim = dim

    def forward(self, x, timestep=None, class_labels=None):
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            x, timestep=timestep, class_labels=class_labels,
            hidden_dtype=x.dtype)
        attn_out = self.attn1(norm_hidden)
        x = x + gate_msa.unsqueeze(1) * attn_out
        norm_ff = self.norm3(x)
        modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = self.ff(modulated_ff)
        x = x + gate_mlp.unsqueeze(1) * ff_out
        return x


class _StubSpecATransformer(nn.Module):
    """Stub transformer that supports SpecA forward (current, cache_dic).

    In the full step path, calls block(x) so forward hooks fire.
    In the Taylor path, uses cache_step_dit with pre-populated cache.
    """

    def __init__(self, num_layers: int = 4, dim: int = 16, seq: int = 4,
                 in_channels: int = 4, out_channels: int = 8,
                 latent_size: int = 4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_StubSpecABlock(dim, seq) for _ in range(num_layers)])
        self.pos_embed = nn.Linear(in_channels * latent_size * latent_size,
                                   dim * seq)
        self.head = nn.Linear(dim, out_channels)
        self.dim = dim
        self.seq = seq
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.config = type("Cfg", (), {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "patch_size": 2,
        })()

    def forward(self, hidden_states, timestep=None, class_labels=None,
                current=None, cache_dic=None, return_dict=True, **kwargs):
        from accelerators.speca import (
            speca_cal_type, cache_step_dit, compute_error_gate,
        )
        use_speca = current is not None and cache_dic is not None
        if use_speca:
            speca_cal_type(cache_dic, current)

        b = hidden_states.shape[0]
        x = hidden_states.flatten(1)
        x = self.pos_embed(x)
        x = x.view(b, self.seq, self.dim)

        for layer_idx, block in enumerate(self.transformer_blocks):
            if use_speca:
                current.layer = layer_idx
            step_type = 'full' if (not use_speca) else current.type

            if step_type == 'full':
                x = block(x, timestep=timestep, class_labels=class_labels)

            elif step_type == 'Taylor':
                distance = current.step - current.activated_steps[-1]
                check_layer = cache_dic.check_layer
                do_check = (layer_idx == check_layer and cache_dic.check)
                _block_input = None
                if do_check:
                    _block_input = x.clone()
                    full_hidden = _block_input

                gate_msa = torch.ones(b, self.dim, device=x.device)
                gate_mlp = torch.ones(b, self.dim, device=x.device)
                x = cache_step_dit(
                    x,
                    cache_dic.cache[-1][layer_idx]['attn'],
                    cache_dic.cache[-1][layer_idx]['mlp'],
                    gate_msa, gate_mlp, distance,
                )

                if do_check:
                    full_hidden = block(full_hidden, timestep=timestep,
                                       class_labels=class_labels)
                    gate_value, _ = compute_error_gate(
                        x, full_hidden,
                        metric=cache_dic.error_metric,
                    )
                    current.last_layer_error = gate_value

        x = x.mean(dim=1)
        out = self.head(x)
        out = out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
        if not return_dict:
            return (out,)
        return out


# ===========================================================================
# Helpers
# ===========================================================================


def _build_speca_state(num_layers=4, num_steps=10):
    """Build a SpecACache + SpecAState pair and populate with one full step.

    Sets step to a value past the first_enhance window so speca_cal_type
    can decide 'Taylor' with check=True.
    """
    cache_dic, current = speca_init(
        num_steps=num_steps,
        base_threshold=0.01,
        decay_rate=0.01,
        min_taylor_steps=2,
        max_taylor_steps=5,
        num_layers=num_layers,
        error_metric="cosine_similarity",
        check_layer=num_layers - 1,
    )
    current.step = num_steps - 1
    current.type = 'full'
    current.last_type = 'full'
    current.activated_steps = [num_steps - 1]
    for layer_idx in range(num_layers):
        for module in ['attn', 'mlp']:
            cache_dic.cache[-1][layer_idx][module] = [
                torch.randn(2, 4, 16) for _ in range(3)
            ]
    cache_dic.full_count = 1
    cache_dic.cache_counter = 0

    target_step = num_steps - 5
    current.step = target_step
    current.type = 'Taylor'
    current.last_type = 'Taylor'
    current.activated_steps = [num_steps - 1, num_steps - 2, num_steps - 3, num_steps - 4]
    cache_dic.taylor_step_counter = 3
    cache_dic.check = True
    cache_dic.full_count = 4
    return cache_dic, current


# ===========================================================================
# Test 1: block_input_hidden stored in event
# ===========================================================================


def test_block_input_stored():
    """make_speca_event stores block_input_hidden in the event."""
    print("=" * 60)
    print("Test 1: block_input_hidden stored in event")
    print("=" * 60)

    torch.manual_seed(10)
    block_input = torch.randn(1, 4, 16)
    true_feat = torch.randn(1, 4, 16)
    pred_feat = torch.randn(1, 4, 16)
    latent = torch.randn(1, 4, 4, 4)

    event = make_speca_event(
        layer_id=2, timestep_val=500,
        step_idx=5, num_steps=10,
        predicted_hidden=pred_feat,
        full_hidden=true_feat,
        error_value=0.05,
        error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module="block",
        latent_input=latent,
        class_labels=torch.tensor([3]),
        block_input_hidden=block_input,
    )

    assert event.block_input_hidden is not None, "block_input_hidden should be stored"
    assert event.block_input_hidden.shape == (1, 4, 16), \
        f"shape mismatch: {event.block_input_hidden.shape}"
    assert event.block_input_hidden.dtype == torch.float16, \
        "block_input_hidden should be fp16"

    # Without block_input_hidden
    event2 = make_speca_event(
        layer_id=2, timestep_val=500,
        step_idx=5, num_steps=10,
        predicted_hidden=pred_feat,
        full_hidden=true_feat,
        error_value=0.05,
        error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module="block",
    )
    assert event2.block_input_hidden is None, \
        "block_input_hidden should be None when not provided"

    # to_dict includes the flag
    d = event.to_dict()
    assert d["has_block_input_hidden"] is True
    d2 = event2.to_dict()
    assert d2["has_block_input_hidden"] is False

    print("  block_input_hidden stored and serialized correctly")


# ===========================================================================
# Test 2: Per-block replay differs from vanilla → loss > 0 at no-op
# ===========================================================================


def test_per_block_replay_loss_positive():
    """Per-block replay on Taylor-predicted input gives loss > 0 at LoRA no-op.

    The block_input_hidden comes from the SpecA Taylor path (accumulated
    prediction errors from prior layers). Running the block on this input
    produces a different output than the vanilla forward, giving non-zero
    supervised loss even when LoRA B=0.
    """
    print("=" * 60)
    print("Test 2: per-block replay loss > 0 at no-op")
    print("=" * 60)

    torch.manual_seed(42)
    num_layers, num_steps = 4, 10
    transformer = _StubSpecATransformer(num_layers=num_layers, dim=16, seq=4)
    transformer.eval()

    cache_dic, current = _build_speca_state(num_layers=num_layers,
                                             num_steps=num_steps)
    check_layer = cache_dic.check_layer

    latent = torch.randn(2, 4, 4, 4)
    cl = torch.tensor([3, 7])

    # Run SpecA Taylor forward to capture block_input at check_layer
    block_input_captured = {}
    def _input_hook(mod, inp):
        block_input_captured['inp'] = inp[0].detach().clone()
    h = transformer.transformer_blocks[check_layer].register_forward_pre_hook(
        _input_hook)
    with torch.no_grad():
        transformer(latent, timestep=torch.tensor([current.step]),
                    class_labels=cl, current=current, cache_dic=cache_dic,
                    return_dict=False)
    h.remove()
    block_input = block_input_captured['inp']

    # Also run vanilla forward and capture block output at check_layer
    target_captured = {}
    def _target_hook(mod, inp, out):
        target_captured['out'] = out.detach().clone()
    h = transformer.transformer_blocks[check_layer].register_forward_hook(
        _target_hook)
    with torch.no_grad():
        transformer(latent, timestep=torch.tensor([current.step]),
                    class_labels=cl, current=None, cache_dic=None,
                    return_dict=False)
    h.remove()
    vanilla_output = target_captured['out']

    # Per-block replay: run check_layer block on Taylor-predicted input
    with torch.no_grad():
        replay_output = _run_block_dit(
            transformer.transformer_blocks[check_layer],
            block_input,
            timestep=torch.tensor([current.step]),
            class_labels=cl,
            dtype=block_input.dtype,
        )

    # The replay output should differ from vanilla
    diff = (replay_output - vanilla_output).abs().max().item()
    mse = F.mse_loss(replay_output, vanilla_output).item()
    assert mse > 1e-6, f"Per-block replay should differ from vanilla, MSE={mse:.2e}"

    print(f"  max |replay - vanilla| = {diff:.6f}")
    print(f"  MSE(replay, vanilla)  = {mse:.6f}")
    print(f"  per-block replay differs from vanilla → loss > 0 at no-op")


# ===========================================================================
# Test 3: Gradient flows to LoRA B through per-block replay
# ===========================================================================


def test_gradient_flows_to_lora_b():
    """Per-block replay with LoRA gives non-zero gradient to LoRA B.

    The block runs on Taylor-predicted input (block_input_hidden), and
    the target is the vanilla output. LoRA modifies the block's submodules,
    so gradient flows through them.
    """
    print("=" * 60)
    print("Test 3: gradient flows to LoRA B")
    print("=" * 60)

    torch.manual_seed(200)
    num_layers, num_steps = 4, 10
    transformer = _StubSpecATransformer(num_layers=num_layers, dim=16, seq=4)

    lora_wrappers = attach_lora_all_layers(
        transformer, rank=4, alpha=1.0, time_conditioned=False)
    n_lora = sum(len(d) for d in lora_wrappers.values())
    transformer.train()

    cache_dic, current = _build_speca_state(num_layers=num_layers,
                                             num_steps=num_steps)
    check_layer = cache_dic.check_layer

    latent = torch.randn(2, 4, 4, 4)
    cl = torch.tensor([3, 7])

    # Capture block_input from SpecA Taylor path
    block_input_captured = {}
    def _input_hook(mod, inp):
        block_input_captured['inp'] = inp[0].clone()
    h = transformer.transformer_blocks[check_layer].register_forward_pre_hook(
        _input_hook)
    with torch.no_grad():
        transformer(latent, timestep=torch.tensor([current.step]),
                    class_labels=cl, current=current, cache_dic=cache_dic,
                    return_dict=False)
    h.remove()
    block_input = block_input_captured['inp']

    # Capture vanilla target
    target_captured = {}
    def _target_hook(mod, inp, out):
        target_captured['out'] = out.detach().clone()
    h = transformer.transformer_blocks[check_layer].register_forward_hook(
        _target_hook)
    with torch.no_grad():
        transformer(latent, timestep=torch.tensor([current.step]),
                    class_labels=cl, current=None, cache_dic=None,
                    return_dict=False)
    h.remove()
    target = target_captured['out']

    # Per-block replay with LoRA (with grad)
    lora_hidden = _run_block_dit(
        transformer.transformer_blocks[check_layer],
        block_input,
        timestep=torch.tensor([current.step]),
        class_labels=cl,
        dtype=block_input.dtype,
    )

    loss = F.mse_loss(lora_hidden, target)
    assert loss.item() > 1e-8, f"Loss should be > 0, got {loss.item():.2e}"
    loss.backward()

    lora_b_grads = []
    for layer_dict in lora_wrappers.values():
        for name, lora in layer_dict.items():
            if hasattr(lora, 'lora_B') and lora.lora_B.grad is not None:
                grad_norm = lora.lora_B.grad.norm().item()
                lora_b_grads.append(grad_norm)

    has_nonzero_grad = len(lora_b_grads) > 0 and any(g > 1e-8 for g in lora_b_grads)
    assert has_nonzero_grad, \
        f"No LoRA B matrix received non-zero gradient. grad norms: {lora_b_grads[:5]}"

    max_grad = max(lora_b_grads) if lora_b_grads else 0
    print(f"  loss = {loss.item():.6f}")
    print(f"  LoRA B params with grad: {len(lora_b_grads)}/{n_lora}")
    print(f"  max LoRA B grad norm = {max_grad:.8f}")
    print(f"  gradient flows to LoRA B")


# ===========================================================================
# Test 4: Backward compatibility — no block_input_hidden → vanilla forward
# ===========================================================================


def test_backward_compat_no_block_input():
    """Events without block_input_hidden fall back to vanilla forward."""
    print("=" * 60)
    print("Test 4: backward compat — no block_input_hidden")
    print("=" * 60)

    torch.manual_seed(300)
    num_layers, num_steps = 4, 10
    transformer = _StubSpecATransformer(num_layers=num_layers, dim=16, seq=4)
    transformer.train()

    # The stub's norm1.emb is a plain nn.Linear that doesn't accept
    # hidden_dtype (the real CombinedTimestepLabelEmbeddings does).
    # Patch it to accept and ignore the extra args so the fallback vanilla
    # forward in compute_training_loss doesn't crash.
    for block in transformer.transformer_blocks:
        orig_emb = block.norm1.emb
        class _PatchedEmb(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, *args, **kwargs):
                # Only pass the first positional arg (timestep) to nn.Linear
                return self.inner(args[0].float().reshape(-1, 1))
        block.norm1.emb = _PatchedEmb(orig_emb)

    latent = torch.randn(2, 4, 4, 4)
    cl = torch.tensor([0, 1])
    true_feat = torch.randn(2, 4, 16)

    event = make_speca_event(
        layer_id=2, timestep_val=500,
        step_idx=5, num_steps=10,
        predicted_hidden=true_feat,
        full_hidden=true_feat,
        error_value=0.05,
        error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module="block",
        latent_input=latent,
        class_labels=cl,
        # block_input_hidden NOT passed → no snapshot
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = compute_training_loss(
            transformer,
            curvature_events=[event],
            anchor_samples=None,
            lambda_anchor=0.0,
        )

    assert torch.isfinite(loss).all(), f"loss not finite: {loss}"
    print(f"  loss = {loss.item():.6f}  vanilla forward fallback works")


# ===========================================================================
# Main
# ===========================================================================


if __name__ == "__main__":
    test_block_input_stored()
    test_per_block_replay_loss_positive()
    test_gradient_flows_to_lora_b()
    test_backward_compat_no_block_input()
    print("\n" + "=" * 60)
    print("All signal-source fix tests passed!")
    print("=" * 60)
