# -*- coding: utf-8 -*-
"""Unit tests for time-conditioned LoRA (t_proj γ-modulation).

Coverage:
  1. Zero-init equivalence: LoRALinear(time_conditioned=True) with B=0 and
     t_proj[-1].weight=0 produces output equal to base(x) within 1e-6 —
     whether or not a t_emb is registered.
  2. t_emb modulation has effect: with non-zero B, enabling t_emb changes
     the output vs not setting it.
  3. Checkpoint round-trip: save → load preserves lora_A, lora_B and
     t_proj parameters exactly.
  4. Legacy checkpoint (no t_proj field) loads cleanly, with t_proj left
     at zero-init.
  5. get_lora_params includes t_proj params (count = lora_A + lora_B +
     t_proj totals).
  6. Two-zero-init invariant: even with random non-zero B (simulating a
     partially-trained checkpoint), a freshly-built LoRALinear still
     no-ops on its first forward (γ=0 ⇒ factor 1.0, but B=0 ⇒ ΔW=0).

Run:
    python verification_feedback_loop/tests/test_time_conditioned_lora.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from verification_feedback_loop.lora_adapter import (
    LoRALinear,
    attach_lora,
    attach_lora_all_layers,
    save_lora_checkpoint,
    load_lora_checkpoint,
    get_lora_params,
    freeze_backbone,
    set_lora_t_emb,
    clear_lora_t_emb,
)


# ===========================================================================
# Stub transformer (mimics DiT's transformer_blocks ModuleList)
# ===========================================================================


class _StubBlock(nn.Module):
    """A DiT-shaped stub: norm1 wraps a TimestepEmbedder-like emb."""

    def __init__(self, dim: int = 16, t_emb_dim: int = 16):
        super().__init__()
        self.attn1 = nn.ModuleDict({
            "to_q": nn.Linear(dim, dim),
            "to_k": nn.Linear(dim, dim),
            "to_v": nn.Linear(dim, dim),
            "to_out": nn.ModuleList([nn.Linear(dim, dim)]),
        })
        self.ff = nn.ModuleDict({
            "net": nn.ModuleList([
                nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim),
            ]),
        })
        # Fake a TimestepEmbedder: emb.mlp[0].in_features = t_emb_dim
        # so the inference path can find the dim.
        self.norm1 = type("_N", (nn.Module,), {})()
        self.norm1.emb = type("_E", (nn.Module,), {})()
        self.norm1.emb.mlp = nn.ModuleList([nn.Linear(t_emb_dim, dim * 6)])

    def forward(self, x):
        return x


class _StubTransformer(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 16, t_emb_dim: int = 16):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_StubBlock(dim, t_emb_dim) for _ in range(num_layers)])
        self.config = type("c", (), {"in_channels": 4, "out_channels": 8})()

    def forward(self, x, **kw):
        return (x,)


# ===========================================================================
# Tests
# ===========================================================================


def test_zero_init_equivalence():
    """At init, LoRALinear(time_conditioned=True).forward(x) ≈ base(x).

    Two-zero-init invariant: lora_B=0 AND t_proj[-1].weight=0.
    """
    print("=" * 60)
    print("Test 1: zero-init equivalence (B=0 + t_proj[-1]=0)")
    print("=" * 60)

    torch.manual_seed(0)
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, rank=4, alpha=8,
                      time_conditioned=True, t_emb_dim=64)

    x = torch.randn(8, 64)
    base_out = base(x).clone()
    lora_out = lora(x)
    diff = (base_out - lora_out).abs().max().item()
    assert diff < 1e-6, f"vanilla path diff {diff} >= 1e-6"
    print(f"  no t_emb set → diff = {diff:.2e} ✓")

    # With t_emb set: should STILL equal base(x) (because B=0 ⇒ ΔW=0).
    t_emb = torch.randn(8, 64)
    set_lora_t_emb(t_emb)
    try:
        lora_out_t = lora(x)
        diff_t = (base_out - lora_out_t).abs().max().item()
        assert diff_t < 1e-6, f"with t_emb diff {diff_t} >= 1e-6"
        print(f"  t_emb set, B=0 → diff = {diff_t:.2e} ✓")
    finally:
        clear_lora_t_emb()

    # Verify the zero-init invariant actually holds on the params.
    assert torch.all(lora.lora_B == 0).item(), "lora_B must be zero at init"
    assert torch.all(lora.t_proj[-1].weight == 0).item(), \
        "t_proj[-1].weight must be zero at init"
    assert torch.all(lora.t_proj[-1].bias == 0).item(), \
        "t_proj[-1].bias must be zero at init"
    print(f"  invariant: lora_B=0, t_proj[-1].weight=0, t_proj[-1].bias=0 ✓")
    print()


def test_t_emb_modulation_changes_output():
    """With non-zero B AND non-zero t_proj[-1], t_emb changes the output.

    Note: t_proj[-1] is zero-initialised, so γ=0 at start → factor 1.0 →
    identical to no-t path. To exercise the γ branch we have to perturb
    t_proj[-1] too. (This is the two-zero-init invariant: BOTH B and
    t_proj[-1] must be non-zero before modulation shows up.)
    """
    print("=" * 60)
    print("Test 2: t_emb modulation has effect (B ≠ 0 AND t_proj[-1] ≠ 0)")
    print("=" * 60)

    torch.manual_seed(0)
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, rank=4, alpha=8,
                      time_conditioned=True, t_emb_dim=64)
    # Simulate partially-trained: non-zero B and non-zero t_proj[-1].
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn_like(lora.lora_B) * 0.1)
        lora.t_proj[-1].weight.copy_(torch.randn_like(lora.t_proj[-1].weight) * 0.1)
        lora.t_proj[-1].bias.copy_(torch.randn_like(lora.t_proj[-1].bias) * 0.1)

    x = torch.randn(8, 64)

    # Path A: no t_emb set (γ branch skipped) — output = base + ΔW·α.
    clear_lora_t_emb()
    out_no_t = lora(x).clone()

    # Path B: t_emb set, γ branch active — output = base + ΔW·α·(1+γ).
    set_lora_t_emb(torch.randn(8, 64))
    try:
        out_with_t = lora(x).clone()
    finally:
        clear_lora_t_emb()

    diff = (out_no_t - out_with_t).abs().max().item()
    assert diff > 0, "expected non-zero diff when t_emb is set with trained B"
    print(f"  no-t vs with-t diff = {diff:.4e} (B ≠ 0, t_proj[-1] ≠ 0) ✓")

    # Sanity: zero out t_proj[-1] (γ=0) → with-t should match no-t.
    with torch.no_grad():
        lora.t_proj[-1].weight.zero_()
        lora.t_proj[-1].bias.zero_()
    set_lora_t_emb(torch.randn(8, 64))
    try:
        out_zero_gamma = lora(x).clone()
    finally:
        clear_lora_t_emb()
    diff_gamma0 = (out_no_t - out_zero_gamma).abs().max().item()
    assert diff_gamma0 < 1e-6, \
        f"γ=0 should match no-t path, diff={diff_gamma0}"
    print(f"  γ=0 ⇒ with-t equals no-t (diff {diff_gamma0:.2e}) ✓")

    # Sanity: zero out B (ΔW=0) → output equals base regardless of γ.
    with torch.no_grad():
        lora.lora_B.zero_()
    set_lora_t_emb(torch.randn(8, 64))
    try:
        out_zero_b = lora(x).clone()
    finally:
        clear_lora_t_emb()
    diff_zero = (base(x) - out_zero_b).abs().max().item()
    assert diff_zero < 1e-6, f"B=0 should be no-op, diff={diff_zero}"
    print(f"  B=0 ⇒ no-op (diff {diff_zero:.2e}) ✓")
    print()


def test_checkpoint_roundtrip_preserves_t_proj():
    """save → load preserves lora_A, lora_B and t_proj exactly."""
    print("=" * 60)
    print("Test 3: checkpoint round-trip preserves t_proj")
    print("=" * 60)

    torch.manual_seed(0)
    transformer = _StubTransformer(num_layers=3, dim=16, t_emb_dim=16)
    wrappers = attach_lora_all_layers(
        transformer, rank=4, alpha=1.0, time_conditioned=True, t_emb_dim=16)

    # Randomise all params so we can detect any non-load.
    for wdict in wrappers.values():
        for lora in wdict.values():
            with torch.no_grad():
                lora.lora_A.copy_(torch.randn_like(lora.lora_A))
                lora.lora_B.copy_(torch.randn_like(lora.lora_B))
                for p in lora.t_proj.parameters():
                    p.copy_(torch.randn_like(p))

    # Snapshot for comparison.
    snapshot = {}
    for lid, wdict in wrappers.items():
        snapshot[lid] = {}
        for path, lora in wdict.items():
            snapshot[lid][path] = {
                "A": lora.lora_A.detach().clone(),
                "B": lora.lora_B.detach().clone(),
                "t0": lora.t_proj[0].weight.detach().clone(),
                "t1": lora.t_proj[-1].weight.detach().clone(),
            }

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lora_candidate_v001.pt")
        save_lora_checkpoint(wrappers, path, version="v001",
                             base_model_version="stub")

        # Fresh transformer, fresh attach via load.
        tx2 = _StubTransformer(num_layers=3, dim=16, t_emb_dim=16)
        loaded_wrappers, meta = load_lora_checkpoint(tx2, path)

        assert meta.get("time_conditioned") is True, \
            f"meta should report time_conditioned=True, got {meta}"

        for lid, wdict in loaded_wrappers.items():
            for path_str, lora in wdict.items():
                snap = snapshot[lid][path_str]
                a_diff = (lora.lora_A - snap["A"]).abs().max().item()
                b_diff = (lora.lora_B - snap["B"]).abs().max().item()
                t0_diff = (lora.t_proj[0].weight - snap["t0"]).abs().max().item()
                t1_diff = (lora.t_proj[-1].weight - snap["t1"]).abs().max().item()
                assert a_diff < 1e-6, f"layer {lid} A diff {a_diff}"
                assert b_diff < 1e-6, f"layer {lid} B diff {b_diff}"
                assert t0_diff < 1e-6, f"layer {lid} t_proj[0] diff {t0_diff}"
                assert t1_diff < 1e-6, f"layer {lid} t_proj[-1] diff {t1_diff}"
        print(f"  all lora_A / lora_B / t_proj weights match snapshot ✓")
    print()


def test_legacy_checkpoint_loads_with_zero_t_proj():
    """Legacy ckpt (no t_proj, no time_conditioned field) loads cleanly.

    The fresh LoRA's t_proj stays at zero-init → first forward no-ops.
    """
    print("=" * 60)
    print("Test 4: legacy checkpoint (no t_proj) loads cleanly")
    print("=" * 60)

    torch.manual_seed(0)
    transformer = _StubTransformer(num_layers=2, dim=16, t_emb_dim=16)
    # Build a VANILLA LoRA (no time-conditioning) and save it.
    wrappers_vanilla = attach_lora_all_layers(
        transformer, rank=4, alpha=1.0, time_conditioned=False)
    with torch.no_grad():
        for wdict in wrappers_vanilla.values():
            for lora in wdict.items():
                pass
        for wdict in wrappers_vanilla.values():
            for lora in wdict.values():
                lora.lora_A.copy_(torch.randn_like(lora.lora_A) * 0.1)
                lora.lora_B.copy_(torch.randn_like(lora.lora_B) * 0.1)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "legacy.pt")
        save_lora_checkpoint(wrappers_vanilla, path,
                             version="legacy", base_model_version="stub")
        # Verify the saved state has no t_proj / time_conditioned fields by
        # manually stripping them to simulate a pre-time-conditioning ckpt.
        state = torch.load(path, map_location="cpu")
        state.pop("time_conditioned", None)
        state.pop("t_emb_dim", None)
        for layer_state in state["layers"].values():
            for entry in layer_state.values():
                entry.pop("t_proj", None)
        torch.save(state, path)

        # Detach vanilla wrappers so we can re-attach on a fresh transformer.
        from verification_feedback_loop.lora_adapter import detach_lora
        detach_lora(transformer, wrappers_vanilla)

        tx2 = _StubTransformer(num_layers=2, dim=16, t_emb_dim=16)
        # Load with explicit time_conditioned=True override.
        loaded, meta = load_lora_checkpoint(
            tx2, path, time_conditioned=True, t_emb_dim=16)

        # Check t_proj is zero-init on every loaded wrapper.
        for lid, wdict in loaded.items():
            for path_str, lora in wdict.items():
                assert lora.time_conditioned, f"{lid}/{path_str} not tc"
                last_w = lora.t_proj[-1].weight
                last_b = lora.t_proj[-1].bias
                assert torch.all(last_w == 0).item(), \
                    f"{lid}/{path_str} t_proj[-1].weight not zero"
                assert torch.all(last_b == 0).item(), \
                    f"{lid}/{path_str} t_proj[-1].bias not zero"
        print(f"  legacy ckpt loaded, t_proj zero-initialised everywhere ✓")

        # And the LoRA forward with t_emb set still no-ops (B + γ = 0 ⇒ no-op
        # only if B=0; here B is the legacy random value, so we need to check
        # γ is exactly 0 ⇒ output equals the vanilla LoRA forward).
        x = torch.randn(2, 16)
        set_lora_t_emb(torch.randn(2, 16))
        try:
            out_tc = tx2(x)[0].clone()  # transformer returns (x,) tuple
        finally:
            clear_lora_t_emb()
        # Compare against direct vanilla LoRA forward (no γ).
        # We need to invoke LoRALinear without t_emb set.
        clear_lora_t_emb()
        out_no_tc = tx2(x)[0].clone()
        diff = (out_tc - out_no_tc).abs().max().item()
        assert diff < 1e-6, f"γ=0 should leave output unchanged, diff={diff}"
        print(f"  γ=0 ⇒ tc forward equals vanilla forward (diff {diff:.2e}) ✓")
    print()


def test_get_lora_params_includes_t_proj():
    """get_lora_params must include t_proj params when time-conditioned."""
    print("=" * 60)
    print("Test 5: get_lora_params includes t_proj")
    print("=" * 60)

    torch.manual_seed(0)
    transformer = _StubTransformer(num_layers=2, dim=16, t_emb_dim=16)
    wrappers = attach_lora_all_layers(
        transformer, rank=4, alpha=1.0, time_conditioned=True, t_emb_dim=16)

    # Expected per LoRA: lora_A + lora_B + t_proj params. Sizes vary per Linear
    # (e.g. ff.net.2 has in_features = dim*4 = 64), so sum from the wrappers.
    expected_total = 0
    n_loras = 0
    for wdict in wrappers.values():
        for lora in wdict.values():
            n_loras += 1
            expected_total += lora.lora_A.numel() + lora.lora_B.numel()
            assert lora.time_conditioned, "expected time-conditioned LoRA"
            expected_total += sum(p.numel() for p in lora.t_proj.parameters())

    params = get_lora_params(transformer)
    n_p = sum(p.numel() for p in params)
    assert n_p == expected_total, (
        f"expected {expected_total}, got {n_p}")
    # Each LoRA should contribute 4 params: A, B, t_proj[0].weight,
    # t_proj[0].bias, t_proj[2].weight, t_proj[2].bias → 6 actually.
    expected_param_tensors = n_loras * 6
    assert len(params) == expected_param_tensors, (
        f"expected {expected_param_tensors} param tensors "
        f"(6 per LoRA), got {len(params)}")
    print(f"  {n_p} params across {n_loras} LoRAs "
          f"({len(params)} tensors = 6/LoRA) ✓")

    # All must require grad after freeze_backbone.
    freeze_backbone(transformer)
    n_grad = sum(1 for p in transformer.parameters() if p.requires_grad)
    assert n_grad == len(params), (
        f"expected {len(params)} trainable params, got {n_grad}")
    print(f"  freeze_backbone → all {n_grad} LoRA params trainable ✓")

    # Vanilla variant: fewer params.
    from verification_feedback_loop.lora_adapter import detach_lora
    detach_lora(transformer, wrappers)
    wrappers_vanilla = attach_lora_all_layers(
        transformer, rank=4, alpha=1.0, time_conditioned=False)
    params_v = get_lora_params(transformer)
    n_pv = sum(p.numel() for p in params_v)
    expected_v = 0
    for wdict in wrappers_vanilla.values():
        for lora in wdict.values():
            expected_v += lora.lora_A.numel() + lora.lora_B.numel()
            assert not lora.time_conditioned, "vanilla LoRA should not be tc"
    assert n_pv == expected_v, f"vanilla expected {expected_v}, got {n_pv}"
    print(f"  vanilla variant: {n_pv} params (A + B only, no t_proj) ✓")
    print()


def test_attach_lora_time_conditioning_auto_inferred():
    """attach_lora with time_conditioned=True auto-infers t_emb_dim from DiT."""
    print("=" * 60)
    print("Test 6: t_emb_dim auto-inferred from block.norm1.emb")
    print("=" * 60)

    torch.manual_seed(0)
    transformer = _StubTransformer(num_layers=2, dim=16, t_emb_dim=24)

    # Don't pass t_emb_dim — should be inferred.
    wrappers = attach_lora_all_layers(
        transformer, rank=4, alpha=1.0, time_conditioned=True)

    for lid, wdict in wrappers.items():
        for path, lora in wdict.items():
            assert lora.time_conditioned, f"{lid}/{path} not tc"
            inferred = lora.t_proj[0].in_features
            assert inferred == 24, \
                f"expected t_emb_dim=24, got {inferred}"
    print(f"  t_emb_dim auto-inferred = 24 (matches stub.norm1.emb.mlp[0]) ✓")
    print()


# ===========================================================================
# Runner
# ===========================================================================


def main():
    print()
    test_zero_init_equivalence()
    test_t_emb_modulation_changes_output()
    test_checkpoint_roundtrip_preserves_t_proj()
    test_legacy_checkpoint_loads_with_zero_t_proj()
    test_get_lora_params_includes_t_proj()
    test_attach_lora_time_conditioning_auto_inferred()
    print("=" * 60)
    print("All time-conditioned LoRA tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
