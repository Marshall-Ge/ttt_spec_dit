# -*- coding: utf-8 -*-
"""Mid-run reload tests: snapshot, pull, swap, dtype conversion, failure fallback.

Verifies that the training daemon can snapshot LoRA weights, the inference
thread can pull and swap them, and that failures are handled gracefully.

Run:
    python verification_feedback_loop/tests/test_mid_run_reload.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from verification_feedback_loop.lora_adapter import (
    LoRALinear,
    attach_lora_all_layers,
    save_lora_checkpoint,
    _swap_lora_weights,
    _load_state_into_wrappers,
)
from verification_feedback_loop.async_trainer import AsyncTrainingWorker
from verification_feedback_loop.config import VFLConfig
from verification_feedback_loop.replay_buffer import StratifiedReplayBuffer


# ===========================================================================
# Stub transformer
# ===========================================================================


class _StubBlock(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.attn1 = nn.ModuleDict({
            "to_q": nn.Linear(dim, dim),
            "to_k": nn.Linear(dim, dim),
            "to_v": nn.Linear(dim, dim),
            "to_out": nn.ModuleList([nn.Linear(dim, dim)]),
        })
        self.ff = nn.ModuleDict({
            "net": nn.ModuleList([
                nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim)
            ]),
        })

    def forward(self, x):
        return x


class _StubTransformer(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 16):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_StubBlock(dim) for _ in range(num_layers)])
        self.config = type("c", (), {"in_channels": 4, "out_channels": 8})()

    def forward(self, x, timestep=None, class_labels=None, **kw):
        return (x,)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_snapshot_from_wrappers(layer_wrappers):
    """Build a snapshot dict in the same format as _snapshot_lora_state."""
    snapshot = {"layers": {}}
    for layer_id, wrappers in layer_wrappers.items():
        layer_state = {}
        for path, lora in wrappers.items():
            entry = {
                "lora_A": lora.lora_A.data.detach().cpu().clone(),
                "lora_B": lora.lora_B.data.detach().cpu().clone(),
            }
            if lora.time_conditioned:
                entry["t_proj"] = {
                    k: v.detach().cpu().clone()
                    for k, v in lora.t_proj.state_dict().items()
                }
            layer_state[path] = entry
        snapshot["layers"][str(layer_id)] = layer_state
    return snapshot


# ===========================================================================
# Tests
# ===========================================================================


def test_snapshot_correctness():
    """_snapshot_lora_state should produce a snapshot matching wrapper weights."""
    print("=" * 60)
    print("Test 1: snapshot correctness")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16)
    buf = StratifiedReplayBuffer(capacity_per_stratum=100)
    cfg = VFLConfig()
    cfg.buffer_ready_min_strata = 1
    cfg.buffer_ready_min_total_samples = 5
    cfg.buffer_ready_min_anchors = 1
    cfg.poll_interval_s = 0.05

    with tempfile.TemporaryDirectory() as outdir:
        worker = AsyncTrainingWorker(
            transformer, buf, config=cfg, output_dir=outdir,
        )
        worker._ensure_train_model()

        # Set non-zero B weights on the train model
        for lid, wdict in worker._layer_wrappers.items():
            for path, lora in wdict.items():
                nn.init.uniform_(lora.lora_B, -0.5, 0.5)
                if lora.time_conditioned:
                    nn.init.uniform_(lora.t_proj[-1].weight, -0.1, 0.1)

        snapshot = worker._snapshot_lora_state()

        # Verify snapshot matches wrappers
        for lid, wdict in worker._layer_wrappers.items():
            lid_str = str(lid)
            assert lid_str in snapshot["layers"]
            for path, lora in wdict.items():
                assert path in snapshot["layers"][lid_str]
                entry = snapshot["layers"][lid_str][path]
                assert torch.allclose(entry["lora_A"], lora.lora_A.data.cpu())
                assert torch.allclose(entry["lora_B"], lora.lora_B.data.cpu())
                if lora.time_conditioned:
                    for k, v in lora.t_proj.state_dict().items():
                        assert torch.allclose(entry["t_proj"][k], v.cpu())

        worker.stop(timeout=1.0)

    print("  snapshot matches wrapper weights ✓")
    print()


def test_pull_isolation():
    """A pulled snapshot should not be mutated by a subsequent _train_once."""
    print("=" * 60)
    print("Test 2: pull isolation")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16)
    buf = StratifiedReplayBuffer(capacity_per_stratum=100)
    cfg = VFLConfig()
    cfg.buffer_ready_min_strata = 1
    cfg.buffer_ready_min_total_samples = 5
    cfg.buffer_ready_min_anchors = 1
    cfg.poll_interval_s = 0.05

    with tempfile.TemporaryDirectory() as outdir:
        worker = AsyncTrainingWorker(
            transformer, buf, config=cfg, output_dir=outdir,
        )
        worker._ensure_train_model()

        # Set B to a known value
        for lid, wdict in worker._layer_wrappers.items():
            for path, lora in wdict.items():
                lora.lora_B.data.fill_(0.1)

        with worker._swap_lock:
            worker._latest_lora_state = worker._snapshot_lora_state()

        # Pull snapshot #1
        snap1 = worker.pull_latest_lora_state()
        assert snap1 is not None

        # Change B on the train model
        for lid, wdict in worker._layer_wrappers.items():
            for path, lora in wdict.items():
                lora.lora_B.data.fill_(0.9)

        # Update snapshot (simulates a new training cycle)
        with worker._swap_lock:
            worker._latest_lora_state = worker._snapshot_lora_state()

        # Pull snapshot #2
        snap2 = worker.pull_latest_lora_state()

        # snap1 should still have B=0.1
        for lid_str in snap1["layers"]:
            for path, entry in snap1["layers"][lid_str].items():
                assert torch.allclose(entry["lora_B"], torch.tensor(0.1)), \
                    f"snap1 B should be 0.1, got {entry['lora_B'].mean().item():.4f}"

        # snap2 should have B=0.9
        for lid_str in snap2["layers"]:
            for path, entry in snap2["layers"][lid_str].items():
                assert torch.allclose(entry["lora_B"], torch.tensor(0.9)), \
                    f"snap2 B should be 0.9, got {entry['lora_B'].mean().item():.4f}"

        worker.stop(timeout=1.0)

    print("  pulled snapshots are isolated from subsequent mutations ✓")
    print()


def test_swap_correctness():
    """_swap_lora_weights should copy new state into inference wrappers."""
    print("=" * 60)
    print("Test 3: swap correctness")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=4, dim=16)
    inf_wrappers = attach_lora_all_layers(transformer, rank=4, alpha=1.0)

    # Verify B starts at zero
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            assert torch.allclose(lora.lora_B.data, torch.zeros_like(lora.lora_B.data))

    # Create a "new state" with non-zero B
    new_state = {"layers": {}}
    for lid, wdict in inf_wrappers.items():
        layer_state = {}
        for path, lora in wdict.items():
            entry = {
                "lora_A": lora.lora_A.data.cpu().clone(),
                "lora_B": torch.ones_like(lora.lora_B.data.cpu()) * 0.42,
            }
            if lora.time_conditioned:
                entry["t_proj"] = {
                    k: v.cpu().clone()
                    for k, v in lora.t_proj.state_dict().items()
                }
            layer_state[path] = entry
        new_state["layers"][str(lid)] = layer_state

    _swap_lora_weights(
        inf_wrappers, new_state,
        device=torch.device("cpu"), dtype=torch.float32)

    # Verify B now equals the new state
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            expected = 0.42
            actual = lora.lora_B.data.mean().item()
            assert abs(actual - expected) < 1e-6, \
                f"layer {lid}/{path}: expected B≈{expected}, got {actual}"

    print("  inference wrapper B updated to new state ✓")
    print()


def test_swap_dtype_conversion():
    """_swap_lora_weights should handle fp32→fp16 conversion correctly."""
    print("=" * 60)
    print("Test 4: swap dtype conversion")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=2, dim=16)
    inf_wrappers = attach_lora_all_layers(transformer, rank=4, alpha=1.0)
    # Convert inference model to fp16
    transformer.half()
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            assert lora.lora_B.dtype == torch.float16

    # Create fp32 new state
    new_state = {"layers": {}}
    for lid, wdict in inf_wrappers.items():
        layer_state = {}
        for path, lora in wdict.items():
            entry = {
                "lora_A": torch.randn_like(lora.lora_A.data).float() * 0.5,
                "lora_B": torch.randn_like(lora.lora_B.data).float() * 0.1,
            }
            if lora.time_conditioned:
                entry["t_proj"] = {}
            layer_state[path] = entry
        new_state["layers"][str(lid)] = layer_state

    _swap_lora_weights(
        inf_wrappers, new_state,
        device=torch.device("cpu"), dtype=torch.float16)

    # Verify inference wrappers remain fp16
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            assert lora.lora_A.dtype == torch.float16
            assert lora.lora_B.dtype == torch.float16
            # Values should be within fp16 range
            assert lora.lora_B.data.abs().max() < 65504

    print("  fp32→fp16 conversion correct ✓")
    print()


def test_swap_failure_fallback():
    """Shape mismatch should raise RuntimeError; inference wrappers stay unchanged."""
    print("=" * 60)
    print("Test 5: swap failure fallback")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=2, dim=16)
    inf_wrappers = attach_lora_all_layers(transformer, rank=4, alpha=1.0)

    # Record original B
    orig_B = {}
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            orig_B[(lid, path)] = lora.lora_B.data.clone()

    # Create mismatched state (wrong lora_A shape)
    new_state = {"layers": {}}
    for lid, wdict in inf_wrappers.items():
        layer_state = {}
        for path, lora in wdict.items():
            entry = {
                "lora_A": torch.randn(8, 32),  # wrong shape
                "lora_B": torch.randn(32, 8) * 0.1,
            }
            layer_state[path] = entry
        new_state["layers"][str(lid)] = layer_state

    try:
        _swap_lora_weights(
            inf_wrappers, new_state,
            device=torch.device("cpu"), dtype=torch.float32)
        assert False, "should have raised RuntimeError"
    except RuntimeError:
        pass

    # Verify inference wrappers are unchanged
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            assert torch.allclose(lora.lora_B.data, orig_B[(lid, path)]), \
                f"layer {lid}/{path} B changed despite swap failure"

    print("  swap failure leaves inference wrappers unchanged ✓")
    print()


def test_load_state_into_wrappers_old_checkpoint():
    """_load_state_into_wrappers with old-format checkpoint (no t_proj)."""
    print("=" * 60)
    print("Test 6: load old checkpoint without t_proj")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=2, dim=16)
    inf_wrappers = attach_lora_all_layers(transformer, rank=4, alpha=1.0,
                                          time_conditioned=True)

    # Verify t_proj starts at zero-init
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            if lora.time_conditioned:
                last_w = lora.t_proj[-1].weight.data
                assert torch.allclose(last_w, torch.zeros_like(last_w)), \
                    "t_proj should start zero-init"

    # Create old-format checkpoint (no t_proj)
    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "old_ckpt.pt")
        # Build a state dict without t_proj
        state = {
            "version": "v001",
            "base_model_version": "stub-v1",
            "rank": 4,
            "alpha": 1.0,
            "time_conditioned": False,  # old format
            "t_emb_dim": None,
            "layers": {},
            "metadata": {},
        }
        for lid, wdict in inf_wrappers.items():
            layer_state = {}
            for path, lora in wdict.items():
                entry = {
                    "lora_A": torch.randn_like(lora.lora_A.data),
                    "lora_B": torch.randn_like(lora.lora_B.data) * 0.1,
                }
                # No t_proj key — old format
                layer_state[path] = entry
            state["layers"][str(lid)] = layer_state
        torch.save(state, ckpt_path)

        _load_state_into_wrappers(
            inf_wrappers, ckpt_path,
            device=torch.device("cpu"), dtype=torch.float32)

    # A/B should be loaded; t_proj should remain zero-init
    for lid, wdict in inf_wrappers.items():
        for path, lora in wdict.items():
            assert not torch.allclose(lora.lora_B.data,
                                       torch.zeros_like(lora.lora_B.data)), \
                "B should have been loaded from checkpoint"
            if lora.time_conditioned:
                last_w = lora.t_proj[-1].weight.data
                assert torch.allclose(last_w, torch.zeros_like(last_w)), \
                    "t_proj should stay zero-init for old-format ckpt"

    print("  old-format checkpoint loads correctly, t_proj stays zero ✓")
    print()


# ===========================================================================
# Runner
# ===========================================================================


def main():
    print()
    test_snapshot_correctness()
    test_pull_isolation()
    test_swap_correctness()
    test_swap_dtype_conversion()
    test_swap_failure_fallback()
    test_load_state_into_wrappers_old_checkpoint()

    print("=" * 60)
    print("All mid-run reload tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
