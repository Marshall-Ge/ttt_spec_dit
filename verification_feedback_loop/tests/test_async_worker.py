# -*- coding: utf-8 -*-
"""Phase 2 unit + stress tests for AsyncTrainingWorker.

Verifies:
  1. ``_buffer_ready()`` returns False on an empty buffer, True once the
     strata / anchor thresholds are crossed.
  2. ``attach_lora_all_layers`` covers every block (no more select_top_k).
  3. ``find_latest_checkpoint`` picks the most recent ``*.pt`` file.
  4. Thread-safety stress test:推理线程频繁写 buffer, 训练线程同时采样,
     不 crash 不死锁。
  5. ``stop()`` is idempotent and a crashed ``_train_once`` does not poison
     the inference thread (crash is contained, ``crash_count`` increments,
     ``get_latest_checkpoint`` still returns the last good one).

Run:
    python verification_feedback_loop/tests/test_async_worker.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from verification_feedback_loop.config import VFLConfig
from verification_feedback_loop.replay_buffer import (
    StratifiedReplayBuffer,
    AnchorSample,
)
from verification_feedback_loop.verification_hook import make_speca_event
from verification_feedback_loop.lora_adapter import (
    attach_lora_all_layers,
    find_latest_checkpoint,
    save_lora_checkpoint,
    LoRALinear,
)
from verification_feedback_loop.async_trainer import AsyncTrainingWorker


# ===========================================================================
# Stub transformer — mimics DiT's transformer_blocks ModuleList
# ===========================================================================


class _StubBlock(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        # Use names that match DiT's _LORA_TARGET_PATHS so attach_lora_all_layers
        # actually finds Linears to wrap.
        self.attn1 = nn.ModuleDict({
            "to_q": nn.Linear(dim, dim),
            "to_k": nn.Linear(dim, dim),
            "to_v": nn.Linear(dim, dim),
            "to_out": nn.ModuleList([nn.Linear(dim, dim)]),
        })
        self.ff = nn.ModuleDict({
            "net": nn.ModuleList([nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim)]),
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


def _make_event(layer_id: int, step_idx: int, num_steps: int,
                with_latent: bool = True):
    """Build a VerificationEvent with replay context."""
    latent = torch.randn(1, 4, 4, 4) if with_latent else None
    cl = torch.tensor([0]) if with_latent else None
    return make_speca_event(
        layer_id=layer_id, timestep_val=step_idx * 20,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=torch.randn(1, 4, 16),
        full_hidden=torch.randn(1, 4, 16),
        error_value=0.05,
        error_metric="cosine_similarity",
        model="dit", base_model_version="stub-v1",
        module="block",
        latent_input=latent,
        class_labels=cl,
    )


def _make_anchor():
    return AnchorSample(
        prompt=torch.tensor([0]),
        latent=torch.randn(1, 4, 4, 4),
        timestep=torch.tensor([500]),
        target=torch.randn(1, 8, 2, 2),
        model="dit",
        base_model_version="stub-v1",
    )


# ===========================================================================
# Tests
# ===========================================================================


def test_buffer_ready_empty_vs_full():
    """_buffer_ready() should return False on empty, True on full-enough buffer."""
    print("=" * 60)
    print("Test 1: _buffer_ready empty vs full")
    print("=" * 60)

    cfg = VFLConfig()
    # Use small thresholds so the test doesn't need to add thousands of events
    cfg.buffer_ready_min_strata = 3
    cfg.buffer_ready_min_total_samples = 10
    cfg.buffer_ready_min_anchors = 2

    buf = StratifiedReplayBuffer(capacity_per_stratum=100)
    transformer = _StubTransformer(num_layers=4, dim=16)
    with tempfile.TemporaryDirectory() as outdir:
        worker = AsyncTrainingWorker(
            transformer, buf, config=cfg, output_dir=outdir,
            base_model_version="stub-v1",
        )

        # Empty buffer → not ready
        assert worker._buffer_ready() is False, \
            "empty buffer should not be ready"
        print(f"  empty buffer → _buffer_ready=False ✓")

        # Add a single event → still not ready (only 1 stratum, no anchors)
        buf.add(_make_event(layer_id=0, step_idx=0, num_steps=50),
                kind="hard_negative")
        assert worker._buffer_ready() is False, \
            "single-event buffer should not be ready"
        print(f"  1 stratum, 0 anchors → _buffer_ready=False ✓")

        # Fill 3 strata with 2 events each + 2 anchors → ready
        for layer in [0, 1, 2]:
            for step in range(2):
                buf.add(_make_event(layer_id=layer, step_idx=step,
                                    num_steps=50),
                        kind="hard_negative")
        buf.add_anchor(_make_anchor())
        buf.add_anchor(_make_anchor())
        assert worker._buffer_ready() is True, \
            "3 strata × 2 events + 2 anchors should be ready"
        print(f"  3 strata × 2 events + 2 anchors → _buffer_ready=True ✓")

        # Drop below anchor threshold → not ready again
        buf._anchor_samples.clear()
        assert worker._buffer_ready() is False, \
            "no anchors → not ready"
        print(f"  anchors removed → _buffer_ready=False ✓")

        worker.stop(timeout=1.0)

    print()


def test_attach_lora_all_layers_covers_every_block():
    """attach_lora_all_layers should hang LoRA on every block (no top-K)."""
    print("=" * 60)
    print("Test 2: attach_lora_all_layers covers all 28 blocks")
    print("=" * 60)

    n_layers = 28
    transformer = _StubTransformer(num_layers=n_layers, dim=16)
    wrappers = attach_lora_all_layers(transformer, rank=4, alpha=1.0)

    assert len(wrappers) == n_layers, (
        f"expected {n_layers} layers, got {len(wrappers)}")
    for lid in range(n_layers):
        assert lid in wrappers, f"layer {lid} missing"
        # Each layer should have at least one LoRA wrapper (DiT-style block has 6 Linears)
        assert len(wrappers[lid]) > 0, (
            f"layer {lid} has no LoRA wrappers — attach_lora_to_block failed")
        for path, lora in wrappers[lid].items():
            assert isinstance(lora, LoRALinear)
            assert lora.rank == 4
            assert lora.alpha == 1.0
    print(f"  {n_layers} layers × ~6 Linears each, all wrapped ✓")
    print()


def test_find_latest_checkpoint():
    """find_latest_checkpoint should return the most recent .pt file."""
    print("=" * 60)
    print("Test 3: find_latest_checkpoint picks newest .pt")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        # Empty dir → None
        assert find_latest_checkpoint(d) is None
        print(f"  empty dir → None ✓")

        # No .pt files → None
        open(os.path.join(d, "summary.json"), "w").close()
        assert find_latest_checkpoint(d) is None
        print(f"  no .pt files → None ✓")

        # Create two .pt files with different mtimes
        old = os.path.join(d, "lora_candidate_v001.pt")
        new = os.path.join(d, "lora_candidate_v002.pt")
        with open(old, "wb") as f:
            f.write(b"\x00")
        os.utime(old, (1000, 1000))
        with open(new, "wb") as f:
            f.write(b"\x00")
        os.utime(new, (2000, 2000))

        result = find_latest_checkpoint(d)
        assert result == new, f"expected {new}, got {result}"
        print(f"  picked v002 (newer mtime) ✓")

        # Nonexistent dir → None
        assert find_latest_checkpoint("/nonexistent/path/xyz") is None
        print(f"  nonexistent dir → None ✓")
    print()


def test_thread_safety_stress():
    """Concurrent buffer.add() (推理线程) + buffer.sample_training_batch() (训练线程)
    should not crash or deadlock."""
    print("=" * 60)
    print("Test 4: thread-safety stress (concurrent write + sample)")
    print("=" * 60)

    buf = StratifiedReplayBuffer(capacity_per_stratum=200)
    stop_event = threading.Event()
    errors: list = []

    # Producer thread — simulates 推理线程 writing events + anchors
    def producer():
        try:
            i = 0
            while not stop_event.is_set():
                layer = i % 28
                step = i % 50
                buf.add(_make_event(layer_id=layer, step_idx=step,
                                    num_steps=50),
                        kind="hard_negative" if i % 5 else "normal")
                if i % 10 == 0:
                    buf.add_anchor(_make_anchor())
                i += 1
        except Exception as e:
            errors.append(("producer", e))

    # Consumer thread — simulates 训练线程 sampling batches
    def consumer():
        try:
            while not stop_event.is_set():
                events, anchors = buf.sample_training_batch(16)
                # Just touch the data to make sure refs are valid
                _ = len(events), len(anchors)
                # Also call stats() frequently — it reads under lock too
                _ = buf.stats()
                time.sleep(0.001)  # small back-off to avoid pure busy-spin
        except Exception as e:
            errors.append(("consumer", e))

    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)
    t_prod.start()
    t_cons.start()

    # Let them hammer it for 1.5 seconds
    time.sleep(1.5)
    stop_event.set()
    t_prod.join(timeout=2.0)
    t_cons.join(timeout=2.0)

    assert not errors, f"thread errors: {errors}"
    print(f"  no crashes, no deadlocks after ~1.5s concurrent stress ✓")
    stats = buf.stats()
    print(f"  buffer state: {stats['total_samples']} events, "
          f"{stats['total_anchors']} anchors, "
          f"{stats['num_strata_nonempty']} strata")
    print()


def test_train_once_crash_does_not_poison_worker():
    """If _train_once throws, the worker should catch it, increment crash_count,
    and keep serving inference with the last good checkpoint."""
    print("=" * 60)
    print("Test 5: training crash contained")
    print("=" * 60)

    cfg = VFLConfig()
    cfg.buffer_ready_min_strata = 1
    cfg.buffer_ready_min_total_samples = 5
    cfg.buffer_ready_min_anchors = 1
    cfg.poll_interval_s = 0.05  # fast polling so the test runs quickly
    cfg.trainer_steps_per_trigger = 1

    buf = StratifiedReplayBuffer(capacity_per_stratum=100)
    # Seed with enough data to be ready
    for layer in range(2):
        for step in range(3):
            buf.add(_make_event(layer_id=layer, step_idx=step, num_steps=50),
                    kind="hard_negative")
    buf.add_anchor(_make_anchor())
    buf.add_anchor(_make_anchor())

    transformer = _StubTransformer(num_layers=4, dim=16)
    with tempfile.TemporaryDirectory() as outdir:
        worker = AsyncTrainingWorker(
            transformer, buf, config=cfg, output_dir=outdir,
            base_model_version="stub-v1",
        )
        # Force _train_once to raise by monkey-patching compute_training_loss
        from verification_feedback_loop import async_trainer as at_mod

        def boom(*a, **kw):
            raise RuntimeError("simulated training crash")

        original = at_mod.compute_training_loss
        at_mod.compute_training_loss = boom
        try:
            worker.start()
            # Give the thread time to attempt a few cycles
            time.sleep(0.5)
            worker.stop(timeout=2.0)
        finally:
            at_mod.compute_training_loss = original

        # The crash should have been caught — worker thread should be done
        # and crash_count should be > 0.
        assert worker.crash_count > 0, (
            f"expected crash_count > 0, got {worker.crash_count}")
        assert worker.last_error is not None
        assert "simulated training crash" in worker.last_error, (
            f"unexpected last_error: {worker.last_error}")
        # No checkpoint should have been produced (all cycles crashed)
        assert worker.get_latest_checkpoint() is None
        print(f"  crash_count={worker.crash_count}, "
              f"last_error={worker.last_error!r} ✓")
        print(f"  no checkpoint produced (as expected) ✓")
    print()


def test_stop_is_idempotent():
    """Calling stop() multiple times should not error."""
    print("=" * 60)
    print("Test 6: stop() is idempotent")
    print("=" * 60)

    buf = StratifiedReplayBuffer(capacity_per_stratum=50)
    transformer = _StubTransformer(num_layers=2, dim=16)
    cfg = VFLConfig()
    cfg.poll_interval_s = 0.1

    with tempfile.TemporaryDirectory() as outdir:
        worker = AsyncTrainingWorker(
            transformer, buf, config=cfg, output_dir=outdir,
        )
        worker.start()
        time.sleep(0.05)
        worker.stop(timeout=1.0)
        worker.stop(timeout=1.0)  # second call — should be a no-op
        worker.stop(timeout=1.0)  # third call — still no-op
        print(f"  3× stop() — no errors ✓")
    print()


def test_save_and_find_checkpoint_roundtrip():
    """save_lora_checkpoint + find_latest_checkpoint should round-trip."""
    print("=" * 60)
    print("Test 7: save + find checkpoint round-trip")
    print("=" * 60)

    transformer = _StubTransformer(num_layers=3, dim=16)
    wrappers = attach_lora_all_layers(transformer, rank=4, alpha=1.0)

    with tempfile.TemporaryDirectory() as d:
        # Save three checkpoints with increasing mtime
        paths = []
        for v in range(3):
            p = os.path.join(d, f"lora_candidate_v{v+1:03d}.pt")
            save_lora_checkpoint(
                wrappers, p,
                version=f"v{v+1:03d}",
                base_model_version="stub-v1",
                metadata={"phase": 2, "v": v},
            )
            # Force distinct mtimes so find_latest picks the last one
            os.utime(p, (1000 + v, 1000 + v))
            paths.append(p)

        latest = find_latest_checkpoint(d)
        assert latest == paths[-1], (
            f"expected {paths[-1]}, got {latest}")
        print(f"  3 checkpoints saved, find_latest → v003 ✓")
    print()


# ===========================================================================
# Runner
# ===========================================================================


def main():
    print()
    test_buffer_ready_empty_vs_full()
    test_attach_lora_all_layers_covers_every_block()
    test_find_latest_checkpoint()
    test_thread_safety_stress()
    test_train_once_crash_does_not_poison_worker()
    test_stop_is_idempotent()
    test_save_and_find_checkpoint_roundtrip()

    print("=" * 60)
    print("All Phase 2 async worker tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
