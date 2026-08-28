# -*- coding: utf-8 -*-
"""Unit tests for the lightweight VFL checkpoint manager in utils.py.

CPU-only — no torch / diffusers / GPU required. Run with:

    pytest test_checkpoint_manager.py -v
"""

import os
import time
import tempfile

from utils import get_vfl_checkpoint_dir, prune_checkpoints


def test_get_vfl_checkpoint_dir_layout():
    """Method-specific subdirectory under output_dir/checkpoints/."""
    d = get_vfl_checkpoint_dir("/tmp/run42", "teacache")
    assert d == "/tmp/run42/checkpoints/teacache"

    d2 = get_vfl_checkpoint_dir("/tmp/run42", "speca")
    assert d2 == "/tmp/run42/checkpoints/speca"
    assert d != d2  # method-scoped segregation


def test_get_vfl_checkpoint_dir_handles_relative():
    d = get_vfl_checkpoint_dir("./output", "baseline")
    assert d == "./output/checkpoints/baseline"


def test_prune_checkpoints_no_dir():
    """Missing directory is a no-op, returns 0."""
    assert prune_checkpoints("/nonexistent/path/foo", keep=3) == 0


def test_prune_checkpoints_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        assert prune_checkpoints(d, keep=3) == 0


def test_prune_checkpoints_keeps_only_n_newest():
    """With 5 files and keep=3, the 2 oldest are deleted."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(5):
            p = os.path.join(d, f"lora_candidate_v{i:03d}.pt")
            with open(p, "wb") as f:
                f.write(b"x")
            # Stagger mtimes so the prune order is deterministic
            os.utime(p, (i, i))  # v0 oldest, v4 newest
            paths.append(p)
            time.sleep(0.01)

        deleted = prune_checkpoints(d, keep=3)
        assert deleted == 2

        remaining = sorted(os.listdir(d))
        # Newest 3 = v2, v3, v4 (by mtime, not by name)
        assert remaining == ["lora_candidate_v002.pt",
                             "lora_candidate_v003.pt",
                             "lora_candidate_v004.pt"]


def test_prune_checkpoints_below_keep_no_op():
    """With 3 files and keep=3, nothing is deleted."""
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            p = os.path.join(d, f"ckpt_{i}.pt")
            with open(p, "wb") as f:
                f.write(b"x")
            time.sleep(0.01)

        assert prune_checkpoints(d, keep=3) == 0
        assert len(os.listdir(d)) == 3


def test_prune_checkpoints_ignores_non_pt_files():
    """Only .pt files are managed; other files are left alone."""
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            with open(os.path.join(d, f"ckpt_{i}.pt"), "wb") as f:
                f.write(b"x")
            time.sleep(0.01)
        # Non-.pt file that should NOT be counted or deleted
        with open(os.path.join(d, "metadata.json"), "w") as f:
            f.write("{}")

        deleted = prune_checkpoints(d, keep=3)
        assert deleted == 2
        assert "metadata.json" in os.listdir(d)


def test_prune_checkpoints_default_keep_is_3():
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            with open(os.path.join(d, f"ckpt_{i}.pt"), "wb") as f:
                f.write(b"x")
            time.sleep(0.01)

        # Default keep=3
        assert prune_checkpoints(d) == 2
