"""Tests for the per-image latent seed formula (--latent-seed-offset).

The (image, latent draw) cell is seeded deterministically by
``latent_seed_for_index`` so that replicate runs over the SAME images can
draw independent latents per image. Load-bearing invariants:

- offset=0 reproduces the legacy ``100000 + absolute_idx`` formula
  bit-identically (every existing run and cached comparison depends on it);
- distinct offsets give disjoint seed sets for the same image indices
  (stride 1_000_000 > 50_000, the largest possible image index);
- the same offset is reproducible across calls;
- the offset is validated by the CLI and persisted into the results config.
"""

import pytest

from main import parse_args, validate_args
from utils import latent_seed_for_index

_INDICES = [0, 1, 499, 49999]


# ===========================================================================
# The seed formula itself
# ===========================================================================


def test_offset_zero_reproduces_legacy_seed_sequence():
    """offset=0 must be bit-identical to the legacy `100000 + idx` formula."""
    for idx in _INDICES:
        assert latent_seed_for_index(idx, 0) == 100000 + idx


def test_offset_zero_reproducible_across_calls():
    for idx in _INDICES:
        assert latent_seed_for_index(idx, 0) == latent_seed_for_index(idx, 0)


def test_same_offset_reproducible_across_calls():
    for offset in (1, 2, 7, 12345):
        for idx in _INDICES:
            a = latent_seed_for_index(idx, offset)
            b = latent_seed_for_index(idx, offset)
            assert a == b


def test_distinct_offsets_disjoint_for_same_images():
    """Different offsets must never collide on the same image index, so
    per-image draws are genuinely independent."""
    for idx in _INDICES:
        seeds = {latent_seed_for_index(idx, o) for o in (0, 1, 2, 3)}
        assert len(seeds) == 4, f"offsets collided at index {idx}"


def test_seed_monotone_within_offset_across_images():
    """Within one offset, seeds are strictly increasing in the image index —
    each image gets a unique latent seed."""
    for offset in (0, 1, 2):
        seq = [latent_seed_for_index(i, offset) for i in _INDICES]
        assert seq == sorted(seq)
        assert len(set(seq)) == len(seq)


def test_negative_offset_rejected():
    with pytest.raises(ValueError):
        latent_seed_for_index(0, -1)


# ===========================================================================
# CLI surface
# ===========================================================================


def _parse(argv):
    """parse_args() reads sys.argv directly; shim it for CLI-surface tests."""
    import sys
    old = sys.argv
    sys.argv = ["main.py"] + argv
    try:
        return parse_args()
    finally:
        sys.argv = old


def test_cli_defaults_latent_seed_offset_to_zero():
    args = _parse([
        "--model", "dit", "--task", "c2i", "--dataset", "imagenet",
        "--method", "baseline", "--metrics", "fid",
    ])
    assert args.latent_seed_offset == 0


def test_cli_accepts_positive_offset():
    args = _parse([
        "--model", "dit", "--task", "c2i", "--dataset", "imagenet",
        "--method", "baseline", "--metrics", "fid",
        "--latent-seed-offset", "2",
    ])
    assert args.latent_seed_offset == 2


def test_cli_rejects_negative_offset():
    args = _parse([
        "--model", "dit", "--task", "c2i", "--dataset", "imagenet",
        "--method", "baseline", "--metrics", "fid",
        "--latent-seed-offset", "-3",
    ])
    assert validate_args(args) is False


# ===========================================================================
# Persistence into results config
# ===========================================================================


def test_latent_seed_offset_persisted_in_dit_config(monkeypatch, tmp_path):
    """The offset must survive into results.json's config so downstream
    analysis can verify two runs really used different latent draws rather
    than trusting a directory name."""
    from types import SimpleNamespace

    import run_dit

    def _unexpected_covr_profiler(*args, **kwargs):
        raise AssertionError("disabled COVR path constructed a profiler")

    monkeypatch.setattr(
        run_dit, "_GenerationProfiler", _unexpected_covr_profiler)
    calls = []

    # Fake the ImageNet dataset so the test needs no real data on disk.
    # run_c2i imports it lazily via `from dataset.imagenet import ImageNetDataset`;
    # patch the module to intercept that import.
    import dataset.imagenet as _ds_mod

    class _FakeDS:
        def __init__(self, imagenet_dir=None, n_images=0, seed=0):
            self._n = n_images

        def __len__(self):
            return self._n

        def __getitem__(self, i):
            return (f"fake/{i}.png", f"class_{i}", i % 1000)

    monkeypatch.setattr(_ds_mod, "ImageNetDataset", _FakeDS)

    # Fake the generator so no model weights are loaded (no GPU here).
    # DiTGenerator.generate's first arg is `seed`; the fake records it.
    class _FakeGen:
        def __init__(self, *a, **kw):
            pass

        def load(self):
            return None

        def generate(self, prompts, seed, **kw):
            calls.append(seed)
            # A 1-image tensor so postprocessing can slice it; no GPU needed.
            import torch
            return (torch.zeros(1, 4, 4, 4),
                    torch.zeros(1, 3, 4, 4, dtype=torch.uint8))

        generate_ttt = generate

    # run_c2i prints the transformer structure after load; give it a stub.
    _FakeGen.transformer = SimpleNamespace(
        parameters=lambda: [],
        transformer_blocks=[],
    )
    _FakeGen.device = "cpu"

    monkeypatch.setattr(run_dit, "DiTGenerator", _FakeGen)

    def fake_profile(self, *a, **kw):
        return []

    monkeypatch.setattr(run_dit.FLOPsMetric, "profile", fake_profile)

    def fake_add_generation(self, teacache):
        return 0.0

    monkeypatch.setattr(run_dit.FLOPsMetric, "add_generation", fake_add_generation)

    # FID prep needs real images on disk; stub it and the computer.
    def fake_ensure_real_299(*a, **kw):
        return None

    monkeypatch.setattr(run_dit, "ensure_real_299", fake_ensure_real_299)

    def fake_fid_add(self, img, **kw):
        return None

    monkeypatch.setattr(run_dit.FIDISComputer, "add", fake_fid_add)

    args = _parse([
        "--model", "dit", "--task", "c2i", "--dataset", "imagenet",
        "--method", "teacache", "--metrics", "fid",
        "--n_prompts", "2", "--batch_size", "1",
        "--latent-seed-offset", "1",
        # Keep the run's artifacts inside pytest's tmp dir rather than the
        # repo's output/, which is a real results tree.
        "--output_dir", str(tmp_path / "run"),
    ])
    assert validate_args(args) is True

    def fake_compute(self, *a, **kw):
        return {"fid": 1.0}

    monkeypatch.setattr(run_dit.FIDISComputer, "compute", fake_compute)

    def fake_cleanup(self):
        return None

    monkeypatch.setattr(run_dit.FIDISComputer, "cleanup", fake_cleanup)

    results = run_dit.run_c2i(args)
    config = results["config"]
    assert config["latent_seed_offset"] == 1
    assert config["covr_shadow"] is False
    assert config["covr_template_bandit"] is False
    assert config["covr_force_template_id"] is None
    assert config["covr_session_id"] is None
    assert config["covr_version_key"] is None
    assert config["covr_profile_stages"] is False
    assert not any(
        key.startswith("covr_") for key in results["aggregate"])
    # The same (image, draw) cell seeds actually handed to the generator.
    assert calls[0] == [latent_seed_for_index(0, 1)]
    assert calls[1] == [latent_seed_for_index(1, 1)]
