"""The out-of-sample gate in ``scripts/analyze_crossover.py``.

The gate decides whether COVR gets GPU budget, so its two failure modes both
need pinning down: it must not declare a crossover where none exists, and it
must not throw away one that does. The rule it uses now (per-arm cost
regression) replaced a 1NN label-transfer rule that failed the second way —
significant p, negative recovered share — on worlds that genuinely cross.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import analyze_crossover as ac  # noqa: E402

_ARM_LEVELS = np.array([2.072, 2.912, 3.148, 3.415])  # observed k6 mean_conf
_TILT = np.array([-1.0, -0.3, 0.3, 1.0])


def _world(rng, n, interaction, noise=0.5, informative_features=True):
    """Additive difficulty + arm offsets, optionally with a real interaction.

    ``interaction=0`` means NO per-image crossover exists: the best arm is the
    same arm for every image, so any gate pass is a false positive.
    """
    difficulty = rng.randn(n) * 0.6
    D = (_ARM_LEVELS[None, :] + difficulty[:, None]
         + rng.randn(n, 4) * noise
         + interaction * difficulty[:, None] * _TILT[None, :])
    if informative_features:
        feat = np.stack([difficulty + rng.randn(n) * 0.3,
                         difficulty * 0.7 + rng.randn(n) * 0.3,
                         rng.randn(n) * 0.3], axis=1)
    else:
        feat = rng.randn(n, 3)
    return D, feat


def _run(rng, n=400, **kw):
    D, feat = _world(rng, n, **kw)
    return ac._out_of_sample(D, list(range(n)), feat, 200, rng)


def test_cost_regression_finds_a_real_crossover_that_label_transfer_misses():
    """The reason the gate moved off 1NN label transfer.

    Same D, same features, same split: the cost regression recovers a positive
    share at p<=0.05 while transferring the neighbour's argmin label recovers a
    NEGATIVE share. Both p-values can be small — the label rule's p says the
    features predict the winner, its frac says acting on that prediction by
    committing to one arm costs more than the whole oracle gain.
    """
    out = _run(np.random.RandomState(3), interaction=0.8)
    assert out["skip"] is None
    assert out["reg_frac"] > 0.0
    assert out["reg_p"] <= 0.05
    assert out["nn_frac"] < 0.0        # the former gate would say "no crossover"
    assert out["reg_frac"] > out["nn_frac"]


def test_no_crossover_does_not_pass_the_gate():
    """interaction=0: one arm is best for every image, so frac must not be >0
    at p<=0.05. Checked over several seeds because a single draw says little
    about a false-positive rate."""
    for seed in range(6):
        out = _run(np.random.RandomState(seed), interaction=0.0)
        assert out["skip"] is None
        assert not (out["reg_frac"] > 0.0 and out["reg_p"] <= 0.05), seed


def test_uninformative_features_degenerate_to_the_best_single_arm():
    """With features unrelated to anything, every arm's fit is its own mean, so
    the argmin is the globally best arm and the recovered share is ~0 — not the
    large negative number label transfer produces. This is what keeps the gate
    from being noisy in the harmless direction."""
    out = _run(np.random.RandomState(11), interaction=0.0,
               informative_features=False)
    assert out["reg_frac"] == pytest.approx(0.0, abs=1e-9)
    assert out["nn_frac"] < out["reg_frac"]


def test_cost_regression_never_beats_the_oracle_it_is_measured_against():
    """A share above 1.0 would mean the transfer rule beat per-image omniscience
    on the same half, which is impossible; catching it here guards against the
    train/test halves being crossed by a future edit."""
    for seed in (0, 5, 9):
        out = _run(np.random.RandomState(seed), interaction=0.8)
        assert out["reg_frac"] <= 1.0 + 1e-9
        assert out["const_frac"] <= 1e-9  # feature-free ceiling


def test_features_with_zero_variance_are_reported_not_crashed():
    n = 40
    rng = np.random.RandomState(0)
    D = _ARM_LEVELS[None, :] + rng.randn(n, 4) * 0.5
    out = ac._out_of_sample(D, list(range(n)), np.ones((n, 3)), 10, rng)
    assert out["skip"] and "degenerate" in out["skip"]


def test_zero_oracle_benefit_is_flagged_rather_than_dividing_by_zero():
    """One arm strictly dominating every image leaves nothing to transfer; frac
    must be nan instead of raising or reporting a bogus percentage."""
    n = 40
    D = np.tile(np.array([0.1, 1.0, 2.0, 3.0]), (n, 1))
    rng = np.random.RandomState(0)
    feat = rng.randn(n, 3)
    out = ac._out_of_sample(D, list(range(n)), feat, 10, rng)
    assert out["oracle_benefit_test"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(out["reg_frac"]) and np.isnan(out["nn_frac"])


def test_renderer_labels_the_gate_and_keeps_the_diagnostic_visible(
        tmp_path, monkeypatch):
    """A reader must not mistake the diagnostic for the gate: the k6 real run
    printed p=0.009 with a negative share, and calling that "pure noise" is the
    wrong conclusion.

    The row is produced by ``_analyze_budget`` on a synthetic probe rather than
    hand-written, so the test keeps working when the row schema changes; only
    the out-of-sample result is substituted.
    """
    import json
    from PIL import Image

    rng = np.random.RandomState(3)
    n = 12
    base = (rng.rand(n, 16, 16, 3) * 255).astype(np.uint8)

    def _write(directory, images, fid, is_mean):
        gen = os.path.join(directory, "generated")
        os.makedirs(gen, exist_ok=True)
        for i, arr in enumerate(images):
            Image.fromarray(arr).save(os.path.join(gen, f"{i:06d}_cls.png"))
        with open(os.path.join(directory, "results.json"), "w") as fh:
            json.dump({"config": {"seed": 42, "num_steps": 50,
                                  "n_prompts": len(images)},
                       "aggregate": {"fid": fid, "is_mean": is_mean}}, fh)

    budget = tmp_path / "k6"
    budget.mkdir()
    (budget / "manifest.json").write_text(json.dumps({"num_steps": 50}),
                                          encoding="utf-8")
    for name, scale, fid, ism in [("back_loaded", 12, 170.0, 8.0),
                                  ("uniform", 3, 130.0, 9.0)]:
        noisy = np.clip(base.astype(np.int16) + rng.randn(n, 16, 16, 3) * scale,
                        0, 255).astype(np.uint8)
        _write(str(budget / "equalflops" / f"arm_{name}"), list(noisy), fid, ism)

    crafted = _run(np.random.RandomState(3), interaction=0.8)
    monkeypatch.setattr(ac, "_out_of_sample",
                        lambda *a, **k: dict(crafted))

    class _A:
        perm_seed = 0
        max_perm = 10
        oos_perm = 10

    row = ac._analyze_budget(str(budget), {i: base[i] for i in range(n)},
                             "pixel_mse", "cpu", _A())
    text = "\n".join(ac._render_budget(row, 5.0))
    assert "COST REGRESSION (the gate)" in text
    assert "[diagnostic, NOT the gate] 1NN label transfer" in text
    assert "VERDICT GATE: crossover UTILIZABLE" in text
