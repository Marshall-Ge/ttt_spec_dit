"""Per-image metrics used by the COVR crossover verdict.

These guard the two properties that make the ``inception`` / ``inception_conf``
options worth trusting as evidence rather than as another number: the entropy is
the real predictive entropy (so "smaller = better" holds), and the
reference-free metric genuinely never reads the reference (so its verdict cannot
be blamed on the reference not being the FID optimum).
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import analyze_crossover as ac  # noqa: E402


class _StubExtractor(torch.nn.Module):
    """Mirrors the real extractor's input contract: uint8 4-D BxCxHxW."""

    def __init__(self, feature):
        super().__init__()
        self.feature = feature
        self.seen_dtypes = []

    def forward(self, x):
        assert torch.is_tensor(x)
        self.seen_dtypes.append(x.dtype)
        assert x.dtype == torch.uint8, f"extractor needs uint8, got {x.dtype}"
        assert x.dim() == 4 and x.shape[1] == 3, x.shape
        dim = 2048 if self.feature == "2048" else 1008
        flat = x.float().flatten(1).mean(dim=1, keepdim=True)
        return (flat.repeat(1, dim) + torch.arange(dim).float()[None] * 1e-3,)


@pytest.fixture
def stub_inception(monkeypatch):
    made = {}

    def _factory(feature, device):
        made.setdefault(feature, _StubExtractor(feature))
        return made[feature]

    monkeypatch.setattr(ac, "_inception_extractor", _factory)
    ac._INCEPTION_CACHE.clear()
    return made


def test_entropy_matches_closed_form_at_both_extremes():
    uniform = np.zeros((1, 1008))
    one_hot = np.zeros((1, 1008))
    one_hot[0, 3] = 60.0
    ent = ac._entropy_from_logits(np.concatenate([uniform, one_hot]))
    assert ent[0] == pytest.approx(np.log(1008.0))
    assert ent[1] == pytest.approx(0.0, abs=1e-9)
    # A large logit shift must not change entropy (softmax is shift-invariant);
    # the naive exp() without the max-subtraction overflows here instead.
    shifted = ac._entropy_from_logits(uniform + 1e4)
    assert shifted[0] == pytest.approx(np.log(1008.0))


def test_inception_distance_is_zero_for_identical_images(stub_inception):
    rng = np.random.RandomState(0)
    ref = (rng.rand(4, 8, 8, 3) * 255).astype(np.uint8)
    D = ac._distance_matrix("inception", "cpu", ref, [ref, ref.copy()])
    assert D.shape == (4, 2)
    assert np.allclose(D, 0.0)
    # uint8 in, never pre-normalized to [0,1] — the real extractor asserts this.
    assert set(stub_inception["2048"].seen_dtypes) == {torch.uint8}


def test_inception_conf_never_reads_the_reference(stub_inception):
    rng = np.random.RandomState(1)
    ref = (rng.rand(5, 8, 8, 3) * 255).astype(np.uint8)
    arms = [(rng.rand(5, 8, 8, 3) * 255).astype(np.uint8) for _ in range(3)]
    first = ac._distance_matrix("inception_conf", "cpu", ref, arms)
    second = ac._distance_matrix("inception_conf", "cpu", ref // 3, arms)
    assert np.allclose(first, second)
    assert first.shape == (5, 3)
    assert np.isfinite(first).all()


def test_unknown_batched_metric_is_rejected(stub_inception):
    ref = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="not a batched metric"):
        ac._distance_matrix("pixel_mse", "cpu", ref, [ref])


def test_resolve_metric_falls_back_and_reports_when_inception_is_missing(
        monkeypatch, capsys):
    monkeypatch.setattr(ac, "_inception_usable",
                        lambda device: (False, "ModuleNotFoundError: nope"))
    monkeypatch.setattr(ac, "_lpips_usable", lambda device: False)

    name, desc = ac._resolve_metric("inception", "cpu")
    assert name == "pixel_mse"
    assert "pixel_mse" in desc
    # The fallback must say that it weakens the verdict, since pixel MSE is not
    # in FID's feature space and its [a] failure closes nothing.
    assert "does not close the reward family" in capsys.readouterr().out

    assert ac._resolve_metric("auto", "cpu")[0] == "pixel_mse"
    monkeypatch.setattr(ac, "_inception_usable", lambda device: (True, ""))
    assert ac._resolve_metric("auto", "cpu")[0] == "inception"


def _write_run(directory, images, fid, is_mean):
    import json
    from PIL import Image
    gen = os.path.join(directory, "generated")
    os.makedirs(gen, exist_ok=True)
    for index, arr in enumerate(images):
        Image.fromarray(arr).save(os.path.join(gen, f"{index:06d}_cls.png"))
    with open(os.path.join(directory, "results.json"), "w") as handle:
        json.dump({"config": {"seed": 42, "num_steps": 50,
                              "n_prompts": len(images)},
                   "aggregate": {"fid": fid, "is_mean": is_mean}}, handle)


def _probe(tmp_path, n=12):
    """Reference + one budget with three arms whose noise scale tracks FID."""
    import json
    rng = np.random.RandomState(3)
    base = (rng.rand(n, 16, 16, 3) * 255).astype(np.uint8)
    _write_run(str(tmp_path / "reference"), list(base), 128.0, 9.0)
    budget = tmp_path / "k6"
    budget.mkdir()
    (budget / "manifest.json").write_text(json.dumps({"num_steps": 50}),
                                          encoding="utf-8")
    for name, scale, fid, is_mean in [("back_loaded", 12, 170.0, 8.0),
                                      ("front_loaded", 6, 150.0, 10.0),
                                      ("uniform", 3, 130.0, 9.0)]:
        noisy = np.clip(base.astype(np.int16)
                        + rng.randn(n, 16, 16, 3) * scale,
                        0, 255).astype(np.uint8)
        _write_run(str(budget / "equalflops" / f"arm_{name}"), list(noisy),
                   fid, is_mean)
    return str(budget), {i: base[i] for i in range(n)}


class _Args:
    perm_seed = 0
    max_perm = 20
    oos_perm = 20


def test_analyze_budget_runs_the_inception_path_end_to_end(
        tmp_path, stub_inception):
    budget_dir, ref_imgs = _probe(tmp_path)
    row = ac._analyze_budget(budget_dir, ref_imgs, "inception", "cpu", _Args())

    assert row["skip"] is None
    assert row["metric"] == "inception"
    assert row["reference_free"] is False
    # [b]'s floor stays in pixel-MSE units even though [a]/[c] are not, so its
    # ratio must not be contaminated by the Inception scale.
    assert row["mean_spread_pmse"] > 0.0
    assert row["floor_ratio"] == pytest.approx(
        row["mean_spread_pmse"] / ac._FLOOR)
    assert row["is_means"]["uniform"] == 9.0
    rendered = "\n".join(ac._render_budget(row, 5.0))
    assert "deliberately in pixel-MSE units" in rendered
    assert "mean_dist" in rendered


def test_reference_free_row_reports_both_correlations(tmp_path,
                                                      stub_inception):
    budget_dir, ref_imgs = _probe(tmp_path)
    row = ac._analyze_budget(budget_dir, ref_imgs, "inception_conf", "cpu",
                             _Args())

    assert row["reference_free"] is True
    assert row["spearman_is"] is not None
    rendered = "\n".join(ac._render_budget(row, 5.0))
    assert "mean_conf" in rendered
    assert "Spearman(mean_conf, -IS)" in rendered
    # The prototypicality bias is a property of the metric, so it must be
    # printed whether or not the gate passes.
    assert "KNOWN BIAS" in rendered


def test_missing_arm_fid_skips_instead_of_crashing_the_renderer(
        tmp_path, stub_inception):
    """A NaN/absent arm FID must not reach the [a] table.

    The table calls float() on every arm FID and reads row["spearman"], so a
    missing one used to raise TypeError from inside the renderer and take down
    the whole analysis after the GPU work was already done.
    """
    import json
    budget_dir, ref_imgs = _probe(tmp_path)
    broken = os.path.join(budget_dir, "equalflops", "arm_uniform",
                          "results.json")
    payload = json.loads(open(broken, encoding="utf-8").read())
    payload["aggregate"]["fid"] = None
    with open(broken, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    row = ac._analyze_budget(budget_dir, ref_imgs, "inception", "cpu", _Args())
    assert row["valid"] is None
    assert row["skip"] == "some arm results.json missing FID"
    rendered = "\n".join(ac._render_budget(row, 5.0))
    assert "SKIPPED: some arm results.json missing FID" in rendered
