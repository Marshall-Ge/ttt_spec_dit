import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from accelerators.covr_bandit import RefreshTemplate, TemplateManifest
from main import parse_args, validate_args
from models.dit import DiTTransformer2D
from run_dit import (
    _GenerationProfiler,
    _compute_generated_fid_is,
    _covr_canonical_json,
    _covr_full_rollout,
    _covr_online_accounting,
    _covr_resume_metadata,
    _covr_scheduler_alphas,
    _covr_scheduler_config_json,
    _covr_scheduler_pair,
    _covr_shadow_full,
    _dataset_generation_window,
    _load_forced_covr_template,
)
from utils import ensure_real_299


class StatefulScheduler:
    def __init__(self):
        self.counter = 7

    def step(self, model_output, timestep, sample, return_dict=False):
        self.counter += 1
        return (sample - model_output * self.counter,)


class RecordingTransformer:
    def __init__(self):
        self.calls = []

    def forward_with_cfg(self, hidden_states, timestep, **kwargs):
        self.calls.append(("cfg", kwargs))
        return hidden_states + 1

    def __call__(self, hidden_states, timestep, **kwargs):
        self.calls.append(("plain", kwargs))
        return (hidden_states + 1,)


def test_covr_version_json_is_canonical_across_container_order():
    left = {
        "_use_default_values": {"beta_end", "beta_start"},
        "nested": {"second": 2, "first": torch.tensor([1, 2])},
    }
    right = {
        "nested": {"first": torch.tensor([1, 2]), "second": 2},
        "_use_default_values": {"beta_start", "beta_end"},
    }

    assert _covr_canonical_json(left) == _covr_canonical_json(right)
    assert json.loads(_covr_canonical_json(left))["_use_default_values"] == [
        "beta_end", "beta_start"]


def test_scheduler_version_ignores_default_metadata_order():
    left = {
        "prediction_type": "epsilon",
        "timestep_spacing": "leading",
        "_use_default_values": ["steps_offset", "trained_betas"],
    }
    right = {
        "_use_default_values": ["trained_betas", "steps_offset"],
        "timestep_spacing": "leading",
        "prediction_type": "epsilon",
    }

    assert _covr_scheduler_config_json(left) == _covr_scheduler_config_json(right)


def test_scheduler_counterfactual_branches_share_pre_step_state():
    scheduler = StatefulScheduler()
    sample = torch.ones(1, 1)
    approx = torch.full((1, 1), 2.0)
    full = torch.full((1, 1), 3.0)

    x_approx, x_full = _covr_scheduler_pair(
        scheduler, approx, full, torch.tensor(10), sample)

    assert scheduler.counter == 7
    assert x_approx.item() == pytest.approx(1.0 - 2.0 * 8.0)
    assert x_full.item() == pytest.approx(1.0 - 3.0 * 8.0)
    scheduler.step(approx, torch.tensor(10), sample, return_dict=False)
    assert scheduler.counter == 8


def test_shadow_full_explicitly_bypasses_all_acceleration_state():
    transformer = RecordingTransformer()
    hidden = torch.zeros(2, 1)
    labels = torch.tensor([1, 1000])

    result = _covr_shadow_full(
        transformer, hidden, torch.tensor([10, 10]), labels, 4.5)

    assert torch.equal(result, hidden + 1)
    mode, kwargs = transformer.calls[-1]
    assert mode == "cfg"
    assert kwargs["current"] is None
    assert kwargs["cache_dic"] is None
    assert kwargs["teacache_state"] is None


def test_scheduler_alphas_use_actual_next_step_and_final_alpha():
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prediction_type="epsilon"),
        alphas_cumprod=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        final_alpha_cumprod=torch.tensor(0.9),
    )
    timesteps = torch.tensor([3, 1])
    assert _covr_scheduler_alphas(
        scheduler, timesteps, 0, torch.tensor(3)) == pytest.approx((0.4, 0.2))
    assert _covr_scheduler_alphas(
        scheduler, timesteps, 1, torch.tensor(1)) == pytest.approx((0.2, 0.9))
    scheduler.config.prediction_type = "v_prediction"
    with pytest.raises(ValueError, match="epsilon-prediction"):
        _covr_scheduler_alphas(scheduler, timesteps, 1, torch.tensor(1))


def test_cfg_wrapper_keeps_tensor_contract_and_records_disagreement():
    class FakeTransformer:
        config = SimpleNamespace(in_channels=1)
        last_cfg_disagreement = 0.0

        def forward(self, hidden_states, timestep, current, cache_dic,
                    teacache_state=None, class_labels=None, return_dict=False):
            output = torch.tensor([
                [[[3.0]], [[0.0]]],
                [[[4.0]], [[0.0]]],
                [[[1.0]], [[0.0]]],
                [[[2.0]], [[0.0]]],
            ])
            return (output,)

    transformer = FakeTransformer()
    hidden = torch.zeros(4, 1, 1, 1)
    output = DiTTransformer2D.forward_with_cfg(
        transformer,
        hidden,
        torch.zeros(4),
        current=None,
        cache_dic=None,
        class_labels=torch.tensor([1, 2, 1000, 1000]),
        cfg_scale=2.0,
        track_cfg_disagreement=True,
    )

    assert torch.is_tensor(output)
    assert output[:, 0, 0, 0].tolist() == pytest.approx([5.0, 6.0, 5.0, 6.0])
    assert transformer.last_cfg_disagreement == pytest.approx(
        (torch.tensor([2.0, 2.0]).norm() / torch.tensor([1.0, 2.0]).norm()).item())


def test_full_rollout_uses_isolated_scheduler_for_exact_horizon():
    class RolloutScheduler:
        def __init__(self):
            self.calls = 0

        def scale_model_input(self, sample, timestep):
            return sample

        def step(self, model_output, timestep, sample, return_dict=False):
            self.calls += 1
            return (sample - model_output,)

    class ConstantTransformer:
        def __init__(self):
            self.calls = 0

        def __call__(self, hidden_states, timestep, **kwargs):
            self.calls += 1
            return (torch.ones_like(hidden_states),)

    scheduler = RolloutScheduler()
    transformer = ConstantTransformer()
    result = _covr_full_rollout(
        transformer, scheduler, torch.tensor([3, 2, 1]),
        start_idx=0, horizon=2, latents=torch.ones(1, 1),
        class_labels=torch.tensor([1]), guidance_scale=1.0,
        in_channels=1,
    )

    assert result.item() == pytest.approx(-1.0)
    assert transformer.calls == 2
    assert scheduler.calls == 0


def test_resume_metadata_requires_per_assignment_sample_counts(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "session_id": "session",
        "run_identity": {"dataset": "imagenet"},
        "assignments": [{"sample_count": 2}, {"sample_count": 1}],
    }), encoding="utf-8")
    assert _covr_resume_metadata(str(path)) == {
        "session_id": "session",
        "processed_samples": 3,
        "run_identity": {"dataset": "imagenet"},
    }

    path.write_text(json.dumps({
        "session_id": "session",
        "assignments": [{"trajectory_id": 0}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="predates sample-count tracking"):
        _covr_resume_metadata(str(path))


def test_generated_fid_is_swaps_inputs_and_restores_them():
    class Metric:
        real_dir = "real"
        gen_dir = "generated"

        def compute(self):
            assert self.real_dir == "generated"
            assert self.gen_dir == "real"
            return {"fid": 1.0, "is_mean": 2.0, "is_std": 3.0}

    metric = Metric()
    assert _compute_generated_fid_is(metric)["is_mean"] == 2.0
    assert (metric.real_dir, metric.gen_dir) == ("real", "generated")


def test_generated_fid_is_restores_inputs_after_error():
    class Metric:
        real_dir = "real"
        gen_dir = "generated"

        def compute(self):
            raise RuntimeError("metric failed")

    metric = Metric()
    with pytest.raises(RuntimeError, match="metric failed"):
        _compute_generated_fid_is(metric)
    assert (metric.real_dir, metric.gen_dir) == ("real", "generated")


def test_dataset_window_composes_base_offset_and_resume():
    assert _dataset_generation_window(0, 10, 0, 10) == (0, 10)
    assert _dataset_generation_window(2048, 2048, 128, 4096) == (2176, 1920)
    with pytest.raises(ValueError, match="consumed"):
        _dataset_generation_window(2048, 2048, 2048, 4096)


def test_real_299_subset_uses_absolute_dataset_indices(tmp_path):
    val_dir = tmp_path / "imagenet" / "val"
    val_dir.mkdir(parents=True)
    items = []
    for index in range(4):
        path = val_dir / f"source_{index}.png"
        Image.new("RGB", (8, 8), color=(index, index, index)).save(path)
        items.append((str(path), f"a photo of a class {index}", index))

    class Dataset:
        def __init__(self):
            self.val_dir = str(val_dir)
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    subset = ensure_real_299(Dataset(), str(tmp_path / "run"), 2, start_index=2)
    assert sorted(path.name for path in Path(subset).iterdir()) == [
        "000002_class_2.png",
        "000003_class_3.png",
    ]


def test_online_accounting_separates_candidate_and_safety_cost():
    accounting = _covr_online_accounting(
        [5.0, 7.0], [1.0, 2.0], n_images=8, safety_full_steps=6,
        candidate_flops_T=3.0, vanilla_flops_T=12.0,
        full_step_flops=0.5e12,
    )
    assert accounting["wall_s_candidate_total"] == pytest.approx(9.0)
    assert accounting["wall_s_safety_total"] == pytest.approx(3.0)
    assert accounting["speed_candidate_img_per_s"] == pytest.approx(8.0 / 9.0)
    assert accounting["safety_full_steps_mean_per_trajectory"] == 3.0
    assert accounting["flops_safety_T"] == pytest.approx(1.5)
    assert accounting["flops_online_T"] == pytest.approx(4.5)
    assert accounting["flops_reduction_online"] == pytest.approx(0.625)


def test_online_accounting_includes_terminal_and_control_cost():
    accounting = _covr_online_accounting(
        [6.0, 8.0], [1.0, 2.0], n_images=8, safety_full_steps=6,
        candidate_flops_T=3.0, vanilla_flops_T=12.0,
        full_step_flops=0.5e12,
        terminal_wall_times=[0.5, 0.5], terminal_full_steps=2,
        control_wall_times=[0.25, 0.25],
    )
    assert accounting["wall_s_candidate_total"] == pytest.approx(9.5)
    assert accounting["wall_s_terminal_total"] == pytest.approx(1.0)
    assert accounting["wall_s_control_total"] == pytest.approx(0.5)
    assert accounting["speed_online_img_per_s"] == pytest.approx(8.0 / 14.0)
    assert accounting["flops_terminal_T"] == pytest.approx(0.5)
    assert accounting["flops_online_T"] == pytest.approx(5.0)


def test_generation_profiler_cpu_collects_nested_stages_without_sync():
    profiler = _GenerationProfiler("cpu", detailed=True)
    generation = profiler.start_gpu("generation_online", required=True)
    safety = profiler.start_gpu("safety_shadow_full", required=True)
    profiler.stop_gpu(safety)
    profiler.stop_gpu(generation)

    assert profiler.synchronize() == 0.0
    summary = profiler.summary()
    assert summary["generation_online"] >= summary["safety_shadow_full"]
    assert "cuda_sync_calls" not in summary


def test_forced_template_loader_validates_version_and_id(tmp_path):
    manifest = TemplateManifest(
        version_key="version-a",
        num_steps=4,
        num_layers=2,
        mandatory_prefix=1,
        max_taylor_gap=2,
        baseline_template_id="prior",
        templates=(
            RefreshTemplate("prior", (True, False, True, False), 4),
            RefreshTemplate("alternate", (True, True, False, False), 4),
        ),
    )
    path = tmp_path / "manifest.json"
    manifest.save(str(path))

    loaded, template = _load_forced_covr_template(
        str(path), "alternate", "version-a")
    assert loaded.manifest_hash == manifest.manifest_hash
    assert template.refresh_mask == (True, True, False, False)

    with pytest.raises(ValueError, match="runtime"):
        _load_forced_covr_template(str(path), "alternate", "version-b")
    with pytest.raises(ValueError, match="not found"):
        _load_forced_covr_template(str(path), "missing", "version-a")


def _parse_main_args(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "--model", "dit",
        "--task", "c2i",
        "--dataset", "imagenet",
        "--method", "speca",
        *extra,
    ])
    return parse_args()


def test_cli_rejects_negative_dataset_start_index(monkeypatch):
    args = _parse_main_args(monkeypatch, "--dataset-start-index", "-1")
    assert validate_args(args) is False


def test_cli_accepts_direct_forced_template(monkeypatch):
    args = _parse_main_args(
        monkeypatch,
        "--dataset-start-index", "2048",
        "--covr-template-manifest", "manifest.json",
        "--covr-force-template-id", "template_01",
    )
    assert validate_args(args) is True


def test_cli_rejects_forced_template_with_bandit(monkeypatch):
    args = _parse_main_args(
        monkeypatch,
        "--covr-template-manifest", "manifest.json",
        "--covr-force-template-id", "template_01",
        "--covr-template-bandit",
    )
    assert validate_args(args) is False
