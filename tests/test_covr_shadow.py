from types import SimpleNamespace

import pytest
import torch

from models.dit import DiTTransformer2D
from run_dit import (
    _covr_scheduler_alphas,
    _covr_scheduler_pair,
    _covr_shadow_full,
)


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
