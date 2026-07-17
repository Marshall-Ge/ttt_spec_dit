# -*- coding: utf-8 -*-
"""CPU tests for verification-guided selective recompute."""

import pytest
import torch

from accelerators.compute_controller import (
    ComputeAction,
    ComputeOpportunity,
    ProbeCorrectController,
    VerificationResult,
)
from accelerators.speca import speca_cal_type, speca_init
from eval.latency import FLOPsMetric


def _opportunity(trajectory_id=0, verification_requested=True):
    return ComputeOpportunity(
        model="dit",
        method="speca",
        trajectory_id=trajectory_id,
        step_idx=4,
        num_steps=6,
        layer_idx=0,
        timestep_bucket=2,
        approximation_distance=-2,
        verification_requested=verification_requested,
    )


def test_probe_correct_controller_reject_policy():
    controller = ProbeCorrectController("reject")
    controller.begin_trajectory(0)

    opportunity = _opportunity()
    assert controller.decide(opportunity) == ComputeAction.VERIFY
    assert not controller.observe(
        opportunity,
        VerificationResult(error_value=0.01, threshold=0.02, accepted=True),
    )
    assert controller.observe(
        opportunity,
        VerificationResult(error_value=0.03, threshold=0.02, accepted=False),
    )

    controller.end_trajectory()
    stats = controller.stats()
    assert stats["verifications"] == 2
    assert stats["corrected_verifications"] == 1
    assert stats["rejected_verifications"] == 1


def test_controller_requires_matching_trajectory():
    controller = ProbeCorrectController("always")
    with pytest.raises(RuntimeError):
        controller.decide(_opportunity())

    controller.begin_trajectory(1)
    with pytest.raises(ValueError):
        controller.decide(_opportunity(trajectory_id=0))
    controller.end_trajectory()


def test_speca_counts_every_step_once():
    cache, current = speca_init(
        num_steps=6,
        base_threshold=0.01,
        decay_rate=0.01,
        min_taylor_steps=1,
        max_taylor_steps=4,
        num_layers=1,
        check_layer=0,
    )

    for step in range(5, -1, -1):
        current.step = step
        current.last_layer_error = 0.0
        speca_cal_type(cache, current)

    assert cache.full_count + cache.taylor_count == 6
    assert cache.full_count > 0
    assert cache.taylor_count > 0


def test_speca_flops_include_probe_blocks_once():
    metric = FLOPsMetric.__new__(FLOPsMetric)
    metric._profiled = True
    metric._flops_full = 100.0
    metric._flops_skip = 20.0
    metric._total_vanilla = 0.0
    metric._total_accel = 0.0
    metric._n = 0

    metric.add_speca_generation(
        full_steps=2,
        taylor_steps=3,
        probe_full_blocks=4,
        num_layers=4,
    )

    assert metric._total_vanilla == 500.0
    assert metric._total_accel == 340.0
    assert metric._n == 1


def _run_tiny_speca(model, controller, monkeypatch):
    import models.dit as dit_module

    events = []

    def record_event(**kwargs):
        events.append({
            "predicted_hidden": kwargs["predicted_hidden"].detach().clone(),
            "full_hidden": kwargs["full_hidden"].detach().clone(),
        })

    monkeypatch.setattr(dit_module, "_vfl_record_speca_event", record_event)
    cache, current = speca_init(
        num_steps=6,
        base_threshold=0.01,
        decay_rate=0.01,
        min_taylor_steps=1,
        max_taylor_steps=4,
        num_layers=1,
        check_layer=0,
        controller=controller,
        trajectory_id=0,
    )
    if controller is not None:
        controller.begin_trajectory(0)

    hidden = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(7))
    labels = torch.tensor([0])
    with torch.no_grad():
        for step_idx, timestep in enumerate([900, 700, 500, 300, 100, 0]):
            current.step = 5 - step_idx
            model(
                hidden,
                timestep=torch.tensor([timestep]),
                current=current,
                cache_dic=cache,
                class_labels=labels,
                return_dict=False,
            )

    if controller is not None:
        controller.end_trajectory()
    return cache, events


def test_probe_correction_preserves_pre_correction_event(monkeypatch):
    from models.dit import DiTTransformer2D

    torch.manual_seed(11)
    model = DiTTransformer2D(
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=8,
        num_layers=1,
        sample_size=4,
        patch_size=2,
        num_embeds_ada_norm=10,
    ).eval()

    baseline_cache, baseline_events = _run_tiny_speca(
        model, controller=None, monkeypatch=monkeypatch)
    controller = ProbeCorrectController("always")
    corrected_cache, corrected_events = _run_tiny_speca(
        model, controller=controller, monkeypatch=monkeypatch)

    assert baseline_events
    assert corrected_events
    assert torch.allclose(
        baseline_events[0]["predicted_hidden"],
        corrected_events[0]["predicted_hidden"],
    )
    assert torch.allclose(
        baseline_events[0]["full_hidden"],
        corrected_events[0]["full_hidden"],
    )
    assert baseline_cache.corrected_probe_blocks == 0
    assert corrected_cache.corrected_probe_blocks == len(corrected_events)
    assert corrected_cache.probe_full_blocks == len(corrected_events)
