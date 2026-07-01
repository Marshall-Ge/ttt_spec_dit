# -*- coding: utf-8 -*-
"""End-to-end VFL integration test.

Verifies:
  1. calibrator=None → inference path unchanged (no crashes, same structure)
  2. calibrator + buffer enabled → events collected, calibrator updated
  3. No import errors, no signature mismatches

Run:
    python verification_feedback_loop/tests/test_integration_e2e.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np

# ===========================================================================
# Test 1: Synthetic event flow (no model needed)
# ===========================================================================


def test_synthetic_event_flow():
    """Simulate the VFL pipeline with synthetic events."""
    print("=" * 60)
    print("Test 1: Synthetic Event Flow")
    print("=" * 60)

    from verification_feedback_loop import (
        VerificationEvent, record_event, make_timestep_bucket,
        StratifiedReplayBuffer, OnlineCalibrator,
    )
    from accelerators.teacache import teacache_init, teacache_decide
    from accelerators.speca import speca_init, speca_cal_type

    # Setup
    buf = StratifiedReplayBuffer(capacity_per_stratum=100)
    cal = OnlineCalibrator(ema_window=50, forget_factor=0.95)
    num_steps = 50

    # ---- TeaCache path ----
    tc_state = teacache_init(num_steps=50, rel_l1_thresh=0.25)
    mod_inputs = [torch.randn(1, 256, 1152) for _ in range(num_steps)]

    decisions_no_cal = []
    decisions_with_cal = []

    for step_idx in range(num_steps):
        mod = mod_inputs[step_idx]

        # Without calibrator
        should1, raw1 = teacache_decide(tc_state, mod, calibrator=None)
        decisions_no_cal.append(should1)

    # Reset and rerun with calibrator
    tc_state2 = teacache_init(num_steps=50, rel_l1_thresh=0.25)
    for step_idx in range(num_steps):
        mod = mod_inputs[step_idx]

        # With calibrator (but not yet trained → uses defaults)
        should2, raw2 = teacache_decide(tc_state2, mod, calibrator=cal, probe_layer=20)
        decisions_with_cal.append(should2)

        # Simulate a VFL probe event (as if a calc step happened)
        if should2:
            event = VerificationEvent(
                layer_id=20, timestep=step_idx,
                timestep_bucket=make_timestep_bucket(step_idx, num_steps),
                predicted_feature=torch.randn(1, 256, 1152),
                true_feature=torch.randn(1, 256, 1152),
                error_value=abs(raw2) * 0.5,  # proxy
                decision="reject", model="dit",
                base_model_version="v1.0", step_idx=step_idx,
            )
            record_event(event, buffer=buf)
            if raw2 > 0:
                cal.update(event)
            else:
                cal.update(event)

    # Verify: decisions should be identical when calibrator is cold (uses defaults)
    assert decisions_no_cal == decisions_with_cal, \
        "Cold calibrator changed TeaCache decisions!"
    print(f"  TeaCache: {len(decisions_no_cal)} steps, "
          f"decisions identical with cold calibrator: PASS")

    # Verify: buffer received events
    assert buf.total_samples > 0, "Buffer received no events!"
    print(f"  Buffer: {buf.total_samples} events collected: PASS")

    # Verify: calibrator received updates
    assert cal.total_updates > 0, "Calibrator received no updates!"
    print(f"  Calibrator: {cal.total_updates} updates: PASS")

    # Verify: calibrator stats are accessible
    stats = cal.get_stats(20, 1)
    assert "threshold" in stats
    print(f"  Calibrator stats: layer=20, bucket=1, "
          f"threshold={stats['threshold']:.4f}: PASS")

    # ---- SpecA path ----
    cache_dic, current = speca_init(
        num_steps=50, base_threshold=0.01, decay_rate=0.01,
        min_taylor_steps=1, max_taylor_steps=4, max_order=4,
        num_layers=28, error_metric="cosine_similarity", check_layer=20,
    )

    # Run a few steps through SpecA decision
    types_no_cal = []
    types_with_cal = []

    for step in [49, 48, 47, 46, 45]:  # reverse denoising order
        current.step = step

        # Without calibrator
        speca_cal_type(cache_dic, current, calibrator=None)
        types_no_cal.append(current.type)

        # Simulate error if full step happened
        if current.type == "full":
            current.last_layer_error = 0.005
            # Simulate a VFL event
            event = VerificationEvent(
                layer_id=20, timestep=step,
                timestep_bucket=make_timestep_bucket(49 - step, 50),
                predicted_feature=torch.randn(1, 256, 1152),
                true_feature=torch.randn(1, 256, 1152),
                error_value=0.005, decision="reject",
                model="dit", base_model_version="v1.0", step_idx=step,
            )
            record_event(event, buffer=buf)
            cal.update(event)

    # Reset and rerun with calibrator
    cache_dic2, current2 = speca_init(
        num_steps=50, base_threshold=0.01, decay_rate=0.01,
        min_taylor_steps=1, max_taylor_steps=4, max_order=4,
        num_layers=28, error_metric="cosine_similarity", check_layer=20,
    )
    for step in [49, 48, 47, 46, 45]:
        current2.step = step
        speca_cal_type(cache_dic2, current2, calibrator=cal)
        types_with_cal.append(current2.type)
        if current2.type == "full":
            current2.last_layer_error = 0.005

    # SpecA decisions may differ with warm calibrator (expected behavior)
    print(f"  SpecA: types_no_cal={types_no_cal}, types_with_cal={types_with_cal}")
    print(f"  SpecA with calibrator: PASS (no crash)")

    print()
    return True


# ===========================================================================
# Test 2: Real DiT generation (model needed)
# ===========================================================================


def test_real_dit_generation():
    """Run real DiT generation with VFL enabled to verify no crashes."""
    print("=" * 60)
    print("Test 2: Real DiT Generation with VFL")
    print("=" * 60)

    from config import DIT_REPO
    from models.dit import (
        DiTTransformer2D, set_vfl_buffer, set_vfl_calibrator,
        get_vfl_buffer, set_vfl_step_info,
    )
    from verification_feedback_loop import StratifiedReplayBuffer, OnlineCalibrator

    # Check model availability
    model_path = os.path.join(DIT_REPO, "transformer", "diffusion_pytorch_model.bin")
    if not os.path.exists(model_path):
        print("  [SKIP] DiT model not found at", model_path)
        return True  # not a failure

    print("  Loading DiT-2-256 (this may take ~30s)...")
    t0 = time.time()

    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        transformer = DiTTransformer2D.from_pretrained(DIT_REPO)
        transformer.to(device=device, dtype=dtype)
        transformer.eval()
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        # Setup VFL
        buf = StratifiedReplayBuffer(capacity_per_stratum=100)
        cal = OnlineCalibrator(ema_window=50, forget_factor=0.95)
        set_vfl_buffer(buf, model_version="dit-v1.0")
        set_vfl_calibrator(cal)
        assert get_vfl_buffer() is buf
        print("  VFL buffer + calibrator registered")

        # Run synthetic forward (no real denoising — just the transformer forward)
        num_steps = 50
        batch_size = 1
        latent_shape = (batch_size, 4, 32, 32)

        with torch.no_grad():
            x = torch.randn(latent_shape, device=device, dtype=dtype)
            t = torch.tensor([981], device=device, dtype=torch.long)
            class_labels = torch.tensor([207], device=device, dtype=torch.long)

            # --- 1. Vanilla forward (no VFL state at all) ---
            set_vfl_step_info(0, num_steps)
            out1 = transformer(x, t, class_labels=class_labels, return_dict=False)[0]
            print(f"  Vanilla forward: output shape={out1.shape}")

            # --- 2. SpecA forward (speca state, no VFL collector) ---
            from accelerators.speca import speca_init
            cache_dic, current = speca_init(
                num_steps=num_steps, base_threshold=0.01, decay_rate=0.01,
                min_taylor_steps=1, max_taylor_steps=4, max_order=4,
                num_layers=28, error_metric="cosine_similarity", check_layer=20,
            )
            current.step = num_steps - 1  # first step
            set_vfl_step_info(0, num_steps)
            out2 = transformer(x, t, current=current, cache_dic=cache_dic,
                              class_labels=class_labels, return_dict=False)[0]
            print(f"  SpecA forward: output shape={out2.shape}, step_type={current.type}")

            # --- 3. TeaCache forward with VFL ---
            from accelerators.teacache import teacache_init
            tc_state = teacache_init(num_steps=num_steps, rel_l1_thresh=0.25)
            set_vfl_step_info(0, num_steps)
            out3 = transformer(x, t, teacache_state=tc_state,
                              class_labels=class_labels, return_dict=False)[0]
            print(f"  TeaCache forward: output shape={out3.shape}")

            # --- 4. TeaCache forward with calibrator passed to decide ---
            # (calibrator flow tested in synthetic test above, just verify no crash)

        # Check VFL data flow
        buf_stats = buf.stats()
        cal_updates = cal.total_updates
        print(f"\n  VFL after 3 forwards:")
        print(f"    Buffer: {buf_stats['total_samples']} events, "
              f"{buf_stats['num_strata_nonempty']} nonempty strata")
        print(f"    Calibrator: {cal_updates} total updates")

        # Clean up VFL state
        set_vfl_buffer(None)
        set_vfl_calibrator(None)

        print(f"\n  Real DiT generation with VFL: PASS")
        if device == "cuda":
            del transformer
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [WARN] DiT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


# ===========================================================================
# Test 3: Determinism check (calibrator=None → identical)
# ===========================================================================


def test_determinism_no_calibrator():
    """TeaCache decisions should be identical with calibrator=None vs no param."""
    print("=" * 60)
    print("Test 3: Determinism (calibrator=None)")
    print("=" * 60)

    from accelerators.teacache import teacache_init, teacache_decide

    torch.manual_seed(42)
    num_steps = 20

    # Run A: no calibrator param
    state_a = teacache_init(num_steps=num_steps, rel_l1_thresh=0.25)
    decisions_a = []
    for _ in range(num_steps):
        mod = torch.randn(1, 256, 1152)
        should, _ = teacache_decide(state_a, mod)
        decisions_a.append(should)

    # Run B: calibrator=None explicitly
    torch.manual_seed(42)  # same seed
    state_b = teacache_init(num_steps=num_steps, rel_l1_thresh=0.25)
    decisions_b = []
    for _ in range(num_steps):
        mod = torch.randn(1, 256, 1152)
        should, _ = teacache_decide(state_b, mod, calibrator=None)
        decisions_b.append(should)

    assert decisions_a == decisions_b, \
        f"Decisions differ! A={decisions_a}, B={decisions_b}"
    print(f"  {num_steps} steps, identical decisions: PASS")

    # Run C: calibrator=None with probe_layer
    torch.manual_seed(42)
    state_c = teacache_init(num_steps=num_steps, rel_l1_thresh=0.25)
    decisions_c = []
    for _ in range(num_steps):
        mod = torch.randn(1, 256, 1152)
        should, _ = teacache_decide(state_c, mod, calibrator=None, probe_layer=20)
        decisions_c.append(should)

    assert decisions_a == decisions_c, \
        f"probe_layer affected decisions when calibrator=None!"
    print(f"  calibrator=None + probe_layer: identical decisions: PASS")

    print()
    return True


# ===========================================================================
# Main
# ===========================================================================


if __name__ == "__main__":
    results = []
    for test_fn, name in [
        (test_synthetic_event_flow, "Synthetic Event Flow"),
        (test_determinism_no_calibrator, "Determinism (calibrator=None)"),
        (test_real_dit_generation, "Real DiT Generation"),
    ]:
        try:
            ok = test_fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}: {name}")
    print("=" * 60)

    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
