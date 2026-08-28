import copy
import hashlib
import json

from scripts.analyze.check_covr_bandit_resume_smoke import validate_artifacts


SESSION_ID = "resume-smoke"
VERSION_KEY = "runtime-version"
STATE_PATH = "/tmp/resume-smoke/template_bandit_state.json"
SAMPLES_PER_LEG = 2
DATASET_START_INDEX = 10


def _manifest():
    return {
        "schema_version": 1,
        "version_key": VERSION_KEY,
        "num_steps": 50,
        "baseline_strategy_id": "uniform",
        "strategies": [
            {
                "strategy_id": "uniform",
                "method": "teacache",
                "params": {"refresh_mask": [True, False]},
                "modeled_flops": 8.0,
                "source": "test",
            },
            {
                "strategy_id": "front_loaded",
                "method": "teacache",
                "params": {"refresh_mask": [True, True]},
                "modeled_flops": 8.0,
                "source": "test",
            },
        ],
    }


def _manifest_hash(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assignment(trajectory_id, manifest_hash):
    return {
        "session_id": SESSION_ID,
        "trajectory_id": trajectory_id,
        "prequential_index": trajectory_id,
        "template_id": "uniform",
        "propensity": 1.0,
        "manifest_hash": manifest_hash,
        "sample_count": 1,
    }


def _state(count, manifest_hash):
    return {
        "schema_version": 4,
        "session_id": SESSION_ID,
        "version_key": VERSION_KEY,
        "manifest_hash": manifest_hash,
        "run_identity": {
            "dataset": "imagenet",
            "dataset_start_index": DATASET_START_INDEX,
            "seed": 42,
            "batch_size": 1,
            "safety_sample_rate": 0.1,
            "sentinel_rate": 0.0,
            "sentinel_horizon": 0,
        },
        "rng_state": {},
        "arm_stats": {},
        "safety": {},
        "completed_trajectories": list(range(count)),
        "assignments": [
            _assignment(trajectory_id, manifest_hash)
            for trajectory_id in range(count)
        ],
        "feedback": [],
    }


def _result(*, processed, resume_offset, generation_start, target, generated,
            manifest_hash):
    return {
        "config": {
            "covr_version_key": VERSION_KEY,
            "covr_session_id": SESSION_ID,
            "covr_template_bandit": True,
            "covr_force_template_id": None,
            "covr_profile_stages": True,
        },
        "aggregate": {
            "covr_template_bandit": {
                "session_id": SESSION_ID,
                "manifest_hash": manifest_hash,
                "state_path": STATE_PATH,
                "dataset_start_index": DATASET_START_INDEX,
                "resume_sample_offset": resume_offset,
                "generation_start_index": generation_start,
                "target_samples": target,
                "generated_samples_this_run": generated,
                "processed_samples": processed,
                "assignments": processed,
                "completed_trajectories": processed,
            },
            "covr_reward_telemetry": {
                "sentinel_rate": 0.0,
                "sentinel_count": 0,
            },
            "generation_profile": {
                "stage_total_s": {"generation_online": 1.0},
            },
        },
    }


def _valid_artifacts():
    manifest = _manifest()
    manifest_hash = _manifest_hash(manifest)
    first_state = _state(SAMPLES_PER_LEG, manifest_hash)
    final_state = _state(2 * SAMPLES_PER_LEG, manifest_hash)
    first_result = _result(
        processed=SAMPLES_PER_LEG,
        resume_offset=0,
        generation_start=DATASET_START_INDEX,
        target=SAMPLES_PER_LEG,
        generated=SAMPLES_PER_LEG,
        manifest_hash=manifest_hash,
    )
    resumed_result = _result(
        processed=2 * SAMPLES_PER_LEG,
        resume_offset=SAMPLES_PER_LEG,
        generation_start=DATASET_START_INDEX + SAMPLES_PER_LEG,
        target=2 * SAMPLES_PER_LEG,
        generated=SAMPLES_PER_LEG,
        manifest_hash=manifest_hash,
    )
    return manifest, first_result, resumed_result, first_state, final_state


def _validate(artifacts):
    manifest, first_result, resumed_result, first_state, final_state = artifacts
    return validate_artifacts(
        manifest=manifest,
        first_result=first_result,
        resumed_result=resumed_result,
        first_state=first_state,
        final_state=final_state,
        samples_per_leg=SAMPLES_PER_LEG,
        dataset_start_index=DATASET_START_INDEX,
        session_id=SESSION_ID,
        state_path=STATE_PATH,
    )


def test_valid_bandit_resume_artifacts_pass():
    assert _validate(_valid_artifacts()) == []


def test_bandit_resume_rejects_identity_mismatches():
    artifacts = list(copy.deepcopy(_valid_artifacts()))
    artifacts[2]["config"]["covr_session_id"] = "other-session"
    artifacts[4]["version_key"] = "other-version"
    artifacts[4]["manifest_hash"] = "other-hash"

    errors = _validate(tuple(artifacts))

    assert any("resumed results covr_session_id" in error for error in errors)
    assert any("final state version_key" in error for error in errors)
    assert any("final state manifest_hash" in error for error in errors)


def test_bandit_resume_rejects_trajectory_and_prefix_corruption():
    artifacts = list(copy.deepcopy(_valid_artifacts()))
    final_assignments = artifacts[4]["assignments"]
    final_assignments[0]["template_id"] = "front_loaded"
    final_assignments[-1]["trajectory_id"] = 2

    errors = _validate(tuple(artifacts))

    assert any("final trajectory IDs" in error for error in errors)
    assert any("do not preserve the first-state prefix" in error for error in errors)


def test_bandit_resume_rejects_window_and_count_mismatches():
    artifacts = list(copy.deepcopy(_valid_artifacts()))
    summary = artifacts[2]["aggregate"]["covr_template_bandit"]
    summary["resume_sample_offset"] = 0
    summary["generation_start_index"] = DATASET_START_INDEX
    summary["processed_samples"] = 3
    summary["generated_samples_this_run"] = 1

    errors = _validate(tuple(artifacts))

    assert any("resumed summary resume_sample_offset" in error for error in errors)
    assert any("resumed summary generation_start_index" in error for error in errors)
    assert any("resumed summary processed_samples" in error for error in errors)
    assert any(
        "resumed summary generated_samples_this_run" in error
        for error in errors)
