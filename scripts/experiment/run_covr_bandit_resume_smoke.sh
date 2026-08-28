#!/usr/bin/env bash
# One-command COVR experimental-bandit persistence/resume smoke test.
#
# This validates runtime state handling only. It does not measure whether the
# adaptive policy improves quality or efficiency.
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_STEPS="${COVR_NUM_STEPS:-50}"
REFRESH_COUNT="${COVR_REFRESH_COUNT:-8}"
SEED="${COVR_SEED:-42}"
GUIDANCE="${COVR_GUIDANCE_SCALE:-4.0}"
SAMPLES_PER_LEG="${COVR_SAMPLES_PER_LEG:-2}"
DATASET_START_INDEX="${COVR_DATASET_START_INDEX:-0}"
SESSION_ID="${COVR_SESSION_ID:-covr-bandit-resume-smoke-seed${SEED}}"

if ! [[ "${SAMPLES_PER_LEG}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COVR_SAMPLES_PER_LEG must be a positive integer" >&2
  exit 2
fi
if ! [[ "${DATASET_START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "COVR_DATASET_START_INDEX must be a non-negative integer" >&2
  exit 2
fi

TOTAL_TARGET=$((2 * SAMPLES_PER_LEG))
if [[ -n "${COVR_BANDIT_SMOKE_ROOT:-}" ]]; then
  ROOT="${COVR_BANDIT_SMOKE_ROOT}"
  mkdir -p "${ROOT}"
else
  ROOT="$(mktemp -d /tmp/covr_bandit_resume_smoke.XXXXXX)"
fi

PROBE_DIR="${ROOT}/version_probe"
MANIFEST="${ROOT}/teacache_k${REFRESH_COUNT}.json"
FIRST_RUN_DIR="${ROOT}/first_run"
RESUMED_RUN_DIR="${ROOT}/resumed_run"
STATE="${ROOT}/template_bandit_state.json"
FIRST_STATE="${ROOT}/state_after_first.json"

if [[ -e "${STATE}" || -e "${FIRST_STATE}" ]]; then
  echo "refusing to reuse an existing bandit state under ${ROOT}" >&2
  echo "set COVR_BANDIT_SMOKE_ROOT to a new or empty directory" >&2
  exit 2
fi

COMMON_ARGS=(
  --model dit
  --task c2i
  --dataset imagenet
  --method teacache
  --num_steps "${NUM_STEPS}"
  --seed "${SEED}"
  --guidance_scale "${GUIDANCE}"
  --dataset-start-index "${DATASET_START_INDEX}"
)

echo "COVR bandit resume smoke root: ${ROOT}"
echo "[1/6] Probing COVR runtime version identity"
python main.py "${COMMON_ARGS[@]}" \
  --metrics latency \
  --n_prompts 1 --batch_size 1 \
  --covr-profile-stages \
  --output_dir "${PROBE_DIR}"

VERSION_KEY="$(python - "${PROBE_DIR}/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["config"].get("covr_version_key")
if not value:
    raise SystemExit("results.json does not contain config.covr_version_key")
print(value)
PY
)"
echo "      version_key=${VERSION_KEY}"

echo "[2/6] Building external TeaCache strategy manifest"
python scripts/experiment/build_budget_manifest.py \
  --output "${MANIFEST}" \
  --method teacache \
  --num-steps "${NUM_STEPS}" \
  --refresh-count "${REFRESH_COUNT}" \
  --mandatory-prefix 3 \
  --baseline-arm uniform \
  --version-key "${VERSION_KEY}"

BANDIT_ARGS=(
  --metrics latency flops speed
  --batch_size 1
  --covr-profile-stages
  --covr-strategy-manifest "${MANIFEST}"
  --covr-strategy-bandit
  --covr-bandit-state "${STATE}"
  --covr-session-id "${SESSION_ID}"
  --covr-sentinel-rate 0
)

echo "[3/6] Running first bandit segment (${SAMPLES_PER_LEG} samples)"
python main.py "${COMMON_ARGS[@]}" "${BANDIT_ARGS[@]}" \
  --n_prompts "${SAMPLES_PER_LEG}" \
  --output_dir "${FIRST_RUN_DIR}"

echo "[4/6] Snapshotting first persisted state"
cp "${STATE}" "${FIRST_STATE}"

echo "[5/6] Resuming to cumulative target ${TOTAL_TARGET}"
python main.py "${COMMON_ARGS[@]}" "${BANDIT_ARGS[@]}" \
  --n_prompts "${TOTAL_TARGET}" \
  --output_dir "${RESUMED_RUN_DIR}"

echo "[6/6] Validating bandit persistence and resume"
python scripts/analyze/check_covr_bandit_resume_smoke.py "${ROOT}" \
  --manifest "${MANIFEST}" \
  --state "${STATE}" \
  --first-state "${FIRST_STATE}" \
  --samples-per-leg "${SAMPLES_PER_LEG}" \
  --dataset-start-index "${DATASET_START_INDEX}" \
  --session-id "${SESSION_ID}"
