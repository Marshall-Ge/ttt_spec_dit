#!/usr/bin/env bash
# Per-image COVR *contextual* strategy-bandit experiment for TeaCache.
#
# Layers deferred-commit contextual LinUCB selection on top of the plain
# strategy-bandit runner (run_covr_teacache_bandit.sh). Each trajectory runs a
# forced-calc prefix of K steps, then commits an arm using causal-prefix
# features as the LinUCB context, with an efficiency-aware reward
# (terminal_MSE + lambda * measured_flops).
#
# This is a learning/telemetry experiment. It intentionally does not run FID/IS.
# The default 1,000-image segment uses batch size 1, a 10% terminal-fidelity
# sentinel, no SpecA safety shadow, and a persisted state (schema v2) that can
# be resumed. A v1 (non-contextual) state cannot resume into this script.
#
# Fresh run:
#   bash scripts/run_covr_teacache_contextual.sh
# Resume to a larger cumulative target:
#   MODE=resume N_PROMPTS=2000 \
#     bash scripts/run_covr_teacache_contextual.sh
# Analyze without GPU work:
#   MODE=analyze bash scripts/run_covr_teacache_contextual.sh
# Sweep lambda (separate roots):
#   COVR_EFFICIENCY_LAMBDA=1e-4 COVR_OUTPUT_ROOT=output/covr_teacache/ctx_1e-4 \
#     bash scripts/run_covr_teacache_contextual.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${MODE:-run}"
NUM_STEPS="${NUM_STEPS:-50}"
N_PROMPTS="${N_PROMPTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"
LATENT_SEED_OFFSET="${LATENT_SEED_OFFSET:-0}"
EPSILON="${COVR_EPSILON:-0.20}"
PRIOR_PENALTY="${COVR_PRIOR_PENALTY:-0.0}"
SENTINEL_RATE="${COVR_SENTINEL_RATE:-0.10}"
WINDOW_SIZE="${COVR_ANALYSIS_WINDOW_SIZE:-100}"
SESSION_ID="${COVR_SESSION_ID:-covr-teacache-contextual-seed${SEED}}"
# Output lands under the canonical output/covr_teacache namespace (relative to
# the repo root). Override COVR_OUTPUT_ROOT for lambda/alpha sweeps.
ROOT="${COVR_OUTPUT_ROOT:-output/covr_teacache/contextual_seed${SEED}}"

# Contextual knobs.
# K = forced-calc prefix steps before the deferred arm commit. 3 matches the
# viability probe default and the mandatory prefix used by the manifest.
PREFIX_STEPS="${COVR_PREFIX_STEPS:-3}"
# LinUCB exploration alpha (LCB bonus weight). 1.0 is the standard default.
LINUCB_ALPHA="${COVR_LINUCB_ALPHA:-1.0}"
# Efficiency reward weight: combined = terminal_MSE + LAMBDA * cost, with
# cost = measured_flops / vanilla_flops in [0,1]. Set to 0.0 for a pure
# contextual-fidelity probe (isolates contextual selection from the efficiency
# term); 1e-3 is the canonical efficiency-aware setting.
LAMBDA="${COVR_EFFICIENCY_LAMBDA:-1e-3}"

# Format: threshold:expected_skip_rate. The rates are planning estimates used
# only as cost metadata; calibrate them from local TeaCache sweeps when possible.
THRESHOLD_ARMS="${COVR_THRESHOLD_ARMS:-0.15:0.30 0.25:0.48 0.40:0.60 0.60:0.70 1.00:0.80}"
BASELINE_ARM="${COVR_BASELINE_ARM:-threshold_0p25}"
# NOTE: contextual deferred-commit supports threshold arms ONLY. commit_arm
# drops refresh_mask and switches onto the rel_l1_thresh dynamic path, so a
# fixed-mask (refresh-count) arm cannot be committed — the bandit rejects mixed
# manifests at construction. Do NOT add --refresh-counts here.

case "${MODE}" in
  run|resume|analyze) ;;
  *)
    echo "MODE must be run, resume, or analyze" >&2
    exit 2
    ;;
esac
if [[ "${BATCH_SIZE}" != "1" ]]; then
  echo "BATCH_SIZE must be 1 for per-image COVR trajectories" >&2
  exit 2
fi
if ! [[ "${PREFIX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COVR_PREFIX_STEPS must be a positive integer (>=1)" >&2
  exit 2
fi
if (( PREFIX_STEPS >= NUM_STEPS )); then
  echo "COVR_PREFIX_STEPS (${PREFIX_STEPS}) must be < NUM_STEPS (${NUM_STEPS})" >&2
  exit 2
fi
if ! [[ "${N_PROMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_PROMPTS must be a positive cumulative target" >&2
  exit 2
fi
if ! [[ "${DATASET_START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "DATASET_START_INDEX must be a non-negative integer" >&2
  exit 2
fi

MANIFEST="${ROOT}/teacache_strategy_manifest.json"
STATE="${ROOT}/teacache_bandit_state.json"
PROBE_DIR="${ROOT}/version_probe"
SEGMENT_DIR="${ROOT}/segment_${N_PROMPTS}"
REPORT_DIR="${ROOT}/analysis_${N_PROMPTS}"

analyze() {
  if [[ ! -f "${STATE}" || ! -f "${MANIFEST}" ]]; then
    echo "analysis requires ${STATE} and ${MANIFEST}" >&2
    exit 2
  fi
  python scripts/analyze/analyze_covr_teacache_bandit.py "${STATE}" \
    --manifest "${MANIFEST}" \
    --output-dir "${REPORT_DIR}" \
    --window-size "${WINDOW_SIZE}"
}

if [[ "${MODE}" == "analyze" ]]; then
  analyze
  exit 0
fi

mkdir -p "${ROOT}"
if [[ "${MODE}" == "run" && -e "${STATE}" ]]; then
  echo "refusing to overwrite existing bandit state: ${STATE}" >&2
  echo "use MODE=resume or choose a new COVR_OUTPUT_ROOT" >&2
  exit 2
fi
if [[ "${MODE}" == "resume" ]]; then
  if [[ ! -f "${STATE}" || ! -f "${MANIFEST}" ]]; then
    echo "MODE=resume requires an existing state and manifest under ${ROOT}" >&2
    exit 2
  fi
fi
if [[ -e "${SEGMENT_DIR}/results.json" ]]; then
  echo "refusing to overwrite completed segment: ${SEGMENT_DIR}" >&2
  exit 2
fi

COMMON_ARGS=(
  --model dit
  --task c2i
  --dataset imagenet
  --method teacache
  --num_steps "${NUM_STEPS}"
  --seed "${SEED}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --dataset-start-index "${DATASET_START_INDEX}"
  --latent-seed-offset "${LATENT_SEED_OFFSET}"
  --batch_size 1
  --img_save_limit 0
)

if [[ "${MODE}" == "run" ]]; then
  echo "COVR TeaCache contextual bandit root: ${ROOT}"
  echo "  prefix_steps=${PREFIX_STEPS} linucb_alpha=${LINUCB_ALPHA} lambda=${LAMBDA}"
  echo "[1/3] Probing runtime version identity"
  python main.py "${COMMON_ARGS[@]}" \
    --metrics latency \
    --n_prompts 1 \
    --covr-profile-stages \
    --output_dir "${PROBE_DIR}"

  VERSION_KEY="$(python - "${PROBE_DIR}/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["config"].get("covr_version_key")
if not value:
    raise SystemExit("results.json lacks config.covr_version_key")
print(value)
PY
)"

  read -r -a threshold_specs <<< "${THRESHOLD_ARMS}"
  manifest_args=(
    --output "${MANIFEST}"
    --method teacache
    --num-steps "${NUM_STEPS}"
    --mandatory-prefix 3
    --baseline-arm "${BASELINE_ARM}"
    --version-key "${VERSION_KEY}"
  )
  for spec in "${threshold_specs[@]}"; do
    manifest_args+=(--threshold-arm "${spec}")
  done

  echo "[2/3] Building TeaCache threshold and fixed-mask arms"
  python scripts/experiment/build_budget_manifest.py "${manifest_args[@]}"
else
  echo "COVR TeaCache contextual bandit resume root: ${ROOT}"
fi

if [[ "${MODE}" == "run" ]]; then
  echo "[3/3] Running ${N_PROMPTS} per-image trajectories"
else
  echo "[1/1] Resuming to cumulative target ${N_PROMPTS}"
fi
python main.py "${COMMON_ARGS[@]}" \
  --metrics latency flops speed \
  --n_prompts "${N_PROMPTS}" \
  --covr-profile-stages \
  --covr-strategy-manifest "${MANIFEST}" \
  --covr-strategy-bandit \
  --covr-contextual-bandit \
  --covr-prefix-steps "${PREFIX_STEPS}" \
  --covr-linucb-alpha "${LINUCB_ALPHA}" \
  --covr-efficiency-lambda "${LAMBDA}" \
  --covr-bandit-state "${STATE}" \
  --covr-session-id "${SESSION_ID}" \
  --covr-bandit-epsilon "${EPSILON}" \
  --covr-bandit-prior-penalty "${PRIOR_PENALTY}" \
  --covr-safety-sample-rate 0 \
  --covr-sentinel-rate "${SENTINEL_RATE}" \
  --output_dir "${SEGMENT_DIR}"

analyze

echo
printf 'state:    %s\n' "${STATE}"
printf 'manifest: %s\n' "${MANIFEST}"
printf 'report:   %s\n' "${REPORT_DIR}"
