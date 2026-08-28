#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORCHESTRATOR="${ROOT_DIR}/scripts/experiment/run_covr_template_experiment.sh"

# ---- config (all overridable via env) ----
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_profile_validation/${RUN_ID}}"
SESSION_ID="${SESSION_ID:-covr-profile-${RUN_ID}-seed${SEED:-42}}"
N_PROMPTS="${N_PROMPTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"
AUDIT_N="${AUDIT_N:-2048}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
TEMPLATE_COUNT="${TEMPLATE_COUNT:-4}"
MANDATORY_PREFIX="${MANDATORY_PREFIX:-3}"
MAX_TAYLOR_GAP="${MAX_TAYLOR_GAP:-5}"
TEMPLATE_ID="${TEMPLATE_ID:-timestep_prior}"
BANDIT_EPSILON="${BANDIT_EPSILON:-0.5}"
SENTINEL_RATE="${SENTINEL_RATE:-1.0}"
SAFETY_SAMPLE_RATE="${SAFETY_SAMPLE_RATE:-0.1}"
SENTINEL_HORIZON="${SENTINEL_HORIZON:-0}"

COVR_GPUS="${COVR_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${COVR_GPUS}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "COVR_GPUS must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi

MANIFEST="${OUTPUT_ROOT}/manifest.json"
BANDIT_STATE="${OUTPUT_ROOT}/bandit_state.json"
AUDIT_DIR="${OUTPUT_ROOT}/audit"
AUDIT_FILE="${AUDIT_DIR}/covr/events_${SESSION_ID}.jsonl"

mkdir -p "${OUTPUT_ROOT}/logs"
MAIN_LOG="${OUTPUT_ROOT}/logs/_main.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

BASE_ENV=(
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "SESSION_ID=${SESSION_ID}"
  "SEED=${SEED:-42}"
  "N_PROMPTS=${N_PROMPTS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "NUM_STEPS=${NUM_STEPS}"
  "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
  "TEMPLATE_COUNT=${TEMPLATE_COUNT}"
  "MANDATORY_PREFIX=${MANDATORY_PREFIX}"
  "MAX_TAYLOR_GAP=${MAX_TAYLOR_GAP}"
  "MANIFEST=${MANIFEST}"
  "BANDIT_STATE=${BANDIT_STATE}"
  "TEMPLATE_ID=${TEMPLATE_ID}"
  "BANDIT_EPSILON=${BANDIT_EPSILON}"
  "SENTINEL_RATE=${SENTINEL_RATE}"
  "SAFETY_SAMPLE_RATE=${SAFETY_SAMPLE_RATE}"
  "SENTINEL_HORIZON=${SENTINEL_HORIZON}"
  "DATASET_START_INDEX=${DATASET_START_INDEX}"
)

EXTRA_ARGS=("$@")

CLEANUP_PIDS=()
cleanup() {
  log "signal received, stopping children..."
  for pid in "${CLEANUP_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  exit 1
}
trap cleanup INT TERM

# ============================================================
# Phase 1: audit (GPU 0) → build manifest
# ============================================================
log "==== Phase 1: audit + manifest ===="

AUDIT_LOG="${OUTPUT_ROOT}/logs/audit.log"
log "running audit on GPU ${GPU_IDS[0]} → ${AUDIT_LOG}"

env "${BASE_ENV[@]}" \
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
  N_PROMPTS="${AUDIT_N}" \
  DATASET_START_INDEX=0 \
  bash "${ORCHESTRATOR}" audit \
  --covr-profile-stages "${EXTRA_ARGS[@]}" \
  >"${AUDIT_LOG}" 2>&1
log "audit ✓"

log "building manifest → ${MANIFEST}"
python "${ROOT_DIR}/scripts/analyze/build_covr_manifest.py" "${AUDIT_FILE}" \
  --output "${MANIFEST}" \
  --num-layers 28 \
  --template-count "${TEMPLATE_COUNT}" \
  --mandatory-prefix "${MANDATORY_PREFIX}" \
  --max-taylor-gap "${MAX_TAYLOR_GAP}" \
  >>"${MAIN_LOG}" 2>&1

# Verify manifest
python - "${MANIFEST}" "${TEMPLATE_ID}" <<'PY'
import json, sys
p, tid = sys.argv[1:]
with open(p) as f:
    m = json.load(f)
tids = {t["template_id"] for t in m.get("templates", [])}
if tid not in tids:
    raise SystemExit(f"template {tid!r} not found in {p!r}; available: {sorted(tids)}")
for t in m["templates"]:
    mask = t["refresh_mask"]
    print(f"  {t['template_id']}: refresh_count={sum(mask)}/{len(mask)}")
PY
log "manifest ✓"

# ============================================================
# Phase 2: 4-way parallel eval
# ============================================================
log "==== Phase 2: 4-way parallel eval ===="

EVAL_ENV=("${BASE_ENV[@]}" "DATASET_START_INDEX=${AUDIT_N}")
PIDS=()
PHASE_NAMES=()

launch_phase() {
  local name="$1" gpu_id="$2" phase="$3"
  local phase_log="${OUTPUT_ROOT}/logs/${name}.log"
  log "launching ${name} on GPU ${gpu_id} → ${phase_log}"
  (
    env "${EVAL_ENV[@]}" CUDA_VISIBLE_DEVICES="${gpu_id}" \
      bash "${ORCHESTRATOR}" "${phase}" \
      --covr-profile-stages "${EXTRA_ARGS[@]}" \
      >"${phase_log}" 2>&1
  ) &
  PIDS+=("$!")
  PHASE_NAMES+=("${name}")
  CLEANUP_PIDS+=("$!")
}

launch_phase baseline       "${GPU_IDS[0]}" baseline
launch_phase speca           "${GPU_IDS[1]}" speca
launch_phase timestep_prior  "${GPU_IDS[2]}" template
launch_phase bandit          "${GPU_IDS[3]}" bandit

log "all 4 phases launched (PIDs: ${PIDS[*]})"
log "tail -f ${MAIN_LOG}"

faild=0
for index in "${!PIDS[@]}"; do
  name="${PHASE_NAMES[index]}"
  pid="${PIDS[index]}"
  if wait "${pid}"; then
    log "${name} ✓ complete"
  else
    log "${name} ✗ FAILED (exit=$?) — see ${OUTPUT_ROOT}/logs/${name}.log"
    faild=1
  fi
done
trap - INT TERM

if [[ ${faild} -ne 0 ]]; then
  log "==== some phases FAILED ===="
  exit 1
fi

log "==== all phases passed, running summary ===="
env "${BASE_ENV[@]}" bash "${ORCHESTRATOR}" summary >>"${MAIN_LOG}" 2>&1
log "==== complete: ${OUTPUT_ROOT} ===="
