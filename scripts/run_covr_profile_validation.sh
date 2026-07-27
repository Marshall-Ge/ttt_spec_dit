#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCHESTRATOR="${ROOT_DIR}/scripts/run_covr_template_experiment.sh"

if [[ -z "${MANIFEST:-}" ]]; then
  if [[ $# -eq 0 || "$1" == --* ]]; then
    echo "usage: $0 MANIFEST [extra main.py args...]" >&2
    echo "   or: MANIFEST=/path/to/manifest.json $0 [extra main.py args...]" >&2
    exit 2
  fi
  MANIFEST="$1"
  shift
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest not found: ${MANIFEST}" >&2
  exit 2
fi

TEMPLATE_ID="${TEMPLATE_ID:-timestep_prior}"
python - "${MANIFEST}" "${TEMPLATE_ID}" <<'PY'
import json, sys
p, tid = sys.argv[1:]
with open(p, encoding="utf-8") as fh:
    m = json.load(fh)
if tid not in {t.get("template_id") for t in m.get("templates", [])}:
    raise SystemExit(f"template {tid!r} not found in manifest {p!r}")
PY

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_profile_validation/${RUN_ID}}"
SESSION_ID="${SESSION_ID:-covr-profile-${RUN_ID}-seed${SEED:-42}}"
BANDIT_STATE="${BANDIT_STATE:-${OUTPUT_ROOT}/bandit_state.json}"
N_PROMPTS="${N_PROMPTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"

COVR_GPUS="${COVR_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${COVR_GPUS}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "COVR_GPUS must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
  if [[ -z "${gpu_id}" ]]; then
    echo "COVR_GPUS contains an empty GPU id" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs"
MAIN_LOG="${OUTPUT_ROOT}/logs/_main.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }
log_n() { echo "[$(date '+%H:%M:%S')] $*" >> "${MAIN_LOG}"; }

VALIDATION_ENV=(
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "SESSION_ID=${SESSION_ID}"
  "MANIFEST=${MANIFEST}"
  "BANDIT_STATE=${BANDIT_STATE}"
  "TEMPLATE_ID=${TEMPLATE_ID}"
  "N_PROMPTS=${N_PROMPTS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "DATASET_START_INDEX=${DATASET_START_INDEX}"
)
EXTRA_ARGS=("$@")
PIDS=()
PHASE_NAMES=()

launch_phase() {
  local name="$1" gpu_id="$2" phase="$3"
  local phase_log="${OUTPUT_ROOT}/logs/${name}.log"

  log "launching ${name} on GPU ${gpu_id} → ${phase_log}"
  (
    env "${VALIDATION_ENV[@]}" CUDA_VISIBLE_DEVICES="${gpu_id}" \
      bash "${ORCHESTRATOR}" "${phase}" \
      --covr-profile-stages "${EXTRA_ARGS[@]}" \
      >"${phase_log}" 2>&1
  ) &
  PIDS+=("$!")
  PHASE_NAMES+=("${name}")
}

cleanup() {
  log "received signal, stopping children..."
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  log "all children stopped"
  exit 1
}
trap cleanup INT TERM

log "==== COVR profile validation start ===="
log "output:  ${OUTPUT_ROOT}"
log "manifest: ${MANIFEST}"
log "gpus:     ${COVR_GPUS}"
log "prompts:  ${N_PROMPTS}  batch: ${BATCH_SIZE}"

launch_phase baseline       "${GPU_IDS[0]}" baseline
launch_phase speca           "${GPU_IDS[1]}" speca
launch_phase timestep_prior  "${GPU_IDS[2]}" template
launch_phase bandit          "${GPU_IDS[3]}" bandit

log "all 4 phases launched (PIDs: ${PIDS[*]})"
log "tail -f ${MAIN_LOG}"

failed=0
for index in "${!PIDS[@]}"; do
  local name="${PHASE_NAMES[index]}"
  local pid="${PIDS[index]}"
  if wait "${pid}"; then
    log "${name} ✓ complete"
  else
    log "${name} ✗ FAILED (exit=$?) — see ${OUTPUT_ROOT}/logs/${name}.log"
    failed=1
  fi
done
trap - INT TERM

if [[ ${failed} -ne 0 ]]; then
  log "==== some phases FAILED ===="
  exit 1
fi

log "==== all phases passed, running summary ===="
env "${VALIDATION_ENV[@]}" bash "${ORCHESTRATOR}" summary >> "${MAIN_LOG}" 2>&1
log "==== complete: ${OUTPUT_ROOT} ===="
