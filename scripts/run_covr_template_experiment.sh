#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
  echo "usage: $0 {smoke|audit|manifest|baseline|speca|bandit} [extra main.py args...]" >&2
  exit 2
fi
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-42}"
if [[ "${PHASE}" == "smoke" ]]; then
  RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_template_smoke/${RUN_ID}}"
  SESSION_ID="${SESSION_ID:-covr-smoke-${RUN_ID}-seed${SEED}}"
  N_PROMPTS="${N_PROMPTS:-64}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  SENTINEL_RATE="${SENTINEL_RATE:-1.0}"
  SAFETY_SAMPLE_RATE="${SAFETY_SAMPLE_RATE:-1.0}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_template_experiment}"
  SESSION_ID="${SESSION_ID:-covr-template-seed42}"
  N_PROMPTS="${N_PROMPTS:-50000}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  SENTINEL_RATE="${SENTINEL_RATE:-0.05}"
  SAFETY_SAMPLE_RATE="${SAFETY_SAMPLE_RATE:-0.1}"
fi
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
TEMPLATE_COUNT="${TEMPLATE_COUNT:-4}"
MANDATORY_PREFIX="${MANDATORY_PREFIX:-3}"
MAX_TAYLOR_GAP="${MAX_TAYLOR_GAP:-5}"
SENTINEL_HORIZON="${SENTINEL_HORIZON:-5}"
BANDIT_EPSILON="${BANDIT_EPSILON:-0.1}"

AUDIT_DIR="${OUTPUT_ROOT}/audit"
AUDIT_FILE="${AUDIT_FILE:-${AUDIT_DIR}/covr/events_${SESSION_ID}.jsonl}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifest.json}"
BANDIT_STATE="${BANDIT_STATE:-${OUTPUT_ROOT}/bandit_state.json}"

COMMON_ARGS=(
  --model dit
  --task c2i
  --dataset imagenet
  --seed "${SEED}"
  --num_steps "${NUM_STEPS}"
  --n_prompts "${N_PROMPTS}"
  --batch_size "${BATCH_SIZE}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --metrics fid is latency flops speed
)

case "${PHASE}" in
  smoke)
    SMOKE_ENV=(
      "OUTPUT_ROOT=${OUTPUT_ROOT}"
      "SESSION_ID=${SESSION_ID}"
      "SEED=${SEED}"
      "N_PROMPTS=${N_PROMPTS}"
      "BATCH_SIZE=${BATCH_SIZE}"
      "NUM_STEPS=${NUM_STEPS}"
      "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
      "TEMPLATE_COUNT=${TEMPLATE_COUNT}"
      "MANDATORY_PREFIX=${MANDATORY_PREFIX}"
      "MAX_TAYLOR_GAP=${MAX_TAYLOR_GAP}"
      "SENTINEL_RATE=${SENTINEL_RATE}"
      "SENTINEL_HORIZON=${SENTINEL_HORIZON}"
      "SAFETY_SAMPLE_RATE=${SAFETY_SAMPLE_RATE}"
      "BANDIT_EPSILON=${BANDIT_EPSILON}"
    )
    if [[ -e "${AUDIT_FILE}" && "${REUSE_AUDIT:-0}" == "1" ]]; then
      echo "reusing audit file: ${AUDIT_FILE}"
    else
      env "${SMOKE_ENV[@]}" bash "$0" audit "$@"
    fi
    env "${SMOKE_ENV[@]}" bash "$0" manifest
    env "${SMOKE_ENV[@]}" bash "$0" bandit "$@"
    echo "smoke complete: ${OUTPUT_ROOT}"
    ;;
  audit)
    if [[ -e "${AUDIT_FILE}" && "${ALLOW_AUDIT_APPEND:-0}" != "1" ]]; then
      echo "audit file already exists: ${AUDIT_FILE}" >&2
      echo "use a new SESSION_ID or set ALLOW_AUDIT_APPEND=1" >&2
      exit 2
    fi
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --method speca \
      --output_dir "${AUDIT_DIR}" \
      --covr-shadow \
      --covr-session-id "${SESSION_ID}" \
      --covr-output-dir "${AUDIT_DIR}/covr" \
      "$@"
    ;;
  manifest)
    python "${ROOT_DIR}/scripts/build_covr_manifest.py" "${AUDIT_FILE}" \
      --output "${MANIFEST}" \
      --num-layers 28 \
      --template-count "${TEMPLATE_COUNT}" \
      --mandatory-prefix "${MANDATORY_PREFIX}" \
      --max-taylor-gap "${MAX_TAYLOR_GAP}" \
      "$@"
    ;;
  baseline)
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --method baseline \
      --output_dir "${OUTPUT_ROOT}/baseline" \
      "$@"
    ;;
  speca)
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --method speca \
      --output_dir "${OUTPUT_ROOT}/speca" \
      "$@"
    ;;
  bandit)
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --method speca \
      --output_dir "${OUTPUT_ROOT}/bandit" \
      --covr-template-bandit \
      --covr-template-manifest "${MANIFEST}" \
      --covr-bandit-state "${BANDIT_STATE}" \
      --covr-session-id "${SESSION_ID}" \
      --covr-bandit-epsilon "${BANDIT_EPSILON}" \
      --covr-safety-sample-rate "${SAFETY_SAMPLE_RATE}" \
      --covr-sentinel-rate "${SENTINEL_RATE}" \
      --covr-sentinel-horizon "${SENTINEL_HORIZON}" \
      "$@"
    ;;
  *)
    echo "unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
