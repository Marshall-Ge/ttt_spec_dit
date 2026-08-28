#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
  echo "usage: $0 {smoke|pilot|validate|eval|summary|audit|manifest|baseline|speca|template|bandit} [extra main.py args...]" >&2
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
elif [[ "${PHASE}" == "pilot" ]]; then
  RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_template_pilot/${RUN_ID}}"
  SESSION_ID="${SESSION_ID:-covr-pilot-${RUN_ID}-seed${SEED}}"
  N_PROMPTS="${N_PROMPTS:-2048}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  SENTINEL_RATE="${SENTINEL_RATE:-0.25}"
  SAFETY_SAMPLE_RATE="${SAFETY_SAMPLE_RATE:-0.25}"
  BANDIT_EPSILON="${BANDIT_EPSILON:-0.2}"
elif [[ "${PHASE}" == "validate" ]]; then
  RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_template_validation/${RUN_ID}}"
  SESSION_ID="${SESSION_ID:-covr-validation-${RUN_ID}-seed${SEED}}"
  N_PROMPTS="${N_PROMPTS:-2048}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  SENTINEL_RATE="${SENTINEL_RATE:-1.0}"
  SAFETY_SAMPLE_RATE="${SAFETY_SAMPLE_RATE:-0.1}"
  BANDIT_EPSILON="${BANDIT_EPSILON:-0.5}"
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
SENTINEL_HORIZON="${SENTINEL_HORIZON:-0}"
BANDIT_EPSILON="${BANDIT_EPSILON:-0.1}"
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"
AUDIT_N="${AUDIT_N:-2048}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

AUDIT_DIR="${OUTPUT_ROOT}/audit"
AUDIT_FILE="${AUDIT_FILE:-${AUDIT_DIR}/covr/events_${SESSION_ID}.jsonl}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifest.json}"
BANDIT_STATE="${BANDIT_STATE:-${OUTPUT_ROOT}/bandit_state.json}"

_maybe_skip() {
  local out_dir="$1"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/results.json" ]]; then
    echo "[skip] ${out_dir}/results.json exists"
    return 0
  fi
  return 1
}

COMMON_ARGS=(
  --model dit
  --task c2i
  --dataset imagenet
  --seed "${SEED}"
  --num_steps "${NUM_STEPS}"
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
  pilot)
    PILOT_ENV=(
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
    env "${PILOT_ENV[@]}" bash "$0" audit "$@"
    env "${PILOT_ENV[@]}" bash "$0" manifest
    env "${PILOT_ENV[@]}" bash "$0" baseline "$@"
    env "${PILOT_ENV[@]}" bash "$0" speca "$@"
    env "${PILOT_ENV[@]}" bash "$0" bandit "$@"
    echo "pilot complete: ${OUTPUT_ROOT}"
    ;;
  validate)
    VALIDATE_ENV=(
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
    env "${VALIDATE_ENV[@]}" N_PROMPTS="${AUDIT_N}" DATASET_START_INDEX=0 bash "$0" audit "$@"
    env "${VALIDATE_ENV[@]}" N_PROMPTS="${AUDIT_N}" DATASET_START_INDEX=0 bash "$0" manifest
    env "${VALIDATE_ENV[@]}" DATASET_START_INDEX="${AUDIT_N}" bash "$0" baseline "$@"
    env "${VALIDATE_ENV[@]}" DATASET_START_INDEX="${AUDIT_N}" bash "$0" speca "$@"
    while IFS= read -r template_id; do
      env "${VALIDATE_ENV[@]}" DATASET_START_INDEX="${AUDIT_N}" \
        TEMPLATE_ID="${template_id}" bash "$0" template "$@"
    done < <(python -c 'import json, sys; print("\n".join(t["template_id"] for t in json.load(open(sys.argv[1]))["templates"]))' "${MANIFEST}")
    env "${VALIDATE_ENV[@]}" DATASET_START_INDEX="${AUDIT_N}" bash "$0" bandit "$@"
    echo "validation complete: ${OUTPUT_ROOT}"
    ;;
  eval)
    # eval-only: skip audit+manifest, run evaluation methods at DATASET_START_INDEX
    DATASET_START_INDEX="${DATASET_START_INDEX:-${AUDIT_N}}"
    _maybe_skip "${OUTPUT_ROOT}/baseline" || bash "$0" baseline "$@"
    _maybe_skip "${OUTPUT_ROOT}/speca" || bash "$0" speca "$@"
    while IFS= read -r template_id; do
      _maybe_skip "${OUTPUT_ROOT}/templates/${template_id}" || \
        TEMPLATE_ID="${template_id}" bash "$0" template "$@"
    done < <(python -c 'import json, sys; print("\n".join(t["template_id"] for t in json.load(open(sys.argv[1]))["templates"]))' "${MANIFEST}")
    _maybe_skip "${OUTPUT_ROOT}/bandit" || bash "$0" bandit "$@"
    echo "eval complete: ${OUTPUT_ROOT}"
    ;;
  summary)
    echo "method,fid,is_mean,flops_T,img_per_s"
    for d in "${OUTPUT_ROOT}/baseline" "${OUTPUT_ROOT}/speca" "${OUTPUT_ROOT}/templates/"* "${OUTPUT_ROOT}/bandit"; do
      [[ -f "${d}/results.json" ]] || continue
      name=$(basename "${d}")
      python3 -c "
import json
d=json.load(open('${d}/results.json'))
a=d.get('aggregate',{})
fid=a.get('fid','-')
is_m=a.get('is_mean','-')
flops=a.get('flops_online_T',a.get('flops_accel_T','-'))
speed=a.get('speed_online_img_per_s',a.get('speed_img_per_s','-'))
print(f'${name},{fid},{is_m},{flops},{speed}')
"
    done
    echo "summary complete: ${OUTPUT_ROOT}"
    ;;
  audit)
    if [[ -e "${AUDIT_FILE}" && "${ALLOW_AUDIT_APPEND:-0}" != "1" ]]; then
      echo "audit file already exists: ${AUDIT_FILE}" >&2
      echo "use a new SESSION_ID or set ALLOW_AUDIT_APPEND=1" >&2
      exit 2
    fi
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --n_prompts "${N_PROMPTS}" \
      --dataset-start-index "${DATASET_START_INDEX}" \
      --method speca \
      --output_dir "${AUDIT_DIR}" \
      --covr-shadow \
      --covr-session-id "${SESSION_ID}" \
      --covr-output-dir "${AUDIT_DIR}/covr" \
      "$@"
    ;;
  manifest)
    python "${ROOT_DIR}/scripts/analyze/build_covr_manifest.py" "${AUDIT_FILE}" \
      --output "${MANIFEST}" \
      --num-layers 28 \
      --template-count "${TEMPLATE_COUNT}" \
      --mandatory-prefix "${MANDATORY_PREFIX}" \
      --max-taylor-gap "${MAX_TAYLOR_GAP}" \
      "$@"
    ;;
  baseline)
    _maybe_skip "${OUTPUT_ROOT}/baseline" || true
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --n_prompts "${N_PROMPTS}" \
      --dataset-start-index "${DATASET_START_INDEX}" \
      --method baseline \
      --output_dir "${OUTPUT_ROOT}/baseline" \
      "$@"
    ;;
  speca)
    _maybe_skip "${OUTPUT_ROOT}/speca" || true
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --n_prompts "${N_PROMPTS}" \
      --dataset-start-index "${DATASET_START_INDEX}" \
      --method speca \
      --output_dir "${OUTPUT_ROOT}/speca" \
      "$@"
    ;;
  template)
    if [[ -z "${TEMPLATE_ID:-}" ]]; then
      echo "TEMPLATE_ID is required for the template phase" >&2
      exit 2
    fi
    _maybe_skip "${OUTPUT_ROOT}/templates/${TEMPLATE_ID}" || true
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --n_prompts "${N_PROMPTS}" \
      --dataset-start-index "${DATASET_START_INDEX}" \
      --method speca \
      --output_dir "${OUTPUT_ROOT}/templates/${TEMPLATE_ID}" \
      --covr-template-manifest "${MANIFEST}" \
      --covr-force-template-id "${TEMPLATE_ID}" \
      "$@"
    ;;
  bandit)
    _maybe_skip "${OUTPUT_ROOT}/bandit" || true
    python "${ROOT_DIR}/main.py" "${COMMON_ARGS[@]}" \
      --n_prompts "${N_PROMPTS}" \
      --dataset-start-index "${DATASET_START_INDEX}" \
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
