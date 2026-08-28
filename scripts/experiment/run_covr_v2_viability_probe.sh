#!/usr/bin/env bash
set -euo pipefail

# Small, paired COVR-on-TeaCache viability probe.  This does not establish
# adaptive quality; it only tests whether causal prefix features (legacy
# pixel-space and/or boundary telemetry) predict arm headroom on an
# out-of-sample split.
#
# Usage:
#   MODE=analyze OUTPUT_ROOT=/path/to/existing/run bash scripts/run_covr_v2_viability_probe.sh
#   MODE=run bash scripts/run_covr_v2_viability_probe.sh
#
# In MODE=analyze, only the CPU analyzer runs against existing artifacts.
# In MODE=run, the full probe (version identity, 4 forced arms, reference,
# analyzer) runs.  Default: MODE=run.

MODE="${MODE:-run}"
ROOT_DIR="${OUTPUT_ROOT:-$(mktemp -d /tmp/covr_v2_viability.XXXXXX)}"
NUM_STEPS="${NUM_STEPS:-50}"
REFRESH_COUNT="${REFRESH_COUNT:-8}"
PREFIX_STEPS="${PREFIX_STEPS:-3}"
N_PROMPTS="${N_PROMPTS:-128}"
SEED="${SEED:-42}"
LATENT_SEED_OFFSET="${LATENT_SEED_OFFSET:-0}"
GUIDANCE="${GUIDANCE:-4.5}"
SESSION_ID="${SESSION_ID:-covr-v2-viability-${SEED}-${LATENT_SEED_OFFSET}}"
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-${N_PROMPTS}}"
IMAGE_FORMAT="${IMAGE_FORMAT:-png}"            # default to PNG (no JPEG confound)
USE_BOUNDARY_TELEMETRY="${USE_BOUNDARY_TELEMETRY:-1}"

mkdir -p "${ROOT_DIR}"
PROBE_DIR="${ROOT_DIR}/version_probe"
MANIFEST="${ROOT_DIR}/budget_k${REFRESH_COUNT}.json"
REFERENCE_DIR="${ROOT_DIR}/reference/generated"
REPORT="${ROOT_DIR}/viability_report.json"

COMMON_ARGS=(
  --model dit
  --task c2i
  --dataset imagenet
  --num_steps "${NUM_STEPS}"
  --seed "${SEED}"
  --latent-seed-offset "${LATENT_SEED_OFFSET}"
  --guidance_scale "${GUIDANCE}"
  --batch_size 1
  --img_save_limit "${IMG_SAVE_LIMIT}"
  --covr-viability-image-format "${IMAGE_FORMAT}"
)

ANALYZER_ARGS=(
  --reference-dir "${REFERENCE_DIR}"
  --output "${REPORT}"
  --permutations 500
)

# ---- Boundary telemetry flag ----
BOUNDARY_FLAG=()
if [[ "${USE_BOUNDARY_TELEMETRY}" == "1" ]]; then
  BOUNDARY_FLAG+=(--covr-viability-boundary-telemetry)
fi

# ============================================================================
# MODE=analyze — CPU-only, no GPU work
# ============================================================================
if [[ "${MODE}" == "analyze" ]]; then
  if [[ ! -d "${REFERENCE_DIR}" ]]; then
    printf 'FATAL: reference directory %s does not exist\n' "${REFERENCE_DIR}"
    printf 'Set REFERENCE_DIR or OUTPUT_ROOT to a completed run\n'
    exit 3
  fi

  ARM_SPECS_ANALYZE=()
  for ARM in front_loaded uniform back_loaded geometric; do
    RECORD="${ROOT_DIR}/${ARM}/viability.jsonl"
    if [[ -f "${RECORD}" ]]; then
      ARM_SPECS_ANALYZE+=("${ARM}=${RECORD}")
    else
      printf 'WARNING: arm %s JSONL not found at %s — skipping\n' "${ARM}" "${RECORD}"
    fi
  done
  if [[ ${#ARM_SPECS_ANALYZE[@]} -lt 2 ]]; then
    printf 'FATAL: fewer than 2 arm JSONL files found\n'
    exit 2
  fi

  printf '[analyze] Running viability gate against %d arms\n' "${#ARM_SPECS_ANALYZE[@]}"
  set +e
  python scripts/analyze_covr_v2_viability.py \
    "${ANALYZER_ARGS[@]}" \
    $(printf -- '--arm %q ' "${ARM_SPECS_ANALYZE[@]}")
  GATE_STATUS=$?
  set -e

  printf '\nCOVR-v2 viability analysis complete\n'
  printf '  report: %s\n' "${REPORT}"
  exit "${GATE_STATUS}"
fi

# ============================================================================
# MODE=run — full probe
# ============================================================================

printf '\n========================================\n'
printf '  COVR-v2 TeaCache viability probe\n'
printf '========================================\n'
printf '  root:            %s\n' "${ROOT_DIR}"
printf '  images:          %d\n' "${N_PROMPTS}"
printf '  arms:            4 (front_loaded, uniform, back_loaded, geometric)\n'
printf '  total images:    %d (1 version + 4*%d arms + %d reference)\n' \
       "$((1 + 4 * N_PROMPTS + N_PROMPTS))" "${N_PROMPTS}" "${N_PROMPTS}"
printf '  refresh count:   %d\n' "${REFRESH_COUNT}"
printf '  steps:           %d\n' "${NUM_STEPS}"
printf '  method:          teacache (forced schedule)\n'
printf '  image format:    %s\n' "${IMAGE_FORMAT}"
printf '  boundary telemetry: %s\n' "${USE_BOUNDARY_TELEMETRY}"
printf '  stop rule:       STOP → halt; INSUFFICIENT_DATA → halt\n'
printf '========================================\n\n'

# ---- Phase 1: Version identity ----
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[1/4] Probing runtime version identity"
  python main.py "${COMMON_ARGS[@]}" \
    --method teacache --metrics latency \
    --n_prompts 1 --covr-profile-stages \
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
  python scripts/analyze/build_budget_manifest.py \
    --output "${MANIFEST}" \
    --num-steps "${NUM_STEPS}" \
    --refresh-count "${REFRESH_COUNT}" \
    --method teacache \
    --mandatory-prefix "${PREFIX_STEPS}" \
    --version-key "${VERSION_KEY}"
else
  echo "[1/4] Reusing manifest ${MANIFEST}"
fi

# ---- Phase 2: Forced arms ----
ARM_SPECS=()
for ARM in front_loaded uniform back_loaded geometric; do
  OUT="${ROOT_DIR}/${ARM}"
  RECORD="${OUT}/viability.jsonl"
  echo "[2/4] Running forced arm ${ARM} (${N_PROMPTS} images)"
  python main.py "${COMMON_ARGS[@]}" \
    "${BOUNDARY_FLAG[@]}" \
    --method teacache --metrics latency flops speed \
    --n_prompts "${N_PROMPTS}" \
    --covr-strategy-manifest "${MANIFEST}" \
    --covr-force-strategy-id "${ARM}" \
    --covr-session-id "${SESSION_ID}-${ARM}" \
    --covr-sentinel-rate 0 \
    --covr-viability-output "${RECORD}" \
    --covr-viability-prefix-steps "${PREFIX_STEPS}" \
    --output_dir "${OUT}"
  ARM_SPECS+=("${ARM}=${RECORD}")
done

# ---- Phase 3: Full reference ----
echo "[3/4] Running full reference"
python main.py "${COMMON_ARGS[@]}" \
  --method baseline --metrics latency flops speed \
  --n_prompts "${N_PROMPTS}" \
  --output_dir "${ROOT_DIR}/reference"

# ---- Phase 4: Viability gate ----
printf '[4/4] Running OOS viability gate\n'
set +e
python scripts/analyze_covr_v2_viability.py \
  "${ANALYZER_ARGS[@]}" \
  $(printf -- '--arm %q ' "${ARM_SPECS[@]}")
GATE_STATUS=$?
set -e

printf '\nCOVR-v2 viability probe complete\n'
printf '  root:       %s\n' "${ROOT_DIR}"
printf '  manifest:   %s\n' "${MANIFEST}"
printf '  report:     %s\n' "${REPORT}"
printf '  boundary:   '
if [[ "${GATE_STATUS}" -eq 0 ]]; then
  printf 'PASS — causal OOS headroom detected; proceed to confirmatory 500-image experiment\n'
elif [[ "${GATE_STATUS}" -eq 1 ]]; then
  printf 'STOP — no reliable OOS headroom; do not escalate to stage C\n'
elif [[ "${GATE_STATUS}" -eq 2 ]]; then
  printf 'VIABILITY ERROR — see report for details\n'
else
  printf 'status %d\n' "${GATE_STATUS}"
fi
exit "${GATE_STATUS}"
