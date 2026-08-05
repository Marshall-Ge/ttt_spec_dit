#!/usr/bin/env bash
set -euo pipefail

# Small, paired COVR-on-TeaCache viability probe.  This does not establish
# adaptive quality; it only tests whether causal prefix features predict arm
# headroom on an out-of-sample split.
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
)

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
  python scripts/build_budget_manifest.py \
    --output "${MANIFEST}" \
    --num-steps "${NUM_STEPS}" \
    --refresh-count "${REFRESH_COUNT}" \
    --method teacache \
    --mandatory-prefix "${PREFIX_STEPS}" \
    --version-key "${VERSION_KEY}"
else
  echo "[1/4] Reusing manifest ${MANIFEST}"
fi

ARM_SPECS=()
for ARM in front_loaded uniform back_loaded geometric; do
  OUT="${ROOT_DIR}/${ARM}"
  RECORD="${OUT}/viability.jsonl"
  echo "[2/4] Running forced arm ${ARM} (${N_PROMPTS} images)"
  python main.py "${COMMON_ARGS[@]}" \
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

echo "[3/4] Running full reference"
python main.py "${COMMON_ARGS[@]}" \
  --method baseline --metrics latency flops speed \
  --n_prompts "${N_PROMPTS}" \
  --output_dir "${ROOT_DIR}/reference"

printf '[4/4] Running OOS viability gate\n'
set +e
python scripts/analyze_covr_v2_viability.py \
  --reference-dir "${REFERENCE_DIR}" \
  --output "${REPORT}" \
  $(printf -- '--arm %q ' "${ARM_SPECS[@]}")
GATE_STATUS=$?
set -e

printf '\nCOVR-v2 viability probe complete\n'
printf '  root:       %s\n' "${ROOT_DIR}"
printf '  manifest:   %s\n' "${MANIFEST}"
printf '  report:     %s\n' "${REPORT}"
printf '  boundary:   PASS only means causal OOS headroom; it is not adaptive FID/IS evidence\n'
exit "${GATE_STATUS}"
