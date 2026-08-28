#!/usr/bin/env bash
# Complete the exact-K=8 plain TeaCache comparator and render the paired verdict.
# Existing replicas with results.json are kept, so the script is safe to resume.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-output/covr_static_k8}"
THRESHOLD="${THRESHOLD:-1.40}"
OFFSETS="${OFFSETS:-0 1 2 3 4}"
SEED="${SEED:-42}"
NUM_STEPS="${NUM_STEPS:-50}"
N_PROMPTS="${N_PROMPTS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GUIDANCE="${GUIDANCE:-4.5}"
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-1}"

if ! "${PYTHON}" - "${THRESHOLD}" "${NUM_STEPS}" "${N_PROMPTS}" \
    "${BATCH_SIZE}" "${IMG_SAVE_LIMIT}" <<'PY' >/dev/null
import math
import sys

threshold = float(sys.argv[1])
assert math.isfinite(threshold) and threshold >= 0, "THRESHOLD must be finite and >= 0"
for name, raw in zip(
        ("NUM_STEPS", "N_PROMPTS", "BATCH_SIZE", "IMG_SAVE_LIMIT"),
        sys.argv[2:]):
    assert int(raw) > 0, f"{name} must be positive"
PY
then
  exit 2
fi

OFFSETS="${OFFSETS//,/ }"
for offset in ${OFFSETS}; do
  if ! [[ "${offset}" =~ ^[0-9]+$ ]]; then
    echo "OFFSETS must contain non-negative integers (got '${offset}')" >&2
    exit 2
  fi

  threshold_dir="${OUT_DIR}/threshold/thresh_${THRESHOLD}"
  if [ "${offset}" = "0" ]; then
    run_dir="${threshold_dir}"
  else
    run_dir="${threshold_dir}/rep_${offset}"
  fi

  if [ -f "${run_dir}/results.json" ]; then
    echo "[skip] threshold=${THRESHOLD} offset=${offset}: results already exist"
    continue
  fi
  if [ -e "${run_dir}" ]; then
    echo "[error] incomplete output directory exists: ${run_dir}" >&2
    echo "        inspect, remove, or rename it before resuming" >&2
    exit 2
  fi

  echo "[run] threshold=${THRESHOLD} offset=${offset}"
  "${PYTHON}" main.py \
    --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${NUM_STEPS}" \
    --n_prompts "${N_PROMPTS}" --latent-seed-offset "${offset}" \
    --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}" \
    --img_save_limit "${IMG_SAVE_LIMIT}" --thresh "${THRESHOLD}" \
    --output_dir "${run_dir}"
done

echo ""
"${PYTHON}" scripts/analyze_static_masks.py "${OUT_DIR}"
