#!/usr/bin/env bash
# =============================================================================
# TeaCache decision-value sweeps — the go/no-go gate for COVR-on-TeaCache.
#
# Answers ONE upstream question before any 5k-50k bandit spend:
#   "On TeaCache, does per-trajectory adaptive decision (mask OR threshold)
#    beat a single static setting at all?"
#
# It runs two cheap sweeps and an analyzer:
#   [A]  Threshold sweep  — plain --method teacache at several thresholds.
#        => the quality-FLOPs Pareto frontier any bandit must beat, and
#           whether the threshold knob is smooth (=> just pick a point).
#   [B1] Equal-FLOPs mask sweep — each manifest arm forced individually via
#        --covr-force-strategy-id (NO bandit), plus same-arm noise replicas.
#        => at fixed calc-count, does WHERE calc sits move FID beyond noise?
#
# Neither sweep touches the bandit — pure forced-schedule / static teacache.
#
# Usage:
#   bash scripts/sweep_teacache_decision.sh [AUDIT_JSONL]
# Env overrides (all optional):
#   N_PROMPTS (default 500)   NUM_STEPS (50)   BATCH_SIZE (32)
#   SEED (42)   GUIDANCE (4.5)   THRESHOLDS ("0.10,0.20,0.30,0.40")
#   OUT_DIR (/tmp/tc_decision_sweep)   TEMPLATE_COUNT (4)
#
# Prereqs: project root, GPU env that can run main.py.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"

N_PROMPTS="${N_PROMPTS:-500}"
NUM_STEPS="${NUM_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
GUIDANCE="${GUIDANCE:-4.5}"
THRESHOLDS="${THRESHOLDS:-0.10,0.20,0.30,0.40}"
TEMPLATE_COUNT="${TEMPLATE_COUNT:-4}"
OUT_DIR="${OUT_DIR:-/tmp/tc_decision_sweep}"

MANIFEST="${OUT_DIR}/tc_mask_manifest.json"
THRESH_DIR="${OUT_DIR}/threshold"
EQF_DIR="${OUT_DIR}/equalflops"
AUDIT_GEN_DIR="${OUT_DIR}/speca_audit"

run_main() {  # run_main <output_dir> <extra args...>
  local out="$1"; shift
  rm -rf "${out}"
  python main.py --model dit --task c2i --dataset imagenet \
      --method teacache --metrics fid is latency flops speed \
      --seed "${SEED}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}" \
      --output_dir "${out}" "$@"
}

# =============================================================================
# [A] Threshold sweep — plain dynamic-threshold teacache at each value.
# =============================================================================
echo ""
echo "########## [A] Threshold sweep: ${THRESHOLDS} ##########"
IFS=',' read -ra THRESH_ARR <<< "${THRESHOLDS}"
for t in "${THRESH_ARR[@]}"; do
  t="$(echo "${t}" | xargs)"   # trim
  echo "  --- teacache thresh=${t} ---"
  run_main "${THRESH_DIR}/thresh_${t}" --thresh "${t}"
done

# =============================================================================
# [B1] Equal-FLOPs mask sweep. Build the mask manifest, then force each arm.
# =============================================================================
echo ""
echo "########## [B1] Building equal-FLOPs mask manifest ##########"

# Locate or bootstrap the SpecA action-audit the mask search needs.
AUDIT="${1:-}"
if [[ -z "${AUDIT}" ]]; then
  AUDIT="$(ls -t output/covr_test/covr/events_*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -z "${AUDIT}" ]]; then
    AUDIT="$(ls -t "${AUDIT_GEN_DIR}"/events_*.jsonl 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -z "${AUDIT}" || ! -f "${AUDIT}" ]]; then
  echo "  no audit found — bootstrapping via SpecA + --covr-shadow"
  rm -rf "${AUDIT_GEN_DIR}"
  python main.py --model dit --task c2i --dataset imagenet \
      --method speca --metrics fid is latency flops speed \
      --covr-shadow --covr-output-dir "${AUDIT_GEN_DIR}" \
      --seed "${SEED}" --num_steps "${NUM_STEPS}" \
      --n_prompts "$(( N_PROMPTS < 80 ? N_PROMPTS : 80 ))" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}"
  AUDIT="$(ls -t "${AUDIT_GEN_DIR}"/events_*.jsonl 2>/dev/null | head -1 || true)"
fi
if [[ -z "${AUDIT}" || ! -f "${AUDIT}" ]]; then
  echo "[FATAL] could not obtain a SpecA action-audit for the mask search." >&2
  exit 1
fi
echo "  audit: ${AUDIT}"

mkdir -p "$(dirname "${MANIFEST}")"
python scripts/build_covr_manifest.py "${AUDIT}" \
    --method teacache-mask --output "${MANIFEST}" \
    --template-count "${TEMPLATE_COUNT}" --num-layers 28 \
    --mandatory-prefix 3 --max-taylor-gap 5 --num-steps "${NUM_STEPS}"

# Enumerate arm IDs from the manifest (no jq dependency).
mapfile -t ARM_IDS < <(python - "${MANIFEST}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for s in m["strategies"]:
    print(s["strategy_id"])
PY
)
echo "  arms: ${ARM_IDS[*]}"

echo ""
echo "########## [B1] Forcing each equal-FLOPs arm (no bandit) ##########"
for arm in "${ARM_IDS[@]}"; do
  echo "  --- forced arm: ${arm} ---"
  run_main "${EQF_DIR}/arm_${arm}" \
      --covr-strategy-manifest "${MANIFEST}" \
      --covr-force-strategy-id "${arm}"
done

# Noise floor: rerun ONE arm (the baseline) under 2 alternate seeds so the
# analyzer can measure same-arm FID spread. Arm-to-arm differences must beat
# this to count as real.
BASELINE_ARM="$(python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["baseline_strategy_id"])
PY
)"
echo ""
echo "########## [B1] Noise-floor replicas (arm=${BASELINE_ARM}, seeds +1/+2) ##########"
for delta in 1 2; do
  s=$(( SEED + delta ))
  echo "  --- noise replica seed=${s} ---"
  rm -rf "${EQF_DIR}/noise_${s}"
  python main.py --model dit --task c2i --dataset imagenet \
      --method teacache --metrics fid is latency flops speed \
      --covr-strategy-manifest "${MANIFEST}" \
      --covr-force-strategy-id "${BASELINE_ARM}" \
      --seed "${s}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}" \
      --output_dir "${EQF_DIR}/noise_${s}"
done

# =============================================================================
# Verdict
# =============================================================================
echo ""
echo "########## Analysis / go-no-go ##########"
python scripts/analyze_teacache_sweeps.py "${OUT_DIR}"

echo ""
echo "outputs under: ${OUT_DIR}"
echo "  threshold sweep : ${THRESH_DIR}/thresh_*/results.json"
echo "  equal-FLOPs arms: ${EQF_DIR}/arm_*/results.json"
echo "  noise replicas  : ${EQF_DIR}/noise_*/results.json"
echo "  mask manifest   : ${MANIFEST}"
