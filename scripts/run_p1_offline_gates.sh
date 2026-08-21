#!/usr/bin/env bash
# =============================================================================
# P1 offline gates, one shot (GPU box).
# Spec: .claude/covr_quality_constrained_budget_20260820.md §3/§5
#
# Runs: conf extraction -> Gate [P1-a] -> costs -> Gate [P1-b] ->
#       Gate [P1-c] FLOPs leg -> Gate [P1-c] quality leg -> verdict summary.
#
# Analysis only: reads existing PNGs/results.json, never generates images.
# The only GPU use is InceptionV3 forwards (conf extraction + torch-fidelity).
# Everything lands in OUT_DIR (default output/p1_offline_gates) incl. a log.
#
# Env overrides:
#   PROBE_DIR      (output/covr_budget_probe)        budget probe run root;
#                  missing -> degrade to Gate [P1-a] only
#   K8_STATIC_DIR  (output/covr_static_k8_random)    K8 static-mask run root
#   OUT_DIR        (output/p1_offline_gates)    artifacts + logs
#   ARM_REGEX      (auto)   [P1-a] offset-0 arm filter
#   REAL_DIR       (auto: <PROBE_DIR>/<ref arm>/real_299)
#   DEVICE (cuda)  BATCH (64)  PYTHON (python3)
#
# Interpretation discipline (spec §3): 三个 GATE 行全 PASS 才规划 GPU 闭环
# 真跑；任一 FAIL → 停，不调参挽救。conf CSV 已存在时跳过重提取（安全重跑）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || PYTHON=python
PROBE_DIR="${PROBE_DIR:-output/covr_budget_probe}"
K8_STATIC_DIR="${K8_STATIC_DIR:-output/covr_static_k8_random}"
OUT_DIR="${OUT_DIR:-output/p1_offline_gates}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-64}"

REF_ARM="k8/equalflops/arm_pattern_uniform"
K6_ARM="k6/equalflops/arm_pattern_back_loaded"
K4_ARM="k4/equalflops/arm_pattern_uniform"

fail() { echo "ERROR: $*" >&2; exit 2; }

# --- preflight: scripts must be the 2026-08-21 toolchain ---------------------
for f in scripts/extract_inception_conf.py scripts/analyze_conf_rank_validity.py \
         scripts/simulate_conf_budget_controller.py scripts/build_conf_costs.py \
         scripts/compute_mixture_fid.py; do
  [ -f "$f" ] || fail "missing $f — sync the repo first (P1 scripts updated 2026-08-21)"
done
grep -q "mixture-out" scripts/simulate_conf_budget_controller.py \
  || fail "simulate_conf_budget_controller.py is the pre-2026-08-21 version — sync the repo"
grep -q "PASS-FID-ONLY" scripts/analyze_conf_rank_validity.py \
  || fail "analyze_conf_rank_validity.py is the pre-2026-08-21 version — sync the repo"

# --- preflight: data dirs ----------------------------------------------------
if [ ! -d "${K8_STATIC_DIR}" ]; then
  echo "K8 static run dir not found: ${K8_STATIC_DIR}" >&2
  echo "output/ contains:" >&2; ls output/ >&2 || true
  fail "set K8_STATIC_DIR=..."
fi
RUN_PROBE=1
if [ ! -d "${PROBE_DIR}" ]; then
  echo "WARNING: budget probe dir not found: ${PROBE_DIR}"
  echo "  -> running Gate [P1-a] only. [P1-b]/[P1-c] need the budget-probe"
  echo "     k8/k6/k4 arms (sweep_budget_probe.sh output); locate that run"
  echo "     or regenerate it, then rerun with PROBE_DIR=..."
  RUN_PROBE=0
else
  for arm in "${REF_ARM}" "${K6_ARM}" "${K4_ARM}"; do
    [ -f "${PROBE_DIR}/${arm}/results.json" ] \
      || fail "missing ${PROBE_DIR}/${arm}/results.json — probe layout differs, check arm paths"
  done
fi

# [P1-a] arm filter: static runs with random nulls use budget-qualified IDs
# (arm_pattern_uniform_k8); fall back to unqualified if that's what exists.
if compgen -G "${K8_STATIC_DIR}/*/equalflops/arm_pattern_*_k8" >/dev/null; then
  ARM_BASE='arm_pattern_[a-z0-9_]+_k8'
else
  ARM_BASE='arm_pattern_[a-z0-9_]+'
fi
ARM_REGEX="${ARM_REGEX:-${ARM_BASE}\$}"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "== P1 offline gates =="
echo "  probe:  ${PROBE_DIR}"
echo "  static: ${K8_STATIC_DIR}"
echo "  out:    ${OUT_DIR}   log: ${LOG}"
echo "  [P1-a] arm regex: ${ARM_REGEX}"

# --- 0) conf extraction (the only heavy step; resumable) ---------------------
CONF_STATIC="${OUT_DIR}/conf_k8static.csv"
CONF_PROBE="${OUT_DIR}/conf_probe.csv"
if [ -f "${CONF_STATIC}" ]; then echo "[skip] ${CONF_STATIC} exists"; else
  "${PYTHON}" scripts/extract_inception_conf.py "${K8_STATIC_DIR}" \
      "${CONF_STATIC}" --device "${DEVICE}" --batch "${BATCH}"
fi
if [ "${RUN_PROBE}" = "1" ]; then
  if [ -f "${CONF_PROBE}" ]; then echo "[skip] ${CONF_PROBE} exists"; else
    "${PYTHON}" scripts/extract_inception_conf.py "${PROBE_DIR}" \
        "${CONF_PROBE}" --device "${DEVICE}" --batch "${BATCH}"
  fi
fi

# --- 1) Gate [P1-a]: 6 mask arms at offset 0 + free per-offset replications --
echo; echo "######## Gate [P1-a] (offset 0) ########"
"${PYTHON}" scripts/analyze_conf_rank_validity.py "${CONF_STATIC}" \
    --results-root "${K8_STATIC_DIR}" --arm-regex "${ARM_REGEX}" \
    | tee "${OUT_DIR}/gate_p1a.txt"
for r in 1 2 3 4; do
  if compgen -G "${K8_STATIC_DIR}/*/equalflops/arm_pattern_*/rep_${r}" >/dev/null; then
    echo; echo "-- [P1-a] replication, latent offset ${r} (informational) --"
    "${PYTHON}" scripts/analyze_conf_rank_validity.py "${CONF_STATIC}" \
        --results-root "${K8_STATIC_DIR}" \
        --arm-regex "${ARM_BASE}/rep_${r}\$" \
        | tee "${OUT_DIR}/gate_p1a_rep${r}.txt"
  fi
done

# --- 2) costs (k6 = measured best arm back_loaded) + Gate [P1-b] -------------
if [ "${RUN_PROBE}" = "0" ]; then
  echo; echo "######## VERDICT SUMMARY (partial: [P1-a] only) ########"
  grep -h "^GATE" "${OUT_DIR}/gate_p1a.txt" || true
  echo "([P1-b]/[P1-c] skipped: budget probe run not found — locate the"
  echo " k8/k6/k4 sweep output or regenerate, then rerun with PROBE_DIR=...)"
  exit 0
fi

echo; echo "######## costs.json ########"
"${PYTHON}" scripts/build_conf_costs.py "${PROBE_DIR}" \
    --arm "8=${REF_ARM}" --arm "6=${K6_ARM}" --arm "4=${K4_ARM}" \
    -o "${OUT_DIR}/costs.json"

echo; echo "######## conf separation (summary; Cohen's d) ########"
"${PYTHON}" scripts/simulate_conf_budget_controller.py "${CONF_PROBE}" summary \
    --reference "${REF_ARM}" --headroom-arm "${K6_ARM}" \
    | tee "${OUT_DIR}/summary.txt"

echo; echo "######## Gate [P1-b] (power) ########"
"${PYTHON}" scripts/simulate_conf_budget_controller.py "${CONF_PROBE}" power \
    --reference "${REF_ARM}" | tee "${OUT_DIR}/gate_p1b.txt"

H_STAR="$(sed -n 's/.*h=[[:space:]]*\([0-9.]*\): .*<- selected.*/\1/p' \
    "${OUT_DIR}/gate_p1b.txt" | head -n1)"
if [ -z "${H_STAR}" ]; then
  echo "WARNING: power selected no h (FA gate unmet -> [P1-b] FAIL); "
  echo "         running simulate with h=5.0 for the record only"
  H_STAR="5.0"
fi
echo "h* = ${H_STAR}"

# --- 3) Gate [P1-c]: closed-loop sim (FLOPs leg) -> mixture FID (quality) ----
echo; echo "######## Gate [P1-c] FLOPs leg (simulate) ########"
"${PYTHON}" scripts/simulate_conf_budget_controller.py "${CONF_PROBE}" simulate \
    --reference "${REF_ARM}" --costs "${OUT_DIR}/costs.json" --h "${H_STAR}" \
    --mixture-out "${OUT_DIR}/plan.json" | tee "${OUT_DIR}/gate_p1c_flops.txt"

echo; echo "######## Gate [P1-c] quality leg (mixture FID) ########"
if ! "${PYTHON}" scripts/compute_mixture_fid.py "${OUT_DIR}/plan.json" \
    --run-root "${PROBE_DIR}" --workdir "${OUT_DIR}/mixture_299" \
    ${REAL_DIR:+--real-dir "${REAL_DIR}"} \
    | tee "${OUT_DIR}/gate_p1c_quality.txt"; then
  echo "quality leg did not run (infra error above, e.g. real_299/torch);"
  echo "fix and rerun this script — conf CSVs and plan.json are reused"
fi

# --- verdict summary ----------------------------------------------------------
echo; echo "######## VERDICT SUMMARY ########"
grep -h "^GATE" \
    "${OUT_DIR}/gate_p1a.txt" "${OUT_DIR}/gate_p1b.txt" \
    "${OUT_DIR}/gate_p1c_flops.txt" "${OUT_DIR}/gate_p1c_quality.txt" || true
echo "(spec §3: 三 gate 全 PASS 才规划 GPU 闭环真跑；任一 FAIL → 停，不调参挽救)"
echo "artifacts: ${OUT_DIR}/{conf_*.csv,costs.json,plan.json,gate_*.txt,summary.txt}"
