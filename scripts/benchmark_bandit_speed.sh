#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# COVR Bandit Speed Bottleneck Diagnosis
# ============================================================================
# 4 个对照实验分散到多张 GPU 并行执行，完成后自动分析 candidate img/s 差异。
# 默认以 nohup 后台方式启动，通过 tail -f 查看日志。
#
# 用法:
#   GPUS=0,1,2,3 bash scripts/benchmark_bandit_speed.sh
#
# 前台模式:
#   FOREGROUND=1 GPUS=0,1,2,3 MANIFEST=... bash scripts/benchmark_bandit_speed.sh
#
# 环境变量 (均可覆盖):
#   GPUS              — 逗号分隔的 GPU ID (需要至少 4 张; 默认 0,1,2,3)
#   MANIFEST          — COVR template manifest 路径 (必须)
#   N_PROMPTS         — 生成图片数 (默认 2048)
#   BATCH_SIZE        — batch size (默认 32)
#   NUM_STEPS         — denoising steps (默认 50)
#   TEMPLATE_ID       — forced template id (默认 timestep_prior)
#   BANDIT_EPSILON    — bandit exploration (默认 0.1)
#   FOREGROUND        — 设为 1 则前台运行 (默认后台 nohup)
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/bandit_speed_diag/${RUN_ID}}"

# Manifest (defaults to tmp/manifest.json in project root)
MANIFEST="${MANIFEST:-${ROOT_DIR}/tmp/manifest.json}"

# ---- Background launch logic ----
# If not already the worker and FOREGROUND!=1, re-exec self via nohup
if [[ "${_BANDIT_SPEED_WORKER:-}" != "1" && "${FOREGROUND:-0}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  NOHUP_LOG="${OUTPUT_ROOT}/logs/_main.log"
  echo ""
  echo "  ┌─────────────────────────────────────────────────────────────┐"
  echo "  │  COVR Bandit Speed Diagnosis — launched in background       │"
  echo "  └─────────────────────────────────────────────────────────────┘"
  echo ""
  echo "  Output:  ${OUTPUT_ROOT}"
  echo "  Log:     ${NOHUP_LOG}"
  echo ""
  echo "  Monitor:"
  echo "    tail -f ${NOHUP_LOG}"
  echo ""
  echo "  Stop:"
  echo "    kill \$(cat ${OUTPUT_ROOT}/logs/_main.pid)"
  echo ""
  export OUTPUT_ROOT MANIFEST RUN_ID
  _BANDIT_SPEED_WORKER=1 \
    nohup bash "${BASH_SOURCE[0]}" "$@" \
    >> "${NOHUP_LOG}" 2>&1 &
  BGPID=$!
  echo "${BGPID}" > "${OUTPUT_ROOT}/logs/_main.pid"
  echo "  PID: ${BGPID}"
  echo ""
  disown "${BGPID}"
  exit 0
fi

# Defaults
GPUS="${GPUS:-0,1,2,3}"
N_PROMPTS="${N_PROMPTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
TEMPLATE_ID="${TEMPLATE_ID:-timestep_prior}"
BANDIT_EPSILON="${BANDIT_EPSILON:-0.1}"
SESSION_ID="speed-diag-${RUN_ID}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ ${#GPU_IDS[@]} -lt 4 ]]; then
  echo "[ERROR] Need at least 4 GPUs. Got: ${GPUS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
MAIN_LOG="${OUTPUT_ROOT}/logs/_main.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

# Common args
COMMON_ARGS=(
  --model dit --task c2i --dataset imagenet
  --method speca
  --metrics fid is flops speed
  --seed "${SEED}"
  --num_steps "${NUM_STEPS}"
  --n_prompts "${N_PROMPTS}"
  --batch_size "${BATCH_SIZE}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --covr-profile-stages
)

log "============================================="
log "COVR Bandit Speed Bottleneck Diagnosis"
log "============================================="
log "GPUs: ${GPUS}"
log "Manifest: ${MANIFEST}"
log "N_PROMPTS: ${N_PROMPTS}, BS: ${BATCH_SIZE}, Steps: ${NUM_STEPS}"
log "Output: ${OUTPUT_ROOT}"
log ""

# ============================================================================
# Experiment A: Bandit + full safety/sentinel (复现慢速)
# ============================================================================
EXP_A_DIR="${OUTPUT_ROOT}/A_bandit_full_safety"
EXP_A_ARGS=(
  "${COMMON_ARGS[@]}"
  --covr-template-bandit
  --covr-template-manifest "${MANIFEST}"
  --covr-bandit-epsilon "${BANDIT_EPSILON}"
  --covr-safety-sample-rate 0.1
  --covr-sentinel-rate 0.05
  --covr-sentinel-horizon 0
  --covr-session-id "${SESSION_ID}-A"
  --output_dir "${EXP_A_DIR}"
)

# ============================================================================
# Experiment B: Bandit + 禁用 safety/sentinel (隔离 dispatch 开销)
# ============================================================================
EXP_B_DIR="${OUTPUT_ROOT}/B_bandit_no_safety"
EXP_B_ARGS=(
  "${COMMON_ARGS[@]}"
  --covr-template-bandit
  --covr-template-manifest "${MANIFEST}"
  --covr-bandit-epsilon "${BANDIT_EPSILON}"
  --covr-safety-sample-rate 0.0
  --covr-sentinel-rate 0.0
  --covr-session-id "${SESSION_ID}-B"
  --output_dir "${EXP_B_DIR}"
)

# ============================================================================
# Experiment C: Forced template (同 refresh_mask, 无 bandit 选择循环)
# ============================================================================
EXP_C_DIR="${OUTPUT_ROOT}/C_forced_template"
EXP_C_ARGS=(
  "${COMMON_ARGS[@]}"
  --covr-force-template-id "${TEMPLATE_ID}"
  --covr-template-manifest "${MANIFEST}"
  --covr-safety-sample-rate 0.0
  --covr-sentinel-rate 0.0
  --output_dir "${EXP_C_DIR}"
)

# ============================================================================
# Experiment D: Adaptive SpecA baseline (无 COVR)
# ============================================================================
EXP_D_DIR="${OUTPUT_ROOT}/D_adaptive_speca"
EXP_D_ARGS=(
  "${COMMON_ARGS[@]}"
  --output_dir "${EXP_D_DIR}"
)

# ============================================================================
# Launch all 4 experiments in parallel
# ============================================================================
PIDS=()
NAMES=("A_bandit_full_safety" "B_bandit_no_safety" "C_forced_template" "D_adaptive_speca")

launch() {
  local name="$1" gpu="$2"
  shift 2
  local logfile="${OUTPUT_ROOT}/logs/${name}.log"
  log "  [${name}] → GPU ${gpu} (log: logs/${name}.log)"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" python main.py "$@" >"${logfile}" 2>&1
  ) &
  PIDS+=("$!")
}

CLEANUP_PIDS=()
cleanup() {
  log "SIGINT/SIGTERM — killing children..."
  for pid in "${CLEANUP_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  exit 130
}
trap cleanup INT TERM

log "Launching 4 experiments..."
launch "A_bandit_full_safety" "${GPU_IDS[0]}" "${EXP_A_ARGS[@]}"
CLEANUP_PIDS+=("${PIDS[-1]}")
launch "B_bandit_no_safety"   "${GPU_IDS[1]}" "${EXP_B_ARGS[@]}"
CLEANUP_PIDS+=("${PIDS[-1]}")
launch "C_forced_template"    "${GPU_IDS[2]}" "${EXP_C_ARGS[@]}"
CLEANUP_PIDS+=("${PIDS[-1]}")
launch "D_adaptive_speca"     "${GPU_IDS[3]}" "${EXP_D_ARGS[@]}"
CLEANUP_PIDS+=("${PIDS[-1]}")

log ""
log "PIDs: ${PIDS[*]}"
log "Waiting for all experiments to finish..."
log ""

FAILED=0
for idx in "${!PIDS[@]}"; do
  name="${NAMES[idx]}"
  pid="${PIDS[idx]}"
  if wait "${pid}"; then
    log "  [${name}] ✓ done"
  else
    log "  [${name}] ✗ FAILED (exit=$?) — check logs/${name}.log"
    FAILED=1
  fi
done
trap - INT TERM

if [[ ${FAILED} -ne 0 ]]; then
  log ""
  log "WARNING: some experiments failed. Analyzing available results..."
fi

# ============================================================================
# Auto-analysis
# ============================================================================
log ""
log "============================================="
log "ANALYSIS"
log "============================================="

ANALYSIS_SCRIPT="${OUTPUT_ROOT}/analysis.py"
cat > "${ANALYSIS_SCRIPT}" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Analyze COVR bandit speed diagnosis results."""
import json, sys, os
from pathlib import Path

output_root = Path(sys.argv[1])
experiments = {
    "A": ("A_bandit_full_safety",  "Bandit + safety + sentinel"),
    "B": ("B_bandit_no_safety",    "Bandit (no safety/sentinel)"),
    "C": ("C_forced_template",     "Forced template (no bandit loop)"),
    "D": ("D_adaptive_speca",      "Adaptive SpecA (no COVR)"),
}

results = {}
for key, (dirname, label) in experiments.items():
    rpath = output_root / dirname / "results.json"
    if not rpath.exists():
        print(f"  [{key}] {label}: MISSING (no results.json)")
        continue
    with open(rpath) as f:
        data = json.load(f)
    results[key] = data

if not results:
    print("\n[ERROR] No results found. Check experiment logs.")
    sys.exit(1)

# Extract key metrics
print("\n" + "=" * 78)
print(f"{'Exp':<4} {'Config':<38} {'img/s':>8} {'cand img/s':>11} "
      f"{'FLOPs(T)':>9} {'safety_s':>9}")
print("-" * 78)

for key in "ABCD":
    if key not in results:
        continue
    data = results[key]
    label = experiments[key][1]

    # Speed: try multiple locations
    overall_speed = data.get("speed_img_per_s", 0)
    candidate_speed = data.get("speed_candidate_img_per_s", overall_speed)

    # FLOPs
    flops_T = data.get("flops_accel_T", data.get("flops_candidate_T", 0))

    # Safety wall time
    safety_total = data.get("wall_s_safety_total", 0)

    print(f"  {key}   {label:<38} {overall_speed:>7.2f}  {candidate_speed:>10.2f} "
          f" {flops_T:>8.3f}  {safety_total:>8.1f}s")

print("=" * 78)

# Diagnosis
print("\n--- Bottleneck Diagnosis ---\n")

d_speed = results.get("D", {}).get("speed_img_per_s", 0)
a_cand = results.get("A", {}).get("speed_candidate_img_per_s",
         results.get("A", {}).get("speed_img_per_s", 0))
b_cand = results.get("B", {}).get("speed_candidate_img_per_s",
         results.get("B", {}).get("speed_img_per_s", 0))
c_speed = results.get("C", {}).get("speed_img_per_s", 0)

if "A" in results and "B" in results and b_cand > 0:
    safety_impact = (b_cand - a_cand) / b_cand * 100
    print(f"1. Safety/Sentinel overhead: B vs A")
    print(f"   B (no safety) = {b_cand:.2f} img/s")
    print(f"   A (with safety) = {a_cand:.2f} img/s")
    if safety_impact > 5:
        print(f"   → Safety shadow causes {safety_impact:.1f}% slowdown ← CONFIRMED BOTTLENECK")
    else:
        print(f"   → Safety shadow impact: {safety_impact:.1f}% (minor)")

    safety_s = results["A"].get("wall_s_safety_total", 0)
    terminal_s = results["A"].get("wall_s_terminal_total", 0)
    if safety_s > 0 or terminal_s > 0:
        print(f"   Safety wall time: {safety_s:.1f}s, Terminal wall time: {terminal_s:.1f}s")
    print()

if "B" in results and "C" in results and c_speed > 0:
    dispatch_impact = (c_speed - b_cand) / c_speed * 100
    print(f"2. Bandit dispatch overhead: C vs B")
    print(f"   C (forced template) = {c_speed:.2f} img/s")
    print(f"   B (bandit, no safety) = {b_cand:.2f} img/s")
    if dispatch_impact > 3:
        print(f"   → Bandit selection/persist causes {dispatch_impact:.1f}% slowdown")
    else:
        print(f"   → Bandit dispatch impact: {dispatch_impact:.1f}% (negligible)")
    print()

if "C" in results and "D" in results and d_speed > 0:
    reinit_impact = (d_speed - c_speed) / d_speed * 100
    print(f"3. Per-trajectory speca_init + profiler sync: D vs C")
    print(f"   D (adaptive, no COVR) = {d_speed:.2f} img/s")
    print(f"   C (forced template) = {c_speed:.2f} img/s")
    if reinit_impact > 3:
        print(f"   → Reinit/profiler overhead: {reinit_impact:.1f}% slowdown")
    else:
        print(f"   → Reinit overhead: {reinit_impact:.1f}% (negligible)")
    print()

if "A" in results and "D" in results and d_speed > 0:
    total_impact = (d_speed - a_cand) / d_speed * 100
    print(f"4. Total bandit slowdown (D vs A): {total_impact:.1f}%")
    print(f"   D = {d_speed:.2f} img/s, A candidate = {a_cand:.2f} img/s")
    print()

# Profile stage breakdown (if available)
for key in "ABCD":
    if key not in results:
        continue
    profile = results[key].get("generation_profile", {})
    stage_totals = profile.get("stage_total_s", {})
    if stage_totals:
        print(f"\n--- [{key}] Stage profiling (total seconds) ---")
        sorted_stages = sorted(stage_totals.items(), key=lambda x: -x[1])
        for stage, secs in sorted_stages[:10]:
            if secs > 0.01:
                print(f"   {stage:<40} {secs:>8.2f}s")

print("\n" + "=" * 78)
print("Done. Full results in each experiment's results.json")

# Save summary
summary = {
    "experiments": {k: experiments[k][1] for k in results},
    "speeds": {},
}
for key in results:
    summary["speeds"][key] = {
        "overall_img_per_s": results[key].get("speed_img_per_s", 0),
        "candidate_img_per_s": results[key].get("speed_candidate_img_per_s", 0),
    }
summary_path = output_root / "speed_diagnosis_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {summary_path}")
PYTHON_EOF

python "${ANALYSIS_SCRIPT}" "${OUTPUT_ROOT}" 2>&1 | tee -a "${MAIN_LOG}"

log ""
log "All outputs: ${OUTPUT_ROOT}"
log "Main log:    ${MAIN_LOG}"
