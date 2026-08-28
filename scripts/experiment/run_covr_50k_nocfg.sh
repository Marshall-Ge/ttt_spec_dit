#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# COVR Bandit 50k 无 CFG 实验
# ============================================================================
# 三组实验分散到 3 张 GPU 并行：baseline / SpecA / COVR Bandit
# 无 CFG (guidance_scale=1.0)，50k ImageNet，DiT-2-256
#
# 优化策略：
#   - safety_sample_rate=0 (禁用 safety shadow，消除 22% FLOPs 开销)
#   - sentinel_rate=0.02 (降低 sentinel 频率，减少 full rollout)
#   - safety_chain_threshold=5 (短 Taylor chain 不采样)
#
# 用法:
#   GPUS=0,1,2 bash scripts/run_covr_50k_nocfg.sh
#
# 前台模式:
#   FOREGROUND=1 GPUS=0,1,2 bash scripts/run_covr_50k_nocfg.sh
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/covr_50k_nocfg/${RUN_ID}}"
MANIFEST="${MANIFEST:-/tmp/manifest.json}"

# Defaults
GPUS="${GPUS:-0,1,2}"
N_PROMPTS="${N_PROMPTS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="1.0"
TEMPLATE_ID="${TEMPLATE_ID:-timestep_prior}"
BANDIT_EPSILON="${BANDIT_EPSILON:-0.1}"
SESSION_ID="covr-50k-nocfg-${RUN_ID}"

# ---- Background launch logic ----
if [[ "${_COVR_50K_WORKER:-}" != "1" && "${FOREGROUND:-0}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  NOHUP_LOG="${OUTPUT_ROOT}/logs/_main.log"
  echo ""
  echo "  ┌─────────────────────────────────────────────────────────────┐"
  echo "  │  COVR 50k No-CFG Experiment — launched in background        │"
  echo "  └─────────────────────────────────────────────────────────────┘"
  echo ""
  echo "  Config: 50k images, guidance_scale=1.0, DiT-2-256, 50 steps"
  echo "  Experiments: baseline / SpecA / COVR Bandit (speed-optimized)"
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
  _COVR_50K_WORKER=1 \
    nohup bash "${BASH_SOURCE[0]}" "$@" \
    >> "${NOHUP_LOG}" 2>&1 &
  BGPID=$!
  echo "${BGPID}" > "${OUTPUT_ROOT}/logs/_main.pid"
  echo "  PID: ${BGPID}"
  echo ""
  disown "${BGPID}"
  exit 0
fi

# ---- Worker starts here ----
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ ${#GPU_IDS[@]} -lt 3 ]]; then
  echo "[ERROR] Need at least 3 GPUs. Got: ${GPUS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
MAIN_LOG="${OUTPUT_ROOT}/logs/_main.log"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "${MAIN_LOG}"; }

log "============================================="
log "COVR 50k No-CFG Experiment"
log "============================================="
log "GPUs: ${GPUS}"
log "N_PROMPTS: ${N_PROMPTS}, BS: ${BATCH_SIZE}, Steps: ${NUM_STEPS}"
log "Guidance Scale: ${GUIDANCE_SCALE} (No CFG)"
log "Manifest: ${MANIFEST}"
log "Output: ${OUTPUT_ROOT}"
log ""

# Common args
COMMON_ARGS=(
  --model dit --task c2i --dataset imagenet
  --metrics fid is flops speed
  --seed "${SEED}"
  --num_steps "${NUM_STEPS}"
  --n_prompts "${N_PROMPTS}"
  --batch_size "${BATCH_SIZE}"
  --guidance_scale "${GUIDANCE_SCALE}"
)

# ============================================================================
# Experiment 1: Baseline (full DDIM, no acceleration)
# ============================================================================
EXP_BASELINE_DIR="${OUTPUT_ROOT}/baseline"
EXP_BASELINE_ARGS=(
  "${COMMON_ARGS[@]}"
  --method baseline
  --output_dir "${EXP_BASELINE_DIR}"
)

# ============================================================================
# Experiment 2: Adaptive SpecA (no COVR)
# ============================================================================
EXP_SPECA_DIR="${OUTPUT_ROOT}/speca"
EXP_SPECA_ARGS=(
  "${COMMON_ARGS[@]}"
  --method speca
  --output_dir "${EXP_SPECA_DIR}"
)

# ============================================================================
# Experiment 3: COVR Bandit (speed-optimized)
#   - safety_sample_rate=0: 完全禁用 safety shadow (省 22% FLOPs)
#   - sentinel_rate=0.02: 低频 sentinel (仅 2% trajectory 做 terminal fidelity)
#   - chain_threshold=5: 短 chain 不采样
# ============================================================================
EXP_BANDIT_DIR="${OUTPUT_ROOT}/covr_bandit"
EXP_BANDIT_ARGS=(
  "${COMMON_ARGS[@]}"
  --method speca
  --covr-template-bandit
  --covr-template-manifest "${MANIFEST}"
  --covr-bandit-epsilon "${BANDIT_EPSILON}"
  --covr-session-id "${SESSION_ID}"
  --covr-safety-sample-rate 0.0
  --covr-sentinel-rate 0.02
  --covr-sentinel-horizon 0
  --covr-safety-chain-threshold 5
  --covr-profile-stages
  --output_dir "${EXP_BANDIT_DIR}"
)

# ============================================================================
# Launch all 3 experiments in parallel
# ============================================================================
PIDS=()
NAMES=("baseline" "speca" "covr_bandit")

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

log "Launching 3 experiments..."
launch "baseline"    "${GPU_IDS[0]}" "${EXP_BASELINE_ARGS[@]}"
CLEANUP_PIDS+=("${PIDS[-1]}")
launch "speca"       "${GPU_IDS[1]}" "${EXP_SPECA_ARGS[@]}"
CLEANUP_PIDS+=("${PIDS[-1]}")
launch "covr_bandit" "${GPU_IDS[2]}" "${EXP_BANDIT_ARGS[@]}"
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
    log "  [${name}] done"
  else
    log "  [${name}] FAILED (exit=$?) — check logs/${name}.log"
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
log "RESULTS"
log "============================================="

python3 - "${OUTPUT_ROOT}" << 'PYTHON_EOF'
import json, sys
from pathlib import Path

output_root = Path(sys.argv[1])
experiments = {
    "Baseline": "baseline",
    "Adaptive SpecA": "speca",
    "COVR Bandit": "covr_bandit",
}

results = {}
for label, dirname in experiments.items():
    rpath = output_root / dirname / "results.json"
    if not rpath.exists():
        print(f"  [{label}]: MISSING")
        continue
    with open(rpath) as f:
        raw = json.load(f)
    data = raw.get("aggregate", raw)
    results[label] = data

if not results:
    print("\n[ERROR] No results found.")
    sys.exit(1)

print()
print("=" * 80)
print(f"  COVR 50k No-CFG Results (guidance_scale=1.0, DiT-2-256, 50 steps)")
print("=" * 80)
print()
print(f"  {'Method':<20} {'FID':>8} {'IS':>8} {'FLOPs(T)':>10} "
      f"{'img/s':>8} {'cand img/s':>12} {'skip%':>7}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*12} {'-'*7}")

for label, data in results.items():
    fid = data.get("fid", data.get("FID", "—"))
    is_score = data.get("is_mean", data.get("IS", data.get("inception_score_mean", "—")))
    flops = data.get("flops_candidate_T", data.get("flops_accel_T",
            data.get("flops_vanilla_T", 0)))
    speed = data.get("speed_img_per_s", 0)
    cand_speed = data.get("speed_candidate_img_per_s", speed)
    skip = data.get("speca_skip_ratio", data.get("skip_ratio", "—"))

    fid_s = f"{fid:.2f}" if isinstance(fid, (int, float)) else str(fid)
    is_s = f"{is_score:.1f}" if isinstance(is_score, (int, float)) else str(is_score)
    flops_s = f"{flops:.3f}" if isinstance(flops, (int, float)) else str(flops)
    speed_s = f"{speed:.2f}" if isinstance(speed, (int, float)) else str(speed)
    cand_s = f"{cand_speed:.2f}" if isinstance(cand_speed, (int, float)) else str(cand_speed)
    skip_s = f"{skip*100:.1f}%" if isinstance(skip, (int, float)) else str(skip)

    print(f"  {label:<20} {fid_s:>8} {is_s:>8} {flops_s:>10} "
          f"{speed_s:>8} {cand_s:>12} {skip_s:>7}")

print()
print("=" * 80)

# Profile breakdown for bandit
bandit_data = results.get("COVR Bandit", {})
profile = bandit_data.get("generation_profile", {})
stage_totals = profile.get("stage_total_s", {})
if stage_totals:
    print()
    print("  COVR Bandit Stage Profiling (top stages):")
    sorted_stages = sorted(stage_totals.items(), key=lambda x: -x[1])
    for stage, secs in sorted_stages[:8]:
        if secs > 0.1:
            print(f"    {stage:<40} {secs:>8.1f}s")
    print()

# Safety/terminal breakdown
safety_s = bandit_data.get("wall_s_safety_total", 0)
terminal_s = bandit_data.get("wall_s_terminal_total", 0)
control_s = bandit_data.get("wall_s_control_total", 0)
if any([safety_s, terminal_s, control_s]):
    print(f"  COVR overhead: safety={safety_s:.1f}s, terminal={terminal_s:.1f}s, "
          f"control={control_s:.1f}s")
    print()

# Save comparison table
summary = {"config": {"guidance_scale": 1.0, "n_prompts": 50000, "num_steps": 50,
                      "model": "dit", "note": "no CFG"}}
for label, data in results.items():
    summary[label] = {k: v for k, v in data.items()
                      if any(x in k for x in ["fid", "is", "flops", "speed", "wall", "skip"])}
summary_path = output_root / "comparison_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Summary saved: {summary_path}")
PYTHON_EOF

log ""
log "All outputs: ${OUTPUT_ROOT}"
log "Done."
