#!/usr/bin/env bash
# =============================================================================
# Hypothesis A — aggressive-budget probe for COVR-on-TeaCache.
#
# The falsified experiment used calc=13/50, where TeaCache's cache is not yet
# "overdrawn": equal-FLOPs mask arms tied within 2x the noise floor, so the
# bandit had nothing to discover. This script re-runs the SAME equal-FLOPs
# arm-spread test at progressively harsher budgets to find the budget (if any)
# where WHERE the calc steps sit starts to matter.
#
#   for each K in BUDGETS:
#     - force each of 4 equal-FLOPs arms (front / uniform / back / geometric)
#     - rerun the baseline arm at 2 extra seeds  -> per-budget noise floor
#     - hand the pair to scripts/analyze_teacache_sweeps.py for the verdict
#
# Arms come from scripts/build_budget_manifest.py, which synthesizes masks
# combinatorially. The audit-derived --method teacache-mask path cannot go
# below the audit's own refresh cardinality (13), which is what blocked this
# probe before.
#
# DECISION CRITERION (per budget, from the analyzer):
#   arm FID spread > 2 x same-arm noise spread  =>  DIFFER: mask selection has
#     headroom at this budget; hypothesis A survives, proceed to reward repair.
#   otherwise                                   =>  TIE at this budget.
#   Tie at every budget down to the harshest     =>  the teacache same-method
#     equal-FLOPs mask direction is permanently closed.
#
# Usage:
#   bash scripts/sweep_budget_probe.sh
# Env overrides:
#   BUDGETS ("8,6,4")  N_PROMPTS (500)  NUM_STEPS (50)  BATCH_SIZE (32)
#   SEED (42)  GUIDANCE (4.5)  OUT_DIR (/tmp/covr_budget_probe)
#   REWARD_MODE (terminal|hstep, default terminal)  SENTINEL_RATE (1.0)
#   SENTINEL_HORIZON (hstep mode only, default 10)  SESSION_ID (auto)
#   REFERENCE (1|0, default 1)  REFERENCE_DIR (default <OUT_DIR>/reference)
#   IMG_SAVE_LIMIT (default N_PROMPTS = save every image; per-image crossover
#       analysis needs the full set, the old default of 50 is not enough)
#
# Per-image crossover (COVR hypothesis for the bandit decision):
#   Global FID cannot tell "one arm is best for every image" (pick it offline,
#   bandit unnecessary) from "arms flip per image" (the only case a
#   per-trajectory bandit can exploit). This sweep now saves ALL generated
#   PNGs per run (--img_save_limit N_PROMPTS) and runs one shared full-compute
#   reference (--method baseline, same seed/window) once; scripts/analyze_
#   crossover.py pairs every arm's PNG against the reference PNG at the same
#   global_idx (same image, same seed) and reports whether crossover is real,
#   utilizable, or buried in the 8-bit quantization floor. The reference costs
#   ~NUM_STEPS/mean(K) arm-runs (printed below) and is opt-out via REFERENCE=0.
#
# Sentinel / reward telemetry (COVR hypothesis C — does the reward proxy
# rank arms like FID does?):
#   REWARD_MODE=terminal -> --covr-sentinel-horizon 0; cost per sentinel
#       trajectory = 1 extra full forward at the last step. NOTE: at K=8 the
#       uniform/back_loaded/geometric arms calc the last step, making the
#       accelerated path structurally identical to the shadow full (MSE ~ 0);
#       this is the structural-degeneration hypothesis the analyzer checks.
#   REWARD_MODE=hstep   -> --covr-sentinel-horizon ${SENTINEL_HORIZON}; cost
#       per sentinel trajectory = H extra full forwards from a random start.
#   SENTINEL_RATE defaults to 1.0 because the reward-vs-FID ranking needs a
#       reward estimate for EVERY arm, and per-arm means over only
#       ceil(N_PROMPTS/BATCH_SIZE) ~ 16 trajectories would drown in noise at
#       the default 0.05. The cost estimate printed below scales linearly
#       with the rate you set.
#
# Session identity (verified against run_dit.py): the default session id is
# timestamped (run_dit.py:1754), and _covr_hash_sample/_covr_hash_index hash
# (session_id, trajectory_id) — a different session id would make every arm
# sample DIFFERENT sentinel trajectories and H-step rollout starts, which
# would confound the cross-arm reward comparison with "which images got
# sampled". All arms of a budget therefore share SESSION_ID (auto-generated
# once per budget unless you set it). Noise replicas get their own
# _noise_<s> session id ON PURPOSE: they rerun the SAME arm at seeds +1/+2,
# where ImageNetDataset(n_images, seed) shifts the image pool — hashing a
# noise replica against the forced-arm session id would select sentinels on
# images that replica never generates, which is the wrong noise floor for
# "same-arm FID spread across seeds". Only runs that share both seed and
# session id are directly comparable.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

BUDGETS="${BUDGETS:-8,6,4}"
N_PROMPTS="${N_PROMPTS:-500}"
NUM_STEPS="${NUM_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
GUIDANCE="${GUIDANCE:-4.5}"
OUT_DIR="${OUT_DIR:-/tmp/covr_budget_probe}"
REWARD_MODE="${REWARD_MODE:-terminal}"
SENTINEL_RATE="${SENTINEL_RATE:-1.0}"
SENTINEL_HORIZON="${SENTINEL_HORIZON:-10}"
SESSION_ID="${SESSION_ID:-}"

# --- full-compute reference + per-image PNG saving (see header note) ---
REFERENCE="${REFERENCE:-1}"            # 1|0 — run the full-compute reference
REFERENCE_DIR="${REFERENCE_DIR:-}"     # existing reference dir to reuse (default: OUT_DIR/reference)
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-}"   # PNGs saved per run (default: N_PROMPTS = all)

case "${REWARD_MODE}" in
  terminal) SENTINEL_HORIZON=0 ;;
  hstep)    SENTINEL_HORIZON="${SENTINEL_HORIZON}" ;;
  *)
    echo "REWARD_MODE must be 'terminal' or 'hstep' (got '${REWARD_MODE}')" >&2
    exit 2
    ;;
esac
if ! python - "$SENTINEL_RATE" <<'PY' >/dev/null
import sys
rate = float(sys.argv[1])
assert 0.0 <= rate <= 1.0, "SENTINEL_RATE must be in [0, 1]"
PY
then
  exit 2
fi
if [ "${REWARD_MODE}" = "hstep" ] && \
   ! python - "$SENTINEL_HORIZON" "$NUM_STEPS" <<'PY' >/dev/null
import sys
h, n = int(sys.argv[1]), int(sys.argv[2])
assert 0 < h < n, "SENTINEL_HORIZON must be in (0, NUM_STEPS)"
PY
then
  exit 2
fi

if [ "${REFERENCE}" != "0" ] && [ "${REFERENCE}" != "1" ]; then
  echo "REFERENCE must be 0 or 1 (got '${REFERENCE}')" >&2
  exit 2
fi
: "${IMG_SAVE_LIMIT:=${N_PROMPTS}}"
if ! python - "$IMG_SAVE_LIMIT" <<'PY' >/dev/null
import sys
assert int(sys.argv[1]) > 0, "IMG_SAVE_LIMIT must be positive"
PY
then
  exit 2
fi
if [ -z "${REFERENCE_DIR}" ]; then
  REFERENCE_DIR="${OUT_DIR}/reference"
fi

run_arm() {  # run_arm <out> <manifest> <arm> <seed> <session-id>
  local out="$1" manifest="$2" arm="$3" seed="$4" session_id="$5"
  rm -rf "${out}"
  python main.py --model dit --task c2i --dataset imagenet \
      --method teacache --metrics fid is latency flops speed \
      --covr-strategy-manifest "${manifest}" \
      --covr-force-strategy-id "${arm}" \
      --covr-session-id "${session_id}" \
      --covr-sentinel-rate "${SENTINEL_RATE}" \
      --covr-sentinel-horizon "${SENTINEL_HORIZON}" \
      --seed "${seed}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}" \
      --img_save_limit "${IMG_SAVE_LIMIT}" \
      --output_dir "${out}"
}

run_reference() {  # run_reference <out>  — full compute, shared cross-budget anchor
  local out="$1"
  python main.py --model dit --task c2i --dataset imagenet \
      --method baseline --metrics fid is latency flops speed \
      --seed "${SEED}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}" \
      --img_save_limit "${IMG_SAVE_LIMIT}" \
      --output_dir "${out}"
}

# ---- cost / disk statement BEFORE any GPU spend ----
python - "$BUDGETS" "$NUM_STEPS" "$N_PROMPTS" "$IMG_SAVE_LIMIT" "$REFERENCE" <<'PY'
import sys
budgets = [int(k) for k in sys.argv[1].split(",") if k.strip()]
num_steps, n_prompts, img_limit = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
reference = sys.argv[5] == "1"
arm_runs = 6 * len(budgets)                 # 4 arms + 2 noise replicas per budget
arm_units = sum(6 * k for k in budgets)     # each run costs K full forwards/trajectory
ref_units = num_steps if reference else 0
mean_k = sum(budgets) / len(budgets)
print("--- cost / disk budget (before any GPU spend) ---")
print(f"  arm runs: {arm_runs} (4 arms + 2 noise per budget, K={budgets})")
print(f"  forward-units per trajectory: arms={arm_units}, reference={ref_units}"
      f"{'' if reference else ' (REFERENCE=0)'}, total={arm_units + ref_units}")
if reference:
    print(f"  reference = {num_steps / mean_k:.1f} average arm-runs "
          f"({num_steps}/{mean_k:.1f}) and {ref_units}/{arm_units + ref_units} "
          f"= {100.0 * ref_units / (arm_units + ref_units):.1f}% of the sweep's "
          "forward budget")
total_runs = arm_runs + (1 if reference else 0)
total_imgs = total_runs * img_limit
raw_gb = total_imgs * 256 * 256 * 3 / 1e9
print(f"  images per run: min(N_PROMPTS, IMG_SAVE_LIMIT) = {img_limit} "
      f"256x256 RGB PNGs")
print(f"  disk: {total_runs} runs x {img_limit} = {total_imgs} images")
print(f"    raw upper bound {raw_gb:.2f} GB; typical PNG ~0.5-1.0 B/px -> "
      f"{total_imgs * 256 * 256 * 0.5 / 1e9:.2f}-{total_imgs * 256 * 256 / 1e9:.2f} GB")
if reference:
    print(f"    (reference itself: {img_limit} imgs = "
          f"{img_limit * 256 * 256 * 3 / 1e9:.2f} GB raw)")
PY

# ---- full-compute reference (cross-budget, shared, run once) ----
echo ""
echo "############################################################"
echo "########## full-compute reference (per-image anchor) ##########"
echo "############################################################"
if [ "${REFERENCE}" = "1" ]; then
  if [ -f "${REFERENCE_DIR}/results.json" ]; then
    echo "  reusing existing reference: ${REFERENCE_DIR} (results.json present)"
  else
    echo "  running --method baseline (${NUM_STEPS} full steps) at seed=${SEED}"
    echo "  dir: ${REFERENCE_DIR}"
    run_reference "${REFERENCE_DIR}"
  fi
else
  echo "  REFERENCE=0 -> skipping (scripts/analyze_crossover.py will report the "
  echo "  missing anchor and the per-image analysis will be unavailable)"
fi

IFS=',' read -ra BUDGET_ARR <<< "${BUDGETS}"
for K in "${BUDGET_ARR[@]}"; do
  K="$(echo "${K}" | xargs)"
  BK="${OUT_DIR}/k${K}"
  MANIFEST="${BK}/manifest.json"

  echo ""
  echo "############################################################"
  echo "########## budget K=${K}/${NUM_STEPS} calc steps ##########"
  echo "############################################################"
  python scripts/build_budget_manifest.py \
      --num-steps "${NUM_STEPS}" --refresh-count "${K}" --output "${MANIFEST}"

  ARMS=()
  while IFS= read -r arm; do
    ARMS+=("${arm}")
  done < <(python - "${MANIFEST}" <<'PY'
import json, sys
for s in json.load(open(sys.argv[1]))["strategies"]:
    print(s["strategy_id"])
PY
)
  BASE="$(python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["baseline_strategy_id"])
PY
)"

  # Session identity: one shared session per budget for ALL forced arms (the
  # hash keys off (session_id, trajectory_id), and the dataset window is
  # identical for them — same seed), so the same images become sentinels in
  # every arm and H-step rollouts start at the same steps. Noise replicas get
  # a distinct session because they rerun at seeds +1/+2, which shifts the
  # image pool; keep their sentinels on the images they actually generate.
  BUDGET_SESSION="${SESSION_ID:-covr-probe-$(date +%Y%m%d-%H%M%S)-k${K}-seed${SEED}}"

  echo ""
  echo "--- [K=${K}] sentinel / reward telemetry ---"
  echo "  mode=${REWARD_MODE}  rate=${SENTINEL_RATE}  horizon=${SENTINEL_HORIZON}"
  echo "  forced-arm session id: ${BUDGET_SESSION}"
  TRAJS=$(python - "$BATCH_SIZE" "$N_PROMPTS" <<'PY'
import math, sys
bs, n = int(sys.argv[1]), int(sys.argv[2])
print(math.ceil(n / bs))
PY
)
  if [ "${REWARD_MODE}" = "terminal" ]; then
    EXTRA_PER_TRAJ=1
  else
    EXTRA_PER_TRAJ="${SENTINEL_HORIZON}"
  fi
  EXTRA_FORWARDS=$(python - "$TRAJS" "$EXTRA_PER_TRAJ" "$SENTINEL_RATE" <<'PY'
import sys
trajs, per_traj, rate = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(int(trajs * per_traj * rate))
PY
)
  echo "  EXTRA FULL FORWARDS: ~${EXTRA_FORWARDS} (${TRAJS} trajectories x "
  echo "    ${EXTRA_PER_TRAJ}/traj x rate ${SENTINEL_RATE}); Ctrl-C now if this"
  echo "    is more than you want to spend. A K-step trajectory itself costs K"
  echo "    full forwards; hstep at H=10 more than doubles the total forward"
  echo "    budget of the sweep."

  echo ""
  echo "--- [K=${K}] forcing each equal-FLOPs arm ---"
  for arm in "${ARMS[@]}"; do
    echo "  arm=${arm}"
    run_arm "${BK}/equalflops/arm_${arm}" "${MANIFEST}" "${arm}" "${SEED}" \
        "${BUDGET_SESSION}"
  done

  echo ""
  echo "--- [K=${K}] noise floor: arm=${BASE} at seeds +1/+2 ---"
  for delta in 1 2; do
    s=$(( SEED + delta ))
    echo "  seed=${s}"
    run_arm "${BK}/equalflops/noise_${s}" "${MANIFEST}" "${BASE}" "${s}" \
        "covr-probe-$(date +%Y%m%d-%H%M%S)-k${K}-seed${s}"
  done

  echo ""
  echo "--- [K=${K}] verdict ---"
  python scripts/analyze_teacache_sweeps.py "${BK}"
done

echo ""
echo "########## cross-budget summary ##########"
echo "########## arm-spread verdict: ##########"
python scripts/analyze_budget_probe.py "${OUT_DIR}"
echo ""
echo "########## reward-proxy verdict (reward vs FID ordering) ##########"
python scripts/analyze_reward_proxy.py "${OUT_DIR}"
echo ""
echo "########## per-image crossover verdict ##########"
echo "########## one arm best for every image, or do arms flip per image? ##########"
echo "########## (requires the full-compute reference; REFERENCE=0 disables it) ##########"
python scripts/analyze_crossover.py "${OUT_DIR}"
