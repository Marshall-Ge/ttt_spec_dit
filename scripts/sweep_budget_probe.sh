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
#   SEED (42)  GUIDANCE (4.5)  OUT_DIR (output/covr_budget_probe, relative to
#       the repo root — the script cds there, so a relative OUT_DIR survives
#       being invoked from anywhere and keeps the PNGs off /tmp, where a reboot
#       or a parallel job can take them)
#   REWARD_MODE (terminal|hstep, default terminal)  SENTINEL_RATE (1.0)
#   SENTINEL_HORIZON (hstep mode only, default 10)  SESSION_ID (auto)
#   ARM_FILTER (comma-separated strategy IDs; empty = all manifest arms)
#   RANDOM_COUNT (default 0)  RANDOM_SEED (default 0)
#   PAIRED_REPLICAS (total latent draws per selected arm, default 1)
#   THRESHOLDS (comma-separated plain TeaCache thresholds; empty = disabled)
#   THRESHOLD_REPLICAS (total latent draws per threshold, default 1)
#   REFERENCE (1|0, default 1)  REFERENCE_DIR (default <OUT_DIR>/reference)
#   IMG_SAVE_LIMIT (default N_PROMPTS = save every image; per-image crossover
#       analysis needs the full set, the old default of 50 is not enough)
#   NOISE_MODE (latent|seed, default seed)  NOISE_REPLICAS (2)  NOISE_ARM (auto)
#   SKIP_ARMS (0|1)  SKIP_NOISE (0|1) — reuse what is already on disk
#
# WHICH NOISE THE FLOOR MEASURES (NOISE_MODE) — this decides what the floor is
# a valid comparison FOR, so it is not a cosmetic knob:
#   seed   (default, legacy) reruns the arm at SEED+1..+R. --seed feeds
#       ImageNetDataset(n, seed)'s shuffle (dataset/imagenet.py:137-140), so
#       each replica draws a DIFFERENT set of N images and classes. The
#       resulting sd therefore mixes latent-sampling noise with
#       which-images-were-drawn noise, and at N=500 the latter can dominate.
#       That is the right floor for "would this arm ranking survive a different
#       dataset draw", which is what analyze_teacache_sweeps.py's global
#       arm-FID spread verdict asks.
#   latent reruns the arm at the SAME --seed with --latent-seed-offset 1..R, so
#       every replica generates the SAME N images and classes from INDEPENDENT
#       latents (utils.latent_seed_for_index gives disjoint seed sets, stride
#       1e6). That is the right floor for analyze_crossover.py's [d], which
#       asks whether arms flip ON THE SAME IMAGE — pairing is by global_idx, so
#       the image draw is held fixed by construction and must not be
#       re-randomized by the floor it is compared against.
#   The two modes write to different dirs (noise_<seed> vs noise_lat<r>) but all
#   three analyzers glob noise_*, so the script REMOVES the other mode's dirs
#   before writing: pooling both into one sd would silently average two
#   different noise sources.
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
OUT_DIR="${OUT_DIR:-output/covr_budget_probe}"
REWARD_MODE="${REWARD_MODE:-terminal}"
SENTINEL_RATE="${SENTINEL_RATE:-1.0}"
SENTINEL_HORIZON="${SENTINEL_HORIZON:-10}"
SESSION_ID="${SESSION_ID:-}"
ARM_FILTER="${ARM_FILTER:-}"
PAIRED_REPLICAS="${PAIRED_REPLICAS:-1}"
THRESHOLDS="${THRESHOLDS:-}"
THRESHOLD_REPLICAS="${THRESHOLD_REPLICAS:-1}"
RANDOM_COUNT="${RANDOM_COUNT:-0}"
RANDOM_SEED="${RANDOM_SEED:-0}"

# --- noise floor: WHICH noise the floor measures (see header) ---
NOISE_MODE="${NOISE_MODE:-seed}"       # latent|seed
NOISE_REPLICAS="${NOISE_REPLICAS:-2}"  # replicas per budget
NOISE_ARM="${NOISE_ARM:-}"             # arm to replicate (default: manifest baseline)
SKIP_ARMS="${SKIP_ARMS:-0}"            # 1 = do not rerun the 4 forced arms
SKIP_NOISE="${SKIP_NOISE:-0}"          # 1 = do not rerun the noise replicas

# --- full-compute reference + per-image PNG saving (see header note) ---
REFERENCE="${REFERENCE:-1}"            # 1|0 — run the full-compute reference
REFERENCE_DIR="${REFERENCE_DIR:-}"     # existing reference dir to reuse (default: OUT_DIR/reference)
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-}"   # PNGs saved per run (default: N_PROMPTS = all)
VERSION_PROBE_DIR="${VERSION_PROBE_DIR:-${OUT_DIR}/version_probe}"

case "${REWARD_MODE}" in
  terminal) SENTINEL_HORIZON=0 ;;
  hstep)    SENTINEL_HORIZON="${SENTINEL_HORIZON}" ;;
  *)
    echo "REWARD_MODE must be 'terminal' or 'hstep' (got '${REWARD_MODE}')" >&2
    exit 2
    ;;
esac
if ! python - "$SENTINEL_RATE" "$PAIRED_REPLICAS" \
        "$THRESHOLD_REPLICAS" "$ARM_FILTER" "$THRESHOLDS" <<'PY' >/dev/null
import math
import sys
rate = float(sys.argv[1])
assert 0.0 <= rate <= 1.0, "SENTINEL_RATE must be in [0, 1]"
paired = int(sys.argv[2])
threshold = int(sys.argv[3])
arms = [value.strip() for value in sys.argv[4].split(",") if value.strip()]
thresholds = [value.strip() for value in sys.argv[5].split(",") if value.strip()]
assert paired >= 1, "PAIRED_REPLICAS must be >= 1"
assert threshold >= 1, "THRESHOLD_REPLICAS must be >= 1"
assert len(arms) == len(set(arms)), "ARM_FILTER contains duplicate strategy IDs"
for value in thresholds:
    parsed = float(value)
    assert math.isfinite(parsed) and parsed >= 0.0, (
        "THRESHOLDS values must be finite and >= 0")
assert (paired == 1 and not thresholds) or rate == 0.0, (
    "paired/threshold static validation requires SENTINEL_RATE=0; static runs "
    "must not pay delayed full-reference shadow cost")
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
if ! python - "${RANDOM_COUNT}" <<'PY' >/dev/null
import sys
assert int(sys.argv[1]) >= 0, "RANDOM_COUNT must be >= 0"
PY
then
  exit 2
fi

case "${NOISE_MODE}" in
  latent|seed) ;;
  *)
    echo "NOISE_MODE must be 'latent' or 'seed' (got '${NOISE_MODE}')" >&2
    exit 2
    ;;
esac
if ! python - "$NOISE_REPLICAS" <<'PY' >/dev/null
import sys
assert int(sys.argv[1]) >= 0, "NOISE_REPLICAS must be >= 0"
PY
then
  exit 2
fi
if [ "${SKIP_ARMS}" = "1" ] && [ "${SKIP_NOISE}" = "1" ] && \
   [ -z "${THRESHOLDS}" ]; then
  echo "SKIP_ARMS=1, SKIP_NOISE=1, and empty THRESHOLDS leave nothing to run" >&2
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

run_teacache() {  # run_teacache <out> <seed> <latent-offset> [extra args...]
  local out="$1" seed="$2" lat_offset="$3"
  shift 3
  rm -rf "${out}"
  python main.py --model dit --task c2i --dataset imagenet \
      --method teacache --metrics fid is latency flops speed \
      --seed "${seed}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
      --latent-seed-offset "${lat_offset}" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}" \
      --img_save_limit "${IMG_SAVE_LIMIT}" \
      --output_dir "${out}" "$@"
}

run_arm() {  # run_arm <out> <manifest> <arm> <seed> <session-id> [latent-offset]
  local out="$1" manifest="$2" arm="$3" seed="$4" session_id="$5"
  local lat_offset="${6:-0}"
  run_teacache "${out}" "${seed}" "${lat_offset}" \
      --covr-strategy-manifest "${manifest}" \
      --covr-force-strategy-id "${arm}" \
      --covr-session-id "${session_id}" \
      --covr-sentinel-rate "${SENTINEL_RATE}" \
      --covr-sentinel-horizon "${SENTINEL_HORIZON}"
}

run_threshold() {  # run_threshold <out> <threshold> [latent-offset]
  local out="$1" threshold="$2"
  local lat_offset="${3:-0}"
  run_teacache "${out}" "${SEED}" "${lat_offset}" --thresh "${threshold}"
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
python - "$BUDGETS" "$NUM_STEPS" "$N_PROMPTS" "$IMG_SAVE_LIMIT" "$REFERENCE" \
        "$NOISE_REPLICAS" "$SKIP_ARMS" "$SKIP_NOISE" "$NOISE_MODE" \
        "$ARM_FILTER" "$PAIRED_REPLICAS" "$THRESHOLDS" \
        "$THRESHOLD_REPLICAS" "$RANDOM_COUNT" <<'PY'
import sys
budgets = [int(k) for k in sys.argv[1].split(",") if k.strip()]
num_steps, n_prompts, img_limit = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
reference = sys.argv[5] == "1"
replicas = int(sys.argv[6])
skip_arms, skip_noise = sys.argv[7] == "1", sys.argv[8] == "1"
noise_mode = sys.argv[9]
arm_filter = [a.strip() for a in sys.argv[10].split(",") if a.strip()]
paired_replicas = int(sys.argv[11])
thresholds = [t.strip() for t in sys.argv[12].split(",") if t.strip()]
threshold_replicas = int(sys.argv[13])
n_random = int(sys.argv[14])
n_selected = len(arm_filter) if arm_filter else 4 + n_random
n_arms = 0 if skip_arms else n_selected * paired_replicas
n_noise = 0 if skip_noise else replicas
per_budget = n_arms + n_noise
arm_runs = per_budget * len(budgets)
arm_units = sum(per_budget * k for k in budgets)  # each run costs K forwards/traj
ref_units = num_steps if reference else 0
version_probe_units = num_steps
threshold_runs = len(thresholds) * threshold_replicas
mean_k = sum(budgets) / len(budgets)
print("--- cost / disk budget (before any GPU spend) ---")
print(f"  arm runs: {arm_runs} ({n_arms} selected-arm replicas + "
      f"{n_noise} noise[{noise_mode}] per budget, K={budgets})")
print(f"  plain TeaCache threshold runs: {threshold_runs}")
if skip_arms:
    print("    SKIP_ARMS=1 -> forced arms are REUSED from disk, not rerun")
if skip_noise:
    print("    SKIP_NOISE=1 -> existing noise_* dirs are REUSED, not rerun")
print(f"  forward-units: arms={arm_units}, reference={ref_units}"
      f"{'' if reference else ' (REFERENCE=0)'}, "
      f"version_probe={version_probe_units}, "
      f"total={arm_units + ref_units + version_probe_units}")
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

echo ""
echo "############################################################"
echo "########## COVR runtime version probe ########################"
echo "############################################################"
echo "  dir: ${VERSION_PROBE_DIR}"
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics latency \
    --seed "${SEED}" --num_steps "${NUM_STEPS}" --n_prompts 1 \
    --guidance_scale "${GUIDANCE}" --batch_size 1 \
    --covr-profile-stages --output_dir "${VERSION_PROBE_DIR}"
VERSION_KEY="$(python - "${VERSION_PROBE_DIR}/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["config"].get("covr_version_key")
if not value:
    raise SystemExit("version probe did not produce config.covr_version_key")
print(value)
PY
)"
echo "  runtime version key: ${VERSION_KEY}"

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
      --num-steps "${NUM_STEPS}" --refresh-count "${K}" \
      --random-count "${RANDOM_COUNT}" --random-seed "${RANDOM_SEED}" \
      --version-key "${VERSION_KEY}" \
      --output "${MANIFEST}"

  ARMS=()
  while IFS= read -r arm; do
    ARMS+=("${arm}")
  done < <(python - "${MANIFEST}" <<'PY'
import json, sys
for s in json.load(open(sys.argv[1]))["strategies"]:
    print(s["strategy_id"])
PY
)
  if [ -n "${ARM_FILTER}" ]; then
    FILTERED_ARMS=()
    IFS=',' read -ra REQUESTED_ARMS <<< "${ARM_FILTER}"
    for requested in "${REQUESTED_ARMS[@]}"; do
      requested="$(echo "${requested}" | xargs)"
      if [ -z "${requested}" ]; then
        continue
      fi
      found=0
      for arm in "${ARMS[@]}"; do
        if [ "${arm}" = "${requested}" ]; then
          FILTERED_ARMS+=("${arm}")
          found=1
          break
        fi
      done
      if [ "${found}" -ne 1 ]; then
        echo "ARM_FILTER contains unknown strategy '${requested}'" >&2
        echo "  available: ${ARMS[*]}" >&2
        exit 2
      fi
    done
    if [ "${#FILTERED_ARMS[@]}" -eq 0 ]; then
      echo "ARM_FILTER selected no strategies" >&2
      exit 2
    fi
    ARMS=("${FILTERED_ARMS[@]}")
  fi
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
  if [ "${SKIP_ARMS}" = "1" ]; then
    echo "  SKIP_ARMS=1 -> reusing existing arm_* dirs"
  else
    for arm in "${ARMS[@]}"; do
      echo "  arm=${arm} latent-seed-offset=0"
      run_arm "${BK}/equalflops/arm_${arm}" "${MANIFEST}" "${arm}" "${SEED}" \
          "${BUDGET_SESSION}" 0
      if [ "${PAIRED_REPLICAS}" -gt 1 ]; then
        for r in $(seq 1 $((PAIRED_REPLICAS - 1))); do
          echo "  arm=${arm} latent-seed-offset=${r}"
          run_arm "${BK}/equalflops/arm_${arm}/rep_${r}" "${MANIFEST}" \
              "${arm}" "${SEED}" "${BUDGET_SESSION}-${arm}-rep${r}" "${r}"
        done
      fi
    done
  fi

  echo ""
  NOISE_ARM_ID="${NOISE_ARM:-${BASE}}"
  echo "--- [K=${K}] noise floor: arm=${NOISE_ARM_ID} mode=${NOISE_MODE} x${NOISE_REPLICAS} ---"
  # One mode's replicas at a time: all three analyzers glob noise_*, so leaving
  # the other mode's dirs behind would pool two DIFFERENT noise sources into one
  # sd. Stale dirs of the mode not being run are removed.
  if [ "${SKIP_NOISE}" = "1" ]; then
    echo "  SKIP_NOISE=1 -> keeping existing noise_* dirs untouched"
  else
    if [ "${NOISE_MODE}" = "latent" ]; then
      rm -rf "${BK}"/equalflops/noise_[0-9]*
      for r in $(seq 1 "${NOISE_REPLICAS}"); do
        echo "  latent-seed-offset=${r} (same seed=${SEED} -> same 500 images/classes)"
        run_arm "${BK}/equalflops/noise_lat${r}" "${MANIFEST}" \
            "${NOISE_ARM_ID}" "${SEED}" \
            "covr-probe-$(date +%Y%m%d-%H%M%S)-k${K}-seed${SEED}-lat${r}" "${r}"
      done
    else
      rm -rf "${BK}"/equalflops/noise_lat*
      for r in $(seq 1 "${NOISE_REPLICAS}"); do
        s=$(( SEED + r ))
        echo "  seed=${s} (shifts the image pool as well as the latents)"
        run_arm "${BK}/equalflops/noise_${s}" "${MANIFEST}" \
            "${NOISE_ARM_ID}" "${s}" \
            "covr-probe-$(date +%Y%m%d-%H%M%S)-k${K}-seed${s}" 0
      done
    fi
  fi

  echo ""
  echo "--- [K=${K}] verdict ---"
  python scripts/analyze_teacache_sweeps.py "${BK}"
done

# Thresholds are shared across budgets and run after ARM_FILTER has been
# validated against every generated manifest.
if [ -n "${THRESHOLDS}" ]; then
  echo ""
  echo "############################################################"
  echo "########## plain TeaCache threshold comparators ##########"
  echo "############################################################"
  IFS=',' read -ra THRESHOLD_ARR <<< "${THRESHOLDS}"
  for threshold in "${THRESHOLD_ARR[@]}"; do
    threshold="$(echo "${threshold}" | xargs)"
    if [ -z "${threshold}" ]; then
      continue
    fi
    echo "  threshold=${threshold} latent-seed-offset=0"
    run_threshold "${OUT_DIR}/threshold/thresh_${threshold}" "${threshold}" 0
    if [ "${THRESHOLD_REPLICAS}" -gt 1 ]; then
      for r in $(seq 1 $((THRESHOLD_REPLICAS - 1))); do
        echo "  threshold=${threshold} latent-seed-offset=${r}"
        run_threshold "${OUT_DIR}/threshold/thresh_${threshold}/rep_${r}" \
            "${threshold}" "${r}"
      done
    fi
  done
fi

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

if [ "${PAIRED_REPLICAS}" -gt 1 ] || [ -n "${THRESHOLDS}" ]; then
  echo ""
  echo "########## paired static-mask verdict ##########"
  if [ "${RANDOM_COUNT}" -gt 0 ]; then
    if [[ "${BUDGETS}" != *,* ]]; then
      STATIC_LEFT="pattern_uniform_k${BUDGETS}"
      STATIC_RIGHT="pattern_geometric_k${BUDGETS}"
      python scripts/analyze_static_masks.py "${OUT_DIR}" \
          --left "${STATIC_LEFT}" --right "${STATIC_RIGHT}"
    else
      echo "  random arms use budget-qualified IDs; run paired analysis per "
      echo "  budget with --left pattern_uniform_k<K> --right "
      echo "  pattern_geometric_k<K>"
    fi
  else
    python scripts/analyze_static_masks.py "${OUT_DIR}"
  fi
fi
