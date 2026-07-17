#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
N_PROMPTS="${N_PROMPTS:-80}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-2.0}"
BATCH="${BATCH:-32}"
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-80}"
SPECA_BASE_THRESHOLD="${SPECA_BASE_THRESHOLD:-0.01}"
SPECA_DECAY_RATE="${SPECA_DECAY_RATE:-0.01}"
SPECA_MIN_TAYLOR_STEPS="${SPECA_MIN_TAYLOR_STEPS:-1}"
SPECA_MAX_TAYLOR_STEPS="${SPECA_MAX_TAYLOR_STEPS:-4}"
SPECA_ERROR_METRIC="${SPECA_ERROR_METRIC:-cosine_similarity}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/selective_recompute_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT"

if [[ "${SELECTIVE_RECOMPUTE_FOREGROUND:-0}" != "1" ]]; then
    nohup env \
        SELECTIVE_RECOMPUTE_FOREGROUND=1 \
        RUN_ROOT="$RUN_ROOT" \
        bash "$SCRIPT_DIR/run_selective_recompute.sh" \
        > "$RUN_ROOT/runner.log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$RUN_ROOT/runner.pid"
    echo "Started selective recompute experiments"
    echo "PID: $pid"
    echo "Log: $RUN_ROOT/runner.log"
    echo "Results: $RUN_ROOT"
    echo "Monitor: tail -f '$RUN_ROOT/runner.log'"
    exit 0
fi

echo "Selective recompute comparison"
echo "Run root: $RUN_ROOT"
echo "Prompts=$N_PROMPTS Seed=$SEED Steps=$STEPS Guidance=$GUIDANCE Batch=$BATCH"
echo "SpecA: base=$SPECA_BASE_THRESHOLD decay=$SPECA_DECAY_RATE min=$SPECA_MIN_TAYLOR_STEPS max=$SPECA_MAX_TAYLOR_STEPS metric=$SPECA_ERROR_METRIC"

run_case() {
    local name="$1"
    local controller="$2"
    local policy="${3:-}"
    local output_dir="$RUN_ROOT/$name"
    local log_file="$output_dir/run.log"

    mkdir -p "$output_dir"

    local cmd=(
        "$PYTHON_BIN" -u main.py
        --model dit
        --task c2i
        --dataset imagenet
        --method speca
        --n_prompts "$N_PROMPTS"
        --seed "$SEED"
        --num_steps "$STEPS"
        --guidance_scale "$GUIDANCE"
        --batch_size "$BATCH"
        --img_save_limit "$IMG_SAVE_LIMIT"
        --speca_base_threshold "$SPECA_BASE_THRESHOLD"
        --speca_decay_rate "$SPECA_DECAY_RATE"
        --speca_min_taylor_steps "$SPECA_MIN_TAYLOR_STEPS"
        --speca_max_taylor_steps "$SPECA_MAX_TAYLOR_STEPS"
        --speca_error_metric "$SPECA_ERROR_METRIC"
        --compute-controller "$controller"
        --metrics fid is latency flops speed
        --output_dir "$output_dir"
    )

    if [[ -n "$policy" ]]; then
        cmd+=(--controller-correction-policy "$policy")
    fi

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $name"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}" 2>&1 | tee "$log_file"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $name"
}

run_case "none" "none"
run_case "reject" "probe_correct" "reject"
run_case "always" "probe_correct" "always"

echo
echo "All experiments completed: $RUN_ROOT"
