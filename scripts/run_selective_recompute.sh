#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"
if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/run_selective_recompute.sh [smoke|diagnostic]" >&2
    exit 2
fi

case "$MODE" in
    smoke)
        DEFAULT_N_PROMPTS=80
        DEFAULT_SEEDS="42"
        DEFAULT_CHECK_LAYERS="20"
        ;;
    diagnostic)
        DEFAULT_N_PROMPTS=500
        DEFAULT_SEEDS="42 43 44"
        DEFAULT_CHECK_LAYERS="20 27"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Usage: bash scripts/run_selective_recompute.sh [smoke|diagnostic]" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
N_PROMPTS="${N_PROMPTS:-$DEFAULT_N_PROMPTS}"
SEEDS="${SEEDS:-${SEED:-$DEFAULT_SEEDS}}"
SPECA_CHECK_LAYERS="${SPECA_CHECK_LAYERS:-${SPECA_CHECK_LAYER:-$DEFAULT_CHECK_LAYERS}}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-2.0}"
BATCH="${BATCH:-32}"
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-80}"
SPECA_BASE_THRESHOLD="${SPECA_BASE_THRESHOLD:-0.01}"
SPECA_DECAY_RATE="${SPECA_DECAY_RATE:-0.01}"
SPECA_MIN_TAYLOR_STEPS="${SPECA_MIN_TAYLOR_STEPS:-1}"
SPECA_MAX_TAYLOR_STEPS="${SPECA_MAX_TAYLOR_STEPS:-4}"
SPECA_ERROR_METRIC="${SPECA_ERROR_METRIC:-cosine_similarity}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/selective_recompute_${MODE}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT"

if [[ "${SELECTIVE_RECOMPUTE_FOREGROUND:-0}" != "1" ]]; then
    nohup env \
        SELECTIVE_RECOMPUTE_FOREGROUND=1 \
        RUN_ROOT="$RUN_ROOT" \
        bash "$SCRIPT_DIR/run_selective_recompute.sh" "$MODE" \
        > "$RUN_ROOT/runner.log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$RUN_ROOT/runner.pid"
    echo "Started selective recompute $MODE"
    echo "PID: $pid"
    echo "Log: $RUN_ROOT/runner.log"
    echo "Results: $RUN_ROOT"
    echo "Monitor: tail -f '$RUN_ROOT/runner.log'"
    exit 0
fi

read -r -a seed_values <<< "$SEEDS"
read -r -a check_layer_values <<< "$SPECA_CHECK_LAYERS"
case_count=$((${#seed_values[@]} * ${#check_layer_values[@]} * 3))

echo "Selective recompute comparison"
echo "Mode: $MODE"
echo "Run root: $RUN_ROOT"
echo "Cases: $case_count"
echo "Prompts=$N_PROMPTS Seeds=[$SEEDS] CheckLayers=[$SPECA_CHECK_LAYERS]"
echo "Steps=$STEPS Guidance=$GUIDANCE Batch=$BATCH"
echo "SpecA: base=$SPECA_BASE_THRESHOLD decay=$SPECA_DECAY_RATE min=$SPECA_MIN_TAYLOR_STEPS max=$SPECA_MAX_TAYLOR_STEPS metric=$SPECA_ERROR_METRIC"

run_case() {
    local group_root="$1"
    local check_layer="$2"
    local seed="$3"
    local name="$4"
    local controller="$5"
    local policy="${6:-}"
    local output_dir="$group_root/$name"
    local log_file="$output_dir/run.log"

    mkdir -p "$output_dir"

    local cmd=(
        "$PYTHON_BIN" -u main.py
        --model dit
        --task c2i
        --dataset imagenet
        --method speca
        --n_prompts "$N_PROMPTS"
        --seed "$seed"
        --num_steps "$STEPS"
        --guidance_scale "$GUIDANCE"
        --batch_size "$BATCH"
        --img_save_limit "$IMG_SAVE_LIMIT"
        --speca_base_threshold "$SPECA_BASE_THRESHOLD"
        --speca_decay_rate "$SPECA_DECAY_RATE"
        --speca_min_taylor_steps "$SPECA_MIN_TAYLOR_STEPS"
        --speca_max_taylor_steps "$SPECA_MAX_TAYLOR_STEPS"
        --speca_error_metric "$SPECA_ERROR_METRIC"
        --speca_check_layer "$check_layer"
        --compute-controller "$controller"
        --metrics fid is latency flops speed
        --output_dir "$output_dir"
    )

    if [[ -n "$policy" ]]; then
        cmd+=(--controller-correction-policy "$policy")
    fi

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting layer=$check_layer seed=$seed policy=$name"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}" 2>&1 | tee "$log_file"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished layer=$check_layer seed=$seed policy=$name"
}

for seed in "${seed_values[@]}"; do
    for check_layer in "${check_layer_values[@]}"; do
        if [[ "$MODE" == "smoke" && ${#seed_values[@]} -eq 1 && ${#check_layer_values[@]} -eq 1 ]]; then
            group_root="$RUN_ROOT"
        else
            group_root="$RUN_ROOT/layer_${check_layer}/seed_${seed}"
        fi
        run_case "$group_root" "$check_layer" "$seed" "none" "none"
        run_case "$group_root" "$check_layer" "$seed" "reject" "probe_correct" "reject"
        run_case "$group_root" "$check_layer" "$seed" "always" "probe_correct" "always"
    done
done

echo
echo "All experiments completed: $RUN_ROOT"
