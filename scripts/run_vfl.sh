#!/usr/bin/env bash
# ===========================================================================
# VFL v4 实验脚本 — DiT + SpecA + 完整三层 VFL (含 ||B||_F 监控)
#
# 用途: 跑 50k ImageNet c2i 验证 VFL LoRA 是否真的学到东西
# 配套: verification_feedback_loop/async_trainer.py 已加 ||B||_F 日志
#
# 启动:
#   bash scripts/run_vfl.sh
#
# 监控 (另一个终端):
#   tail -f vfl.log | grep -E 'cycle #| RESULTS'
#   tail -f vfl.log | grep '||B||_F'    # 看 LoRA B 范数是否 > 1e-6
#
# 停止:
#   pkill -f 'main.py.*vfl' 或 kill <PID>
#
# 可覆盖参数 (示例):
#   N_PROMPTS=500 STEPS=20 bash scripts/run_vfl.sh   # 冒烟
#   FRESH_LOG=0 bash scripts/run_vfl.sh              # 追加而非轮转 log
# ===========================================================================
set -euo pipefail

# ---- 可覆盖参数 ----
SEED="${SEED:-42}"
N_PROMPTS="${N_PROMPTS:-50000}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-2.0}"
BATCH="${BATCH:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-output/c2i_dit_imagenet_speca_vfl_v4_cfg2.0}"
LOG_FILE="${LOG_FILE:-vfl.log}"

# ---- SpecA 参数 (与 config.py 默认对齐) ----
SPECA_BASE="${SPECA_BASE_THRESHOLD:-0.01}"
SPECA_DECAY="${SPECA_DECAY_RATE:-0.01}"
SPECA_MIN="${SPECA_MIN_TAYLOR_STEPS:-1}"
SPECA_MAX="${SPECA_MAX_TAYLOR_STEPS:-4}"
SPECA_METRIC="${SPECA_ERROR_METRIC:-cosine_similarity}"

METRICS="fid is flops latency"

echo "============================================================"
echo "VFL v4 Experiment — DiT + SpecA + VFL (||B||_F monitored)"
echo "============================================================"
echo "  N:         ${N_PROMPTS}"
echo "  Steps:     ${STEPS}"
echo "  Guidance:  ${GUIDANCE}"
echo "  Batch:     ${BATCH}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Log:       ${LOG_FILE}"
echo "============================================================"
echo ""
echo "Monitor LoRA training signal (another terminal):"
echo "  tail -f ${LOG_FILE} | grep '||B||_F'"
echo ""

# ---- 轮转旧 log (默认开, FRESH_LOG=0 关闭) ----
if [ "${FRESH_LOG:-1}" = "1" ] && [ -f "${LOG_FILE}" ]; then
    ROTATED="${LOG_FILE}.$(date +%Y%m%d_%H%M%S).bak"
    mv "${LOG_FILE}" "${ROTATED}"
    echo "[run_vfl] rotated old log → ${ROTATED}"
fi

# ---- 启动 ----
nohup python main.py \
    --model dit --task c2i --dataset imagenet \
    --method speca --metrics ${METRICS} \
    --guidance_scale "${GUIDANCE}" --seed "${SEED}" --num_steps "${STEPS}" \
    --n_prompts "${N_PROMPTS}" --batch_size "${BATCH}" \
    --output_dir "${OUTPUT_DIR}" \
    --speca_base_threshold "${SPECA_BASE}" --speca_decay_rate "${SPECA_DECAY}" \
    --speca_min_taylor_steps "${SPECA_MIN}" --speca_max_taylor_steps "${SPECA_MAX}" \
    --speca_error_metric "${SPECA_METRIC}" \
    --vfl \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[run_vfl] launched PID=${PID}"
echo "[run_vfl] log:  ${LOG_FILE}"
echo "[run_vfl] kill: kill ${PID}"
echo "[run_vfl] tail: tail -f ${LOG_FILE}"
