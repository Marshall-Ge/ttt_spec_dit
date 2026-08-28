#!/usr/bin/env bash
# ===========================================================================
# TTT-DiT 全配置 benchmark 脚本 — 显式架构 v2
#
# 覆盖: DiT × 4 + PixArt c2i × 8 + PixArt t2i × 8 = 20 combos
#
# 机器: RTX 4090 D (23.5 GB VRAM)
# 原则: CFG 下 transformer 实际 B = batch_size × 2
# ===========================================================================
set -euo pipefail

# ---- 可覆盖参数 ----
SEED="${SEED:-42}"
N_PROMPTS="${N_PROMPTS:-50}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-4.5}"
BATCH="${BATCH:-4}"
IMG_SAVE_LIMIT="${IMG_SAVE_LIMIT:-50}"

# ---- 共享指标 ----
T2I_METRICS="latency flops speed imagereward geneval"
C2I_METRICS="latency flops speed fid is"

# ---- SpecA 参数 (与 config.py 默认对齐) ----
SPECA_BASE="${SPECA_BASE_THRESHOLD:-0.01}"
SPECA_DECAY="${SPECA_DECAY_RATE:-0.01}"
SPECA_MIN="${SPECA_MIN_TAYLOR_STEPS:-1}"
SPECA_MAX="${SPECA_MAX_TAYLOR_STEPS:-4}"
SPECA_METRIC="${SPECA_ERROR_METRIC:-cosine_similarity}"

BASE_CMD="python main.py"
BASE="${BASE_CMD} --n_prompts ${N_PROMPTS} --seed ${SEED} --guidance_scale ${GUIDANCE} --batch_size ${BATCH} --img_save_limit ${IMG_SAVE_LIMIT}"

# =========================================================================
# 参数速查表
# =========================================================================
#  PixArt-XL-2 (2.5B, fp16):
#    CFG on  (gs=4.5)  → max bs=4  (transformer B=8)
#
#  DiT-2-256 (675M, fp16):
#    CFG on  (gs=4.5)  → max bs=64 (transformer B=128)
# =========================================================================

echo "============================================================"
echo "TTT-DiT Benchmark Suite — 显式架构 v2"
echo "GPU: RTX 4090 D (23.5 GB) | Seed: ${SEED} | Steps: ${STEPS}"
echo "N: ${N_PROMPTS} | Guidance: ${GUIDANCE} | Batch: ${BATCH}"
echo "============================================================"

# =========================================================================
# Section 1 — DiT-2-256  c2i  imagenet  (4 combos)
# =========================================================================

echo ""
echo "========== DiT c2i imagenet =========="
echo ""

echo "[1/20] DiT c2i imagenet baseline"
$BASE --model dit --task c2i --dataset imagenet --method baseline \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[2/20] DiT c2i imagenet teacache"
$BASE --model dit --task c2i --dataset imagenet --method teacache \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[3/20] DiT c2i imagenet ddim"
$BASE --model dit --task c2i --dataset imagenet --method ddim \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[4/20] DiT c2i imagenet speca"
$BASE --model dit --task c2i --dataset imagenet --method speca \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS} \
    --speca_base_threshold "${SPECA_BASE}" --speca_decay_rate "${SPECA_DECAY}" \
    --speca_min_taylor_steps "${SPECA_MIN}" --speca_max_taylor_steps "${SPECA_MAX}" \
    --speca_error_metric "${SPECA_METRIC}"
echo ""

# =========================================================================
# Section 2 — PixArt-α  c2i  (8 combos: coco × 4 + imagenet × 4)
# =========================================================================

echo ""
echo "========== PixArt c2i =========="
echo ""

echo "[5/20] PixArt c2i coco baseline"
$BASE --model pixart --task c2i --dataset coco --method baseline \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[6/20] PixArt c2i coco teacache"
$BASE --model pixart --task c2i --dataset coco --method teacache \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[7/20] PixArt c2i coco ddim"
$BASE --model pixart --task c2i --dataset coco --method ddim \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[8/20] PixArt c2i coco speca"
$BASE --model pixart --task c2i --dataset coco --method speca \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS} \
    --speca_base_threshold "${SPECA_BASE}" --speca_decay_rate "${SPECA_DECAY}" \
    --speca_min_taylor_steps "${SPECA_MIN}" --speca_max_taylor_steps "${SPECA_MAX}" \
    --speca_error_metric "${SPECA_METRIC}"
echo ""

echo "[9/20] PixArt c2i imagenet baseline"
$BASE --model pixart --task c2i --dataset imagenet --method baseline \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[10/20] PixArt c2i imagenet teacache"
$BASE --model pixart --task c2i --dataset imagenet --method teacache \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[11/20] PixArt c2i imagenet ddim"
$BASE --model pixart --task c2i --dataset imagenet --method ddim \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS}
echo ""

echo "[12/20] PixArt c2i imagenet speca"
$BASE --model pixart --task c2i --dataset imagenet --method speca \
    --num_steps "${STEPS}" --metrics ${C2I_METRICS} \
    --speca_base_threshold "${SPECA_BASE}" --speca_decay_rate "${SPECA_DECAY}" \
    --speca_min_taylor_steps "${SPECA_MIN}" --speca_max_taylor_steps "${SPECA_MAX}" \
    --speca_error_metric "${SPECA_METRIC}"
echo ""

# =========================================================================
# Section 3 — PixArt-α  t2i  (8 combos: drawbench × 4 + geneval × 4)
# =========================================================================

echo ""
echo "========== PixArt t2i =========="
echo ""

echo "[13/20] PixArt t2i drawbench baseline"
$BASE --model pixart --task t2i --dataset drawbench --method baseline \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS}
echo ""

echo "[14/20] PixArt t2i drawbench teacache"
$BASE --model pixart --task t2i --dataset drawbench --method teacache \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS}
echo ""

echo "[15/20] PixArt t2i drawbench ddim"
$BASE --model pixart --task t2i --dataset drawbench --method ddim \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS}
echo ""

echo "[16/20] PixArt t2i drawbench speca"
$BASE --model pixart --task t2i --dataset drawbench --method speca \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS} \
    --speca_base_threshold "${SPECA_BASE}" --speca_decay_rate "${SPECA_DECAY}" \
    --speca_min_taylor_steps "${SPECA_MIN}" --speca_max_taylor_steps "${SPECA_MAX}" \
    --speca_error_metric "${SPECA_METRIC}"
echo ""

echo "[17/20] PixArt t2i geneval baseline"
$BASE --model pixart --task t2i --dataset geneval --method baseline \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS}
echo ""

echo "[18/20] PixArt t2i geneval teacache"
$BASE --model pixart --task t2i --dataset geneval --method teacache \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS}
echo ""

echo "[19/20] PixArt t2i geneval ddim"
$BASE --model pixart --task t2i --dataset geneval --method ddim \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS}
echo ""

echo "[20/20] PixArt t2i geneval speca"
$BASE --model pixart --task t2i --dataset geneval --method speca \
    --num_steps "${STEPS}" --metrics ${T2I_METRICS} \
    --speca_base_threshold "${SPECA_BASE}" --speca_decay_rate "${SPECA_DECAY}" \
    --speca_min_taylor_steps "${SPECA_MIN}" --speca_max_taylor_steps "${SPECA_MAX}" \
    --speca_error_metric "${SPECA_METRIC}"
echo ""

echo "============================================================"
echo "ALL 20 DONE"
echo "============================================================"
