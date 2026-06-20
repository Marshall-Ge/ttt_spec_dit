#!/usr/bin/env bash
# ===========================================================================
# TTT-DiT 全配置启动脚本 — 每种组合已标注 GPU 最大稳定 batch_size
#
# 机器: RTX 4090 D (23.5 GB VRAM)
# 原则: CFG 下 transformer 实际 B = batch_size × 2
# ===========================================================================
set -euo pipefail

SEED="${SEED:-42}"
N_PROMPTS="${N_PROMPTS:-1}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-4.5}"
BATCH="${BATCH:-4}"

echo "══════════════════════════════════════════════════════════════"
echo " TTT-DiT Benchmark Suite"
echo " GPU: RTX 4090 D (23.5 GB)"
echo " Seed: ${SEED}  |  Steps: ${STEPS}"
echo "══════════════════════════════════════════════════════════════"

# =========================================================================
# Section 0 — 参数速查表
# =========================================================================
#  PixArt-XL-2 (2.5B, fp16):
#    CFG on  (gs=4.5)  → max bs=4  (transformer B=8)
#    CFG off (gs=1.0)  → max bs=6  (transformer B=6)
#
#  DiT-2-256 (675M, fp16):
#    CFG on  (gs=4.0)  → max bs=64 (transformer B=128)
#    CFG off (gs=1.0)  → max bs=80 (transformer B=80)
# =========================================================================

# =========================================================================
# Section 1 — PixArt-α  t2i  (drawbench / geneval)
# =========================================================================

echo ""
echo ">>> PixArt-α  T2I  <<<"
echo ""

# --- baseline ---
echo "  [1.1] PixArt baseline (DPM++)  CFG"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method baseline --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4

echo "  [1.2] PixArt baseline (DPM++)  no CFG"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method baseline --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 6

# --- speca ---
echo "  [1.3] PixArt SpecA  CFG  (↓65% FLOPs)"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method speca --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4 \
    --speca_base_threshold 0.01 --speca_decay_rate 0.01 \
    --speca_min_taylor_steps 1 --speca_max_taylor_steps 2 \
    --speca_error_metric cosine_similarity

echo "  [1.4] PixArt SpecA  no CFG"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method speca --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 6 \
    --speca_base_threshold 0.01 --speca_decay_rate 0.01 \
    --speca_min_taylor_steps 1 --speca_max_taylor_steps 2 \
    --speca_error_metric cosine_similarity

# --- teacache ---
echo "  [1.5] PixArt TeaCache  CFG  (γ=0.25)"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method teacache --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4 --thresh 0.25

echo "  [1.6] PixArt TeaCache  no CFG"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method teacache --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 6 --thresh 0.25

# --- ddim ---
echo "  [1.7] PixArt DDIM step-skip  CFG"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method ddim --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${DDIM_FLOP_MATCHED_STEPS:-10}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4

echo "  [1.8] PixArt DDIM step-skip  no CFG"
python main.py \
    --model pixart --task t2i --dataset drawbench \
    --method ddim --metrics latency flops speed \
    --seed "${SEED}" --num_steps "${DDIM_FLOP_MATCHED_STEPS:-10}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 6

# --- geneval dataset ---
echo "  [1.9] PixArt SpecA  geneval  CFG"
python main.py \
    --model pixart --task t2i --dataset geneval \
    --method speca --metrics geneval latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4 \
    --speca_base_threshold 0.01 --speca_decay_rate 0.01 \
    --speca_min_taylor_steps 1 --speca_max_taylor_steps 2 \
    --speca_error_metric cosine_similarity

echo "  [1.10] PixArt baseline  geneval  CFG"
python main.py \
    --model pixart --task t2i --dataset geneval \
    --method baseline --metrics geneval latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4

# =========================================================================
# Section 2 — PixArt-α  c2i  (coco)
# =========================================================================

echo ""
echo ">>> PixArt-α  C2I  (COCO)  <<<"
echo ""

echo "  [2.1] PixArt baseline  coco  CFG"
python main.py \
    --model pixart --task c2i --dataset coco \
    --method baseline --metrics fid is clip lpips mse latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4

echo "  [2.2] PixArt SpecA  coco  CFG"
python main.py \
    --model pixart --task c2i --dataset coco \
    --method speca --metrics fid is clip lpips mse latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4 \
    --speca_base_threshold 0.01 --speca_decay_rate 0.01 \
    --speca_min_taylor_steps 1 --speca_max_taylor_steps 2 \
    --speca_error_metric cosine_similarity

echo "  [2.3] PixArt TeaCache  coco  CFG"
python main.py \
    --model pixart --task c2i --dataset coco \
    --method teacache --metrics fid is clip lpips mse latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.5 --batch_size 4 --thresh 0.25

# =========================================================================
# Section 3 — DiT-2-256  c2i  (imagenet)
# =========================================================================

echo ""
echo ">>> DiT-2-256  C2I  (ImageNet)  <<<"
echo ""

# --- baseline ---
echo "  [3.1] DiT baseline  CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method baseline --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.0 --batch_size 64

echo "  [3.2] DiT baseline  no CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method baseline --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 80

# --- speca ---
echo "  [3.3] DiT SpecA  CFG  (↓65% FLOPs)"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.0 --batch_size 64 \
    --speca_base_threshold 0.01 --speca_decay_rate 0.01 \
    --speca_min_taylor_steps 1 --speca_max_taylor_steps 2 \
    --speca_error_metric cosine_similarity

echo "  [3.4] DiT SpecA  no CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 80 \
    --speca_base_threshold 0.01 --speca_decay_rate 0.01 \
    --speca_min_taylor_steps 1 --speca_max_taylor_steps 2 \
    --speca_error_metric cosine_similarity

# --- teacache ---
echo "  [3.5] DiT TeaCache  CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.0 --batch_size 64 --thresh 0.25

echo "  [3.6] DiT TeaCache  no CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 80 --thresh 0.25

# --- ddim ---
echo "  [3.7] DiT DDIM step-skip  CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method ddim --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${DDIM_FLOP_MATCHED_STEPS:-10}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 4.0 --batch_size 64

echo "  [3.8] DiT DDIM step-skip  no CFG"
python main.py \
    --model dit --task c2i --dataset imagenet \
    --method ddim --metrics fid is latency flops speed \
    --seed "${SEED}" --num_steps "${DDIM_FLOP_MATCHED_STEPS:-10}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale 1.0 --batch_size 80

# =========================================================================
# Section 4 — 快速冒烟测试 (n_prompts=1, 只测不崩)
# =========================================================================

smoke() {
    echo ""
    echo ">>> Smoke Tests (n_prompts=1) <<<"
    echo ""

    local S_N=1

    echo "  [S1] PixArt+baseline+CFG bs=4"
    python main.py --model pixart --task t2i --dataset drawbench --method baseline \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.5 --batch_size 4

    echo "  [S2] PixArt+speca+CFG bs=4"
    python main.py --model pixart --task t2i --dataset drawbench --method speca \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.5 --batch_size 4

    echo "  [S3] PixArt+teacache+CFG bs=4"
    python main.py --model pixart --task t2i --dataset drawbench --method teacache \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.5 --batch_size 4 --thresh 0.25

    echo "  [S4] PixArt+ddim+CFG bs=4"
    python main.py --model pixart --task t2i --dataset drawbench --method ddim \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.5 --batch_size 4

    echo "  [S5] DiT+baseline+CFG bs=64"
    python main.py --model dit --task c2i --dataset imagenet --method baseline \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.0 --batch_size 64

    echo "  [S6] DiT+speca+CFG bs=64"
    python main.py --model dit --task c2i --dataset imagenet --method speca \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.0 --batch_size 64

    echo "  [S7] DiT+teacache+CFG bs=64"
    python main.py --model dit --task c2i --dataset imagenet --method teacache \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 4.0 --batch_size 64 --thresh 0.25

    echo "  [S8] PixArt+speca noCFG bs=6 (backward compat)"
    python main.py --model pixart --task t2i --dataset drawbench --method speca \
        --metrics latency --seed 42 --num_steps 5 --n_prompts ${S_N} \
        --guidance_scale 1.0 --batch_size 6

    echo ""
    echo "  ✅ All smoke tests passed."
}

# 取消下面一行注释即可只跑冒烟测试:
# smoke; exit 0

echo ""
echo "══════════════════════════════════════════════════════════════"
echo " Done."
echo "══════════════════════════════════════════════════════════════"
