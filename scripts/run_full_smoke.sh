#!/bin/bash
# Full 20-combo smoke test v2
# Parameters: guidance_scale=4.5, batch_size=4, n=50, seed=42
set -e

BASE="python main.py --n_prompts 50 --seed 42 --guidance_scale 4.5 --batch_size 4"
METRICS="latency flops speed imagereward geneval fid is clip lpips mse"
METRICS_T2I="latency flops speed imagereward geneval"
METRICS_C2I="latency flops speed fid is"

echo "============================================================"
echo "SMOKE TEST v2 — 20 combinations"
echo "guidance=4.5, batch=4, n=50, seed=42"
echo "============================================================"

# ========================== DiT c2i (4 combos) ==========================
echo ""
echo "========== DiT c2i imagenet =========="
echo ""

echo "[1/20] DiT c2i imagenet baseline"
$BASE --model dit --task c2i --dataset imagenet --method baseline --metrics $METRICS_C2I
echo ""

echo "[2/20] DiT c2i imagenet teacache"
$BASE --model dit --task c2i --dataset imagenet --method teacache --metrics $METRICS_C2I
echo ""

echo "[3/20] DiT c2i imagenet ddim"
$BASE --model dit --task c2i --dataset imagenet --method ddim --metrics $METRICS_C2I
echo ""

echo "[4/20] DiT c2i imagenet speca"
$BASE --model dit --task c2i --dataset imagenet --method speca --metrics $METRICS_C2I
echo ""

# ========================== PixArt c2i (8 combos) ==========================
echo ""
echo "========== PixArt c2i =========="
echo ""

echo "[5/20] PixArt c2i coco baseline"
$BASE --model pixart --task c2i --dataset coco --method baseline --metrics $METRICS_C2I
echo ""

echo "[6/20] PixArt c2i coco teacache"
$BASE --model pixart --task c2i --dataset coco --method teacache --metrics $METRICS_C2I
echo ""

echo "[7/20] PixArt c2i coco ddim"
$BASE --model pixart --task c2i --dataset coco --method ddim --metrics $METRICS_C2I
echo ""

echo "[8/20] PixArt c2i coco speca"
$BASE --model pixart --task c2i --dataset coco --method speca --metrics $METRICS_C2I
echo ""

echo "[9/20] PixArt c2i imagenet baseline"
$BASE --model pixart --task c2i --dataset imagenet --method baseline --metrics $METRICS_C2I
echo ""

echo "[10/20] PixArt c2i imagenet teacache"
$BASE --model pixart --task c2i --dataset imagenet --method teacache --metrics $METRICS_C2I
echo ""

echo "[11/20] PixArt c2i imagenet ddim"
$BASE --model pixart --task c2i --dataset imagenet --method ddim --metrics $METRICS_C2I
echo ""

echo "[12/20] PixArt c2i imagenet speca"
$BASE --model pixart --task c2i --dataset imagenet --method speca --metrics $METRICS_C2I
echo ""

# ========================== PixArt t2i (8 combos) ==========================
echo ""
echo "========== PixArt t2i =========="
echo ""

echo "[13/20] PixArt t2i drawbench baseline"
$BASE --model pixart --task t2i --dataset drawbench --method baseline --metrics $METRICS_T2I
echo ""

echo "[14/20] PixArt t2i drawbench teacache"
$BASE --model pixart --task t2i --dataset drawbench --method teacache --metrics $METRICS_T2I
echo ""

echo "[15/20] PixArt t2i drawbench ddim"
$BASE --model pixart --task t2i --dataset drawbench --method ddim --metrics $METRICS_T2I
echo ""

echo "[16/20] PixArt t2i drawbench speca"
$BASE --model pixart --task t2i --dataset drawbench --method speca --metrics $METRICS_T2I
echo ""

echo "[17/20] PixArt t2i geneval baseline"
$BASE --model pixart --task t2i --dataset geneval --method baseline --metrics $METRICS_T2I
echo ""

echo "[18/20] PixArt t2i geneval teacache"
$BASE --model pixart --task t2i --dataset geneval --method teacache --metrics $METRICS_T2I
echo ""

echo "[19/20] PixArt t2i geneval ddim"
$BASE --model pixart --task t2i --dataset geneval --method ddim --metrics $METRICS_T2I
echo ""

echo "[20/20] PixArt t2i geneval speca"
$BASE --model pixart --task t2i --dataset geneval --method speca --metrics $METRICS_T2I
echo ""

echo ""
echo "============================================================"
echo "ALL 20 DONE"
echo "============================================================"
