#!/bin/bash
# ===========================================================================
# Download ImageNet 2012 validation set for ImageNet 50K FID/IS evaluation.
#
# ImageNet 50K protocol (SpeCa / ADM / DiT standard):
#   - 50 images per class across all 1,000 ImageNet classes
#   - Real reference: 50K real ImageNet validation images
#   - Generated: 50K images from class-name prompts via PixArt-α
#
# Downloads (total ~6.5 GB):
#   1. ILSVRC2012_img_val.tar (~6.3 GB, 50,000 images)
#   2. ILSVRC2012_devkit_t12.tar.gz (~2.5 MB, class labels/mappings)
#
# Mirror options:
#   A) image-net.org (requires free account registration)
#   B) Kaggle (requires API token)
#   C) Academic Torrents
#
# Target directory: /root/autodl-fs/data/imagenet
# ===========================================================================

set -euo pipefail

DATA_DIR="${1:-/root/autodl-fs/data/imagenet}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "ImageNet 2012 Validation Set Download"
echo "Target: $DATA_DIR"
echo "============================================"

# ---- Method selection ----
METHOD="${2:-auto}"

if [ "$METHOD" = "manual" ]; then
    echo ""
    echo "Please download the following files manually:"
    echo ""
    echo "  1. ILSVRC2012_img_val.tar (~6.3 GB)"
    echo "     URL: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
    echo "     (requires free account at https://image-net.org/)"
    echo ""
    echo "  2. ILSVRC2012_devkit_t12.tar.gz (~2.5 MB)"
    echo "     URL: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
    echo ""
    echo "  Place both files in: $DATA_DIR"
    echo "  Then re-run:  bash $0 $DATA_DIR extract"
    echo ""
    exit 0
fi

if [ "$METHOD" = "extract" ]; then
    # Only extract, skip download
    echo "Extracting existing archives..."
else
    # Try auto-download from image-net.org (requires wget with cookies)
    echo ""
    echo "Attempting download from image-net.org..."
    echo "If this fails, use manual mode:"
    echo "  bash $0 $DATA_DIR manual"
    echo ""

    # Devkit (small, try first)
    DEVKIT="$DATA_DIR/ILSVRC2012_devkit_t12.tar.gz"
    if [ -f "$DEVKIT" ]; then
        echo "[1/2] Devkit already exists: $DEVKIT"
    else
        echo "[1/2] Downloading devkit..."
        wget -c --show-progress \
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz" \
            -O "$DEVKIT" 2>&1 || {
            echo "  Download failed. Please use manual mode."
            exit 1
        }
    fi

    # Validation images
    VAL_TAR="$DATA_DIR/ILSVRC2012_img_val.tar"
    if [ -f "$VAL_TAR" ]; then
        echo "[2/2] Val images already exist: $VAL_TAR ($(du -h "$VAL_TAR" | cut -f1))"
    else
        echo "[2/2] Downloading ILSVRC2012_img_val.tar (~6.3 GB)..."
        wget -c --show-progress \
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" \
            -O "$VAL_TAR" 2>&1 || {
            echo "  Download failed. Please use manual mode:"
            echo "  bash $0 $DATA_DIR manual"
            exit 1
        }
    fi
fi

# ---- Extract ----
echo ""
echo "============================================"
echo "Extracting..."
echo "============================================"

VAL_DIR="$DATA_DIR/val"
VAL_TAR="$DATA_DIR/ILSVRC2012_img_val.tar"
DEVKIT="$DATA_DIR/ILSVRC2012_devkit_t12.tar.gz"

if [ -f "$VAL_TAR" ]; then
    if [ -d "$VAL_DIR" ] && [ "$(find "$VAL_DIR" -name '*.JPEG' 2>/dev/null | wc -l)" -gt 40000 ]; then
        echo "[1/2] Val images already extracted."
    else
        echo "[1/2] Extracting val images (this takes a few minutes)..."
        mkdir -p "$VAL_DIR"
        tar -xf "$VAL_TAR" -C "$VAL_DIR"
        echo "  Done. ($(find "$VAL_DIR" -name '*.JPEG' 2>/dev/null | wc -l) images)"
    fi
else
    echo "[1/2] Val tar not found — skipping."
fi

if [ -f "$DEVKIT" ]; then
    if [ -f "$DATA_DIR/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt" ]; then
        echo "[2/2] Devkit already extracted."
    else
        echo "[2/2] Extracting devkit..."
        tar -xzf "$DEVKIT" -C "$DATA_DIR"
        echo "  Done."
    fi
else
    echo "[2/2] Devkit not found — class labels will be embedded in eval script."
fi

# ---- Organize val images into class subdirectories ----
GROUND_TRUTH="$DATA_DIR/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
if [ -f "$GROUND_TRUTH" ] && [ -d "$VAL_DIR" ]; then
    echo ""
    echo "Organizing val images into class subdirectories..."
    python3 -c "
import os, shutil
val_dir = '$VAL_DIR'
gt_file = '$GROUND_TRUTH'
with open(gt_file) as f:
    labels = [int(l.strip()) - 1 for l in f]  # 1-indexed to 0-indexed
jpgs = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
for jpg, label in zip(jpgs, labels):
    class_dir = os.path.join(val_dir, f'{label:04d}')
    os.makedirs(class_dir, exist_ok=True)
    src = os.path.join(val_dir, jpg)
    dst = os.path.join(class_dir, jpg)
    if not os.path.exists(dst):
        shutil.move(src, dst)
print(f'  Organized {len(jpgs)} images into class subdirectories')
" 2>&1
fi

echo ""
echo "============================================"
echo "ImageNet setup complete!"
echo "  Images: $VAL_DIR"
echo "============================================"
