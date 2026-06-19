#!/bin/bash
# ===========================================================================
# Download MS-COCO 2014 validation set for COCO 30K FID/IS evaluation.
#
# COCO 30K FID protocol (standard for PixArt-α, DALL-E 2, Imagen, SD, etc.):
#   - 30,000 images + captions randomly sampled from COCO 2014 validation set
#   - Real reference: the same 30K real images
#   - Generated: 30K images from the 30K captions using model + acceleration
#
# Downloads (total ~6.3 GB):
#   1. val2014.zip          (~6.2 GB, 40,504 images)
#   2. annotations_trainval2014.zip  (~241 MB, captions)
#
# Target directory: /root/autodl-fs/data/coco
# ===========================================================================

set -euo pipefail

DATA_DIR="${1:-/root/autodl-fs/data/coco}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "Downloading MS-COCO 2014 to: $DATA_DIR"
echo "============================================"

# ---- Annotations (small: ~241 MB) ----
ANNO_ZIP="$DATA_DIR/annotations_trainval2014.zip"
if [ -f "$ANNO_ZIP" ]; then
    echo "[1/2] Annotations zip already exists: $ANNO_ZIP ($(du -h "$ANNO_ZIP" | cut -f1))"
else
    echo "[1/2] Downloading annotations_trainval2014.zip (~241 MB)..."
    wget -c --show-progress \
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
        -O "$ANNO_ZIP"
    echo "  Done."
fi

# ---- Validation images (big: ~6.2 GB) ----
VAL_ZIP="$DATA_DIR/val2014.zip"
if [ -f "$VAL_ZIP" ]; then
    echo "[2/2] Val images zip already exists: $VAL_ZIP ($(du -h "$VAL_ZIP" | cut -f1))"
else
    echo "[2/2] Downloading val2014.zip (~6.2 GB, may take a while)..."
    wget -c --show-progress \
        "http://images.cocodataset.org/zips/val2014.zip" \
        -O "$VAL_ZIP"
    echo "  Done."
fi

# ---- Extract ----
echo ""
echo "============================================"
echo "Extracting..."
echo "============================================"

ANNO_DIR="$DATA_DIR/annotations"
if [ -d "$ANNO_DIR" ] && [ -f "$ANNO_DIR/captions_val2014.json" ]; then
    echo "[1/2] Annotations already extracted."
else
    echo "[1/2] Extracting annotations..."
    unzip -qo "$ANNO_ZIP" -d "$DATA_DIR"
    echo "  Done."
fi

VAL_DIR="$DATA_DIR/val2014"
if [ -d "$VAL_DIR" ] && [ "$(ls "$VAL_DIR"/*.jpg 2>/dev/null | wc -l)" -gt 30000 ]; then
    echo "[2/2] Val images already extracted ($(ls "$VAL_DIR"/*.jpg 2>/dev/null | wc -l) images)."
else
    echo "[2/2] Extracting val2014.zip (this will take a minute)..."
    unzip -qo "$VAL_ZIP" -d "$DATA_DIR"
    echo "  Done. ($(ls "$VAL_DIR"/*.jpg 2>/dev/null | wc -l) images)"
fi

echo ""
echo "============================================"
echo "COCO 2014 download complete!"
echo "  Images:      $VAL_DIR"
echo "  Captions:    $ANNO_DIR/captions_val2014.json"
echo "============================================"
echo ""
echo "Next step: run COCO 30K evaluation"
echo "  python -m phase2.eval_coco --help"
