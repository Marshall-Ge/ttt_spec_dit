# -*- coding: utf-8 -*-
"""Shared configuration for TeaCache PixArt-α project.

Centralises model paths, dataset paths, coefficients, and defaults.
"""

import os

# ---------------------------------------------------------------------------
# HuggingFace mirror (China)
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
HF_CACHE_DIR = "/root/autodl-fs/models"
PIXART_REPO = "PixArt-alpha/PixArt-XL-2-512x512"
CLIP_PATH = "/root/autodl-fs/models/models/clip"

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
COCO_DIR = "/root/autodl-fs/data/coco"
IMAGENET_DIR = "/root/autodl-fs/data/imagenet"
DRAWBENCH_PATH = os.path.join(os.path.dirname(__file__), "drawbench200.txt")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output"

# ---------------------------------------------------------------------------
# TeaCache defaults
# ---------------------------------------------------------------------------
DEFAULT_REL_L1_THRESH = 0.25
DEFAULT_NUM_STEPS = 20

# Where the calibrated PixArt polynomial coefficients are stored.
COEF_PATH = os.path.join(os.path.dirname(__file__), "pixart_coef.json")

# Fallback coefficients (FLUX official). Overwritten by calibration output.
FLUX_COEFFICIENTS = [4.98651651e02, -2.83781631e02, 5.58554382e01,
                     -3.82021401e00, 2.64230861e-01]


def load_coefficients(path: str = None):
    """Load calibrated PixArt coefficients, falling back to FLUX defaults.

    Returns a list of 5 floats (highest degree first, np.poly1d convention).
    """
    import json
    path = path or COEF_PATH
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            coef = data.get("coefficients")
            if coef and len(coef) == 5:
                return coef
        except Exception as e:
            print(f"  [WARN] coef load failed ({e}); using FLUX coefficients")
    return FLUX_COEFFICIENTS
