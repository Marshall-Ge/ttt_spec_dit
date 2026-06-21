# -*- coding: utf-8 -*-
"""Shared configuration for TeaCache PixArt-α project.

Centralises model paths, dataset paths, coefficients, and defaults.
"""

import os

# ---------------------------------------------------------------------------
# HuggingFace mirror (China)
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Suppress tokenizers parallelism warning from torch-fidelity fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
HF_CACHE_DIR = "/root/autodl-fs/models"
PIXART_REPO = "PixArt-alpha/PixArt-XL-2-512x512"
DIT_REPO = "/root/autodl-fs/models/dit_2_256"
CLIP_PATH = "/root/autodl-fs/models/models/clip"

# ---------------------------------------------------------------------------
# DiT-2-256 constants
# ---------------------------------------------------------------------------
DIT_IMAGE_SIZE = 256
DIT_LATENT_SIZE = 32

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
COCO_DIR = "/root/autodl-fs/data/coco"
IMAGENET_DIR = "/root/autodl-fs/data/imagenet"
IMAGENET_299_DIR = "/root/autodl-fs/data/imagenet/val_299"
COCO_299_DIR = "/root/autodl-fs/data/coco/val_299"
DRAWBENCH_PATH = os.path.join(os.path.dirname(__file__), "drawbench200.txt")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output"

# ---------------------------------------------------------------------------
# CFG & batch defaults
# ---------------------------------------------------------------------------
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_BATCH_SIZE = 4   # max stable on RTX 4090 D (23.5 GB) with PixArt+CFG+SpecA

# ---------------------------------------------------------------------------
# TeaCache defaults
# ---------------------------------------------------------------------------
DEFAULT_REL_L1_THRESH = 0.25
DEFAULT_NUM_STEPS = 20

# DDIM step count that FLOP-matches TeaCache (γ=0.25, ~50% skip) on PixArt-XL-2.
# Single full step ≈ 1.236 T FLOPs; TeaCache accel ≈ 12.43 T → 10 steps.
DDIM_FLOP_MATCHED_STEPS = 10

# ---------------------------------------------------------------------------
# SpecA defaults (from DiT speca-dit)
# ---------------------------------------------------------------------------
SPECA_DEFAULT_BASE_THRESHOLD = 0.01       # cosine error scale [0, 1]
SPECA_DEFAULT_DECAY_RATE = 0.01
SPECA_DEFAULT_MIN_TAYLOR_STEPS = 1
SPECA_DEFAULT_MAX_TAYLOR_STEPS = 4
SPECA_DEFAULT_MAX_ORDER = 4
SPECA_DEFAULT_ERROR_METRIC = "cosine_similarity"   # catches FF errors at check_layer

# ---------------------------------------------------------------------------
# Image save limit (disk-space guard for large-scale c2i generation)
# ---------------------------------------------------------------------------
IMG_SAVE_LIMIT = 50

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
