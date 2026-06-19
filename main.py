# -*- coding: utf-8 -*-
"""
PixArt-α Evaluation Pipeline v2 — Thin CLI entry point.

Two task types × two methods:

    t2i (text-to-image):
      python main.py --task t2i --dataset drawbench --method teacache ...
      python main.py --task t2i --dataset geneval --method baseline ...

    c2i (class/caption-to-image):
      python main.py --task c2i --dataset coco --method teacache ...
      python main.py --task c2i --dataset imagenet --method baseline ...
"""

import argparse
import sys

from config import (DEFAULT_NUM_STEPS, DEFAULT_REL_L1_THRESH,
                     COCO_DIR, IMAGENET_DIR)

# ---------------------------------------------------------------------------
# Valid task/dataset combos
# ---------------------------------------------------------------------------

TASK_DATASET_MAP = {
    "t2i": ["drawbench", "geneval"],
    "c2i": ["coco", "imagenet"],
}

ALL_METRICS = {
    "imagereward", "geneval", "fid", "is",
    "clip", "lpips", "mse",
    "latency", "flops", "speed",
}

DATASET_DEFAULTS = {
    "drawbench": 200,
    "geneval": 553,
    "coco": 30000,
    "imagenet": 50000,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PixArt-α Evaluation Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # t2i + drawbench + baseline
  python main.py --task t2i --dataset drawbench --n_prompts 5 \\
      --method baseline --metrics imagereward latency flops speed

  # t2i + geneval + teacache
  python main.py --task t2i --dataset geneval --n_prompts 10 \\
      --method teacache --thresh 0.25 \\
      --metrics geneval latency flops speed

  # c2i + coco + teacache
  python main.py --task c2i --dataset coco --n_prompts 100 \\
      --method teacache --thresh 0.25 \\
      --metrics fid is clip lpips mse latency flops speed

  # c2i + imagenet + baseline
  python main.py --task c2i --dataset imagenet --n_prompts 50 \\
      --method baseline --metrics fid is latency flops speed
        """,
    )

    # ---- Task / Dataset / Method ----
    parser.add_argument("--task", type=str, required=True,
                        choices=["t2i", "c2i"],
                        help="Task type: t2i (text-to-image) or c2i (class-to-image)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["drawbench", "geneval", "coco", "imagenet"],
                        help="Evaluation dataset")
    parser.add_argument("--method", type=str, default="teacache",
                        choices=["baseline", "teacache"],
                        help="Acceleration method (default: teacache)")
    parser.add_argument("--n_prompts", type=int, default=None,
                        help="Number of prompts/images (default varies by dataset)")

    # ---- Model ----
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"Denoising steps (default: {DEFAULT_NUM_STEPS})")
    parser.add_argument("--thresh", type=float, default=DEFAULT_REL_L1_THRESH,
                        help=f"TeaCache threshold γ (default: {DEFAULT_REL_L1_THRESH})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ---- Metrics ----
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["latency", "flops", "speed"],
                        help="Metrics to compute (see docs for valid combos)")

    # ---- Paths ----
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--coef_path", type=str, default=None,
                        help="Path to TeaCache coefficient JSON")
    parser.add_argument("--coco_dir", type=str, default=COCO_DIR,
                        help="COCO dataset root")
    parser.add_argument("--imagenet_dir", type=str, default=IMAGENET_DIR,
                        help="ImageNet dataset root")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args):
    """Validate task/dataset combo and set defaults. Returns True if valid."""
    # Task + dataset
    valid_datasets = TASK_DATASET_MAP.get(args.task, [])
    if args.dataset not in valid_datasets:
        print(f"[ERROR] --task {args.task} does not support --dataset {args.dataset}.")
        print(f"        Valid task/dataset combos:")
        for task, dss in TASK_DATASET_MAP.items():
            for ds in dss:
                print(f"          --task {task} --dataset {ds}")
        return False

    # Default n_prompts
    if args.n_prompts is None:
        args.n_prompts = DATASET_DEFAULTS.get(args.dataset, 200)
        print(f"  [INFO] --n_prompts defaulting to {args.n_prompts} for {args.dataset}")

    # Method
    if args.method == "baseline" and args.thresh != DEFAULT_REL_L1_THRESH:
        print(f"  [INFO] --thresh is ignored when --method baseline")

    # Metrics: warn about unknown, but don't remove yet (pipelines filter)
    unknown = set(args.metrics) - ALL_METRICS
    if unknown:
        print(f"  [WARN] Unknown metrics: {sorted(unknown)} — will be skipped")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not validate_args(args):
        sys.exit(1)

    # ---- Dispatch ----
    if args.task == "t2i":
        from pipelines.t2i import run_t2i
        run_t2i(args)
    elif args.task == "c2i":
        from pipelines.c2i import run_c2i
        run_c2i(args)


if __name__ == "__main__":
    main()
