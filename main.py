# -*- coding: utf-8 -*-
import argparse
import sys

from config import *

# ---------------------------------------------------------------------------
# Valid task/dataset combos
# ---------------------------------------------------------------------------

TASK_DATASET_MAP = {
    "t2i": ["drawbench", "geneval"],
    "c2i": ["coco", "imagenet"],
}

# Model constraints
MODEL_TASK_MAP = {
    "pixart": ["t2i", "c2i"],
    "dit": ["c2i"],  # DiT is class-conditional, no text encoder
}
MODEL_DATASET_MAP = {
    ("dit", "c2i"): ["imagenet"],  # DiT only supports ImageNet classes
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

  # t2i + drawbench + speca
  python main.py --task t2i --dataset drawbench --n_prompts 1 \\
      --method speca --num_steps 20 \\
      --speca_base_threshold 0.1 --speca_decay_rate 0.01 \\
      --speca_min_taylor_steps 2 --speca_max_taylor_steps 5 \\
      --metrics imagereward latency flops speed
        """,
    )

    # ---- Task / Dataset / Method ----
    parser.add_argument("--model", type=str, default="pixart",
                        choices=["pixart", "dit"],
                        help="Base model (default: pixart)")
    parser.add_argument("--task", type=str, required=True,
                        choices=["t2i", "c2i"],
                        help="Task type: t2i (text-to-image) or c2i (class-to-image)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["drawbench", "geneval", "coco", "imagenet"],
                        help="Evaluation dataset")
    parser.add_argument("--method", type=str, default="teacache",
                        choices=["baseline", "teacache", "ddim", "speca"],
                        help="Acceleration/sampling method (default: teacache). "
                             "baseline=DPMSolver++ full, ddim=DDIM step-skipping, "
                             "speca=Speculative Acceleration")
    parser.add_argument("--n_prompts", type=int, default=None,
                        help="Number of prompts/images (default varies by dataset)")

    # ---- Model ----
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"Denoising steps (default: {DEFAULT_NUM_STEPS})")
    parser.add_argument("--thresh", type=float, default=DEFAULT_REL_L1_THRESH,
                        help=f"TeaCache threshold γ (default: {DEFAULT_REL_L1_THRESH})")
    # SpecA hyperparameters
    parser.add_argument("--speca_base_threshold", type=float,
                        default=SPECA_DEFAULT_BASE_THRESHOLD,
                        help=f"SpecA base threshold (default: {SPECA_DEFAULT_BASE_THRESHOLD})")
    parser.add_argument("--speca_decay_rate", type=float,
                        default=SPECA_DEFAULT_DECAY_RATE,
                        help=f"SpecA decay rate (default: {SPECA_DEFAULT_DECAY_RATE})")
    parser.add_argument("--speca_min_taylor_steps", type=int,
                        default=SPECA_DEFAULT_MIN_TAYLOR_STEPS,
                        help=f"SpecA min Taylor steps (default: {SPECA_DEFAULT_MIN_TAYLOR_STEPS})")
    parser.add_argument("--speca_max_taylor_steps", type=int,
                        default=SPECA_DEFAULT_MAX_TAYLOR_STEPS,
                        help=f"SpecA max Taylor steps (default: {SPECA_DEFAULT_MAX_TAYLOR_STEPS})")
    parser.add_argument("--speca_error_metric", type=str,
                        default=SPECA_DEFAULT_ERROR_METRIC,
                        choices=["l1", "l2", "relative_l1", "relative_l2",
                                 "cosine_similarity", "all"],
                        help="SpecA error metric for gate/threshold comparison "
                             "(default: cosine_similarity)")
    parser.add_argument("--speca_check_layer", type=int, default=None,
                        help="DiT SpecA probe layer index (default: 20)")
    parser.add_argument("--compute-controller", type=str, default="none",
                        choices=["none", "probe_correct"],
                        help="Verification-guided compute controller")
    parser.add_argument("--controller-correction-policy", type=str,
                        default="reject", choices=["reject", "always"],
                        help="When probe_correct adopts the verified full output")
    parser.add_argument("--controller-suffix-blocks", type=int, default=0,
                        help="Additional full blocks after a verified probe")
    parser.add_argument("--controller-suffix-budget", type=int, default=0,
                        help="Per-trajectory budget for suffix full blocks")
    # ---- COVR-Spec counterfactual shadow auditing (DiT + SpecA) ----
    parser.add_argument("--covr-shadow", action="store_true", default=False,
                        help="Record paired Taylor/full counterfactual labels "
                             "without changing the SpecA trajectory")
    parser.add_argument("--covr-output-dir", type=str, default=None,
                        help="COVR event directory (default: <output_dir>/covr)")
    parser.add_argument("--covr-session-id", type=str, default=None,
                        help="Stable COVR session ID (default: timestamp + seed)")
    parser.add_argument("--covr-max-events", type=int, default=None,
                        help="Maximum scalar COVR events to record")
    parser.add_argument("--covr-base-model-version", type=str, default=None,
                        help="Base model version recorded in COVR event metadata")
    # ---- TTT (Test-Time Training plugin, DiT-only) ----
    parser.add_argument("--ttt", action="store_true", default=False,
                        help="Enable online TTT plugin on top of TeaCache "
                             "(DiT-only). Uses --ttt_lr and --ttt_micro_epochs.")
    parser.add_argument("--ttt_lr", type=float, default=1e-4,
                        help="TTT plugin AdamW learning rate (default: 1e-4)")
    parser.add_argument("--ttt_micro_epochs", type=int, default=3,
                        help="Per calc-step micro-epochs to reuse teacher signal "
                             "(default: 3). 1=single-pass, 3-5=better efficiency.")
    # ---- VFL (Verification Feedback Loop) ----
    parser.add_argument("--vfl", action="store_true", default=False,
                        help="Enable VFL (Verification Feedback Loop) for online "
                             "cache-decision improvement. Works alongside "
                             "speca/teacache methods.")
    parser.add_argument("--vfl_output_dir", type=str, default=None,
                        help="VFL output directory (default: <output_dir>/vfl)")
    parser.add_argument("--vfl-no-train", action="store_true", default=False,
                        help="Threshold-only mode: collect scalar errors for "
                             "the OnlineCalibrator (no GPU→CPU tensor "
                             "transfer, no buffer writes) and skip all "
                             "training (no deepcopy, no LoRA, no background "
                             "worker). Combine with --vfl for the lightweight "
                             "threshold-only baseline.")
    parser.add_argument("--vfl-no-time-lora", action="store_true", default=False,
                        help="Disable time-conditioned LoRA modulation "
                             "(γ(t_emb) gating on the A→B bottleneck). "
                             "Default: time-conditioning is ON; pass this "
                             "flag to fall back to vanilla LoRA.")
    parser.add_argument("--vfl-lambda-identity", type=float, default=1.0,
                        help="Weight for L_identity (no-op regularization) "
                             "in VFL L3 loss. L_total = (1 + lambda_identity) "
                             "* MSE(lora_hidden, true_feature) + lambda_anchor "
                             "* L_anchor (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help=f"CFG guidance scale (default: {DEFAULT_GUIDANCE_SCALE}). "
                             f"1.0 = no CFG.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Max prompts per generate() call — different prompts "
                             f"batched in parallel (default: {DEFAULT_BATCH_SIZE})")

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
    parser.add_argument("--img_save_limit", type=int, default=IMG_SAVE_LIMIT,
                        help=f"Max generated images to save to disk "
                             f"(default: {IMG_SAVE_LIMIT}). Set to a large "
                             f"number for full saves.")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode: 1-layer DiT + MPS + fp32, for "
                             "local smoke-test only. Metrics are meaningless.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args):
    """Validate task/dataset combo and set defaults. Returns True if valid."""
    # Model + task
    valid_tasks = MODEL_TASK_MAP.get(args.model, [])
    if args.task not in valid_tasks:
        print(f"[ERROR] --model {args.model} does not support --task {args.task}.")
        print(f"        Valid tasks for {args.model}: {valid_tasks}")
        return False

    # Model + task + dataset
    model_task_key = (args.model, args.task)
    if model_task_key in MODEL_DATASET_MAP:
        valid_dss = MODEL_DATASET_MAP[model_task_key]
        if args.dataset not in valid_dss:
            print(f"[ERROR] --model {args.model} + --task {args.task} "
                  f"only supports --dataset in {valid_dss}.")
            return False

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
    if args.method == "ddim" and args.thresh != DEFAULT_REL_L1_THRESH:
        print(f"  [INFO] --thresh is ignored when --method ddim")
    if args.method == "speca" and args.thresh != DEFAULT_REL_L1_THRESH:
        print(f"  [INFO] --thresh is ignored when --method speca")
    if getattr(args, "speca_check_layer", None) is not None:
        if args.model != "dit" or args.method != "speca":
            print("[ERROR] --speca_check_layer requires --model dit --method speca.")
            return False
    if args.compute_controller != "none":
        if args.model != "dit" or args.method != "speca":
            print("[ERROR] --compute-controller currently requires "
                  "--model dit --method speca.")
            return False
    elif args.controller_correction_policy != "reject":
        print("  [INFO] --controller-correction-policy is ignored when "
              "--compute-controller none")

    suffix_blocks = getattr(args, "controller_suffix_blocks", 0)
    suffix_budget = getattr(args, "controller_suffix_budget", 0)
    if suffix_blocks < 0 or suffix_budget < 0:
        print("[ERROR] suffix blocks and budget must be non-negative.")
        return False
    if (suffix_blocks == 0) != (suffix_budget == 0):
        print("[ERROR] suffix blocks and budget must both be zero or positive.")
        return False
    if suffix_blocks > 0 and args.compute_controller == "none":
        print("[ERROR] suffix recompute requires --compute-controller probe_correct.")
        return False

    if args.covr_shadow:
        if args.model != "dit" or args.method != "speca":
            print("[ERROR] --covr-shadow requires --model dit --method speca.")
            return False
        if args.compute_controller != "none":
            print("[ERROR] --covr-shadow cannot be combined with the legacy "
                  "--compute-controller experiment.")
            return False
        if args.vfl:
            print("[ERROR] --covr-shadow cannot be combined with VFL; Gate data "
                  "must use an unchanged backbone and static SpecA policy.")
            return False
    if args.covr_max_events is not None and args.covr_max_events <= 0:
        print("[ERROR] --covr-max-events must be positive.")
        return False

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
    if args.model == "dit":
        from run_dit import run_c2i
        run_c2i(args)
    elif args.model == "pixart":
        if args.task == "t2i":
            from run_pixart import run_t2i
            run_t2i(args)
        elif args.task == "c2i":
            from run_pixart import run_c2i
            run_c2i(args)
        else:
            print(f"[ERROR] Unknown task: {args.task}")
            sys.exit(1)
    else:
        print(f"[ERROR] Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
