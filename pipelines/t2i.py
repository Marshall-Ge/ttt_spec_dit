# -*- coding: utf-8 -*-
"""Text-to-Image (t2i) pipeline for PixArt-α evaluation.

Handles:
  - drawbench: 200 natural-language prompts, ImageReward scoring
  - geneval:    553 compositional prompts, GenEval scoring

Usage (via main.py):
    python main.py --task t2i --dataset drawbench --method teacache ...
    python main.py --task t2i --dataset geneval --method baseline ...
"""

import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List

from config import (OUTPUT_DIR, DEFAULT_REL_L1_THRESH, DEFAULT_NUM_STEPS,
                     load_coefficients)
from utils import CudaTimer, decode_latent, save_image

from models.pixart import PixArtGenerator
from models.dit import DiTGenerator
from accelerators.teacache import TeaCacheAccelerator
from accelerators.speca import SpecAAccelerator

from eval.image_reward import ImageRewardScorer
from eval.gen_eval import GenEvalScorer
from eval.latency import LatencyMetric, FLOPsMetric


# ---------------------------------------------------------------------------
# Metric validity
# ---------------------------------------------------------------------------

T2I_VALID_METRICS = {
    "drawbench": {"imagereward", "latency", "flops", "speed"},
    "geneval":   {"geneval", "latency", "flops", "speed"},
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_t2i(args) -> Dict:
    """Run a t2i evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Expected fields: dataset, n_prompts, num_steps, thresh, seed,
        method, metrics, output_dir, coef_path

    Returns
    -------
    dict with keys: config, aggregate, per_prompt, [geneval]
    """
    dataset_name = args.dataset
    model_name = getattr(args, "model", "pixart")
    if dataset_name not in T2I_VALID_METRICS:
        raise ValueError(
            f"Unknown t2i dataset: {dataset_name}. "
            f"Valid: {list(T2I_VALID_METRICS.keys())}")

    valid_metrics = T2I_VALID_METRICS[dataset_name]
    requested = set(args.metrics)
    for m in sorted(requested - valid_metrics):
        print(f"  [WARN] '{m}' is not valid for t2i/{dataset_name} — skipping")
    selected = sorted(requested & valid_metrics)
    if not selected:
        print(f"  [ERROR] No valid metrics remain for t2i/{dataset_name}.")
        print(f"          Valid choices: {sorted(valid_metrics)}")
        return {}

    device = "cuda"
    dtype = torch.float16

    # --- Output dir ---
    # Append num_steps for ddim to avoid DDIM@10/DDIM@20 collision.
    # Keep baseline/teacache dir names unchanged (their results already exist).
    dir_suffix = f"{args.method}_{args.num_steps}" if args.method == "ddim" else args.method
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"t2i_{model_name}_{dataset_name}_{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = load_coefficients(args.coef_path) if args.coef_path else load_coefficients()

    model_label = {"pixart": "PixArt-α", "dit": "DiT-2-256"}.get(model_name, model_name)
    print("=" * 70)
    print(f"{model_label} T2I Evaluation — {dataset_name.upper()}")
    print(f"  Method:   {args.method}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  N:        {args.n_prompts}")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Metrics:  {selected}")
    print(f"  Output:   {output_dir}")
    if args.method == "teacache":
        print(f"  γ:        {args.thresh}")
        print(f"  Coef:     pixart_coef.json")
    if args.method == "speca":
        print(f"  SpecA:    base_thresh={args.speca_base_threshold} "
              f"decay={args.speca_decay_rate} "
              f"taylor=[{args.speca_min_taylor_steps},{args.speca_max_taylor_steps}] "
              f"metric={getattr(args, 'speca_error_metric', 'relative_l1')}")
    print("=" * 70)

    # =====================================================================
    # 1. Load dataset
    # =====================================================================
    print("\n[1] Loading dataset...")

    if dataset_name == "drawbench":
        from dataset.drawbench import DrawBenchDataset
        ds = DrawBenchDataset(n_prompts=args.n_prompts, base_seed=args.seed)
        items = [(ds[i][0], ds[i][1], None) for i in range(len(ds))]
    elif dataset_name == "geneval":
        from dataset.geneval import GenEvalDataset
        n = args.n_prompts if args.n_prompts else 553
        ds = GenEvalDataset(n_prompts=n, base_seed=args.seed)
        items = [ds[i] for i in range(len(ds))]  # (prompt, seed, tag)

    n = len(items)

    # =====================================================================
    # 2. Load model
    # =====================================================================
    if model_name == "pixart":
        print("\n[2] Loading PixArt-α model...")
        generator = PixArtGenerator(
            num_steps=args.num_steps, device=device, dtype=dtype)
    elif model_name == "dit":
        print("\n[2] Loading DiT-2-256 model...")
        generator = DiTGenerator(
            num_steps=args.num_steps, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    generator.load()

    # =====================================================================
    # 3. Setup metrics
    # =====================================================================
    print("\n[3] Setting up metrics...")
    metrics = {}
    need_imagereward = "imagereward" in selected
    need_geneval = "geneval" in selected
    need_flops = "flops" in selected
    need_latency = "latency" in selected or "speed" in selected

    if need_imagereward:
        metrics["imagereward"] = ImageRewardScorer(device=device)
    if need_geneval:
        metrics["geneval"] = GenEvalScorer(device=device)
    if need_flops:
        metrics["flops"] = FLOPsMetric(generator)
        metrics["flops"].profile()  # MUST profile before TeaCache is installed
    if need_latency:
        metrics["latency"] = LatencyMetric()

    # =====================================================================
    # 4. Setup accelerator (if method == teacache)
    # =====================================================================
    accelerator = None
    if args.method == "teacache":
        accelerator = TeaCacheAccelerator(
            num_steps=args.num_steps,
            rel_l1_thresh=args.thresh,
            coefficients=coefficients,
        )
        accelerator.install(generator)
        print(f"  TeaCache installed (γ={args.thresh})")
    elif args.method == "ddim":
        print(f"  DDIM sampling ({args.num_steps} steps, no caching)")
    elif args.method == "speca":
        accelerator = SpecAAccelerator(
            num_steps=args.num_steps,
            base_threshold=args.speca_base_threshold,
            decay_rate=args.speca_decay_rate,
            min_taylor_steps=args.speca_min_taylor_steps,
            max_taylor_steps=args.speca_max_taylor_steps,
            error_metric=getattr(args, "speca_error_metric", "relative_l1"),
        )
        accelerator.install(generator)
        print(f"  SpecA installed (base_thresh={args.speca_base_threshold})")
    else:
        print(f"  Baseline (no TeaCache)")

    # =====================================================================
    # 5. Generate + score  (batch = different prompts in parallel)
    # =====================================================================
    total_images = n  # one image per prompt
    bs = args.batch_size
    print(f"\n[4] Generating {total_images} images ({args.method}, "
          f"{n} prompts in batches of ≤{bs})...")
    t_start = time.time()

    all_results = []
    batch_count = 0
    for batch_start in tqdm(range(0, n, bs), desc=f"t2i/{dataset_name}", ncols=80):
        batch_end = min(batch_start + bs, n)
        batch_items = items[batch_start:batch_end]
        batch_prompts = [it[0] for it in batch_items]
        batch_seeds   = [it[1] for it in batch_items]
        actual_bs = len(batch_prompts)

        # --- Generate one batch ---
        t0 = time.time()
        if args.method in ("teacache", "speca"):
            accelerator.reset()
        if args.method == "ddim":
            latent, img = generator.generate_ddim(
                batch_prompts, batch_seeds,
                guidance_scale=args.guidance_scale)
        else:
            latent, img = generator.generate(
                batch_prompts, batch_seeds,
                guidance_scale=args.guidance_scale)
        wall_s = time.time() - t0

        per_img_s = wall_s / actual_bs

        # --- Batch eval (no per-image loop for scoring) ---
        if need_imagereward:
            metrics["imagereward"].add_batch(img, prompts=batch_prompts)

        if need_geneval:
            metrics["geneval"].add_batch(img, prompts=batch_prompts)

        if need_latency:
            metrics["latency"].add_pairs_batch([per_img_s] * actual_bs,
                                               [per_img_s] * actual_bs)

        # --- Per-prompt results (wall_s, metadata only) ---
        for b, (prompt, seed, tag) in enumerate(batch_items):
            result = {"prompt": prompt, "seed": seed, "wall_s": wall_s,
                      "images": 1}
            if tag is not None:
                result["tag"] = tag
            all_results.append(result)

        # FLOPs accounting (per batch, once per generate call)
        if need_flops:
            if args.method == "teacache":
                metrics["flops"].add_generation(accelerator.teacache)
            elif args.method == "speca":
                metrics["flops"].add_generation(accelerator.speca)
            else:
                metrics["flops"].add_vanilla_steps(args.num_steps)

        batch_count += 1

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min "
          f"({elapsed/total_images:.2f} s/image, {batch_count} batches)")

    # =====================================================================
    # 6. Aggregate metrics
    # =====================================================================
    agg = _aggregate_t2i(all_results, dataset_name)

    if need_imagereward:
        agg.update(metrics["imagereward"].compute())
    if need_geneval:
        geneval_agg = metrics["geneval"].compute()
        agg["geneval_overall"] = geneval_agg.get("geneval_overall", float("nan"))
        for k, v in geneval_agg.items():
            agg[k] = v
    if need_latency:
        agg.update(metrics["latency"].compute())
    if need_flops:
        agg.update(metrics["flops"].compute())

    # Skip ratio
    if args.method == "teacache" and accelerator is not None:
        st = accelerator.stats
        agg["skip_ratio"] = st.get("skip_ratio", 0.0)
        agg["total_calc"] = st.get("total_calc", 0)
        agg["total_skip"] = st.get("total_skip", 0)
    elif args.method == "speca" and accelerator is not None:
        st = accelerator.stats
        agg["skip_ratio"] = st.get("skip_ratio", 0.0)
        agg["taylor_steps"] = st.get("total_taylor", 0)
        agg["full_steps"] = st.get("total_full", 0)
        agg["total_calc"] = st.get("total_full", 0)
        agg["total_skip"] = st.get("total_taylor", 0)

    # Speed (images/sec) — deduplicate wall_s per batch
    if "speed" in selected and all_results:
        unique_walls = list(dict.fromkeys(r["wall_s"] for r in all_results))
        agg["speed_img_per_s"] = float(n / np.sum(unique_walls)) if unique_walls else 0.0

    results = {
        "config": {
            "model": model_name,
            "task": "t2i",
            "dataset": dataset_name,
            "method": args.method,
            "n_prompts": n,
            "batch_size": args.batch_size,
            "total_images": total_images,
            "num_steps": args.num_steps,
            "rel_l1_thresh": args.thresh if args.method == "teacache" else None,
            "coefficients": coefficients if args.method == "teacache" else None,
            "speca_base_threshold": args.speca_base_threshold if args.method == "speca" else None,
            "speca_decay_rate": args.speca_decay_rate if args.method == "speca" else None,
            "speca_min_taylor_steps": args.speca_min_taylor_steps if args.method == "speca" else None,
            "speca_max_taylor_steps": args.speca_max_taylor_steps if args.method == "speca" else None,
            "speca_error_metric": getattr(args, "speca_error_metric", "relative_l1") if args.method == "speca" else None,
        },
        "aggregate": agg,
        "per_prompt": all_results,
    }

    # =====================================================================
    # 7. Save & report
    # =====================================================================
    print("\n[5] Saving results...")

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  Results → {results_path}")

    report = _build_t2i_report(results)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report  → {report_path}")

    # --- Summary ---
    _print_t2i_summary(results, selected, args.method)

    return results


# ===========================================================================
# Aggregation helper
# ===========================================================================

def _aggregate_t2i(per_prompt: List[Dict], dataset_name: str) -> Dict:
    """Compute mean/std over per-prompt numeric fields."""
    if not per_prompt:
        return {"n_prompts": 0}

    numeric_keys = ["wall_s", "imagereward", "geneval"]
    agg = {"n_prompts": len(per_prompt)}
    for k in numeric_keys:
        vals = [r[k] for r in per_prompt if k in r and r[k] is not None
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    return agg


# ===========================================================================
# Report
# ===========================================================================

def _build_t2i_report(results: Dict) -> str:
    """Build Markdown report for t2i results."""
    lines = []
    cfg = results.get("config", {})
    dataset_name = cfg.get("dataset", "?").upper()
    method = cfg.get("method", "?")
    model = cfg.get("model", "pixart")
    model_display = {"pixart": "PixArt-XL-2 512×512", "dit": "DiT-2-256"}.get(model, model)

    lines.append(f"# {model_display} T2I Evaluation: {dataset_name}\n")
    lines.append(f"**Model:** {model_display} | "
                 f"**Method:** {method} | "
                 f"**Steps:** {cfg.get('num_steps')} | "
                 f"**N:** {cfg.get('n_prompts', '?')}\n")
    if method == "teacache":
        lines.append(f"**γ:** {cfg.get('rel_l1_thresh')} | "
                     f"**Coefficients:** `{cfg.get('coefficients')}`\n")
    elif method == "speca":
        lines.append(f"**Base thresh:** {cfg.get('speca_base_threshold', 0.1)} | "
                     f"**Decay:** {cfg.get('speca_decay_rate', 0.01)} | "
                     f"**Taylor steps:** {cfg.get('speca_min_taylor_steps', 2)}–"
                     f"{cfg.get('speca_max_taylor_steps', 5)}\n")
    lines.append("---\n")

    agg = results.get("aggregate", {})

    lines.append("## Results\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    # Latency
    lat_mean = agg.get("latency_vanilla_mean", None) or agg.get("wall_s_mean", 0)
    lat_std = agg.get("latency_vanilla_std", 0) or agg.get("wall_s_std", 0)
    if lat_mean:
        lines.append(f"| **Latency (s/image) ↓** | {lat_mean:.3f} ± {lat_std:.3f} |")

    # FLOPs
    flops_v = agg.get("flops_vanilla_T")
    flops_a = agg.get("flops_accel_T")
    if flops_v is not None and not np.isnan(flops_v) and flops_v > 0:
        line = f"| **FLOPs (T) ↓** | vanilla: {flops_v:.3f}"
        if flops_a is not None and not np.isnan(flops_a):
            line += f" / accel: {flops_a:.3f}"
        line += " |"
        lines.append(line)

    # Speed
    speed = agg.get("speed_img_per_s")
    if speed:
        lines.append(f"| **Speed ↑** | {speed:.3f} img/s |")

    # ImageReward
    if "imagereward_mean" in agg:
        lines.append(f"| **ImageReward ↑** | {agg['imagereward_mean']:.3f} ± {agg.get('imagereward_std', 0):.3f} |")

    # GenEval
    if dataset_name == "GENEVAL":
        gv = agg.get("geneval_overall")
        if gv is not None and not np.isnan(gv):
            lines.append(f"| **GenEval Overall ↑** | {gv:.1%} |")
            for key, label in [
                ("geneval_single_object", "  Single Object"),
                ("geneval_two_object", "  Two Object"),
                ("geneval_counting", "  Counting"),
                ("geneval_colors", "  Colors"),
                ("geneval_position", "  Position"),
                ("geneval_attribute_binding", "  Attr. Binding"),
            ]:
                v = agg.get(key)
                if v is not None and not np.isnan(v):
                    lines.append(f"| {label} | {v:.1%} |")

    # Skip ratio
    sr = agg.get("skip_ratio")
    if sr is not None and sr > 0:
        lines.append(f"| **Skip Ratio** | {sr:.0%} ({agg.get('total_skip', 0)}/{agg.get('total_calc', 0) + agg.get('total_skip', 0)} steps) |")

    lines.append("")
    lines.append("## Method\n")
    if method == "teacache":
        lines.append(f"TeaCache (Liu et al., CVPR 2025) accelerated {model_display} generation.\n")
    elif method == "ddim":
        lines.append("DDIM (Song et al., 2021) step-skipping baseline — full 28-block forward every step, "
                     f"only the number of sampling steps is reduced to {cfg.get('num_steps')}.\n")
    elif method == "speca":
        lines.append("SpecA (Speculative Acceleration) — Taylor-series feature prediction "
                     "with adaptive full/Taylor step selection and last-block error checking.\n")
    else:
        lines.append(f"{model_display} baseline (all steps computed).\n")

    return "\n".join(lines)


# ===========================================================================
# Summary printer
# ===========================================================================

def _print_t2i_summary(results: Dict, selected_metrics: List[str], method: str):
    """Print a console summary."""
    agg = results.get("aggregate", {})
    cfg = results.get("config", {})
    dataset_name = cfg.get("dataset", "?")

    print("\n" + "=" * 70)
    print(f"Summary — t2i/{dataset_name}/{method}")
    print("=" * 70)

    if "latency" in selected_metrics:
        lat = agg.get("latency_vanilla_mean", None) or agg.get("wall_s_mean", 0)
        print(f"  Latency (s/img): {lat:.3f}")

    if "flops" in selected_metrics:
        fv = agg.get("flops_vanilla_T")
        fa = agg.get("flops_accel_T")
        if fv is not None and not np.isnan(fv):
            if fa is not None and not np.isnan(fa) and abs(fa - fv) > 1e-9:
                red = agg.get("flops_reduction", 0)
                print(f"  FLOPs (T):       vanilla={fv:.3f}  accel={fa:.3f}  ↓{red:.0%}")
            else:
                print(f"  FLOPs (T):       {fv:.3f}")

    if "speed" in selected_metrics:
        s = agg.get("speed_img_per_s")
        if s:
            print(f"  Speed:           {s:.3f} img/s")

    if "imagereward" in selected_metrics:
        print(f"  ImageReward:     {agg.get('imagereward_mean', 0):.3f} ± {agg.get('imagereward_std', 0):.3f}")

    if "geneval" in selected_metrics:
        gv = agg.get("geneval_overall", float("nan"))
        print(f"  GenEval overall: {gv:.1%}" if not np.isnan(gv) else "  GenEval overall: NaN")

    if method == "teacache":
        sr = agg.get("skip_ratio", 0)
        print(f"  Skip ratio:      {sr:.0%}")
    elif method == "speca":
        sr = agg.get("skip_ratio", 0)
        print(f"  Taylor ratio:    {sr:.0%} ({agg.get('taylor_steps', 0)}/{agg.get('taylor_steps', 0) + agg.get('full_steps', 0)})")
    print("=" * 70)


# ===========================================================================
# JSON cleaner
# ===========================================================================

def _clean(o):
    """Strip non-serializable values (numpy, torch)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, torch.Tensor):
        return None
    return o
