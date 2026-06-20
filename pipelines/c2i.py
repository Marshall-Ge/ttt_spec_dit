# -*- coding: utf-8 -*-
"""Class/Caption-to-Image (c2i) pipeline for PixArt-α evaluation.

Handles:
  - coco:     COCO 30K captions → generated images vs real COCO
  - imagenet: ImageNet 50K class prompts → generated images vs real ImageNet

Usage (via main.py):
    python main.py --task c2i --dataset coco --method teacache ...
    python main.py --task c2i --dataset imagenet --method baseline ...
"""

import json
import os
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, List

from config import (OUTPUT_DIR, COCO_DIR, IMAGENET_DIR,
                     DEFAULT_REL_L1_THRESH, DEFAULT_NUM_STEPS,
                     load_coefficients)
from utils import CudaTimer, save_image, pil_to_tensor

from models.pixart import PixArtGenerator
from models.dit import DiTGenerator
from accelerators.teacache import TeaCacheAccelerator
from accelerators.speca import SpecAAccelerator

from eval.fid_is import FIDISComputer
from eval.clip_score import CLIPScorer
from eval.lpips import LPIPSScorer
from eval.mse import MSEMetric, compute_pixel_mse
from eval.latency import LatencyMetric, FLOPsMetric


# ---------------------------------------------------------------------------
# Metric validity
# ---------------------------------------------------------------------------

C2I_VALID_METRICS = {
    "coco":     {"fid", "is", "clip", "lpips", "mse", "latency", "flops", "speed"},
    "imagenet": {"fid", "is", "latency", "flops", "speed"},
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_c2i(args) -> Dict:
    """Run a c2i evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Expected fields: dataset, n_prompts, num_steps, thresh, seed,
        method, metrics, output_dir, coef_path

    Returns
    -------
    dict with keys: config, aggregate, [fid], [is], [clip], [lpips], [mse]
    """
    dataset_name = args.dataset
    model_name = getattr(args, "model", "pixart")
    if dataset_name not in C2I_VALID_METRICS:
        raise ValueError(
            f"Unknown c2i dataset: {dataset_name}. "
            f"Valid: {list(C2I_VALID_METRICS.keys())}")

    valid_metrics = C2I_VALID_METRICS[dataset_name]
    requested = set(args.metrics)
    for m in sorted(requested - valid_metrics):
        print(f"  [WARN] '{m}' is not valid for c2i/{dataset_name} — skipping")
    selected = sorted(requested & valid_metrics)
    if not selected:
        print(f"  [ERROR] No valid metrics remain for c2i/{dataset_name}.")
        print(f"          Valid choices: {sorted(valid_metrics)}")
        return {}

    device = "cuda"
    dtype = torch.float16

    # --- Output dir ---
    # Append num_steps for ddim to avoid DDIM@10/DDIM@20 collision.
    # Keep baseline/teacache dir names unchanged (their results already exist).
    dir_suffix = f"{args.method}_{args.num_steps}" if args.method == "ddim" else args.method
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"c2i_{model_name}_{dataset_name}_{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = load_coefficients(args.coef_path) if args.coef_path else load_coefficients()

    model_label = {"pixart": "PixArt-α", "dit": "DiT-2-256"}.get(model_name, model_name)
    print("=" * 70)
    print(f"{model_label} C2I Evaluation — {dataset_name.upper()}")
    print(f"  Method:   {args.method}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  N:        {args.n_prompts}")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Metrics:  {selected}")
    print(f"  Output:   {output_dir}")
    if args.method == "teacache":
        print(f"  γ:        {args.thresh}")
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

    if dataset_name == "coco":
        from dataset.coco import COCO30KDataset
        ds = COCO30KDataset(
            coco_dir=getattr(args, "coco_dir", COCO_DIR),
            n_images=args.n_prompts, seed=args.seed)
    elif dataset_name == "imagenet":
        from dataset.imagenet import ImageNetDataset
        ds = ImageNetDataset(
            imagenet_dir=getattr(args, "imagenet_dir", IMAGENET_DIR),
            n_images=args.n_prompts, seed=args.seed)

    n = len(ds)

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
    need_fid = "fid" in selected
    need_is = "is" in selected
    need_clip = "clip" in selected
    need_lpips = "lpips" in selected
    need_mse = "mse" in selected
    need_flops = "flops" in selected
    need_latency = "latency" in selected or "speed" in selected
    need_fid_is = need_fid or need_is

    if need_clip:
        metrics["clip"] = CLIPScorer(device=device, dtype=dtype)
    if need_lpips:
        metrics["lpips"] = LPIPSScorer(device=device)
    if need_mse:
        metrics["mse"] = MSEMetric(which="pixel")
    if need_flops:
        metrics["flops"] = FLOPsMetric(generator)
        metrics["flops"].profile()  # MUST profile BEFORE TeaCache install
    if need_latency:
        metrics["latency"] = LatencyMetric()

    # =====================================================================
    # 4. Setup accelerator
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
    # 5. Generate images  (batch = different prompts in parallel)
    # =====================================================================
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    total_images = n  # one image per prompt
    bs = args.batch_size
    print(f"\n[4] Generating {total_images} images ({args.method}, "
          f"{n} prompts in batches of ≤{bs})...")
    t_start = time.time()

    wall_times = []       # per-batch wall times (deduplicated)
    all_results = []
    global_idx = 0        # running image index

    for batch_start in tqdm(range(0, n, bs), desc=f"c2i/{dataset_name}", ncols=80):
        batch_end = min(batch_start + bs, n)
        batch_indices = list(range(batch_start, batch_end))
        actual_bs = len(batch_indices)

        # Collect prompts and seeds for this batch
        batch_inputs = []
        batch_seeds = []
        for idx in batch_indices:
            data = ds[idx]
            gen_input = data[2] if len(data) > 2 and model_name == "dit" else data[1]
            batch_inputs.append(gen_input)
            batch_seeds.append(100000 + idx)

        # --- Generate one batch ---
        t0 = time.time()
        if args.method in ("teacache", "speca"):
            accelerator.reset()
        if args.method == "ddim":
            latent, img = generator.generate_ddim(
                batch_inputs, batch_seeds,
                guidance_scale=args.guidance_scale)
        else:
            latent, img = generator.generate(
                batch_inputs, batch_seeds,
                guidance_scale=args.guidance_scale)
        wall_s = time.time() - t0
        wall_times.append(wall_s)

        per_img_s = wall_s / actual_bs

        # --- Save (per-image, only this loop iterates per-image) ---
        img_limit = getattr(args, "img_save_limit", 50)
        for b, idx in enumerate(batch_indices):
            if global_idx < img_limit:
                out_path = os.path.join(gen_dir, f"{global_idx:06d}.png")
                save_image(img[b:b+1], out_path)
            global_idx += 1

        # --- Batch eval ---
        if need_clip:
            batch_prompts_text = [ds[idx][1] for idx in batch_indices]
            metrics["clip"].add_batch(img, prompts=batch_prompts_text)

        if need_lpips or need_mse:
            real_tensors = []
            for idx in batch_indices:
                _img_path = ds[idx][0]
                try:
                    real_pil = Image.open(_img_path).convert("RGB")
                    gen_size = (img.shape[-1], img.shape[-2])
                    real_pil = real_pil.resize(gen_size, Image.BICUBIC)
                    real_tensors.append(pil_to_tensor(real_pil).to(device))
                except Exception:
                    real_tensors.append(None)

            valid_indices = [i for i, rt in enumerate(real_tensors) if rt is not None]
            if valid_indices:
                valid_imgs = img[valid_indices]
                valid_refs = torch.stack([real_tensors[i] for i in valid_indices])
                if need_lpips:
                    metrics["lpips"].add_batch(valid_imgs, valid_refs)
                if need_mse:
                    metrics["mse"].add_batch(valid_imgs, valid_refs)

        if need_latency:
            metrics["latency"].add_pairs_batch([per_img_s] * actual_bs,
                                               [per_img_s] * actual_bs)

        if need_flops:
            if args.method == "teacache":
                metrics["flops"].add_generation(accelerator.teacache)
            elif args.method == "speca":
                metrics["flops"].add_generation(accelerator.speca)
            else:
                metrics["flops"].add_vanilla_steps(args.num_steps)

        for idx in batch_indices:
            data = ds[idx]
            all_results.append({
                "idx": idx,
                "prompt": data[1][:120],
                "wall_s": wall_s,
                "images": 1,
            })

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min "
          f"({elapsed/total_images:.2f} s/image, {len(wall_times)} batches)")

    # =====================================================================
    # 6. FID/IS computation
    # =====================================================================
    fid_is_results = {}
    if need_fid_is:
        real_299_dir = _prepare_real_dir(ds, output_dir, n)

        # Resize first (and only) image per prompt (n total) for FID
        gen_299_dir = os.path.join(output_dir, "generated_299")
        os.makedirs(gen_299_dir, exist_ok=True)
        print(f"\n  Resizing {n} generated images (first per prompt) to 299×299...")
        for idx in tqdm(range(n), desc="resize gen", ncols=80):
            src_idx = idx  # one image per prompt, global index = prompt index
            out_path = os.path.join(gen_299_dir, f"{idx:06d}.png")
            if os.path.exists(out_path):
                continue
            src_path = os.path.join(gen_dir, f"{src_idx:06d}.png")
            if not os.path.exists(src_path):
                continue
            pil = Image.open(src_path).convert("RGB")
            pil = pil.resize((299, 299), Image.BICUBIC)
            pil.save(out_path)

        # Compute FID/IS
        fid_computer = FIDISComputer(
            real_dir=real_299_dir,
            gen_dir=gen_299_dir,
        )
        fid_is_results = fid_computer.compute()

    # =====================================================================
    # 7. Aggregate
    # =====================================================================
    agg: Dict = {"n_images": total_images}

    # Wall-time stats — wall_s is per generate() call (B different prompts)
    if wall_times:
        agg["wall_s_mean"] = float(np.mean(wall_times))
        agg["wall_s_std"] = float(np.std(wall_times))
        agg["speed_img_per_s"] = float(n / np.sum(wall_times)) if wall_times else 0.0

    # Metric objects
    if need_clip:
        agg.update(metrics["clip"].compute())
    if need_lpips:
        agg.update(metrics["lpips"].compute())
    if need_mse:
        agg.update(metrics["mse"].compute())
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

    results = {
        "config": {
            "model": model_name,
            "task": "c2i",
            "dataset": dataset_name,
            "method": args.method,
            "n_prompts": n,
            "batch_size": args.batch_size,
            "n_images": total_images,
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
    }

    if fid_is_results:
        results["fid_is"] = fid_is_results

    # =====================================================================
    # 8. Save & report
    # =====================================================================
    print("\n[5] Saving results...")

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  Results → {results_path}")

    report = _build_c2i_report(results, selected)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report  → {report_path}")

    # --- Summary ---
    _print_c2i_summary(results, selected, args.method)

    return results


# ===========================================================================
# Helpers
# ===========================================================================

def _prepare_real_dir(dataset, output_dir: str, n: int) -> str:
    """Resize real reference images to 299×299 flat directory.

    Returns path to the flat directory.
    """
    real_dir = os.path.join(output_dir, "real_299")
    os.makedirs(real_dir, exist_ok=True)

    # Count existing
    existing = len([f for f in os.listdir(real_dir) if f.endswith(".png")])
    if existing >= n:
        print(f"  Real images already prepared ({existing} PNGs)")
        return real_dir

    print(f"\n  Preparing {n} real images → 299×299...")
    for idx in tqdm(range(n), desc="real→299", ncols=80):
        out_path = os.path.join(real_dir, f"{idx:06d}.png")
        if os.path.exists(out_path):
            continue
        img_path, _ = dataset[idx][0], dataset[idx][1]
        try:
            pil = Image.open(img_path).convert("RGB")
            pil = pil.resize((299, 299), Image.BICUBIC)
            pil.save(out_path)
        except Exception as e:
            print(f"\n  [WARN] Failed to process {img_path}: {e}")
    return real_dir


# ===========================================================================
# Report
# ===========================================================================

def _build_c2i_report(results: Dict, selected_metrics: List[str]) -> str:
    """Build Markdown report for c2i results."""
    lines = []
    cfg = results.get("config", {})
    dataset_name = cfg.get("dataset", "?").upper()
    method = cfg.get("method", "?")
    model = cfg.get("model", "pixart")
    model_display = {"pixart": "PixArt-XL-2 512×512", "dit": "DiT-2-256"}.get(model, model)

    lines.append(f"# {model_display} C2I Evaluation: {dataset_name}\n")
    lines.append(f"**Model:** {model_display} | "
                 f"**Method:** {method} | "
                 f"**Steps:** {cfg.get('num_steps')} | "
                 f"**N:** {cfg.get('n_images', '?')}\n")
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

    # Quality metrics
    lines.append("## Quality\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    fid_is = results.get("fid_is", {})
    if "fid" in fid_is:
        lines.append(f"| **FID ↓** | {fid_is['fid']:.4f} |")
    if "is_mean" in fid_is:
        lines.append(f"| **IS ↑** | {fid_is['is_mean']:.4f} ± {fid_is.get('is_std', 0):.4f} |")

    if "clip_score_mean" in agg:
        lines.append(f"| **CLIP Score ↑** | {agg['clip_score_mean']:.2f} ± {agg.get('clip_score_std', 0):.2f} |")
    if "lpips_mean" in agg:
        lines.append(f"| **LPIPS ↓** | {agg['lpips_mean']:.4f} ± {agg.get('lpips_std', 0):.4f} |")
    if "pixel_mse_mean" in agg:
        lines.append(f"| **Pixel MSE ↓** | {agg['pixel_mse_mean']:.4e} ± {agg.get('pixel_mse_std', 0):.4e} |")
    lines.append("")

    # Efficiency metrics
    lines.append("## Efficiency\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    lat = agg.get("latency_vanilla_mean", None) or agg.get("wall_s_mean")
    if lat:
        lines.append(f"| **Latency (s/img) ↓** | {lat:.3f} ± {agg.get('latency_vanilla_std', agg.get('wall_s_std', 0)):.3f} |")

    flops_v = agg.get("flops_vanilla_T")
    flops_a = agg.get("flops_accel_T")
    if flops_v is not None and not np.isnan(flops_v):
        line = f"| **FLOPs (T) ↓** | {flops_v:.3f}"
        if flops_a is not None and not np.isnan(flops_a) and abs(flops_a - flops_v) > 1e-9:
            line += f" (accel: {flops_a:.3f}, ↓{agg.get('flops_reduction', 0):.0%})"
        line += " |"
        lines.append(line)

    speed = agg.get("speed_img_per_s")
    if speed:
        lines.append(f"| **Speed ↑** | {speed:.3f} img/s |")

    sr = agg.get("skip_ratio")
    if sr is not None and sr > 0:
        lines.append(f"| **Skip Ratio** | {sr:.0%} |")
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

def _print_c2i_summary(results: Dict, selected_metrics: List[str], method: str):
    """Print a console summary."""
    agg = results.get("aggregate", {})
    cfg = results.get("config", {})

    print("\n" + "=" * 70)
    print(f"Summary — c2i/{cfg.get('dataset', '?')}/{method}")
    print("=" * 70)

    fid_is = results.get("fid_is", {})
    if "fid" in fid_is:
        print(f"  FID:           {fid_is['fid']:.4f}")
    if "is_mean" in fid_is:
        print(f"  IS:            {fid_is['is_mean']:.4f} ± {fid_is.get('is_std', 0):.4f}")

    if "clip_score_mean" in agg:
        print(f"  CLIP Score:    {agg['clip_score_mean']:.2f}")
    if "lpips_mean" in agg:
        print(f"  LPIPS:         {agg['lpips_mean']:.4f}")
    if "pixel_mse_mean" in agg:
        print(f"  Pixel MSE:     {agg['pixel_mse_mean']:.4e}")

    lat = agg.get("latency_vanilla_mean", None) or agg.get("wall_s_mean")
    if lat:
        print(f"  Latency:       {lat:.3f} s/img")

    flops_v = agg.get("flops_vanilla_T")
    if flops_v is not None and not np.isnan(flops_v):
        fa = agg.get("flops_accel_T")
        if fa is not None and not np.isnan(fa) and abs(fa - flops_v) > 1e-9:
            print(f"  FLOPs (T):     vanilla={flops_v:.3f}  accel={fa:.3f}")
        else:
            print(f"  FLOPs (T):     {flops_v:.3f}")

    speed = agg.get("speed_img_per_s")
    if speed:
        print(f"  Speed:         {speed:.3f} img/s")

    if method == "teacache":
        sr = agg.get("skip_ratio", 0)
        print(f"  Skip ratio:    {sr:.0%}")
    elif method == "speca":
        sr = agg.get("skip_ratio", 0)
        print(f"  Taylor ratio:  {sr:.0%} ({agg.get('taylor_steps', 0)}/{agg.get('taylor_steps', 0) + agg.get('full_steps', 0)})")
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
