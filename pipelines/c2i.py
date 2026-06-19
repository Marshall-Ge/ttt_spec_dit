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
from models.teacache import TeaCacheAccelerator

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
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"c2i_{dataset_name}_{args.method}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = load_coefficients(args.coef_path) if args.coef_path else load_coefficients()

    print("=" * 70)
    print(f"PixArt-α C2I Evaluation — {dataset_name.upper()}")
    print(f"  Method:   {args.method}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  N:        {args.n_prompts}")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Metrics:  {selected}")
    print(f"  Output:   {output_dir}")
    if args.method == "teacache":
        print(f"  γ:        {args.thresh}")
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
    print("\n[2] Loading PixArt-α model...")
    generator = PixArtGenerator(
        num_steps=args.num_steps, device=device, dtype=dtype)
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
    else:
        print(f"  Baseline (no TeaCache)")

    # =====================================================================
    # 5. Generate images
    # =====================================================================
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    print(f"\n[4] Generating {n} images ({args.method})...")
    t_start = time.time()

    wall_times = []
    all_results = []

    for idx in tqdm(range(n), desc=f"c2i/{dataset_name}", ncols=80):
        img_path, prompt = ds[idx]
        seed = 100000 + idx

        # --- Generate ---
        t0 = time.time()
        if args.method == "teacache":
            accelerator.reset()
            latent, img = generator.generate_teacache(
                prompt, seed, accelerator.teacache)
        else:
            latent, img = generator.generate(prompt, seed)
        wall_s = time.time() - t0
        wall_times.append(wall_s)

        # --- Save generated image ---
        out_path = os.path.join(gen_dir, f"{idx:06d}.png")
        save_image(img, out_path)

        # --- Per-image metrics ---
        if need_clip:
            metrics["clip"].add(img, prompt=prompt)

        if need_lpips or need_mse:
            # Load real image as tensor (at generated resolution)
            try:
                real_pil = Image.open(img_path).convert("RGB")
                gen_size = (img.shape[-1], img.shape[-2])  # (W, H)
                real_pil = real_pil.resize(gen_size, Image.BICUBIC)
                real_tensor = pil_to_tensor(real_pil).to(device)
            except Exception:
                real_tensor = None

            if need_lpips and real_tensor is not None:
                metrics["lpips"].add(img, reference=real_tensor.unsqueeze(0))
            if need_mse and real_tensor is not None:
                metrics["mse"].add(img, reference=real_tensor.unsqueeze(0))

        if need_latency:
            metrics["latency"].add_pair(wall_s, wall_s)

        if need_flops:
            if args.method == "teacache":
                metrics["flops"].add_generation(accelerator.teacache)
            else:
                metrics["flops"].add_vanilla_steps(args.num_steps)

        all_results.append({
            "idx": idx,
            "prompt": prompt[:120],
            "wall_s": wall_s,
        })

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min ({elapsed/n:.2f} s/image)")

    # =====================================================================
    # 6. FID/IS computation
    # =====================================================================
    fid_is_results = {}
    if need_fid_is:
        real_299_dir = _prepare_real_dir(ds, output_dir, n)

        # Resize generated images to 299×299
        gen_299_dir = os.path.join(output_dir, "generated_299")
        os.makedirs(gen_299_dir, exist_ok=True)
        print(f"\n  Resizing {n} generated images to 299×299...")
        for idx in tqdm(range(n), desc="resize gen", ncols=80):
            out_path = os.path.join(gen_299_dir, f"{idx:06d}.png")
            if os.path.exists(out_path):
                continue
            src_path = os.path.join(gen_dir, f"{idx:06d}.png")
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
    agg: Dict = {"n_images": n}

    # Wall-time stats
    if wall_times:
        agg["wall_s_mean"] = float(np.mean(wall_times))
        agg["wall_s_std"] = float(np.std(wall_times))
        agg["speed_img_per_s"] = float(1.0 / np.mean(wall_times))

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

    # Skip ratio (TeaCache only)
    if args.method == "teacache" and accelerator is not None:
        st = accelerator.stats
        agg["skip_ratio"] = st.get("skip_ratio", 0.0)
        agg["total_calc"] = st.get("total_calc", 0)
        agg["total_skip"] = st.get("total_skip", 0)

    results = {
        "config": {
            "task": "c2i",
            "dataset": dataset_name,
            "method": args.method,
            "n_images": n,
            "num_steps": args.num_steps,
            "rel_l1_thresh": args.thresh if args.method == "teacache" else None,
            "coefficients": coefficients if args.method == "teacache" else None,
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
        img_path, _ = dataset[idx]
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

    lines.append(f"# PixArt-α C2I Evaluation: {dataset_name}\n")
    lines.append(f"**Model:** PixArt-XL-2 512×512 | "
                 f"**Method:** {method} | "
                 f"**Steps:** {cfg.get('num_steps')} | "
                 f"**N:** {cfg.get('n_images', '?')}\n")
    if method == "teacache":
        lines.append(f"**γ:** {cfg.get('rel_l1_thresh')} | "
                     f"**Coefficients:** `{cfg.get('coefficients')}`\n")
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
        lines.append("TeaCache (Liu et al., CVPR 2025) accelerated PixArt-α generation.\n")
    else:
        lines.append("PixArt-α DPMSolver++ baseline (all steps computed).\n")

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
