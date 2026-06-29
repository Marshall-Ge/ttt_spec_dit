# -*- coding: utf-8 -*-
"""TTT benchmark runner: integrates the TTT plugin into the standard DiT c2i
ImageNet evaluation pipeline for direct comparison with static TeaCache.

Plugin (φ) persists across ALL images regardless of class — testing whether a
single 0.92M-param network can learn a class-agnostic correction to TeaCache
cached residuals. This is a STRESS TEST: the original TTT design assumes
within-class sequential adaptation; cross-class generalisation is a harder
(and more interesting) question.

Metrics: FID, IS, FLOPs, Latency, Speed — identical set to the static baseline.
"""

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import (
    DIT_REPO, IMAGENET_DIR, OUTPUT_DIR,
    DEFAULT_REL_L1_THRESH, DEFAULT_NUM_STEPS, load_coefficients,
)
from utils import save_image, pil_to_tensor, ensure_real_299

from models.dit import DiTGenerator
from models.ttt_plugin import (
    SessionAdaLNModulator, ttt_state_init, ttt_reset_for_image,
    ttt_session_stats,
)
from accelerators.teacache import teacache_init, teacache_reset, teacache_stats

from eval.fid_is import FIDISComputer
from eval.latency import LatencyMetric, FLOPsMetric


# ===========================================================================
# TTT c2i benchmark runner
# ===========================================================================

def run_ttt_benchmark(args) -> Dict:
    device = "cuda"
    backbone_dtype = torch.float16

    n_images = args.n_prompts
    num_steps = args.num_steps
    guidance_scale = args.guidance_scale

    # Output dir
    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, f"c2i_dit_imagenet_ttt_{num_steps}steps")
    gen_dir = os.path.join(output_dir, "generated")
    gen_299_dir = os.path.join(output_dir, "generated_299")
    os.makedirs(gen_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coefficients = (load_coefficients(args.coef_path) if args.coef_path
                    else _load_dit_coefficients())

    print("=" * 70)
    print("TTT Benchmark — DiT-2-256 C2I ImageNet")
    print(f"  N:          {n_images}")
    print(f"  Steps:      {num_steps}")
    print(f"  Guidance:   {guidance_scale}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Plugin LR:  {args.lr}")
    print(f"  Micro-Epoch:{args.micro_epochs}")
    print(f"  γ (TeaCache):{args.thresh}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # [1] Dataset
    # ------------------------------------------------------------------
    print("\n[1] Loading dataset...")
    from dataset.imagenet import ImageNetDataset
    ds = ImageNetDataset(
        imagenet_dir=getattr(args, "imagenet_dir", IMAGENET_DIR),
        n_images=n_images, seed=args.seed)
    n = len(ds)
    print(f"  [ImageNet] {n} image-prompt pairs loaded")

    # ------------------------------------------------------------------
    # [2] Load DiT + freeze backbone
    # ------------------------------------------------------------------
    print("\n[2] Loading DiT-2-256...")
    generator = DiTGenerator(num_steps=num_steps, device=device,
                             dtype=backbone_dtype)
    generator.load()

    transformer = generator.transformer
    vae = generator.vae
    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    transformer.eval()
    vae.eval()
    n_bb = sum(p.numel() for p in transformer.parameters())
    n_bb_grad = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"  backbone frozen: {n_bb_grad}/{n_bb} params require grad")

    # ------------------------------------------------------------------
    # [3] Build plugin (persistent across ALL images/classes)
    # ------------------------------------------------------------------
    print("\n[3] Building TTT plugin...")
    hidden_dim = (transformer.config.attention_head_dim *
                  transformer.config.num_attention_heads)
    plugin = SessionAdaLNModulator(hidden_dim=hidden_dim, mid_dim=192).to(
        device=device, dtype=torch.float32)
    plugin.train()
    n_plug = plugin.num_parameters()
    print(f"  plugin params: {n_plug:,} ({n_plug/1e6:.3f}M)")

    ttt_state = ttt_state_init(num_steps=num_steps, plugin=plugin,
                               lr=args.lr,
                               micro_epochs=args.micro_epochs)

    # ------------------------------------------------------------------
    # [4] Metrics (FID/IS + FLOPs + Latency)
    # ------------------------------------------------------------------
    print("\n[4] Setting up metrics...")
    metrics = {}
    fid_is = FIDISComputer(gen_dir=gen_299_dir)
    flops_metric = FLOPsMetric(generator)
    flops_metric.profile()  # MUST run before accelerator setup
    latency_metric = LatencyMetric()

    # ------------------------------------------------------------------
    # [5] Generate
    # ------------------------------------------------------------------
    bs = args.batch_size
    print(f"\n[5] Generating {n} images (TTT, {num_steps} steps, "
          f"γ={args.thresh}, batches ≤{bs})...")
    t_start = time.time()

    wall_times = []
    all_results = []
    global_idx = 0

    for batch_start in tqdm(range(0, n, bs), desc=f"c2i/imagenet-ttt", ncols=80):
        batch_end = min(batch_start + bs, n)
        batch_indices = list(range(batch_start, batch_end))
        actual_bs = len(batch_indices)

        # Collect prompts + seeds
        batch_prompts, batch_seeds = [], []
        for idx in batch_indices:
            data = ds[idx]
            # ImageNet: data = (img_path, prompt_text, DiT_class_id)
            class_id = data[2] if len(data) > 2 else data[1]
            batch_prompts.append(class_id)
            batch_seeds.append(100000 + idx)

        # Each batch gets a fresh TeaCache state (same γ for all)
        teacache_state = teacache_init(
            num_steps=num_steps,
            rel_l1_thresh=args.thresh,
            coefficients=coefficients,
        )
        ttt_reset_for_image(ttt_state)  # reset per-image telemetry only

        # Generate one image at a time in the batch (TTT needs per-image
        # denoising loop for the autograd context).  We batch via list
        # comprehension — each image gets its own denoising trajectory.
        t0 = time.time()
        latent_k, image_k = generator.generate_ttt(
            batch_prompts, batch_seeds,
            guidance_scale=guidance_scale,
            teacache_state=teacache_state,
            ttt_state=ttt_state,
        )
        wall_s = time.time() - t0
        wall_times.append(wall_s)
        per_img_s = wall_s / actual_bs

        # Save + feed metrics
        img_limit = getattr(args, "img_save_limit", 50)
        for b, idx in enumerate(batch_indices):
            if global_idx < img_limit:
                cls_name = ds[idx][1].replace("a photo of a ", "").replace(" ", "_")
                out_path = os.path.join(gen_dir, f"{global_idx:06d}_{cls_name}.png")
                save_image(image_k[b:b+1], out_path)
            tag = ds[idx][1].replace("a photo of a ", "").replace(" ", "_")
            fid_is.add(image_k[b], tag=tag)
            global_idx += 1

        # FLOPs: count calc (teacher) steps × (backbone + micro_epoch overhead)
        # and skip (student) steps × (plugin forward only).
        st = teacache_stats(teacache_state)
        n_calc = st.get("total_calc", 0)
        n_skip = st.get("total_skip", 0)

        # TTT-adjusted FLOPs: calc steps include the backbone + µE training
        flops_full = flops_metric._flops_full
        flops_skip = flops_metric._flops_skip
        flops_ttt_calc = flops_full + args.micro_epochs * 19e6  # plugin fwd/bwd ~19M per µE
        flops_metric._total_vanilla += (n_calc + n_skip) * flops_full
        flops_metric._total_accel += n_calc * flops_ttt_calc + n_skip * flops_skip
        flops_metric._n += 1

        latency_metric.add_pairs_batch(
            [per_img_s] * actual_bs, [per_img_s] * actual_bs)

        for idx in batch_indices:
            data = ds[idx]
            all_results.append({
                "idx": idx,
                "prompt": str(data[1])[:120],
                "wall_s": wall_s,
                "images": 1,
            })

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min "
          f"({elapsed/n:.2f} s/image, {len(wall_times)} batches)")

    # ------------------------------------------------------------------
    # [6] FID/IS
    # ------------------------------------------------------------------
    print("\n[6] Computing FID/IS...")
    real_299_dir = ensure_real_299(ds, output_dir, n)
    fid_is.real_dir = real_299_dir
    fid_is_results = fid_is.compute()
    fid_is.cleanup()

    # ------------------------------------------------------------------
    # [7] Aggregate
    # ------------------------------------------------------------------
    agg: Dict = {"n_images": n}
    if wall_times:
        agg["wall_s_mean"] = float(np.mean(wall_times))
        agg["wall_s_std"] = float(np.std(wall_times))
        agg["speed_img_per_s"] = float(n / np.sum(wall_times)) if wall_times else 0.0

    agg.update(fid_is_results)
    agg.update(latency_metric.compute())
    agg.update(flops_metric.compute())

    # TTT stats
    ttt_stats = ttt_session_stats(ttt_state)
    agg["ttt_trained_steps"] = ttt_stats["trained_steps"]
    agg["ttt_session_loss_mean"] = ttt_stats["session_loss_mean"]
    agg["ttt_plugin_params"] = ttt_stats["plugin_params"]

    results = {
        "config": {
            "model": "dit",
            "task": "c2i",
            "dataset": "imagenet",
            "method": "ttt",
            "n_prompts": n,
            "batch_size": args.batch_size,
            "num_steps": num_steps,
            "rel_l1_thresh": args.thresh,
            "coefficients": coefficients,
            "micro_epochs": args.micro_epochs,
            "lr": args.lr,
        },
        "aggregate": agg,
    }

    # ------------------------------------------------------------------
    # [8] Save
    # ------------------------------------------------------------------
    print("\n[7] Saving results...")
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  Results → {results_path}")

    print("\n" + "=" * 50)
    print("TTT BENCHMARK RESULTS")
    print("=" * 50)
    for k, v in sorted(agg.items()):
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.4f}")
        else:
            print(f"  {k:28s}: {v}")
    print("=" * 50)

    return results


# ===========================================================================
# Helpers
# ===========================================================================

def _load_dit_coefficients(coef_path: Optional[str] = None):
    if coef_path:
        with open(coef_path) as f:
            return json.load(f).get("coefficients", load_coefficients())
    dit_coef_path = os.path.join(os.path.dirname(__file__), "dit_coef.json")
    if os.path.exists(dit_coef_path):
        with open(dit_coef_path) as f:
            return json.load(f).get("coefficients", load_coefficients())
    return load_coefficients()


def _clean(obj, _seen=None):
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        return {k: _clean(v, _seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v, _seen) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item()
    return obj


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="TTT Benchmark — DiT-2-256 C2I ImageNet (full eval pipeline)")
    p.add_argument("--num_steps", type=int, default=50,
                   help="Denoising steps (default 50).")
    p.add_argument("--n_prompts", type=int, default=5000,
                   help="Number of images (default 5000).")
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--thresh", type=float, default=0.25,
                   help="TeaCache γ threshold.")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Plugin AdamW learning rate.")
    p.add_argument("--micro_epochs", type=int, default=3,
                   help="Per calc-step micro-epochs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--coef_path", type=str, default=None)
    p.add_argument("--imagenet_dir", type=str, default=IMAGENET_DIR)
    p.add_argument("--img_save_limit", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(DIT_REPO):
        print(f"[ERROR] DIT_REPO not found: {DIT_REPO}")
        sys.exit(1)
    run_ttt_benchmark(args)


if __name__ == "__main__":
    main()
