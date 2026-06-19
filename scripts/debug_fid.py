"""Debug: check Inception features for FID computation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from phase2.eval_coco import InceptionExtractor, load_real_image, compute_fid_from_features, compute_is_from_logits

device = "cuda"
coco_dir = "/root/autodl-fs/data/coco"
output_dir = "./output/phase2/coco30k"

# Load 10 real images and 10 generated images
extractor = InceptionExtractor(device=device)

# --- Real images ---
print("=" * 60)
print("Real COCO images")
print("=" * 60)
real_feats = []
from phase2.eval_coco import COCO30KDataset
ds = COCO30KDataset(coco_dir, n_images=10, seed=42)
for idx, (img_path, caption) in enumerate(ds):
    img = load_real_image(img_path, size=299).unsqueeze(0)
    print(f"  [{idx}] {os.path.basename(img_path)}: tensor {img.shape}, "
          f"range=[{img.min():.4f},{img.max():.4f}], dtype={img.dtype}")
    feat = extractor.extract_features(img)
    real_feats.append(feat)
    print(f"       feat shape={feat.shape}, mean={feat.mean():.6f}, std={feat.std():.6f}, "
          f"min={feat.min():.6f}, max={feat.max():.6f}")

real_feat = np.concatenate(real_feats, axis=0)
print(f"\n  All real features: shape={real_feat.shape}, mean={real_feat.mean():.6f}, "
      f"std={real_feat.std():.6f}")

# Are features identical?
same_count = 0
for i in range(len(real_feats)):
    for j in range(i+1, len(real_feats)):
        if np.allclose(real_feats[i], real_feats[j], atol=1e-6):
            same_count += 1
print(f"  Identical feature pairs: {same_count}/{len(real_feats)*(len(real_feats)-1)//2}")

# Covariance rank
cov = np.cov(real_feat, rowvar=False)
eigvals = np.linalg.eigvalsh(cov)
print(f"  Cov rank (non-zero eigvals): {(eigvals > 1e-8).sum()} / {len(eigvals)}")
print(f"  Cov eigval range: [{eigvals.min():.6f}, {eigvals.max():.6f}]")

# --- Generated (vanilla) images ---
print("\n" + "=" * 60)
print("Generated (vanilla) images")
print("=" * 60)
gen_feats = []
for idx in range(10):
    path = os.path.join(output_dir, "generated_vanilla", f"{idx:05d}.png")
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((299, 299), Image.BICUBIC)
    arr = np.array(pil, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    print(f"  [{idx}] tensor {img.shape}, range=[{img.min():.4f},{img.max():.4f}]")
    feat = extractor.extract_features(img)
    gen_feats.append(feat)
    print(f"       feat shape={feat.shape}, mean={feat.mean():.6f}, std={feat.std():.6f}, "
          f"min={feat.min():.6f}, max={feat.max():.6f}")

gen_feat = np.concatenate(gen_feats, axis=0)
print(f"\n  All gen features: shape={gen_feat.shape}, mean={gen_feat.mean():.6f}, "
      f"std={gen_feat.std():.6f}")

cov_g = np.cov(gen_feat, rowvar=False)
eigvals_g = np.linalg.eigvalsh(cov_g)
print(f"  Cov rank: {(eigvals_g > 1e-8).sum()} / {len(eigvals_g)}")

# --- FID test ---
print("\n" + "=" * 60)
print("FID Computation")
print("=" * 60)
print(f"  mu_r: mean={real_feat.mean(0).mean():.6f}, std={real_feat.mean(0).std():.6f}")
print(f"  mu_g: mean={gen_feat.mean(0).mean():.6f}, std={gen_feat.mean(0).std():.6f}")
mu_diff = np.linalg.norm(real_feat.mean(0) - gen_feat.mean(0))
print(f"  ||mu_r - mu_g|| = {mu_diff:.6f}")

fid = compute_fid_from_features(real_feat, gen_feat)
print(f"  FID = {fid:.4f}")

# Also test: FID of real vs real should be ~0
print(f"\n  Sanity: FID(real[:5], real[5:]) = "
      f"{compute_fid_from_features(real_feat[:5], real_feat[5:]):.4f}")

# --- IS test ---
print("\n" + "=" * 60)
print("IS Computation")
print("=" * 60)
logits_list = []
for idx in range(10):
    path = os.path.join(output_dir, "generated_vanilla", f"{idx:05d}.png")
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((299, 299), Image.BICUBIC)
    arr = np.array(pil, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    logits = extractor.extract_logits(img)
    logits_list.append(logits)
    probs = torch.softmax(torch.from_numpy(logits), dim=1)
    top5 = probs.topk(5)
    print(f"  [{idx}] logits shape={logits.shape}, range=[{logits.min():.3f},{logits.max():.3f}]")
    print(f"       top5 classes: {top5.indices[0].tolist()}, probs: {top5.values[0].tolist()}")

logits_all = np.concatenate(logits_list, axis=0)
is_m, is_s = compute_is_from_logits(logits_all, splits=5)
print(f"\n  IS(10 images) = {is_m:.4f} ± {is_s:.4f}")
