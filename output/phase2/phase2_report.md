# Phase 2: SOTA Baseline Strengthening — Full Report

**Model:** PixArt-Alpha XL-2 512x512 | **Steps:** 50 | **Seed:** 42 | **γ:** 0.1 | **Max Skip:** 3 | **Early Block:** 2 | **Probe Block:** 14

---

## 🌊 "Deep Water" Bottleneck Report

> **What this shows:** Even with a SOTA-inspired adaptive cache
> (TeaCache-style relative L1 residual thresholding), certain
> denoising stages force frequent full re-computation — the
> "Deep Water" where features change too fast for caching to help.

- **Cache config:** γ = 0.1, max_skip = 3
- **Total full forwards:** 13
- **Total skips:** 37
- **Skip ratio:** 74.00%
- **Total rejections:** 12

### Rejection Frequency by Denoising Stage

| Stage | Steps | Rejections | Rejection Rate | Mean Residual @ Reject | Max Residual @ Reject | Max-Skip Rejects | Threshold Rejects |
|-------|-------|------------|----------------|------------------------|------------------------|------------------|-------------------|
| Early (Noise) | 16 | 3 | 18.8% | 0.05129 | 0.053315 | 3 | 0 |
| Middle (Structure) | 16 | 4 | 25.0% | 0.053825 | 0.05508 | 4 | 0 |
| Late (Details) | 18 | 5 | 27.8% | 0.08421 | 0.123703 | 4 | 1 |

### 🔴 Primary Bottleneck: **Late (Details)**

The **Late (Details)** stage exhibits a rejection rate of **27.8%**, meaning that even with adaptive thresholding, 28% of steps in this stage cannot be cached and require full re-computation.

Most rejections (4/5) are **max-skip** rejections — the cache drifts too far after 3 consecutive skips and must re-calibrate.

**Implication for Phase 3 TTT:** This stage is where a learned, online 
predictor (Test-Time Training) can potentially outperform the static 
residual-threshold cache by learning to predict feature evolution 
adaptively, reducing the rejection rate.

---

## 📊 SOTA Baseline Performance Table

| Metric | Vanilla DPMSolver++ | Our SOTA Caching Baseline |
|--------|---------------------|---------------------------|
| **Latency (s)** | 1.82 ± 0.00 | 0.57 ± 0.00 |
| **Speedup Ratio** | 1.00× | **3.19×** |
| **CLIP Score** | nan ± 0.00 | nan ± 0.00 |
| **CLIP Δ** | — | nan |
| **Pixel MSE** | — | 0.003036 |
| **Latent MSE** | — | 0.044116 |

*Evaluated on 1 prompts.*

## 🔬 Per-Step Verification MSE

| Stage | Avg Probe MSE | Avg Tail MSE | Skipped Steps |
|-------|---------------|-------------|---------------|
| Early (Noise) | 1.229405e-02 | 1.071944e-03 | 12 |
| Middle (Structure) | 6.665752e-03 | 1.689781e-03 | 12 |
| Late (Details) | 7.358743e-03 | 8.169115e-03 | 13 |
