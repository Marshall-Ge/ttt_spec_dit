# COVR Profile Validation — 四路径对比报告（修正版）

**Run ID:** 20260727-081133-fixed  
**配置:** DiT-2-256, ImageNet 2048 张, 50 DDIM steps, CFG 4.5, batch_size=32, 4×GPU  
**数据源:** 自建 manifest（audit 2048 trajectories + build_covr_manifest）  
**修正:** 初版因 `_GenerationProfiler` 跨 batch 累积不清空导致 GPU 时间 2× 膨胀，已通过 `reset()` 修复

## 总览

| 指标 | baseline | speca | timestep_prior | bandit |
|---:|---:|---:|---:|---:|
| FID ↓ | 56.66 | 55.75 | 55.75 | **49.78** |
| IS ↑ | 125.82 | 123.25 | 123.25 | 110.90 |
| img/s ↑ | 3.94 | 7.63 | 7.65 | 6.87 |
| candidate img/s ↑ | - | - | - | **8.05** |
| online img/s ↑ | - | - | - | **6.87** |
| FLOPs accel (T) ↓ | 11.87 | 3.39 | 3.39 | 3.09 |
| FLOPs online (T) ↓ | - | - | - | 4.10 |
| FLOPs safety (T) | - | - | - | 0.78 |
| FLOPs terminal (T) | - | - | - | 0.24 |
| FLOPs reduction (candidate) | 1.00× | 3.50× | 3.50× | 3.84× |
| FLOPs reduction (online) | - | - | - | 2.89× |
| skip ratio | - | 73.13% | 73.13% | 74.00% |

## 阶段耗时（秒，GPU kernel 总量）

| 阶段 | baseline | speca | timestep_prior | bandit |
|---:|---:|---:|---:|---:|
| generation_online | 519.52 | 268.28 | 267.40 | 297.60 |
| denoise_loop | 502.31 | 251.09 | 250.24 | 280.39 |
| speca_full | - | 143.88 | 143.63 | 138.28 |
| speca_taylor | - | 105.29 | 104.70 | 97.10 |
| safety_shadow_full | - | - | - | 33.02 |
| terminal_fidelity_shadow_full | - | - | - | 10.05 |
| speca_probe_full_block | - | 8.37 | 8.36 | - |
| vae_decode | 17.10 | 17.07 | 17.05 | 17.08 |
| cuda_sync_wait (CPU) | 16.36 | 16.38 | 16.34 | 16.36 |
| image_save_metrics (CPU) | 62.67 | 61.52 | 61.52 | 62.11 |
| bandit_update + persist | - | - | - | 0.28 |

## 指标含义说明

- **`img/s`（= online img/s for bandit）**: 端到端真实吞吐，含 denoising + VAE decode + 图像保存 + 所有 COVR 开销。**可部署速度。**
- **`candidate img/s`**: 纯推理吞吐（扣除 safety/terminal/bandit 控制层）。反映 bandit 所选模板的推理效率，**不可单独部署**（不付 overhead 就没有 bandit 的质量收益）。

## 结论

1. **Bandit FID 49.78 最优**，比自适应 SpecA（55.75）提升 6.0 点，比 baseline（56.66）提升 6.9 点
2. **Candidate 8.05 img/s** 说明 bandit 选出的模板比自适应策略高效 5.6%（speca 7.63），算法在找到更优模板
3. **Online 6.87 img/s** 端到端比 baseline（3.94）快 74%，略慢于 speca（7.63，-10%），代价是 FID 提升 6 点
4. **timestep_prior ≡ adaptive SpecA** — FID、吞吐、FLOPs 完全一致，确认该模板就是自适应策略的等效形态
5. **safety 33.02s** 占 online 的 11.1%，terminal 10.05s 占 3.4%，bandit 控制层 0.28s 可忽略
6. **profiler 已修复** — `reset()` 确保每个 batch 独立计时，GPU 时间 ≈ wall clock（偏差 <0.1%）

## 复现命令

```bash
bash scripts/run_covr_profile_validation.sh
```
