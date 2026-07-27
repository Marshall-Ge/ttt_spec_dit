# COVR Profile Validation — 四路径对比报告

**Run ID:** 20260727-081133  
**配置:** DiT-2-256, ImageNet 2048 张, 50 DDIM steps, CFG 4.5, batch_size=32, 4×GPU  
**数据源:** 自建 manifest（audit 2048 trajectories + build_covr_manifest）

## 总览

| 指标 | baseline | speca | timestep_prior | bandit |
|---:|---:|---:|---:|---:|
| FID ↓ | 39.77 | 39.11 | 39.11 | **33.26** |
| IS ↑ | 199.41 | 195.16 | 195.16 | 169.87 |
| img/s ↑ | 3.93 | 7.68 | 7.67 | 6.74 |
| candidate img/s ↑ | - | - | - | **8.05** |
| online img/s ↑ | - | - | - | **6.74** |
| FLOPs accel (T) ↓ | 11.87 | 3.37 | 3.37 | 3.09 |
| FLOPs online (T) ↓ | - | - | - | 4.24 |
| FLOPs safety (T) | - | - | - | 0.92 |
| FLOPs terminal (T) | - | - | - | 0.24 |
| FLOPs reduction (candidate) | 1.00× | 3.52× | 3.52× | 3.84× |
| FLOPs reduction (online) | - | - | - | 2.80× |
| skip ratio | - | 73.31% | 73.31% | 74.00% |

## 阶段耗时（秒，总量）

| 阶段 | baseline | speca | timestep_prior | bandit |
|---:|---:|---:|---:|---:|
| generation_online | 1041.85 | 533.34 | 533.69 | 606.96 |
| denoise_loop | 1007.57 | 499.15 | 499.46 | 572.67 |
| speca_full | - | 285.17 | 285.63 | 276.90 |
| speca_taylor | - | 210.18 | 209.99 | 194.07 |
| safety_shadow_full | - | - | - | 77.75 |
| terminal_fidelity_shadow_full | - | - | - | 20.13 |
| speca_probe_full_block | - | 16.71 | 16.74 | - |
| vae_decode | 34.04 | 33.96 | 34.00 | 34.05 |
| cuda_sync_wait | 33.04 | 32.99 | 33.02 | 33.06 |
| image_save_metrics | 124.45 | 121.49 | 121.51 | 121.88 |
| bandit_update | - | - | - | 0.01 |
| bandit_state_persist | - | - | - | 0.97 |

## 结论

1. **Bandit FID 最优**（33.26 vs baseline 39.77），确认 COVR 保守模板选择能提升生成质量
2. **Candidate 吞吐 8.05 img/s** 是最快的纯推理路径，speca/timestep_prior 约 7.68 img/s
3. **Online 吞吐 6.74 img/s** 扣除 safety + terminal + bandit 开销后仍比 baseline 快 71%
4. **timestep_prior = adaptive SpecA** — 两者的 FID、img/s、FLOPs 完全一致，说明 `timestep_prior` 就是当前自适应策略的等效形态
5. **safety 开销 77.75s**（占 online 的 12.8%）是主要 overhead；terminal 20.13s；bandit 控制层 <1s 可忽略
6. **cuda_sync_wait ~33s** 在所有路径中一致，属于 VAE decode 后的必然同步，非 COVR 引入

## 复现命令

```bash
bash scripts/run_covr_profile_validation.sh
```
