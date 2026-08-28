---
name: fid-lever-reconstruction-confirmation
description: FID 层面确认:TeaCache linear 残差漂移 + γ=0.70 在 5k 上 skip 32%→70%、速度 2 倍、FID 34.82→34.40;50k 全量验证进行中
metadata:
  type: project
  modified: 2026-08-27
---

# FID 层面确认:杠杆再造(2026-08-27)

## 背景

[[lever-reconstruction-and-structure-aware-design]] 的 latent-MSE 验证显示 TeaCache linear 残差漂移在同 skip 率下 MSE -53~66%。本实验把该变体实装进 main.py 管线(`--teacache-residual-mode linear`, teacache.py 新增 residual_mode/previous_residual2),用真实 FID 确认。

## 5k 快速确认(DiT-2-256, ImageNet, 50 步, CFG=4.5, seed=42, batch=32, 5000 图, 同分布组内对比)

| 配置 | FID ↓ | IS ↑ | skip% | img/s | 生成时间 |
|------|-------|------|-------|-------|---------|
| teacache plain γ=0.25(现版) | 34.82 | 230.65 | 32% | 5.44 | 19.4 min |
| **teacache linear γ=0.70** | **34.40** | 229.09 | **70%** | **10.92** | 11.7 min |

**结论:线性残差漂移 + γ=0.70 严格支配 plain γ=0.25——skip 翻倍、速度 2 倍、FID 略好(-0.42)、IS 略降(-1.56, 0.7%)。**

- 方向确认:杠杆再造对最终生成质量(FID)也是正的,不仅对 latent MSE
- 5k FID 方差 ~±0.5,50k 需确认 -0.42 的显著性;即使 50k FID 持平,同质量 2 倍速度已是强结果

## 50k 全量验证(进行中, ~7.5h)

顺序(ROI): teacache plain γ=0.25 → teacache linear γ=0.70 → speca uniform → speca sched_b (8,4,1)
- 脚本: `/tmp/run_fid_50k.sh`(远程), 日志 `/tmp/fid_50k.log`
- 50k 历史参考(memory, CFG=4.5): Baseline 24.84 / TeaCache 24.73 (IS 458.4) / SpecA 23.96 / COVR Bandit 17.48 (IS 351.2)
- 输出: `output/fid50k_*`(远程), 拉回 `experiments/fid_50k/`(本地)

## 代码改动(远程, 未提交)

- `accelerators/teacache.py`: teacache_init 加 `residual_mode` ("plain"/"linear"/"damped"); cache_residual 存 previous_residual2; apply_residual 按 mode 加漂移项(damped 用 Hermite, u=min(streak/10,1)); reset 清 previous_residual2
- `main.py`: 加 `--teacache-residual-mode` / `--teacache-residual-decay` / `--speca-bucket-schedule` 参数
- `run_dit.py`: teacache_init 传 residual_mode; speca 分支按时间桶设置 cache_dic.max_taylor_steps(bucket_schedule 属性); run_c2i 解析 schedule

## 相关

[[lever-reconstruction-and-structure-aware-design]], [[error-vs-distance-mechanism]], [[final-framework-conclusion]]
