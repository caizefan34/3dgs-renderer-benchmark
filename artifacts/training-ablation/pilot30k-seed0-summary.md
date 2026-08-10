# HiGS Ablation Pilot 30k — Seed-0 结果摘要 (2026-08-07)

## 实验设计
- 协议: `benchmark/higs-ablation-protocol.json`（独立于冻结 210-job `higs-paper-protocol.json`）
- 矩阵: 8 方法 × 5 场景 (drjohnson, bicycle, garden, room, train) × seed 0 = 40 jobs, A100, 30k from-SfM
- 场景排序 scene-major 交错, 每 job 独立进程, resume/原子结果写入
- 全部结果回传本地并 SHA 校验, `validate_higs_ablation_results.py --require-complete` 通过
  (complete=40, failed=0, missing=0)

## 汇总 (均值, n=5/scene)
| method | PSNR | dPSNR vs full | SSIM | LPIPS | wall s | speedup | mem MiB |
|---|---|---|---|---|---|---|---|
| higs_full | 27.165 | 0 | 0.8682 | 0.0953 | 1145.8 | 1.00x | 4674 |
| higs_visible_only | 27.101 | -0.065 | 0.8672 | 0.1007 | 1074.8 | 1.07x | 4110 |
| higs_switch_12k | 27.071 | -0.094 | 0.8630 | 0.1015 | 940.5 | 1.22x | 3593 |
| higs_switch_18k | 26.901 | -0.264 | 0.8559 | 0.1107 | 872.5 | 1.31x | 3185 |
| higs_switch_21k | 26.957 | -0.208 | 0.8565 | 0.1095 | 872.1 | 1.31x | 3188 |
| higs_current | 26.725 | -0.440 | 0.8523 | 0.1150 | 855.6 | 1.34x | 3176 |
| higs_progressive_only | 26.817 | -0.348 | 0.8564 | 0.1053 | 873.5 | 1.31x | 3599 |
| higs_switch_15k | 26.754 | -0.412 | 0.8564 | 0.1103 | 884.3 | 1.30x | 3168 |

## Pareto 候选 (质量损失 ≤ 0.30 dB 且加速 ≥ 1.15x)
1. **higs_switch_12k**: dPSNR -0.094 dB, 1.22x wall, -1081 MiB 显存
2. **higs_switch_21k**: dPSNR -0.208 dB, 1.31x wall, -1486 MiB 显存
3. **higs_switch_18k**: dPSNR -0.264 dB, 1.31x wall, -1489 MiB 显存

## 初步解读 (pilot 阶段, 不做最终结论)
- switch_12k 用最小的质量代价 (≈0.1 dB) 换 22% 加速, 是"质量最接近 full"的操作点。
- 更晚的切换点 (18k/21k) 加速相似 (1.31x) 但质量略低, 说明可见性掩码在早期密化阶段作用显著。
- visible_only (无 progressive) 质量最接近 full 但加速有限 (1.07x), 支持"progressive 是主要加速来源"的假设。
- 30k 步 seed-0 pilot 仅用于筛选; 正式结论需多 seed 确认矩阵 (3 seeds × 11 scenes)。
