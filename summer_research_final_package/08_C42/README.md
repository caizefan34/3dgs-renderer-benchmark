# 08 · C42（候选 C42 全证据）

## 核心结论

**C42 并非对所有 baseline 都是 universal additive optimization。**
它对不同 baseline（Speedy / FastGS / Faster-GS）的效果不一致，叠加性被否定。
C42 的正确性（与完整、正确实现相比）取决于其定义与 baseline 的差异；若将 naive
版本的 C42 视为"免费的加法优化"，则存在误差，必须在报告中以正确的表述保留。

## 关键数据文件（原始位置）

| 内容 | 仓库路径 |
|---|---|
| 13 场景原始测量 | `results/c42_scene_raw/*.json`（或 `results/c42_13scene/` 变体） |
| 汇总表格 | `results/c42_13scene/aggregate.*.{csv,json}` |
| 自适应尺度扫描 | `results/c42_adaptive/`（b9/b12/b20 等） |
| 与 Speedy 对照 | `results/a100/phase-c42/speedysplat_c42_*.json` |
| 与 FastGS 对照 | `results/a100/phase-c42/fastgs_c42_*.json` |
| 与 Faster-GS 对照 | `results/a100/phase-c42/faster_gs_c42_*.json` |
| 最终决策 | `reports/a100_validation/final_c42_decision.md` |

## 报告写作提示

- 引用"13 场景 raw"前先查该场景是否在最终列表中（有些场景因数据质量被剔除）。
- C42 的 baseline 定义必须写清楚：与 Reference V1（`absgrad, 8e-4`）对照时才是
  `+2 场景` 之外的那套结论。
