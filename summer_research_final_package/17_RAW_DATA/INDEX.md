# INDEX.md — 原始数据索引（17_RAW_DATA）

> 本文件把 `results/`、`reports/` 中的原始数据按主题建立索引，使最终报告的作者能
> “按图索骥”：每个数字都能找到原始 JSON/CSV。**不要在本文件中虚构路径**；
> 所有条目都基于仓库当前实际存在的文件。

---

## 快速导航

| 主题 | 数据位置 | 说明 |
|---|---|---|
| 基线 & 训练数据 | `results/training/`、`results/a100/` | 训练回合的 metrics/checkpoints |
| C42 实验 | `results/c42_adaptive/`、`results/a100/phase-c42/` | 自适应缩放扫描 |
| C25/C26 反向 | `results/a100/phase-c25/`、`phase-c26/` | 稀疏反向 profile、1.7ms 地板 |
| C1/C17 正向 | `results/a100/phase-c1/`、`phase-c17*/` | 深度压缩 / 瓦片排序 |
| HiGS | `results/higs-round*/`、`results/a100/higs*` | T_HIGS 训练/正确性 |
| 复现/CI | `results/confirmatory-*` | 确认性矩阵（3 seeds / 132 jobs） |
| 压缩实验 | `results/measured-compression/` | 量化/压缩数据 |
| 诊断 | `results/diagnostics/` | 群组 profile、误差分析 |

---

## 主要目录细读

### 1. `results/confirmatory-*`（确认性结果，报告必须引用）
- `confirmatory-consumer-720p/` — 消费级 720p 推理基准（62 文件）。
- `confirmatory-matrix/` — 主结果矩阵（63 文件），含 speedup CI 表。
- `confirmatory-db/` — Deep Blending 子集（27 文件）。
- 每个目录中的 `*_summary.json` 或 `*_table.csv` 可直接引用为“最终结果表”。

### 2. `results/higs-round*`（HiGS 各轮次）
- 每个 round 一个子目录；round>=50 的文件为有效训练结果；`higs-round61..64` 为最终一致版本。
- `results/higs-round61/summary.json` 等包含 13 场景最终指标。

### 3. `results/a100/`（A100 实验汇总）
- 子目录按 `phase-*` / `c*` 组织；文件名为 `*.json` / `*.csv` / `*.log`。
- `phase-c25/`、`phase-c26/` 为反向稀疏关键实验；`phase-c42/` 为 C42 跨 baseline 对照。

### 4. `results/measured/`（主测量数据）
- 1040 文件，包含原始 timing 采样、GPU 指标（SM/带宽/占用率）。
- 子目录 `measured/train`、`measured/infer` 等；另见 `results/measured/timing-samples.json`。

### 5. `reports/`（叙述性报告与审计）
- `reports/reference-v1-baseline-lock.md` — 基线锁定声明（**任何对比的锚点**）。
- `reports/phase-c42/`、`reports/r3_1/`、`reports/r4/` 等为审计与结论性报告。
- `reports/final-conclusions.md` 为旧版总结；最终报告应以本包 04/08/13 目录为最新依据。

---

## 使用规则

1. **只引用存在的文件**：先 `Test-Path` 或用 `dir` 确认，不要照抄本索引而不验证。
2. **数字必须带路径**：例如
   > “13 场景均值 PSNR 提升 0.02 dB（`results/higs-round61/summary.json`，字段 `mean_psnr_delta`）”
3. **两套环境数据不混用**：`results/a100/…`（EPIC-05）与 `results/rtx5070…`（Windows）
   分属不同 cohort；同一指标不得跨 cohort 比较。
4. **优先最新轮次**：对同一实验存在多个轮次文件时，用时间戳最新的；若无法确认，按
   `13_TRAINABLE_HIGS/TRAINABLE_HIGS_INDEX.md` 或 `04_CRITICAL_FINDINGS` 中标注的版本为准。

---

*本文件由 evidence-collection agent 于 2026-09-18 生成。*
