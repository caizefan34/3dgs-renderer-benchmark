# 12 · C49 属性解耦（Attribute-Decoupled Backward）

## 定义（重要区分）

- **G_dens**（densification gradient）＝ `means2d.absgrad`（用于增长/剪枝 的梯度）
- **G_opt**（optimization gradient）＝ 每个参数 `.grad`（用于优化器更新的梯度）

旧项目错误地把 G_dens 当作 G_opt 使用（R0.1 修正前的 C49），造成**假阳性**。
R0.1 将两者分开测量后：

| 指标 | 修正前 (C49 原始) | 修正后 (最佳代理) |
|---|---|---|
| 浓度 (top-Gaussians) | 82.0% (uncorrected) | 75.8% (signed, corrected) |
| 有效上限 (E2E) | — | ≈ 1.13%（绝不可能达到 15%+） |
| SH 属性 | top-10% → ~95% 正收益 | 20% → ~95% |

## 结论

C49 的"收益"在修正后**只能支撑约 1.1–1.3%**，不足以作为独立论文级性能主张；
它被吸收进 C51 选择性反向的 Utility 公式（`U_loss = −g^T Δθ`）中。

## 数据位置

| 数据 | 路径 |
|---|---|
| 报告 | `reports/phase_c49_gaussian_lifecycle_research.md` |
| 修正证明 | `reports/phase-r0.1-evidence-correction.md` |
| 依赖图 | `reports/r2-gradient-dependency-graph.md` |
| 数据 | `results/phase-c49/*.json`（摘要文件见 _drafts/s7_c49.md） |
