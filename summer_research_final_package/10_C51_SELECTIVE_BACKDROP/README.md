# 11 · C51 选择性反向（Selective Backward）

## 这是什么

C51 系列 = 在训练中仅对"高预期收益"（高 utility/高梯度预测）的 Gaussian 子集执行
反向传播。包含多个变体（V1、V2、B1、B2、B3），以及门控（Gates A/B/C/D）系统。

## 核心数字

| 指标 | 数值 | 出处 |
|---|---|---|
| K50-B1 kernel 加速 | ≈ +10.2% | `results/a100/phase-c51-stage4a/kernel_microbenchmark.json` |
| 5K E2E | ≈ +6.7% | `results/a100/phase-c51st4a/…`（见摘要） |
| 30K E2E | ≈ +7.8% | 同上 |
| 前向 bit-exact | YES（K50-B1） | `reports/phase_c51_sparse_backward.md` |
| 余弦相似度 | ≈ 1.0（K50-B1） | 同上 |
| B2 失败 | −0.49 dB / −0.0054 SSIM | 部分更新不一致 |

## 为什么 B2 失败

B2 只更新了 mask 中给定的参数（如位置）而不同时更新外观 → 每次迭代不一致，
训练质量下降。B1/B3 更新所有被掩盖参数的对应部分，保持一致性。

## 最终状态

`KEEP`，作为 `C51_SYSTEMS_COMPONENT` 进入最终系统建议（不单独作为独立候选发表）。

## 数据位置

| 数据 | 路径 |
|---|---|
| 训练结果 | `results/a100/phase-c51/`、`phase-c51-stage4a/`、`phase-c51-stage4b/`、`phase-c51-stage5/` |
| 摘要 | `reports/phase_c51_sparse_backward.md`、`reports/phase-c51-stage4a.md` 等 |
