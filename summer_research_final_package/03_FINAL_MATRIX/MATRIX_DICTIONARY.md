# MATRIX_DICTIONARY.md — FINAL_RESULT_MATRIX.csv 字段字典

## 行（Row）

每一行 = 一个 **候选(Candidate / 管道配置)**。同一候选在不同阶段出现多次的，用 `pipeline_stage` 与 `version` 区分；最终报告中建议**按行合并为最终状态**（历史行放入附录）。

## 列（Column）

| 列名 | 类型 | 说明 |
|---|---|---|
| `candidate` | str | 候选代号，如 `C1`, `C17-1`, `C25`, `C42`, `C49`, `C51`, `Candidate_C`, `gsat_30k_fused_prune10_rclip20` |
| `category` | enum | `FORWARD` / `BACKWARD` / `TRAINING` / `STORAGE` / `CERTIFICATION` / `BASELINE` / `INFRA` |
| `pipeline_stage` | enum | `preprocess` / `sort` / `raster` / `backward` / `optimizer` / `training-loop` / `compression` / `certification` / `baseline` |
| `mechanism` | str | 一句话说明候选做了什么（机制的精确形式）。如 `depth key 32->16bit`, `Segmented radix (Segment-local)`, `selective backward (k% Gaussians)`, `DSIM downsampled scale s`, `attribute-utility gating`, `certificate bound`, `fused prune + rclip=0.5` |
| `correctness` | enum | `PASS` / `CONDITIONAL` / `FAIL` / `NOT_MEASURED` / `FALSIFIED`（描述实验后的正确性状态，不包含"没跑"以外的含义）|
| `quality` | str | 质量判据结果：PSNR/SSIM/LPIPS 在哪个 gate 下的 delta。例：`dPSNR -0.02 (gate -0.10)` |
| `kernel_speedup` | float or "-" | 纯内核时间比（新/基线），有方向必须标注（如 +10.2% 表示比基线慢的**优化后快 10.2%**）。分母一定是内核时间。|
| `backward_speedup` | float or "-" | 反向阶段时间比。 |
| `forward_speedup` | float or "-" | 前向阶段时间比。 |
| `e2e_speedup` | float or "-" | 端到端（wall-clock，含 Python 开销和 IO）时间比。 |
| `scenes` | str | 评测场景与场景数（例如 `room` / `11 scenes x 3 seeds` / `13 scenes`），注明属于哪个场景清单（`benchmark/scene_lists/….*.json`）。 |
| `hardware` | enum | 硬件族代号：`A100-SXM4-80GB`、`A100-PCIE-40GB`、`RTX5070-Laptop`、`RTX4090`（模板）。**不同硬件族绝不合并。** |
| `cohort` | str | 完整 cohort 名（含驱动/CUDA 版本），如 `mx-A100-PCIE-40GB / CUDA 11.8 / PyTorch 2.7.1+cu118`。 |
| `final_verdict` | enum | 最终管理状态：`CONFIRMED`（已通过全部预注册判据）/ `KEEP-AS-COMPONENT` / `KEEP-EST`（估计，未完成验证）/ `MODIFY` / `DROP` / `REPEALED`（曾经发布后来撤销）/ `BLOCKED`（缺条件）/ `FALSIFIED`（假设被证伪）。 |
| `main_evidence_paths` | str | 以逗号分隔的相对路径（在该包内），指向报告/JSON/CSV/审计产物。 |

## 速度口径（Speedup semantics — 必须遵守）

1. `kernel_speedup` 只指 **CUDA 内核时间**（不含 launch 与 Python）。
2. `e2e_speedup` 指 **wall-clock**（含 Python 层、IO、启动）；两者不混用，报告里永远同时给分子和分母。
3. 所有 `>=` / `<=` / `>` 的判据都要引用配置文件中的具体数值（如 `benchmark/benchmark/cli_mode.py` 中 threshold）。
4. "速度提升" 永远要在同一硬件 cohost、同一场景、同一基线版本下表示。

## 阶段 / 版本语义（Ver）

- 每个候选可标记 `v1`, `v2`, `final`。
- `REPEALED` 候选不能被当作有效结果引用（例如 flat-lohmann）。
- 历史时间线上的数据（如 Windows 早期）只能用于说明"为什么被放弃"。

## 时间线列

`PHASE`、`DELIVERABLE`、`SPAN` 列（在 `02_Timeline/PHASE_LOG.md` 中有详细定义）：
- `phase`：Phase 0 … R0.1 … R4
- `deliverable`：该阶段完成的报告/数据/代码
- `span`：该阶段时间窗口（git commit 日期范围）

## 状态机（Status value semantics）

```
DRAFT → RUNNING → DONE → BLOCKED
                    ├──→ CONFIRMED   (所有 gate 通过)
                    ├──→ KEEP-AS-COMPONENT (结果有效但候选不复存在；并入系统配置)
                    ├──→ KEEP-EST     (只做了估算/模拟)
                    ├──→ MODIFY       (需要修改后再测)
                    ├──→ DROP         (正式放弃)
                    └──→ REPEALED     (曾经 DONE/CONFIRMED，后来因新证据被撤销)
```

## 关于 "13 scenes" 的注意

`13 scenes` 特指 **C42 官方场景清单**（`benchmark/c42_scene_lists/*.json`）；**HiGS 论文用的是 11 scenes × 3 seeds**。两者不互换。报告中凡出现场景数，必须同时给出 scene-list 文件名。
