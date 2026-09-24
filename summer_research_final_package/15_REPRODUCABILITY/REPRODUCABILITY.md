# REPRODUCABILITY.md — 环境、命令、种子与校验

> 本文件给最终的结题报告提供“复现实验”所需的最小信息。所有命令均在
> 仓库根目录 `C:\Users\36570\3dgs-renderer-benchmark` 下执行（除非另注明）。
> 置顶规则：**不要在本文件之外编造版本号或命令；引用时以仓库内 `configs/`、
> `requirements*.txt`、`pyproject.toml` 与 CI 配置文件为准。**

---

## 1. 硬件与软件版本矩阵（cohort 模型）

本项目使用 **cohort 模型**：不同机器/GPU 的绝对时间**禁止互相比较**。

| Cohort | 机器/GPU | 软件栈 | 用途 |
|---|---|---|---|
| **EPIC-05 (Linux)** | NVIDIA A100-SXM4-80GB | CUDA 12.8；PyTorch 2.9.1+cu128（推测来自环境日志，见 `reports/epic5/` 环境检查），Python 3.10+ | 主实验/训练 |
| **EPIC-05 (CPUs)** | A100 集群 CPU 节点 | 无 GPU | 数据准备/解析/校验 |
| **RTX 5070 Laptop (Windows)** | NVIDIA RTX 5070 Laptop 8GB | CUDA 13.0；PyTorch 2.x | 单机小规模验证 |
| **MX 集群（历史）** | 旧机器 | 未确认 | 早期实验（结果仅作背景参考） |

> 精确版本请以仓库内实际日志/锁文件为准，如 `reports/epic5/EPIC5_environment_check.md`、
> `requirements*.txt`、CI 工作流文件中 `torch`/`cuda` 固定版本。

---

## 2. 标准实验命令

### 2.1 训练（TRN）
```bash
python scripts/train.py --config configs/reference_v1/room_30k.yaml --seed 42
```
参数必须与 `scripts/train.py` 当前接口校对；seed 选择：训练用 42；复现性测试额外使用
seed {42, 2026, 0}（见 CI 的 `--seed` 参数）。

### 2.2 评测（EVL）
```bash
python scripts/evaluate.py --config configs/reference_v1/room_30k.yaml --checkpoint <ckpt_path> --out results/<run_id>/
```

### 2.3 单元测试（UT）
```bash
python -m pytest tests/ -k "candidate"       # 候选算子正确性/数值门
python -m pytest tests/ -k "seed"             # 种子复现性测试
```
（具体测试名以 `tests/` 目录为准）

---

## 3. 数据集与校验

| 数据集 | 内容 | 校验方式 |
|---|---|---|
| Mip-NeRF 360 | bicycle / garden / bonsai / room / counter / kitchen | `data/` 下脚本做 SHA-256 校验 |
| Tanks & Temples | train / truck 等 4 场景 | 同上 |
| Deep Blending | playroom / drjohnson | 同上 |

> 数据集 SHA-256 清单见 `configs/datasets.json` 或仓库根 `DATA_CHECKSUMS.txt`（若存在）。
> 若报告中引用“数据质量检查”，只引用该文件中的 hash 值。

---

## 4. 随机性控制（seed 策略）

- 每个 `(候选, 场景, seed)` 组合独立运行；主实验结果取 3 seeds 的
  **均值 ± 标准差**（或 CI）。
- 任何“速度提升”结论必须包含：
  - 加速分母：`kernel` / `forward` / `e2e` / `wall`（必须写明哪一种）
  - 样本数：`N=11 scenes × 3 seeds`（或具体数值）
  - 该结论的 p 值/CI 或一致性描述

---

## 5. 目录与日志位置

| 内容 | 位置 |
|---|---|
| 实验配置 | `configs/reference_v1/room_30k.yaml`、`configs/epic05/*.json` |
| 训练输出/JSON | `results/training/`、`results/a100/phase-c5*/` 等 |
| 日志 | `logs/`（每实验一个子目录） |
| CI 配置 | `.github/workflows/*.yml` |
| 测试 | `tests/` |

---

## 6. 已知环境问题（引用时注意）

1. **RTX 5070 历史数据**（`reports/rtx5070/`）与 EPIC-05 数据**不能混用**。
2. **cross-compile 问题**：CUDA 11.8（MX）与 CUDA 12.8（EPIC-05）编译产物不互通；
   报告不得声称“同一二进制跨机器运行”。
3. **时间戳问题**：早期 `@{...}` PowerShell 日志中的时间戳为 UTC，与文件名中本地
   时间存在时差，引用时以文件 mtime 为准。
4. **DNN 数值误差**：不同 GPU 上 float32 舍入差异可导致质量指标小幅波动；
   报告结论差异必须大于该误差带（见 `reports/` 中误差分析）。

---

## 7. 提交锚点

- 主报告引用基线：`b562562`（C17 tile-local prototype）
- 机制修正：`fc756f4`（C1 机制）、`8c2d7b0`（R3.1 修复）
- R4：`9e40f11`（13 场景训练启动）

> 这些锚点是仓库历史中的 commit hash；报告中写“commit `b562562` 起引入 C17 原型”即可。

---

*本文件由 evidence-collection agent 于 2026-09-18 生成。*
