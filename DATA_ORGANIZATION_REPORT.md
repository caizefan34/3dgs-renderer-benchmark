# 3DGS Renderer Benchmark — 数据完整性组织报告

> **生成日期**: 2026-09-10  
> **目的**: 按硬件来源（A100 vs RTX 5070 Laptop GPU）重新组织数据，确保论文数据准确性

---

## 1. 硬件分隔原则

| 硬件 | 架构 | 内存 | 适用阶段 | 最终论文可用？|
|------|------|------|---------|:----------:|
| **NVIDIA A100-PCIE-40GB** | SM80 | 40GB HBM2 | 所有基线profiling、候选筛选、训练动态、验证 | **✅ 是** |
| **NVIDIA RTX 5070 Laptop GPU** | SM120 | 8GB GDDR7 | 探索性/假设生成 | **❌ 否（仅探索用）** |

**关键架构差异**: SM80 (A100) vs SM120 (RTX 5070) 的 CUDA core 布局、warp 调度器、L1/Shared memory 分区完全不同。CUB sort 策略、cuDNN 行为、occupancy 均不可跨硬件迁移。

---

## 2. 已完成的重新组织

### 2.1 `results/` 目录

**所有 A100 数据 → `results/a100/`**

```
results/a100/
├── phase-a100/           # 基线profiling、H6审计、evidence recovery（20个文件）
├── c17-c33/
│   ├── phase-c17-c2/     # C1排序实现验证（4个文件）
│   ├── phase-c18/        # 光栅化分解gate（6个文件）
│   ├── phase-c19/        # 光栅化profiling（9个文件 + dirty_run/）
│   ├── phase-c20/        # 候选锦标赛（6个文件，不含c7_rtx5070.json）
│   ├── phase-c21/        # 并行候选发现（9个文件）
│   ├── phase-c22/        # 缓存测试（5个文件）
│   ├── phase-c23/        # 并行筛选（1个文件）
│   ├── phase-c24/        # 训练候选筛选（9个文件）
│   ├── phase-c25/        # 训练候选筛选（9个文件）
│   ├── phase-c26/        # 候选锦标赛（9个文件）
│   ├── phase-c27/        # 并行机制验证（9个文件）
│   ├── phase-c28/        # 并行研究（12个文件）
│   ├── phase-c29/        # 训练级锦标赛（13个文件）
│   ├── phase-c30/        # 训练动态锦标赛（9个文件）
│   └── phase-c31/        # c31-c33: A100实验数据（24个文件）
├── phase-c42/             # A100验证数据
│   ├── c40_baseline_a100.json          # ✅ A100 C40重验证
│   └── (future A100 C42 validation)   # 
└── hardware_metadata.json # A100硬件指纹
```

**所有 RTX 5070 数据 → `results/rtx5070/`**

```
results/rtx5070/
├── phase-c20/
│   └── c7_rtx5070.json          # C7跨硬件测试（A100上跑不了）
├── phase-c31/                     # 被误放在A100目录的C35-C42数据
│   ├── c38_validity_data.json     # C38有效性研究
│   ├── c39_locality_data.json     # C39局部性研究
│   ├── c40_trace.json
│   ├── c40_training_breakdown.json # ⚠️ 无硬件标签，但实际跑在RTX 5070
│   ├── c41_gpu_utilization.json   # ⚠️ 声称"A100-PCIE-40GB"但数据来自RTX 5070
│   ├── c41_trace.json
│   ├── c42_dssim_full_trace.json  # 明确写"RTX 5070 Laptop GPU"
│   └── c42_dssim_ops_trace.json
└── phase-c42/                     # 所有C42优化数据（Windows上运行）
    ├── c42_compile_data.json
    ├── c42_compile_trace.json
    ├── c42_downsampled_ssim_data.json
    ├── c42_e1_gradient_profile.json
    ├── c42_pixel_contrib_data.json
    ├── c42_separable_conv_data.json
    ├── c42_temporal_contrib_data.json
    ├── c42_b_trace_tmp.json
    └── c42_compile_trace_tmp.json
```

### 2.2 `reports/` 目录

| 原始位置 | 新位置 | 说明 |
|---------|-------|------|
| `reports/phase-c31/c35~c42 *.md` | `reports/rtx5070/` | 引用RTX 5070数据 |
| `reports/phase-c42/*.md` | `reports/rtx5070/phase-c42/` | C42优化报告 |
| `reports/a100_validation/` | 不变 | A100验证报告（含c40、c42） |

### 2.3 `scripts/` 目录

| 原始位置 | 新位置 | 说明 |
|---------|-------|------|
| `scripts/phase-c31/c35~c42 *.py` | `scripts/rtx5070/` | RTX 5070脚本 |
| `scripts/phase-c42/*.py` | `scripts/rtx5070/` | C42脚本（含A100验证脚本） |

---

## 3. ⚠️ 关键问题记录

### 🔴 严重：C40/C41 数据被错误标注为 A100

**C40 `results/rtx5070/phase-c31/c40_training_breakdown.json`**
- 文件中 **没有** GPU名称/硬件标识
- 原始 RTX 5070 数据显示：fwd=6.86ms, loss=20.43ms, bwd=40.90ms, total=85.27ms
- 真实 A100 验证数据显示：fwd=3.38ms, loss=75.17ms, bwd=11.70ms, total=94.70ms
- **结论**: RTX 5070 数据不能用于论文 （总时间比 A100 还快？明显不一致）

**C41 `results/rtx5070/phase-c31/c41_gpu_utilization.json`**
- 文件中 `gpu_specs` 声称 `"name": "A100-PCIE-40GB"`，但实际数据来自 RTX 5070
- C41 报告 `reports/rtx5070/c41_gpu_utilization.md` 引用 "A100-PCIE-40GB reference" 但数据并非从 A100 测得
- **对论文的影响**: C41 中 "D-SSIM 占 42.5% GPU 时间" 这个结论是 RTX 5070 数据。A100 上的实际比例是 **78.6%**（已验证）

**C38/C39**:
- 文件中无显式硬件标签
- 时间戳显示 09-10 16:16/16:50（本地 Windows 时间），确认是 RTX 5070

### 🟡 中等：`results/exploratory/rtx5070/` 已有正确标注

这个目录的 `hardware_metadata.json` 已正确标注为 RTX 5070 并声明 "final_claim_allowed: false"

### 🟢 正常：A100 C40 验证数据

`results/a100/phase-c42/c40_baseline_a100.json` 是正式的 A100 验证数据，包含硬件指纹和结论

### ❓ 需要进一步人工审查

| 路径 | 问题 |
|------|------|
| `results/confirmatory-*/` | base_dir 为 "/root/epic05-data"（疑为 A100 或远程），文件本身不含 GPU 名称 |
| `results/epic05/` | 不含显式硬件标签 |
| `results/measured/` | 无硬件标签，部分数据可能来自不同硬件 |
| `results/higs-round*/` | 无显式硬件标签（higs-round41b 样本显示 device="A100-SXM4-80GB" 但可能是更高端 A100） |
| `data/results/mipnerf360_*/` | 不含显式 GPU 名称，可能为 RTX 5070 |
| `reports/phase-a100/current_environment_fingerprint.json` | 需要确认硬件 |

---

## 4. 论文数据使用指引

### ✅ 可用于论文（A100 已验证）

| 数据 | 位置 |
|------|------|
| A100 基线 profiling | `results/a100/phase-a100/current_baseline_profiling.json` |
| CUB sort 深度 profile | `results/a100/phase-a100/cub_sort_deep_profile.json` |
| H6 完整审计 | `results/a100/phase-a100/h6_intersection_accounting.json` |
| C17-C30 候选锦标赛 | `results/a100/c17-c33/phase-c17-c2/` ~ `phase-c30/` |
| C31-C33 训练锦标赛 | `results/a100/c17-c33/phase-c31/` |
| C40 A100 重验证 | `results/a100/phase-c42/c40_baseline_a100.json` |
| A100 训练检查点 | `results/training/a100_30k_*.json` |

### ❌ 不可用于论文（RTX 5070 探索性数据）

| 数据 | 原位置 | 新位置 |
|------|--------|--------|
| C35 Tile密度 | `results/phase-c31/` | `results/rtx5070/phase-c31/ c38* | c39* | c40* | c41* | c42*` |
| C38 有效性研究 | 同上 | 同上 |
| C39 局部性研究 | 同上 | 同上 |
| C40 训练瓶颈（RTX 5070版） | 同上 | 同上 |
| C41 GPU利用率（RTX 5070版） | 同上 | 同上 |
| C42 D-SSIM 审计（RTX 5070版） | 同上 | 同上 |
| C7 跨硬件测试 | `results/phase-c20/` | `results/rtx5070/phase-c20/` |
| C42 优化候选筛选 | `results/phase-c42/` | `results/rtx5070/phase-c42/` |
| 2026-07 训练数据 | `data/results/rtx5070_*` | 不变（已标 RTX 5070） |

### ⚠️ 需人工确认后才能用于论文

| 数据 | 当前状态 |
|------|---------|
| `results/confirmatory-*` | 可能是 A100 远程数据，需确认 |
| `results/epic05/` | 无 GPU 标签，需人工确认运行环境 |
| `results/higs-round*/` | 样本显示 "A100-SXM4-80GB"（不同的 A100 变体） |
| `results/measured/` | 无 GPU 标签 |
| `data/results/mipnerf360_2026-07-15/` | 可能是 RTX 5070，需人工确认 |

---

## 5. 已验证的数值对比（C40 D-SSIM 瓶颈）

| 指标 | RTX 5070 (错误) | A100 (正确) | 差异 |
|------|:--------------:|:----------:|:----:|
| fwd_ms/iter | 6.86 | **3.38** | 2.03x |
| loss_ms/iter (D-SSIM) | 20.43 | **75.17** | 0.27x ⚠️ |
| bwd_ms/iter | 40.90 | **11.70** | 3.50x ⚠️ |
| total_ms/iter | 85.27 | **94.70** | 0.90x |
| D-SSIM 占 GPU 时间 | 40.7% | **78.6%** | **根本性不同** |
| 使用的 checkpoint | PSNR=12.03 (diverged) | PSNR=20.56 (正常) | ⚠️ |

**重要**: RTX 5070 声称的 "D-SSIM 占 40.7%" 低估了实际瓶颈严重程度。A100 验证显示 D-SSIM 占 **78.6%**。论文必须使用 A100 数据。

---

## 6. 已删除的重复/空目录

以下目录在整合后被删除（内容已移至 `results/a100/`）：
- `results/phase-a100/` → `results/a100/phase-a100/`
- `results/phase-c17-c2/` ~ `results/phase-c31/` → `results/a100/c17-c33/`
