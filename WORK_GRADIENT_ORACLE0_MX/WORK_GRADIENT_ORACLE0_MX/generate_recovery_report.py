#!/usr/bin/env python3
"""
Generate recovery report and update RUN_STATUS.
"""
import json
from datetime import datetime

# Load all audit results
storage_audit = {
    "root_partition": {
        "device": "/dev/sda1",
        "total_gb": 439,
        "used_gb": 429,
        "available_gb": 0,
        "usage_percent": 100
    },
    "storage_pool": {
        "device": "/dev/md0",
        "total_gb": 3584,
        "used_gb": 3553,
        "available_gb": 31,
        "usage_percent": 100
    },
    "temp_dir_created": "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp",
    "temp_dir_writeable": True,
    "solution": "TMPDIR environment variable pointing to storage pool"
}

source_identity = {
    "repo_path": "/home/liaoyuanjun/3dgs-renderer-benchmark",
    "head_commit": "32ab80e773f74f4d8e40ff4e338c86b29c7957f5",
    "branch": "master",
    "remotes": "NONE",
    "benchmark_file_in_worktree": False,
    "benchmark_file_in_head": False,
    "benchmark_file_in_history": True,
    "benchmark_extracted_from": "3bd1592",
    "benchmark_path": "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/run_higs_train_benchmark.py",
    "benchmark_sha256": "f2e24110d7080e35581bd4f62481042f18e6707643000ce21f9fbac9f99868ee",
    "benchmark_lines": 1881
}

checkpoint_inventory = {
    "total": 6,
    "loadable": 6,
    "garden": [
        {"stage": "EARLY_5K", "N": 2047889, "iteration": 5000},
        {"stage": "MIDDLE_10K", "N": 2404471, "iteration": 10000},
        {"stage": "LATE_FINAL", "N": 3104845, "iteration": 15000}
    ],
    "bicycle": [
        {"stage": "EARLY_5K", "N": 2456697, "iteration": 5000},
        {"stage": "MIDDLE_10K", "N": 3849060, "iteration": 10000},
        {"stage": "LATE_FINAL", "N": 4430283, "iteration": 15000}
    ]
}

gsplat_audit = {
    "version": "1.5.3",
    "path": "/home/liaoyuanjun/miniforge3/envs/nerficg/lib/python3.11/site-packages/gsplat",
    "has_experimental_module": False,
    "experimental_import_error": "cannot import name 'experimental' from 'gsplat'",
    "has_higs_dynamic": False,
    "sparse_tile_api": False,
    "benchmark_apis_present": True,
    "note": "gsplat 1.5.3 base install lacks experimental.HiGS APIs required by benchmark"
}

# Gate status
gates = {
    "TEMP": "PASS",
    "SOURCE": "PARTIAL",
    "BINARY": "BLOCKED",
    "CHECKPOINT": "PASS",
    "CAMERA": "UNKNOWN",
    "SPARSE_KERNEL": "BLOCKED"
}

# Generate report
report = f"""# WORK-GRADIENT-ORACLE0-MX 恢复报告

**执行日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**任务**: S0-S1R 环境恢复与阻塞解除  
**状态**: **PARTIAL_RECOVERY**

---

## 执行摘要

本轮成功解除了**磁盘空间阻塞**，使 Python、PyTorch 和 gsplat 恢复可导入。但在验证 CUDA sparse tile API 时发现 **gsplat 安装缺少 experimental.HiGS 模块**，导致无法执行真正的 sparse tile-mask smoke test。

### 主要发现

1. ✅ **存储恢复**: 创建隔离 TMPDIR，解决 0GB 磁盘空间问题
2. ✅ **Checkpoint 验证**: 6 个 FasterGSFused checkpoints 全部可加载，N 值已确认
3. ⚠️ **源码身份**: benchmark 文件从 git 历史提取，但当前 HEAD 不包含该文件
4. ❌ **CUDA Binary**: gsplat 1.5.3 缺少 experimental.HiGS sparse tile API
5. ❌ **Sparse Kernel**: 无法验证 tile-mask sparse rendering 功能

---

## P1 — 存储与 Python 恢复 ✅

### 磁盘空间审计
- **根分区 (/dev/sda1)**: 439G 总量，0G 可用 (100% 满)
- **Storage Pool (/dev/md0)**: 3.5T 总量，31G 可用 (100% 满)
- **Inode**: 根分区 27M 空闲，Storage pool 63M 空闲 (充足)

### 解决方案
创建隔离临时目录: `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp`

设置环境变量:
```bash
export TMPDIR=/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp
export TMP=/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp
export TEMP=/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp
```

### 验证结果
- ✅ 临时文件写入: PASS (10MB 测试文件)
- ✅ tempfile 模块: PASS (gettempdir() 返回新路径)
- ✅ PyTorch 导入: PASS (2.7.1+cu118, CUDA 11.8, A100 detected)
- ✅ gsplat 导入: PASS (1.5.3)

**结论**: Python 环境完全恢复可用。

---

## P2 — 源码身份审计 ⚠️

### 仓库状态
- **路径**: /home/liaoyuanjun/3dgs-renderer-benchmark
- **HEAD**: 32ab80e773f74f4d8e40ff4e338c86b29c7957f5
- **Branch**: master
- **Remotes**: NONE (本地仓库，无远程)
- **Dirty files**: 167 个修改/未跟踪文件

### Benchmark 文件状态
- **In worktree**: ❌ False
- **In HEAD**: ❌ False  
- **In target commit 32ab80e**: ❌ False
- **In git history**: ✅ True (25 个历史提交)

### 提取历史版本
从提交 `3bd1592` (最新包含该文件的提交) 提取:
- **路径**: `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/run_higs_train_benchmark.py`
- **SHA256**: `f2e24110d7080e35581bd4f62481042f18e6707643000ce21f9fbac9f99868ee`
- **行数**: 1881 行

### API 检查
所有关键 API 在提取的文件中**存在**:
- ✅ make_forward_fn
- ✅ higs_dynamic_ts
- ✅ tile_mask
- ✅ rasterize_gaussian_higs_dynamic
- ✅ _error_guided_mask
- ✅ _importance_l1_loss
- ✅ _masked_l1_loss

**问题**: 当前 HEAD (32ab80e) 不包含 benchmark 文件，而该文件依赖的 gsplat.experimental.HiGS API 在已安装的 gsplat 1.5.3 中不存在。

---

## P3 — Checkpoint 审计 ✅

### 盘点结果
**总计**: 6 个 checkpoints  
**可加载**: 6/6 (100%)

### Garden
| Stage | File | N | Iteration | Size |
|-------|------|---|-----------|------|
| EARLY_5K | 0005000.pt | 2,047,889 | 5,000 | 461 MB |
| MIDDLE_10K | 0010000.pt | 2,404,471 | 10,000 | 541 MB |
| LATE_FINAL | final.pt | 3,104,845 | 15,000 | 699 MB |

### Bicycle
| Stage | File | N | Iteration | Size |
|-------|------|---|-----------|------|
| EARLY_5K | 0005000.pt | 2,456,697 | 5,000 | 553 MB |
| MIDDLE_10K | 0010000.pt | 866.3 MB |
| LATE_FINAL | final.pt | 4,430,283 | 15,000 | 997 MB |

### 元数据验证
所有 checkpoints 包含:
- ✅ `model_state_dict`: Gaussian 参数 (means, scales, rotations, opacities, sh_coefficients)
- ✅ `num_iterations_trained`: 迭代数
- ✅ `model_name`: "fp1_garden" / "fp1_bicycle"
- ✅ `SH_DEGREE`: 3
- ✅ `creation_date`: 训练时间戳
- ✅ `output_directory`: 训练输出路径

**结论**: Checkpoint 身份明确，可作为 S1A 实验的真实训练状态。

---

## P4 — 最小可运行性验证 ❌

### gsplat 安装审计
- **版本**: 1.5.3
- **路径**: /home/liaoyuanjun/miniforge3/envs/nerficg/lib/python3.11/site-packages/gsplat
- **experimental 模块**: ❌ **不存在**
- **HiGS API**: ❌ **缺失**

### 关键问题
```python
from gsplat import experimental
# ImportError: cannot import name 'experimental' from 'gsplat'
```

benchmark 文件需要:
```python
from gsplat.experimental import rasterize_gaussian_higs_dynamic
```

但已安装的 gsplat 1.5.3 是**基础版本**，不包含 `experimental` 子模块和 HiGS 扩展 API。

### 影响
- ❌ 无法执行 tile-mask sparse rendering smoke test
- ❌ 无法验证 CUDA sparse kernel 是否跳过未选中 tile
- ❌ 无法确认 forward/backward/loss weighting 正确性
- ❌ **BLOCKED_SOURCE_IDENTITY**: 源码与已安装二进制不匹配

### 需要的操作
1. 确认 benchmark 源码版本 (3bd1592) 对应的正确 gsplat fork/commit
2. 检查是否需要从特定分支安装 gsplat with HiGS extensions
3. 或从源码重新编译包含 HiGS 的 gsplat

---

## Gate 状态总结

| Gate | 状态 | 说明 |
|------|------|------|
| **TEMP** | ✅ **PASS** | TMPDIR 创建成功，Python 环境恢复 |
| **SOURCE** | ⚠️ **PARTIAL** | Benchmark 文件已提取，但与当前 HEAD 不匹配 |
| **BINARY** | ❌ **BLOCKED** | gsplat 1.5.3 缺少 experimental.HiGS API |
| **CHECKPOINT** | ✅ **PASS** | 6 个 checkpoints 全部验证，N 值和 iteration 已确认 |
| **CAMERA** | ❓ **UNKNOWN** | 未验证（被 BINARY gate 阻塞） |
| **SPARSE_KERNEL** | ❌ **BLOCKED** | 无法测试（缺少 HiGS API） |

---

## 阻塞项与下一步

### 当前阻塞
**BLOCKED_BINARY_MISMATCH**: 已安装的 gsplat 1.5.3 与 benchmark 源码期望的 HiGS API 不匹配。

### 可能的根因
1. **gsplat 版本不匹配**: benchmark 文件 (3bd1592) 可能依赖特定 gsplat fork/commit，而当前安装的是上游 1.5.3
2. **未安装 HiGS 扩展**: HiGS 模块可能需要单独安装或编译
3. **源码分支错误**: 当前 HEAD (32ab80e) 不包含 benchmark，可能不是正确的工作分支

### 建议行动
1. **检查 gsplat 安装来源**:
   ```bash
   pip show gsplat
   ls -la /home/liaoyuanjun/miniforge3/envs/nerficg/lib/python3.11/site-packages/gsplat/
   ```

2. **查找 HiGS 相关 gsplat fork**:
   - 搜索 `caizefan34/3dgs-renderer-benchmark` 仓库的 setup/install 文档
   - 检查是否需要 `pip install -e .` 特定分支
   - 查找 `gsplat.experimental` 的安装说明

3. **验证 benchmark 源码版本兼容性**:
   - 检查提交 3bd1592 的 gsplat 依赖
   - 对比与 32ab80e 的 gsplat 版本要求

4. **备选方案**:
   - 如果 HiGS 扩展不可用，考虑使用 dense fallback + masked loss
   - 或从源码重新安装包含 HiGS 的 gsplat

---

## 已交付文件

### 审计结果
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/CHECKPOINT_INVENTORY.json`
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/CHECKPOINT_DEEP_RESULTS.json`
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/GSPLAT_CUDA_AUDIT.json`
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/source_audit.py` (源码审计脚本)
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/checkpoint_audit.py` (checkpoint 盘点脚本)
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/checkpoint_deep_inspect_v2.py` (深度检查脚本)
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/gsplat_audit_v2.py` (gsplat 审计脚本)

### 提取的源码
- `/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/run_higs_train_benchmark.py` (1881 行)

---

## 最终裁决

**PARTIAL_RECOVERY**

- ✅ 存储阻塞已解除
- ✅ Checkpoint 身份已验证
- ⚠️ 源码身份部分恢复（文件提取，但版本匹配性未知）
- ❌ CUDA binary 与源码不匹配（BLOCKED_BINARY_MISMATCH）
- ❌ 无法执行 smoke test

**需要**: 解决 gsplat HiGS API 缺失问题，或找到正确的 gsplat fork/版本。

**预计下一步时间**: 取决于 gsplat 安装问题的诊断和解决速度（数小时到一天）。
"""

# Save report
report_path = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/RECOVERY_REPORT.md"
with open(report_path, 'w') as f:
    f.write(report)

print(f"Recovery report saved to: {report_path}")
print("\n" + "="*60)
print("RECOVERY REPORT GENERATION COMPLETE")
print("="*60)

# Update RUN_STATUS
run_status = {
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "campaign": "WORK-GRADIENT-ORACLE0-MX",
    "status": "PARTIAL_RECOVERY",
    "stage": "S0-S1R_RECOVERY",
    
    "gates": {
        "TEMP": "PASS",
        "SOURCE": "PARTIAL",
        "BINARY": "BLOCKED",
        "CHECKPOINT": "PASS",
        "CAMERA": "UNKNOWN",
        "SPARSE_KERNEL": "BLOCKED"
    },
    
    "recovery_summary": {
        "storage_resolved": True,
        "python_restored": True,
        "checkpoints_verified": True,
        "gsplat_higs_api_missing": True,
        "smoke_test_executed": False
    },
    
    "blocking_issue": {
        "category": "BINARY_MISMATCH",
        "severity": "CRITICAL",
        "description": "Installed gsplat 1.5.3 lacks experimental.HiGS API required by benchmark source",
        "impact": "Cannot execute sparse tile-mask smoke test or verify CUDA sparse kernel"
    },
    
    "next_action_required": "Identify and install correct gsplat fork/version with HiGS extensions",
    
    "environment": {
        "hostname": "bms-39468022-001",
        "python": "/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python",
        "torch_version": "2.7.1+cu118",
        "cuda_version": "11.8",
        "gsplat_version": "1.5.3 (base install, missing HiGS)",
        "tmpdir": "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
    },
    
    "source": {
        "repo": "/home/liaoyuanjun/3dgs-renderer-benchmark",
        "head_commit": "32ab80e773f74f4d8e40ff4e338c86b29c7957f5",
        "benchmark_extracted_from": "3bd1592",
        "benchmark_sha256": "f2e24110d7080e35581bd4f62481042f18e6707643000ce21f9fbac9f99868ee"
    },
    
    "checkpoints": {
        "total": 6,
        "verified": 6,
        "garden_N_range": [2047889, 3104845],
        "bicycle_N_range": [2456697, 4430283]
    }
}

status_path = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/RUN_STATUS.json"
with open(status_path, 'w') as f:
    json.dump(run_status, f, indent=2)

print(f"RUN_STATUS saved to: {status_path}")
