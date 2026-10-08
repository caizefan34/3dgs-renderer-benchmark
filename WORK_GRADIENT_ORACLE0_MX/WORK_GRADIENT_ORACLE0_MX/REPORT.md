# WORK-GRADIENT-ORACLE0-MX 执行报告

**实验日期**: 2026-10-08  
**执行阶段**: S0 (环境审计与源码检查)  
**最终裁决**: **BLOCKED_ENVIRONMENT**  
**Fresh MX 测量**: ❌ 未完成（因磁盘空间阻塞）

---

## 摘要

本实验在 MX 节点 (bms-39468022-001) 上启动时遭遇**严重磁盘空间不足**问题，导致所有后续阶段（S1A 梯度 Oracle、S1B 在线代理和 GPU 成本验证）无法执行。

**主要阻塞原因**:
1. **磁盘空间耗尽**: 根分区 0GB 可用 (100% 满)，Storage Pool 31GB 可用 (100% 满)
2. **Python 无法运行**: gsplat 导入失败，临时目录不可用
3. **源码缺失**: benchmark/run_higs_train_benchmark.py 主文件不存在于预期路径
4. **Checkpoint 未验证**: 无法加载和验证训练状态元数据

---

## OBSERVED (已观察证据)

### 环境配置 ✅
- **主机**: bms-39468022-001 (MX 节点)
- **GPU**: 8× NVIDIA A100-PCIE-40GB (SM80, 40960 MiB)
- **Python**: /home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
- **Torch**: 2.7.1+cu118
- **CUDA**: 11.8
- **存储路径**: /mnt/storage_pool/liaoyuanjun
- **数据集路径**: /mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360

### 磁盘空间状态 ❌
```
/dev/sda1 (根分区):     439G 总量, 429G 已用,   0G 可用  (100% 满)
/dev/md0 (storage):   3584G 总量, 3553G 已用,  31G 可用  (100% 满)
```

**最大目录占用**:
- `/mnt/storage_pool/liaoyuanjun/data`: 2.1 TB
- `/mnt/storage_pool/liaoyuanjun/hf_home`: 47 GB
- `/mnt/storage_pool/liaoyuanjun/runs`: 43 GB
- `/home/liaoyuanjun/projects`: 90 GB
- `/home/liaoyuanjun/miniforge3`: 63 GB

### 数据集结构 ✅
**garden**:
- 185 张训练图像
- COLMAP 稀疏点云 (cameras.bin, images.bin, points3D.bin)
- 无 eval_cameras.json / eval_images
- 无 .pt 训练 checkpoint

**bicycle**:
- 完整结构：images/, eval_images/, eval_cameras.json (82KB), point_cloud.ply
- 无 .pt 训练 checkpoint

### 可用 Checkpoints (未验证)
**FasterGSFused 输出** (位于 /mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/):
- garden_5k: 0005000.pt (483 MB)
- garden_10k: 0010000.pt (567 MB)
- garden_final: final.pt (732 MB)
- bicycle_5k: 0005000.pt (579 MB)
- bicycle_10k: 0010000.pt (908 MB)
- bicycle_final: final.pt (1.0 GB)

**问题**: 无训练日志、配置或元数据，无法确认 provenance 和 N 值。

### 源码状态 ❌
- **仓库**: /home/liaoyuanjun/3dgs-renderer-benchmark
- **Git commit**: 32ab80e773f74f4d8e40ff4e338c86b29c7957f5
- **Git status**: dirty (3 个修改文件，多个未跟踪文件)
- **关键文件缺失**: benchmark/run_higs_train_benchmark.py (1881 行，主 benchmark 文件)
- **其他文件存在**: baseline/reference_v1/*.py (已修改)

---

## DERIVED (推导结论)

### 磁盘空间影响链
1. **Python 临时目录不可用** → gsplat 导入失败 (FileNotFoundError: No usable temporary directory)
2. **无法执行任何 Python 脚本** → 无法验证 checkpoint、运行 smoke test、执行实验
3. **Storage pool 31GB 不足** → 实验需要存储：
   - 临时数据、梯度张量、mask 样本
   - 结果文件（tile-level 梯度、时序数据）
   - 预估需求：至少 100-200 GB

### 实验协议要求 vs 实际能力
**PROTOCOL_LOCK.json 要求**:
- 64 tiles × 32 masks × 8 states × 2 cameras = 32,768 次 tile-mask 渲染
- 每次需要 CUDA 事件、梯度张量、成本特征
- 需要真实 GPU 计时（30 warmup + 100 timed × 3 paired）

**实际能力**: 
- 磁盘 0GB 可用 → 无法存储任何中间数据
- 主 benchmark 文件缺失 → 无法验证 tile_mask API
- gsplat 未验证 → 无法确认稀疏渲染支持

---

## FAILURES / CONFOUNDERS

### 主要失败原因

#### 1. 磁盘空间耗尽 (BLOCKED_ENVIRONMENT)
- **现象**: 根分区和 storage pool 均 100% 满
- **根因**: 
  - `/mnt/storage_pool/liaoyuanjun/data` 占用 2.1 TB
  - 其他大目录：hf_home (47GB), runs (43GB), strong_baselines (19GB)
  - 无自动清理机制
- **影响**: 
  - Python 无法运行（临时目录不可用）
  - 无法执行任何科学计算
  - 无法存储实验结果
- **是否需要清理**: 是（需要清理至少 100-200 GB）

#### 2. 源码主文件缺失 (BLOCKED_SOURCE_IDENTITY)
- **现象**: benchmark/run_higs_train_benchmark.py 不存在
- **可能原因**:
  - 仓库未完整同步
  - 文件被误删除或移动到其他位置
  - Git 分支不正确
- **影响**: 
  - 无法验证 tile_mask 稀疏渲染 API
  - 无法确认 make_forward_fn 签名
  - 无法审计 `_error_guided_mask` 和 `_importance_l1_loss` 实现

#### 3. Checkpoint Provenance 未验证 (BLOCKED_MISSING_REAL_STATES)
- **现象**: FasterGSFused checkpoints 存在但无元数据
- **缺失信息**:
  - N (Gaussian 数量)
  - 训练配置、seed
  - 训练日志、iteration 标记
- **风险**: 
  - 可能是 580K 误标为 6M (如 SOURCE_INSPECTION.md 第 24 行所述)
  - 可能不是真实训练状态
  - 可能来自其他场景或配置

---

## HYPOTHESIS TESTS (未执行)

由于磁盘空间阻塞，所有科学假设均未测试：

### S1A 梯度价值 Oracle
- ❌ **H1**: O (真实梯度 oracle) 的 geometry NMSE/variance 相对 U/E 可减少 ≥30%
- ❌ **前置条件**: 需要 64 tile pool × 32 masks × 8 states 的真实梯度数据

### S1B 在线代理和 GPU 成本
- ❌ **H2**: P (在线代理) 相对最强 U/E 可减少 ≥20% gradient NMSE
- ❌ **H3**: P 的 scheduler overhead ≤ 5%
- ❌ **前置条件**: 需要真实 CUDA 计时和 matched pixel budgets

---

## VERDICT & NEXT ACTION

### 最终裁决
**BLOCKED_ENVIRONMENT**

**理由**:
1. 磁盘空间严重不足（根分区 0GB，storage pool 31GB）
2. Python 无法执行（临时目录不可用）
3. 主源码文件缺失
4. 无法验证 checkpoint provenance

### 已完成的证据收集
- ✅ SSH 连接测试
- ✅ GPU 库存
- ✅ 磁盘空间分析
- ✅ 数据集结构检查
- ✅ 可用 checkpoint 路径记录
- ✅ Git 状态快照

### 缺失的证据
- ❌ gsplat 版本和 .so SHA256
- ❌ tile_mask sparse rendering smoke test
- ❌ Checkpoint N 值和 provenance
- ❌ 真实 GPU 计时基线

### 下一步行动 (需要用户干预)

#### 紧急行动 (阻塞实验)
1. **清理磁盘空间** (需要 100-200 GB):
   ```bash
   # 建议清理目标（需要用户确认安全性）:
   - /mnt/storage_pool/liaoyuanjun/data (2.1 TB) → 检查是否可归档/删除
   - /mnt/storage_pool/liaoyuanjun/hf_home (47 GB) → 检查 HuggingFace 缓存
   - /mnt/storage_pool/liaoyuanjun/runs (43 GB) → 检查旧运行结果
   - /home/liaoyuanjun/projects (90 GB) → 检查非活跃项目
   ```

2. **验证源码完整性**:
   ```bash
   cd /home/liaoyuanjun/3dgs-renderer-benchmark
   git status
   ls -la benchmark/run_higs_train_benchmark.py
   # 如果缺失，需要重新同步或定位实际路径
   ```

3. **验证 checkpoint provenance** (磁盘清理后):
   ```python
   import torch
   ckpt = torch.load("path/to/checkpoint.pt")
   print("N:", ckpt["means"].shape[0])
   print("Iteration:", ckpt.get("iteration", "UNKNOWN"))
   print("Config:", ckpt.get("config", {}))
   ```

#### 恢复实验条件后
1. 重新执行 S0 环境审计（验证 gsplat、.so、tile_mask）
2. 验证 8 个 checkpoint 的 provenance (garden/bicycle × early/late)
3. 执行 S1A 梯度 Oracle
4. 执行 S1B 在线代理和 GPU 计时

### 代码修改需求
- ❌ 无（阻塞是环境问题，非代码问题）

### 是否需要 Codex 介入
- **短期**: 否（需要用户清理磁盘和验证源码）
- **长期**: 可能需要（如果 checkpoint provenance 不清楚，需要重新训练或找到真实来源）

---

## MANIFEST

### 已交付文件
```
C:\Users\36570\3dgs-renderer-benchmark\WORK_GRADIENT_ORACLE0_MX\WORK_GRADIENT_ORACLE0_MX\
├── CAPABILITY_AUDIT.json          # 环境审计结果 (BLOCKED)
├── PROGRESS.md                     # 详细执行日志
├── REPORT.md                       # 本报告
└── (其他源文件：DSH_EXECUTION_PROMPT.md, PROTOCOL_LOCK.json, etc.)
```

### 已收集的证据
- SSH 连接日志
- GPU 库存 (nvidia-smi)
- 磁盘使用报告 (df -h, du -sh)
- Git 状态 (git rev-parse, git status)
- 数据集结构 (ls -la)

### 缺失的证据 (因阻塞)
- gsplat 版本和 .so SHA256
- tile_mask smoke test 结果
- Checkpoint 元数据和 N 值
- GPU 计时基线

---

## 附录：原始命令日志

### 磁盘空间检查
```bash
ssh mx "df -h /tmp /var/tmp /home/liaoyuanjun /mnt/storage_pool"
# 结果：所有分区 100% 满，0GB-31GB 可用

ssh mx "du -sh /mnt/storage_pool/liaoyuanjun/* | sort -rh | head -20"
# 结果：data (2.1T), hf_home (47G), runs (43G), 等

ssh mx "du -sh /home/liaoyuanjun/* | sort -rh | head -20"
# 结果：projects (90G), miniforge3 (63G), 3dgs-renderer-benchmark (46G)
```

### 源码检查
```bash
ssh mx "cd /home/liaoyuanjun/3dgs-renderer-benchmark && git rev-parse HEAD"
# 结果：32ab80e773f74f4d8e40ff4e338c86b29c7957f5

ssh mx "cd /home/liaoyuanjun/3dgs-renderer-benchmark && git status --porcelain"
# 结果：3 个修改文件，多个未跟踪文件

ssh mx "find /home/liaoyuanjun/3dgs-renderer-benchmark -name 'run_higs_train_benchmark.py'"
# 结果：文件不存在
```

### Checkpoint 检查
```bash
ssh mx "ls -la /mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/"
# 结果：找到 6 个 .pt 文件 (garden/bicycle × 5K/10K/final)

ssh mx "/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python -c 'import torch; ...' 2>&1"
# 结果：FileNotFoundError (临时目录不可用)
```

---

**报告生成时间**: 2026-10-08T14:35:00Z  
**执行者**: DSH (DeepSeek Harness)  
**状态**: 已阻塞，等待用户干预
