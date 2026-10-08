# WORK-GRADIENT-ORACLE0-MX 执行状态

**当前阶段**: S0 环境审计与源码检查  
**状态**: IN_PROGRESS  
**日期**: 2026-10-08

## S0 进度

### 环境审计
- ✅ SSH 连接成功：bms-39468022-001 (MX)
- ✅ GPU 检测：8× NVIDIA A100-PCIE-40GB (SM80)
- ✅ Conda 环境：/home/liaoyuanjun/miniforge3/envs/nerficg
- ✅ 存储路径：/mnt/storage_pool/liaoyuanjun
- ✅ 数据集路径：/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360
  - garden: 185 张图像
  - bicycle: 完整结构 (eval_cameras.json, eval_images, point_cloud.ply)

### 源码审计
- ✅ 仓库路径：/home/liaoyuanjun/3dgs-renderer-benchmark
- ✅ Git commit: 32ab80e773f74f4d8e40ff4e338c86b29c7957f5
- ⚠️ 工作树状态：有未提交修改 (baseline/reference_v1/*.py, scripts/phase-r0.1/collect_checkpoints.sh)
- ✅ 关键文件存在：benchmark/run_higs_train_benchmark.py (1881 行)
- ✅ 核心 API 确认：
  - `make_forward_fn(backend, ..., tile_mask=...)` 支持 higs_dynamic_ts
  - `higs_dynamic_ts` backend 调用 `rasterize_gaussian_higs_dynamic`
  - `_error_guided_mask` 使用 `torch.multinomial(replacement=True)`
  - `_importance_l1_loss` 使用权重 `m/(k*p)`

### Checkpoint 可用性
**garden**:
- ✅ PLY 格式：/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/garden/sparse/0/ (COLMAP 结构)
  - cameras.bin, images.bin, points3D.bin
- ❌ 无 .pt 训练 checkpoint（仅有 COLMAP 稀疏点云）
- 需要：从其他来源获取 garden 的 early (如 5K) 和 late (如 30K) checkpoint

**bicycle**:
- ✅ PLY 格式：/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/point_cloud.ply
- ✅ eval_cameras.json：82,083 字节（完整评估相机配置）
- ❌ 无 .pt 训练 checkpoint（仅有最终 PLY）

**FasterGSFused checkpoints (已发现)**:
- ✅ garden 5K: /mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints/0005000.pt (483 MB)
- ✅ garden 10K: .../0010000.pt (567 MB)
- ✅ garden final: .../final.pt (732 MB)
- ✅ bicycle 5K: /mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt (579 MB)
- ✅ bicycle 10K: .../0010000.pt (908 MB)
- ✅ bicycle final: .../final.pt (1.0 GB)

### 环境验证
- ✅ Torch 版本：2.7.1+cu118 (符合预期)
- ❌ gsplat 导入失败：TMPDIR 磁盘空间不足 (/tmp, /var/tmp, /home 均 100% 满)
- ❌ Storage pool 磁盘空间：31G 可用 (3.5T 总量，100% 使用)
- ❌ benchmark/run_higs_train_benchmark.py 文件不存在于预期路径

### 磁盘空间阻塞
**当前状态**:
- `/dev/sda1` (根分区): 439G 总量，429G 已用，**0G 可用** (100% 满)
- `/dev/md0` (storage pool): 3.5T 总量，3.5T 已用，**31G 可用** (100% 满)

**大目录分析**:
- `/mnt/storage_pool/liaoyuanjun/data`: 2.1T (最大占用)
- `/home/liaoyuanjun/projects`: 90G
- `/home/liaoyuanjun/miniforge3`: 63G
- `/home/liaoyuanjun/3dgs-renderer-benchmark`: 46G

**影响**:
- Python 无法创建临时文件（gsplat 导入失败）
- 无法运行任何需要临时存储的 Python 脚本
- 磁盘不足以进行实验（需要存储 checkpoints、临时数据、结果）

## 阻塞项
- **BLOCKED_ENVIRONMENT**: 磁盘空间严重不足
  - 根分区 100% 满 (0G 可用)
  - Storage pool 100% 满 (31G 可用)
  - Python 无法运行（临时目录不可用）
  - 无法清理他人数据
  - 需要用户清理或扩容磁盘

- **BLOCKED_SOURCE_IDENTITY**: benchmark/run_higs_train_benchmark.py 不存在
  - 需要确认实际源码路径或同步完整仓库

## 建议行动
1. **紧急**: 清理 `/mnt/storage_pool/liaoyuanjun/data` (2.1T) 或其他非关键数据
2. 确认 3dgs-renderer-benchmark 仓库完整性
3. 验证 gsplat 安装是否完整
4. 如磁盘问题无法解决，需 BLOCKED_ENVIRONMENT 并交付诊断
