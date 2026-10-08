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

### 环境验证待完成
- [ ] Torch 版本与 CUDA 11.8 兼容性
- [ ] gsplat 版本确认 (应为 1.5.3)
- [ ] .so 动态库路径与 SHA256
- [ ] tile_mask 稀疏渲染功能 smoke test

## 下一步
1. 确认 FasterGSFused checkpoints 是否符合实验要求（early/late 定义）
2. 验证环境依赖版本
3. 执行 tile_mask sparse rendering smoke test
4. 如全部通过，进入 S1A 阶段

## 阻塞项
- **待验证**: FasterGSFused checkpoint 的 provenance 和 N 值（是否是真实训练 state）
- **待验证**: garden 场景缺少完整 PLY 文件（仅有 COLMAP sparse），需要转换或寻找替代
