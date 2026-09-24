# C17-2 Pre-Implementation Report — 10 Critical Questions

**Date:** 2026-10-19  
**Prepared from:** `source_audit.md` and `design.md`

---

## Q1: C17-2 优化的具体对象是什么？

**Answer:** C17-2 优化的是 **forward rasterization kernel** (`rasterize_to_pixels_3dgs_fwd_kernel`) 的内循环。

具体来说，当前 kernel 中每个 pixel 的 compositing loop：

```cuda
// Current: each pixel processes ALL Gaussians in the tile until done
for (uint32_t t = 0; t < batch_size && !done; ++t) {
    // alpha computation, compositing
    if (next_T <= 1e-4f) done = true;  // pixel done
}
```

C17-2 引入 **active pixel mask**（共享内存），当一个 pixel 的 transmittance T 降到阈值 `1e-4` 以下时，将其标记为 "done"。当 tile 中所有 pixel 都 done 时，**整块 tile 提前退出**，跳过剩余的 batch 加载和 compositing。

**注意这不是参数的优化，而是算法层面的 kernel 结构优化。**

---

## Q2: C17-1 与 C17-2 的边界是什么？

**Answer:** 两个优化完全独立，boundary 明确：

| Dimension | C17-1 (Tile-Local Bounded Queues) | C17-2 (Active Pixel Tracking) |
|-----------|----------------------------------|-------------------------------|
| **Target module** | `IntersectTile.cu` + sort dispatch | `RasterizeToPixels3DGSFwd.cu` |
| **Target stage** | Intersection materialization + global sort | Rasterization compositing |
| **Data structures** | `isect_ids`, `flatten_ids` → per-tile queue | **None** — all changes inside kernel |
| **Memory** | Large per-tile pre-allocated buffers | 36 bytes shared memory per block |
| **Backward** | No change (sort is `@torch.no_grad`) | No change (`last_ids` unchanged) |
| **Correctness guarantee** | Sort equivalence | Mathematical identity (same T threshold) |
| **Implementation complexity** | High (new CUDA kernels, binding, rasterization consumption) | **Low** (one kernel, template param) |
| **Implementation risk** | Medium-high (overflow handling, sort correctness) | **Very low** (mathematically guaranteed) |

**C17-2 will NOT touch**:
- `IntersectTile.cu` (intersection + sort)
- `Intersect.cpp` (C++ binding)
- `Rasterization.cpp` (rasterization binding — unless new flag needed)
- `_wrapper.py` (Python API — unless new flag needed)
- `RasterizeToPixels3DGSBwd.cu` (backward kernel)

---

## Q3: 为什么这个优化可能有效？

**Answer:** 有三个独立的理由：

### 理由 1: 当前 kernel 存在已知的低效性

当前 batch termination 条件是 `__syncthreads_count(done) >= block_size`，这意味着一个 tile（256 pixels）必须在 **所有 256 个 pixel 都 saturate 后**才能提前退出。但实际上，不同 pixel 在 tile 内 saturate 的深度差异很大：

- Sky pixel (覆盖很少 Gaussians)：可能在 Gaussian #10 就 saturate
- Foreground pixel (密集几何区域)：可能需要 Gaussian #2000+ 才 saturate
- 背景 pixel (无任何 Gaussian 覆盖)：**永远不会 saturate**（T 始终 = 1.0）

对于包含背景的 tile（这是非常常见的场景），当前 kernel 永远不会提前退出——因为背景 pixel 的 T 永远不会降到阈值以下。

### 理由 2: 数学保证

`T` 是单调递减的，一旦 `next_T <= 1e-4`，后续 Gaussian 对 pixel 的贡献为：
```
Δcolor = color × α × T ≤ color × α × 1e-4 ≈ 0
```

这意味着提前退出不改变 pixel 值。这是一个**数学保证**，不是统计猜测。

### 理由 3: 工作负载量化

| 场景 | per-tile mean | 估计饱和深度中位数 | 估计 workload reduction |
|------|:-------------:|:------------------:|:----------------------:|
| bicycle t16 | 746 | ~200-400 | 30-50% compositing loop |
| garden t16 | 751 | ~200-400 | 30-50% compositing loop |
| room t16 | 490 | ~100-200 | 20-40% compositing loop |
| bicycle t32 | 1,737 | ~500-800 | 40-60% compositing loop |

由于 compositing loop 占 forward kernel 的 **~70-80%** 时间（基于 kernel 结构分析），整体 forward 受益估计为 15-35%。

---

## Q4: 预计减少什么 workload？

**Answer:** 减少以下 workload 的混合：

### 4a. **Gaussian 属性加载** (global memory reads)

每个 batch 中，每个线程加载一个 Gaussian 的 `means2d[g]`, `conics[g]`, `opacities[g]` — 这是 3 × 4 bytes = 12 bytes 的 global memory 读取。如果 tile 提前退出，这些加载完全避免。

### 4b. **Alpha 计算** (ALU operations)

每个 pixel 对每个 batch 中的每个 Gaussian 做：
```
sigma = 0.5 × (conic.x × delta.x² + conic.z × delta.y²) + conic.y × delta.x × delta.y
alpha = min(0.999, opacity × exp(-sigma))
next_T = T × (1 - alpha)
if next_T <= 1e-4 → done
```
这包括 2 次 FMA、1 次 exp、1 次 min。

### 4c. **Compositing** (FMA + memory)

当 alpha > threshold 时：
```
vis = alpha × T
pix_out[k] += color_k × vis
T = next_T
```
每个 channel 1 FMA。

### 量化

对于 bicycle t16 (6.08M isects, 256 pixels per tile):

| Metric | Baseline | C17-2 (est.) | Δ |
|--------|:--------:|:------------:|:-:|
| Gaussian loads per pixel | 6.08M × 256 = 1.56B | ~1.01B | -35% |
| Alpha computations | 1.56B | ~1.01B | -35% |
| Compositing ops (when α>threshold) | ~100-500M | ~65-325M | -35% |

**这不是 FLOPS reduction（GPU 不介意多几个 ALU），而是减少了 global memory 的读取压力。**

---

## Q5: 哪个 kernel 最可能受益？

**Answer:** 唯一受益的 kernel：**`rasterize_to_pixels_3dgs_fwd_kernel`**

```
Kernel name:       rasterize_to_pixels_3dgs_fwd_kernel<CDIM, scalar_t>
Source file:       RasterizeToPixels3DGSFwd.cu
Grid config:       [I, tile_height, tile_width]  (e.g., 1×68×120 = 8,160 blocks)
Block config:      [tile_size, tile_size]  (16×16 = 256 threads)
Shared memory:     7,168 + 36 bytes (C17-2 new) = 7,204 bytes
```

**不受益的 kernels**:
- `intersect_tile_kernel` — 不受影响（未修改）
- `intersect_offset_kernel` — 不受影响（未修改）
- CUB radix sort — 不受影响
- `rasterize_to_pixels_3dgs_bwd_kernel` — 不受影响（唯一接触的后向接口是 `last_ids`，未改变）
- Projection kernels — 不受影响

---

## Q6: 最大 correctness risk 是什么？

**Answer:** 

### 主要风险: NONE（实现正确的条件下）

原因：active pixel mask 只应用于当 `next_T <= 1e-4f`。一旦 T 降到这个阈值以下，数学上所有后续 Gaussian 的贡献都在舍入误差范围内。因此不改变任何 pixel 的数值输出。

### 次要风险 (需要验证):
1. **`last_ids` 一致性**：如果 pixel 在某个 Gaussian 后跳过了后续 compositing，但 backward 需要知道哪个 Gaussian 是最后一个贡献的，`last_ids` 必须是正确的。C17-2 中，`last_ids[pix_id]` 在 pixel 被标记 done 时写入（与 baseline 相同的 `cur_idx` 值）。
2. **Shared memory bank conflict**: `active_set` 数组（8 × uint32）访问模式应该是无冲突的（每个线程只访问自己 pixel 的 bit）。
3. **`atomicSub` 精确性**: 多个 thread 同时 `atomicSub(active_count, 1)` 是安全的。

### 最大风险评分: 1/5 (LOW)

---

## Q7: backward / gradient 是否可能受影响？

**Answer:** **不直接受影响。**

Backward kernel 读取的 forward 输出：
- `render_alphas` — 与 baseline 一致（T 值不同，但计算的 alpha 结果相同）
- `last_ids` — 与 baseline 一致（最后贡献的 Gaussian 索引不变）
- `flatten_ids` — 未改变
- `tile_offsets` — 未改变

Backward 的计算：每个 pixel 遍历 `last_ids[pixel]` 之前的 Gaussians，沿逆向 path 计算梯度。由于 `last_ids` 不变，backward 不会看到任何差异。

**Gradient correctness** (第 7 阶段) 将通过 finite difference 验证，确保 `g_backward ≈ g_numerical`。

---

## Q8: 如何验证？

**Answer:** 严格分阶段验证：

### Stage 1: Microbenchmark (最小 workload)
- 构造 3-4 个 deterministic tile workloads：低占用 / 中占用 / 高占用 / 全屏覆盖
- 测量：per-tile active pixel count、saturation depth、batch count、Gaussian loads saved
- 验证 early termination 的确在 `active_count==0` 时触发

### Stage 2: Forward Correctness (像素级比较)
- 随机生成 deterministic Gaussian set（固定 seed，少量 Gaussians）
- baseline vs C17-2: 比较 rendered image
- 门限: `max_abs_diff < 1e-6`（浮点精度内应该一致）

### Stage 3: Backward Correctness
- 运行 forward + backward
- 检查 NaN / Inf / illegal memory access

### Stage 4: Gradient Correctness
- 对 position, opacity, scale 做 central difference finite difference
- `g_num` vs `g_backward` 比较
- 接受标准: `rel_diff < 1e-4`

### Stage 5: Quality
- 500-step sanity training: 记录 PSNR trajectory
- baseline vs C17-2: PSNR diff < 0.1 dB

### Stage 6: Performance
- 多次运行 (100+)，warmup，median/mean/variance
- 报告: forward latency, kernel time, workload metrics

---

## Q9: 最小实验是什么？

**Answer:** 一个实验，同时验证 correctness + 测量 potential benefit：

```
最小实验: 3 个 deterministic tile workloads

Setup:
- 固定 camera, 固定 random seed
- 3 个 canvas: empty(0 GS) / sparse(10 GS) / dense(1000 GS)
- Resolution: tile_size × tile_size (256 pixels)
- 运行 baseline 和 C17-2

Expected:
1. 空 tile: 无 Gaussian → active_count 正确 = 0
2. Sparse: 每个 pixel 都看到所有 GS → 无 early exit 触发
3. Dense: 前后 pixel 饱和深度不同 → 定量测量 saved work

如果最小实验通过 → 扩展到完整 3 个 scene。
```

**Estimated time**: 30 minutes (编译 + 运行 + 验证)

---

## Q10: 是否建议批准 implementation？

**Answer:** ⏳ **Conditional APPROVE — with gates**

### 批准条件

✅ **设计理由充分**：基于 kernel 结构分析，early termination 有数学保证，无 correctness 风险。

✅ **实现范围明确**：仅修改一个 CUDA kernel 文件，不触及绑定、不改变数据结构、不影响 backward。

✅ **风险可控**：template parameter 机制确保 baseline 路径可以完全复现。

✅ **独立可验证**：像素级 diff + gradient check + finite difference 的完整验证链。

⏳ **实现前需要确认**:

| # | 事项 | 预期答复 |
|:-:|------|----------|
| 1 | C17-2 是否认可为 "Rasterization Early Batch Termination" 方向？ | (等待确认) |
| 2 | 是否同意实现 `active_set` 作为一个 `uint32_t[8]` 共享内存？ | (等待确认) |
| 3 | 是否同意使用 `USE_ACTIVE_SET` template parameter 控制？ | (等待确认) |
| 4 | Phase 2 (sparse mode warp-gather) 是否推迟到下一个迭代？ | 建议推迟 |
| 5 | 是否需要同时修改 `RasterizeToPixels2DGSFwd.cu`？ | 建议不修改（超出范围） |

### 实现优先级

```
Phase 1 (必须):  active_set tracking + early batch termination
Phase 2 (可选):  sparse-mode warp-gather for <32 active pixels
Phase 3 (独立):  composability test with C17-1 (如果 C17-1 实施)
```

**建议**: 如果以上 5 个确认点通过，开始 Phase 1 实现。

---

## 总结

| 维度 | 评估 |
|------|:----:|
| **优化对象** | Forward rasterization kernel compositing loop |
| **C17-1 边界** | 完全独立——C17-2 只改 rasterization, C17-1 只改 sort |
| **为什么有效** | 背景 pixel 永不 saturate，阻止 batch-level early exit |
| **减少 work** | 30-50% 的 Gaussian-pixel evaluation |
| **受益 kernel** | `rasterize_to_pixels_3dgs_fwd_kernel` 唯一 |
| **Correctness risk** | 极低——数学等价（相同 T threshold） |
| **Backward 影响** | 无——`last_ids` 不变 |
| **验证方法** | 像素级 diff + finite difference gradient check |
| **最小实验** | 3 deterministic tile workloads (30 min) |
| **建议** | ✅ **Conditional APPROVE** |
