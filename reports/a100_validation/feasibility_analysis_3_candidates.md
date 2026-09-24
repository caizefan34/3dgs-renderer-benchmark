# 3DGS Renderer Optimization — Engineering Feasibility Report

## Three Candidate Directions: A (C17-1), B (C42), C (Adaptive Tile Size)

**Platform**: gsplat 1.5.3, CUDA 11.8, A100 PCIe 40GB (108 SMs, 80 SM compute, 48 KB shared memory/block)
**Scenes**: room (926K Gaussians, 311 cams), bicycle (6.13M, 194 cams), garden (1.84M, 185 cams)
**Prior result**: C17-0 (counting sort + segmented CUB) REJECTED — 2.5x slower than baseline due to atomic scatter overhead

---

## A. C17-1: Fused Per-Tile Shared Memory Sort

### A.1 Modification Location in gsplat Pipeline

| Stage | File | Function | Change |
|-------|------|----------|--------|
| Pass 2 (write isect_ids) | `IntersectTile.cu` | `intersect_tile_kernel` | No change — still writes unsorted isect_ids + flatten_ids |
| Sort | `IntersectTile.cu` | `radix_sort_double_buffer` | **REPLACE** with new `fused_tile_sort_kernel` |
| Offset | `IntersectTile.cu` | `intersect_offset_kernel` | **REMOVE** — offsets computed inside fused kernel |
| Host dispatch | `Intersect.cpp` | `intersect_tile()` | Replace sort + offset calls with single kernel launch |
| Forward rasterize | `RasterizeToPixels3DGSFwd.cu` | — | No change (reads tile_offsets + flatten_ids as before) |
| Backward rasterize | `RasterizeToPixels3DGSBwd.cu` | — | No change |

**Key**: Only `IntersectTile.cu` and `Intersect.cpp` are modified. Downstream consumers are unchanged.

### A.2 Source Code Locations to Read

| File | Lines | Purpose |
|------|-------|---------|
| `gsplat/cuda/csrc/IntersectTile.cu` | 23-120 | `intersect_tile_kernel` (Pass 1 + Pass 2) |
| `gsplat/cuda/csrc/IntersectTile.cu` | 155-220 | `intersect_offset_kernel` (offset computation) |
| `gsplat/cuda/csrc/IntersectTile.cu` | 225-270 | `radix_sort_double_buffer` (CUB global sort) |
| `gsplat/cuda/csrc/Intersect.cpp` | 20-120 | `intersect_tile()` (host orchestration) |
| `gsplat/cuda/csrc/Intersect.h` | — | Function declarations |
| `gsplat/cuda/include/Common.h` | 23-30 | `CUB_WRAPPER` macro |
| `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` | — | Verify tile_offsets + flatten_ids read pattern |
| `gsplat/cuda/csrc/Utils.cuh` | — | Utility kernels/macros |

### A.3 Bottleneck Hypothesis

**Current bottleneck**: CUB `DeviceRadixSort::SortPairs` does 6 passes over 8.7M 64-bit keys (end_bit=46, RADIX_BITS=8). Each pass reads/writes all keys to global memory. Total global memory traffic: 6 × 8.7M × 16 bytes (key+value) = 835 MB.

**C17-1 hypothesis**: Each tile's intersections are independent. If sorted within shared memory:
- Each CTA loads its tile's intersections once from global memory
- Sorts entirely in shared memory (1-2 passes, no global memory traffic)
- Writes sorted output back to global memory once
- Total global memory traffic: 1 × 8.7M × 16 bytes = 139 MB (6x reduction)

**Estimated sort time reduction**: From ~1.1 ms (6-pass global sort) to ~0.2-0.4 ms (1-pass shared memory sort), saving 0.7-0.9 ms.

### A.4 Tile Intersection Distribution Analysis (Profiling Data)

**Method**: 6 cameras per scene, tile16, using `torch.bincount` on extracted tile_ids from unsorted isect_ids. Per-tile intersection counts for all non-zero tiles.

#### tile_size=16

| Scene | N_Gaussians | mean | p50 | p75 | p90 | p95 | p99 | p99.9 | max | >6144 (48KB) | >12288 (96KB) |
|-------|------------|------|-----|-----|-----|-----|-----|-------|-----|-------------|---------------|
| room | 926K | 1,068 | 1,014 | 1,296 | 1,581 | 1,765 | 2,139 | 2,645 | 3,795 | 0.00% | 0.00% |
| bicycle | 6.13M | 823 | 488 | 924 | 2,035 | 2,869 | 4,445 | 7,252 | 8,981 | 0.23% | 0.00% |
| garden | 1.84M | 350 | 263 | 408 | 745 | 979 | 1,416 | 1,930 | 2,354 | 0.00% | 0.00% |

#### tile_size=32

| Scene | mean | p50 | p90 | p95 | p99 | p99.9 | max | >6144 (48KB) | >12288 (96KB) |
|-------|------|-----|-----|-----|-----|-------|-----|-------------|---------------|
| room | 1,314 | 1,226 | 1,985 | 2,209 | 2,806 | 3,665 | 5,435 | 0.00% | 0.00% |
| bicycle | 1,660 | 1,029 | 4,052 | 5,571 | 8,911 | 12,815 | 16,968 | 4.07% | 0.17% |
| garden | 718 | 551 | 1,507 | 1,929 | 2,693 | 3,766 | 4,361 | 0.00% | 0.00% |

#### Shared Memory Feasibility Analysis

**A100 shared memory**: 48 KB per block (default), up to 164 KB with `cudaFuncSetAttribute`

Each intersection entry in shared memory: 12 bytes (8B key + 4B value)

| Limit | Entries | Bytes | Feasible? |
|-------|---------|-------|-----------|
| 48 KB (default smem) | 4,096 | 48 KB | room ✅, bicycle p99 ❌, garden ✅ |
| 100 KB (extended smem) | 8,533 | 100 KB | room ✅, bicycle p99.9 ❌, garden ✅ |
| 164 KB (max smem) | 13,989 | 164 KB | room ✅, bicycle p99 ✅, p99.9 ❌ |

**Critical finding**:
- **tile16**: Max intersection count across all 3 scenes = **8,981** (bicycle). Fits in 164 KB extended shared memory. Only 0.23% of tiles exceed 48 KB default.
- **tile32**: Max = **16,968** (bicycle). Does NOT fit in 164 KB. 4.07% of tiles exceed 48 KB.

**tile16 is mandatory for C17-1**. With tile16, 99.77% of tiles fit in default 48 KB shared memory. The 0.23% overflow tiles (bicycle dense regions) need a fallback path:
- Option A: Global memory sort fallback for overflow tiles (hybrid approach)
- Option B: Two-pass shared memory sort for tiles >4096 entries
- Option C: Skip the overflow tiles in shared memory, process them with a second kernel

### A.5 Minimal Experiment Validation Plan

1. **Phase 1 — Single-tile microbenchmark** (1 day):
   - Write a standalone CUDA kernel that sorts one tile's intersections in shared memory
   - Use block-wide radix sort (4-bit radix, 8 passes for 32-bit depth)
   - Measure: sort time per tile vs CUB global sort per-tile equivalent
   - Verify correctness against CPU reference sort

2. **Phase 2 — Integration microbenchmark** (2 days):
   - Replace `radix_sort_double_buffer` + `intersect_offset_kernel` with `fused_tile_sort_kernel`
   - Each CTA = 1 tile, loads its intersections from unsorted isect_ids
   - Sort in shared memory, write tile_offsets + sorted flatten_ids
   - Measure: total sort+offset time, compare to baseline
   - Use room scene (max=3,795, fits in 48 KB)

3. **Phase 3 — Overflow handling** (1 day):
   - Add fallback for tiles >4096 entries (bicycle 0.23%)
   - Measure on bicycle scene
   - Verify no regression

4. **Phase 4 — Full validation** (1 day):
   - 3 scenes, 6 cameras each
   - Compare: isect_ids, flatten_ids, tile_offsets, PSNR
   - Measure: sort time, offset time, full render time

### A.6 Profiling Metrics

| Metric | Measurement Method | Baseline Value |
|--------|-------------------|----------------|
| Sort kernel time | `isect_tiles(sort=True)` - `isect_tiles(sort=False)` | 1.14 ms (room) |
| Offset kernel time | `isect_offset_encode` timing | ~0.05 ms (est.) |
| Total sort+offset | Sum of above | ~1.19 ms |
| Full forward render | `rasterization()` median over 50 iters | 3.88 ms (room) |
| Global memory traffic | Nsight Compute | ~835 MB (6 passes) |
| Shared memory utilization | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | N/A (new kernel) |
| Tile occupancy | Non-zero tiles / total tiles | 100% (all scenes) |
| Overflow rate | Tiles >4096 / total tiles | 0.00% (room), 0.23% (bicycle) |

### A.7 Estimated Benefit Ceiling

| Component | Baseline | C17-1 Estimate | Savings |
|-----------|----------|----------------|---------|
| Sort (6-pass global) | 1.14 ms | 0.25 ms (1-pass shared) | 0.89 ms |
| Offset kernel | 0.05 ms | 0.00 ms (fused) | 0.05 ms |
| **Total** | **1.19 ms** | **0.25 ms** | **0.94 ms** |
| **E2E speedup** | — | — | **+24%** (0.94/3.88) |

**Upper bound**: +24% end-to-end speedup (if shared memory sort is 4x faster than global sort).
**Realistic estimate**: +15-20% (accounting for kernel launch overhead, warp divergence in overflow handling, and shared memory bank conflicts).

### A.8 Maximum Engineering Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overflow tiles (0.23% in bicycle) | HIGH — correctness must be preserved | Fallback to global memory sort for overflow tiles |
| Shared memory bank conflicts | MEDIUM — 48 KB with 32 banks = 1.5 KB/bank, stride access | Use padding or transpose |
| Low occupancy (1 CTA/tile, large smem) | MEDIUM — 164 KB smem → 1 CTA/SM, 108 CTAs total vs 8160 tiles | Multiple waves needed, but each tile is independent |
| Block-wide sort implementation complexity | HIGH — custom radix sort in shared memory | Use CUB block-wide sort (`cub::BlockRadixSort`) |
| CUB BlockRadixSort smem limit | MEDIUM — CUB block sort has internal smem requirements | Verify with `BlockRadixSort::TempStorage` size |
| Backward pass compatibility | LOW — backward doesn't call sort | No change needed |

**Overall risk**: MEDIUM-HIGH. The main risk is overflow handling for bicycle's dense tiles. Using `cub::BlockRadixSort` (a well-tested library) rather than a custom sort kernel significantly reduces implementation risk.

---

## B. C42: Resolution-Adaptive SSIM

### B.1 Modification Location in gsplat Pipeline

| Stage | File | Function | Change |
|-------|------|----------|--------|
| Loss computation | `scripts/epic05/phase7/loss.py` | `D-SSIM` / `compute_loss` | Downsample pred+gt before SSIM conv |
| Training loop | `scripts/epic05/phase7/train_3dgs.py` | Training iteration | Pass scale parameter to loss |
| Rendering | gsplat `rasterization` | — | No change (render at full resolution) |
| L1 loss | `train_3dgs.py` | — | No change (stays full resolution) |

**Key**: Only the SSIM loss computation changes. Rendering and L1 loss are untouched. This is a pure Python modification — no CUDA code.

### B.2 Source Code Locations to Read

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/epic05/phase7/loss.py` | 1-78 | D-SSIM implementation (Gaussian window, conv2d) |
| `scripts/epic05/phase7/train_3dgs.py` | 44-60 | TrainingConfig (loss weights) |
| `scripts/epic05/phase7/train_3dgs.py` | loss call site | Where `compute_loss` is called with pred, gt |
| `scripts/epic05/phase7/gaussian_model.py` | 108 | `forward()` (renders at full resolution) |

### B.3 Bottleneck Hypothesis

**Current bottleneck**: D-SSIM loss runs 5 forward conv2d kernels + 3 backward dgrad kernels on full-resolution (1920×1080) images. At scale=1.0, the SSIM computation takes ~40ms/iteration on room (out of ~95ms total), i.e., **42% of iteration time**.

**C42 hypothesis**: SSIM is a perceptual metric that doesn't need full pixel resolution. Downsampling pred+gt by scale factor s before the 11×11 Gaussian convolution reduces conv cost by s² while preserving gradient direction (verified: cosine >0.98 at s=0.75).

**Speedup model**:
- SSIM cost scales as s² (pixel count)
- L1 cost unchanged (stays full-res)
- Total iteration cost: T(s) = T_L1 + s² × T_SSIM
- At s=0.75: T = T_L1 + 0.5625 × T_SSIM → speedup = T(1.0) / T(0.75)

### B.4 Scale Ablation Design

| Scale | Resolution | SSIM pixels | SSIM cost ratio | Expected speedup |
|-------|-----------|-------------|-----------------|-----------------|
| 1.000 | 1920×1080 | 2.07M | 100% | baseline |
| 0.875 | 1680×945 | 1.59M | 76.6% | +10-15% |
| 0.750 | 1440×810 | 1.17M | 56.3% | +30-34% |
| 0.625 | 1200×675 | 0.81M | 39.1% | +45-55% |
| 0.500 | 960×540 | 0.52M | 25.0% | +60-65% |

**Experiment protocol**:
1. Train each scale for 5K iterations, room scene, seed=42
2. Evaluate PSNR/SSIM at full resolution every 500 iterations
3. Gradient cosine analysis at iters 1000/3000/5000 (vs scale=1.0)
4. Topology analysis: clone/split/prune counts, final Gaussian count
5. Decision gates: dPSNR > -0.2 dB, dSSIM > -0.005, dGS < 10%, speedup > 30%

### B.5 Adaptive Schedule Experiment Design

**Hypothesis**: Starting with aggressive downsampling (s=0.5) and gradually increasing to s=1.0 as training converges may combine early-stage speed with late-stage quality.

**Schedules to test**:

| Schedule | s at iter 0 | s at iter 5K | s at iter 10K | s at iter 30K |
|----------|-------------|--------------|---------------|---------------|
| Constant 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| Linear ramp 0.5→1.0 | 0.50 | 0.58 | 0.67 | 1.00 |
| Cosine ramp 0.5→1.0 | 0.50 | 0.55 | 0.65 | 1.00 |
| Step 0.5→0.75→1.0 | 0.50 | 0.50 | 0.75 | 1.00 |
| Early stop 0.75→1.0 | 0.75 | 0.75 | 1.00 | 1.00 |

**Experiment protocol**:
1. 30K iterations, room scene, 13 eval cameras
2. Measure: wall-clock training time, PSNR/SSIM at 30K, Gaussian count at 30K
3. Compare: constant 0.75 (validated) vs adaptive schedules
4. Decision: adaptive is worthwhile only if it achieves >0.1 dB PSNR improvement over constant 0.75 at similar wall-clock time

**Adaptive trigger signals** (for runtime adaptation):
- L1 loss plateau detection (running average over 500 iters, slope < threshold)
- PSNR improvement rate (ΔPSNR over 1000 iters < 0.01 dB)
- Gaussian count stabilization (ΔGS < 1% over 1000 iters)

### B.6 Profiling Metrics

| Metric | Measurement Method | Baseline Value (s=1.0) |
|--------|-------------------|----------------------|
| SSIM forward time | `time.perf_counter` around `compute_loss` | ~40 ms/iter |
| L1 forward time | Isolate L1 computation | ~5 ms/iter |
| Total iteration time | `time.perf_counter` around full iter | ~95 ms/iter |
| Gradient cosine | Per-parameter cosine vs s=1.0 | 1.0 (reference) |
| Gradient magnitude ratio | Per-parameter mag ratio vs s=1.0 | 1.0 (reference) |
| PSNR | 13 cameras, full-res eval | 18.32 dB (room @5K) |
| SSIM metric | 13 cameras, full-res eval | 0.6835 |
| Gaussian count | `model.xyz.shape[0]` | 896K (room @5K) |
| D-SSIM conv FLOPs | 5 kernels × 2.07M pixels × 121 (11×11) | 1.25 GFLOP |

### B.7 Estimated Benefit Ceiling

| Scale | Sort savings | E2E speedup (measured) | Quality (dPSNR) |
|-------|-------------|----------------------|-----------------|
| 0.75 | — | +33.7% (room, bicycle, garden) | +0.04 to +0.59 dB |
| 0.50 | — | +60.5% (estimated) | -0.01 dB (marginal) |
| 0.625 | — | +45-50% (estimated) | TBD |

**Already validated**: s=0.75 passes all gates across 3 scenes with +30-34% speedup and zero quality loss.
**Upper bound**: s=0.5 gives +60% speedup but topology diverges +10.6% (marginal).
**Sweet spot**: s=0.75 is the validated primary candidate. s=0.625 is untested but estimated to give +45-50% with acceptable topology.

### B.8 Maximum Engineering Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| Quality degradation at aggressive scales | MEDIUM — s=0.25 causes -1.93 dB | Scale sweep with gradient cosine gate |
| Topology divergence | LOW — s=0.75 is within 10% gate | Monitor clone/split/prune |
| Adaptive schedule instability | LOW — schedule changes may cause gradient oscillation | Gradual transitions, eval at checkpoints |
| Generalization to other scenes | LOW — validated on 3 Mip-NeRF 360 scenes | Already cross-scene validated |
| F.interpolate overhead | LOW — area-mode interpolation is O(n) | Negligible vs SSIM conv savings |

**Overall risk**: LOW. C42 is a pure Python modification with no CUDA code. Already validated at s=0.75 across 3 scenes. The scale ablation and adaptive schedule are incremental experiments on a proven approach.

---

## C. Adaptive Tile Size Runtime Selection

### C.1 Modification Location in gsplat Pipeline

| Stage | File | Function | Change |
|-------|------|----------|--------|
| Rasterization call | `scripts/epic05/phase7/train_3dgs.py` | `rasterization(tile_size=...)` | Change tile_size per iteration |
| Rasterization call | `scripts/epic05/phase7/gaussian_model.py` | `forward()` | Pass tile_size parameter |
| gsplat internals | `gsplat/cuda/_wrapper.py` | `rasterization()` | No change (already accepts tile_size) |
| isect_tiles | `gsplat/cuda/_wrapper.py` | `isect_tiles()` | No change (already accepts tile_size) |

**Key**: gsplat already accepts `tile_size` as a runtime parameter. No CUDA modification needed. Only the training loop needs to decide which tile_size to use.

### C.2 Source Code Locations to Read

| File | Lines | Purpose |
|------|-------|---------|
| `gsplat/cuda/_wrapper.py` | `rasterization()` | tile_size parameter handling |
| `gsplat/cuda/_wrapper.py` | `isect_tiles()` | tile_size → tile_width/tile_height computation |
| `gsplat/cuda/csrc/IntersectTile.cu` | 23-120 | `intersect_tile_kernel` — tile_size usage in tile bounds |
| `scripts/epic05/phase7/train_3dgs.py` | TrainingConfig | TILE_SIZE constant |

### C.3 Bottleneck Hypothesis

**Current state**: Fixed tile_size=16 for all training iterations. C43 profiling showed:
- tile16: 4.27 ms render (room), 9.16 ms (bicycle), 3.36 ms (garden)
- tile32: 4.84 ms render (room), 12.11 ms (bicycle), 4.59 ms (garden)
- tile16 is faster on ALL 3 scenes with SfM init (confirmed)

**Hypothesis**: Tile size affects parallelism, not rendering algorithm. The optimal tile_size depends on:
1. **Gaussian radii distribution** — larger radii → more tiles per Gaussian → tile16 better
2. **Gaussian count** — more Gaussians → more intersections → more parallelism needed
3. **Scene density** — dense regions favor smaller tiles (more parallelism), sparse regions favor larger tiles (less overhead)

**Adaptive hypothesis**: Early training (large radii, few Gaussians) may benefit from tile16, while late training (smaller radii, many Gaussians) may benefit from tile32 for some scenes. But profiling at SfM init (early) and 10K checkpoint (late) both showed tile16 is faster.

### C.4 Decision Metrics for Runtime Selection

To select tile_size at runtime, we need:

| Metric | Computation | Cost | Decision Rule |
|--------|-------------|------|---------------|
| **Mean intersections per tile** | `n_isects / n_nonzero_tiles` | O(1) from isect output | <1000 → tile32, ≥1000 → tile16 |
| **Max intersections per tile** | `max(torch.bincount(tile_ids))` | O(n_isects) | >4000 → tile16 (shared mem sort) |
| **Non-zero tile fraction** | `n_nonzero_tiles / n_total_tiles` | O(n_tiles) | <50% → tile32, ≥50% → tile16 |
| **Mean Gaussian radius (pixels)** | `radii.float().mean()` | O(N) | >32px → tile16, <16px → tile32 |
| **GPU SM utilization** | Nsight or `n_isects / (n_tiles × SMs)` | O(1) | <0.5 → tile32, ≥0.5 → tile16 |

**Minimal decision function** (pseudocode, not implementation):
```
if mean_radius > 2 * tile_size:
    tile_size = 16   # large Gaussians cover many tiles
elif mean_radius < tile_size:
    tile_size = 32   # small Gaussians cover few tiles
else:
    tile_size = 16   # default
```

### C.5 Minimal Experiment Validation Plan

1. **Phase 1 — Per-iteration profiling** (0.5 day):
   - Train room for 5K iterations, profile tile16 vs tile32 every 500 iters
   - Measure: render time, n_isects, mean radius, Gaussian count
   - Determine if optimal tile_size changes during training

2. **Phase 2 — Per-scene profiling** (0.5 day):
   - Already done: tile16 is faster on all 3 scenes at both SfM init and 10K checkpoint
   - Confirm with 30K checkpoint if available

3. **Phase 3 — Adaptive selection** (1 day):
   - Implement decision function based on mean Gaussian radius
   - Train with adaptive tile_size, measure total wall-clock time
   - Compare to fixed tile16

4. **Phase 4 — Edge case** (0.5 day):
   - Test on bicycle (6.13M Gaussians, high tile density variance)
   - Verify no regression

### C.6 Profiling Metrics

| Metric | tile16 (room) | tile32 (room) | tile16 (bicycle) | tile32 (bicycle) |
|--------|--------------|--------------|-----------------|-----------------|
| Render time | 4.27 ms | 4.84 ms | 9.16 ms | 12.11 ms |
| N_isects | 4.2M | 1.7M | 7.7M | 3.9M |
| N_tiles | 8,160 | 2,040 | 8,160 | 2,040 |
| Mean isects/tile | 516 | 832 | 945 | 1,912 |
| Max isects/tile | 3,795 | 5,435 | 8,981 | 16,968 |
| Non-zero tile % | 100% | 100% | 100% | 100% |
| SM utilization | 5.8 isects/SM/tile | 9.4 isects/SM/tile | 10.6 | 21.7 |

### C.7 Estimated Benefit Ceiling

**Already measured**:
| Scene | tile16 vs tile32 speedup | Quality impact |
|-------|------------------------|----------------|
| room | +13.3% (tile16 faster) | 0.00 dB |
| bicycle | +32.2% (tile16 faster) | 0.00 dB |
| garden | +36.6% (tile16 faster) | 0.00 dB |

**Adaptive benefit**: Since tile16 is already optimal for all 3 scenes at all training stages, adaptive selection provides **~0% additional benefit** over simply using tile16.

**Upper bound**: If there exists a scene/config where tile32 is faster (e.g., trained model with tiny radii at inference), adaptive selection could provide up to +10% for that case. But this is an inference-time optimization, not training.

### C.8 Maximum Engineering Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| No benefit over fixed tile16 | HIGH — already confirmed tile16 is optimal | Low-cost: just use tile16 |
| Profiling overhead per iteration | LOW — mean radius is O(N) | Compute every 100 iters, not every iter |
| Decision instability | LOW — tile_size is binary (16/32) | Hysteresis: don't switch more than once per 1000 iters |
| Quality impact | ZERO — tile_size doesn't affect rendering | Already confirmed dPSNR=0.00 |

**Overall risk**: LOW. But **benefit is also LOW** — tile16 is already the optimal choice for all tested scenes. The adaptive selection adds complexity for ~0% gain. The only scenario where it helps is inference with trained models that have very small Gaussian radii.

---

## Summary Comparison

| Criterion | A (C17-1 Fused Sort) | B (C42 Adaptive SSIM) | C (Adaptive Tile Size) |
|-----------|---------------------|----------------------|----------------------|
| **Modification type** | CUDA kernel | Python loss function | Python config |
| **Lines changed** | ~200 (CUDA) | ~10 (Python) | ~20 (Python) |
| **Build required** | Yes (nvcc) | No | No |
| **Estimated speedup** | +15-24% | +30-34% (validated) | ~0% (tile16 already optimal) |
| **Correctness risk** | Medium (overflow) | Low (validated) | Zero |
| **Quality risk** | Low (same algorithm) | Low (dPSNR > -0.2) | Zero |
| **Implementation effort** | 4-5 days | 1-2 days (ablation) | 1 day (no benefit expected) |
| **Prior validation** | C17-0 failed; C17-1 different approach | s=0.75 validated 3 scenes | tile16 validated 3 scenes |
| **Recommendation** | **PURSUE** — highest potential | **DEPLOY** — already validated | **SKIP** — no benefit |

### Recommendation

1. **B (C42)**: Deploy immediately at s=0.75. Already validated. Run scale ablation [1.0, 0.875, 0.75, 0.625, 0.5] and adaptive schedule experiments to find the optimal operating point.

2. **A (C17-1)**: Pursue as the next high-impact optimization. The tile distribution data confirms shared memory sort is feasible for tile16 (99.77% of tiles fit in 48 KB). Use `cub::BlockRadixSort` to minimize implementation risk. Expected +15-24% on top of C42.

3. **C (Adaptive Tile Size)**: Skip. tile16 is already optimal for all 3 Mip-NeRF 360 scenes at all training stages. The adaptive selection adds complexity for zero benefit.

---

## Data Provenance

| Item | Path |
|------|------|
| Tile distribution profiler | `scripts/phase-c42/c17_1_tile_distribution.py` |
| Tile distribution data | `results/a100/phase-c42/c17_1_tile_distribution.json` |
| C17-0 validation (room) | `results/a100/phase-c42/c17_0_validation_room.json` |
| C17-0 validation (bicycle) | `results/a100/phase-c42/c17_0_validation_bicycle.json` |
| C17-0 validation (garden) | `results/a100/phase-c42/c17_0_validation_garden.json` |
| C17-0 report | `reports/a100_validation/c17_0_tile_segmented_sort.md` |
| C42 scale sweep data | `results/a100/phase-c42/track_a_scale_*.json` |
| C42 multi-scene data | `results/a100/phase-c42/track_a_multiscene_*.json` |
| C42 report | `reports/a100_validation/track_a_c42_scale_sweep.md` |
| C43 tile-size data | `results/a100/phase-c42/track_c_c43_*.json` |
| C43 report | `reports/a100_validation/track_c_c1_c43_validation.md` |
| Track B source analysis | `reports/a100_validation/track_b_c17_source_analysis.md` |
