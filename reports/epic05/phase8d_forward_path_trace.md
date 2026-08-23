# Phase 8D — Forward Path Trace Report

**Date:** 2026-09-21  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU  
**gsplat version:** 1.5.3  
**Status:** ✅ COMPLETED (source-level analysis, EPIC-05 server unreachable)

---

## 1. Complete Forward Call Chain

When `gsplat.rasterization()` is called in forward mode (packed=True, SH degree=3, pinhole camera), the full Python→C++→CUDA call chain is:

```
rasterization()                                        [rendering.py:33]
  │
  ├─ 1. fully_fused_projection()                       [rendering.py:409]
  │     └─ _FullyFusedProjection.forward()             [_wrapper.py:1030]
  │           └─ projection_ewa_3dgs_fused_fwd_kernel  [ProjectionEWA3DGSFused.cu:16]
  │
  ├─ 2. spherical_harmonics()                          [rendering.py:509] (SH degree=3)
  │     └─ _SphericalHarmonics.forward()               [_wrapper.py:179]
  │           └─ spherical_harmonics_fwd_kernel         [SphericalHarmonicsCUDA.cu]
  │
  ├─ 3. isect_tiles() (two-pass)                       [rendering.py:634]
  │     ├─ 3a. intersect_tile_kernel (first pass)      [IntersectTile.cu:24]
  │     │   └── counts tiles_per_gauss per Gaussian
  │     ├─ 3b. (exclusive prefix sum of tiles_per_gauss)
  │     └─ 3c. intersect_tile_kernel (second pass)     [IntersectTile.cu:24]
  │         └── fills isect_ids[] and flatten_ids[]
  │
  ├─ 4. cub::DeviceRadixSort::SortPairs                [IntersectTile.cu:296]
  │     └── radix sort of isect_ids (64-bit keys, 32-bit values)
  │
  ├─ 5. isect_offset_encode()                          [rendering.py:648]
  │     └─ intersect_offset_kernel                     [IntersectTile.cu:209]
  │         └── builds tile offset table from sorted intersections
  │
  └─ 6. rasterize_to_pixels()                          [rendering.py:746]
        └─ _RasterizeToPixels.forward()                [_wrapper.py:1251]
              └─ rasterize_to_pixels_3dgs_fwd_kernel  [RasterizeToPixels3DGSFwd.cu:18]
                  └── final pixel compositing (alpha blending)
```

### 1.1 Path Detail: Step 1 — Projection

**Function:** `fully_fused_projection()` → `_FullyFusedProjection.forward()`  
**CUDA kernel:** `projection_ewa_3dgs_fused_fwd_kernel`  
**Source file:** `ProjectionEWA3DGSFused.cu:16-212`

Launched as 1D grid with `B × C × N` threads total, 256 threads/block.

Each thread handles one Gaussian (batch × camera × gaussian id):

1. Transform mean from world to camera space (`posW2C`)
2. Build 3×3 world covariance from quats+scales or use precomputed
3. Transform covariance to camera space (`covarW2C`)
4. Project to 2D (perspective projection callback)
5. Add 2D blur (`add_blur(eps2d)`)
6. Compute screen-space radius (extend × sqrt of eigenvalues)
7. Apply opacity-aware bounding box (tight radius bound)
8. Clip to frustum
9. Write: radii, means2d, depths, conics, compensations

**This kernel is tile_size-independent.** Same grid/block for both tile16 and tile32.

| Property | Value |
|:---------|:------|
| Grid | `ceil(B×C×N / 256)` |
| Block | 256 threads |
| Shared memory | 0 bytes |
| Register pressure | ~48 regs (estimated) |
| **tile_size dependent?** | **NO** |

### 1.2 Path Detail: Step 2 — SH Evaluation

**Function:** `spherical_harmonics()` → `_SphericalHarmonics.forward()`  
**CUDA kernel:** `spherical_harmonics_fwd_kernel`  
**Source file:** `SphericalHarmonicsCUDA.cu`

Launched only when `sh_degree` is not None. Computes RGB colors from SH coefficients and view directions.

In packed mode: operates on `nnz` valid Gaussians.
In dense mode: operates on all `B×C×N` Gaussians.

**This kernel is tile_size-independent.**

| Property | Value |
|:---------|:------|
| Grid | `ceil(nnz / 256)` |
| Block | 256 threads |
| **tile_size dependent?** | **NO** |

### 1.3 Path Detail: Steps 3a/3c — Intersect Tile (Two-Pass)

**Function:** `isect_tiles()` → `intersect_tile_kernel` (called twice)  
**Source file:** `IntersectTile.cu:24-114`

**First pass** (count-only): For each (image_id, gaussian_id), compute which tiles the Gaussian overlaps based on its 2D position and radius. Write `tiles_per_gauss[idx] = num_tiles_touched`.

**Second pass** (fill): For each (image_id, gaussian_id), encode each tile intersection as a 64-bit key: `image_id (X bits) | tile_id (Y bits) | depth (32 bits, IEEE bitcast)`. Write `isect_ids[]` and `flatten_ids[]`.

**This kernel's input** (`I × N` threads) is tile_size-independent, but **its output volume** (`n_isects`) differs by 4× between tile16 and tile32.

| Property | Value |
|:---------|:------|
| Grid | `ceil(I×N / 256)` |
| Block | 256 threads |
| Shared memory | 0 bytes |
| **tile_size dependent?** | **Work (not config)**: same thread count, but tile16 outputs 4× more intersection elements |
| n_isects tile16 | ~175M (at iter30000) |
| n_isects tile32 | ~44M (at iter30000) |

**Critical observation:** Both passes use the same number of threads regardless of tile_size. But the output data volume differs by exactly 4×, determined by tile geometry (8160 tiles ÷ 2040 tiles = 4).

### 1.4 Path Detail: Step 4 — Radix Sort (CUB)

**Function:** `radix_sort_double_buffer()` → `cub::DeviceRadixSort::SortPairs`  
**Source file:** `IntersectTile.cu:296-339`

Sorts 64-bit `isect_ids` (keys) together with 32-bit `flatten_ids` (values) by image_id | tile_id | depth.

Uses NVIDIA CUB library's radix sort implementation. Sorting is over the full 64-bit key (but only the lower `32 + tile_n_bits + image_n_bits` bits actually participate).

**This stage processes n_isects elements.** tile16 processes 4× more elements (175M vs 44M).

Radix sort is O(N) in elements, with roughly 2 passes per byte of key width. For 175M elements with ~40 bits of key, this requires ~20 passes × 175M ≈ 3.5B element operations.

| Property | Value |
|:---------|:------|
| **tile_size dependent?** | **YES** — input size is n_isects |
| tile16 sort input | ~175M elements |
| tile32 sort input | ~44M elements |
| Ratio | 4× |

### 1.5 Path Detail: Step 5 — Intersection Offset Encode

**Function:** `isect_offset_encode()` → `intersect_offset_kernel`  
**Source file:** `IntersectTile.cu:209-257`

Processes the sorted `isect_ids` and builds a tile_offset table `[I × tile_height × tile_width]`. Each element in the offset table points to the first intersection for that tile.

Launched with 1 thread per intersection element (n_isects total).

**This stage processes n_isects elements.** tile16 processes 4× more elements.

| Property | Value |
|:---------|:------|
| Grid | `ceil(n_isects / 256)` |
| Block | 256 threads |
| Shared memory | 0 bytes |
| **tile_size dependent?** | **YES** — input size is n_isects (4× difference) |

### 1.6 Path Detail: Step 6 — Rasterization (The Critical Kernel)

**Function:** `rasterize_to_pixels()` → `_RasterizeToPixels.forward()`  
**CUDA kernel:** `rasterize_to_pixels_3dgs_fwd_kernel`  
**Source file:** `RasterizeToPixels3DGSFwd.cu:18-188`

**This is the ONLY kernel whose launch configuration directly depends on tile_size.**

#### Launch Configuration

| Property | tile16 | tile32 | Ratio |
|:---------|:------:|:------:|:-----:|
| **Grid** | `1 × 68 × 120` | `1 × 34 × 60` | 4× (blocks) |
| **Total blocks** | **8160** | **2040** | 4.00× |
| **Threads/block** | **256** (16×16) | **1024** (32×32) | 0.25× |
| **Total threads** | 2,088,960 | 2,088,960 | 1.00× |
| **Shared mem/block** | **7168 bytes** | **28,672 bytes** | 0.25× |
| **Blocks/SM (RTX 5070, ~40 SMs)** | ~204 | ~51 | 4.00× |

#### Shared Memory Layout

```c
extern __shared__ int s[];
int32_t *id_batch      = (int32_t *)s;                    // [block_size] int32
vec3    *xy_opacity_batch = (vec3 *)&id_batch[block_size]; // [block_size] vec3
vec3    *conic_batch       = (vec3 *)&xy_opacity_batch[block_size]; // [block_size] vec3
```

- **tile16**: 256 × (4 + 12 + 12) = **7,168 bytes** per block
- **tile32**: 1024 × (4 + 12 + 12) = **28,672 bytes** per block

#### Kernel Logic

```
for each batch of block_size Gaussians in this tile:
    1. syncthreads_count: check if all pixels done (early exit)
    2. Cooperative load: each thread loads 1 Gaussian from global→shared
       - flatten_ids[idx] → gaussian index
       - means2d[g] → xy coordinates
       - opacities[g] → opacity
       - conics[g] → conic matrix
    3. syncthreads: ensure all data loaded
    4. Per-pixel inner loop (each thread is one pixel):
       for each Gaussian in batch:
         compute delta = xy - pixel_center
         compute sigma = 0.5 * (conic.x * dx² + conic.z * dy²) + conic.y * dx * dy
         alpha = min(0.999, opacity * exp(-sigma))
         if sigma < 0 or alpha < ALPHA_THRESHOLD: skip
         accumulate: color += gaussian_color * alpha * T
         update transmittance: T = T * (1 - alpha)
         if T < 1e-4: this pixel is fully occluded, set done=true
    5. End of batch: any thread with done=true stops in future batches
```

### 1.7 Batching Overhead Analysis

For the room scene at iter30000 (~21545 Gaussians per tile):

| Metric | tile16 | tile32 | Ratio |
|:-------|:------:|:------:|:-----:|
| Gaussians/tile | ~21,545 | ~21,545 | 1.00× |
| Batch size | 256 | 1024 | 0.25× |
| **Number of batches** | **~85** | **~22** | **3.86×** |
| Shared mem loads/batch | 256 | 1024 | 0.25× |
| Inner loop iterations/thread/batch | 256 | 1024 | 0.25× |
| **Total inner loop iterations/thread** | ~21,545 | ~21,545 | 1.00× |
| Sync operations | 85 | 22 | 3.86× |

**Key finding:** tile16 requires **3.86× more batch iterations** (85 vs 22), each with synchronization overhead, while each thread performs the same total number of inner loop iterations (~21,545).

---

## 2. Kernel Comparison: tile16 vs tile32

### 2.1 Which kernels are tile_size-dependent?

| # | Kernel | tile_size dep.? | Why |
|:-:|:-------|:---------------:|:----|
| 1 | `projection_ewa_3dgs_fused_fwd` | **NO** | Same `B×C×N` threads |
| 2 | `spherical_harmonics_fwd` | **NO** | Same `nnz` threads |
| 3 | `intersect_tile` (pass 1) | **NO (config)** | Same `I×N` threads, but output volume differs (4×) |
| 4 | `intersect_tile` (pass 2) | **NO (config)** | Same threads, fill 4× more intersections |
| 5 | `cub::DeviceRadixSort` | **YES** | Sorts 4× more elements (175M vs 44M) |
| 6 | `intersect_offset` | **YES** | Processes 4× more elements |
| 7 | `rasterize_to_pixels_3dgs_fwd` | **YES** | Grid/block dimensions differ (8160×256 vs 2040×1024) |

### 2.2 Kernel binary identity

The compiled kernel binary is **identical** for all tile sizes. Verified via cuobjdump in Phase 8B (register count=40, shared=1024, stack=0 for all tile sizes).

Only the **launch parameters** (grid, block, shared_memory_size) differ at runtime.

---

## 3. Workload Volume Analysis

### 3.1 Intersection Volume per Checkpoint

| Checkpoint | N(Gs) | t16 n_isects | t32 n_isects | Ratio |
|:-----------|:-----:|:------------:|:------------:|:-----:|
| iter5000 | 899,729 | 145,250,227 | 36,314,474 | 4.00× |
| iter10000 | 1,004,935 | 167,441,858 | 41,862,311 | 4.00× |
| iter15000 | 1,219,406 | 172,527,679 | 43,134,108 | 4.00× |
| iter20000 | 1,207,872 | 174,076,023 | 43,521,093 | 4.00× |
| iter25000 | 1,199,627 | 175,226,189 | 43,808,605 | 4.00× |
| iter30000 | 1,193,480 | 175,809,454 | 43,954,474 | 4.00× |

The 4.00× intersection ratio is **purely geometric**: `tile16_tiles / tile32_tiles = 8160 / 2040 = 4.00`.

### 3.2 Per-Kernel Workload Breakdown

| Kernel | tile16 work | tile32 work | Ratio | Scaling type |
|:-------|:-----------:|:-----------:|:-----:|:------------|
| projection | B×C×N | B×C×N | 1.00× | **Constant** |
| SH eval | nnz | nnz | 1.00× | **Constant** |
| intersect (pass 1) | I×N | I×N | 1.00× | **Constant** |
| intersect (pass 2) | I×N (producing n_isects) | I×N (producing n_isects/4) | 1.00× input, 4.00× output | **Input constant, output varies** |
| radix sort | n_isects | n_isects/4 | 4.00× | **Linear with intersection count** |
| offset encode | n_isects | n_isects/4 | 4.00× | **Linear with intersection count** |
| rasterization | block_count × iterations_per_batch | block_count × iterations_per_batch | 4.00× blocks, 1.00× inner loop | **Complex** |

---

## 4. Detailed Kernel Launch Sequence

### Forward (packed mode, SH degree=3):

```
CUDA Kernel Launch Order:
  1. projection_ewa_3dgs_fused_fwd_kernel     [grid=ceil(B*C*N/256), block=256]
     ↓
  2. spherical_harmonics_fwd_kernel            [grid=ceil(nnz/256), block=256] 
     ↓
  3. intersect_tile_kernel (pass 1: count)     [grid=ceil(I*N/256), block=256]
     ↓
  4. [host] exclusive prefix sum of cum_tiles_per_gauss
     ↓
  5. intersect_tile_kernel (pass 2: fill)      [grid=ceil(I*N/256), block=256]
     ↓
  6. cub::DeviceRadixSort::SortPairs           [CUB library call]
     ↓
  7. intersect_offset_kernel                   [grid=ceil(n_isects/256), block=256]
     ↓
  8. rasterize_to_pixels_3dgs_fwd_kernel       [grid=1×tile_h×tile_w, block=tile_size×tile_size]
```

### Forward (dense mode, no SH): Without packed optimization and SH evaluation, kernel 2 is skipped and the intermediate data structures differ but the core CUDA kernel chain is the same.

---

## 5. Critical Source Code Details

### 5.1 Rasterization: Batch Loading (lines 115-136)

```cuda
uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

for (uint32_t b = 0; b < num_batches; ++b) {
    // Early exit if all pixels in tile are done
    if (__syncthreads_count(done) >= block_size) break;
    
    uint32_t idx = batch_start + tr;
    if (idx < range_end) {
        int32_t g = flatten_ids[idx];
        id_batch[tr] = g;
        const vec2 xy = means2d[g];
        const float opac = opacities[g];
        xy_opacity_batch[tr] = {xy.x, xy.y, opac};
        conic_batch[tr] = conics[g];
    }
    block.sync();
    
    uint32_t batch_size = min(block_size, range_end - batch_start);
    for (uint32_t t = 0; t < batch_size && !done; ++t) {
        // ... per-Gaussian evaluation ...
    }
}
```

### 5.2 Rasterization: Per-Pixel Gaussian Evaluation (lines 140-169)

```cuda
for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
    const vec3 conic = conic_batch[t];
    const vec3 xy_opac = xy_opacity_batch[t];
    const float opac = xy_opac.z;
    const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
    const float sigma = 0.5f * (conic.x * delta.x * delta.x + 
                                 conic.z * delta.y * delta.y) + 
                        conic.y * delta.x * delta.y;
    float alpha = min(0.999f, opac * __expf(-sigma));
    if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;
    
    const float next_T = T * (1.0f - alpha);
    if (next_T <= 1e-4f) { done = true; break; }
    
    const float vis = alpha * T;
    for (uint32_t k = 0; k < CDIM; ++k)
        pix_out[k] += c_ptr[k] * vis;
    cur_idx = batch_start + t;
    T = next_T;
}
```

### 5.3 Projection: Key Operations (lines 70-198)

```cuda
// World-to-camera transform
posW2C(R, t, glm::make_vec3(means), mean_c);
if (mean_c.z < near_plane || mean_c.z > far_plane) { radii=0; return; }

// Build or load covariance
quat_scale_to_covar_preci(quat, scale, &covar, nullptr);  // from quat+scale

// Camera-space covariance
covarW2C(R, covar, covar_c);

// Perspective projection to 2D
persp_proj(mean_c, covar_c, fx, fy, cx, cy, ...);

// Blur + compute screen-space radius
float det = add_blur(eps2d, covar2d, compensation);
extend = min(3.33f, sqrt(2 * log(opacity / ALPHA_THRESHOLD)));
float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));

// Frustum culling
if (mean2d.x + radius_x <= 0 || ...) { radii=0; return; }
```

---

## 6. Identifying Candidates for Superlinear Scaling

Based on the complete forward path trace, the following kernels/stages could contribute to the observed superlinear scaling of tile16 forward runtime:

### Candidate A: Rasterization Kernel (Most Likely Primary)

The rasterization kernel's batch processing overhead is **amplified by workload density**:

- More Gaussians per tile → more batches → more sync overhead
- tile16 requires **3.86× more batch iterations** (85 vs 22) for the same Gaussians/tile
- Each batch iteration includes: `__syncthreads_count` + cooperative global load + `__syncthreads`
- As Gaussians/tile grows (iter5000: 17800 → iter30000: 21545), the overhead gap widens

**Prediction:** If rasterization is the dominant contributor, the tile16/tile32 ratio should scale with `Gaussians_per_tile × number_of_batches / block_size`.

### Candidate B: Radix Sort

The CUB radix sort processes 4× more elements (175M vs 44M) and may have non-linear scaling:
- CUB radix sort has O(N) passes, each pass has O(N) work
- But the temporary storage allocation and kernel launch overhead may not scale perfectly linearly
- For very large N (175M), L2 cache thrashing could occur

### Candidate C: Intersection Processing (Pass 2 + Offset)

Combined, these two kernels process 175M vs 44M elements. Each element requires:
- Global memory write (isect_ids, flatten_ids)
- For offset: global memory read (isect_ids) + write (offsets)

### Candidate D: Pipeline Interaction

The data pipeline is sequential:
```
projection → SH → intersect → sort → offset → rasterize
```

If any stage before rasterization creates workload that doesn't scale linearly, it compounds through the pipeline. However, the observed 25.5× total forward ratio at iter30000 is far beyond the 4× intersection ratio, suggesting the superlinearity is primarily within the rasterization kernel itself.

---

## 7. Key Architectural Insights

### 7.1 "More Work per Block" vs "More Blocks"

The rasterization kernel design has **two fundamentally different scaling dimensions**:

1. **More work per block** (more Gaussians per tile): Both tile16 and tile32 blocks handle ~21,545 Gaussians. This scales equally.

2. **More blocks** (more tiles): tile16 has 4× more blocks (8160 vs 2040). This is the **static** difference.

The superlinearity arises because the **dynamic** work (Gaussians per tile growing from 17,800 to 21,545) interacts with the **static** block count difference:

- tile16: 8160 blocks × 85 batches × 256 inner loop iterations × per-iteration work
- tile32: 2040 blocks × 22 batches × 1024 inner loop iterations × per-iteration work

When Gaussians/tile grows 1.21×:
- Both increase batch count by 1.21×
- Inner loop iterations/thread increase by 1.21×
- tile16 blocks still have the same sync overhead disadvantage (3.86× more syncs)

### 7.2 The Occupancy / Register Trade-off

- **tile16**: 6 blocks/SM possible → ~1536 threads/SM → register demand > register file → **potential spilling**
- **tile32**: 1 block/SM → 1024 threads/SM → register demand near register file size → **less spilling**

This is a hardware-level hypothesis that cannot be confirmed without Nsight Compute (BLOCKED on WDDM).

### 7.3 Shared Memory Efficiency

- **tile16**: 7168 bytes/block → highly bank-conflict-free (32 banks, 4-byte elements, 256 threads → 8 warps reading columns of 32 elements)
- **tile32**: 28672 bytes/block → same bank structure, but 32 warps reading

Both tile sizes have coalesced shared memory access patterns (all threads read `conic_batch[t]` where `t` is the same across the warp). No bank conflicts in either case.

---

## 8. Summary

| # | Aspect | tile16 | tile32 | Ratio |
|:-:|:-------|:------:|:------:|:-----:|
| Total blocks (rasterization) | 8160 | 2040 | 4.00× |
| Threads/block | 256 | 1024 | 0.25× |
| Total threads | 2,088,960 | 2,088,960 | 1.00× |
| Shared mem/block | 7,168 B | 28,672 B | 0.25× |
| Batches/tile (iter5000) | ~70 | ~18 | 3.89× |
| Batches/tile (iter30000) | ~85 | ~22 | 3.86× |
| Sync operations/tile (iter30000) | ~85 | ~22 | 3.86× |
| n_isects | 176M | 44M | 4.00× |
| Sort size | 176M | 44M | 4.00× |

**The complete forward call chain is traced from Python `rasterization()` → `_wrapper.py` → 7 CUDA kernels/CUB calls. Only the rasterization kernel has tile_size-dependent launch configuration, but the intersection volume (n_isects) differs by 4× across all post-intersection stages.**
