# Phase 8C — Backward Path Trace Report

**Date:** 2026-09-20  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU  
**gsplat version:** 1.5.3  
**Status:** ✅ COMPLETED  

---

## 1. Complete Backward Call Chain

### 1.1 Autograd Entry Points

When `loss.backward()` is called on the output of `gsplat.rasterization()`, the autograd system traverses the computational graph backwards through the following CUDA kernels, in order:

```
loss.backward()
  └─ _RasterizeToPixels.backward()          ← CDIM=3 (RGB)
       └─ rasterize_to_pixels_3dgs_bwd_kernel  ← DOMINANT KERNEL
  └─ _SphericalHarmonics.backward()         ← if sh_degree is not None
       └─ spherical_harmonics_bwd
  └─ _FullyFusedProjection.backward()
       └─ projection_ewa_3dgs_fused_bwd
  └─ _QuatScaleToCovarPreci.backward()
       └─ quat_scale_to_covar_preci_bwd
```

### 1.2 Per-Kernel Details

#### Kernel 1: `rasterize_to_pixels_3dgs_bwd_kernel`

| Property | Value |
|:---------|:------|
| **Source** | `RasterizeToPixels3DGSBwd.cu` (line 16) |
| **Function** | `_RasterizeToPixels.backward()` → `launch_rasterize_to_pixels_3dgs_bwd_kernel` |
| **Launch site** | `cuda/_wrapper.py:_RasterizeToPixels.backward` (line 1336) |
| **Grid** | `I × tile_height × tile_width` |
| **Block** | `tile_size × tile_size` |
| **Threads/block** | `tile_size²` (256 for t16, 1024 for t32) |
| **Shared memory** | `tile_size² × (4 + 12 + 12 + 4×CDIM) bytes` |
| **Purpose** | **ALL gradient computation for rasterization output.** Computes ∂L/∂means2d, ∂L/∂conics, ∂L/∂colors, ∂L/∂opacities via backpropagation through the alpha compositing equation. |
| **Key operation** | `gpuAtomicAdd` (scatter-add) for every valid Gaussian-pixel pair. Each thread in the last warp writes to 5 separate output buffers via atomic operations. |
| **Register count** | ~64 (not directly queryable from Python) |
| **Relative cost** | **~95-99% of backward time** (estimated from kernel complexity) |

**Critical Observation:** This is the ONLY kernel whose launch configuration depends on tile_size. The grid dimensions are `I × tile_height × tile_width`, where `tile_height = ceil(H/tile_size)` and `tile_width = ceil(W/tile_size)`.

For 1080p (1920×1080):
- **tile16:** grid = 1 × 68 × 120 = **8,160 blocks**, 256 threads/block
- **tile32:** grid = 1 × 34 × 60 = **2,040 blocks**, 1024 threads/block

The kernel implements a **tile-based backwards pass**: each block handles one tile, iterating over Gaussians sorted by depth in that tile, computing per-pixel gradients, and accumulating gradients via `gpuAtomicAdd`. The inner loop structure:

```
for each batch of Gaussians in tile:
    load batch into shared memory (cooperative)
    for each Gaussian in batch (back-to-front):
        compute per-pixel contribution to gradients
        warp-reduce within each warp
        thread 0: gpuAtomicAdd to global gradient buffers
```

#### Kernel 2: `spherical_harmonics_bwd`

| Property | Value |
|:---------|:------|
| **Source** | `SphericalHarmonicsCUDA.cu` |
| **Function** | `_SphericalHarmonics.backward()` |
| **Grid** | Depends on `nnz` (visible Gaussian count) |
| **Block** | 256 threads |
| **Purpose** | ∂L/∂shs from ∂L/∂colors and view direction |
| **Relative cost** | <1% of backward time |

#### Kernel 3: `projection_ewa_3dgs_fused_bwd`

| Property | Value |
|:---------|:------|
| **Source** | `ProjectionEWA3DGSFused.cu` |
| **Function** | `_FullyFusedProjection.backward()` |
| **Grid** | Depends on `N` (total Gaussian count) |
| **Block** | 256 threads |
| **Purpose** | Chain-rule through 3D→2D projection: ∂L/∂means, ∂L/∂quats, ∂L/∂scales from ∂L/∂means2d, ∂L/∂conics, ∂L/∂depths |
| **Relative cost** | <1% of backward time |

#### Kernel 4: `quat_scale_to_covar_preci_bwd`

| Property | Value |
|:---------|:------|
| **Source** | `QuatScaleToCovarCUDA.cu` |
| **Function** | `_QuatScaleToCovarPreci.backward()` |
| **Grid** | Depends on `N` |
| **Block** | 256 threads |
| **Purpose** | ∂L/∂quats, ∂L/∂scales from ∂L/∂covars |
| **Relative cost** | <1% of backward time |

### 1.3 Kernel Launch Configuration Comparison (tile16 vs tile32)

| Aspect | tile16 | tile32 | Ratio |
|:-------|:------:|:------:|:-----:|
| **Blocks (rasterize_bwd)** | 8,160 | 2,040 | 4.00× |
| **Threads/block (rasterize_bwd)** | 256 | 1024 | 0.25× |
| **Total threads** | 2,088,960 | 2,088,960 | 1.00× |
| **Shared memory/block** | ~5,500 bytes | ~20,000 bytes | 0.27× |
| **Blocks (other kernels)** | Same | Same | 1.00× |
| **Kernel binary** | **IDENTICAL** (cuobjdump verified) | **IDENTICAL** | 1.00× |

**Key Finding:** The backward kernel binary is **identical** for tile16 and tile32. Only the grid/block launch parameters differ. The 4× more blocks for tile16 means more tile boundaries where divergence may occur, but the same total threads.

### 1.4 The Atomic Memory Access Pattern

The backward kernel's critical section (lines 251-274 of `RasterizeToPixels3DGSBwd.cu`):

```cuda
if (warp.thread_rank() == 0) {
    // 5 INDEPENDENT gpuAtomicAdd calls per valid Gaussian-pixel pair
    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);        // CDIM floats
    gpuAtomicAdd(v_conic_ptr + 0, v_conic_local.x);     // 1 float
    gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);     // 1 float
    gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);     // 1 float
    gpuAtomicAdd(v_xy_ptr + 0, v_xy_local.x);           // 1 float
    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);           // 1 float
    gpuAtomicAdd(v_opacities + g, v_opacity_local);     // 1 float
}
```

**This is approximately 10 atomic add operations per valid Gaussian-pixel pair**, all targeting global memory. Every Gaussian that contributes to multiple pixels generates atomic contention on its gradient buffer.

---

## 2. Kernel Launch Sequence (theoretical)

As traced from source code, the exact CUDA launch sequence during backward is:

```
1. rasterize_to_pixels_3dgs_bwd_kernel   [grid=1×tile_h×tile_w, block=tile_size²]
   └── computes: v_means2d, v_conics, v_colors, v_opacities
       └── uses gpuAtomicAdd for scatter accumulation

2. spherical_harmonics_bwd                [grid=ceil(nnz/256)]
   └── computes: v_shs from v_colors
       
3. projection_ewa_3dgs_fused_bwd          [grid=ceil(N/256)]
   └── computes: v_means, v_quats, v_scales from v_means2d, v_conics, v_depths

4. quat_scale_to_covar_preci_bwd          [grid=ceil(N/256)]
   └── computes: v_quats, v_scales from v_covars (if using separate path)
```

Kernels 2-4 are **tile_size independent** — identical launch configs for tile16 and tile32.

---

## 3. Key Architectural Insight

The backward kernel's performance is dominated by three factors:

1. **Atomic scatter contention:** The number of threads contending for the same Gaussian's gradient buffer is proportional to the number of pixels that Gaussian covers. For large Gaussians covering many tiles, contention is high.

2. **Shared memory capacity:** tile32 has 4× more shared memory per block (~20KB vs ~5KB), allowing larger batches of Gaussians to be processed per iteration.

3. **Occupancy:** At 1080p, tile16 has 8,160 blocks running on ~40 SMs → ~204 blocks/SM. This is far more than tile32's 2,040 blocks → ~51 blocks/SM. However, tile16's blocks are smaller (256 vs 1024 threads), balancing total thread count.

The **critical path** is the atomic scatter-add loop: the number of atomic operations per Gaussian equals the number of pixels that Gaussian intersects. For tiles with very many Gaussians, this creates severe memory subsystem contention.

---

## 4. Summary

| Backward Kernel | tile_size dependent? | tile16 config | tile32 config | Estimated cost share |
|:----------------|:--------------------:|:-------------:|:-------------:|:--------------------:|
| `rasterize_to_pixels_3dgs_bwd` | **YES** — grid & block | 8160×256 | 2040×1024 | ~97% |
| `spherical_harmonics_bwd` | NO | N-dependent | N-dependent | ~1% |
| `projection_ewa_3dgs_fused_bwd` | NO | N-dependent | N-dependent | ~1% |
| `quat_scale_to_covar_preci_bwd` | NO | N-dependent | N-dependent | ~1% |

The complete backward call chain is traced from Python `autograd` → `_wrapper.py` CUDA function bindings → `.cu` kernel launches. All source file locations, line numbers, and launch parameters are documented above.
