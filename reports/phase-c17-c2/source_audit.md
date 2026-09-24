# C17-2 Source Audit Report

> **⚠️ Deprecated — superseded by `c1_source_audit_final.md` (2026-09-05).**
> This is a Phase C17-2 predecessor report containing unverified "12 CUB passes" claims.
> The current evidence chain uses `c1_source_audit_final.md` which provides source-verified
> end_bit values (46→30) without assuming pass counts. Keep this file for historical record only.

**Date:** 2026-10-19  
**Repo Commit:** `0237503` (dirty — untracked files only)  
**Baseline:** gsplat v1.5.3 (installed at `Lib/site-packages/gsplat`)  
**C1 Applied:** ⚠️ **CORRECTION: NO** — verified installed `IntersectTile.cu` is true v1.5.3 baseline (46-bit key, 12 CUB passes). C1 exists only in `patches/IntersectTile.c1.cu` but is **not deployed**.  
**Hardware (current):** NVIDIA RTX 5070 Laptop GPU (8 GB VRAM, Blackwell)  
**Hardware (validation):** A100-PCIE-40GB (mx server, distinct PyTorch 2.4.1 + gsplat 1.5.3 build)

---

## 1. Repository State

### 1.1 Current Status
```
Commit: 0237503 — "Merge remote-tracking branch 'origin/master' into master — resolve conflicts, retain both sides"
Modified: results/epic05/eligible_modules.json, research_alignment_matrix.json, scripts/epic05/phase7/run_full.py, run_sanity.py
Untracked: 30+ temporary scripts (_tmp_*), patches/, reports/epic05/phase* reports/
```

### 1.2 C1 Deployment State
The installed `gsplat/cuda/csrc/IntersectTile.cu` **already contains C1 changes** (depth compression from 32→16 bits). The original gsplat v1.5.3 baseline is preserved at:
- `patches/gsplat_orig/IntersectTile.cu` — intermediate version (includes C1 comments but is the baseline-aligned copy)
- `patches/gsplat_orig/IntersectTile_orig_v153.cu` — true upstream v1.5.3 (no C1 comments)

### 1.3 Build Mechanism
- **JIT compilation** via `torch.utils.cpp_extension._jit_compile` (triggered by `_backend.py`)
- Source: `gsplat/cuda/csrc/*.cu` + `*.cpp` + `ext.cpp`
- `build_baseline.bat` → sets `FAST_COMPILE=1`, triggers import → rebuilds
- `build_c1.bat` → same, but after patching C1 into source
- No pre-built wheel; CUDA sources compiled at import time

---

## 2. C17-1: What Was Actually Changed

### 2.1 Summary
C17-1 (Phase 17B) was a **feasibility study only** — no CUDA implementation was deployed. The design proposed replacing the global intersection materialization + CUB radix sort with per-tile bounded queues + block-level local sorts. However, Phase 17B-0 proved correctness via Python simulation, and Phase 17B-1 established performance estimates — but **no C17-1 CUDA code was integrated into gsplat**.

### 2.2 C1 (Depth Key Compression) — Already Deployed
C1 is **separate from C17-1**. C1 (Phase 16) reduced the sort key width from 47 bits to 31 bits by truncating depth from 32 bits to its upper 16 bits:

| Aspect | Baseline (v1.5.3) | C1 (Current) |
|--------|------------------|--------------|
| Key encoding | `image_id(1) \| tile_id(14) \| depth(32)` | `image_id(1) \| tile_id(14) \| depth_upper(16)` |
| Sort range | 47 bits → 12 CUB passes | 31 bits → 8 CUB passes (33% fewer) |
| Offset kernel shift | `>> 32` | `>> 16` |
| Performance impact | — | <1% end-to-end improvement (measured) |

### 2.3 What C17-1 Proposed But Did NOT Implement

| Component | Proposed Change | Actual Status |
|-----------|----------------|---------------|
| Per-tile queues | `tile_buf[n_tiles][capacity]` replaces `isect_ids/flatten_ids` | Not implemented |
| Single-pass intersect | Direct queue write (no cumsum) | Not implemented |
| Block-level sort | Per-tile bitonic sort replaces CUB radix | Not implemented |
| Offset elimination | Buffer pointers replace offset kernel | Not implemented |
| Rasterization changes | Direct buffer reads replace `flatten_ids` indirection | Not implemented |

### 2.4 C17-1 Key Deliverables (Completed)
1. **Intersection set equivalence** — proven across 9/9 configs (34.8M intersections, zero missing/extra/duplicate)
2. **Queue capacity analysis** — max per-tile = 14,174 (bicycle t32); p99.9 < 10,358
3. **Overflow characterization** — cap=8192: 0.99% overflow (bicycle t16); cap=16384: 0.98% overflow (bicycle t32)
4. **Depth ordering equivalence** — 25,932/25,932 tiles exact match (100%)
5. **Performance model** — estimated ~1.1–1.5× speedup, at cost of 1.8× memory for uniform-cap queue

---

## 3. Current Pipeline (with C1 Applied)

### 3.1 Complete Forward Path

```
Gaussian Parameters (means, quats, scales, colors, opacities)
  │
  ▼
[1] fully_fused_projection()  ← _FullyFusedProjection.forward()
  ├── quat_scale_to_covar_preci_fwd  (quats+scales → covariances)
  ├── projection_ewa_3dgs_fused_fwd (world→camera→screen, compute means2d, conics, radii, depths)
  └── returns: (radii, means2d, depths, conics)
  │
  ▼
[2] isect_tiles()  ← @torch.no_grad()
  ├── Pass 1: intersect_tile_kernel [count] → tiles_per_gauss[nnz]
  ├── cumsum → n_isects (host sync via .item())
  ├── Allocate isect_ids[n_isects×int64] + flatten_ids[n_isects×int32]
  ├── Pass 2: intersect_tile_kernel [materialize] → isect_ids + flatten_ids
  ├── CUB DeviceRadixSort::SortPairs (C1: 31-bit key, 8 passes)
  │   └── DoubleBuffer: isect_ids↔isect_ids_sorted, flatten_ids↔flatten_ids_sorted
  └── returns: (tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted)
  │
  ▼
[3] isect_offset_encode()  ← @torch.no_grad()
  ├── intersect_offset_kernel (decode tile_id from sorted isect_ids)
  └── returns: offsets[I×tile_height×tile_width] (per-tile start/end indices into flatten_ids)
  │
  ▼
[4] rasterize_to_pixels()  ← _RasterizeToPixels.forward()
  ├── rasterize_to_pixels_3dgs_fwd_kernel
  │   └── 1 block × 1 tile (tile_size × tile_size threads)
  │   └── For each tile:
  │        ├── Read offsets[tile] → (range_start, range_end)
  │        ├── Batch-load (flatten_ids → means2d, conics, colors, opacities)
  │        ├── Alpha compositing front-to-back
  │        └── Write render_colors, render_alphas, last_ids
  └── returns: (render_colors, render_alphas)
```

### 3.2 Complete Backward Path

```
[5] _RasterizeToPixels.backward()
  ├── rasterize_to_pixels_3dgs_bwd_kernel
  │   └── 1 block × 1 tile (tile_size × tile_size threads)
  │   └── For each tile (back-to-front):
  │        ├── Read offsets[tile] → (range_start, range_end)
  │        ├── Batch-load (flatten_ids → means2d, conics, colors, opacities)
  │        ├── Gradient computation per Gaussian:
  │        │   ├── v_means2d (per-pixel → warpReduce → atomicAdd)
  │        │   ├── v_conics (atomicAdd)
  │        │   ├── v_colors (atomicAdd)
  │        │   └── v_opacities (atomicAdd)
  │        └── Returns to: _FullyFusedProjection.backward() → v_means, v_quats, v_scales
  └── Gradient flows continue through projection and SH
```

### 3.3 Data Structure Summary

| Buffer | Type | Size | Role |
|--------|------|------|------|
| `means2d` | float32[nnz,2] | Projected 2D positions (input to intersection + rasterization) |
| `radii` | int32[nnz,2] | Projected radius in pixels |
| `depths` | float32[nnz] | Z-depth per Gaussian |
| `conics` | float32[nnz,3] | Inverse 2D covariance (upper triangle) |
| `tiles_per_gauss` | int32[nnz] | Per-Gaussian tile count (Pass 1 output) |
| `cum_tiles_per_gauss` | int64[nnz] | Cumsum for indexing Pass 2 |
| `isect_ids` | int64[n_isects] | **64-bit sort key**: `image_id\|tile_id\|depth_upper16` |
| `flatten_ids` | int32[n_isects] | **Gaussian index** (GID) per intersection |
| `isect_ids_sorted` | int64[n_isects] | Double-buffer: CUB sorted output |
| `flatten_ids_sorted` | int32[n_isects] | Double-buffer: CUB sorted output |
| `tile_offsets` | int32[I,th,tw] | Per-tile start index into `flatten_ids_sorted` |
| `render_colors` | float32[I,H,W,CDIM] | Output rendered image |
| `render_alphas` | float32[I,H,W] | Output accumulated alpha |
| `last_ids` | int32[I,H,W] | Index of last contributing Gaussian per pixel |

### 3.4 Kernel Launch Configuration

| Kernel | Grid | Block | Shared Mem |
|--------|------|-------|------------|
| `intersect_tile_kernel` | `ceil(nnz/256)` | 256 | 0 |
| `intersect_offset_kernel` | `ceil(n_isects/256)` | 256 | 0 |
| `rasterize_to_pixels_3dgs_fwd_kernel` | `[I, tile_height, tile_width]` | `[tile_size, tile_size]` | `tile_size² × (4+12+12) bytes` |
| `rasterize_to_pixels_3dgs_bwd_kernel` | `[I, tile_height, tile_width]` | `[tile_size, tile_size]` | `tile_size² × (4+12+12+CDIM×4) bytes` |

---

## 4. C17-2 Optimization Target Identification

### 4.1 Current Bottleneck Analysis

**Primary bottleneck: Rasterization kernel** (50%+ of forward time per Phase 17B measurements).  
**Secondary bottleneck: Intersection materialization** (Pass 1 + Pass 2 + CUB sort ≈ 35-50% of forward time for tile16).

Baseline stage breakdown (bicycle t16, 1080p, RTX 5070):

| Stage | Time (ms) | % of Forward | Bound By |
|-------|:---------:|:------------:|----------|
| Projection | ~2.0 | 8% | Compute (covar calc) |
| Intersect Pass 1 (count) | ~2.5 | 10% | Compute + memory write |
| **Intersect Pass 2 (write)** | **~4.0** | **15%** | **Memory write (64-bit key encoding)** |
| **CUB radix sort (6.08M × 31-bit)** | **~5.5** | **21%** | **Memory bandwidth (C1 reduced from 12→8 passes)** |
| Offset kernel | ~0.8 | 3% | Compute |
| **Rasterization** | **~13.0** | **50%** | **Compute + memory (batched Gaussian reads)** |

### 4.2 C17-2 Hypothesis

**C17-2 targets the rasterization kernel exclusively**, not the intersection/sort pipeline (which is C17-1's domain).

The key observation from the rasterization forward kernel:

```
┌─────────────────────────────────────────────┐
│  rasterize_to_pixels_3dgs_fwd_kernel        │
│                                             │
│  For each batch [b=0..num_batches]:         │
│    Each thread loads 1 Gaussian from tile:   │
│      g = flatten_ids[batch_start + tr]       │  ← indirect load
│      xy = means2d[g]                         │  ← global memory
│      opac = opacities[g]                     │  ← global memory  
│      conic = conics[g]                       │  ← global memory
│    Block.sync()                              │
│    For each Gaussian [t=0..batch_size]:      │
│      If pixel inside footprint:              │
│        Compute alpha = exp(-0.5*delta*conic)  │  ← compute bound
│        If alpha > THRESHOLD && T > 1e-4:     │
│          Accumulate: pix_out += color * vis   │  ← compute
│          Update T                            │
└─────────────────────────────────────────────┘
```

**Current inefficiency**: Forward kernel **does not exploit spatial coherence of alpha saturation**.

### 4.3 The Key Insight

The forward kernel processes **all** Gaussians in a tile, from front to back, until T (transmittance) drops below `1e-4`. At that point, pixel `T < 1e-4` means the pixel is "done" — but:

1. **Different pixels in the same tile saturate at different Gaussians.** Some pixels reach `T < 1e-4` early, others late.

2. **The current kernel continues loading and checking Gaussians for saturated pixels** — it sets `done = True` per pixel but still participates in batch loads and computes alpha checks.

3. **The `__syncthreads_count(done) >= block_size` check can end a batch early**, but only when **every pixel in the tile** is saturated. This is rare for most scenes (background sky pixels never saturate, thin objects have low coverage).

### 4.4 C17-2 Optimization: Early Batch Termination

**Idea**: Track **how many pixels in the tile are still active** (T > threshold) and detect when the remaining active pixels are sparse enough to justify **switching to a different processing mode** or terminating the tile entirely.

**Three possible approaches** (need to decide which one in Design):

1. **Fine-grained batch termination**: After each batch, check if active pixel count is below a threshold. If so, either:
   - Switch to a warp-level gather (process only active pixels)
   - Exit entirely (if all pixels done)

2. **Per-pixel active mask tracking**: Maintain a 256-bit mask of which pixels are still active. When mask == 0 → exit. When mask has < N active bits → switch to warp-gather mode.

3. **Tile-level early exit with completion check**: After processing `X` batches equal to the *max* per-pixel Gaussian count in the tile (instead of processing all batches for all pixels), exit.

### 4.5 Expected Workload Reduction

| Scenario | Current Behavior | C17-2 Expected | Reduction |
|----------|-----------------|----------------|-----------|
| Tile with 746 Gaussians (mean), sky region | ~746 loads/checks per pixel | Early exit after ~200-400 | ~30-50% fewer loads |
| Tile with max 6,882 Gaussians (bicycle t16) | ~6,882 loads/checks | Early exit after ~1,000-2,000 | ~50-70% fewer loads |
| Tile with 490 Gaussians (room t16) | ~490 loads/checks | Early exit after ~200-400 | ~20-50% fewer loads |

**Key metric**: `workload = Σ_pixels min(Gaussians_in_tile, gaussian_idx_at_saturation)`

Currently this is implicitly `n_isects_per_tile × tile_size²`. After optimization, it becomes `Σ_pixels saturation_depth(pixel)`.

### 4.6 Why This Doesn't Overlap with C17-1

| Dimension | C17-1 | C17-2 |
|-----------|-------|-------|
| **Target stage** | Intersection + sort pipeline | Rasterization kernel |
| **Mechanism** | Per-tile queue + local sort | Early batch termination |
| **Workload reduced** | Global sort passes (memory bandwidth) | Per-pixel alpha compositing (compute + memory) |
| **Data structure affected** | `isect_ids`, `flatten_ids` → tile buffers | No data structure change |
| **Kernel affected** | `intersect_tile_kernel`, sort, offset | `rasterize_to_pixels_3dgs_fwd_kernel` only |
| **Memory impact** | Large (per-tile pre-allocated buffers) | Negligible (just a counter/bitset in shared memory) |
| **Backward impact** | None (sort is no_grad) | `last_ids` (used by backward) unchanged |
| **Correctness risk** | Overflow handling | Pixel values identical (same math, fewer iterations) |
| **Independence** | Replaces intersection pipeline entirely | Pure rasterization optimization inside the tile block |

**C17-1 and C17-2 address different bottlenecks** in different stages. They can be composed: if both are applied, the intersection pipeline is replaced by per-tile queues + local sorts, AND the rasterization kernel terminates batches earlier. The correctness of each is independent.

---

## 5. What Cannot Be Modified

### INVARIANT — No Changes Allowed

1. **Gradient semantics**: The autograd graph edge count and connectivity must be preserved. `_RasterizeToPixels.forward() → save_for_backward()` produces the exact same saved tensors.

2. **Output ordering assumptions**: `render_colors[I,H,W,CDIM]` and `render_alphas[I,H,W]` maintain the same pixel processing order (identical to baseline). `last_ids[I,H,W]` encodes the same final-Gaussian-per-pixel.

3. **Alpha compositing semantics**: `T = T × (1-α)`, `pix_out += color × α × T_prev`, `T < 1e-4 → done`. This alpha compositing formula is the mathematical contract.

4. **Visibility / depth ordering**: Gaussians consumed front-to-back within each tile. C17-2 does not change ordering — it only changes **how many Gaussians each pixel processes**.

5. **Tensor shape contracts**: `means2d[nnz,2]`, `conics[nnz,3]`, `colors[nnz,CDIM]`, `opacities[nnz]` — unchanged shapes and semantics.

6. **Python API compatibility**: `rasterize_to_pixels()` signature unchanged. No new required arguments. Optional arguments get sensible defaults.

7. **Backward dependency chain**: `_RasterizeToPixels.backward()` reads `render_alphas, last_ids, isect_offsets, flatten_ids` — these exact tensors, with the same semantics. C17-2 does not change `last_ids` values (same "last contributing Gaussian" for each pixel).

8. **Numerical determinism**: Same floating-point operations in the same order for non-terminated pixels. No order-of-operations changes.

### What IS Modified

- Inside the forward rasterization kernel: **added early-exit logic for saturated pixels**
- Shared memory layout: may add a `uint32_t active_pixel_mask` or `uint32_t active_count`
- Loop bound: `num_batches` → dynamic termination based on active pixel count

---

## 6. Audit Summary

### Key Files and Their Roles

| File | Role | Will C17-2 modify? |
|------|------|-------------------|
| `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` | Forward rasterization kernel | **YES — primary target** |
| `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` | Backward rasterization kernel | **MAYBE — last_ids validation** |
| `gsplat/cuda/csrc/RasterizeToPixels2DGSFwd.cu` | 2DGS forward | **NO** |
| `gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu` | 2DGS backward | **NO** |
| `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu` | World-space forward | **NO (separate kernel)** |
| `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu` | World-space backward | **NO** |
| `gsplat/cuda/csrc/IntersectTile.cu` | Intersection + sort | **NO** |
| `gsplat/cuda/csrc/Intersect.cpp` | Intersection binding | **NO** |
| `gsplat/cuda/csrc/Rasterization.cpp` | Rasterization binding | **MAYBE — if new args needed** |
| `gsplat/cuda/_wrapper.py` | Python wrapper | **MAYBE — if new flag needed** |
| `gsplat/cuda/csrc/Rasterization.h` | Header signatures | **MAYBE — if new args needed** |
| `gsplat/cuda/csrc/Common.h` | Utility macros/types | **NO** |
| `gsplat/cuda/csrc/Utils.cuh` | CUDA utilities (warpSum) | **NO** |
| `gsplat/cuda/csrc/Ops.h` | Op registration | **NO** |
| `gsplat/cuda/csrc/ext.cpp` | Extension entry point | **NO** |

### Existing Test Infrastructure

| Test | Location | Use for C17-2? |
|------|----------|----------------|
| Phase 17B C1 correctness | `reports/epic05/phase16_c1_correctness.md` | Reference for test methodology |
| Phase 17B C17-1 correctness | `reports/epic05/phase17b_c17_1_correctness.md` | Reference for intersection equivalence testing |
| Phase 17B C17-1 benchmark | `reports/epic05/phase17b_c17_1_benchmark.md` | Reference for performance methodology |
| Unit tests | `tests/unit/` | Untracked — need inspection |
| Integration tests | `tests/integration/` | Untracked — need inspection |
| Phase 8B snapshot | Historical forward pattern | Reference render for correctness |
| 500-step training scripts | `scripts/epic05/phase7/run_sanity.py`, `run_full.py` | Training validation |
| 30K training scripts | `scripts/epic05/phase7/run_full.py` | Full training validation |

---

*End of Source Audit — proceed to Design phase*
