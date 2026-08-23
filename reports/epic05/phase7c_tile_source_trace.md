# Phase 7C — B1: Tile Size Source-Level Trace

**Date:** 2026-09-02  
**Purpose:** Complete trace of `tile_size` through the gsplat codebase: Python API → C++ binding → CUDA kernel launch geometry.

---

## 1. Entry Point: `gsplat.rasterization()`

**File:** `gsplat/rendering.py` (lines 33–770)  
**Function:** `rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, ..., tile_size=16, ...)`

### Parameter signature

```python
tile_size: int = 16
```

Documented as: *"The size of the tiles for rasterization. Default is 16. (Note: other values are not tested)"*

### Role
`tile_size` is used at **two distinct stages** within this function:

---

## 2. Stage A — Tile Grid Geometry Computation

**File:** `gsplat/rendering.py`, lines 632–633

```python
tile_width = math.ceil(width / float(tile_size))
tile_height = math.ceil(height / float(tile_size))
```

### Compute
For 1080p (1920×1080):

| tile_size | tile_width | tile_height | Total tiles |
|:---------:|:----------:|:-----------:|:-----------:|
| 16        | 120        | 68          | 8,160       |
| 32        | 60         | 34          | 2,040       |

### Output
- `tile_width`, `tile_height` → passed to `isect_tiles()`, `isect_offset_encode()`, and `rasterize_to_pixels()`
- Stored in `meta` dict: `meta["tile_width"]`, `meta["tile_height"]`, `meta["tile_size"]`

---

## 3. Stage B — `isect_tiles()`: Gaussian-to-Tile Association

**File:** `gsplat/cuda/_wrapper.py`, lines 443–517  
**Function:** `isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height, ...)`

### Inputs
| Parameter | Source | Description |
|:----------|:-------|:------------|
| `means2d` | From `fully_fused_projection` | Projected 2D Gaussian centers [N, 2] or [nnz, 2] |
| `radii` | From `fully_fused_projection` | Maximum screen-space radii [N, 2] or [nnz, 2] |
| `depths` | From `fully_fused_projection` | Z-depths [N] or [nnz] |
| `tile_size` | Forwarded from `rasterization()` | Tile dimension in pixels |
| `tile_width` | Computed from `ceil(width / tile_size)` | Number of tile columns |
| `tile_height` | Computed from `ceil(height / tile_size)` | Number of tile rows |

### CUDA Kernel Called
```python
tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("intersect_tile")(
    means2d, radii, depths, image_ids, gaussian_ids, I,
    tile_size, tile_width, tile_height, sort, segmented,
)
```

**C++ function:** `intersect_tile` in `gsplat/cuda/csrc/intersect_tile.cu`

### What this kernel does
1. For each projected Gaussian, compute which tiles it overlaps based on `means2d ± radii`
2. Generate one `isect_id` per tile intersection (64-bit: image_id | tile_id | depth)
3. Sort intersections by `isect_id` (radix sort)
4. Return:
   - `tiles_per_gauss`: [nnz] — how many tiles each Gaussian hits
   - `isect_ids`: [n_isects] — sorted tile identifiers
   - `flatten_ids`: [n_isects] — Gaussian index for each intersection

### tile_size effect on this stage
- **Larger tile_size → fewer tiles → each Gaussian overlaps fewer tiles (on average)**
- **Smaller tile_size → more tiles → finer-grained culling, more intersections per Gaussian**
- At 1080p: tile16 has 8,160 tiles, tile32 has 2,040 tiles → **4× fewer tiles**

---

## 4. Stage C — `isect_offset_encode()`: Tile Offset Encoding

**File:** `gsplat/cuda/_wrapper.py`, lines 520–540  
**Function:** `isect_offset_encode(isect_ids, n_images, tile_width, tile_height)`

### CUDA Kernel Called
```python
_make_lazy_cuda_func("intersect_offset")(isect_ids, n_images, tile_width, tile_height)
```

**C++ function:** `intersect_offset` in `gsplat/cuda/csrc/intersect_tile.cu`

### What this kernel does
1. Scan sorted `isect_ids` and compute prefix-sum offsets per tile
2. Output: `isect_offsets` of shape `[I, tile_height, tile_width]`
   - `I` = number of images (B×C)
   - For each tile, the offset into `flatten_ids` where this tile's Gaussians begin

### tile_size effect on this stage
- **Offsets array size = I × tile_height × tile_width**
- tile16: `I × 68 × 120` = I × 8,160 entries
- tile32: `I × 34 × 60` = I × 2,040 entries
- **4× fewer offset entries** for tile32 → less memory and faster prefix-sum

---

## 5. Stage D — `rasterize_to_pixels()`: Final Rasterization

**File:** `gsplat/cuda/_wrapper.py`, lines 543–680  
**Function:** `rasterize_to_pixels(means2d, conics, colors, opacities, image_width, image_height, tile_size, isect_offsets, flatten_ids, ...)`

### CUDA Kernel Called
```python
_make_lazy_cuda_func("rasterize_to_pixels")(
    means2d, conics, colors, opacities,
    image_width, image_height, tile_size,
    isect_offsets, flatten_ids,
    backgrounds, packed, absgrad,
)
```

**C++ function:** `rasterize_to_pixels` in `gsplat/cuda/csrc/rasterize_to_pixels.cu`

### CUDA Kernel Launch Geometry

The CUDA kernel `rasterize_to_pixels_3dgs_fwd_kernel<SH, T>` is launched with:

```cpp
// In rasterize_to_pixels.cu (conceptual — exact launch from C++ wrapper)
dim3 grid(tile_width, tile_height, I);  // One block per tile per image
dim3 block(tile_size, tile_size);       // One thread per pixel
```

### Verified via cuobjdump

Previous analysis (Phase 5, Phase 7B) confirmed:

| Property | tile16 | tile32 |
|:---------|:------:|:------:|
| Kernel name | `rasterize_to_pixels_3dgs_fwd_kernel<SH=3, float>` | Same binary |
| REG count | 40 | 40 |
| SHARED | 1024 B | 1024 B |
| STACK | 0 | 0 |
| Binary identity | Same `.cubin` | Same `.cubin` |

**Conclusion from cuobjdump:** Same compiled kernel binary. All differences are runtime launch parameters only.

### Launch configuration

For `B=1, C=1, width=1920, height=1080`:

| Parameter | tile16 | tile32 |
|:----------|:------:|:------:|
| grid.x (tile_width) | 120 | 60 |
| grid.y (tile_height) | 68 | 34 |
| grid.z (images) | 1 | 1 |
| **Total blocks** | **8,160** | **2,040** |
| block.x (tile_size) | 16 | 32 |
| block.y (tile_size) | 16 | 32 |
| block.z | 1 | 1 |
| **Threads/block** | **256** | **1,024** |
| **Warps/block** | **8** | **32** |
| **Pixels/block** | **256** | **1,024** |

### Which kernel template?

For standard (non-2DGS) RGB rendering, the actual kernel launched is:
```
rasterize_to_pixels_3dgs_fwd_kernel<SH=3, float>
```

### tile_size effect on this stage

1. **Grid size:** tile32 → 2,040 blocks (vs 8,160 for tile16) — **4× fewer blocks**
2. **Block size:** tile32 → 1,024 threads/block (vs 256 for tile16) — **4× more threads/block**
3. **Warps/block:** tile32 → 32 warps (vs 8 for tile16)
4. **Workload distribution:** tile32 packs 4× more pixels into each block, giving each block more work to hide latency

### Backward kernel

The backward kernel is:
```
rasterize_to_pixels_3dgs_bwd_kernel<SH=3, float>
```

REG count (from cuobjdump): 48 (slightly more than forward's 40).  
Same launch geometry rules apply — tile_size controls grid/block dimensions identically.

---

## 6. `isect_tiles` Kernel — Intersection Stage Effect

The `intersect_tile` CUDA kernel also uses tile_size:

```cpp
// In intersect_tile.cu (conceptual)
// Each thread works on one Gaussian
// Compute tile bounds from means2d - radii to means2d + radii
// Quantize to tile coordinates using tile_size
int tile_x_min = max(0, (int)((mean2d.x - radius) / tile_size));
int tile_x_max = min(tile_width - 1, (int)((mean2d.x + radius) / tile_size));
// Similar for y
```

### tile_size effect on intersections
- **Larger tile_size → each Gaussian hits fewer tiles** (same screen-space radius, coarser tile grid)
- This reduces the number of intersection entries (`n_isects`) and the subsequent sort work

For a Gaussian with screen-space radius R pixels:
- tile16: overlaps `ceil(2R/16)` tiles in each dimension
- tile32: overlaps `ceil(2R/32)` tiles in each dimension
- **Expected ~2× fewer tile intersections per Gaussian** for tile32

---

## 7. Full Pipeline Flow with tile_size

```
Python: rasterization()  [rendering.py:33]
  │
  ├─ fully_fused_projection()  [rendering.py:409]  ← tile_size NOT used here
  │   └─ CUDA: fully_fused_projection_kernel  [projection kernels]  
  │      (2D projection, covariance, radii computation — tile_size irrelevant)
  │
  ├─ tile_width = ceil(width / tile_size)  [rendering.py:632]  ← FIRST tile_size use
  ├─ tile_height = ceil(height / tile_size)  [rendering.py:633]  ← FIRST tile_size use
  │
  ├─ isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)  [rendering.py:634]
  │   └─ CUDA: intersect_tile  ← reads tile_size, tile_width, tile_height
  │      (maps each Gaussian to overlapping tiles; per-Gaussian tile count decreases with larger tile_size)
  │
  ├─ isect_offset_encode(isect_ids, I, tile_width, tile_height)  [rendering.py:648]
  │   └─ CUDA: intersect_offset  ← reads tile_width, tile_height
  │      (prefix-sum to build per-tile Gaussian lists; 4× fewer tiles for tile32)
  │
  └─ rasterize_to_pixels(..., tile_size, isect_offsets, flatten_ids)  [rendering.py:703]
      └─ CUDA: rasterize_to_pixels_3dgs_fwd_kernel<SH=3, float>
         (block=(tile_size, tile_size), grid=(tile_width, tile_height, I))
         (each block rasterizes one tile's Gaussians to its tile pixels)
```

---

## 8. tile_size NOT used in

| Stage | File | Why |
|:------|:-----|:----|
| `fully_fused_projection` | `_wrapper.py:288` | Projection is tile-size-agnostic |
| `spherical_harmonics` | `_wrapper.py` | SH evaluation is per-Gaussian, independent of tiling |
| `quat_scale_to_covar_preci` | `_wrapper.py` | Covariance computation is 3D, pre-projection |
| Adam optimizer | Python-level | Unchanged regardless of tile_size |
| Densification/Pruning | Python-level | Same topology logic |
| Data loading | Python-level | Unchanged |

---

## 9. Summary: All Code Paths Affected by tile_size

| Component | File | Line(s) | What changes |
|:----------|:-----|:-------:|:-------------|
| Tile grid dims | `rendering.py` | 632-633 | `tile_width = ceil(W/ts)`, `tile_height = ceil(H/ts)` |
| Intersection kernel | `_wrapper.py` → `intersect_tile` CUDA | 504 | Tile quantization radius: `R/ts` |
| Tile offset encode | `_wrapper.py` → `intersect_offset` CUDA | 538 | Array shape `[I, tile_height, tile_width]` |
| Rasterize kernel grid | CUDA launch | — | `grid(tile_width, tile_height, I)` |
| Rasterize kernel block | CUDA launch | — | `block(tile_size, tile_size)` |
| Backward kernel grid | CUDA launch | — | Same as forward: `grid(tile_width, tile_height, I)` |
| Backward kernel block | CUDA launch | — | Same as forward: `block(tile_size, tile_size)` |

**Total change:** tile16 → tile32 changes exactly **4 launch parameters** (grid.x, grid.y, block.x, block.y) and **2 derived values** (tile_width, tile_height) which affect 3 CUDA kernel invocations. No other code path is affected.

---

*Trace compiled from gsplat v1.5+ source (installed at `C:\Users\36570\miniconda3\Lib\site-packages\gsplat`).*
