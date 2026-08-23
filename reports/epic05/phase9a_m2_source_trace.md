# Phase 9A — M2 Packed/Dense Source Trace

**Date:** 2026-09-15  
**Author:** DSH coding agent  
**Status:** COMPLETE

---

## 1. Purpose

Trace exactly what changes between `packed=True` and `packed=False` in the gsplat
`rasterization()` pipeline — not from variable names, but from the actual source code
control flow.

**Source file traced:** `gsplat/rendering.py` (lines 33–770)  
**CUDA autograd functions:** `gsplat/cuda/_wrapper.py`  
**gsplat version:** installed at `C:\Users\36570\miniconda3\Lib\site-packages\gsplat\`

---

## 2. Entry Point

`gsplat.rendering.rasterization()` is the public API. The `packed` parameter (default `True`)
is passed through to three downstream calls:

1. `fully_fused_projection(..., packed=packed, ...)` — projection
2. `isect_tiles(..., packed=packed, ...)` — tile intersection
3. `rasterize_to_pixels(..., packed=packed, ...)` — pixel rasterization

Plus two intermediate processing blocks:
- SH color computation (packed vs dense indexing)
- Meta dict metadata

---

## 3. Stage 1: Projection

### 3.1 Dense path (`packed=False`)

Dispatches to:

```python
radii, means2d, depths, conics, compensations = _FullyFusedProjection.apply(...)
```

**CUDA kernel:** `projection_ewa_3dgs_fused_fwd`  
**Output shape:** `[..., C, N, ...]` — all Gaussians × all cameras

- `radii`: `[..., C, N, 2]` — for every Gaussian-camera pair (zero if not visible)
- `means2d`: `[..., C, N, 2]`
- `depths`: `[..., C, N]`
- `conics`: `[..., C, N, 3]`
- No `batch_ids`, `camera_ids`, `gaussian_ids` returned

Dense path does **not** filter out invalid Gaussians — it keeps all `N*C` entries,
marking invalid ones with `radii=0`.

### 3.2 Packed path (`packed=True`)

Dispatches to:

```python
batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations \
    = _FullyFusedProjectionPacked.apply(...)
```

**CUDA kernel:** `projection_ewa_3dgs_fused_fwd_packed`  
**Output shape:** `[nnz, ...]` — only valid (visible) Gaussian-camera pairs

- `batch_ids`: `[nnz]` — batch index for each valid pair
- `camera_ids`: `[nnz]` — camera index for each valid pair
- `gaussian_ids`: `[nnz]` — Gaussian index for each valid pair
- `radii`: `[nnz, 2]`
- `means2d`: `[nnz, 2]`
- `depths`: `[nnz]`
- `conics`: `[nnz, 3]`

The packed CUDA kernel internally **compacts** the projection output — it only writes
entries for Gaussians that are actually visible in the camera frustum.

### 3.3 Memory Footprint — Projection Stage

| Tensor | Dense | Packed |
|--------|-------|--------|
| radii | C × N × 2 × 4B | nnz × 2 × 4B |
| means2d | C × N × 2 × 4B | nnz × 2 × 4B |
| depths | C × N × 4B | nnz × 4B |
| conics | C × N × 3 × 4B | nnz × 3 × 4B |
| **Total** | **C × N × 8 × 4B** | **nnz × 8 × 4B** |

For a real scene with N=1M Gaussians, C=1 camera, where ~250K are visible:
- Dense: ~32 MB (projection tensors)
- Packed: ~8 MB (projection tensors) + 3 × nnz × 4B (ids) ≈ ~11 MB

The compact memory is especially important for multi-camera batch rendering (C > 1).

---

## 4. Stage 1b: Opacity Extraction

### Dense:
```python
opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))
# Shape: [..., C, N]
```

### Packed:
```python
opacities = opacities.view(B, N)[batch_ids, gaussian_ids]  # [nnz]
```
Uses fancy indexing from the compacted ids.

---

## 5. Stage 2: SH Color Computation

### 5.1 Packed path

```python
# means for valid pairs only
dirs = means.view(B, N, 3)[batch_ids, gaussian_ids] - campos.view(B, C, 3)[batch_ids, camera_ids]
# SH coefficients for valid pairs
shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]
# SH evaluation — only on nnz elements
colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
```

SH evaluation is computed only for `nnz` valid pairs → O(nnz) compute.

### 5.2 Dense path

```python
# directions for ALL Gaussian-camera pairs
dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
# SH evaluation on ALL pairs
shs = torch.broadcast_to(colors[..., None, :, :, :], batch_dims + (C, N, -1, 3))
colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [..., C, N, 3]
```

SH evaluation is computed for all `C × N` pairs → O(C×N) compute.
For C=1, N=1M: dense evaluates SH on 1M Gaussian-camera directions, versus ~250K for packed.

---

## 6. Stage 3: Tile Intersection

### 6.1 Common signature

```python
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d, radii, depths,
    tile_size, tile_width, tile_height,
    segmented=segmented,
    packed=packed,
    n_images=I, image_ids=image_ids, gaussian_ids=gaussian_ids,
)
```

### 6.2 Dense path

- Input shapes: `means2d [..., C, N, 2]`, `radii [..., C, N, 2]`, `depths [..., C, N]`
- CUDA kernel iterates over all `C × N` Gaussian-camera pairs
- For each pair with `radii > 0`, computes tile coverage and writes intersection record
- Total candidate elements = C × N (but most have radii=0 for invisible pairs)

### 6.3 Packed path

- Input shapes: `means2d [nnz, 2]`, `radii [nnz, 2]`, `depths [nnz]`
- CUDA kernel iterates over nnz elements only
- All nnz elements are valid by definition (compact from projection)
- Total candidate elements = nnz (typically 25–50% of C×N)

### 6.4 Sorting Workload

Both paths perform CUB radix sort on the intersection ids. Input size for sort:
- Dense: all tile-Gaussian intersections of visible pairs (≈ tiles_per_gauss × visible_pairs)
- Packed: same but fewer input pairs

For a real scene where each visible Gaussian intersects ~50 tiles on average:
- Dense sort input ≈ visible_pairs × 50
- Packed sort input ≈ nnz × 50

Since nnz ≈ visible_pairs, the **sort input size scales with the number of visible
pairs**, which is the same for both paths. However, the packed path has already paid
the compaction cost in the projection stage, while the dense path still carries all
C×N elements through the intersection test.

---

## 7. Stage 4: Rasterization

### 7.1 Common function

```python
render_colors, render_alphas = rasterize_to_pixels(
    means2d, conics, colors, opacities,
    width, height, tile_size,
    isect_offsets, flatten_ids,
    backgrounds=backgrounds,
    packed=packed,
    absgrad=absgrad,
)
```

### 7.2 Dense path

- `means2d [..., C, N, 2]`, `conics [..., C, N, 3]`, `colors [..., C, N, 3]`, `opacities [..., C, N]`
- CUDA kernel reads these dense tensors; tiles read Gaussians from their camera's
  full array, using `flatten_ids` to index into `[camera * N + gaussian_idx]`

### 7.3 Packed path

- `means2d [nnz, 2]`, `conics [nnz, 3]`, `colors [nnz, 3]`, `opacities [nnz]`
- CUDA kernel reads compacted tensors; tiles use `flatten_ids` to index into `[0..nnz)`

### 7.4 Rasterization Kernel Difference

The rasterization kernel `rasterize_to_pixels_3dgs_fwd` is the **same compiled binary**
in both cases. The difference is only in how the input pointers are set up:
- Dense: stride = C × N in the flattened index space
- Packed: stride = nnz in the packed index space

Per-pixel loop count is identical (same number of Gaussian contributions per pixel
since the same Gaussians are visible either way for a given camera).

---

## 8. Stage 5: Backward

### 8.1 Projection backward

**Dense:** `_FullyFusedProjection.backward()` → CUDA kernel `projection_ewa_3dgs_fused_bwd`
- Computes gradients for all C×N entries
- Gradients for invisible pairs are zero

**Packed:** `_FullyFusedProjectionPacked.backward()` → CUDA kernel `projection_ewa_3dgs_fused_bwd_packed`
- Computes gradients only for nnz valid entries
- Gradient tensor shapes are compacted

### 8.2 Rasterization backward

Both paths use the same kernel `rasterize_to_pixels_3dgs_bwd`.
Input differences mirror the forward:
- Dense: `[..., C, N, ...]` gradients
- Packed: `[nnz, ...]` gradients (scatter back via `gaussian_ids`)

---

## 9. Summary of What Changes

| Aspect | Dense (`packed=False`) | Packed (`packed=True`) |
|--------|----------------------|----------------------|
| **Projection kernel** | `projection_ewa_3dgs_fused_fwd` | `projection_ewa_3dgs_fused_fwd_packed` |
| **Projection output** | `[C, N, ...]` full tensor | `[nnz, ...]` compacted |
| **Valid pair count** | C × N (most invalid) | nnz (all valid) |
| **Memory (projection)** | ~8 × C × N × 4B | ~8 × nnz × 4B + 3 × nnz × 4B |
| **SH compute** | C × N SH evaluations | nnz SH evaluations |
| **Isect tiles input** | C × N pairs → O(C×N) tests | nnz pairs → O(nnz) tests |
| **Sort input** | Intersections of visible pairs | Same as dense (same Gaussians visible) |
| **Rasterize kernel** | Same binary, stride = C×N | Same binary, stride = nnz |
| **Projection backward** | `_fused_bwd` (dense grads) | `_fused_bwd_packed` (compacted grads) |
| **Rasterize backward** | Same kernel, dense inputs | Same kernel, packed inputs |

### Critical Insight

The packed/dense difference is **not** about sorting workload (sort input is the
same number of tile-Gaussian intersections in both cases). The real differences are:

1. **Compaction overhead:** Packed mode pays the cost of an extra compaction step
   in the forward projection kernel to eliminate invalid pairs.

2. **Memory bandwidth:** Packed mode reduces memory traffic for SH computation,
   tile intersection test, and rasterization by keeping only valid pairs.

3. **Memory footprint:** Packed mode uses less GPU memory for intermediate tensors,
   which can matter for large scenes with many cameras.

4. **Indexing overhead:** Packed requires batch/camera/gaussian id maps for scatter/gather.

5. **SH compute reduction:** For C=1, if 25% of Gaussians are visible, packed mode
   evaluates SH on 1/4 the pairs.

For a **single-camera render** where most Gaussians are visible (close-up scene),
packed and dense are nearly identical because nnz ≈ visible fraction of N → both
process similar numbers of Gaussian-camera pairs.
