# Phase 15 — Renderer Source Map

> System-level trace: Python API → C++ binding → CUDA kernel for the gsplat differentiable rasterizer (packed mode).

---

## 1. Call Chain Overview (Forward, Packed Mode)

```
rendering.py:rasterization()
  ├── fully_fused_projection()           # _wrapper.py:fully_fused_projection (packed=True)
  │   └── _FullyFusedProjectionPacked.apply()
  │       └── projection_ewa_3dgs_packed_fwd_kernel    # ProjectionEWA3DGSPacked.cu
  │           Output: batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics
  ├── spherical_harmonics()             # _wrapper.py:spherical_harmonics (if sh_degree set)
  │   └── _SphericalHarmonics.apply()
  │       └── sh_coeffs_to_color_fast kernel          # SphericalHarmonicsCUDA.cu
  │           Output: colors [nnz, 3]
  ├── isect_tiles()                      # _wrapper.py:isect_tiles
  │   ├── [First pass] intersect_tile_kernel (count)  # IntersectTile.cu:24
  │   │   Output: tiles_per_gauss [nnz]
  │   ├── at::cumsum()                                 # Intersect.cpp:79
  │   │   Output: cum_tiles_per_gauss [nnz], n_isects
  │   ├── [Second pass] intersect_tile_kernel (fill)   # IntersectTile.cu:24
  │   │   Output: isect_ids [n_isects], flatten_ids [n_isects]
  │   └── radix_sort_double_buffer()                   # IntersectTile.cu:296
  │       └── cub::DeviceRadixSort::SortPairs
  │           Output: isect_ids_sorted, flatten_ids_sorted
  ├── isect_offset_encode()             # _wrapper.py:isect_offset_encode
  │   └── intersect_offset_kernel                      # IntersectTile.cu:209
  │       Output: isect_offsets [I, tile_h, tile_w]
  └── rasterize_to_pixels()              # _wrapper.py:rasterize_to_pixels
      └── _RasterizeToPixels.apply()
          └── rasterize_to_pixels_3dgs_fwd_kernel      # RasterizeToPixels3DGSFwd.cu:17
              Output: render_colors, render_alphas, last_ids
```

---

## 2. Call Chain Overview (Backward)

```
loss.backward()
  ├── _RasterizeToPixels.backward()
  │   └── rasterize_to_pixels_3dgs_bwd_kernel          # RasterizeToPixels3DGSBwd.cu
  │       Output: v_means2d, v_conics, v_colors, v_opacities
  │       (Reads: means2d, conics, colors, opacities, tile_offsets,
  │        flatten_ids, render_alphas, last_ids, v_render_colors, v_render_alphas)
  ├── (if SH) _SphericalHarmonics.backward()
  │   └── sh_coeffs_to_color_fast_vjp kernel           # SphericalHarmonicsCUDA.cu
  │       Output: v_coeffs, v_dirs
  ├── _FullyFusedProjectionPacked.backward()
  │   └── projection_ewa_3dgs_packed_bwd_kernel        # ProjectionEWA3DGSPacked.cu
  │       (OR projection_ewa_3dgs_fused_bwd_kernel if dense)
  │       Output: v_means, v_quats, v_scales, v_covars
  │       (Recomputes geometry: covar from quats/scales, transforms, projection VJP)
```

---

## 3. Key Source File Locations

| File | Path | Purpose |
|------|------|---------|
| `rendering.py` | `gsplat/rendering.py` | Python orchestration: calls all stages in order |
| `_wrapper.py` | `gsplat/cuda/_wrapper.py` | Autograd Functions + tensor shape validation |
| `_backend.py` | `gsplat/cuda/_backend.py` | Lazy CUDA C extension loader |
| `Intersect.cpp` | `gsplat/cuda/csrc/Intersect.cpp` | C++ binding: host-side launch orchestration for intersect+sort |
| `IntersectTile.cu` | `gsplat/cuda/csrc/IntersectTile.cu` | CUDA kernels: intersect_tile, intersect_offset, radix_sort, segmented_sort |
| `ProjectionEWA3DGSFused.cu` | `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` | Fused forward+backward projection (dense mode) |
| `ProjectionEWA3DGSPacked.cu` | `gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu` | Packed projection with block-level prefix sum |
| `RasterizeToPixels3DGSFwd.cu` | `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` | Forward rasterization: tile-based alpha compositing |
| `RasterizeToPixels3DGSBwd.cu` | `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` | Backward rasterization: VJP with atomicAdd |
| `SphericalHarmonicsCUDA.cu` | `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu` | SH evaluation forward + VJP |
| `Common.h` | `gsplat/cuda/csrc/Common.h` | Shared types, constants (ALPHA_THRESHOLD), camera models |
| `Rasterization.h` | `gsplat/cuda/csrc/Rasterization.h` | Rasterization kernel launch declarations |
| `Intersect.h` | `gsplat/cuda/csrc/Intersect.h` | Intersect + sort kernel declarations |

---

## 4. Data Tensor Lifecycle (Packed Mode)

```
                      B*N gaussians
                           │
                    Projection (fused)
                    ┌──────┴──────┐
                    │              │
               radii[nnz]    means2d[nnz,2]
                depths[nnz]   conics[nnz,3]
                    │              │
                    │       ┌──────┘
                    │       │
               isect_tiles (first pass) → tiles_per_gauss[nnz]
                    │       │
                    │    cumsum → cum_tiles_per_gauss[nnz], n_isects
                    │       │
                    │    isect_tiles (second pass)
                    │    ┌──┴──┐
                    │    │     │
                    │ isect_ids    flatten_ids
                    │ [n_isects]   [n_isects]
                    │    │           │
                    │  RADIX SORT ──┘
                    │    │
                    │ isect_ids_sorted, flatten_ids_sorted
                    │    │
                    │ offset_encode → isect_offsets[I, th, tw]
                    │    │
                    └────┤
                         │
                    rasterize_to_pixels_fwd
                         │
                    rendered colors, alpha
```

---

## 5. Key Observations

### 5.1 The n_isects data explosion

The intersection materialization (S4) is the most volume-expanding operation in the pipeline:
- Input: `nnz` gaussians (typical: 100K–3M visible gaussians)
- Each Gaussian covers `tile_count` tiles: `(ceil(radius_x/tile_size)*2+1) × (ceil(radius_y/tile_size)*2+1)`
- For tile_size=16: each large Gaussian can cover 4+ tiles → n_isects ~4-6× nnz
- For tile_size=16 at 1080p: tile_width=68, tile_height=38, n_tiles=2584

### 5.2 The sort is the bottleneck

- `radix_sort_double_buffer()` sorts all n_isects by 64-bit keys (image_id | tile_id | depth)
- CUB's DeviceRadixSort::SortPairs requires ~12 passes for 48-bit key range
- Each pass reads/writes the full n_isects × (8+4) bytes
- Temporary storage allocation via CUB_WRAPPER goes through PyTorch caching allocator

### 5.3 Two-pass intersect design

The two-pass design in Intersect.cpp avoids allocating worst-case intersection storage:
1. First pass: count tiles per gaussian (lightweight)
2. CPU-side cumsum + `n_isects = cumsum[-1].item<int64_t>()` — **forces a host-device synchronization**
3. Allocate exact-size tensors
4. Second pass: write intersections

The `.item<int64_t>()` call at line 80 of Intersect.cpp is a **synchronization point** that blocks the CPU until the first-pass kernel completes.

### 5.4 Offset encode re-scans

`intersect_offset_kernel` runs with n_isects threads and re-reads the entire sorted `isect_ids` array to produce tile offsets. This is an O(n_isects) pass that reads the same data the sort just wrote.

### 5.5 Shared memory in rasterization

Only the rasterization kernels (both fwd and bwd) use shared memory — for batched gaussian data. The projection and intersection kernels use zero shared memory.

### 5.6 Atomic accumulation in backward

`rasterize_to_pixels_3dgs_bwd_kernel` uses `gpuAtomicAdd` for gradient writes. Multiple pixel threads within a tile may contend on the same gaussian ID (especially for large Gaussians that span many pixels).

---

## 6. CUDA Stream Behavior

All kernels launch on `at::cuda::getCurrentCUDAStream()` — the default stream. No explicit CUDA events or stream synchronization beyond the default stream ordering. This means:
- All kernels serialize on the default stream
- No overlap between kernels is possible
- Memory allocation/deallocation (temp buffers) also serialize on this stream
