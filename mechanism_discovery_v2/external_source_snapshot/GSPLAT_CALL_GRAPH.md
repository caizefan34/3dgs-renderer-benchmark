# gsplat 1.4.0 Rasterization Pipeline Call Graph

> **Version note**: The installed package is `gsplat==1.4.0`. The reference baseline
> (`baseline/reference_v1/`) was designed for `gsplat==1.5.3` semantics. The topology of
> the call graph is identical between these versions; the source files below are from the
> **installed 1.4.0** package at:
> `C:\Users\36570\miniconda3\Lib\site-packages\gsplat`

Each call path is written as:

```
training_function → Python wrapper → autograd Function → CUDA/C++ binding → CUDA kernel
```

with exact file paths and function names.

---

## 1. Entry Point — Trainer → Public `rasterization()` API

```
baseline/reference_v1/trainer.py
  render_with_meta()  [line 100]
    ↓
gsplat.rendering  (imported from gsplat.__init__ as `rasterization`)
  rasterization()  [rendering.py:28-583]
```
The trainer calls `rasterization()` at line 109 with:
- `packed=False` (dense [C, N, ...] tensors, not sparse)
- `sh_degree` set (enables SH evaluation path)
- `absgrad=True`

---

## 2. Projection Path (Forward)

```
rendering.py::rasterization  [line 297]
  ↓
cuda/_wrapper.py::fully_fused_projection  [line 185]
  ↓
cuda/_wrapper.py::_FullyFusedProjection  [line 757]
    torch.autograd.Function
  ↓ forward() [line 761]
  ↓
_make_lazy_cuda_func("fully_fused_projection_fwd")
  ↓
cuda/_backend.py
  → import:  gsplat.csrc._C
  (JIT-compiled or pre-built pybind11 module)
  ↓
cuda/csrc/ext.cpp  [line 28-33]
  m.def("fully_fused_projection_fwd", &gsplat::fully_fused_projection_fwd_tensor)
  ↓
cuda/csrc/fully_fused_projection_fwd.cu
  fully_fused_projection_fwd_tensor()  [line 196]
    → allocates radii [C,N], means2d [C,N,2], depths [C,N], conics [C,N,3]
    ↓
  FULLY_FUSED_PROJECTION_FWD_KERNEL<float>  [line 20-194]
    __global__ void  |  grid: (C*N + 255)/256  |  blocks: 1D
    Per Gaussian per Camera (each thread = 1 Gaussian × 1 camera view)

    Inside the kernel:
    1. pos_world_to_cam()      — apply viewmat rotation+translation (helpers.cuh)
    2. quat_scale_to_covar_preci() — build 3×3 world covariance from quat+scale (utils.cuh)
    3. covar_world_to_cam()    — R * covar * R^T (helpers.cuh)
    4. persp_proj() / ortho_proj() / fisheye_proj() — project to pixel (utils.cuh)
    5. add_blur()              — add eps2d to eigenvalues (utils.cuh)
    6. inverse() of 2D covar   — compute conic matrix (utils.cuh)
    7. radius = ceil(3 * sqrt(v1))  — 3-sigma screen-space radius
    8. near/far plane clip + image-boundary clip → radius=0 skips
    9. Write: radii, means2d, depths, conics, compensations
```

### Sub‑path: Projection Backward

```
_FullyFusedProjection.backward()  [line 814]
  ↓
_make_lazy_cuda_func("fully_fused_projection_bwd")
  ↓
cuda/csrc/ext.cpp  [line 31-33]
  m.def("fully_fused_projection_bwd", &gsplat::fully_fused_projection_bwd_tensor)
  ↓
cuda/csrc/fully_fused_projection_bwd.cu
  fully_fused_projection_bwd_tensor()
  ↓
FULLY_FUSED_PROJECTION_BWD_KERNEL<float>  __global__
    → computes v_means, v_covars/v_quats/v_scales, v_viewmats
```

---

## 3. Spherical Harmonics Evaluation Path (Forward)

**(Only when `sh_degree` is not None — the trainer always sets it.)**

```
rendering.py::rasterization  [line 380]
    → dirs = means - camtoworld_center  (view direction per Gaussian)
    → colors = spherical_harmonics(sh_degree, dirs, shs, masks=radii>0)
  ↓
cuda/_wrapper.py::spherical_harmonics  [line 29-55]
  ↓
cuda/_wrapper.py::_SphericalHarmonics  [line 1208]
    torch.autograd.Function
  ↓ forward() [line 1212]
  ↓
_make_lazy_cuda_func("compute_sh_fwd")
  ↓
cuda/csrc/ext.cpp  [line 10]
  m.def("compute_sh_fwd", &gsplat::compute_sh_fwd_tensor)
  ↓
cuda/csrc/compute_sh_fwd.cu
  compute_sh_fwd_tensor()  [line 40]
  ↓
COMPUTE_SH_FWD_KERNEL<float>  [line 12-38]
    __global__ void  |  grid: (N*3 + 255)/256
    Per color-channel per Gaussian (each thread = 1 channel of 1 Gaussian)
    Calls sh_coeffs_to_color_fast()  — inline function in spherical_harmonics.cuh
    Evaluates SH basis functions up to degree `degrees_to_use`
```

### Sub‑path: SH Backward

```
_SphericalHarmonics.backward()  [line 1222]
  ↓
_make_lazy_cuda_func("compute_sh_bwd")
  ↓
cuda/csrc/ext.cpp  [line 11]
  m.def("compute_sh_bwd", &gsplat::compute_sh_bwd_tensor)
  ↓
cuda/csrc/compute_sh_bwd.cu
  compute_sh_bwd_tensor()
  ↓
COMPUTE_SH_BWD_KERNEL<float>  __global__
    → computes v_coeffs and (optionally) v_dirs
```

---

## 4. Tile Intersection and Sorting Path

```
rendering.py::rasterization  [lines 497-510]
    → tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(...)
    → isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
  ↓
cuda/_wrapper.py::isect_tiles  [line 324]
    @torch.no_grad()
  ↓
_make_lazy_cuda_func("isect_tiles")
  ↓
cuda/csrc/ext.cpp  [line 35]
  m.def("isect_tiles", &gsplat::isect_tiles_tensor)
  ↓
cuda/csrc/isect_tiles.cu
  isect_tiles_tensor()  [line ~350]
    → Allocates: tiles_per_gauss, isect_ids, flatten_ids
    ↓ === TWO-PASS ALGORITHM ===

    PASS 1:
    ISECT_TILES<T>  __global__
      first_pass=true
      Per Gaussian per Camera — counts how many tiles this Gaussian touches
      → tile_min/tile_max from means2d/radius in tile-space
      → writes tiles_per_gauss[idx]

    → Exclusive Prefix Sum on CPU via GSPLAT_CUB_WRAPPER (cub::DeviceScan::ExclusiveSum)
    → Produces cum_tiles_per_gauss  (offsets into output arrays)

    PASS 2:
    ISECT_TILES<T>  __global__
      first_pass=false
      For each tile intersected, packs a 64-bit key:
        camera_id (top bits) | tile_id (middle 32 bits) | depth_bits (bottom 32 bits)
      Also writes flatten_ids = global flat index in [C*N] or [nnz]

    → SORT: CUB DeviceRadixSort::SortPairs
      Sorts (isect_ids, flatten_ids) by 64-bit isect_id key
      → Result: Gaussians sorted by camera → tile → depth (front-to-back within each tile)

  ↓
cuda/_wrapper.py::isect_offset_encode  [line 399]
    @torch.no_grad()
  ↓
_make_lazy_cuda_func("isect_offset_encode")
  ↓
cuda/csrc/ext.cpp  [line 36]
  m.def("isect_offset_encode", &gsplat::isect_offset_encode_tensor)
  ↓
cuda/csrc/isect_tiles.cu
  isect_offset_encode_tensor()
    → Compacts sorted isect_ids into per-tile start offsets
    → output: [C, tile_height, tile_width] int32 — index into flatten_ids per tile
```

---

## 5. Rasterization to Pixels (Forward)

```
rendering.py::rasterization  [line 558]
  ↓
cuda/_wrapper.py::rasterize_to_pixels  [line 418]
  ↓
cuda/_wrapper.py::_RasterizeToPixels  [line 883]
    torch.autograd.Function
  ↓ forward() [line 887]
  ↓
_make_lazy_cuda_func("rasterize_to_pixels_fwd")
  ↓
cuda/csrc/ext.cpp  [line 38]
  m.def("rasterize_to_pixels_fwd", &gsplat::rasterize_to_pixels_fwd_tensor)
  ↓
cuda/csrc/rasterize_to_pixels_fwd.cu
  rasterize_to_pixels_fwd_tensor()  [line 291]
    → reads channels = colors.size(-1)
    → dispatches via switch(channels) { __GS__CALL_(3) ... }
  ↓
CALL_KERNEL_WITH_DIM<CDIM>()  [line 189]
    → allocates renders [C, H, W, CDIM], alphas [C, H, W, 1], last_ids [C, H, W]
    ↓
RASTERIZE_TO_PIXELS_FWD_KERNEL<CDIM, float>  [line 17-186]
    __global__ void
    grid: dim3(C, tile_height, tile_width)
    block: dim3(tile_size, tile_size, 1)   — e.g. 16×16 = 256 threads

    Each block covers ONE tile. Each thread covers ONE pixel.

    tile_size × tile_size shared memory:
      id_batch[256]     — int32_t: flatten Gaussian IDs
      xy_opacity_batch  — vec3<float>: mean2d.x, mean2d.y, opacity
      conic_batch       — vec3<float>: conic[0], conic[1], conic[2]

    Algorithm (tile-cooperative, front-to-back):
    for each batch of 256 Gaussians in this tile's sorted list:
        1. Sync → each thread loads 1 Gaussian into shared memory
        2. Sync → each thread iterates over the batch
        3. For each Gaussian covering this pixel:
           delta = pixel_center - mean2d
           sigma = 0.5 * (conic.x * dx^2 + conic.z * dy^2) + conic.y * dx * dy
           alpha = min(0.999, opacity * exp(-sigma))
           if alpha < 1/255: skip
           vis = alpha * transmittance
           pixel_color += color * vis
           transmittance *= (1 - alpha)
           if transmittance ≤ 1e-4: mark `done`
        4. Early exit if whole block is done

    → writes render_colors, render_alphas, last_ids
```

---

## 6. Backward Pass — `rasterize_to_pixels_bwd`

```
_RasterizeToPixels.backward()  [line 940]
  ↓
_make_lazy_cuda_func("rasterize_to_pixels_bwd")
  ↓
cuda/csrc/ext.cpp  [line 39]
  m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor)
  ↓
cuda/csrc/rasterize_to_pixels_bwd.cu
  rasterize_to_pixels_bwd_tensor()
  ↓
RASTERIZE_TO_PIXELS_BWD_KERNEL<CDIM, float>  __global__
    Same grid/block layout as forward.
    Re-traverses each tile's Gaussian list using stored last_ids + render_alphas.
    Computes:
      v_means2d      — gradient w.r.t. projected 2D means
      v_means2d_abs  — absolute gradient (for AbsGS, if absgrad=True)
      v_conics       — gradient w.r.t. conic matrix
      v_colors       — gradient w.r.t. input colors
      v_opacities    — gradient w.r.t. opacities
```

---

## 7. Complete Forward Call Chain (Summary Diagram)

```
trainer.py::render_with_meta()
  │
  ├─→ gsplat.rendering::rasterization()                    [rendering.py:28]
  │     │
  │     ├─→ fully_fused_projection()                        [_wrapper.py:185]
  │     │     └─→ _FullyFusedProjection.forward()            [_wrapper.py:761]
  │     │           └─→ fully_fused_projection_fwd_tensor()  [fully_fused_projection_fwd.cu:196]
  │     │                 └─→ fully_fused_projection_fwd_kernel<> (CUDA kernel)
  │     │
  │     ├─→ spherical_harmonics()                            [_wrapper.py:29]
  │     │     └─→ _SphericalHarmonics.forward()              [_wrapper.py:1212]
  │     │           └─→ compute_sh_fwd_tensor()              [compute_sh_fwd.cu:40]
  │     │                 └─→ compute_sh_fwd_kernel<> (CUDA kernel)
  │     │
  │     ├─→ isect_tiles()              (@torch.no_grad)      [_wrapper.py:324]
  │     │     └─→ isect_tiles_tensor()  [isect_tiles.cu]
  │     │           ├─ isect_tiles<> (kernel Pass 1: count)
  │     │           ├─ cub::DeviceScan::ExclusiveSum (prefix sum)
  │     │           ├─ isect_tiles<> (kernel Pass 2: encode)
  │     │           └─ cub::DeviceRadixSort::SortPairs (sort)
  │     │
  │     ├─→ isect_offset_encode()      (@torch.no_grad)      [_wrapper.py:399]
  │     │     └─→ isect_offset_encode_tensor()  [isect_tiles.cu]
  │     │
  │     └─→ rasterize_to_pixels()                             [_wrapper.py:418]
  │           └─→ _RasterizeToPixels.forward()                [_wrapper.py:887]
  │                 └─→ rasterize_to_pixels_fwd_tensor()      [rasterize_to_pixels_fwd.cu:291]
  │                       └─→ call_kernel_with_dim<CDIM>()
  │                             └─→ rasterize_to_pixels_fwd_kernel<CDIM,float> (CUDA kernel)
  │
  └─→ returns (render_colors, render_alphas, meta)
```

---

## 8. Backward Call Chain (Summary)

```
torch.autograd (loss.backward())
  │
  ├─→ _RasterizeToPixels.backward()                         [_wrapper.py:940]
  │     └─→ rasterize_to_pixels_bwd_tensor()                 [rasterize_to_pixels_bwd.cu]
  │           └─→ rasterize_to_pixels_bwd_kernel<CDIM,float> (CUDA kernel)
  │           Returns: v_means2d, v_conics, v_colors, v_opacities
  │
  ├─→ _SphericalHarmonics.backward()                        [_wrapper.py:1222]
  │     └─→ compute_sh_bwd_tensor()                          [compute_sh_bwd.cu]
  │           └─→ compute_sh_bwd_kernel<float> (CUDA kernel)
  │           Returns: v_coeffs, v_dirs
  │
  └─→ _FullyFusedProjection.backward()                      [_wrapper.py:814]
        └─→ fully_fused_projection_bwd_tensor()              [fully_fused_projection_bwd.cu]
              └─→ fully_fused_projection_bwd_kernel<float> (CUDA kernel)
              Returns: v_means, v_covars/v_quats/v_scales, v_viewmats
```

---

## 9. Key Utility Files

| File | Role |
|------|------|
| `cuda/csrc/helpers.cuh` | Cooperative-group warp reductions, `pos_world_to_cam()`, `covar_world_to_cam()` |
| `cuda/csrc/utils.cuh`   | `quat_scale_to_covar_preci()`, `persp_proj()`, `ortho_proj()`, `fisheye_proj()`, `add_blur()`, `inverse()` |
| `cuda/csrc/types.cuh`   | `vec2`, `vec3`, `mat2`, `mat3` type aliases (via glm) |
| `cuda/csrc/spherical_harmonics.cuh` | `sh_coeffs_to_color_fast()` — Legendre polynomial SH evaluation |
| `cuda/csrc/bindings.h`  | All C++ tensor function declarations; `CameraModelType` enum; `GSPLAT_N_THREADS=256` |
| `cuda/csrc/ext.cpp`     | pybind11 module definition — maps Python names → C++ tensor functions |
| `cuda/_backend.py`      | JIT compilation / prebuilt-load logic → exposes `_C` |
| `cuda/_wrapper.py`      | All `torch.autograd.Function` classes + public Python API wrappers |
| `cuda/_torch_impl.py`   | Pure-PyTorch reference implementations (no CUDA needed, uses nerfacc) |

---

## 10. Notebook-style view of `fully_fused_projection_fwd_kernel`

```
fully_fused_projection_fwd_kernel<float>            (1 thread = 1 Gaussian × 1 camera)
  │
  ├─ pos_world_to_cam(R, t, mean, mean_c)           helpers.cuh
  │     mean_c = R * mean + t
  │
  ├─ if mean_c.z < near_plane || > far_plane → radii=0, return
  │
  ├─ quat_scale_to_covar_preci(quat, scale, covar)  utils.cuh
  │     R_q = quat_to_rotmat(quat)
  │     covar = R_q * diag(scale)^2 * R_q^T
  │
  ├─ covar_world_to_cam(R, covar, covar_c)           helpers.cuh
  │     covar_c = R * covar * R^T
  │
  ├─ persp_proj(mean_c, covar_c, fx,fy,cx,cy, ...)   utils.cuh
  │     J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
  │     covar2d = J * covar_c * J^T
  │     mean2d = (fx*x/z + cx, fy*y/z + cy)
  │
  ├─ add_blur(eps2d, covar2d, compensation)          utils.cuh
  │     det_orig = det(covar2d)
  │     covar2d += eps2d * I
  │     compensation = sqrt(det_orig / det(covar2d))
  │
  ├─ inverse(covar2d, conic)                         utils.cuh
  │     conic = covar^{-1}  (only upper tri stored)
  │
  ├─ radius = ceil(3*sqrt(v1))
  │     b = 0.5*(covar2d[0][0] + covar2d[1][1])
  │     v1 = b + sqrt(max(0.01, b^2 - det))
  │
  ├─ if radius <= radius_clip → radii=0, return
  ├─ if outside image → radii=0, return
  │
  └─ Write radii, means2d, depths, conics, compensations
```

## 11. Packed Mode (the `_FullyFusedProjectionPacked` variant)

The trainer uses `packed=False`. For reference, the packed variant exists:

```
_FullyFusedProjectionPacked.forward()                 [_wrapper.py:1017]
  ↓
_make_lazy_cuda_func("fully_fused_projection_packed_fwd")
  ↓
fully_fused_projection_packed_fwd_tensor()             [fully_fused_projection_packed_fwd.cu]
  ↓
fully_fused_projection_packed_fwd_kernel<>  (CUDA kernel)
  → Parallelizes over C×N with block-level prefix sums (CUB)
  → Produces packed [nnz] outputs: camera_ids, gaussian_ids, radii, means2d, depths, conics
```

---

## Source Snapshot Location

All source files traced above have been copied to:
`mechanism_discovery_v2/external_source_snapshot/gsplat_1_5_3/`

> The directory is labeled `gsplat_1_5_3` per the baseline's target version, but
> contains the **actual installed version 1.4.0** source. The call graph topology
> is identical between 1.4.0 and 1.5.3 in the paths traced above.
