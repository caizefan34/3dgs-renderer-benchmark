# R2.1 Profile-Only Patch Notes — Shared Raster Attribute Gating

## 1. Patch Status

| Property | Value |
|----------|-------|
| **Status** | `NOT_APPLIED` to canonical baseline — profiling-only patch for gsplat-1.5.3 |
| **Git status** | 0 modified tracked files (clean tree at commit `84f29bb`) |
| **Repository** | `/tmp/gsplat_baseline/gsplat-1.5.3` on remote A100 server `mx` |
| **Target** | gsplat 1.5.3 source tree (NOT local gsplat 1.4.0) |
| **Purpose** | Profile-only measurement of per-branch gradient gating — NOT a candidate implementation |
| **Source files** | `r21_patch.py` (patch script), `r21_build.py` (build script) |

## 2. Files Modified by the Patch

The patch (`experiments/r2_profile_only/r21_patch.py`) modifies 4 files in the remote gsplat-1.5.3 source tree:

| File in gsplat-1.5.3 | Original backup | Changes |
|----------------------|-----------------|---------|
| `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` | `*.orig_r21` | Kernel signature, shared memory, mask loading, gradient computation, warp reduction, atomicAdd |
| `gsplat/cuda/csrc/Rasterization.cpp` | `*.orig_r21` | C++ launch function — add 3 mask params to kernel dispatch |
| `gsplat/cuda/_wrapper.py` | `*.orig_r21` | Python autograd `_RasterizeToPixels` — add mask tensors to `save_for_backward`, pass to C++ |
| `gsplat/rendering.py` | `*.orig_r21` | `rasterization()` top-level function — add 3 mask params with default `None` |

## 3. Kernel Signature Changes

Three new mask parameters added to `rasterize_to_pixels_3dgs_bwd_kernel`:

```cuda
const uint8_t *__restrict__ r2_geo_mask,       // [N] or [nnz]
const uint8_t *__restrict__ r2_app_mask,        // [N] or [nnz]
const uint8_t *__restrict__ r2_opacity_mask,    // [N] or [nnz]
```

When all three are non-null, **per-branch mode** is enabled. When null, the original C51 single-mask behavior is preserved.

## 4. Shared Memory Layout (per-kernel)

```
rgbs_batch[block_size * CDIM]     — Gaussian colors (unchanged)
mask_batch[block_size]             — legacy importance mask (unchanged)
app_mask_batch[block_size]         — R2.1 app mask (NEW)
geo_mask_batch[block_size]         — R2.1 geo mask (NEW)
opacity_mask_batch[block_size]     — R2.1 opacity mask (NEW)
```

## 5. Which Operations Remain Unconditional

The following operations are **always executed** (not gated by any mask):

| Operation | Why | Source (in modified kernel) |
|-----------|-----|-----------------------------|
| **Traversal** | Loading flatten_ids, iterating per-pixel Gaussian list | Required for all gradient modes |
| **T recompute** (`ra = 1.0/(1.0 - alpha)`; `T *= ra`) | Required for downstream pixel composition | Always executed if `valid` |
| **Buffer update** (`buffer[k] += rgb[k] * fac`) | Required for T/buffer correctness of later Gaussians | Always executed if `valid` |
| **delta computation** (px-py offset) | Required for geometry gradient | Always computed |
| **sigma recompute** | Required for alpha recompute | Always computed |
| **alpha recompute** (`alpha = min(0.999, opac * exp(-sigma))`) | Required for everything | Always computed |
| **vis computation** (`vis = alpha * T_before`) | Required for opacity gradient and buffer | Always computed |

Reminder: traversal includes reconstructing `tile_offsets` and `flatten_ids` from saved state, plus the per-pixel front-to-back Gaussian list walk. These are part of the existing canonical backward kernel and are not saved/cached by the patch.

## 6. Which Operations Are Gated by Which Mask

| Gradient output | Gate condition | Computed from |
|----------------|---------------|--------------|
| `v_colors` (v_rgb) | `do_app = valid && app_mask_batch[t]` | `fac * v_render_c[k]` — no v_alpha needed |
| `v_opacities` | `do_opacity = valid && opacity_mask_batch[t]` | `vis * v_alpha` — v_alpha needed |
| `v_means2d` (v_xy) | `do_geo = valid && geo_mask_batch[t]` | `-opac * vis * v_alpha * (conic matrix) * delta` — v_alpha needed |
| `v_conics` | `do_geo_full = valid && geo_mask_batch[t]` | Derived from v_sigma — v_alpha needed |
| `v_means2d_abs` | Same as v_means2d | `abs(v_xy_local)` — only computed when v_means2d is |

**Key**: `v_alpha` is an intermediate that is only computed when at least one of `do_opacity` or `do_geo` is true. If only `do_app` is true, v_alpha is completely skipped.

## 7. Conditionals Within the Geometry Branch

```
do_geo_full (geo_mask):  full gradient = v_xy + v_conic + v_xy_abs
do_geo (without full):   B2 densify-only = only v_xy + v_xy_abs
```

The denisfy-only path (`else if (do_geo && !do_geo_full && ...)`) preserves the legacy C51 B2 behavior where only `v_means2d` gradient is computed for densification.

## 8. AtomicAdd Pattern (per-branch)

```cuda
if (app_mask_batch[t])   → atomicAdd into v_colors[g]
if (opacity_mask_batch[t]) → atomicAdd into v_opacities[g]
if (geo_mask_batch[t])   → atomicAdd into v_conics[g] + v_means2d[g] + v_means2d_abs[g]
```

Each branch's atomicAdd is independent. When combining this with the legacy C51 `importance_mask`, the more restrictive of the two masks is effectively applied per branch.

## 9. Implementation Notes

- The patch was deployed as a **Python script** (`r21_patch.py`) that performs string replacement on the C++/CUDA/Python sources, then `r21_build.py` compiles and re-links the CUDA kernel.
- **`r21_patch.py` is not a standard diff/patch** — it is a procedural script that performs find-and-replace on the exact string content of gsplat-1.5.3 files. It expects the specific C51-patched version of the gsplat code.
- The build uses a gcc-10 wrapper to compile the modified CUDA kernel, as the remote server lacks the exact compiler expected by PyTorch 2.7.1+cu118.
- The R2.1 patch was developed as **profiling-only** — it was never tested for training correctness or numerical equivalence to the canonical baseline. Its purpose was to measure per-branch gradient computation time in isolation.
