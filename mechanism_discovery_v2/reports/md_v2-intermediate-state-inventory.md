# Intermediate State Inventory — gsplat 3DGS Rasterization Pipeline

> **Source**: gsplat 1.5.3 (`gsplat/cuda/csrc/*.cu`) + reference\_v1 trainer  
> **Objective**: Catalog every tensor/intermediate produced by the forward pass that the backward pass or training loop consumes, and identify which intermediates are unnecessary after densification ends.

---

## 1. Pipeline Stage Overview

The rasterization pipeline in `rasterization()` (`rendering.py`) orchestrates these stages in order:

```
fully_fused_projection_fwd  ──►  spherical_harmonics  ──►  isect_tiles (2-pass)  ──►  isect_offset_encode  ──►  rasterize_to_pixels_fwd
 (CUDA kernel)                     (CUDA kernel)            (CUDA kernel)               (CUDA kernel)                (CUDA kernel)
```

The backward pass runs the same stages in reverse, consuming saved forward intermediates:

```
loss.backward()
  └─► rasterize_to_pixels_bwd   ──►  fully_fused_projection_bwd   ──► spherical_harmonics_bwd
      (CUDA kernel)                     (CUDA kernel)                    (CUDA kernel)
```

---

## 2. Intermediate Tensor Table

**Notation**:
- `C` = number of cameras (always 1 in reference\_v1 trainer)
- `N` = number of 3D Gaussians (grows from ~10K → ~200K during training)
- `nnz` = number of **visible** Gaussian instances after frustum culling (packed mode only; reference\_v1 uses `packed=False`, so all tensors are `[C, N, ...]`)
- `K` = SH coefficient count = `(max_sh_degree + 1)²` (max 16 for degree 3)
- `H, W` = image height, width
- `T` = tile count = `ceil(W/16) × ceil(H/16)`
- `M` = intersection count (tile-Gaussian pairs, up to ~100× `nnz`)

| # | Tensor | Shape | dtype | Approx bytes (N=100K, C=1) | Created in (kernel/function) | Consumed in (kernel/function) | Lifetime | Required only for training? | Required only during densification? | Required after densification ends? |
|---|--------|-------|-------|----------------------------|-----------------------------|-----------------------------|----------|----------------------------|-------------------------------------|-------------------------------------|
| 1 | **means2d** | `[C, N, 2]` | float32 | 0.8 MB | `fully_fused_projection_fwd` | `isect_tiles`, `rasterize_to_pixels_fwd`, `rasterize_to_pixels_bwd`, `fully_fused_projection_bwd` | FWD+BWD | Yes (gradients to means) | No | **Yes** — consumed by backward every iteration |
| 2 | **means2d.absgrad** | `[C, N, 2]` | float32 | 0.8 MB | `rasterize_to_pixels_bwd` (attached as `.absgrad` attribute) | `add_densification_stats` in trainer | BWD-only | Yes | **Yes** — densification gradient | **No** — unused after `densify_until_iter` |
| 3 | **radii** | `[C, N]` | int32 | 0.4 MB | `fully_fused_projection_fwd` | `isect_tiles` (pass 1 & 2), trainer → visibility filter | FWD | Partially | Partially | **Partially** — still produced; visibility_filter computed but **unused** after densification |
| 4 | **depths** | `[C, N]` | float32 | 0.4 MB | `fully_fused_projection_fwd` | `isect_tiles` (pass 1 & 2), `fully_fused_projection_bwd` (z-gradient) | FWD+BWD | Yes (when depth rendering active) | No | **Yes*** — consumed in `fully_fused_projection_bwd` via `v_depths` |
| 5 | **conics** | `[C, N, 3]` | float32 | 1.2 MB | `fully_fused_projection_fwd` | `rasterize_to_pixels_fwd`, `rasterize_to_pixels_bwd`, `fully_fused_projection_bwd` | FWD+BWD | Yes (gradients to quats/scales) | No | **Yes** — consumed by backward every iteration |
| 6 | **compensations** | `[C, N]` | float32 | 0.4 MB | `fully_fused_projection_fwd` (only when `rasterize_mode="antialiased"`) | `fully_fused_projection_bwd` | FWD+BWD | Yes (antialiased only) | No | Only if antialiased mode used |
| 7 | **colors** (SH-evaluated) | `[C, N, 3]` | float32 | 1.2 MB | `spherical_harmonics` | `rasterize_to_pixels_fwd`, `rasterize_to_pixels_bwd` (via `v_colors`) | FWD+BWD | Yes (gradients to SH coeffs) | No | **Yes** — consumed by backward every iteration |
| 8 | **tiles_per_gauss** | `[C, N]` | int32 | 0.4 MB | `isect_tiles` (1st pass) | trainer instrumentation only | FWD | Yes (instrumentation) | Yes (instrumentation) | **No** — only consumed by C53 instrumentation at checkpoints |
| 9 | **isect_ids** | `[M]` | int64 | ~3.2 MB (M=400K) | `isect_tiles` (2nd pass) | `isect_offset_encode` | FWD | Yes | No | **Yes** — required to produce offsets for rasterization |
| 10 | **flatten_ids** | `[M]` | int32 | ~1.6 MB (M=400K) | `isect_tiles` (2nd pass) | `rasterize_to_pixels_fwd`, `rasterize_to_pixels_bwd` | FWD+BWD | Yes | No | **Yes** — consumed in both forward and backward rasterization |
| 11 | **isect_offsets** | `[C, T_y, T_x]` | int32 | ~32 KB (T=800) | `isect_offset_encode` | `rasterize_to_pixels_fwd`, `rasterize_to_pixels_bwd` | FWD+BWD | Yes | No | **Yes** — consumed in both forward and backward rasterization |
| 12 | **render_alphas** | `[C, H, W, 1]` | float32 | ~0.6 MB (H=800, W=800) | `rasterize_to_pixels_fwd` | `rasterize_to_pixels_bwd`, trainer → loss (SSIM) | FWD+BWD | Yes | No | **Yes** — consumed by backward and loss computation |
| 13 | **last_ids** | `[C, H, W]` | int32 | ~2.5 MB (H=800, W=800) | `rasterize_to_pixels_fwd` | `rasterize_to_pixels_bwd` | FWD→BWD | Yes | No | **Yes** — consumed by backward to bound the walk-back per pixel |
| 14 | **opacities** (activated) | `[C, N]` | float32 | 0.4 MB | `rendering.py` (sigmoid + broadcast) | `rasterize_to_pixels_fwd`, `rasterize_to_pixels_bwd` | FWD+BWD | Yes (gradients to raw opacities) | No | **Yes** — consumed by backward every iteration |
| 15 | **camera_ids** | `[nnz]` | int64 | ~0 (packed=False, =None) | `fully_fused_projection_packed_fwd` | `isect_tiles` (packed mode only) | FWD | N/A | N/A | Not applicable (packed=False in ref. trainer) |
| 16 | **gaussian_ids** | `[nnz]` | int64 | ~0 (packed=False, =None) | `fully_fused_projection_packed_fwd` | `isect_tiles` (packed mode only) | FWD | N/A | N/A | Not applicable (packed=False in ref. trainer) |

> \* `depths` is consumed by `fully_fused_projection_bwd` only when `v_depths` is non-zero, which requires `render_mode` that includes depth. In pure RGB mode, `v_depths` is zero and the backward still reads `depths` but the gradient contribution is zero. The kernel always passes `v_depths` to `fully_fused_projection_bwd` and always reads from the `depths` tensor pointer even if the gradient is zero — so the memory cannot be freed.

---

## 3. Additional Convolved Intermediates (internal to CUDA kernels)

These exist only in registers/shared memory within the kernel and are **never materialized as global-memory tensors**:

| Intermediate | Kernel | Description | Materialized? |
|---|---|---|---|
| `R` (rotation matrix) | `proj_fwd`, `proj_bwd` | 3×3 rotation from `viewmats` | No — computed from pointers |
| `covar` (3D covariance) | `proj_fwd`, `proj_bwd` | 3×3 world-space covariance | No — computed from quats/scales or loaded from covars |
| `covar_c` (camera covariance) | `proj_fwd`, `proj_bwd` | 3D covariance in camera space | No — `R @ covar @ R^T` |
| `covar2d` (2D projection) | `proj_fwd`, `proj_bwd` | 2×2 projected covariance | No — Jacobian projection |
| `covar2d_inv` (conic) | `proj_fwd`, `proj_bwd` | Inverse 2×2 covariance (never fully materialized as 2×2; written as conics 3-vector) | Conics is the written output |
| `sigma` (per-pixel) | `rasterize_fwd`, `bwd` | Scalar: `½ Δ^T · conic · Δ` | No — computed inline in warp |
| `alpha` (per-pixel) | `rasterize_fwd`, `bwd` | Scalar: `opacity · exp(-sigma)` | No — computed inline |
| `T` (transmittance) | `rasterize_fwd`, `bwd` | Scalar running transmittance | No — register variable per thread |
| `pix_out` | `rasterize_fwd` | Accumulated per-pixel color | No — register array per thread |
| SH bases | `spherical_harmonics` | Evaluated SH basis functions | No — kernel-internal |

---

## 4. Zero-Work Analysis (After Densification Ends)

The reference trainer densifies from `densify_from_iter=500` to `densify_until_iter=15,000`. After iteration 15,000:

### 4.1 Intermediates that become **zero-work candidates** (no consumer)

| Intermediate | Why zero-work? | Impact | Removal priority |
|---|---|---|---|
| **means2d.absgrad** (item #2) | Only consumed by `add_densification_stats()` which is gated by `iteration < densify_until_iter`. The CUDA backward always computes it when `absgrad=True`, but the Python consumer doesn't read it. | Saves ~0.8 MB N=100K; more importantly, avoids `absgrad` rounding from the backward kernel. | **HIGH** — set `absgrad=False` after densification ends |
| **radii** → visibility_filter (item #3 derived) | `visibility_filter = (radii > 0).any(dim=-1)` is computed every iteration, but the only consumers (`max_radii2D` update, `add_densification_stats`) are gated by `iteration < densify_until_iter`. | Saves ~0.4 MB read. The radii tensor is still needed by `isect_tiles` and thus cannot be eliminated. The post-compute visibility_filter is wasteful but negligible. | **LOW** — minimal compute, difficult to restructure |
| **tiles_per_gauss** (item #8) | Only consumed by C53 instrumentation (at checkpoint iterations). Not consumed by any backward kernel. In `isect_tiles` it is produced as an intermediate then stored. | Saves ~0.4 MB. But gsplat already materializes it; the instrumentation is an *additional* read. | **MEDIUM** — if not instrumenting, don't read it |

### 4.2 Intermediates that remain **necessary** after densification

| Intermediate | Why still needed? |
|---|---|
| **means2d** (item #1) | Consumed by `isect_tiles`, both `rasterize_to_pixels_fwd` and `bwd`, and `fully_fused_projection_bwd`. The backward pass produces gradients for `means` (xyz positions). |
| **conics** (item #5) | Consumed by both rasterization forward and backward kernels. The backward pass produces gradients for `quats` and `scales`. |
| **colors** / SH-evaluated (item #7) | Consumed by both rasterization forward and backward kernels. The backward pass produces gradients for SH coefficients. |
| **opacities** (item #14) | Consumed by both rasterization forward and backward kernels. The backward pass produces gradients for raw `_opacity` logits. |
| **flatten_ids** (item #10) | Consumed by both rasterization forward and backward kernels to index into Gaussian data by tile intersection order. |
| **isect_offsets** (item #11) | Consumed by both rasterization forward and backward kernels as tile range pointers. |
| **render_alphas** (item #12) | Consumed by `rasterize_to_pixels_bwd` and by the loss computation (SSIM needs alpha). |
| **last_ids** (item #13) | Consumed by `rasterize_to_pixels_bwd` to bound the per-pixel reverse traversal through Gaussians. |
| **isect_ids** (item #9) | Produced only within the forward pass for `isect_offset_encode`; lives only until offsets are written. |

### 4.3 Intermediates that are **scalars or constants** (always cheap)

- `tile_size`, `tile_width`, `tile_height`, `width`, `height` (metadata) — trivial.
- `n_cameras` — trivial.

---

## 5. Answer to the Key Question

> **"Are we producing/storing/writing intermediates during phases where no later computation consumes them?"**

**Yes, two clear instances:**

### Instance A: `absgrad` after densification ends (iteration ≥ `densify_until_iter`)

In the reference trainer, `absgrad=True` is hard-coded in the `rasterization()` call (line 125 of `trainer.py`):

```python
absgrad=True,  # Required: makes means2d require grad for view-space gradient
```

The comment says "Required" but it is **only** required during the densification phase (≤15,000 iter). After that, `add_densification_stats()` is never called. However, the CUDA backward kernel in `rasterize_to_pixels_bwd.cu` (lines 232–234, 267–271) **always** computes `v_means2d_abs` when `absgrad=True`:

```cpp
if (v_means2d_abs != nullptr) {
    v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
}
// ...
if (v_means2d_abs != nullptr) {
    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
}
```

This produces a full `[C, N, 2]` float32 tensor of absolute gradients that is **never read** by the Python training loop after densification ends. The `torch.zeros_like(means2d)` allocation (line 349 of `rasterize_to_pixels_bwd.cu` call site) and the atomic-add writes are pure waste.

**Fix**: After iteration ≥ `densify_until_iter`, switch to `absgrad=False`. This both:
- Saves the `v_means2d_abs` allocation (~0.8 MB at N=100K)
- Removes the `abs()` computation and atomic-add instructions in the backward kernel

### Instance B: `tiles_per_gauss` outside of instrumentation checkpoints

`tiles_per_gauss` is produced by the first pass of `isect_tiles` and returned in `meta`. It is consumed **only** by the C53 instrumentation at specific checkpoint iterations (e.g., 1,000, 10,000, 20,000, 30,000). At all other iterations, this tensor is allocated, written, and then immediately discarded. The `isect_tiles` kernel always runs both passes (because the second pass needs `cum_tiles_per_gauss` from the first pass), so the first pass output is an intrinsic cost of the intersection computation — it cannot be skipped. But **reading** it back into Python and storing it in `meta` is optional.

**Fix**: Don't store `tiles_per_gauss` in `meta` unless instrumentation is active at the current iteration. This is a trivial change in `rendering.py` or the trainer.

### Instance C: `visibility_filter` after densification ends (negligible)

The trainer computes `visibility_filter = (radii > 0).any(dim=-1)` every iteration (line 336), but after `densify_until_iter` the result is never used. This is a single elementwise comparison (~2 CUDA warps for N=100K) — negligible relative to the rasterization kernel.

---

## 6. Detailed Tensor Lifecycle Diagrams

### 6.1 Means2d lifecycle

```
fully_fused_projection_fwd     isect_tiles        rasterize_to_pixels_fwd
        │                           │                      │
        ├─── means2d [C,N,2] ──────►│◄──── reads ──────────┤
        │                           │                      │
        │                    rasterize_to_pixels_bwd       │
        │                           │                      │
        │◄─── v_means2d ◄─── reads ─┤◄─── reads ──────────┤
        │                 ┌─────────┘                     │
        │                 ▼                                │
        │      fully_fused_projection_bwd                  │
        │              │                                   │
        │              └──► v_means, v_quats, v_scales     │
        │                                                  │
        │ (only if densification active)                   │
        │      trainer: add_densification_stats            │
        │              │                                   │
        │              └──► xyz_gradient_accum, denom      │
        │                                                  │
        │ (via .absgrad attribute set by rasterize_to_pixels_bwd)
```

### 6.2 Radii lifecycle

```
fully_fused_projection_fwd     isect_tiles (1st pass)
        │                           │
        ├─── radii [C,N] (int32) ──►│ (tile count computation)
        │                           │
        │                    isect_tiles (2nd pass)
        │                           │
        │◄──────────────────────────│ (read for tile range encoding)
        │
        │    trainer: visibility_filter = (radii > 0).any(dim=-1)
        │         │
        │         ├── if < densify_until_iter:
        │         │      max_radii2D.update(visibility_filter)
        │         │      add_densification_stats(means2d, visibility_filter)
        │         │
        │         └── if ≥ densify_until_iter:
        │                visibility_filter is DEAD (no consumer)
```

### 6.3 Intersection structure lifecycle (isect_ids / flatten_ids / isect_offsets)

```
isect_tiles (2-pass) ──► isect_ids [M] ──► isect_offset_encode ──► isect_offsets [C, T]
                           │                                               │
                           └── (used only by isect_offset_encode)         │
                                                                           │
                    rasterize_to_pixels_fwd          rasterize_to_pixels_bwd
                           │                                  │
                    ◄──── reads isect_offsets ────────── reads ◄────
                    ◄──── reads flatten_ids  ────────── reads ◄────
```

---

## 7. Summary of Optimization Opportunities

| Opportunity | Tensor | Savings | Complexity | Risk |
|---|---|---|---|---|
| **Disable absgrad** after densification | means2d.absgrad | ~0.8 MB + atomic-add in BWD | Low — one flag change in trainer | None — consumer gated by `densify_until_iter` |
| **Skip tiles_per_gauss read** when not instrumenting | tiles_per_gauss | ~0.4 MB + Python overhead | Low — conditional read in trainer | None — pure instrumentation artifact |
| **Skip visibility_filter** after densification | radii (derived) | Negligible | Medium — requires restructuring training loop | Low — cosmetic change |
| **Use packed=True** with camera_ids/gaussian_ids | Many tensors | Large — `[C,N,...] → [nnz,...]` sparsity win | High — changes all indexing in trainer | Medium — requires adapting densification code |

### Priority recommendation

1. ✅ **Immediate**: Set `absgrad=False` after densification ends. Single-line change, confirmed waste-free.
2. ✅ **Immediate**: Avoid reading `tiles_per_gauss` into Python dict outside instrumented iterations.
3. ⏳ **Future**: Consider `packed=True` for post-densification phase to exploit the fact that ~50% of Gaussians may be invisible per view (nnz << N).

---

## 8. Appendix: Kernel-by-Kernel Input/Output Map

### `fully_fused_projection_fwd` (projection CUDA kernel)
| Inputs | Outputs |
|---|---|
| `means` [N,3] | **radii** [C,N] (int32) |
| `quats` [N,4] | **means2d** [C,N,2] |
| `scales` [N,3] | **depths** [C,N] |
| `viewmats` [C,4,4] | **conics** [C,N,3] |
| `Ks` [C,3,3] | **compensations** [C,N] (optional) |
| `eps2d`, `near_plane`, `far_plane`, `radius_clip` | |

### `spherical_harmonics` (SH evaluation)
| Inputs | Outputs |
|---|---|
| `sh_degree` | **colors** [C,N,3] |
| `dirs` [C,N,3] | |
| `coeffs` [C,N,K,3] | |
| `masks` [C,N] | |

### `isect_tiles` (tile intersection, 2-pass)
| Inputs (pass 1) | Output (pass 1) |
|---|---|
| `means2d`, `radii`, `depths` | **tiles_per_gauss** [C,N] |
| `tile_size`, `tile_width`, `tile_height` | (used for cumsum to size pass 2) |

| Inputs (pass 2) | Outputs (pass 2) |
|---|---|
| `means2d`, `radii`, `depths` | **isect_ids** [M] (int64, encoded) |
| `cum_tiles_per_gauss` | **flatten_ids** [M] (int32) |
| Plus optional camera_ids/gaussian_ids (packed mode) | |

### `isect_offset_encode`
| Inputs | Output |
|---|---|
| `isect_ids` [M] | **isect_offsets** [C, tile_height, tile_width] |

### `rasterize_to_pixels_fwd`
| Inputs | Outputs |
|---|---|
| `means2d` [C,N,2] | **render_colors** [C,H,W,ch] |
| `conics` [C,N,3] | **render_alphas** [C,H,W,1] |
| `colors` [C,N,ch] | **last_ids** [C,H,W] (int32) |
| `opacities` [C,N] | |
| `isect_offsets` [C,Ty,Tx] | |
| `flatten_ids` [M] | |
| `width`, `height`, `tile_size` | |

### `rasterize_to_pixels_bwd`
| Inputs (fwd data) | Inputs (grad outputs) | Outputs (grad inputs) |
|---|---|---|
| `means2d` [C,N,2] | `v_render_colors` [C,H,W,ch] | **v_means2d** [C,N,2] |
| `conics` [C,N,3] | `v_render_alphas` [C,H,W,1] | **v_means2d_abs** [C,N,2] (if absgrad) |
| `colors` [C,N,ch] | | **v_conics** [C,N,3] |
| `opacities` [C,N] | | **v_colors** [C,N,ch] |
| `render_alphas` [C,H,W,1] | | **v_opacities** [C,N] |
| `last_ids` [C,H,W] | | |
| `isect_offsets`, `flatten_ids` | | |

### `fully_fused_projection_bwd`
| Inputs (fwd data) | Inputs (grad outputs) | Outputs (grad inputs) |
|---|---|---|
| `means` [N,3] | `v_means2d` [C,N,2] | **v_means** [N,3] |
| `quats/scales` or `covars` | `v_depths` [C,N] | **v_quats** [N,4] / **v_scales** [N,3] |
| `viewmats` [C,4,4], `Ks` [C,3,3] | `v_conics` [C,N,3] | **v_viewmats** [C,4,4] (optional) |
| `radii` [C,N] (for visibility mask) | `v_compensations` [C,N] (optional) | |
| `conics` [C,N,3] | | |
| `compensations` [C,N] (optional) | | |

---

## 9. Conclusion

The pipeline has **2 clear zero-work opportunities** after densification ends (iteration ≥ 15,000):

1. **`absgrad=False`** — saves ~0.8 MB tensor allocation and atomic-add operations in the backward kernel. The `.absgrad` tensor is produced but never consumed.

2. **`tiles_per_gauss` not read** — trivial save of ~0.4 MB and CPU-side Python overhead outside instrumented iterations.

All other intermediates (`means2d`, `conics`, `colors`, `opacities`, `depths`, `flatten_ids`, `isect_offsets`, `render_alphas`, `last_ids`) are consumed by the backward pass every iteration and cannot be eliminated.
