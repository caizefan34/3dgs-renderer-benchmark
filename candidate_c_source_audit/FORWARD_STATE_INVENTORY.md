# FORWARD STATE INVENTORY — Candidate C Source Audit

> **Scope:** Tensors produced or consumed during the forward pass of 3DGS rasterization,
> tracked through their origin, shape, and availability at the start of the backward pass.
>
> **Baseline scene:** Room (~10K Gaussians at convergence), 1080p (1920×1080), single camera (C=1).
> The trainer at `baseline/reference_v1/trainer.py` runs with `packed=False`, `absgrad=True`.
>
> **File versions audited:**
> - `trainer.py` — reference_v1 training loop
> - `rasterize_to_pixels_fwd.cu` — forward CUDA kernel
> - `rasterize_to_pixels_bwd.cu` — backward CUDA kernel (defines what is consumed)
> - `gsplat_python/cuda/_wrapper.py` — autograd `_RasterizeToPixels` (the ctx.save_for_backward / ctx.saved_tensors contract)
> - `gsplat_python/rendering.py` — `rasterization()` orchestrator (projection → SH → tile encode → rasterize)

---

## Autograd Boundary

There are **two separate autograd Functions** in the forward path:

1. **`_FullyFusedProjection`** (rendering.py line 757–880, wrapper.py line 757–880)
   — Projects Gaussians to 2D. Saves its own `ctx` for its own backward.
2. **`_RasterizeToPixels`** (wrapper.py lines 883–1010)
   — Rasterizes per-pixel. Saves its own `ctx` for its own backward.

The `rasterization()` orchestrator in `rendering.py` connects them:
- `fully_fused_projection()` (line 297) — autograd `_FullyFusedProjection.apply()`.
- `isect_tiles()` (line 497) — `@torch.no_grad()` — **no autograd**, executed eagerly (detached).
- `isect_offset_encode()` (line 510) — `@torch.no_grad()` — same.
- SH evaluation `spherical_harmonics()` (line 380 or 710) — its own autograd `_SphericalHarmonics`.
- `rasterize_to_pixels()` (line 558) — autograd `_RasterizeToPixels.apply()` — **this is the forward pass whose saved state we audit**.

> **Key insight:** `isect_ids`, `tiles_per_gauss`, `tile_width`, `tile_height` are computed outside
> the autograd graph (under `@torch.no_grad`) and are **never saved for backward**. They are
> available in the `meta` dict (Python scope) but not inside `_RasterizeToPixels.backward()`.

---

## Tensor Inventory Table

### Legend

| Column | Meaning |
|--------|---------|
| **AVAILABLE_BEFORE_BACKWARD** | The tensor is available inside `_RasterizeToPixels.backward()`. "YES" = saved via `ctx.save_for_backward()`. "NO" = not saved (only in `meta` or not retained). |
| **Where created** | The stage / kernel call that produces this tensor |
| **Where consumed (bwd)** | The CUDA bwd kernel parameter list or the Python backward method |

---

### 1. `render_colors` — Rendered pixel colors

| Property | Value |
|----------|-------|
| **Shape** | `[C, H, W, D]` — `[1, 1080, 1920, 3]` |
| **Dtype** | `float32` |
| **Source file / function** | `rasterize_to_pixels_fwd.cu` kernel output, lines 232–234 (allocated as `torch::empty`) |
| **Created at** | `_RasterizeToPixels.forward`, wrapper.py line 902–916 (returned by CUDA fwd) |
| **Where consumed (bwd)** | Gradient arrives as `v_render_colors` from autograd (wrapper.py line 942). **The forward value is NOT passed to backward.** Only its gradient is consumed. |
| **Approx size (Room 10K @ 1080p)** | `1 × 1080 × 1920 × 3 × 4 B` ≈ **24.9 MB** |
| **AVAILABLE_BEFORE_BACKWARD** | **NO** — Not in `ctx.save_for_backward()` (wrapper.py lines 918–929). Only `v_render_colors` is available. |

---

### 2. `render_alphas` — Per-pixel accumulated opacity

| Property | Value |
|----------|-------|
| **Shape** | `[C, H, W, 1]` — `[1, 1080, 1920, 1]` |
| **Dtype** | `float32` |
| **Source file / function** | `rasterize_to_pixels_fwd.cu` kernel output, lines 236–238 (allocated as `torch::empty`) |
| **Created at** | `_RasterizeToPixels.forward`, wrapper.py line 902–916 (returned by CUDA fwd) |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 37**: `const S *__restrict__ render_alphas`. Used at line 106 to reconstruct `T_final = 1.0f - render_alphas[pix_id]`, and at line 210 for `v_alpha += T_final * ra * v_render_a`. |
| **Approx size (Room 10K @ 1080p)** | `1 × 1080 × 1920 × 1 × 4 B` ≈ **8.3 MB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 927: `ctx.save_for_backward(…, render_alphas, …)`. |

---

### 3. `last_ids` — Index (in flatten_ids) of last contributing Gaussian per pixel

| Property | Value |
|----------|-------|
| **Shape** | `[C, H, W]` — `[1, 1080, 1920]` |
| **Dtype** | `int32` |
| **Source file / function** | `rasterize_to_pixels_fwd.cu` kernel output, lines 240–242 (allocated as `torch::empty`) |
| **Created at** | `_RasterizeToPixels.forward`, wrapper.py line 902–916 (returned by CUDA fwd) |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 38**: `const int32_t *__restrict__ last_ids`. Used at line 111: `const int32_t bin_final = inside ? last_ids[pix_id] : 0;` to limit the backward traversal to only Gaussians that actually contributed to each pixel. |
| **Approx size (Room 10K @ 1080p)** | `1 × 1080 × 1920 × 4 B` ≈ **8.3 MB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 928: `ctx.save_for_backward(…, last_ids)`. |

---

### 4. `tile_offsets` (aliased as `isect_offsets`) — Starting index into flatten_ids per tile

| Property | Value |
|----------|-------|
| **Shape** | `[C, tile_h, tile_w]` — `[1, 68, 120]` for 1080p, tile_size=16 |
| **Dtype** | `int32` |
| **Source file / function** | `rendering.py` line 510: `isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)` — `@torch.no_grad()` CUDA call |
| **Created at** | After `isect_tiles`, before `rasterize_to_pixels()` |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 34**: `const int32_t *__restrict__ tile_offsets`. Used to compute `range_start` and `range_end` per tile (lines 87–91), exactly as in the forward kernel. |
| **Approx size (Room 10K @ 1080p)** | `1 × 68 × 120 × 4 B` ≈ **32.6 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 925: `ctx.save_for_backward(…, isect_offsets, …)`. |

---

### 5. `flatten_ids` — Global Gaussian indices (into means2d/colors/opacities) for every intersection

| Property | Value |
|----------|-------|
| **Shape** | `[n_isects]` — depends on scene; estimated **~500K–1M** for Room 10K @ 1080p |
| **Dtype** | `int32` |
| **Source file / function** | `rendering.py` line 497: `isect_tiles()` — `@torch.no_grad()` CUDA call, returns `tiles_per_gauss, isect_ids, flatten_ids` |
| **Created at** | After `fully_fused_projection`, before `rasterize_to_pixels()` |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 35**: `const int32_t *__restrict__ flatten_ids`. Used at line 140 to get the Gaussian index: `int32_t g = flatten_ids[idx];`. Serves as the indirection layer into all Gaussian attribute arrays. |
| **Approx size (Room 10K @ 1080p)** | ~500K × 4 B ≈ **2.0 MB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 926: `ctx.save_for_backward(…, flatten_ids, …)`. |

---

### 6. `means2d` — Projected 2D Gaussian centers

| Property | Value |
|----------|-------|
| **Shape** | `[C, N, 2]` (non-packed) — `[1, 10000, 2]` |
| **Dtype** | `float32` |
| **Source file / function** | `_FullyFusedProjection.forward` → CUDA `fully_fused_projection_fwd` (wrapper.py line 783–799) |
| **Created at** | `rendering.py` line 297–314: `fully_fused_projection()`, before `isect_tiles` |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 23**: `const vec2<S> *__restrict__ means2d`. Used at line 142 to compute pixel-space delta: `const vec2<S> xy = means2d[g]` and `delta = {xy_opac.x - px, xy_opac.y - py}`. |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 2 × 4 B` ≈ **80 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 919: `ctx.save_for_backward(means2d, …)`. |

---

### 7. `conics` — Inverse 2D covariance (upper-triangular)

| Property | Value |
|----------|-------|
| **Shape** | `[C, N, 3]` (non-packed) — `[1, 10000, 3]` |
| **Dtype** | `float32` |
| **Source file / function** | `_FullyFusedProjection.forward` → CUDA `fully_fused_projection_fwd` (wrapper.py line 783–799) |
| **Created at** | `rendering.py` line 297–314: `fully_fused_projection()` |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 24**: `const vec3<S> *__restrict__ conics`. Used at line 172 to recompute the 2D Gaussian PDF: `sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y`. |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 3 × 4 B` ≈ **120 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 920: `ctx.save_for_backward(conics, …)`. |

---

### 8. `colors` — Per-Gaussian RGB (post-SH-evaluation)

| Property | Value |
|----------|-------|
| **Shape** | `[C, N, D]` (non-packed) — `[1, 10000, 3]` for RGB mode |
| **Dtype** | `float32` |
| **Source file / function** | `rendering.py` lines 368–392: SH evaluation via `spherical_harmonics()` (autograd `_SphericalHarmonics`). With `sh_degree=None` (direct RGB), the colors tensor is expanded from the model's raw features. |
| **Created at** | Between `fully_fused_projection` and `rasterize_to_pixels` in `rendering.py` |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 25**: `const S *__restrict__ colors`. Used at line 148 to reload `rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k]`, and at line 206 to compute the partial derivative `v_alpha`: `(rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) * v_render_c[k]`. |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 3 × 4 B` ≈ **120 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 921: `ctx.save_for_backward(colors, …)`. |

---

### 9. `opacities` — Per-Gaussian opacity (view-adjusted)

| Property | Value |
|----------|-------|
| **Shape** | `[C, N]` (non-packed) — `[1, 10000]` |
| **Dtype** | `float32` |
| **Source file / function** | `rendering.py` line 327 (packed) or line 331 (non-packed): `opacities = opacities.repeat(C, 1)`. Then multiplied by compensations at line 335 if antialiased mode. |
| **Created at** | After `fully_fused_projection`, before `isect_tiles` |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 26**: `const S *__restrict__ opacities`. Used at line 143: `const S opac = opacities[g];` — reloaded per Gaussian to recompute `alpha = min(0.999f, opac * vis)`. |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 1 × 4 B` ≈ **40 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 922: `ctx.save_for_backward(opacities, …)`. |

---

### 10. `backgrounds` (optional) — Background color per camera

| Property | Value |
|----------|-------|
| **Shape** | `[C, D]` — `[1, 3]` or `None` |
| **Dtype** | `float32` |
| **Source file / function** | Passed through from the `rasterization()` call; `trainer.py` uses `background="black"` (line 189 of dataset setup) → `backgrounds=None` (default black is handled as zero). |
| **Created at** | In `dataset.py` / passed by caller |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 27**: `const S *__restrict__ backgrounds`. Used at line 213–219 for the background gradient contribution: `accum += backgrounds[k] * v_render_c[k]` → `v_alpha += -T_final * ra * accum`. |
| **Approx size (Room 10K @ 1080p)** | Negligible (`1 × 3 × 4 B` = 12 B, or `None`) |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 923: `ctx.save_for_backward(backgrounds, …)`. |

---

### 11. `masks` (optional) — Per-tile mask

| Property | Value |
|----------|-------|
| **Shape** | `[C, tile_h, tile_w]` or `None` |
| **Dtype** | `bool` |
| **Source file / function** | Passed through from the `rasterization()` call; not used by trainer. |
| **Created at** | Caller |
| **Where consumed (bwd)** | **`rasterize_to_pixels_bwd.cu` kernel param line 28**: `const bool *__restrict__ masks`. Checked at line 71: `if (masks != nullptr && !masks[tile_id])` → early return (no gradients). |
| **Approx size** | `1 × 68 × 120 × 1 B` ≈ **8.2 KB** when present |
| **AVAILABLE_BEFORE_BACKWARD** | **YES** — Explicitly saved at wrapper.py line 924: `ctx.save_for_backward(masks, …)`. |

---

### 12. `isect_ids` — Encoded (camera_id | tile_id | depth) per intersection

| Property | Value |
|----------|-------|
| **Shape** | `[n_isects]` — ~500K for Room 10K |
| **Dtype** | `int64` |
| **Source file / function** | `rendering.py` line 497: `isect_tiles()` (`@torch.no_grad()`) |
| **Created at** | After `fully_fused_projection`, before `isect_offset_encode` |
| **Where consumed (bwd)** | **NOT consumed by backward kernel.** Only used to produce `isect_offsets` and `flatten_ids` in forward via `@torch.no_grad()`. |
| **Approx size (Room 10K @ 1080p)** | ~500K × 8 B ≈ **4.0 MB** |
| **AVAILABLE_BEFORE_BACKWARD** | **NO** — Not saved in `ctx.save_for_backward()`. Computed under `@torch.no_grad()`, it is effectively a free intermediate that could be discarded after `isect_offset_encode`. |

---

### 13. `radii` — Maximum 2D projected radius (in pixels) per Gaussian

| Property | Value |
|----------|-------|
| **Shape** | `[C, N]` (non-packed) — `[1, 10000]` |
| **Dtype** | `int32` |
| **Source file / function** | `_FullyFusedProjection.forward` → CUDA `fully_fused_projection_fwd` |
| **Created at** | `rendering.py` line 297–314 |
| **Where consumed (bwd)** | **NOT consumed by `rasterize_to_pixels_bwd`.** Not passed to the backward kernel at all. Used in the trainer (line 335) for densification and visibility filtering — but that's post-backward Python code, not in the autograd backward. |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 4 B` ≈ **40 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **NO** — Not saved in `_RasterizeToPixels.ctx`. It is saved in `_FullyFusedProjection.ctx` (wrapper.py line 803–804: `ctx.save_for_backward(…, radii, conics, compensations)`) but that's a separate autograd scope. |

> **Note:** `radii` IS available in the `meta` dict returned by `rasterization()` — the trainer uses `meta["radii"][0]` at line 335. But it is NOT available inside `_RasterizeToPixels.backward()`.

---

### 14. `depths` — Z-depth of projected Gaussians

| Property | Value |
|----------|-------|
| **Shape** | `[C, N]` (non-packed) — `[1, 10000]` |
| **Dtype** | `float32` |
| **Source file / function** | `_FullyFusedProjection.forward` → CUDA `fully_fused_projection_fwd` |
| **Created at** | `rendering.py` line 297–314 |
| **Where consumed (bwd)** | **NOT consumed by `rasterize_to_pixels_bwd`.** Used only in forward for depth sorting (`isect_tiles` encodes depth into the 64-bit `isect_ids`). |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 4 B` ≈ **40 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **NO** — Not saved in `_RasterizeToPixels.ctx`. |

---

### 15. `compensations` — View-dependent opacity compensation (antialiasing)

| Property | Value |
|----------|-------|
| **Shape** | `[C, N]` or `None` — `None` in "classic" (default) mode |
| **Dtype** | `float32` |
| **Source file / function** | `_FullyFusedProjection.forward` → CUDA `fully_fused_projection_fwd`. Only produced when `calc_compensations=True` (i.e., `rasterize_mode="antialiased"`). |
| **Created at** | `rendering.py` lines 297–314 |
| **Where consumed (bwd)** | **N/A for classic mode.** The trainer uses `rasterize_mode="classic"` (default), so compensations are `None`. |
| **Approx size** | N/A (None in classic mode) |
| **AVAILABLE_BEFORE_BACKWARD** | **NO** — Not saved in `_RasterizeToPixels.ctx`. If antialiased, compensations are saved in `_FullyFusedProjection.ctx`. |

---

### 16. `tiles_per_gauss` — Number of tiles intersected by each Gaussian

| Property | Value |
|----------|-------|
| **Shape** | `[C, N]` (non-packed) — `[1, 10000]` |
| **Dtype** | `int32` |
| **Source file / function** | `rendering.py` line 497: `isect_tiles()` (`@torch.no_grad()`) |
| **Created at** | After `fully_fused_projection`, before `rasterize_to_pixels` |
| **Where consumed (bwd)** | **NOT consumed by backward.** Used only in the trainer (line 355: `meta["tiles_per_gauss"][0]`) for workload instrumentation (C53). |
| **Approx size (Room 10K @ 1080p)** | `1 × 10000 × 4 B` ≈ **40 KB** |
| **AVAILABLE_BEFORE_BACKWARD** | **NO** — Computed under `@torch.no_grad()`, not in `_RasterizeToPixels.ctx`. |

---

### 17. `backgrounds` gradient computation (Python-side only)

Although `backgrounds` is saved for backward (row 10 above), the actual gradient
`v_backgrounds` is **not computed by the CUDA kernel**. It is computed in Python at
wrapper.py lines 990–993:

```python
if ctx.needs_input_grad[4]:
    v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(dim=(1, 2))
```

This uses `render_alphas` (from `ctx.saved_tensors`) and `v_render_colors` (the incoming
gradient). The CUDA kernel does not touch this.

---

## Summary: What the backward kernel actually reads

The CUDA backward kernel `rasterize_to_pixels_bwd_kernel` (rasterize_to_pixels_bwd.cu lines 17–49)
receives **10 input tensors** (plus gradients and outputs):

| Input tensor | Fwd param line | Bwd uses |
|---|---|---|
| `means2d` | 23 | Reload xy to recompute delta |
| `conics` | 24 | Reload conic matrix to recompute sigma |
| `colors` | 25 | Reload RGB values for v_alpha partial derivative |
| `opacities` | 26 | Reload opacity to recompute alpha |
| `backgrounds` | 27 (optional) | Background gradient contribution |
| `masks` | 28 (optional) | Skip masked tiles |
| `image_width/height/tile_size` | 29–33 | Pixel coordinate math |
| `tile_offsets` | 34 | Tile range start/end |
| `flatten_ids` | 35 | Indirection to Gaussians |
| `render_alphas` | 37 | Reconstruct T_final |
| `last_ids` | 38 | Limit traversal to contributing Gaussians |

**All 10 input tensors are saved by `ctx.save_for_backward()`** — the backward kernel
has everything it needs to recompute `sigma`, `alpha`, `T`, and `vis` exactly as the
forward kernel computed them.

---

## What is NOT saved for backward (and why)

| Tensor | Reason skipped |
|--------|----------------|
| `render_colors` | Only its gradient (`v_render_colors`) is needed; forward value is irrelevant for backward math. |
| `isect_ids` | Computed eagerly (`@torch.no_grad`), used only to derive `isect_offsets` + `flatten_ids` which ARE saved. |
| `tiles_per_gauss` | Eagerly computed, used only for instrumentation (C53) — not needed for autograd. |
| `radii` | Used by trainer for densification (post-backward Python), not needed inside `_RasterizeToPixels.backward()`. Exists in `meta` dict. |
| `depths` | Used only for tile-sorting in forward; the depth is encoded into `isect_ids`. Not needed in backward. |
| `compensations` | Modulates opacities once in forward; if antialiased, saved in `_FullyFusedProjection.ctx`, not here. |

---

## absgrad flag

**Status in trainer:** `absgrad=True` (trainer.py line 125).

**How it flows:**
1. `rasterization()` passes `absgrad=True` to `rasterize_to_pixels()` (rendering.py line 570).
2. `rasterize_to_pixels()` passes `absgrad` to `_RasterizeToPixels.apply()` (wrapper.py line 533–546).
3. `_RasterizeToPixels.forward()` stores `ctx.absgrad = absgrad` (wrapper.py line 933).
4. `_RasterizeToPixels.backward()` passes `absgrad=True` to the CUDA bwd function (wrapper.py line 984).
5. The CUDA host function `call_kernel_with_dim` allocates `v_means2d_abs` when `absgrad=True` (bwd.cu lines 348–350).
6. The kernel computes both `v_means2d` (standard gradient) and `v_means2d_abs` (abs gradient) in the same loop (bwd.cu lines 232–234, 246–248, 267–271).
7. Back in Python, `means2d.absgrad = v_means2d_abs` (wrapper.py line 988).

The `absgrad` feature doubles the `means2d` gradient storage but reuses the same computation
— the abs gradient is just `abs(v_xy_local.x)` and `abs(v_xy_local.y)` taken from the same
per-Gaussian gradient.

---

## Memory Budget Summary (Room 10K @ 1080p, C=1)

### Saved for backward (must be in GPU memory at backward start)

| Tensor | Size |
|--------|------|
| `render_alphas` `[1,1080,1920,1]` | 8.3 MB |
| `last_ids` `[1,1080,1920]` | 8.3 MB |
| `tile_offsets` `[1,68,120]` | 32.6 KB |
| `flatten_ids` `[~500K]` | ~2.0 MB |
| `means2d` `[1,10000,2]` | 80 KB |
| `conics` `[1,10000,3]` | 120 KB |
| `colors` `[1,10000,3]` | 120 KB |
| `opacities` `[1,10000]` | 40 KB |
| `backgrounds` `[1,3]` | 12 B |
| **Subtotal (saved)** | **~19 MB** |

### Additional forward allocations (can be freed before backward)

| Tensor | Size |
|--------|------|
| `render_colors` `[1,1080,1920,3]` | 24.9 MB |
| `isect_ids` `[~500K]` int64 | ~4.0 MB |
| `radii` `[1,10000]` | 40 KB |
| `depths` `[1,10000]` | 40 KB |
| `tiles_per_gauss` `[1,10000]` | 40 KB |
| **Subtotal (freeable)** | **~29 MB** |

| Total forward allocation | **~48 MB** |
| Minimum persistent (saved) | **~19 MB** |

This means the forward pass allocates roughly **2.5×** more memory than is strictly needed
for backward computation. The three largest freeable tensors are `render_colors` (24.9 MB),
`isect_ids` (~4.0 MB), and the other meta tensors (~0.1 MB combined).

---

## Line Reference Quick Index

| Description | File | Lines |
|---|---|---|
| `ctx.save_for_backward()` — the saved-state contract | `_wrapper.py` | 918–929 |
| `ctx.saved_tensors` restoration in backward | `_wrapper.py` | 945–956 |
| CUDA fwd kernel signature (outputs: render_colors, alphas, last_ids) | `rasterize_to_pixels_fwd.cu` | 17–38 |
| CUDA bwd kernel signature (reads: 14 params, writes: 5 grads) | `rasterize_to_pixels_bwd.cu` | 17–49 |
| `rasterization()` orchestration in Python | `rendering.py` | 28–582 |
| Trainer calling `rasterization()` with `absgrad=True` | `trainer.py` | 109–132 |
| Trainer reading `meta["radii"]`, `meta["tiles_per_gauss"]` | `trainer.py` | 335, 355 |
| `absgrad`→`v_means2d_abs` allocation in CUDA host | `rasterize_to_pixels_bwd.cu` | 347–350 |
| `absgrad`→`v_xy_abs_local` computation in kernel | `rasterize_to_pixels_bwd.cu` | 232–234, 246–248, 267–271 |
| `means2d.absgrad` attribute set in Python | `_wrapper.py` | 987–988 |
