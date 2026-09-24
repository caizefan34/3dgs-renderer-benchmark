# Phase 11 — M4 `radius_clip` Source Trace

**Date:** 2026-09-21  
**Author:** DSH coding agent  
**Status:** COMPLETE  
**gsplat version:** 1.5.3  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (Compute 12.0)

---

## 1. Purpose

Trace exactly what `radius_clip` does in the gsplat `rasterization()` pipeline — not from parameter names, but from actual CUDA kernel source code control flow.

---

## 2. Entry Point

`gsplat.rendering.rasterization()` accepts `radius_clip: float = 0.0`.

The parameter is forwarded through:

```
rasterization(radius_clip=...)
  └─ fully_fused_projection(radius_clip=...)     # packed or dense
       └─ _FullyFusedProjectionPacked.forward(radius_clip=...)
            └─ projection_ewa_3dgs_packed_fwd (CUDA kernel)
```

**radius_clip is ONLY passed to the projection kernel.** It is NOT passed to:
- `isect_tiles` (tile intersection)
- `_RasterizeToPixels` (pixel rasterization)
- Any backward kernel

---

## 3. CUDA Kernel: `projection_ewa_3dgs_packed_fwd_kernel`

### 3.1 Kernel Signature

```cpp
// ProjectionEWA3DGSPacked.cu line 266-280
void launch_projection_ewa_3dgs_packed_fwd_kernel(
    const at::Tensor means,
    const at::optional<at::Tensor> covars,
    const at::optional<at::Tensor> quats,
    const at::optional<at::Tensor> scales,
    const at::optional<at::Tensor> opacities,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,          // <--- HERE
    ...
);
```

### 3.2 Where radius_clip Takes Effect

**File:** `ProjectionEWA3DGSPacked.cu`, inside the per-Gaussian processing loop:

```cpp
// Line 179-209
float radius_x, radius_y;
if (valid) {
    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[col_idx];
        if (compensations != nullptr) {
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            valid = false;
        }
        // Opacity-aware bounding box (Mip-Splatting)
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // Compute tight rectangular bounding box (non-differentiable)
    radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    // *** radius_clip FILTER ***
    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        valid = false;  // Gaussian EXCLUDED from packed output
    }

    // Frustum culling
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        valid = false;
    }
}
```

### 3.3 Exact Semantics

| Property | Value |
|:---------|:------|
| **What is checked** | Both `radius_x` AND `radius_y` must be **≤** `radius_clip` |
| **Effect when triggered** | `valid = false` |
| **Consequence** | Gaussian is **excluded from packed output entirely**: no batch_id, camera_id, gaussian_id, radius, means2d, depth, conics, or compensation written |
| **Downstream impact** | Excluded Gaussians never reach `isect_tiles` or `_RasterizeToPixels` |
| **Is this filtering or clipping?** | **Filtering** (not clipping) — the Gaussian is completely removed from the pipeline, not modified |
| **Gradient flow** | No gradient for excluded Gaussians through the projection → rasterization path (they were never in the computational graph for this camera) |

### 3.4 Key Finding

**`radius_clip` is a packed-output filtering mechanism, not a numerical clipping operation.** It acts as a second visibility gate after frustum culling and opacity thresholding, all inside the projection kernel. Gaussians with both 2D radii ≤ `radius_clip` are treated as invisible.

---

## 4. Impact on Downstream Stages

### 4.1 Projection Output

When `radius_clip > 0`, the packed output (`nnz`) has fewer entries. The exclusion chain:

```
Projection (excluded if both radii <= radius_clip)
  → radii output for remaining Gaussians (unmodified values)
  → means2d, conics, depths for remaining Gaussians (unmodified)
```

### 4.2 Tile Intersection (`isect_tiles`)

The `intersect_tile_kernel` (in `IntersectTile.cu`) receives the reduced packed set. It processes only the Gaussians that survived projection. The kernel:

```cpp
// IntersectTile.cu line 55-77
const float radius_x = radii[idx * 2];
const float radius_y = radii[idx * 2 + 1];
if (radius_x <= 0 || radius_y <= 0) { return; }  // skip zero-radius

float tile_radius_x = radius_x / static_cast<float>(tile_size);
float tile_radius_y = radius_y / static_cast<float>(tile_size);
// compute tile range and write intersection indices
```

**radius_clip reduces the number of Gaussians entering tile intersection**, which reduces:
- `tiles_per_gauss` counts
- `n_isects` (total tile-Gaussian intersections)
- Sorting input size
- Rasterization workload

### 4.3 Sorting

Fewer Gaussians × fewer tiles → fewer sort elements. The sort is a segmented radix sort over `n_isects` elements.

### 4.4 Rasterization (`_RasterizeToPixels`)

The pixel rasterization kernel only sees Gaussians in its tile's intersection list. If a Gaussian was excluded at projection, it contributes zero pixels and zero computation in the rasterization kernel.

### 4.5 Backward

**radius_clip is NOT passed to any backward kernel.** The backward pass operates on the reduced set of Gaussians that survived forward projection. This means:

- `_FullyFusedProjectionPacked.backward` does NOT re-filter by radius_clip — it projects gradients back through the same surviving Gaussians
- **Gradient for excluded Gaussians = 0** (they were never in the computation for this camera)
- This is exactly the correct semantics: a Gaussian that didn't contribute to the output gets zero gradient for this view

### 4.6 Straight-Through / Mask Semantics

**None.** There is no straight-through estimator, no gradient mask, and no learned threshold for `radius_clip`. It is a purely deterministic filter applied during the forward projection based on the computed 2D bounding box size. The gradient computation is faithful to the forward computation: only the subset of Gaussians that actually contributed to the output receive gradients.

---

## 5. Interaction with Other Parameters

| Parameter | Interaction with radius_clip |
|:----------|:----------------------------|
| `eps2d` | `eps2d` is added to covariance eigenvalues before radius computation, so it affects the effective bounding box size. Larger `eps2d` → larger `radius_x/radius_y` → fewer Gaussians clipped |
| `packed` | radius_clip only has meaningful effect in packed mode (dense mode processes all Gaussians regardless) |
| `opacities` | Gaussian with opacity below threshold is already excluded before radius_clip check |
| `tile_size` | Not related to radius_clip (clip happens at projection, tile intersection uses the surviving radii) |

---

## 6. Summary: What radius_clip Actually Does

```
Pipeline position: During projection, after EWA computation and bounding box calculation
What it checks:    Both 2D axes radii <= radius_clip
Effect:            Gaussian excluded from packed output (not written to outputs)
Downstream:        Never reaches tile intersection, sorting, or rasterization
Gradient:          Zero gradient for excluded Gaussians (correct autograd semantics)
Backward pass:     Only sees surviving Gaussians; backward kernel does NOT filter
Threshold nature:  Deterministic, hard, non-differentiable
```

**Critical insight:** `radius_clip` does NOT modify the radius values of surviving Gaussians. It only removes entire Gaussians from the computation. This means:
- Surviving Gaussians have exactly the same radii, means2d, conics as without clipping
- The output image for non-excluded pixels is identical to the unclipped case
- Differences only appear where excluded Gaussians contributed visible pixels

The trade-off is:
- **Speed benefit:** Fewer Gaussians in tile intersection, sorting, and rasterization
- **Quality cost:** Gaussians that were visible (even if tiny) are removed, potentially causing missing splats in the final image
