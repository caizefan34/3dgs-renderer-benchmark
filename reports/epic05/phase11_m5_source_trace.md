# Phase 11 — M5 `eps2d` Source Trace

**Date:** 2026-09-21  
**Author:** DSH coding agent  
**Status:** COMPLETE  
**gsplat version:** 1.5.3  

---

## 1. Purpose

Trace exactly what `eps2d` does in the gsplat `rasterization()` pipeline — from API through CUDA kernels and backward pass.

---

## 2. Entry Point

`gsplat.rendering.rasterization()` accepts `eps2d: float = 0.3`.

The parameter is forwarded to:

```
rasterization(eps2d=...)
  └─ fully_fused_projection(eps2d=...)
       └─ _FullyFusedProjectionPacked.forward(eps2d=...)
            └─ projection_ewa_3dgs_packed_fwd CUDA kernel (eps2d)
```

`eps2d` is also saved in `ctx.eps2d` for the backward pass of the projection kernel.

---

## 3. The `add_blur` Function (CUDA)

**File:** `gsplat/cuda/include/Utils.cuh` (lines 380-388)

```cpp
inline __device__ float
add_blur(const float eps2d, mat2 &covar, float &compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;      // ADD eps2d to diagonal
    covar[1][1] += eps2d;      // ADD eps2d to diagonal
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}
```

### 3.1 What `eps2d` Actually Changes

| Aspect | How eps2d affects it |
|:-------|:--------------------|
| **2D Covariance (`covar2d`)** | Diagonal elements increased by `eps2d`. This is **in-place modification**. |
| **Determinant** | Increased (blurred determinant ≥ original determinant) |
| **Inverse covariance (conic)** | Computed from the **blurred** covariance matrix |
| **Bounding box radii** | `radius_x = ceilf(extend * sqrt(covar2d[0][0]))` — **larger** due to increased variance |
| **Tile coverage** | Larger radii → Gaussians cover **more tiles** |
| **Pixel alpha** | Gaussian response function uses the conic (blurred inverse covariance) → **wider, softer splats** |
| **Compensation factor** | `compensation = sqrt(det_orig / det_blur)` — always ≤ 1.0 |

### 3.2 Compensation (Only Applied in "antialiased" Mode)

```python
# From rasterization() (line 420-421):
if compensations is not None:
    opacities = opacities * compensations
```

In **classic mode** (default): compensation is computed but **NOT** applied to opacities. Only the blurred covariance affects the rendering.

In **antialiased mode**: opacities are multiplied by compensation factor to reduce the opacity of blurred Gaussians (Mip-Splatting approach).

---

## 4. Downstream Effects in Forward Pass

### 4.1 Conic (Inverse Covariance)

The conic is computed from the blurred covariance:
```cpp
covar2d_inv = glm::inverse(covar2d);  // covar2d already has eps2d added
```

The conic directly controls the Gaussian response at each pixel:
```
response = exp(-0.5 * [dx, dy] @ conic @ [dx, dy]^T)
```

Larger `eps2d` → wider covariance → smaller conic values → smoother, wider splats.

### 4.2 Radius / Tile Coverage

```cpp
radius_x = ceilf(extend * sqrtf(covar2d[0][0]));  // after eps2d added
```

Larger `eps2d` → larger radii → Gaussians cover more tiles → **more tile-Gaussian intersections**. This is the opposite of `radius_clip`.

### 4.3 Numerical Stability

A key secondary effect: `eps2d` prevents singular covariance matrices (when `det` approaches 0). Without `eps2d`, very small covariances can produce degenerate conics, causing numerical issues.

---

## 5. Backward Pass: `add_blur_vjp`

**File:** `Utils.cuh` (lines 390-423)

```cpp
inline __device__ void add_blur_vjp(
    const float eps2d,
    const mat2 conic_blur,
    const float compensation,
    const float v_compensation,
    mat2 &v_covar
) {
```

The backward pass correctly accounts for the blur:
- Gradients of the compensation factor are back-propagated through the eps2d perturbation
- `v_covar` receives contributions from both the rendering gradient and the compensation gradient
- The `eps2d` parameter itself is a constant (not a learned parameter), so no gradient w.r.t. eps2d

---

## 6. Summary: What `eps2d` Actually Does

```
eps2d = 0.0     → no blur, original covariance used directly
eps2d = 0.3     → diagonal of 2D covariance increased by 0.3 (default)
eps2d = large   → Gaussians become wider, softer, cover more tiles
```

### Impact Dimensions

| Dimension | Impact |
|:----------|:-------|
| **Covariance** | ✅ Adds eps2d to diagonal (in-place modification of `covar[0][0]` and `covar[1][1]`) |
| **Conic** | ✅ Inverse of blurred covariance (affects pixel alpha) |
| **Radius** | ✅ Larger radii → more tiles covered |
| **Numerical stability** | ✅ Prevents degenerate covariances (determinant artificially increased) |
| **Tile coverage** | ✅ INCREASED (more intersections, more work) |
| **Pixel alpha** | ✅ Wider, softer splats |
| **Compensation (antialiased)** | ✅ Opacity scaling by sqrt(det_orig/det_blur) |
| **Backward** | ✅ Correct VJP through the blur operation |

### Key Trade-off

Larger `eps2d` → more tile intersections (MORE work), wider splats (softer image).
Smaller `eps2d` → fewer tile intersections (LESS work), sharper splats (potential aliasing).

This makes `eps2d` different from `radius_clip` in a critical way:
- **radius_clip** removes work (fewer Gaussians, fewer intersections)
- **eps2d** ADDS work (larger radii, more intersections at larger values)

Testing should verify: does reducing `eps2d` from 0.3 (default) to 0.01 reduce workload and speed up rendering while preserving quality?
