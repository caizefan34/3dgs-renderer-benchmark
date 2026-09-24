# R2 — Gradient Dependency Graph

## gsplat 1.5.3 Backward Autograd Graph for REFERENCE_V1_ABSGRAD

### Forward Path

```
trainer.py
  → rasterization(means, quats, scales, opacities, colors[SH], viewmats, Ks, ...)
    │
    ├─ fully_fused_projection(means, quats, scales, viewmats, Ks)
    │   └─ CUDA: projection_ewa_3dgs_fused_fwd
    │      → radii, means2d, depths, conics
    │
    ├─ spherical_harmonics(sh_degree, dirs=means-campos, coeffs=SH)
    │   └─ CUDA: spherical_harmonics_fwd
    │      → colors [N, 3] (RGB from SH evaluation)
    │
    ├─ isect_tiles(means2d, radii, depths)
    │   → tiles_per_gauss, isect_ids, flatten_ids
    │
    └─ rasterize_to_pixels(means2d, conics, colors[3], opacities, ...)
        └─ CUDA: rasterize_to_pixels_3dgs_fwd
           → render_colors [H, W, 3], render_alphas [H, W, 1]
```

### Backward Path

```
loss.backward()
  │
  ├─ v_render_colors [H, W, 3], v_render_alphas [H, W, 1]
  │   (from L1 + SSIM backward)
  │
  ├─ _RasterizeToPixels.backward(v_render_colors, v_render_alphas)
  │   │  CUDA: rasterize_to_pixels_3dgs_bwd
  │   │  Timing: 11.26ms (41.4% E2E) — SHARED
  │   │
  │   ├─ v_colors [N, 3]     → APPEARANCE (feeds SH backward)
  │   ├─ v_opacities [N]     → OPACITY_ONLY
  │   ├─ v_means2d [N, 2]    → GEOMETRY (feeds projection backward)
  │   └─ v_conics [N, 3]     → GEOMETRY (feeds projection backward)
  │
  ├─ _SphericalHarmonics.backward(v_colors)
  │   │  CUDA: spherical_harmonics_bwd
  │   │  Timing: 0.28ms (1.0% E2E) — COUPLED
  │   │
  │   ├─ v_coeffs [N, K, 3]  → dL/dSH (APPEARANCE)
  │   └─ v_dirs [N, 3]       → dL/dxyz (GEOMETRY, via dirs = means - campos)
  │
  ├─ _FullyFusedProjection.backward(v_means2d, v_conics)
  │   │  CUDA: projection_ewa_3dgs_fused_bwd
  │   │  Timing: 0.17ms (0.6% E2E) — GEOMETRY_ONLY
  │   │
  │   ├─ v_means [N, 3]      → dL/dxyz (GEOMETRY)
  │   ├─ v_scales [N, 3]     → dL/dscale (GEOMETRY)
  │   └─ v_quats [N, 4]      → dL/drot (GEOMETRY)
  │
  └─ Activation backward
      ├─ sigmoid_bwd → dL/dopacity_raw
      ├─ exp_bwd → dL/dscaling_raw
      └─ normalize_bwd → dL/drot_raw
```

### Key Dependency: SH → xyz

The SH evaluation uses `dirs = means - campos` as input. The SH backward therefore produces `v_dirs` which flows back to `means` (xyz).

Source: `gsplat/cuda/_wrapper.py:1831-1845`
```python
class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        compute_v_dirs = ctx.needs_input_grad[1]  # True if dirs (→means) requires grad
        v_coeffs, v_dirs = _make_lazy_cuda_func("spherical_harmonics_bwd")(
            num_bases, sh_degree, dirs, coeffs, masks,
            v_colors.contiguous(), compute_v_dirs)
        return None, v_dirs, v_coeffs, None
```

The `compute_v_dirs` flag is True whenever xyz requires gradient (which is always in training). This means:
- **v_coeffs** → dL/dSH (appearance parameter)
- **v_dirs** → dL/dxyz (geometry parameter, via view direction)

Both are computed in the same `spherical_harmonics_bwd` kernel call.

### Raster Backward Internal Structure

Within `rasterize_to_pixels_3dgs_bwd`, the per-pixel loop:

```cuda
for each pixel (x, y):
    T = 1.0  // transmittance
    for each gaussian i in tile (depth-sorted):
        if importance_mask[i] == 0:
            // Skip gradient computation but still update T and buffer
            alpha = opacity[i] * gaussian_2d_contribution(...)
            T *= (1 - alpha)
            continue

        // Gradient computation (skippable via importance_mask):
        v_colors[i] += atomicAdd(...)   // 3 atomicAdds (RGB) → APPEARANCE
        v_opacities[i] += atomicAdd(...) // 1 atomicAdd → OPACITY
        v_means2d[i] += atomicAdd(...)   // 2 atomicAdds → GEOMETRY
        v_conics[i] += atomicAdd(...)    // 3 atomicAdds → GEOMETRY

        T *= (1 - alpha)  // SHARED: always needed
```

**CDIM = 3** (RGB channels), not 48 SH coefficients. SH evaluation happens BEFORE rasterization in the forward pass; the raster kernel only sees the 3-channel RGB colors.

### Derivative Work Classification

| Operation | Type | Exclusive? | Cost |
|-----------|------|:----------:|------|
| rasterize_to_pixels_3dgs_bwd | SHARED | No | 11.26ms |
| spherical_harmonics_bwd | COUPLED (SH+xyz) | No | 0.28ms |
| projection_ewa_3dgs_fused_bwd | GEOMETRY_ONLY | Yes | 0.17ms |
| Loss + SSIM + autograd | SHARED | No | 10.63ms |

### SH Decomposition Verdict

**SH_PATH_COUPLED**: The dSH computation cannot be skipped for selected Gaussians while preserving the geometry contribution from view-dependent color. Both v_coeffs (→dL/dSH) and v_dirs (→dL/dxyz) are computed in the same kernel from the same SH basis evaluation.

Separating them would require:
1. A new CUDA kernel computing only v_dirs (view direction gradient)
2. This kernel would still evaluate SH basis functions
3. Savings would be minimal (0.28ms total for both paths)
