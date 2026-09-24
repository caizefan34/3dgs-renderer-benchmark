# C51 Sparse Backward Patch Notes — Reference Snapshot

**Label**: C51_SPARSE_BACKWARD_REFERENCE  
**Date**: 2026-09-14  
**Purpose**: Historical snapshot of the C51 sparse-backward CUDA implementation for reference in mechanism discovery.  
**Status**: ARCHIVED — not used in current or future training.

---

## 1. Patch Location

The C51 Stage 4A patch modifies the gsplat 1.5.3 CUDA rasterization backend to support `importance_mask` for skipping gradient computation for low-importance Gaussians in the backward pass.

**Original files patched** (inside gsplat 1.5.3 source tree):
- `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` — core backward kernel
- `gsplat/cuda/csrc/Rasterization.h` — declaration
- `gsplat/cuda/csrc/Rasterization.cpp` — C++ binding
- `gsplat/cuda/_wrapper.py` — Python autograd function
- `gsplat/rendering.py` — high-level API (`rasterization` function)

**Patch script**: `scripts/phase-c51-stage4a/patch_cuda.py`

---

## 2. Where Mask Is Loaded

The `importance_mask` is loaded as an additional kernel argument in `RasterizeToPixels3DGSBwd.cu`:

```cuda
template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    ...
    const bool *__restrict__ importance_mask,  // [N] — 0 = skip, 1 = compute
    ...
)
```

The mask is a 1D boolean tensor of shape `[N]` where:
- `importance_mask[i] = 1` → compute ALL gradients for Gaussian i
- `importance_mask[i] = 0` → skip gradient computation for Gaussian i

The mask is passed from Python through:
1. `rendering.py` → `rasterization()` function, parameter `importance_mask: Optional[torch.Tensor] = None`
2. `_wrapper.py` → `RasterizeToPixels3DGS.forward()` stores it in `ctx.importance_mask`
3. `_wrapper.py` → `RasterizeToPixels3DGS.backward()` passes it to the CUDA kernel

---

## 3. What Work Is Skipped

When `importance_mask[gaussian_idx] = 0`, the backward kernel skips:

### In the per-Gaussian backward loop (the main raster backward body):
1. **Color/SH gradient computation**: All `dL/dcolors` computation is skipped. The SH backward path is not entered.
2. **Conic/opacity gradient computation**: `dL/dconics` and `dL/dopacities` computation is skipped.
3. **Mean2D gradient computation**: `dL/dmeans2d` is skipped.

### In the atomicAdd accumulation:
4. **All atomicAdd operations** for the masked Gaussian's contributions are skipped:
   - `atomicAdd` for `dL/dconics` 
   - `atomicAdd` for `dL/dopacities`
   - `atomicAdd` for `dL/dmeans2d`
   - `atomicAdd` for `dL/dcolors`

### Design B variants:
- **B1/B3** (`compute_densify_grad=False`): Skip ALL gradient compute + atomicAdd for masked Gaussians.
- **B2** (`compute_densify_grad=True`): Compute `v_means2d` only (for densification gradient), skip `v_colors`/`v_conics`/`v_opacities` compute + atomicAdd.

---

## 4. What Work Still Executes

The following is **NOT** affected by the mask and always executes:

1. **Tile intersection** (forward only) — unchanged
2. **Forward rasterization** — unchanged
3. **Transmittance/buffer reconstruction** in backward — the backward kernel still reconstructs the per-pixel transmittance and blending state from the sorted Gaussians (reads `means2d`, `conics`, `colors`, `opacities` for T/buffer). Only gradient writes are gated.
4. **Pixel gradient** (`dL/drgb`) — computed from the loss, always
5. **Projection forward** — unchanged (always runs)
6. **Projection backward** — unchanged (always runs all gradient computation)
7. **Spherical Harmonics forward** — unchanged (always runs)
8. **Sorting** — unchanged (always runs tile intersection, prefix sum, sort)

---

## 5. Which Gradients Are Suppressed

When `importance_mask[i] = 0`, Gaussian i's gradients for the following parameters are suppressed to zero:

| Gradient | Suppressed? | Mechanism |
|----------|------------|-----------|
| `dL/dmean2d` (screen-space) | YES — skip in kernel | Not computed in raster bwd |
| `dL/dconics` | YES — skip in kernel | Not computed in raster bwd |
| `dL/dopacities` | YES — skip in kernel | Not computed in raster bwd |
| `dL/dcolors` (SH input) | YES — skip in kernel | Not computed in raster bwd |
| `dL/dxyz` (world-space) | IMPLICITLY YES | Through dL/dmean2d → projection bwd |
| `dL/dscale` | IMPLICITLY YES | Through dL/dconics path (skipped) |
| `dL/drotation` | IMPLICITLY YES | Through dL/dconics path (skipped) |
| `dL/dSH_coefficients` | IMPLICITLY YES | Through dL/dcolors (skipped) |

The densification gradient (`means2d.absgrad`) behavior is controlled by the `compute_densify_grad` parameter:
- B1/B3: Also suppressed (zero densification gradient for masked Gaussians)
- B2: Computed even for masked Gaussians (to preserve densification signal)

---

## 6. Exact Kernel/Function Names

| Component | File | Function/Kernel Name |
|-----------|------|---------------------|
| Forward raster CUDA kernel | `RasterizeToPixels3DGSFwd.cu` | `rasterize_to_pixels_3dgs_fwd_kernel` |
| Backward raster CUDA kernel | `RasterizeToPixels3DGSBwd.cu` | `rasterize_to_pixels_3dgs_bwd_kernel` |
| Forward C++ wrapper | `Rasterization.cpp` | `RasterizeToPixels3DGS::forward` |
| Backward C++ wrapper | `Rasterization.cpp` | `RasterizeToPixels3DGS::backward` |
| Python autograd forward | `_wrapper.py` | `RasterizeToPixels3DGS.forward` |
| Python autograd backward | `_wrapper.py` | `RasterizeToPixels3DGS.backward` |
| SH forward CUDA kernel | `compute_sh_fwd.cu` | `compute_sh_fwd_kernel` |
| SH backward CUDA kernel | `compute_sh_bwd.cu` | `compute_sh_bwd_kernel` |
| Projection forward CUDA | `fully_fused_projection_fwd.cu` | `fully_fused_projection_fwd_kernel` |
| Projection backward CUDA | `fully_fused_projection_bwd.cu` | `fully_fused_projection_bwd_kernel` |
| Tile intersection CUDA | `isect_tiles.cu` | `isect_tiles_kernel` |
| Sort (CUB) | `isect_tiles.cu` | `radix_sort_pairs` (CUB DeviceRadixSort) |

---

## 7. Patch Mode Summary

| Mode | Mask Behavior | compute_densify_grad | Use Case |
|------|--------------|---------------------|----------|
| `None` (baseline) | No mask | N/A | Original gsplat backward |
| B1 | Skip all gradient | False | Max speed, densification affected |
| B2 | Skip non-densify gradients | True | Speed + densification preserved |
| B3 | Skip all gradient | False | Same as B1 (alternative implementation) |

---

## 8. Compatibility Notes

- Designed for gsplat 1.5.3.
- The patch was applied and tested on A100-PCIE-40GB with CUDA 11.8.
- All C51 Stage 4A/4B/5 experiments used this patch.
- The patch is NOT compatible with gsplat 1.4.0 (currently installed local version).
- C51-R used a different approach: Python-level gradient masking after full backward (not CUDA kernel modification).
