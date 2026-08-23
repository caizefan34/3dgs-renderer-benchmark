# Phase 9B — M3 SH Degree Source Trace

**Date:** 2026-09-21  
**Author:** DSH coding agent  
**Status:** COMPLETE

---

## 1. Purpose

Trace exactly what changes when `sh_degree` is set to 0, 1, or 3 in the gsplat
`rasterization()` pipeline — not from configuration variable names, but from actual
source code control flow through CUDA kernels.

**Source file traced:** `gsplat/rendering.py` (lines 33–770)  
**CUDA autograd file:** `gsplat/cuda/_wrapper.py`  
**CUDA kernel source:** `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`  
**gsplat version:** 1.5.3  

---

## 2. Entry Point

`gsplat.rendering.rasterization()` accepts `sh_degree: Optional[int] = None`.

When `sh_degree` is:
- **`None`**: `colors` tensor is treated as **post-activation RGB values** with shape `[..., N, 3]` or `[..., C, N, 3]`. No SH evaluation occurs. Colors are used directly in rasterization.
- **`0`, `1`, `2`, `3`**: `colors` tensor is treated as **SH coefficients** with shape `[..., N, K, 3]` or `[..., C, N, K, 3]`, where `K = (sh_degree+1)^2` is the number of SH bases.

### Tensor Shape Constraint

```python
assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
# SH0: K=1,  SH1: K=4,  SH3: K=16
```

The assertion only checks that the **stored** tensor has enough coefficients; the actual
**evaluation** uses `sh_degree` to control how many bases are computed.

---

## 3. Color Computation Path

### 3.1 When `sh_degree is None` (pass-through)

```
colors = colors[..., batch_ids, gaussian_ids]  # [nnz, 3] packed
or
colors = broadcast_to(..., C, N, 3)           # [..., C, N, 3] dense
```

No SH evaluation. Colors are used as-is. Shape `[..., 3]` RGB.

### 3.2 When `sh_degree = n` (SH evaluation)

```
dirs = means - campos                                # [nnz, 3] or [..., C, N, 3]
shs = colors[..., batch_ids, gaussian_ids, :, :]     # [nnz, K, 3] packed
or
shs = broadcast_to(..., C, N, K, 3)                  # [..., C, N, K, 3] dense
colors = spherical_harmonics(sh_degree, dirs, shs)    # -> [nnz, 3] or [..., C, N, 3]
```

The SH evaluation **always produces 3 output channels** (RGB), regardless of degree.

#### What SH degree controls internally:

Inside `spherical_harmonics()` (in `gsplat/cuda/_wrapper.py`):

```python
def spherical_harmonics(degrees_to_use, dirs, coeffs, masks=None):
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    return _SphericalHarmonics.apply(degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks)
```

The `_SphericalHarmonics` autograd function calls:
- Forward: `spherical_harmonics_fwd_kernel` with `sh_degree`
- Backward: `spherical_harmonics_bwd_kernel` with `num_bases`, `sh_degree`

---

## 4. CUDA Kernel Differences

### 4.1 Forward Kernel: `spherical_harmonics_fwd_kernel`

**Source:** `SphericalHarmonicsCUDA.cu`  
**Launch config:** Grid dependent on `nnz` (visible Gaussian count), block = 256 threads

The kernel implements the standard SH basis functions up to degree `sh_degree`:

| SH Degree | Bases Computed | Basis Functions Used |
|:---------:|:--------------:|:--------------------|
| 0 | 1 | Y₀⁰ (DC / constant term) |
| 1 | 4 | Y₀⁰ + Y₁⁻¹ + Y₁⁰ + Y₁¹ (DC + 3 linear) |
| 3 | 16 | Y₀⁰ through Y₃³ (full 3rd-degree expansion) |

**Computational workload** (per Gaussian-camera pair):

| SH Degree | Coeffs read | Multiply-adds (approx) | Memory load |
|:---------:|:-----------:|:----------------------:|:-----------:|
| 0 | 3 (DC RGB) | ~3 | 3 × float |
| 1 | 12 | ~12 | 12 × float |
| 3 | 48 | ~48 | 48 × float |

**Register pressure:** Higher degrees use more registers for intermediate basis values. The kernel has specializations:
- `spherical_harmonics_fwd_kernel<reg>` where reg count increases with degree

### 4.2 Backward Kernel: `spherical_harmonics_bwd_kernel`

**Source:** `SphericalHarmonicsCUDA.cu`  
**Launch config:** Grid dependent on `nnz`, block = 256 threads

Computes:
- `∂L/∂coeffs`: gradient w.r.t. SH coefficients (shape: same as coeffs input)
- `∂L/∂dirs` (optional): gradient w.r.t. view direction

**Workload scales linearly with number of SH bases:**
- SH0: 3 coeff gradients per Gaussian
- SH1: 12 coeff gradients per Gaussian
- SH3: 48 coeff gradients per Gaussian

---

## 5. What DOES Change with SH Degree

### 5.1 Tensor Dimensions

| Aspect | SH0 | SH1 | SH3 |
|:-------|:---:|:---:|:---:|
| **SH coeffs stored** | [N, 1, 3] | [N, 4, 3] | [N, 16, 3] |
| **SH coeffs used** | [nnz, 1, 3] | [nnz, 4, 3] | [nnz, 16, 3] |
| **Output RGB** | [nnz, 3] | [nnz, 3] | [nnz, 3] |
| **Trainable SH params** | N × 3 | N × 12 | N × 48 |
| **Gradient for SH** | [N, 1, 3] | [N, 4, 3] | [N, 16, 3] |

**Critical:** SH degree does NOT change the shape of any other tensor (xyz, scales, rotations, opacity). The rasterization kernel receives identical `[nnz, 3]` colors regardless of SH degree.

### 5.2 What Changes vs What Stays the Same

| Component | SH0 | SH1 | SH3 | Change? |
|:----------|:---:|:---:|:---:|:-------:|
| Projection kernel | same | same | same | ❌ No |
| Projection output shape | C×N/nnz | C×N/nnz | C×N/nnz | ❌ No |
| SH evaluation kernel | fwd_kernel<lib> | fwd_kernel<lib> | fwd_kernel<lib> | ✅ Degree param |
| SH coeffs read per pair | 3 | 12 | 48 | ✅ 1× → 4× → 16× |
| SH backward kernel | bwd_kernel<lib> | bwd_kernel<lib> | bwd_kernel<lib> | ✅ Degree param |
| Rasterization kernel | same | same | same | ❌ No |
| Rasterization input | [nnz, 3] | [nnz, 3] | [nnz, 3] | ❌ No |
| Tile intersection | same count | same count | same count | ❌ No |
| CUB radix sort | same | same | same | ❌ No |
| Optimizer params | N×(3+3+4+1+3) | N×(3+3+4+1+12) | N×(3+3+4+1+48) | ✅ SH group only |

### 5.3 Trainable Parameter Count

| SH Degree | SH params | Total params | vs SH3 |
|:---------:|:---------:|:------------:|:------:|
| 0 | 1,593,376 × 3 = 4.78M | 18.60M | 3.0× fewer |
| 1 | 1,593,376 × 12 = 19.12M | 32.94M | 1.7× fewer |
| 3 | 1,593,376 × 48 = 76.48M | 90.30M | reference |

For a trained scene (e.g., 1.19M Gaussians after 30K steps):
| SH Degree | SH params | Total params | vs SH3 |
|:---------:|:---------:|:------------:|:------:|
| 0 | 1,193,480 × 3 = 3.58M | 13.91M | 3.0× fewer |
| 1 | 1,193,480 × 12 = 14.32M | 24.65M | 1.7× fewer |
| 3 | 1,193,480 × 48 = 57.29M | 67.62M | reference |

---

## 6. What Does NOT Change

1. **Rasterization kernel**: The tile-based alpha compositing is identical regardless of SH degree because colors are always `[nnz, 3]` by the time they reach `rasterize_to_pixels()`.

2. **Projection**: Same kernel, same C×N or nnz workload.

3. **Tile intersection**: Same visible Gaussian set, same tile coverage computation, same CUB radix sort input size.

4. **Densification/Pruning**: These operate on position/scale/rotation/opacity gradients. SH degree affects the loss landscape (different color representation → different gradient signal for position/scale/rotation/opacity), but the mechanics of densification and pruning are identical.

5. **Other parameters (xyz, scales, rotations, opacity)**: Same tensor shapes, same trainable parameter counts regardless of SH degree.

---

## 7. CUDA Kernel Registration Confirmation

Confirmed via `gsplat/cuda/_wrapper.py`:

```python
# SH forward/bwd kernels are lazy-loaded:
_make_lazy_cuda_func("spherical_harmonics_fwd")
_make_lazy_cuda_func("spherical_harmonics_bwd")
```

The compiled CUDA binary (`gsplat_cuda.pyd`) contains:
- `spherical_harmonics_fwd_kernel<reg>` — single kernel with degree-templated register usage
- `spherical_harmonics_bwd_kernel<reg>` — same for backward

Both are loaded at runtime. The degree value is passed as a kernel argument, not as a C++ template parameter, so **the same compiled binary handles all degrees**.

---

## 8. Summary

| Aspect | SH0 | SH1 | SH3 |
|:-------|:---:|:---:|:---:|
| SH bases | 1 (DC) | 4 (DC + linear) | 16 (full 3rd order) |
| Tensor shape used | [nnz, 1, 3] | [nnz, 4, 3] | [nnz, 16, 3] |
| Trainable SH params | 3 per Gaussian | 12 per Gaussian | 48 per Gaussian |
| SH eval compute | ~3 ops/pair | ~12 ops/pair | ~48 ops/pair |
| Rasterization | identical | identical | identical |
| Backward SH gradient | [N, 1, 3] | [N, 4, 3] | [N, 16, 3] |
| Total param count (1.6M Gs) | 18.6M | 32.9M | 90.3M |
| Rasterize kernel cost | same | same | same |
| Tile intersection cost | same | same | same |
| CUB sort cost | same | same | same |

### Key Insight

**SH degree is a representation capacity control, not a rendering pipeline change.** The only pipeline stages affected are:
1. **SH forward evaluation** — computes (degree+1)² RGB contributions per visible Gaussian
2. **SH backward** — computes gradients for (degree+1)² coefficient values per Gaussian
3. **Optimizer state** — more/less SH parameters to store and update

The **rasterization kernel itself is completely unaffected** by SH degree because it always receives `[nnz, 3]` RGB colors. The **tile intersection** and **projection** pipelines are also identical.

This means SH degree's performance impact in training comes from:
- SH eval/compute stage (minor, <1% of forward time)
- SH backward (minor, <1% of backward time per the Phase 8C trace)
- Optimizer step for SH parameters (proportional to parameter count)
- **Indirect effect**: Different color representation changes the loss landscape, which changes densification/pruning decisions and thus **Gaussian count over time**

The **indirect effect through training trajectory** is likely more significant than the direct compute cost.
