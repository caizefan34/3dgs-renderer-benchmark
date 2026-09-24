# Tensor Shapes — Candidate C Source Audit

## Canonical Room Configuration

Source: `configs/reference_room_30k.yaml` and `baseline/reference_room_30k/trainer.py`

| Configuration | Value |
|---------------|-------|
| Scene | `room` |
| Iterations | 30,000 |
| Resolution | 1080p (1920×1080) |
| Initial Gaussians | 112,627 (SfM points) |
| Final Gaussians | 952,353 (peak during densification) |
| SH degree | 0→3 (progression every 1,000 iters) |
| Tile size | 16×16 |
| Batch size | 1 (single camera) |

## Tensor Shapes at Key Points

### Input / Scene Data
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `points3D` (SfM xyz) | `[N_init, 3]` | float32 | Real-SfM-initialized |
| Initial `means` | `[N_init, 3]` | float32 | Same as SfM |
| Initial `colors` | `[N_init, 3]` | float32 | SH DC color |

### Forward Path (per iteration)

#### Rendering
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `render_colors` | `[H, W, 3]` | float32 | Output image (also `[1, H, W, 3]` inside renderer) |
| `render_alphas` | `[H, W, 1]` | float32 | Alpha output (accumulated) |
| `means2d` | `[N, 2]` or `[1, N, 2]` | float32 | Projected 2D means; saved for backward if `absgrad` |
| `radii` | `[N]` or `[1, N]` | int32/int64 | Max 2D extent in pixels; saved for backward |
| `conics` | `[N, 3]` or `[1, N, 3]` | float32 | Inverse covariance 2D (upper triangle) |
| `opacities` | `[N]` or `[1, N]` | float32 | Sigmoid(activated) |
| `colors` | `[N, 3]` or `[1, N, 3]` | float32 | After SH eval (RGB) |

#### Loss
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `loss` | scalar | float32 | Weighted L1 + D-SSIM |
| `L1` | scalar | float32 | `(1 - λ) * mean(|image - gt|)` |
| `dssim` | scalar | float32 | `λ * (1 - SSIM)` |

#### Backward Gradient Tensors
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `v_render_colors` | `[H, W, 3]` | float32 | From loss backward |
| `v_render_alphas` | `[H, W, 1]` | float32 | From loss backward (typically zeros) |
| `v_colors` | `[N, 3]` or `[nnz, 3]` | float32 | From raster backward |
| `v_opacities` | `[N]` or `[nnz]` | float32 | From raster backward |
| `v_means2d` | `[N, 2]` or `[nnz, 2]` | float32 | From raster backward (includes `absgrad` if used) |
| `v_conics` | `[N, 3]` or `[nnz, 3]` | float32 | From raster backward |
| `v_means2d_abs` | `[N, 2]` or `[nnz, 2]` | float32 | If `absgrad=True` |

### Packed vs Dense Mode Note
- `packed=False` (Current): Dense tensors `[1, N, ...]` — full N arrays passed to CUDA every iteration.
- `packed=True`: Dense tensors of shape `[nnz, ...]` where `nnz = number of visible (non-culled) Gaussians`.

### Optimization State (per Gaussian parameter)
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `exp_avg` (per param) | Same as param | float32 | Adam first moment |
| `exp_avg_sq` (per param) | Same as param | float32 | Adam second moment |
| `max_exp_avg_sq` | Same as param | float32 | Only for scale/rotation (AMSGrad) |

### Densification State
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `max_radii2D` | `[N]` | float32 | Per-Gaussian max projected radius |
| `xyz_gradient_accum` | `[N, 1]` | float32 | Accumulated view-space gradient norm |
| `denom` | `[N]` | float32 | Accumulation counter (number of views) |
| `G_dens` | `[N]` | float32 | Densification gradient (mean2d abs) |

## Dtype Summary
All tensors in the canonical path use float32. Camera/light intrinsics use `torch.float32`. The only non-float32 tensors are index/int tensors:
- `radii`: dtype `int32` in newer gsplat versions (check)
- `tile_indices`, `flatten_ids`, `isect_ids`: `int32`/`int64` (CUDA intermediates)

## Approximate Sizes (Room, 1080p)
- Image: 1920×1080×3 = 6.2 MB per camera
- Initial SfM points: `112,627 × 3` = 1.35 MB
- Final Gaussians at peak: `952,353 × 3` ≈ 11.4 MB (xyz only)
- Per-target gradient shapes: `[N,3]` / `[N,1]` for each of 5+ params
  - `xyz`: `[N,3]`, `scale`: `[N,3]`, `rotation`: `[N,4]`, `opacity`: `[N,1]`, `colors`: `[N,K,3]`
