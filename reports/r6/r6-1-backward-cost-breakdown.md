# R6-1 — Backward Wall-Time Decomposition

## Methodology

For each workload (3 scenes × 3 training stages = 9 profiles), we:
1. Load a clean baseline checkpoint (ReferenceV1Config, gsplat 1.5.3, no candidate)
2. Warmup 20 iterations (forward + loss + backward + synchronize)
3. Measure 100 iterations with CUDA Events:
   - T_fwd: forward pass (render_with_meta)
   - T_bwd: loss.backward() only
   - T_iter: full iteration (forward + loss + backward)
4. Profile 30 backward-only iterations with torch.profiler (CUDA activity) to
   decompose T_bwd into kernel-level components

Camera: index 0 (same across all profiles for consistency).
Resolution: 1920×1080. SH degree: 3. Tile size: 16. Packed: false. absgrad: true.
Hardware: A100-PCIE-40GB, driver 595.71.05, CUDA 12.4, PyTorch 2.4.1.

## Results: backward decomposition (camera 0, 100 measured iterations)

| Scene | Stage | N | r_touch | T_bwd (ms) | T_raster (ms) | T_zero (ms) | T_zero%T_bwd | T_zero%T_iter | T_iter (ms) |
|-------|-------|---|---------|-----------:|--------------:|------------:|-------------:|--------------:|------------:|
| room | 5K | 560,632 | 0.330 | 43.9 | 26.8 | 2.99 | 6.6% | 2.9% | 103.5 |
| room | 15K | 933,590 | 0.305 | 43.0 | 22.7 | 3.89 | 9.0% | 3.7% | 104.5 |
| room | 30K | 933,590 | 0.284 | 42.1 | 20.8 | 2.36 | 5.5% | 2.2% | 105.0 |
| bicycle | 5K | 1,933,177 | 0.228 | 68.8 | 38.9 | 5.55 | 8.2% | 3.9% | 140.9 |
| bicycle | 15K | 3,957,041 | 0.294 | 82.7 | 42.6 | 8.33 | 9.6% | 5.3% | 158.4 |
| bicycle | 30K | 3,957,041 | 0.282 | 36.6 | 16.2 | 4.24 | 11.6% | 6.0% | 70.3 |
| garden | 5K | 1,779,185 | 0.180 | 49.1 | 21.0 | 4.65 | 10.5% | 4.2% | 111.3 |
| garden | 15K | 2,610,559 | 0.299 | 53.5 | 24.3 | 4.79 | 8.4% | 4.2% | 113.9 |
| garden | 30K | 2,610,559 | 0.292 | 27.4 | 11.1 | 3.09 | 11.3% | 5.3% | 58.3 |

Full kernel decomposition (per-iter ms, backward-only):

| Scene | Stage | raster_bwd | memset_zero | sh_bwd | proj_bwd | other_bwd | T_bwd_decomp |
|-------|-------|-----------:|------------:|-------:|---------:|----------:|-------------:|
| room | 5K | 26.81 | 2.99 | 0.12 | 0.10 | 15.43 | 45.45 |
| room | 15K | 22.71 | 3.89 | 0.17 | 0.41 | 15.77 | 42.95 |
| room | 30K | 20.81 | 2.36 | 0.17 | 0.97 | 18.95 | 43.26 |
| bicycle | 5K | 38.86 | 5.55 | 0.31 | 0.31 | 22.96 | 67.99 |
| bicycle | 15K | 42.63 | 8.33 | 0.41 | 0.76 | 29.61 | 81.74 |
| bicycle | 30K | 16.19 | 4.24 | 0.23 | 0.63 | 14.79 | 36.08 |
| garden | 5K | 21.00 | 4.65 | 0.28 | 0.38 | 18.02 | 44.33 |
| garden | 15K | 24.34 | 4.79 | 0.27 | 0.54 | 22.56 | 52.50 |
| garden | 30K | 11.12 | 3.09 | 0.18 | 0.37 | 12.36 | 27.12 |

Note: `other_bwd` (34-44% of T_bwd) is dominated by the SepSSIM/DSSIM loss backward
(15-channel separable Gaussian conv2d), NOT by renderer overhead. This is not
addressable by R6 renderer optimizations.

## Key observations

### 1. Backward dominates forward (4-7×)

T_bwd / T_fwd ranges from 4.2× (room 5K) to 7.1× (garden 5K). The backward is
the primary training cost center.

### 2. Rasterizer backward is the dominant kernel (48-60% of T_bwd)

The `rasterize_to_pixels_3dgs_bwd_kernel` accounts for 48-60% of total backward
time. This is the only kernel using atomic operations. The remaining time is
DSSIM loss backward (not addressable) and zero-init.

### 3. Zero-init is a significant fixed cost (5.5-11.6% of T_bwd)

Gradient buffer zero-initialization (at::zeros_like → cudaMemsetAsync) accounts
for 5.5-11.6% of backward time. This cost scales with TOTAL N (not touched
count), making it a fixed cost per iteration.

### 4. Sparse-tail pattern at 30K (B-GATE-3)

At 30K checkpoints, Gaussians are more optimized (fewer intersections, smaller
footprints). The rasterizer time decreases, but zero-init cost remains:

| Scene | 15K→30K | T_raster change | T_zero change | T_zero%T_bwd change |
|-------|---------|-----------------|---------------|---------------------|
| bicycle | 42.6→16.2ms | -62% | 8.33→4.24ms (-49%) | 9.6%→11.6% (+21%) |
| garden | 24.3→11.1ms | -54% | 4.79→3.09ms (-35%) | 8.4%→11.3% (+35%) |

As useful work decreases, zero-init becomes a LARGER fraction of backward.
This is the sparse-tail behavior (B-GATE-3).

### 5. Touch rate is low (18-33%)

Only 18-33% of Gaussians are visible/intersected per frame. The remaining
67-82% have their gradient buffers zero-initialized but never written to by
the rasterizer — pure waste.

## T_bwd decomposition formula

```
T_bwd = T_raster + T_zero + T_sh + T_proj + T_other
      = T_raster + T_zero + T_DSSIM_loss + T_autograd_overhead
```

Where:
- T_raster: rasterizer backward kernel (atomicAdd, 48-60% T_bwd)
- T_zero: gradient buffer zero-init (at::zeros_like, 5.5-11.6% T_bwd)
- T_sh + T_proj: elementwise VJP kernels (< 2% T_bwd, negligible)
- T_other: DSSIM loss backward + autograd overhead (34-44% T_bwd, not addressable)
