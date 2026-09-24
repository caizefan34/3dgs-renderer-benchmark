# C40-0: Training Kernel Bottleneck Profiling — Final Report

## Summary

The gsplat renderer is **NOT the training bottleneck**. The D-SSIM loss computation (cudnn convolution) dominates at **40.7%** of GPU kernel time. The actual rendering pipeline (projection + intersection + sort + rasterize forward) is only **4.3%**.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Scene | room (Mip-NeRF 360) |
| Resolution | 1080p (1920×1080) |
| Camera | Fixed camera 0 |
| Checkpoint | iter 5000 (1,000,684 Gaussians, SH degree 3) |
| Densification | DISABLED |
| Warmup | 30 iterations |
| CUDA events iterations | 20 |
| Profiler iterations | 10 (active) |
| tile_size | 16, packed=True |

---

## Method 1: CUDA Events (High-Level Breakdown)

| Stage | Mean (ms) | Median (ms) | P90 (ms) | % of Total |
|-------|-----------|-------------|----------|------------|
| **Forward** (render) | 6.86 | 6.86 | 7.75 | **8.0%** |
| **Loss** (L1 + D-SSIM) | 20.43 | 20.25 | 24.03 | **24.0%** |
| **Backward** | 40.90 | 39.24 | 51.79 | **48.0%** |
| **Optimizer** (Adam) | 13.81 | 14.22 | 15.41 | **16.2%** |
| **Total** | 85.27 | 83.91 | 102.06 | 100% |

**Backward is 48%** — just under the 50% threshold. Forward rendering is only 8%.

---

## Method 2: torch.profiler (Kernel-Level Breakdown)

10 profiled iterations, 2590 CUDA kernel events, 88.10 ms/iter total GPU kernel time.

| Category | Time/iter (ms) | % of Total | Kernels/iter |
|----------|---------------|------------|--------------|
| **loss_dssim** | 35.827 | **40.7%** | 9 |
| **adam** | 17.484 | **19.8%** | 41 |
| **elementwise_misc** | 14.944 | **17.0%** | 130 |
| **rasterize_bwd** | 10.329 | **11.7%** | 1 |
| other (incl. intersection) | 3.512 | 4.0% | 17 |
| sorting | 2.016 | 2.3% | 19 |
| rasterize_fwd | 1.600 | 1.8% | 1 |
| memory | 1.221 | 1.4% | 31 |
| projection_bwd | 0.557 | 0.6% | 1 |
| reduction | 0.434 | 0.5% | 9 |
| projection_fwd | 0.177 | 0.2% | 1 |

### Rendering pipeline subtotal

| Rendering stage | Time/iter (ms) | % of Total |
|----------------|---------------|------------|
| projection_fwd | 0.177 | 0.2% |
| intersection (in "other") | 1.603 | 1.8% |
| sorting | 2.016 | 2.3% |
| rasterize_fwd | 1.600 | 1.8% |
| **Forward subtotal** | **5.396** | **6.1%** |
| rasterize_bwd | 10.329 | 11.7% |
| projection_bwd | 0.557 | 0.6% |
| **Backward subtotal** | **10.886** | **12.4%** |
| **Rendering total** | **16.282** | **18.5%** |

### Non-rendering subtotal

| Non-rendering stage | Time/iter (ms) | % of Total |
|---------------------|---------------|------------|
| loss_dssim | 35.827 | 40.7% |
| adam | 17.484 | 19.8% |
| elementwise_misc | 14.944 | 17.0% |
| memory + reduction + other | 5.167 | 5.9% |
| **Non-rendering total** | **73.422** | **83.4%** |

---

## Top Kernels Per Category

### loss_dssim (40.7% — THE BOTTLENECK)

```
cudnn::dgrad2d_grouped_direct_kernel:  19.690 ms/iter  (3 calls/iter)
cudnn::conv2d_grouped_direct_kernel:   16.136 ms/iter  (6 calls/iter)
```

These are cuDNN convolution kernels used for D-SSIM (structural similarity) loss computation. The D-SSIM loss requires:
1. Gaussian blur of rendered and GT images (convolution)
2. Mean/variance/covariance computation (more convolutions)
3. Gradient backward through convolutions (dgrad)

**9 kernel calls per iteration consume 40.7% of GPU time.**

### adam (19.8%)

```
multi_tensor_apply_kernel (pointwise):   3.480 ms/iter  (6 calls/iter)
multi_tensor_apply_kernel (pointwise):   2.457 ms/iter  (6 calls/iter)
multi_tensor_apply_kernel (ternary):     2.426 ms/iter  (6 calls/iter)
```

Adam optimizer with 5 parameter groups (xyz, rotations, scales, opacity, SHs). Each group requires:
- Update step (m, v, param)
- Bias correction
- 41 kernel calls per iteration

### rasterize_bwd (11.7%)

```
gsplat::rasterize_to_pixels_3dgs_bwd_kernel:  10.329 ms/iter  (1 call/iter)
```

The single gsplat backward rasterization kernel. This is the ONLY gsplat kernel that appears in the top-5 categories.

### elementwise_misc (17.0%)

```
elementwise_kernel (binary div):       3.170 ms/iter  (10 calls)
vectorized_elementwise_kernel (add):   3.101 ms/iter  (16 calls)
vectorized_elementwise_kernel (mul):   2.074 ms/iter  (10 calls)
```

130 small elementwise kernels per iteration. These include:
- Gaussian activations (sigmoid, exp, normalize)
- L1 loss computation
- Gradient clipping
- SH evaluation helper ops
- Tensor arithmetic in loss/backward

---

## Decision Gate Evaluation

| Criterion | Threshold | Measured | Triggered? |
|-----------|-----------|----------|------------|
| Backward > 50% | >50% | 48.0% (events) / 12.4% (profiler) | ❌ No |
| Projection > 30% | >30% | 0.2% | ❌ No |
| Intersection > 30% | >30% | 1.8% | ❌ No |
| None dominant | — | — | ✅ Yes |

### Protocol decision: **Perform roofline/memory analysis**

However, the data reveals a finding beyond the protocol's decision gate:

**The actual bottleneck is the D-SSIM loss at 40.7%**, which is not one of the three stages checked by the decision gate (backward, projection, intersection). The gsplat renderer itself accounts for only 18.5% of total GPU time.

---

## Key Findings

### Finding 1: Renderer is NOT the bottleneck

| What was expected | What was measured |
|-------------------|-------------------|
| Rendering (fwd+bwd) dominates | Rendering = **18.5%** of GPU time |
| Projection is expensive | Projection fwd = **0.2%** |
| Intersection is expensive | Intersection = **1.8%** |
| Backward rasterization dominates | rasterize_bwd = **11.7%** |

The gsplat CUDA kernels are highly optimized. With 1M Gaussians at 1080p, the entire forward render (project + intersect + sort + rasterize) takes only **5.4ms** (6.1% of total).

### Finding 2: D-SSIM loss is the #1 bottleneck

| D-SSIM component | Time/iter | % of total |
|------------------|-----------|------------|
| Forward conv (blur) | ~16ms | ~18% |
| Backward conv (dgrad) | ~20ms | ~23% |
| **D-SSIM total** | **35.8ms** | **40.7%** |

The D-SSIM loss uses cuDNN 2D convolutions for Gaussian blur, which are compute-intensive. With λ_dssim=0.2, the D-SSIM contributes 20% to the loss value but 40.7% to the compute cost.

### Finding 3: Kernel launch overhead is significant

| Category | Kernels/iter | Time/iter | % of total |
|----------|-------------|-----------|------------|
| elementwise_misc | 130 | 14.9ms | 17.0% |
| adam | 41 | 17.5ms | 19.8% |
| memory (fill) | 31 | 1.2ms | 1.4% |

**202 small kernels** (elementwise + adam + memory) consume **38.2%** of GPU time. Each kernel averages ~0.1ms, suggesting kernel launch overhead and memory bandwidth limitations rather than compute.

### Finding 4: Sorting is negligible

| Sort component | Time/iter | % of total |
|----------------|-----------|------------|
| CUB radix sort | 2.0ms | 2.3% |

Sorting 3.2M intersections costs only 2ms. This confirms C37-B's finding that sort optimization has negligible ROI.

---

## Implications for Optimization

### What NOT to optimize (confirmed by measurement)

| Direction | Measured cost | Reason |
|-----------|-------------|--------|
| Projection kernel | 0.2% | Already optimally fused in CUDA |
| Intersection kernel | 1.8% | Already fast tile-based approach |
| Sorting | 2.3% | CUB radix sort is near-optimal |
| Rasterize forward | 1.8% | 1 kernel, 1.6ms for 1M GS at 1080p |
| State reuse (C38/C39) | N/A | K=0, 98.5% dirty tiles |

### Where optimization opportunity EXISTS

| Direction | Potential savings | Difficulty |
|-----------|------------------|------------|
| **D-SSIM loss optimization** | **~35ms (40.7%)** | Medium — replace cudnn conv with separable blur or FFT |
| **Adam kernel fusion** | ~10ms (11.4%) | Medium — fuse 41 calls into fewer kernels |
| **Elementwise fusion** | ~10ms (11.4%) | Medium — fuse 130 small kernels |
| **rasterize_bwd** | ~5ms (5.7%) | Hard — requires CUDA kernel modification |

### D-SSIM optimization paths

The D-SSIM loss uses 3×3 (or larger) Gaussian blur via cuDNN convolution. Possible optimizations:

1. **Separable blur**: Replace 2D conv with two 1D convs (3× savings for 3×3 kernel)
2. **Box filter approximation**: Replace Gaussian with box filter (no conv needed, just averaging)
3. **Downsampled SSIM**: Compute SSIM at 1/2 or 1/4 resolution (4-16× savings)
4. **Skip D-SSIM**: Use L1-only loss (saves 40.7%, but may reduce quality)
5. **Precomputed blur**: Cache the blur kernel weights (minor savings)

### Adam optimization paths

1. **Fused Adam**: Use `torch.optim.Adam` with `foreach=False` or custom fused kernel
2. **Gradient clipping fusion**: Combine clip + Adam into single kernel
3. **Reduced parameter groups**: Merge SHs into single large tensor (reduces kernel launches)

---

## Comparison with C32-A (prior profiling)

| Metric | C32-A (3000 iter) | C40 (5000 iter) | Change |
|--------|-------------------|-----------------|--------|
| N Gaussians | ~500K (est.) | 1,000,684 | 2× |
| Forward (events) | 8.83ms | 6.86ms | -22% |
| Backward (events) | 22.33ms | 40.90ms | +83% |
| Optimizer (events) | 7.97ms | 13.81ms | +73% |
| Total (events) | 39.13ms | 85.27ms | +118% |
| Kernels/iter | 350 | ~260 | -26% |

The backward and optimizer times increased disproportionately with N (2× N → 1.8× backward, 1.7× optimizer). This is expected: Adam is O(N) per parameter, and rasterize_bwd processes more GS-pixel pairs.

---

## Decision

Per protocol: **No single stage dominant → perform roofline/memory analysis**

But the data shows the TRUE dominant cost is D-SSIM loss (40.7%), not a rendering stage. The protocol's decision gate doesn't account for loss computation as a potential bottleneck.

### Recommended next investigation

1. **C40-1: D-SSIM loss cost analysis** — Measure D-SSIM vs L1-only training. If D-SSIM can be simplified or approximated, ~35ms/iter (41%) can be saved.
2. **C40-2: Adam + elementwise fusion** — Measure potential of fusing 202 small kernels into fewer launches. Potential ~20ms/iter (23%) savings.
3. **Roofline analysis** for rasterize_bwd (the only gsplat kernel worth optimizing at 11.7%).

---

## Artifacts

- Experiment script: `scripts/phase-c31/c40_training_breakdown.py`
- Raw data: `results/phase-c31/c40_training_breakdown.json`
- Chrome trace: `results/phase-c31/c40_trace.json`
- Checkpoint: `results/epic05/phase7/phase7_room_30k_16/phase7_room_30k_16_iter5000.pt`
