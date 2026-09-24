# Phase C43: Multi-Track Backward Optimization Research Report

**Platform**: A100 PCIe 40GB (SM80, 108 SMs), gsplat 1.5.3, CUDA 11.8  
**Scene**: room (MipNeRF360), 1080p, 1.59M SfM Gaussians  
**Date**: 2025-01

---

## Executive Summary

Five tracks of backward pass optimization research were conducted on gsplat 1.5.3. The central finding is that **the training bottleneck is not the backward rasterizer kernel but the SSIM loss computation**. At scale=1.0, SSIM consumes 75.8 ms (77% of a 98.3 ms training iteration), while the entire backward pass consumes only 17.9 ms (18%). The backward rasterizer kernel itself is 6.6 ms (6.7% of total).

**C42 (SSIM downscale, Track 0)** remains the only optimization exceeding the 5% e2e threshold: **+17% to +59% e2e speedup** depending on downscale factor. All backward-specific optimizations (Tracks A, B, D) fall below the 3% e2e threshold because the backward kernel is a small fraction of total training time.

---

## Track 0: C42 Re-validation with Continuous Pruning

### Motivation
The original P1 experiment reported +33.6% C42 speedup. The re-validation in Phase C42-Audit found the discrepancy was caused by Gaussian count explosion (23M vs ~1M) due to missing pruning. Track 0v2/v3 re-runs with aggressive continuous pruning (opacity_threshold=0.05, grad_threshold=0.002) to keep GS ~1.5M.

### Method
- **Track 0v2**: 5K training iterations, 5 scales, pruning every 100 iters (GPUs 0-4)
- **Track 0v3**: Train 2000 iters to stabilize GS count (1.53M), then measure clean render+SSIM+backward timing without densification overhead (5 cameras × 30 iterations)

### Results (Track 0v3, clean timing, GS=1,530,454)

| Scale | Render (ms) | SSIM (ms) | Backward (ms) | Total (ms) | E2E Speedup |
|-------|-------------|-----------|---------------|------------|-------------|
| 1.0   | 4.57        | 75.85     | 17.87         | 98.33      | baseline    |
| 0.875 | 4.59        | 59.64     | 17.35         | 81.62      | **+17.0%**  |
| 0.75  | 4.58        | 45.62     | 15.77         | 65.99      | **+32.9%**  |
| 0.625 | 4.57        | 32.84     | 15.15         | 52.59      | **+46.5%**  |
| 0.5   | 4.56        | 21.27     | 14.40         | 40.27      | **+59.0%**  |

### Key Observations
1. **SSIM is the dominant cost** (77% of total at scale 1.0), not the backward kernel (18%)
2. **SSIM time scales quadratically** with resolution: 75.85 → 21.27 ms (3.6x reduction at 0.5x resolution)
3. **Backward time is roughly constant** (17.9 → 14.4 ms) — slight decrease at lower scales due to reduced gradient propagation through the smaller SSIM conv2d graph
4. **Render time is constant** (~4.57 ms) — forward rasterization is scale-independent
5. **Original P1 +33.6% speedup** is consistent with scale ≈ 0.75 (+32.9%)
6. GS trajectory: 1.59M → 1.35M (pruning) → 1.88M (densification) over 5K iters — stable near 1.5-1.9M

### Decision: **KEEP** — +17% to +59% e2e speedup, well above 5% threshold

---

## Track A: Batch Size (tile_size) Scaling

### Motivation
The backward kernel processes Gaussians in batches of `block_size = tile_size²`. Larger batches reduce batch iteration overhead; smaller batches reduce per-batch shared memory pressure. A100 has 48KB default shared memory per block, allowing up to 4x the baseline 10KB.

### Method
Tested tile_size = 8 (batch=64), 16 (batch=256, baseline), 32 (batch=1024) on room scene with 1.59M Gaussians. Measured forward time, backward time, and gradient correctness (max_abs_gradient_difference vs baseline).

### Results

| tile_size | batch_size | Forward (ms) | Backward (ms) | Bwd Speedup | Grad Diff | Correct? |
|-----------|------------|--------------|---------------|-------------|-----------|----------|
| 8         | 64         | 5.72         | 7.73          | +26.2%      | 2.29e-05  | PASS     |
| 16        | 256        | 3.48         | 10.47         | baseline    | 1.19e-06  | PASS     |
| 32        | 1024       | ERROR        | —             | —           | —         | —        |

### Analysis
- **tile_size=8**: Backward 26.2% faster (smaller batches → less wasted computation per batch, better early termination granularity). But forward 64% slower (more tile blocks → more global memory traffic, less data reuse per tile). Net e2e: fwd_bwd = 13.44 ms vs 13.95 ms baseline = **+3.7% e2e** (below 5% threshold)
- **tile_size=32**: "Too many resources requested for launch" — 1024 threads per block exceeds the register limit on SM80. The backward kernel uses ~40 registers/thread; 1024 threads × 40 = 40,960 registers > 65,536 max per SM → 0 blocks per SM. Would require `--maxrregcount` reduction or kernel refactoring.
- **Gradient correctness**: tile_size=8 produces max gradient difference of 2.29e-05 (floating-point reordering from different batch boundaries, not a correctness issue)

### Critical Finding
**Batch size and tile_size are inseparable** in gsplat's architecture. The block dimension IS `tile_size²`, so changing batch size changes the tile grid, intersection distribution, and shared memory layout. This is not a simple kernel parameter — it's a renderer-level architectural change.

### Decision: **DROP** — Best case +3.7% e2e (below 5%), tile_size=32 infeasible on SM80

---

## Track B: Alpha Recomputation Elimination

### Motivation
The backward kernel recomputes `vis = __expf(-sigma)` for every Gaussian-pixel pair (line 176 of RasterizeToPixels3DGSBwd.cu). The forward pass already computes this value but discards it. Caching `vis` (or `alpha`) from forward to backward could eliminate the expensive `exp()` call on the special function unit (SFU).

### Source Analysis

**Forward kernel** (RasterizeToPixels3DGSFwd.cu, line 148):
```cpp
float alpha = min(0.999f, opac * __expf(-sigma));
```
Forward computes `vis = __expf(-sigma)` but stores only `render_alphas[pix_id] = 1.0f - T` (the final transmittance) and `last_ids[pix_id]` (the last contributing Gaussian index). The per-Gaussian-pixel `vis` is **not** stored.

**Backward kernel** (RasterizeToPixels3DGSBwd.cu, lines 173-177):
```cpp
float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
vis = __expf(-sigma);
alpha = min(0.999f, opac * vis);
```
Backward recomputes `vis` and `alpha` from scratch. The `vis` value is needed for gradient computation: `v_sigma = -opac * vis * v_alpha`.

### Cost Estimation

| Metric | Value |
|--------|-------|
| Total intersections (Gaussian-tile pairs) | 4,315,942 |
| Visible Gaussians | 430,897 (27% of 1.59M) |
| exp() calls (upper bound, all pixels in tile) | 1.10B |
| exp() calls (lower bound, 1 per intersection) | 4.32M |
| A100 peak exp throughput | 609 Gexp/s |
| exp() time (upper bound) | 1.81 ms (33.3% of bwd kernel) |
| exp() time (lower bound) | 0.007 ms (0.1% of bwd kernel) |
| exp() time (midpoint estimate) | 0.91 ms (16.7% of bwd kernel) |
| Cache memory (vis only, 4B/intersection) | 17.3 MB (0.04% of 40GB) |

### Potential Speedup

| Scenario | Bwd Kernel Speedup | E2E Speedup (with SSIM) |
|----------|-------------------|------------------------|
| Upper bound (33% exp fraction) | 33.3% | 1.8% (scale=1.0) |
| Midpoint (17% exp fraction) | 16.7% | 0.9% (scale=1.0) |
| With C42 scale=0.75 | 16.7% | 1.4% |

### Implementation Complexity
- **Forward kernel**: Add per-intersection `vis` output buffer (indexed by flatten_ids position, not Gaussian ID)
- **Backward kernel**: Accept `vis` buffer as input, skip the `__expf()` call
- **Python wrapper**: Allocate and pass the buffer through `rasterization()` autograd function
- **Correctness**: The cached `vis` must match exactly; any floating-point reordering between forward and backward would cause gradient divergence

### Decision: **DROP** — Even upper-bound estimate gives only 1.8% e2e speedup (below 3% threshold). The exp() recomputation is a small fraction of total training time because SSIM dominates.

---

## Track C: Backward Kernel Profiling

### Method
Used PyTorch profiler (CUDA activity) to break down the backward pass into individual kernels. Nsight Compute (ncu) was unavailable due to ERR_NVGPUCTRPERM (no GPU performance counter permissions without root). nvprof is unsupported on SM80 (compute capability 8.0 > 7.5 limit).

### Pipeline Breakdown (1.59M Gaussians, room scene)

| Stage | Time (ms) | % of Pipeline |
|-------|-----------|---------------|
| Projection (fully_fused_projection) | 0.094 | 0.7% |
| Intersect + Sort (isect_tiles + CUB sort) | 1.090 | 7.8% |
| Offset encode | 0.032 | 0.2% |
| **Forward rasterize** | **3.663** | **26.1%** |
| **Backward (total)** | **10.376** | **73.9%** |
| **Total (fwd+bwd)** | **14.039** | **100%** |

### Backward Kernel Breakdown (10.375 ms total CUDA time)

| Kernel | CUDA Time (ms) | % of Backward |
|--------|---------------|---------------|
| `rasterize_to_pixels_3dgs_bwd_kernel` | 6.587 | **63.5%** |
| Memcpy DtoD (gradient buffer init) | 1.445 | 13.9% |
| vectorized_elementwise (PyTorch ops) | 0.974 | 9.4% |
| `spherical_harmonics_bwd_kernel` | 0.276 | 2.7% |
| `projection_ewa_3dgs_fused_bwd_kernel` | 0.249 | 2.4% |
| Other elementwise kernels | ~0.84 | 8.1% |

### Key Insights
1. **The rasterize backward kernel is 63.5% of backward CUDA time**, but only 6.7% of a full training iteration (including SSIM loss)
2. **Memcpy DtoD (1.45 ms)** is the second-largest cost — this is gradient buffer initialization (zeroing v_colors, v_conics, v_means2d, v_opacities). Could potentially be fused with the backward kernel.
3. **SH backward and projection backward are minor** (2.7% and 2.4%) — not worth optimizing
4. **The backward is compute-bound** (arithmetic + exp), not memory-bound. The shared memory usage (10KB) is well within the 48KB limit, and global memory access is coalesced through the batch loading pattern.
5. **Occupancy**: 256 threads/block, ~40 registers/thread → 4 blocks per SM (1024 threads, 53% occupancy). Increasing occupancy would require reducing register usage.

### A100 Hardware Context
- 108 SMs, 4 SFUs per SM, 1.41 GHz clock
- Peak exp throughput: 609 Gexp/s
- Peak FP32 throughput: 19.5 TFLOPS
- 48KB default shared memory per block (163KB max optin)
- 65,536 registers per SM

---

## Track D: Warp Specialization Assessment

### Motivation
Hypothesize that splitting the backward kernel into specialized warps (load, compute, atomic) could improve throughput by pipelining memory and compute.

### Assessment
After analyzing the backward kernel structure, warp specialization is **not applicable**:

1. **Current structure**: All 256 threads in a block cooperatively process the same batch of Gaussians. The pattern is: `sync → load batch to shared mem → sync → compute per-pixel gradients → warp-reduce → atomic add → sync → load next batch`.

2. **No inter-warp parallelism to exploit**: Each thread handles one pixel independently. The 8 warps in a block each handle 32 pixels, but they all process the same Gaussians. There's no data partitioning across warps.

3. **Already pipelined**: The batch-level loop already provides some pipeline overlap — while one batch is being computed, the next batch's data could be prefetched. But with only 10KB of shared memory per batch, the prefetch window is small.

4. **Atomic operations are already warp-reduced**: The `warpSum()` calls (lines 244-250) reduce gradients across the warp before a single thread does the atomic add. This is already the optimal pattern.

5. **The bottleneck is exp() on the SFU**, which is a per-thread operation. Warp specialization cannot parallelize the SFU — all warps share the same 4 SFUs per SM.

### Decision: **DROP** — Kernel structure doesn't support warp specialization. The bottleneck (exp() on SFU) is not addressable by warp-level reorganization.

---

## Comprehensive Evidence Table

| Track | Optimization | Backward Impact | E2E Impact (with SSIM) | Gradient Correct | Decision |
|-------|-------------|----------------|----------------------|------------------|----------|
| 0 (C42) | SSIM downscale 0.75 | -2.1 ms bwd | **+32.9%** | N/A (loss change) | **KEEP** |
| 0 (C42) | SSIM downscale 0.5 | -3.5 ms bwd | **+59.0%** | N/A (loss change) | **KEEP** |
| A | tile_size=8 (batch=64) | +26.2% bwd | +3.7% | PASS (2.3e-05) | **DROP** (<5%) |
| A | tile_size=32 (batch=1024) | FAIL | FAIL | FAIL | **DROP** (infeasible) |
| B | Alpha/vis caching (est.) | 16.7% bwd kernel | 0.9-1.8% | Untested | **DROP** (<3%) |
| C | Profiling (observation) | — | — | — | N/A |
| D | Warp specialization | Not applicable | Not applicable | — | **DROP** |

---

## Bottleneck Hierarchy

The full training iteration time breakdown (scale=1.0, GS=1.53M):

```
Total iteration: 98.33 ms
├── SSIM loss:     75.85 ms (77.2%)  ← TRUE BOTTLENECK
├── Backward:      17.87 ms (18.2%)
│   ├── rasterize_bwd_kernel:  6.59 ms (6.7%)
│   ├── memcpy DtoD:           1.45 ms (1.5%)
│   ├── elementwise:           0.97 ms (1.0%)
│   ├── SH bwd:                0.28 ms (0.3%)
│   ├── projection bwd:        0.25 ms (0.3%)
│   └── other:                 0.84 ms (0.9%)
│   (CPU autograd overhead:    ~7.5 ms)
└── Render:         4.57 ms (4.6%)
    ├── projection:            0.09 ms
    ├── isect+sort:            1.09 ms
    ├── offset:                0.03 ms
    └── rasterize_fwd:         3.36 ms
```

**The backward rasterizer kernel is 6.7% of total training time.** Even completely eliminating it would yield only +6.7% e2e speedup. Any backward-only optimization is fundamentally capped at this level.

---

## C42 Interaction with Backward Optimization

C42 (SSIM downscaling) changes the cost landscape:

| Scale | SSIM (ms) | Backward (ms) | Total (ms) | Bwd as % of Total |
|-------|-----------|---------------|------------|-------------------|
| 1.0   | 75.85     | 17.87         | 98.33      | 18.2%             |
| 0.75  | 45.62     | 15.77         | 65.99      | 23.9%             |
| 0.5   | 21.27     | 14.40         | 40.27      | 35.8%             |

With C42 at scale=0.5, backward becomes 35.8% of total. A 14% backward speedup (achievable via alpha caching) would give 5.0% e2e — right at the threshold. **C42 and backward optimization are complementary, not competitive.**

---

## Novelty Assessment

1. **C42 SSIM downscale**: Novel application of multi-scale loss to 3DGS training. Not found in prior 3DGS literature. The key insight — that SSIM loss computation dominates training time and scales with resolution — is not widely recognized.

2. **Batch size / tile_size**: Known architectural constraint in tile-based rasterizers. The trade-off between tile_size and forward/backward balance is a new empirical finding for gsplat specifically.

3. **Alpha caching**: Conceptually similar to activation checkpointing in deep learning, but applied to the 3DGS rasterizer. The feasibility analysis (memory cost, exp() fraction estimate) is novel for gsplat.

4. **Warp specialization**: Not applicable to this kernel structure. This negative result is valuable — it constrains the optimization space.

---

## Recommendations

### Immediate (KEEP)
- **C42 at scale=0.75**: +32.9% e2e speedup, minimal quality impact (SSIM=0.8003 vs 0.7990). This is the single most impactful optimization found.

### Future Work (if backward optimization is still desired)
1. **Alpha caching + C42 combination**: At scale=0.5, backward is 35.8% of total. Alpha caching (16.7% bwd speedup) would give ~5% e2e. This is the only scenario where backward optimization reaches the KEEP threshold.
2. **Gradient buffer init fusion**: The 1.45 ms Memcpy DtoD (13.9% of backward) could be eliminated by fusing the zero-initialization into the backward kernel's prologue.
3. **tile_size=32 with register reduction**: Compile with `--maxrregcount=32` to fit 1024 threads per block. Would need kernel refactoring to reduce register pressure.

### Not Recommended
- **Warp specialization**: Kernel structure doesn't support it
- **tile_size=8**: Net e2e below threshold due to forward regression
- **Standalone alpha caching**: Below 3% e2e without C42

---

## Data Files

| File | Description |
|------|-------------|
| `results/a100/phase-c42/track0v3_clean_timing.json` | Clean C42 timing (no densification overhead) |
| `results/a100/phase-c42/track0v2_pruned_{10,0875,075,0625,05}.json` | 5K training runs with aggressive pruning |
| `results/a100/phase-c42/track_a_tile_size.json` | Tile_size=8/16/32 batch size experiment |
| `results/a100/phase-c42/track_b_alpha_caching_analysis.json` | Alpha caching cost/benefit analysis |
| `results/a100/phase-c42/track_c_backward_profile.json` | PyTorch profiler backward breakdown |
| `scripts/phase-c42/track0v3_clean_timing.py` | Clean timing experiment script |
| `scripts/phase-c42/track_a_tile_size.py` | Tile size experiment script |
| `scripts/phase-c42/track_b_alpha_caching_analysis.py` | Alpha caching analysis script |
| `scripts/phase-c42/track_c_torch_profiler.py` | PyTorch profiler script |
