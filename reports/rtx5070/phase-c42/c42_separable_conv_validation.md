# C42 Phase 1-B: Separable Gaussian Convolution Validation

## Executive Summary

| Metric | Result |
|--------|--------|
| **Correctness** | PASS — max_abs_error = 6.56e-07 (gate: <1e-5) |
| **Single blur forward** | 1.41x speedup (2.80ms -> 1.99ms) |
| **Full D-SSIM forward** | 0.87x — **REGRESSION** (16.47ms -> 19.02ms) |
| **Full D-SSIM fwd+bwd** | 1.30x speedup (46.32ms -> 35.65ms, 23% reduction) |
| **Estimated total training** | 9.8% speedup (89.5ms -> 80.7ms) |
| **Decision** | **MAYBE** — D-SSIM reduced 23% but total training impact below 20% threshold |

---

## 1. Correctness Result (Experiment A)

| Input type | max_abs_error | mean_abs_error | relative_error |
|-----------|---------------|----------------|----------------|
| Random (N(0,1)) | 5.36e-07 | 2.79e-08 | 8.96e-07 |
| Image [0,1] | 6.56e-07 | 7.49e-08 | — |

**Gate**: max_abs_error < 1e-5
**Result**: **PASS** — error is 15x below the gate threshold.

The separable convolution (`conv2d(x, kh[3,1,1,11]) → conv2d(result, kv[3,1,11,1])`) produces results numerically identical to the 2D convolution (`conv2d(x, kernel[3,1,11,11])`) to within float32 precision. The residual error (6.56e-07) is from summation order differences in floating-point arithmetic and is far below any threshold that would affect SSIM values or gradient quality.

---

## 2. Forward Speedup (Experiment B)

### B1: Single blur call

| Method | mean (ms) | std (ms) | min (ms) | max (ms) |
|--------|----------|---------|---------|---------|
| 2D conv [3,1,11,11] | 2.797 | 0.646 | 1.665 | 3.752 |
| Separable [3,1,1,11]+[3,1,11,1] | 1.987 | 0.547 | 1.272 | 2.937 |
| **Speedup** | **1.41x** | | | |

A single separable blur is 1.41x faster than a single 2D blur. Each 1D convolution processes fewer MACs (11 vs 121 per pixel), and the two sequential 1D passes (2.0ms total) are cheaper than one 2D pass (2.8ms).

### B3: Full D-SSIM forward (5 blur calls + elementwise)

| Method | mean (ms) | std (ms) |
|--------|----------|---------|
| 2D | 16.475 | 2.738 |
| Separable | 19.021 | 4.219 |
| **Speedup** | **0.87x (REGRESSION)** | |

**The full D-SSIM forward is 15.5% SLOWER with separable convolution.**

Root cause: The separable approach launches **10 conv kernels** (5 blur x 2 convs each) vs **5 conv kernels** (5 blur x 1 conv each). The extra 5 kernel launches and 5 intermediate tensor allocations (`[1,3,1080,1920]` = 23.7MB each, 118.5MB total extra memory traffic) outweigh the per-kernel FLOP savings.

### B4: Profiler kernel count (single blur, 10 profiled iters)

| Method | Kernel | ms/iter | calls/iter |
|--------|--------|---------|------------|
| 2D | `DepthwiseConv2d_cu::conv_depthwise2d_forward_kernel_generic` | 2.233 | 1.1 |
| Separable | `DepthwiseConv2d_cu::conv_depthwise2d_forward_kernel_generic` | 2.317 | 2.3 |

**Critical observation**: Both 2D and separable use the same PyTorch native kernel (`DepthwiseConv2d_cu`), NOT cuDNN. In the C41 training pipeline, the kernel was `cudnn::conv2d_grouped_direct`. This means the isolated benchmark uses a **different convolution backend** than the training pipeline. PyTorch's dispatcher selects native depthwise for this isolated context, while the training pipeline triggers cuDNN.

This backend discrepancy means the isolated speedup numbers are a **lower bound** — the actual pipeline behavior with cuDNN may differ.

---

## 3. Backward Speedup (Experiment C)

### C1: Single blur backward (fwd+bwd)

| Method | mean (ms) | std (ms) |
|--------|----------|---------|
| 2D | 7.109 | 0.825 |
| Separable | 6.376 | 1.544 |
| **Speedup** | **1.12x** | |

### C2: Full D-SSIM fwd+bwd (3 grad blur + 2 no-grad blur + elementwise)

| Method | mean (ms) | std (ms) |
|--------|----------|---------|
| 2D | 46.324 | 4.994 |
| Separable | 35.647 | 1.966 |
| **Speedup** | **1.30x (23% reduction)** | |

The backward benefits more from separability than the forward because:
- dgrad (backward convolution) is more expensive than forward conv (6.31ms vs 3.19ms per call in cuDNN), so per-kernel FLOP savings are proportionally larger
- The backward graph has fewer intermediate allocations (gradients flow through existing graph nodes)
- The 6 dgrad kernels (3 blur x 2 directions) vs 3 dgrad kernels is less overhead than the 10 vs 5 forward case

### C3: Profiler breakdown (full D-SSIM fwd+bwd, 10 profiled iters)

| Category | 2D ms/iter | 2D calls | Sep ms/iter | Sep calls | Conv speedup |
|----------|-----------|---------|------------|----------|-------------|
| **conv2d (forward+bwd)** | 35.256 | 9.6 | 20.632 | 19.2 | **1.71x** |
| elementwise | 20.661 | 60.0 | 21.726 | 60.0 | 0.95x (slightly worse) |
| reduction | 0.048 | 1.2 | 0.058 | 1.2 | — |
| **TOTAL** | **55.965** | **70.8** | **42.416** | **80.4** | **1.32x** |

**Convolution speedup: 1.71x** (35.3ms -> 20.6ms) — the separable approach significantly reduces convolution time.

**Elementwise: no improvement** (20.7ms -> 21.7ms, slightly worse) — the extra intermediate tensors from separable conv add ~1ms of elementwise overhead for memory operations.

**Kernel count: 70.8 -> 80.4** (+9.6 kernels/iter) — the separable approach adds ~10 extra kernel launches (5 forward + 5 backward), but each is cheaper.

### Convolution kernel detail

| Method | Kernel | ms/iter | calls/iter | ms/call |
|--------|--------|---------|------------|---------|
| 2D | `DepthwiseConv2d_cu` (fwd) | 21.175 | 6.0 | 3.53 |
| 2D | `DepthwiseConv2d_cu` (bwd) | 14.081 | 3.6 | 3.91 |
| Sep | `DepthwiseConv2d_cu` (fwd) | 13.109 | 12.0 | 1.09 |
| Sep | `DepthwiseConv2d_cu` (bwd) | 7.523 | 7.2 | 1.04 |

Each 1D convolution call costs ~1.04-1.09ms vs 3.53-3.91ms for 2D — a consistent **3.4-3.7x per-kernel speedup**. The total speedup is lower (1.71x) because twice as many kernels are needed.

---

## 4. Total D-SSIM Impact Estimation

### Isolated measurement (this benchmark)

| Metric | 2D | Separable | Speedup |
|--------|-----|-----------|---------|
| Single blur fwd | 2.80ms | 1.99ms | 1.41x |
| Single blur fwd+bwd | 7.11ms | 6.38ms | 1.12x |
| Full D-SSIM fwd | 16.47ms | 19.02ms | 0.87x (regression) |
| Full D-SSIM fwd+bwd | 46.32ms | 35.65ms | 1.30x |

### Pipeline impact estimate (from C41 data)

| Metric | Current (C41) | Estimated with separable | Change |
|--------|-------------|------------------------|-------|
| D-SSIM time | 38.0 ms/iter | 29.2 ms/iter | -8.8ms (-23%) |
| Total training | 89.5 ms/iter | 80.7 ms/iter | -8.8ms (-9.8%) |

**Estimation method**: `new_dssim = 38.0 / 1.30 = 29.2ms; new_total = 89.5 - 38.0 + 29.2 = 80.7ms`

### Caveats on pipeline estimate

1. **Backend discrepancy**: This benchmark uses PyTorch native `DepthwiseConv2d_cu`, while C41 measured cuDNN `conv2d_grouped_direct`. The pipeline may use a different backend for 1D convs, producing different speedup characteristics.

2. **Forward regression**: The 0.87x forward regression (from extra kernel launches + intermediate memory) may be less severe in the pipeline where memory reuse and CUDA stream overlap are more effective. Or it may be worse if the pipeline has less memory headroom.

3. **Memory overhead**: Separable conv allocates 5 intermediate tensors of [1,3,1080,1920] (23.7MB each) during forward. In the training pipeline with 1M Gaussians loaded, this 118.5MB overhead may cause memory pressure or cache eviction.

4. **The 9.8% estimate is a rough projection** — actual pipeline measurement is needed for a definitive number.

---

## 5. Decision: MAYBE

### Evidence Summary

| Criterion | Threshold | Result | Pass? |
|-----------|-----------|--------|-------|
| Correctness | max_abs_error < 1e-5 | 6.56e-07 | YES |
| D-SSIM reduction | >30% for KEEP, <15% for DROP | 23% | MAYBE (15-30% range) |
| Total training speedup | >20% for KEEP, <10% for DROP | 9.8% (estimated) | BORDERLINE DROP |
| Forward regression | None expected (identical math) | 0.87x (15.5% slower) | CONCERN |

### Why MAYBE, not DROP

1. **Convolution speedup is real**: 1.71x on the conv portion (35.3ms -> 20.6ms) proves the separable FLOP reduction translates to GPU time savings.

2. **Forward regression is overhead-bound, not algorithm-bound**: The 0.87x forward regression comes from extra kernel launches and intermediate memory allocation — issues that may be mitigated by torch.compile (Candidate C) or CUDA Graphs in the pipeline.

3. **Backend discrepancy**: The isolated benchmark uses native depthwise, not cuDNN. Pipeline behavior may differ — possibly better (if cuDNN 1D is even slower than native 1D, the switch to native would help more) or worse (if cuDNN 1D is faster than native 1D).

4. **Combined potential**: If forward overhead is mitigated (e.g., by torch.compile fusing the intermediate operations), the D-SSIM speedup could improve from 1.30x to 1.71x, yielding ~17ms savings (19% total training speedup — close to the KEEP threshold).

### Why MAYBE, not KEEP

1. **Total training speedup (9.8%) is below the 20% threshold** — even with the optimistic estimate, this alone doesn't meet the KEEP gate.

2. **Forward regression (0.87x) is a red flag** — in the training pipeline, the forward pass is part of every iteration. A 15.5% forward regression partially offsets the 1.30x backward speedup.

3. **Pipeline validation needed** — the isolated benchmark cannot confirm the actual training speedup due to the backend discrepancy and memory behavior differences.

### Conditions for KEEP

The separable convolution candidate should be **re-evaluated in the full training pipeline** before a final decision. Specifically:

1. **Pipeline-level measurement**: Apply separable conv in `loss.py`, run the C41 profiling configuration (same checkpoint, same scene, 10 profiled iterations), and measure the actual D-SSIM time and total iteration time.

2. **Combined with torch.compile (Candidate C)**: Test B+C together to determine whether torch.compile can mitigate the forward regression by fusing the intermediate elementwise operations and reducing launch overhead.

3. **If pipeline B+C total speedup >15%**: Upgrade to KEEP.
4. **If pipeline B alone total speedup >15%**: Upgrade to KEEP.
5. **If pipeline B alone total speedup <10%**: Downgrade to DROP.

---

## 6. Prior-Art Boundary

Separable Gaussian filtering is a **textbook DSP technique** (Oppenheim & Schafer, 1975). Applying it to SSIM computation is a **known engineering optimization** in image processing. This is **not novel** — the original 3DGS implementation (graphdeco-inria) uses 2D convolution as an inherited implementation choice, not a deliberate algorithmic decision.

This candidate is evaluated solely as an **engineering optimization component** of C42, not as a research contribution. No novelty is claimed.

---

## Artifacts

- Benchmark script: `scripts/phase-c42/c42_separable_conv_benchmark.py`
- Raw data: `results/phase-c42/c42_separable_conv_data.json`
- Profiler traces: `results/phase-c42/c42_b_trace_tmp.json`
- C41 reference data: `results/phase-c31/c41_gpu_utilization.json`
- D-SSIM source: `scripts/epic05/phase7/loss.py`
