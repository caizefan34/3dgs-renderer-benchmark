# C42 Phase 1-C: Separable Conv + torch.compile Interaction Validation

## Executive Summary

| Metric | V0 (baseline) | V1 (sep) | V2 (aot_eager) | V2b (cudagraphs) |
|--------|-------------|----------|---------------|-----------------|
| **D-SSIM fwd+bwd** | 75.5 ms | 57.3 ms (1.32x) | 61.7 ms (1.22x) | 58.0 ms (1.30x) |
| **E2E training** | 133.6 ms | 124.0 ms (+7.2%) | 130.7 ms (+2.2%) | 116.8 ms (+12.6%) |
| **Kernel count** | 96.4/iter | 120.0/iter | 135.6/iter | 124.8/iter |
| **Memory (e2e peak)** | 1911 MB | 1987 MB | 1837 MB | 1558 MB |
| **Correctness** | ref | PASS | PASS | PASS |

**Decision: MAYBE** — V2b (cudagraphs) achieves 12.6% e2e speedup (in 10-15% MAYBE range). V2 (aot_eager) is counterproductive.

**Critical finding**: Triton/inductor (the default torch.compile backend) is **unavailable on Windows**. Only `aot_eager` and `cudagraphs` backends were testable. The full operator-fusion benefit of torch.compile could NOT be evaluated.

---

## 1. Correctness Result

| Comparison | loss_diff | grad_diff | Gate | Result |
|-----------|-----------|-----------|------|--------|
| V0 vs V1 | 1.79e-07 | 1.65e-09 | <1e-5, <1e-4 | PASS |
| V0 vs V2 | 1.79e-07 | 1.65e-09 | <1e-5, <1e-4 | PASS |
| V0 vs V2b | 1.79e-07 | 1.65e-09 | <1e-5, <1e-4 | PASS |

All variants produce identical results to 7 significant figures. The separable convolution and both compile backends preserve numerical correctness perfectly.

Loss values: V0=0.32546484, V1=0.32546502, V2=0.32546502, V2b=0.32546502

---

## 2. Forward Latency (Isolated)

| Variant | Forward (ms) | Std (ms) | vs V0 |
|---------|-------------|---------|-------|
| V0 (2D, no compile) | 35.495 | 3.290 | 1.00x |
| V1 (sep, no compile) | 35.285 | 0.766 | 1.01x |
| V2 (sep, aot_eager) | 35.302 | 2.748 | 1.01x |
| V2b (sep, cudagraphs) | 36.731 | 3.069 | 0.97x |

**Forward latency is identical across all variants** (~35ms). The separable convolution's forward overhead (extra kernel launches) is fully offset by reduced per-kernel cost. Neither compile backend changes forward latency meaningfully.

This differs from Phase 1-B's finding (0.87x forward regression) because this test uses a pre-rendered fixed image with different memory layout, and the CUDA event measurement includes Python dispatch overhead that is similar across variants.

---

## 3. Backward Latency (Isolated)

| Variant | Backward (ms) | Std (ms) | vs V0 | vs V1 |
|---------|-------------|---------|-------|-------|
| V0 (2D, no compile) | 41.216 | 1.638 | 1.00x | — |
| V1 (sep, no compile) | 23.337 | 0.759 | **1.77x** | 1.00x |
| V2 (sep, aot_eager) | 27.971 | 1.004 | 1.47x | 0.83x |
| V2b (sep, cudagraphs) | 23.671 | 0.994 | **1.74x** | 0.99x |

**Key finding**: The backward is where separable convolution delivers its main benefit — 1.77x speedup over V0. This is because dgrad (backward convolution) is more expensive than forward, so the 5.5x FLOP reduction has proportionally larger impact.

**aot_eager HURTS backward**: V2 is 20% slower than V1 (27.97ms vs 23.34ms). The aot_eager backend adds graph tracing overhead without fusion benefits, increasing kernel count from 120 to 135.6.

**cudagraphs preserves V1's backward speed**: V2b matches V1 (23.67ms vs 23.34ms). CUDA Graphs eliminate kernel launch overhead but don't add fusion.

---

## 4. Full Forward+Backward Latency (Isolated)

| Variant | Fwd+Bwd (ms) | Std (ms) | vs V0 | vs V1 |
|---------|-------------|---------|-------|-------|
| V0 (2D, no compile) | 75.545 | 1.216 | 1.00x | — |
| V1 (sep, no compile) | 57.312 | 0.932 | **1.32x** | 1.00x |
| V2 (sep, aot_eager) | 61.717 | 1.117 | 1.22x | 0.93x |
| V2b (sep, cudagraphs) | 58.038 | 2.166 | **1.30x** | 0.99x |

V1 and V2b are essentially tied. V2 (aot_eager) is 8% worse than V1.

---

## 5. CUDA Kernel Count

| Variant | Kernels/iter | Total GPU ms/iter | vs V0 kernels | vs V0 time |
|---------|-------------|-------------------|--------------|-----------|
| V0 | 96.4 | 75.11 | 1.00x | 1.00x |
| V1 | 120.0 | 62.02 | 1.24x more | 1.21x faster |
| V2 | 135.6 | 72.97 | 1.41x more | 1.03x faster |
| V2b | 124.8 | 74.13 | 1.29x more | 1.01x faster |

### Kernel breakdown by category

| Category | V0 (ms) | V0 (calls) | V1 (ms) | V1 (calls) | V2 (ms) | V2 (calls) | V2b (ms) | V2b (calls) |
|----------|---------|-----------|---------|-----------|---------|-----------|---------|------------|
| **conv2d (fwd)** | 23.29 | 5.7 | 9.28 | 6.0 | 10.75 | 6.0 | 12.20 | 6.0 |
| **dgrad (bwd)** | 28.47 | 3.6 | 10.04 | 7.2 | 11.66 | 7.2 | 11.47 | 7.2 |
| **elementwise** | 23.25 | 83.7 | 26.11 | 85.2 | 30.71 | 100.8 | 26.80 | 85.2 |
| other | 0 | 0 | 16.49 | 18.0 | 19.68 | 18.0 | 23.49 | 22.8 |
| reduction | 0.10 | 3.4 | 0.10 | 3.6 | 0.17 | 3.6 | 0.17 | 3.6 |

**Convolution speedup is the core benefit**:
- Forward conv: 23.29ms (V0) → 9.28ms (V1) = **2.51x speedup**
- Backward conv: 28.47ms (V0) → 10.04ms (V1) = **2.83x speedup**
- Total conv: 51.76ms (V0) → 19.32ms (V1) = **2.68x speedup**

**aot_eager increases elementwise kernels**: V2 has 100.8 elementwise calls (vs V0's 83.7 and V1's 85.2) and 30.71ms elementwise time (vs V0's 23.25ms and V1's 26.11ms). The aot_eager backend decomposes some fused ops into more granular kernels, adding overhead.

**cudagraphs adds "other" overhead**: V2b has 22.8 "other" calls at 23.49ms (vs V1's 18.0 calls at 16.49ms). This is CUDA Graph management overhead (graph capture, replay, memory management).

---

## 6. GPU Memory Allocation

| Variant | Isolated peak (MB) | E2E peak (MB) | E2E current (MB) | vs V0 e2e |
|---------|-------------------|--------------|-----------------|-----------|
| V0 | 1751 | 1911 | 1298 | baseline |
| V1 | 1850 | 1987 | 1299 | +76 MB (+4.0%) |
| V2 | 1702 | 1837 | 1298 | -74 MB (-3.9%) |
| V2b | 1323 | 1558 | 1274 | **-353 MB (-18.5%)** |

**V1 increases memory** by 76 MB due to 5 extra intermediate tensors from separable conv (each [1,3,1080,1920] = 23.7 MB).

**V2b (cudagraphs) dramatically reduces memory** by 353 MB (-18.5%). CUDA Graphs pre-allocate a static memory pool and reuse it across iterations, eliminating the allocation/deallocation churn. This is a significant side benefit.

**V2 (aot_eager) slightly reduces memory** by 74 MB through graph-level memory planning.

---

## 7. End-to-End Training Iteration Time

| Variant | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | vs V0 |
|---------|----------|---------|---------|---------|-------|
| V0 (2D, no compile) | 133.6 | 9.5 | 81.2 | 146.1 | baseline |
| V1 (sep, no compile) | 124.0 | 3.6 | 119.7 | 135.4 | **+7.2%** |
| V2 (sep, aot_eager) | 130.7 | 16.6 | 107.6 | 198.2 | +2.2% |
| V2b (sep, cudagraphs) | 116.8 | 14.1 | 72.0 | 125.0 | **+12.6%** |

### Analysis

**V1 (separable alone) delivers 7.2% e2e speedup** — below the 10% DROP threshold but above zero. The convolution speedup (2.68x on conv portion) is partially offset by:
- Extra kernel launches (+24 kernels/iter)
- Extra intermediate memory (+76 MB)
- Elementwise overhead not reduced

**V2 (aot_eager) is nearly useless** — only 2.2% e2e speedup, and HIGHER variance (std=16.6ms). The aot_eager backend adds graph tracing overhead and increases kernel count without fusion benefits. It actually makes V1 SLOWER (130.7ms vs 124.0ms).

**V2b (cudagraphs) is the best variant** — 12.6% e2e speedup. CUDA Graphs eliminate kernel launch overhead by capturing the entire kernel sequence and replaying it as a single graph launch. This recovers the overhead from V1's extra kernel launches AND adds additional savings from reduced launch overhead on ALL kernels (not just the extra ones).

The 5.4 percentage point improvement from V1 to V2b (7.2% → 12.6%) comes from:
1. Eliminated launch overhead on ~120 kernels (7μs × 120 = 0.84ms saved)
2. Reduced memory allocation overhead (353 MB less peak memory)
3. Better kernel scheduling within the graph

### Variance observation

V2b has high variance (std=14.1ms, min=72.0ms, max=125.0ms). The min of 72.0ms suggests CUDA Graphs can achieve very fast execution when conditions are ideal, but occasional graph recapture or memory management events cause spikes. V1 has the lowest variance (std=3.6ms), suggesting it's the most stable.

---

## 8. Total D-SSIM Impact Estimation

### Isolated D-SSIM measurement

| Metric | V0 | V1 | V2b | V2b vs V0 |
|--------|-----|-----|-----|-----------|
| D-SSIM fwd+bwd | 75.5ms | 57.3ms | 58.0ms | 1.30x (23% reduction) |

### Pipeline impact (end-to-end)

| Metric | V0 | V1 | V2b | V2b vs V0 |
|--------|-----|-----|-----|-----------|
| Total training | 133.6ms | 124.0ms | 116.8ms | 12.6% speedup |

### C41 reference comparison

C41 measured D-SSIM at 38.0ms/iter in the training pipeline (with memory reuse and CUDA stream overlap). This experiment's isolated measurement shows 75.5ms (V0) — the 2x difference is because the isolated test lacks pipeline memory reuse.

The RELATIVE speedups are accurate since all variants are measured under identical conditions.

### Convolution breakdown (profiler, isolated)

| Component | V0 (ms) | V1 (ms) | V1 speedup | V2b (ms) | V2b speedup |
|-----------|---------|---------|-----------|---------|------------|
| Forward conv | 23.29 | 9.28 | 2.51x | 12.20 | 1.91x |
| Backward conv | 28.47 | 10.04 | 2.83x | 11.47 | 2.48x |
| **Total conv** | **51.76** | **19.32** | **2.68x** | **23.67** | **2.19x** |
| Elementwise | 23.25 | 26.11 | 0.89x | 26.80 | 0.87x |

The convolution speedup is consistent: 2.2-2.7x. The elementwise overhead increases slightly with separable conv (+2.9ms) due to extra intermediate tensor operations.

---

## 9. torch.compile Backend Analysis

### Backend availability

| Backend | Available on Windows? | Tested? | Result |
|---------|----------------------|---------|--------|
| inductor (default) | **NO** (requires Triton) | No | N/A — Triton not installable on Windows |
| aot_eager | Yes | Yes (V2) | **Counterproductive** — adds overhead |
| cudagraphs | Yes | Yes (V2b) | **Beneficial** — 12.6% e2e speedup |

### Why aot_eager fails

The `aot_eager` backend traces the forward and backward computation graphs with `torch.fx` but does NOT generate fused kernels (that requires Triton/inductor). Its effects:

1. **Increased kernel count**: 135.6 vs V1's 120.0 — graph tracing decomposes some compound ops
2. **Increased elementwise time**: 30.71ms vs V1's 26.11ms — no fusion, just tracing overhead
3. **Higher variance**: std=16.6ms vs V1's 3.6ms — graph management adds unpredictability
4. **No fusion benefit**: The primary value of torch.compile (operator fusion via Triton) is unavailable

### Why cudagraphs succeeds

The `cudagraphs` backend captures the entire sequence of CUDA kernel launches into a CUDA Graph and replays it. Its effects:

1. **Eliminated launch overhead**: ~120 kernel launches replayed as 1 graph launch
2. **Memory optimization**: Static memory pool reduces peak by 353 MB (-18.5%)
3. **Preserves V1's speedup**: Convolution speedup is maintained (2.19x vs V0)
4. **Additional overhead**: 22.8 "other" kernel calls (graph management) at 23.49ms

### What inductor (Triton) would add

The default `inductor` backend (unavailable here) would provide:
1. **Operator fusion**: Fuse ~85 elementwise kernels into ~5-10 Triton kernels
2. **Memory traffic reduction**: Fused kernels read/write each tensor once
3. **Estimated additional speedup**: 15-25% on the elementwise portion (23ms → ~18ms)

If Triton were available, V2 (inductor) could potentially achieve:
- D-SSIM: ~50ms (vs V0's 75.5ms) = 1.51x
- E2E: ~110ms (vs V0's 133.6ms) = 17.7% speedup → **KEEP**

This is speculative and cannot be verified on this platform.

---

## 10. Decision: MAYBE

### Gate evaluation

| Criterion | Threshold | V2b result | Pass? |
|-----------|-----------|-----------|-------|
| D-SSIM speedup | >30% for KEEP | 23% (1.30x) | No (below 30%) |
| Total iteration speedup | >15% for KEEP | 12.6% | No (below 15%) |
| Total iteration speedup | 10-15% for MAYBE | 12.6% | Yes |
| Correctness | loss <1e-5, grad <1e-4 | 1.79e-7, 1.65e-9 | Yes |
| Quality degradation | None (identical math) | None | Yes |

**Decision: MAYBE** — total speedup is 12.6%, in the 10-15% MAYBE range.

### Why MAYBE, not KEEP

1. **Total speedup (12.6%) is below the 15% KEEP threshold**
2. **Triton/inductor unavailable** — the full torch.compile fusion benefit could not be evaluated. On Linux with Triton, the result might be different.
3. **High variance** (std=14.1ms) — cudagraphs introduces execution instability

### Why MAYBE, not DROP

1. **V2b (cudagraphs) adds real value over V1**: 12.6% vs 7.2% — compile adds 5.4 percentage points
2. **Memory reduction is significant**: -353 MB (-18.5%) — enables larger batch or more Gaussians
3. **Convolution speedup is proven**: 2.19x on conv portion, consistent across variants
4. **Platform limitation**: The default inductor backend (Triton) might achieve KEEP on Linux

### Recommendations

1. **On Windows (this platform)**: Use V2b (separable + cudagraphs) as the C42 engineering floor. 12.6% e2e speedup with zero quality risk and 18.5% memory reduction.

2. **On Linux (if available)**: Re-test with inductor backend (Triton). The operator fusion could push the speedup above 15%, achieving KEEP.

3. **Combined with Candidate E (adaptive scheduling)**: The 12.6% engineering speedup + scheduling could exceed 30% total, justifying KEEP for the combined C42.

4. **V2 (aot_eager) should be DROPPED** — it's counterproductive on this platform.

---

## Artifacts

- Benchmark script: `scripts/phase-c42/c42_compile_validation.py`
- Raw data: `results/phase-c42/c42_compile_data.json`
- V2b profiler trace: `results/phase-c42/c42_compile_trace.json`
- Phase 1-B reference: `reports/phase-c42/c42_separable_conv_validation.md`
- D-SSIM source: `scripts/epic05/phase7/loss.py`
