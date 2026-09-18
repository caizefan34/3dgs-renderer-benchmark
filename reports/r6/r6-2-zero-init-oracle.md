# R6-2 — Zero-Init / Fixed-Cost Oracle (R6-B)

## Question

> How much of backward cost is spent merely preparing gradient storage rather
> than computing useful derivatives?

## Measured T_zero

T_zero = time spent in `at::zeros_like` (allocation + cudaMemsetAsync) for all
gradient buffers during backward. Measured via torch.profiler (backward-only).

Three C++ launchers zero-init gradient buffers every backward call:
- Rasterizer: 5 buffers (v_means2d, v_conics, v_colors, v_opacities, v_means2d_abs) = 11N floats
- Projection fused: 3 buffers (v_means, v_quats, v_scales) = 10N floats
- SH backward: 1 buffer (v_coefficients) = 3×(d+1)²×N floats

Total: 21N–69N floats = 84N–276N bytes per backward (degree 0–3).

## B-GATE evaluation

### B-GATE-1: T_zero ≥ 5% T_bwd

| Scene | Stage | T_zero%T_bwd | PASS? |
|-------|-------|-------------|-------|
| room | 5K | 6.6% | ✅ |
| room | 15K | 9.0% | ✅ |
| room | 30K | 5.5% | ✅ |
| bicycle | 5K | 8.2% | ✅ |
| bicycle | 15K | 9.6% | ✅ |
| bicycle | 30K | 11.6% | ✅ |
| garden | 5K | 10.5% | ✅ |
| garden | 15K | 8.4% | ✅ |
| garden | 30K | 11.3% | ✅ |

**B-GATE-1: PASS (9/9 profiles)**

### B-GATE-2: T_zero ≥ 3% T_iter

| Scene | Stage | T_zero%T_iter | PASS? |
|-------|-------|--------------|-------|
| room | 5K | 2.9% | ❌ |
| room | 15K | 3.7% | ✅ |
| room | 30K | 2.2% | ❌ |
| bicycle | 5K | 3.9% | ✅ |
| bicycle | 15K | 5.3% | ✅ |
| bicycle | 30K | 6.0% | ✅ |
| garden | 5K | 4.2% | ✅ |
| garden | 15K | 4.2% | ✅ |
| garden | 30K | 5.3% | ✅ |

**B-GATE-2: PASS (7/9 profiles)**

### B-GATE-3: Sparse-tail behavior

As useful backward work decreases (15K→30K, fewer intersections), T_zero%T_bwd
INCREASES:

| Scene | 15K T_zero%T_bwd | 30K T_zero%T_bwd | Trend |
|-------|------------------:|------------------:|-------|
| bicycle | 9.6% | 11.6% | ↑ +21% |
| garden | 8.4% | 11.3% | ↑ +35% |

The zero-init cost persists while useful work shrinks — classic sparse-tail.

**B-GATE-3: PASS (sparse-tail pattern confirmed for bicycle and garden)**

## R6-B Gate Verdict: **PASS**

B-GATE-1 passes universally (5.5-11.6% T_bwd). At least one gate condition is
met for all 9 profiles.

## Zero-cost oracle

### Upper bound (eliminate ALL zero-init)

$$S^{oracle}_{zero} = \frac{T_{iter}}{T_{iter} - T_{zero}}$$

| Scene | Stage | S_zero (upper) | Speedup % |
|-------|-------|---------------|-----------|
| room | 5K | 1.030 | 3.0% |
| room | 15K | 1.039 | 3.9% |
| room | 30K | 1.023 | 2.3% |
| bicycle | 5K | 1.041 | 4.1% |
| bicycle | 15K | 1.055 | 5.5% |
| bicycle | 30K | 1.064 | 6.4% |
| garden | 5K | 1.044 | 4.4% |
| garden | 15K | 1.044 | 4.4% |
| garden | 30K | 1.056 | 5.6% |

### Conservative (touched-only zero — skip zeroing untouched buffers)

$$\Delta T_{conservative} = T_{zero} \times (1 - r_{touch})$$

| Scene | Stage | r_touch | ΔT_cons (ms) | % T_iter |
|-------|-------|---------|-------------|----------|
| room | 5K | 0.330 | 2.00 | 1.9% |
| room | 15K | 0.305 | 2.70 | 2.6% |
| room | 30K | 0.284 | 1.69 | 1.6% |
| bicycle | 5K | 0.228 | 4.29 | 3.0% |
| bicycle | 15K | 0.294 | 5.88 | 3.7% |
| bicycle | 30K | 0.282 | 3.04 | 4.3% |
| garden | 5K | 0.180 | 3.81 | 3.4% |
| garden | 15K | 0.299 | 3.36 | 2.9% |
| garden | 30K | 0.292 | 2.19 | 3.8% |

Conservative E2E oracle: 1.6-4.3% (below 5% threshold).

### Buffer reuse potential

The measured T_zero includes both allocation and zeroing. If gradient buffers
are allocated once and reused across iterations (instead of at::zeros_like
every call), the allocation overhead is eliminated. Only touched entries need
re-zeroing. This could push the achievable savings toward the upper bound.

## Active/touched sparsity

| Scene | Stage | N_total | N_visible | r_touch | Grad buffer (MB) |
|-------|-------|---------|-----------|---------|-----------------:|
| room | 5K | 560,632 | 184,895 | 0.330 | 154.7 |
| room | 30K | 933,590 | 265,137 | 0.284 | 257.7 |
| bicycle | 5K | 1,933,177 | 441,011 | 0.228 | 508.8 |
| bicycle | 30K | 3,957,041 | 1,116,000 | 0.282 | 1092.1 |
| garden | 5K | 1,779,185 | 320,875 | 0.180 | 468.3 |
| garden | 30K | 2,610,559 | 762,000 | 0.292 | 720.5 |

67-82% of gradient buffer bytes are zero-initialized but never written to.
