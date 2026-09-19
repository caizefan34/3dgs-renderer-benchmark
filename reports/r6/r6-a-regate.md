# R6-A — Re-Gate with Repaired Evidence

## Re-gate conditions (per user protocol)

R6-A may advance to CUDA prototype ONLY if ALL five conditions are met:

1. **Atomic duplication directly measured** (not inferred from forward metadata)
2. **Block aggregation potential ≥ 2×**
3. **Conservative E2E opportunity ≥ 5%**
4. **At least two real workloads pass**
5. **Opportunity is not measurement noise**

## Condition 1: Direct measurement

**PASS** — `r6_a_direct_atomic.py` performs pixel-level alpha evaluation for
200 tiles per camera, directly counting warp-Gaussian pairs where any pixel
in the warp has alpha > threshold. This is a direct simulation of the backward
kernel's atomic pattern, not an inference from forward metadata.

See `r6-a-direct-atomic-instrumentation.md` for methodology.

## Condition 2: Block aggregation potential ≥ 2×

| Scene | Stage | R_atomic (direct) | ≥ 2? |
|-------|-------|------------------:|------|
| room | 5K | 2.86 | ✅ |
| room | 15K | 2.77 | ✅ |
| room | 30K | 2.73 | ✅ |
| bicycle | 5K | 2.51 | ✅ |
| bicycle | 15K | 2.07 | ✅ |
| bicycle | 30K | 2.11 | ✅ |
| garden | 5K | 2.69 | ✅ |
| garden | 15K | 2.41 | ✅ |
| garden | 30K | 2.37 | ✅ |

**PASS (9/9)** — R_atomic = 2.07-2.86, all ≥ 2. But the margin is thin
(minimum 2.07, only 3.5% above threshold).

## Condition 3: Conservative E2E opportunity ≥ 5%

Using the corrected R_atomic, the conservative E2E oracle is recomputed:

```
T_atomic = 5 × R_atomic × n_isects / 30e9 × 1000  (ms, at 30 G atomics/sec)
T_atomic_ideal = 5 × n_isects / 30e9 × 1000  (ms, block-aggregated)
Savings = (T_atomic - T_atomic_ideal) × 0.70  (30% overhead for __syncthreads)
E2E = Savings / T_iter × 100
```

| Scene | Stage | R_atomic | n_isects | T_atomic (ms) | Savings (ms) | E2E (%) | ≥ 5%? |
|-------|-------|----------|----------|--------------:|-------------:|--------:|-------|
| room | 5K | 2.86 | 4.62M | 2.20 | 0.95 | 0.92 | ❌ |
| room | 15K | 2.77 | 4.51M | 2.08 | 0.89 | 0.85 | ❌ |
| room | 30K | 2.73 | 4.02M | 1.83 | 0.78 | 0.74 | ❌ |
| bicycle | 5K | 2.51 | 6.14M | 2.57 | 1.11 | 0.79 | ❌ |
| bicycle | 15K | 2.07 | 8.35M | 2.88 | 0.98 | 0.68 | ❌ |
| bicycle | 30K | 2.11 | 7.37M | 2.59 | 0.99 | 1.42 | ❌ |
| garden | 5K | 2.69 | 6.01M | 2.70 | 1.15 | 1.03 | ❌ |
| garden | 15K | 2.41 | 6.70M | 2.69 | 1.06 | 0.93 | ❌ |
| garden | 30K | 2.37 | 5.74M | 2.27 | 0.92 | 1.58 | ❌ |

**FAIL (0/9)** — Conservative E2E = 0.68-1.58%, ALL below 5%.

The original v1 estimate (1.8-10.0%) was inflated because R_atomic was
overestimated by 2.6-2.7×. With the corrected R_atomic, the atomic stall time
is 2.6× lower, and the savings from block aggregation are proportionally lower.

## Condition 4: At least two real workloads pass

**FAIL** — 0/9 workloads pass condition 3 (conservative E2E ≥ 5%).

## Condition 5: Opportunity not measurement noise

**N/A** — Conditions 3 and 4 fail. The conservative E2E (0.68-1.58%) is
consistent across workloads (low variance), so it is NOT noise — it is a
genuine but SMALL opportunity. The issue is that the opportunity is too small
to justify implementation, not that it is noisy.

## Re-gate verdict

### R6-A = DEFER

R6-A does NOT advance to CUDA prototype. The block-aggregation reduction
factor (2.1-2.9×) is real but the E2E impact (0.68-1.58%) is below the 5%
conservative threshold for ALL 9 workloads.

### Why not DROP?

R6-A is not dropped because:
1. The mechanism is sound (block-level aggregation is correct and novel)
2. R_atomic ≥ 2 confirms real duplication (condition 2 passes)
3. The opportunity may become significant in composition with other optimizations
4. Future hardware (more atomics contention) or larger scenes may increase the benefit

### What would change the verdict?

- If ncu profiling (requires root) shows actual atomic stall fraction > 30% of
  T_raster (vs. our model's 20-50%), the E2E could reach 5%
- If the __syncthreads overhead is < 10% (vs. our 30% assumption), savings increase
- If composed with R6-B (which eliminates zero-init), the relative E2E of R6-A
  could be larger (smaller T_iter denominator)

## Comparison: v1 vs v2

| Metric | v1 (footprint est.) | v2 (direct pixel sim.) | Change |
|--------|--------------------:|-----------------------:|--------|
| R_atomic range | 5.50-7.61 | 2.07-2.86 | -2.6× correction |
| A-GATE-3 pass | 2/9 | 0/9 | All false positives eliminated |
| Best E2E | 10.0% | 1.58% | 6.3× correction |
| Median E2E | 3.1% | 0.93% | 3.3× correction |
