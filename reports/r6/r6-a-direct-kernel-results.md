# R6-A — Direct Kernel Counter Results

## Evidence level: 2 (direct debug instrumentation)

A `__device__ unsigned long long` counter was injected into the
`rasterize_to_pixels_3dgs_bwd_kernel` at the exact warp-leader atomic execution
point (`if (warp.thread_rank() == 0)`). The counter is incremented with
`::atomicAdd(&r6a_debug_count, 1ULL)` once per warp-leader atomicAdd execution
set. This counts the ACTUAL number of warp-leader atomic operations, using the
kernel's real `ALPHA_THRESHOLD = 1/255 ≈ 0.00392` on ALL tiles (not a sample).

## Results (all 9 workloads, A100, camera 0, 5 measured iterations)

| Scene | Stage | N | n_isects | actual_atomic | R_atomic_direct | T_iter (ms) | T_atomic (ms) | Savings (ms) | E2E (%) | Gate |
|-------|-------|---|----------|--------------:|----------------:|------------:|--------------:|-------------:|--------:|------|
| room | 5K | 560,632 | 4,482,292 | 17,829,372 | 3.978 | 60.59 | 2.97 | 1.56 | 2.57 | ❌ |
| room | 15K | 933,590 | 4,312,157 | 14,239,098 | 3.302 | 59.15 | 2.37 | 1.16 | 1.96 | ❌ |
| room | 30K | 933,590 | 4,002,983 | 11,470,825 | 2.866 | 55.61 | 1.91 | 0.87 | 1.57 | ❌ |
| bicycle | 5K | 1,933,177 | 6,669,895 | 22,032,638 | 3.303 | 76.04 | 3.67 | 1.79 | 2.36 | ❌ |
| bicycle | 15K | 3,957,041 | 8,877,402 | 20,225,740 | 2.278 | 101.90 | 3.37 | 1.32 | 1.30 | ❌ |
| bicycle | 30K | 3,957,041 | 7,898,651 | 16,609,715 | 2.103 | 93.63 | 2.77 | 1.02 | 1.09 | ❌ |
| garden | 5K | 1,779,185 | 4,166,629 | 13,234,996 | 3.176 | 61.49 | 2.21 | 1.06 | 1.72 | ❌ |
| garden | 15K | 2,610,559 | 5,879,867 | 14,619,764 | 2.486 | 73.77 | 2.44 | 1.02 | 1.38 | ❌ |
| garden | 30K | 2,610,559 | 5,502,981 | 12,848,502 | 2.335 | 70.66 | 2.14 | 0.86 | 1.21 | ❌ |

## Key observations

### 1. R_atomic_direct = 2.10–3.98 (all ≥ 2, condition 1 passes 9/9)

The direct kernel counter gives R_atomic = 2.103–3.978. This is HIGHER than
the pixel-level simulation (2.07–2.86) because the kernel uses
ALPHA_THRESHOLD = 1/255 ≈ 0.0039 (lower than the simulation's 0.01), so more
warps have pixels with valid alpha → more warp-leader atomics.

### 2. Counter is perfectly deterministic (std = 0.0 for all workloads)

The atomic count is identical across all 5 measured iterations for every
workload. This confirms the atomic pattern is fully determined by the scene
geometry and camera viewpoint — no runtime variance.

### 3. E2E = 1.09–2.57% (all < 5%, condition 2 fails 0/9)

Even with the higher direct R_atomic, the conservative E2E (using the hardware
throughput model at 30 G atomics/s with 30% __syncthreads overhead) remains
below 5% for all 9 workloads. The best case is room 5K at 2.57%.

### 4. The footprint estimate was 1.4–1.9× too high vs direct (not 2.6×)

| Method | R_atomic range | vs direct |
|--------|---------------|-----------|
| Footprint estimate (v1) | 5.50–7.61 | 1.4–1.9× too high |
| Pixel simulation (v2) | 2.07–2.86 | 0.7–1.0× (slightly underestimated) |
| **Direct kernel counter** | **2.10–3.98** | **ground truth** |

The pixel simulation underestimated because it used alpha_threshold=0.01
(2.5× higher than the kernel's 1/255). The footprint estimate overestimated
because bounding-box radii are much larger than actual alpha coverage.

### 5. Block aggregation reduces atomics by 2.1–4.0×, not 5.5–7.6×

The block-aggregation reduction factor = R_atomic = 2.1–4.0. This is the
factor by which warp-leader atomic operations are reduced (from R_atomic per
tile-Gaussian pair to 1). It is NOT the factor by which total backward time
is reduced.

## Gate evaluation

| Condition | Result |
|-----------|--------|
| 1. Direct reduction potential ≥ 2× | **PASS 9/9** (2.10–3.98) |
| 2. Conservative E2E ≥ 5% | **FAIL 0/9** (1.09–2.57%) |
| 3. At least 2 real workloads pass | **FAIL** (0/9 pass both) |
| 4. Opportunity not noise | N/A (conditions 2,3 fail) |

## Verdict: DEFER

R6-A does NOT advance to minimal CUDA implementation. The block-aggregation
reduction factor (2.1–4.0×) is real and directly measured, but the E2E impact
(1.09–2.57%) is below the 5% conservative threshold for ALL 9 workloads.

### Why not DROP?

1. The mechanism is sound (block-level aggregation is correct and novel)
2. R_atomic ≥ 2 confirms real duplication (condition 1 passes 9/9)
3. The opportunity may become significant in composition with other optimizations
4. Future hardware (more atomics contention) or larger scenes may increase benefit
5. The direct measurement is deterministic and reproducible

### What would change the verdict?

- If ncu profiling (requires root) shows actual atomic stall fraction > 50% of
  T_raster (vs. our model's implied ~10-15%), the E2E could reach 5%
- If the __syncthreads overhead is < 10% (vs. our 30% assumption), savings increase
- If composed with other exact optimizations, the relative E2E could be larger

## Comparison across all three evidence levels

| Metric | Footprint (L5) | Pixel sim (L2 proxy) | **Direct counter (L2)** | Correction |
|--------|---------------:|---------------------:|------------------------:|------------|
| R_atomic range | 5.50–7.61 | 2.07–2.86 | **2.10–3.98** | L5 was 1.4–1.9× too high |
| A-GATE-3 pass | 2/9 | 0/9 | **0/9** | All v1 passes were false positives |
| Best E2E | 10.0% | 1.58% | **2.57%** | L5 was 3.9× too high |
| Median E2E | 3.1% | 0.93% | **1.38%** | L5 was 2.2× too high |
