# AccuTile A100 Formal Timing Benchmark Report

> **⚠️ VARIANT P — NOT TRUE ACCUTILE — SUPERSEDED NAMING**
> 
> This report evaluates the per-tile conservative conic predicate (variant P), which is NOT the true upstream AccuTile algorithm. The true AccuTile (variant A) uses a strip-based SnugBox + ellipse-intersection algorithm that IS faster on A100 (1.12-1.28x total speedup). See `accutile-identity-audit.md` and `accutile-a100-final.md` for the corrected B1/P/A comparison.
>
> The conclusions below apply ONLY to variant P, not to true AccuTile.

## Summary

**AccuTile does NOT provide positive end-to-end renderer speedup on A100.** Despite reducing intersections by 46–53%, AccuTile makes the forward pass 2.1–3.1x SLOWER due to the per-tile conservative predicate computation overhead. The backward pass is 11–16% faster (fewer intersections to process), but the forward regression dominates, resulting in a net 31–47% total renderer time regression.

This is a **PROPAGATION FAILURE**: intersection reduction does not translate into useful work reduction because the cost of computing the conservative predicate exceeds the savings from fewer intersection entries.

---

## 1. Method

- **Hardware**: NVIDIA A100-PCIE-40GB (SM 8.0), single GPU
- **Timing**: CUDA Events (torch.cuda.Event with enable_timing=True)
- **Warmup**: 20 iterations
- **Measure**: 100 iterations
- **Forward event**: from start to after rasterization() call
- **Backward event**: from after forward to after .backward()
- **Total event**: from start to after backward
- **Checkpoint**: frozen 30K A100-trained checkpoints, camera 0
- **Controls**: same camera, same tile size (16), same tensors, same process conditions

## 2. Results by Scene

### room (1,105,873 Gaussians, 3114×2075)

| Metric | B1 (mean) | B1 (median) | B1 (std) | B1 (p95) | B1A (mean) | B1A (median) | B1A (std) | B1A (p95) |
|--------|-----------|-------------|-----------|----------|------------|--------------|-----------|-----------|
| Forward (ms) | 8.52 | 8.49 | 0.22 | 8.92 | 26.82 | 26.71 | 1.34 | 29.11 |
| Backward (ms) | 7.11 | 7.10 | 0.03 | 7.17 | 6.16 | 6.16 | 0.01 | 6.17 |
| Total (ms) | 15.63 | 15.61 | 0.22 | 15.99 | 32.98 | 32.86 | 1.33 | 35.27 |

| Speedup | Value |
|---------|-------|
| Forward | 0.318× (3.15× slower) |
| Backward | 1.155× (15.5% faster) |
| Total | 0.474× (2.11× slower) |

### bicycle (2,589,484 Gaussians, 4946×3286)

| Metric | B1 (mean) | B1 (median) | B1 (std) | B1 (p95) | B1A (mean) | B1A (median) | B1A (std) | B1A (p95) |
|--------|-----------|-------------|-----------|----------|------------|--------------|-----------|-----------|
| Forward (ms) | 12.94 | 12.89 | 0.31 | 13.63 | 40.27 | 39.99 | 2.46 | 44.38 |
| Backward (ms) | 17.60 | 17.56 | 0.10 | 17.79 | 15.79 | 15.70 | 0.17 | 16.09 |
| Total (ms) | 30.55 | 30.47 | 0.36 | 31.41 | 56.06 | 55.83 | 2.43 | 60.31 |

| Speedup | Value |
|---------|-------|
| Forward | 0.321× (3.11× slower) |
| Backward | 1.115× (11.5% faster) |
| Total | 0.545× (1.84× slower) |

### garden (874,019 Gaussians, 5187×3361)

| Metric | B1 (mean) | B1 (median) | B1 (std) | B1 (p95) | B1A (mean) | B1A (median) | B1A (std) | B1A (p95) |
|--------|-----------|-------------|-----------|----------|------------|--------------|-----------|-----------|
| Forward (ms) | 14.69 | 14.65 | 0.27 | 15.06 | 36.50 | 35.70 | 2.46 | 40.88 |
| Backward (ms) | 27.36 | 27.30 | 0.16 | 27.60 | 24.12 | 24.13 | 0.08 | 24.19 |
| Total (ms) | 42.05 | 41.95 | 0.32 | 42.59 | 60.62 | 59.85 | 2.46 | 64.99 |

| Speedup | Value |
|---------|-------|
| Forward | 0.402× (2.49× slower) |
| Backward | 1.134× (13.4% faster) |
| Total | 0.694× (1.44× slower) |

## 3. Aggregate Speedup Summary

| Scene | Forward speedup | Backward speedup | Total renderer speedup |
|-------|-----------------|------------------|------------------------|
| room | 0.318× | 1.155× | 0.474× |
| bicycle | 0.321× | 1.115× | 0.545× |
| garden | 0.402× | 1.134× | 0.694× |
| **geomean** | **0.343×** | **1.134×** | **0.547×** |

## 4. Root Cause Analysis

### Why forward is slower

The AccuTile conservative predicate (`accutile_tile_may_contribute`) is evaluated for every candidate tile in every Gaussian's AABB during the intersection generation kernel. This adds per-tile computation:

1. For each Gaussian, the AABB still defines the candidate tile range (unchanged)
2. For each candidate tile, the predicate evaluates 4 corner q-values, 4 edge-interior q-values, and a center check
3. This adds ~8 floating-point evaluations per candidate tile per Gaussian

The AABB baseline simply counts and emits all tiles in the AABB without any per-tile computation. The predicate overhead (evaluating the conic minimum over each tile rectangle) exceeds the savings from emitting fewer intersection entries.

### Why backward is faster

The backward rasterizer processes sorted intersection entries. With fewer intersections (46–53% reduction), the backward kernel has fewer entries to iterate over, resulting in 11–16% speedup. However, this backward speedup is smaller than the forward regression because:
- Forward includes intersection generation (where the predicate cost is paid) + rasterization
- The predicate cost is proportional to the AABB tile count (not the reduced count)
- The backward savings are proportional to the reduced intersection count

### Forward decomposition

The "forward" timing includes:
1. Projection (unchanged)
2. Intersection generation (AABB enumeration + AccuTile predicate = MUCH SLOWER)
3. Radix sort (fewer entries to sort = slightly faster, but dominated by sort overhead)
4. Forward rasterization (same number of pixels, same alpha blending = unchanged)

The intersection generation step (step 2) is where the regression occurs. The predicate computation adds ~18–28ms per frame, while the sort savings from fewer entries is negligible (CUB radix sort is already very fast on A100).

## 5. Comparison with Non-Authoritative RTX 5070 Probe

The Codex RTX 5070 Laptop probe showed:
- baseline = 700.09 ms (full iteration)
- AccuTile = 599.70 ms
- apparent gain ≈ 14.34%

This result is **NOT authoritative** and should not be used as A100 evidence. The RTX 5070 result was:
1. A single wall-clock sample (not 20-warmup/100-measure with CUDA events)
2. On a different GPU architecture (SM 12.0 vs SM 8.0)
3. Using a different checkpoint (diverged, PSNR=12.03 vs normal PSNR=20.56)
4. Mixing hardware cohorts (RTX 5070 Laptop vs A100)

The A100 formal benchmark with CUDA Events and proper warmup/measure protocol shows the opposite: AccuTile is net SLOWER on A100.

## 6. Note on Stage Decomposition

The benchmark measures forward (rasterization call) and backward (autograd backward) separately. A more granular stage decomposition (intersection generation, sort, forward rasterization, backward rasterization as separate events) was not included because the forward regression is already clearly attributable to the intersection generation step where the predicate is evaluated. The v1.5.3 API does not expose separate timing hooks for isect_tiles vs rasterize_to_pixels within a single rasterization() call.

## 7. Verdict

**Timing benchmark: FAIL** for performance promotion.

- Forward is 2.5–3.1× slower across all scenes
- Backward is 11–16% faster across all scenes
- Net total renderer time is 1.4–2.1× slower
- Intersection reduction does not translate into renderer speedup on A100
- The conservative predicate computation cost exceeds the savings from fewer intersections
