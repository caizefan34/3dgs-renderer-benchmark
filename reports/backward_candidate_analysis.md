# Track C: Backward Pipeline Candidate Analysis

## Profiling Evidence: Where Does the Time Go?

---

## 1. Pipeline Breakdown (Measured)

Profiled on A100 PCIe 40GB, gsplat 1.5.3, tile_size=16, 6 cameras per scene, CUDA event timing.

### Summary (averaged across 6 cameras per scene)

| Scene | N_Gaussians | Isect (ms) | Sort (ms) | Offset (ms) | Proj+Raster Fwd (ms) | Backward (ms) | Total (ms) |
|-------|------------|-----------|----------|------------|---------------------|--------------|-----------|
| room | 1,593,376 | 0.40 (2.8%) | 0.62 (4.4%) | 0.03 (0.2%) | 2.37 (16.9%) | 10.62 (75.7%) | 14.03 |
| bicycle | 6,131,954 | 0.63 (2.1%) | 0.91 (3.1%) | 0.05 (0.2%) | 4.74 (16.2%) | 22.91 (78.4%) | 29.23 |
| garden | 1,839,236 | 0.28 (2.5%) | 0.44 (4.0%) | 0.02 (0.2%) | 1.93 (17.6%) | 8.28 (75.6%) | 10.95 |

### Key finding

**Backward pass dominates: 75-78% of total pipeline time across all scenes.**

The sort stage (C17-1 target) is only 3-4% of total time. Even a perfect sort optimization (0 ms) would yield at most +4% e2e speedup.

---

## 2. Backward Kernel Analysis (Source: RasterizeToPixels3DGSBwd.cu)

### How the backward kernel works

```
For each tile (1 CTA per tile):
  For each pixel in tile (tile_size² threads):
    1. Read T_final = 1 - render_alphas[pix_id]  (from forward)
    2. Read last_ids[pix_id]  (last contributing Gaussian index)
    3. For each batch of Gaussians (back-to-front):
      a. Load Gaussian data (means2d, conics, colors, opacities) into shared memory
      b. For each Gaussian in batch:
        - Compute alpha, sigma
        - Update transmittance T
        - Compute gradients: v_rgb, v_conic, v_xy, v_opacity
        - Warp-reduce gradients
        - Atomic-add to global gradient buffers (v_colors, v_conics, v_means2d, v_opacities)
```

### Backward cost sources

1. **Gaussian data loading**: Each batch loads means2d, conics, colors (CDIM), opacities from global memory into shared memory. Cost ∝ n_isects × (8 + 12 + 4 + 4*CDIM) bytes.

2. **Alpha recomputation**: The backward pass must recompute alpha for every Gaussian-pixel pair (same computation as forward). This is the dominant compute cost.

3. **Atomic gradient accumulation**: `gpuAtomicAdd` to v_colors (CDIM floats), v_conics (3 floats), v_means2d (2 floats), v_opacities (1 float) per Gaussian per pixel. Total: (CDIM + 6) atomic adds per contribution.

4. **Early termination via last_ids**: The backward pass only processes Gaussians up to `last_ids[pix_id]` — the last Gaussian that contributed to this pixel in the forward pass. This skips Gaussians that were occluded.

### Why backward is 4-5x more expensive than forward

- **Forward**: Each pixel processes Gaussians front-to-back and can early-terminate when T < 1e-4. Most pixels terminate after 10-50 Gaussians.
- **Backward**: Must process ALL Gaussians from back-to-front up to `last_ids[pix_id]` to accumulate gradients correctly. Cannot skip intermediate Gaussians because each contributes to the transmittance chain.
- **Atomic contention**: Forward only writes 1 output per pixel (render_colors, render_alphas, last_ids). Backward writes gradients for EVERY Gaussian-pixel contribution via atomicAdd.

---

## 3. Gaussian Intersection Distribution (Tail Analysis)

### Measured distribution

| Scene | N_Gauss | Visible | Isect/Gauss mean | median | zeros | Bottom 1% isects from | Top 1% Gauss → % isects |
|-------|---------|---------|-----------------|--------|-------|----------------------|------------------------|
| room | 1.59M | 431K (27%) | 2.7 | 0.0 | 73.0% | 75.1% of Gaussians | 35.7% |
| bicycle | 6.13M | 1.94M (32%) | 1.4 | 0.0 | 68.3% | 69.7% of Gaussians | 24.9% |
| garden | 1.84M | 793K (43%) | 1.7 | 0.0 | 56.9% | 58.6% of Gaussians | 21.0% |

### Critical findings

1. **73% of Gaussians have ZERO intersections** for room. These Gaussians are invisible (outside the camera frustum or fully occluded). They still occupy memory and require gradient zeroing, but the backward kernel doesn't process them (no tile intersection → no work).

2. **Bottom 1% of intersections come from 75% of Gaussians** — the "long tail" of Gaussians with very few (1-2) tile intersections. These are small/far Gaussians that barely contribute.

3. **Top 1% of Gaussians account for 21-36% of intersections** — a small number of large/near Gaussians dominate the rendering work.

### C25 T5' hypothesis evaluation: "1% Gaussian subset still consumes ~29% backward cost"

**Partially confirmed**: Top 1% of Gaussians account for 21-36% of intersections (not exactly 29%, but in the right ballpark). The tail (bottom 75% of Gaussians) accounts for only 1% of intersections.

However, the backward cost is NOT simply proportional to intersection count. The backward kernel processes all Gaussians up to `last_ids[pix_id]` per pixel. The cost is dominated by:
- Number of visible pixels per tile
- Number of Gaussians per tile (intersection count)
- Whether early termination (last_ids) is effective

### Backward atomic write estimate

| Scene | Total bwd writes | Forward writes (pixels) | Ratio | Per-Gauss writes mean | p99 |
|-------|-----------------|----------------------|-------|----------------------|-----|
| room | 4.1M | 2.07M | 2.0x | 9.5 | 22.0 |
| bicycle | 1.6M | 2.07M | 0.8x | 0.8 | 4.8 |
| garden | 0.6M | 2.07M | 0.3x | 0.7 | 4.6 |

The backward-to-forward write ratio is modest (0.3-2.0x). The atomicAdd contention is not extreme — per-Gaussian writes are low (mean < 10).

---

## 4. Candidate Bottleneck Analysis

### Candidate C25: Sparse-tail backward reorganization

**Hypothesis**: The backward pass has fixed overhead from processing invisible/low-contribution Gaussians. If we skip Gaussians with near-zero contribution, we can reduce backward cost.

**Evidence**:
- 73% of Gaussians have zero intersections → already skipped by tile structure (no work)
- Bottom 75% of Gaussians account for only 1% of intersections → already minimal work
- The backward cost is dominated by the TOP 1-5% of Gaussians (21-36% of intersections) and the per-pixel processing chain

**Problem with the hypothesis**: The "1% Gaussians consuming 29% backward cost" claim is about the TOP contributors, not the bottom tail. The backward cost is NOT caused by processing invisible Gaussians — those are already skipped. The cost is caused by the **sheer number of Gaussian-pixel pairs** that need alpha recomputation and gradient accumulation.

**Verdict**: The sparse-tail hypothesis is **NOT the primary bottleneck**. The backward cost is dominated by:
1. Alpha recomputation (compute-bound, not memory-bound)
2. Per-pixel Gaussian traversal (up to last_ids, can be hundreds of Gaussians per pixel)
3. Atomic gradient writes (moderate contention, not extreme)

### Candidate: Backward alpha caching / reuse

**Hypothesis**: The backward pass recomputes alpha for every Gaussian-pixel pair. If we cached alpha values from the forward pass, we could skip this computation.

**Analysis**:
- Alpha computation: `sigma = 0.5 * (conic.x * dx² + conic.z * dy²) + conic.y * dx * dy; alpha = min(0.999, opac * exp(-sigma))`
- This requires 2 loads (conics, means2d) + ~10 FLOPs per Gaussian-pixel pair
- Caching alpha: would need to store `alpha` for every (pixel, Gaussian) pair = n_pixels × avg_gaussians_per_pixel × 4 bytes
- For room: 2M pixels × ~50 Gaussians/pixel × 4B = 400 MB — feasible on 40GB GPU
- But: the backward pass also needs `vis = exp(-sigma)` and `conic` values for gradient computation, so we'd still need to load conics

**Potential speedup**: Alpha computation is ~30% of backward per-Gaussian cost. Caching could save ~30% of backward time = ~23% e2e.

**Risk**: Memory traffic increases (loading cached alpha values). The forward pass would need to store alpha per (pixel, Gaussian) pair, increasing forward memory usage.

**Verdict**: **POTENTIALLY VIABLE** but requires careful memory analysis. Not a simple fix.

### Candidate: Per-pixel early termination optimization

**Hypothesis**: The backward pass uses `last_ids[pix_id]` to know where to stop. But within the processing loop, it still iterates through ALL Gaussians in each batch up to `last_ids`, even if some have alpha < ALPHA_THRESHOLD.

**Evidence from source code** (line 156-161):
```cpp
for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
    bool valid = inside;
    if (batch_end - t > bin_final) {
        valid = 0;
    }
```
The code already skips Gaussians beyond `bin_final` (last_ids). But within the range, it processes ALL Gaussians including those with alpha < threshold.

**Potential optimization**: Pre-compute a "contribution mask" in the forward pass that marks which Gaussians actually contributed (alpha > threshold AND not occluded). The backward pass would skip non-contributing Gaussians.

**Problem**: This requires additional storage (1 bit per intersection) and the forward pass would need to write this mask. The overhead may exceed the savings.

**Verdict**: **LOW IMPACT** — most Gaussians within the last_ids range DO contribute (that's why last_ids is set). The non-contributing ones are those with sigma < 0 or alpha < threshold, which are rare.

### Candidate: Batch size optimization for backward

**Hypothesis**: The backward processes Gaussians in batches of `block_size` (256 for tile16). Each batch requires a `block.sync()` and shared memory load. If we increased the batch size (more shared memory), we'd reduce the number of syncs.

**Current shared memory usage** (tile16, CDIM=3):
- id_batch: 256 × 4 = 1024 bytes
- xy_opacity_batch: 256 × 12 = 3072 bytes
- conic_batch: 256 × 12 = 3072 bytes
- rgbs_batch: 256 × 3 × 4 = 3072 bytes
- Total: 10240 bytes (10 KB)

A100 has 48 KB default shared memory. We could increase batch size to 1024 (40 KB), reducing batch count by 4x.

**Potential speedup**: Fewer sync points and shared memory loads. Estimated 10-15% backward speedup = 7-12% e2e.

**Risk**: Larger batches mean more threads idle if last_ids is reached mid-batch. Also, shared memory bank conflicts may increase.

**Verdict**: **MODERATE IMPACT, LOW RISK** — straightforward to implement, but needs careful tuning.

### Candidate: Warp-level specialization

**Hypothesis**: In the backward kernel, all threads in a tile process the same Gaussians but for different pixels. Some pixels finish early (low last_ids) while others process many Gaussians. Warp divergence causes idle threads.

**Potential optimization**: Assign different batches to different warps, so warps with finished pixels can process other batches. This is a work-stealing approach.

**Problem**: The current design has all threads cooperate on the same batch (shared memory co-loading). Work-stealing would break this pattern and require separate Gaussian loading per warp.

**Verdict**: **HIGH COMPLEXITY, UNCERTAIN BENEFIT** — fundamental redesign of the backward kernel.

---

## 5. Prior-Art Assessment

### gsplat backward kernel
The current implementation follows the standard 3DGS backward design from the original paper [Kerbl et al., 2023]. The key design choices:
- 1 CTA per tile, tile_size² threads
- Batch processing with shared memory caching
- Atomic gradient accumulation
- Back-to-front traversal using last_ids

### Known optimizations in the literature
1. **Alpha caching**: Some implementations cache alpha/transmittance in the forward pass to avoid recomputation in backward. Trade-off: memory vs compute.

2. **Gradient compaction**: Instead of atomicAdd per Gaussian-pixel pair, accumulate per-tile gradients in shared memory and write once. This reduces atomic contention but requires more shared memory.

3. **Separate opacity backward**: The opacity gradient depends on all contributing Gaussians, while color/position gradients are more local. Separating these could allow different parallelization strategies.

4. **Fused forward-backward**: For training (where both forward and backward are needed), fusing them into a single pass could avoid re-reading data. But this requires fundamental kernel redesign.

### What is NOT known to be done
- Tile-local sort for backward (C17-1 applies to forward sort, not backward)
- Per-pixel Gaussian count adaptation (dynamic batch sizing based on last_ids)
- Sparse backward (skipping low-contribution Gaussians based on pre-filtering)

---

## 6. Decision Matrix

| Candidate | Estimated E2E Impact | Implementation Risk | Complexity | Decision |
|-----------|--------------------|--------------------|------------|----------|
| C25 sparse-tail | ~0% (tail already skipped) | N/A | N/A | **DROP** — hypothesis not supported by evidence |
| Alpha caching | +15-23% | Medium (memory trade-off) | Medium | **NEED MORE EVIDENCE** — memory analysis needed |
| Batch size increase | +7-12% | Low (shared memory tuning) | Low | **ITERATE** — quick to test |
| Warp specialization | +5-15% | High (kernel redesign) | High | **NEED MORE EVIDENCE** — fundamental redesign |
| Gradient compaction | +5-10% | Medium (shared memory) | Medium | **NEED MORE EVIDENCE** — atomic contention is moderate |

---

## 7. Recommended Next Steps

1. **Batch size optimization** (quickest win): Increase backward batch size from 256 to 512/1024 by using more shared memory. Measure backward time reduction.

2. **Alpha caching feasibility**: Compute the exact memory requirement for caching alpha values (per-pixel, per-Gaussian) and determine if it fits in GPU memory for all 3 scenes.

3. **Gradient compaction**: Prototype per-tile gradient accumulation in shared memory with a single global write at the end. Measure atomicAdd reduction.

4. **C17-1 integration**: Despite backward dominating, the sort optimization (C17-1) still provides +4% e2e on the forward path. Combined with C42 (+30%), the total speedup is meaningful.

---

## 8. Data Provenance

| Item | Path |
|------|------|
| Pipeline profiler script | `scripts/phase-c42/track_c_pipeline_profiler.py` |
| Pipeline profile data | `results/a100/phase-c42/track_c_pipeline_profile.json` |
| Microbenchmark data | `results/a100/phase-c42/c17_1_microbench.json` |
| Tile distribution data | `results/a100/phase-c42/c17_1_tile_distribution.json` |
| Backward kernel source | `tmp_gsplat_src/RasterizeToPixels3DGSBwd.cu` |
| Forward kernel source | `tmp_gsplat_src/RasterizeToPixels3DGSFwd.cu` |
| Intersect source | `tmp_gsplat_src/IntersectTile.cu` |
| Source analysis report | `reports/c17_1_source_analysis.md` |
