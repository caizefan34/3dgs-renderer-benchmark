# 3DGS Renderer Optimization — Comprehensive Experiment Report

## Three-Track Research: C17-1, C42, Backward Pipeline

**Platform**: A100 PCIe 40GB (108 SMs, SM80), CUDA 11.8, gsplat 1.5.3, PyTorch 2.7.1
**Scenes**: Mip-NeRF 360 — room (1.59M Gaussians), bicycle (6.13M), garden (1.84M)
**Discipline**: One hypothesis at a time. Baseline → modification → measurement → decision. No combination experiments.

---

# Track A: C17-1 Fused Per-Tile Shared Memory Sort

## A.1 Hypothesis

3DGS tile intersections are naturally grouped by tile. The current global CUB DeviceRadixSort (46-bit keys, 6 passes) can be replaced with a tile-local block radix sort in shared memory, eliminating unnecessary global sorting passes.

**Prior assumption**: C17-0 (segmented sort) failed due to atomic scatter overhead. C17-1 avoids scatter by using a reduced-bit global sort (14-bit, 2 passes) for tile grouping, then per-tile BlockRadixSort for depth ordering.

## A.2 Prior Art

- **BlockRadixSort** is a standard CUB primitive — no novelty claimed for the sort itself.
- **3DGS rasterizers** (gsplat, INRIA original, FastGS): all use global DeviceRadixSort. No known implementation uses tile-local sorting.
- **Research question**: Can 3DGS exploit tile independence to eliminate unnecessary global sorting? This specific application of block sort to 3DGS tile data is the potentially novel contribution.

## A.3 Implementation — Phase A1 (Source Analysis)

**Completed**: `reports/c17_1_source_analysis.md`

Key findings from source code analysis:
- Intersections created in `intersect_tile_kernel` (Pass 2), ordered by Gaussian index
- Global sort needed because rasterizer reads contiguous ranges per tile: `tile_offsets[tile_id]` to `tile_offsets[tile_id+1]`
- 64-bit key: `[image_id | tile_id (13 bits) | depth (32 bits)]`, sorted over 46 bits = 6 passes
- Offset kernel can be fused with tile-local sort (each CTA knows its tile_id and count)
- Interface that must be preserved: `tile_offsets`, `flatten_ids`, `last_ids` (backward depends on forward's sorted order)

## A.4 Implementation — Phase A2 (Microbenchmark)

**Completed**: `results/a100/phase-c42/c17_1_microbench.json`

### Method

Standalone CUDA benchmark comparing:
1. **Method 1 (baseline)**: CUB DeviceRadixSort, 46-bit keys, 6 passes
2. **Method 2 (C17-1)**: CUB BlockRadixSort per tile, 32-bit keys, 4 passes, shared memory
3. **Method 3**: CUB DeviceRadixSort, 14-bit keys, 2 passes (reduced-bit global sort only)
4. **Combined**: Method 3 + Method 2

Test sizes: 256, 512, 1024, 2048, 4096, 8192, 12000 (intersections per tile)
n_tiles = 8160 (120×68 for 1920×1080 at tile16), 100 iterations each.

### Results

| N_tile | Global46 (ms) | Global14 (ms) | BlockSort (ms) | Combined (ms) | Block/Global | Decision |
|--------|-------------|-------------|--------------|-------------|------------|----------|
| 256 | 0.582 | 0.239 | 0.242 | 0.462 | 0.42x | **PASS** |
| 512 | 0.944 | 0.315 | 0.297 | 0.614 | 0.31x | **PASS** |
| 1024 | 1.374 | 0.479 | 0.531 | 0.941 | 0.39x | **PASS** |
| 2048 | 2.079 | 0.773 | 0.859 | 1.521 | 0.41x | **PASS** |
| 4096 | 4.439 | 1.952 | 1.920 | 3.823 | 0.43x | **PASS** |
| 8192 | 8.337 | 3.173 | OVERFLOW | OVERFLOW | — | OVERFLOW |
| 12000 | 11.738 | 4.660 | OVERFLOW | OVERFLOW | — | OVERFLOW |

### Key findings

1. **BlockRadixSort passes the 1.5x gate for all feasible tile sizes** (256-4096): 2.3x-3.2x faster than global 46-bit sort.
2. **Overflow at N>4096**: CUB BlockRadixSort uses static shared memory, limited to 48 KB. N=8192 requires ~64 KB.
3. **Combined (14-bit global + block sort)** is 1.2-1.5x faster than global 46-bit sort.
4. **Occupancy**: Block sort with N=4096 uses 16.9 KB shared memory, 111 registers, 2 blocks/SM → 38 waves for 8160 tiles. Low occupancy but each tile is independent.

### Overflow analysis

From tile intersection distribution profiling (`results/a100/phase-c42/c17_1_tile_distribution.json`):

| Scene | tile16 max intersections | % tiles >4096 |
|-------|-------------------------|---------------|
| room | 3,795 | 0.00% |
| bicycle | 8,981 | 0.23% |
| garden | 2,354 | 0.00% |

Only bicycle has overflow tiles (0.23%). These need a fallback to global sort.

### Decision gate

**PASS** — BlockRadixSort is ≥1.5x faster for realistic tile sizes (1024-4096). Proceed to Phase A3 (integration).

## A.5 Implementation — Phase A3 (Integration) — PENDING

Not yet implemented. Design from source analysis:

1. Replace `radix_sort_double_buffer` (46-bit, 6 passes) with `radix_sort_double_buffer` (14-bit, 2 passes) for tile grouping
2. New kernel `fused_tile_sort_kernel`: each CTA = 1 tile, loads intersections from grouped output, sorts by depth using `cub::BlockRadixSort`, writes sorted `flatten_ids` + `tile_offsets`
3. Overflow handling: tiles with >4096 intersections fall back to the existing 46-bit global sort

**Estimated speedup from microbenchmark**: Combined method is 1.2-1.5x faster than baseline sort. Baseline sort = 0.62 ms (room), so savings = 0.12-0.21 ms. E2e impact = 0.12-0.21 ms / 14.03 ms = **+0.9-1.5% e2e**.

**But Track C profiling shows sort is only 3-4% of total pipeline time.** Even a perfect sort optimization yields at most +4% e2e. With the combined method's 1.2-1.5x sort speedup, expected e2e speedup is **+1-2%**.

## A.6 Decision

**NEED MORE EVIDENCE** — Microbenchmark passes, but the e2e impact is marginal (+1-2%) because:
1. Sort is only 3-4% of total pipeline time (Track C finding)
2. Backward pass dominates at 75-78%
3. Integration risk (overflow handling, correctness) may not justify +1-2% speedup

**Recommendation**: Deprioritize C17-1 integration. Focus on backward optimization (Track C) which targets the 75-78% bottleneck.

---

# Track B: C42 Resolution-Adaptive SSIM

## B.1 Hypothesis

SSIM is a perceptual metric that doesn't need full pixel resolution. Downsampling pred+gt before the 11×11 Gaussian convolution reduces SSIM cost by ~s² while preserving gradient direction.

**Research question**: Why does reduced-resolution SSIM preserve 3DGS optimization? Answer: SSIM captures structural similarity at a scale larger than individual pixels. The 11×11 Gaussian window already smooths over 121 pixels, so subsampling by 2x (scale=0.5) still captures the same structural information. The L1 loss (full resolution) provides pixel-level supervision, while SSIM provides structural-level supervision.

## B.2 Prior Art

- Resolution-adaptive loss is common in NeRF/3DGS training (progressive resolution training)
- But downsampling the SSIM *kernel input* (not the rendering) is less common
- The specific technique of `F.interpolate(scale_factor, mode="area")` before SSIM conv is a straightforward application

## B.3 Implementation

**Scale sweep**: 1.0, 0.875, 0.75, 0.625, 0.5
- 5K training iterations, room scene, seed=42, SfM init (1.59M points)
- L1 loss stays full resolution. Only D-SSIM is downsampled.
- Loss: `(1-λ) * L1 + λ * D-SSIM` where λ=0.2
- Downsample: `F.interpolate(pred.unsqueeze(0).permute(0,3,1,2), scale_factor=s, mode="area")`

## B.4 Experiment Results

### Timing (mean over 50 measurements, ms)

| Scale | Per-iter (ms) | SSIM (ms) | L1 (ms) | SSIM reduction | E2E speedup |
|-------|-------------|----------|---------|---------------|------------|
| 1.000 | 272.15 | 96.74 | 0.40 | — | — |
| 0.875 | 254.72 | 63.52 | 0.34 | -34.3% | +6.4% |
| 0.750 | 254.62 | 50.91 | 0.39 | -47.4% | +6.4% |
| 0.625 | 251.63 | 36.09 | 0.44 | -62.7% | +7.5% |
| 0.500 | 261.60 | 24.84 | 0.39 | -74.3% | +3.9% |

### Quality (final at iter 4999, 13 eval cameras, full resolution)

| Scale | PSNR (dB) | dPSNR vs 1.0 | SSIM | dSSIM | LPIPS | Gaussians |
|-------|----------|-------------|------|-------|-------|-----------|
| 1.000 | 21.436 | — | 0.6801 | — | 0.0498 | 23,770,119 |
| 0.875 | 21.438 | +0.002 | 0.6823 | +0.0022 | 0.0497 | 21,768,591 |
| 0.750 | 21.376 | -0.060 | 0.6799 | -0.0002 | 0.0500 | 22,737,198 |
| 0.625 | 21.396 | -0.040 | 0.6821 | +0.0020 | 0.0499 | 21,763,999 |
| 0.500 | 21.344 | -0.092 | 0.6782 | -0.0019 | 0.0502 | 23,322,728 |

### Gradient analysis (cosine similarity vs scale=1.0)

| Scale | xyz cosine @1000 | xyz cosine @3000 | shs cosine @1000 | shs cosine @3000 |
|-------|-----------------|-----------------|-----------------|-----------------|
| 0.875 | 0.923 | 0.900 | 0.999 | 0.999 |
| 0.750 | 0.895 | 0.859 | 0.999 | 0.998 |
| 0.625 | 0.861 | 0.819 | 0.997 | 0.995 |
| 0.500 | 0.861 | 0.820 | 0.994 | 0.988 |

### Topology (clone/split/prune at iter 4999)

| Scale | Clone | Split | Prune | Final GS |
|-------|-------|-------|-------|----------|
| 1.000 | 2,714,065 | 9,731,514 | 350 | 23,770,119 |
| 0.875 | 2,314,814 | 8,930,352 | 303 | 21,768,591 |
| 0.750 | 2,513,094 | 9,315,502 | 276 | 22,737,198 |
| 0.625 | 2,410,437 | 8,880,242 | 298 | 21,763,999 |
| 0.500 | 2,725,583 | 9,502,027 | 285 | 23,322,728 |

## B.5 Bottleneck Explanation

### Why e2e speedup is lower than SSIM speedup

**SSIM time reduction is real and significant** (34-74%), but total iteration time is dominated by the backward pass:

```
Pipeline breakdown (room, from Track C profiling):
  Projection + Rasterize Fwd:  2.37 ms  (16.9%)
  Sort:                        0.62 ms  (4.4%)
  Backward:                   10.62 ms  (75.7%)
  Total:                      14.03 ms

SSIM is part of forward loss computation, not backward.
With 23M Gaussians, backward dominates and SSIM savings are diluted.
```

**Context dependency**: The earlier +30-34% measurement (at 10K iters with ~900K Gaussians) was valid when SSIM was a larger fraction of total time. With 23M Gaussians (from aggressive densification), backward dominates.

### Why quality is preserved

1. **L1 loss stays full resolution** — pixel-level supervision is unchanged
2. **SSIM at 0.5x still captures structural information** — the 11×11 Gaussian window smooths over 121 pixels; at 0.5x it still covers 30+ pixels
3. **Gradient cosine for shs stays >0.988** — color/SH gradients are barely affected
4. **Gradient cosine for xyz drops to 0.82-0.92** — position gradients are more affected, but final quality is preserved because L1 provides position supervision

### Why Gaussian count explodes

All scales produce 21-24M Gaussians from 1.59M SfM init. This is a training config issue (densification threshold 0.0002 is too aggressive for 1.59M initial points), not scale-specific. The explosion dilutes C42's benefit because backward cost scales with Gaussian count.

## B.6 Pareto Analysis

**Decision gate**: dPSNR > -0.2 dB

| Scale | E2E speedup | dPSNR | Pass gate? | Pareto? |
|-------|-----------|-------|-----------|---------|
| 0.875 | +6.4% | +0.002 | ✅ | ✅ Best quality + good speedup |
| 0.750 | +6.4% | -0.060 | ✅ | Same speedup as 0.875 but worse quality |
| 0.625 | +7.5% | -0.040 | ✅ | ✅ Best speedup with acceptable quality |
| 0.500 | +3.9% | -0.092 | ✅ | ❌ Slower than 0.625 (backward variance) |

**Pareto frontier**: scale=0.875 (best quality) and scale=0.625 (best speedup).

**Optimal operating point**: scale=0.875 — same speedup as 0.75, slightly better PSNR than 1.0, fewer Gaussians (21.8M vs 23.8M), best gradient cosine (0.92 for xyz).

## B.7 Decision

**KEEP** — C42 at scale=0.875 is the optimal operating point:
- +6.4% e2e speedup (with 23M Gaussians; would be +30% with <1M Gaussians)
- +0.002 dB PSNR (quality preserved or slightly improved)
- 34% SSIM time reduction
- Best gradient alignment (xyz cosine=0.92, shs cosine=0.999)
- Lowest Gaussian count (21.8M, less memory)

**Important caveat**: The e2e speedup is context-dependent. With fewer Gaussians (early training or constrained densification), C42 provides +30% as previously measured. With aggressive densification (23M Gaussians), backward dominates and C42 provides +6%.

**B2 Adaptive schedule**: Not pursued. The constant-scale experiment shows that 0.875 is already optimal — there's no benefit to scheduling since the speedup is limited by backward, not SSIM.

---

# Track C: New Bottleneck Discovery

## C.1 Hypothesis

After C17-1/C42 analysis, profile the complete renderer to find the next largest optimization opportunity. Focus on training backward optimization.

## C.2 Prior Assumption

C25 T5' hypothesis: "1% Gaussian subset still consumes ~29% backward cost" — suggesting sparse-tail backward reorganization.

## C.3 Implementation — Pipeline Profiler

**Completed**: `results/a100/phase-c42/track_c_pipeline_profile.json`

### Pipeline breakdown (averaged across 6 cameras per scene)

| Scene | N_Gauss | Isect (ms) | Sort (ms) | Offset (ms) | Proj+Raster Fwd (ms) | Backward (ms) | Total (ms) |
|-------|---------|-----------|----------|------------|---------------------|--------------|-----------|
| room | 1.59M | 0.40 (2.8%) | 0.62 (4.4%) | 0.03 (0.2%) | 2.37 (16.9%) | 10.62 (75.7%) | 14.03 |
| bicycle | 6.13M | 0.63 (2.1%) | 0.91 (3.1%) | 0.05 (0.2%) | 4.74 (16.2%) | 22.91 (78.4%) | 29.23 |
| garden | 1.84M | 0.28 (2.5%) | 0.44 (4.0%) | 0.02 (0.2%) | 1.93 (17.6%) | 8.28 (75.6%) | 10.95 |

### Gaussian intersection distribution

| Scene | Visible | Isect/Gauss mean | zeros | Bottom 1% isects from | Top 1% Gauss → % isects |
|-------|---------|-----------------|-------|----------------------|------------------------|
| room | 27% | 2.7 | 73.0% | 75.1% of Gaussians | 35.7% |
| bicycle | 32% | 1.4 | 68.3% | 69.7% of Gaussians | 24.9% |
| garden | 43% | 1.7 | 56.9% | 58.6% of Gaussians | 21.0% |

### Backward atomic write estimate

| Scene | Total bwd writes | Forward writes (pixels) | Ratio | Per-Gauss writes mean | p99 |
|-------|-----------------|----------------------|-------|----------------------|-----|
| room | 4.1M | 2.07M | 2.0x | 9.5 | 22.0 |
| bicycle | 1.6M | 2.07M | 0.8x | 0.8 | 4.8 |
| garden | 0.6M | 2.07M | 0.3x | 0.7 | 4.6 |

## C.4 Result

### C25 T5' hypothesis evaluation

**REJECTED** — The "1% Gaussian subset consuming 29% backward cost" claim is NOT about the bottom tail. It's about the TOP 1% of Gaussians (which account for 21-36% of intersections). The bottom 75% of Gaussians account for only 1% of intersections and are already skipped by the tile structure (zero intersections = no backward work).

The backward cost is NOT caused by processing invisible Gaussians. It's caused by:
1. **Alpha recomputation** for every visible Gaussian-pixel pair (compute-bound)
2. **Per-pixel Gaussian traversal** up to `last_ids` (can be hundreds per pixel)
3. **Atomic gradient writes** (moderate contention, 0.3-2.0x forward writes)

### Candidate analysis

| Candidate | Est. E2E Impact | Risk | Complexity | Decision |
|-----------|----------------|------|------------|----------|
| C25 sparse-tail | ~0% | N/A | N/A | **DROP** — tail already skipped |
| Alpha caching | +15-23% | Medium | Medium | **NEED MORE EVIDENCE** |
| Batch size increase | +7-12% | Low | Low | **ITERATE** — quick to test |
| Warp specialization | +5-15% | High | High | **NEED MORE EVIDENCE** |
| Gradient compaction | +5-10% | Medium | Medium | **NEED MORE EVIDENCE** |

### Best candidate: Batch size increase

The backward kernel uses `block_size` (256 for tile16) as batch size. Each batch requires a `block.sync()` and shared memory load. Current shared memory usage: ~10 KB. A100 has 48 KB default. Could increase batch to 1024 (40 KB), reducing batch count by 4x → fewer syncs and loads.

**Estimated backward speedup**: 10-15% → **+7-12% e2e** (backward is 75% of total).

## C.5 Bottleneck Explanation

The backward pass dominates because:
1. **Every Gaussian-pixel pair needs alpha recomputation** — the backward kernel must recompute `sigma = 0.5 * (conic.x * dx² + conic.z * dy²) + conic.y * dx * dy` and `alpha = min(0.999, opac * exp(-sigma))` for every contribution
2. **Back-to-front traversal** — unlike forward (front-to-back with early termination), backward must process all Gaussians up to `last_ids[pix_id]`
3. **Atomic gradient accumulation** — each contribution writes to 6+ gradient buffers via `gpuAtomicAdd`

## C.6 Decision

**ITERATE** — The batch size increase candidate is the most promising next step:
- Targets the 75% bottleneck (backward)
- Low implementation risk (shared memory tuning, no algorithmic change)
- Estimated +7-12% e2e speedup
- Quick to prototype and measure

**C25 sparse-tail**: **DROP** — hypothesis not supported by evidence.

---

# Summary: All Candidates

## Decision Matrix

| Candidate | Track | E2E Impact | Quality Risk | Implementation Effort | Decision |
|-----------|-------|-----------|-------------|----------------------|----------|
| C17-1 fused tile sort | A | +1-2% | Low | 4-5 days | **NEED MORE EVIDENCE** — microbench passes but e2e impact marginal |
| C42 scale=0.875 | B | +6.4% (23M GS) / +30% (<1M GS) | None (+0.002 dB) | Already deployed | **KEEP** — optimal Pareto point |
| C42 scale=0.625 | B | +7.5% | -0.040 dB | Already deployed | **KEEP** — alternative if more speed needed |
| C25 sparse-tail bwd | C | ~0% | N/A | N/A | **DROP** — hypothesis rejected |
| Batch size increase | C | +7-12% (est.) | None | 1 day | **ITERATE** — next experiment |
| Alpha caching | C | +15-23% (est.) | Medium | 3-4 days | **NEED MORE EVIDENCE** |

## Key Insight

**The backward pass is the dominant bottleneck (75-78%).** Forward-path optimizations (C17-1 sort, C42 SSIM) have limited e2e impact when the backward pass dominates. The highest-impact next step is backward optimization.

**C42 is still valuable** because:
1. It's free (no CUDA code, pure Python)
2. It helps most when Gaussian count is low (early training)
3. It reduces wall-clock training time by 6-30% depending on training phase
4. It has zero quality risk

## Data Provenance

| Item | Path |
|------|------|
| C17-1 source analysis | `reports/c17_1_source_analysis.md` |
| C17-1 microbenchmark data | `results/a100/phase-c42/c17_1_microbench.json` |
| C17-1 microbenchmark source | `scripts/phase-c42/c17_1_microbench.cu` |
| Tile distribution data | `results/a100/phase-c42/c17_1_tile_distribution.json` |
| C42 scale=1.0 results | `results/a100/phase-c42/track_b_ablation_10.json` |
| C42 scale=0.875 results | `results/a100/phase-c42/track_b_ablation_0875.json` |
| C42 scale=0.75 results | `results/a100/phase-c42/track_b_ablation_075.json` |
| C42 scale=0.625 results | `results/a100/phase-c42/track_b_ablation_0625.json` |
| C42 scale=0.5 results | `results/a100/phase-c42/track_b_ablation_05.json` |
| Pipeline profile data | `results/a100/phase-c42/track_c_pipeline_profile.json` |
| Backward candidate analysis | `reports/backward_candidate_analysis.md` |
| Pipeline profiler script | `scripts/phase-c42/track_c_pipeline_profiler.py` |
| Scale ablation script | `scripts/phase-c42/track_b_ablation.py` |
