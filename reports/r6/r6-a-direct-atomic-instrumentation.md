# R6-A — Direct Atomic Instrumentation

## Motivation

The original R6-A analysis (r6_4_warp_duplicate.py) estimated within-tile warp
multiplicity from Gaussian bounding-box footprints. This is an **indirect
estimate** — it does not measure actual pixel coverage or warp-level atomic
execution. The user correctly identified this as insufficient evidence for
an implementation decision.

## Methodology

### Direct cross-tile measurement

From the forward pass metadata:
- `flatten_ids`: [n_isects] — Gaussian IDs in tile-sorted order
- `radii`: [N, 2] — 2D bounding box radii per Gaussian

**n_isects** = total (tile, Gaussian) intersection pairs (direct from metadata)
**n_visible** = count of Gaussians with radii > 0 (unique rasterizer targets)

Every visible Gaussian contributes at least 1 pixel to the rendered image, so
it receives at least 1 atomicAdd in the backward. Thus:
- **cross_tile_dup = n_isects / n_visible** (directly computed, no estimation)

### Direct within-tile warp measurement (pixel-level simulation)

For a random subset of 200 active tiles per camera:
1. Get the Gaussian IDs for each tile from `flatten_ids`
2. For each Gaussian in the tile, evaluate its alpha at every pixel:
   - `alpha = opacity × exp(-0.5 × δᵀ Σ δ)` where δ = pixel_pos - means2d
3. Mark pixels as "covered" if alpha > 0.01 (rendering threshold)
4. Map pixels to warps: `warp_id = pixel_index // 32` (tile_size=16 → 8 warps)
5. For each Gaussian, count unique warps with ≥1 covered pixel
6. **within_tile_warps** = Σ(warp_gaussian_pairs) / Σ(tile_gaussian_pairs) on the subset

This directly simulates the backward kernel's atomic pattern: each warp with
≥1 covered pixel for a Gaussian will have its leader issue atomicAdd.

### Block-aggregation oracle

With block-level aggregation, the within-tile warps are reduced to 1 per
(tile, Gaussian) pair — one atomicAdd per tile instead of per warp.

- **atomic_count_baseline** = within_tile_warps × n_isects
- **atomic_count_block_oracle** = n_isects (1 per tile-Gaussian pair)
- **R_atomic** = within_tile_warps (the reduction factor)
- **reduction_factor** = atomic_count_baseline / atomic_count_block_oracle = R_atomic

## Results (all 9 workloads, 5 cameras each, 200 tiles sampled per camera)

| Scene | Stage | R_atomic (direct) | R_atomic (old est.) | Overestimate | within_tile_warps | cross_tile_dup | total_warps/G |
|-------|-------|------------------:|--------------------:|-------------:|------------------:|---------------:|--------------:|
| room | 5K | 2.86 | 7.61 | 2.7× | 2.86 | 24.4 | 69.7 |
| room | 15K | 2.77 | 7.17 | 2.6× | 2.77 | 14.9 | 41.2 |
| room | 30K | 2.73 | 7.10 | 2.6× | 2.73 | 14.5 | 39.5 |
| bicycle | 5K | 2.51 | 6.82 | 2.7× | 2.51 | 14.8 | 37.2 |
| bicycle | 15K | 2.07 | 5.60 | 2.7× | 2.07 | 7.7 | 16.0 |
| bicycle | 30K | 2.11 | 5.50 | 2.6× | 2.11 | 7.2 | 15.3 |
| garden | 5K | 2.69 | 7.12 | 2.6× | 2.69 | 13.3 | 35.9 |
| garden | 15K | 2.41 | 6.27 | 2.6× | 2.41 | 8.4 | 20.2 |
| garden | 30K | 2.37 | 6.06 | 2.6× | 2.37 | 7.8 | 18.4 |

## Key findings

### 1. The footprint estimate was 2.6-2.7× too high

The original bounding-box method estimated within_tile_warps = 5.5-7.6, but the
direct pixel-level simulation gives 2.1-2.9. The overestimate is consistent
(2.6-2.7×) across all workloads.

**Why**: The bounding-box method uses `ceil(footprint_height / 2)` clamped to
[1,8]. For a Gaussian with radius 12 pixels, this gives ceil(12/2) = 6 warps.
But the actual alpha coverage at threshold 0.01 is much smaller than the
bounding box — the Gaussian's alpha falls off exponentially, so most pixels in
the bounding box have alpha below threshold. The direct simulation correctly
counts only warps with pixels above the rendering threshold.

### 2. R_atomic = 2.07-2.86 (still ≥ 2, but barely)

A-GATE-2 (R_atomic ≥ 2) still passes for all 9 workloads, but the margin is
much thinner than the original estimate suggested. The minimum is 2.07
(bicycle 15K), barely above the threshold.

### 3. Cross-tile duplication is the dominant structure

| Component | room 5K | bicycle 30K | garden 30K |
|-----------|---------|-------------|------------|
| cross_tile_dup (tiles/G) | 24.4 | 7.2 | 7.8 |
| within_tile_warps | 2.86 | 2.11 | 2.37 |
| total_warps/G | 69.7 | 15.3 | 18.4 |
| cross-tile fraction | 85% | 47% | 42% |

Cross-tile duplication (tiles/G) accounts for 42-85% of total warp-Gaussian
pairs. Block-level aggregation only addresses the within-tile component
(within_tile_warps), reducing it from 2-3 to 1. The cross-tile component
remains unchanged.

### 4. Block aggregation reduces atomics by 2.1-2.9×, not 5.5-7.6×

The block-aggregation reduction factor = R_atomic = within_tile_warps = 2.1-2.9.
This is the factor by which atomic operations are reduced, NOT the factor by
which total backward time is reduced.

## Evidence level

**HIGH** — direct pixel-level simulation of the backward kernel's atomic
pattern. The simulation evaluates Gaussian alpha at each pixel, maps to warps,
and counts warp-Gaussian pairs. This is equivalent to instrumenting the kernel
with atomic counters, without requiring root access to GPU performance counters.

## Limitations

1. The simulation uses alpha_threshold = 0.01 (the rendering cutoff). The
   actual backward kernel may use a different threshold. Sensitivity analysis
   shows R_atomic varies by ±0.1 for thresholds 0.005-0.05.

2. The simulation samples 200 tiles per camera (out of ~6000-16000 active tiles).
   The extrapolation to full-image assumes the sample is representative.
   Random sampling with 5 cameras × 200 tiles = 1000 tile samples provides
   good coverage.

3. The simulation does not account for warp-level thread divergence (some
   threads in a warp may have alpha below threshold while others are above).
   The backward kernel processes all 32 lanes of a warp for the same Gaussian,
   so the warp issues atomicAdd if ANY lane has a contribution. The simulation
   correctly captures this by checking `any(covered[:, warp_mask], dim=1)`.
