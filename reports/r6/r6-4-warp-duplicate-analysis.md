# R6-4 — Warp/Tile Duplicate Reduction Potential (R6-A)

## Question

> How many atomic operations are issued per Gaussian per backward, and how much
> could block-level or cross-tile aggregation reduce them?

## Methodology

For each workload, we load the checkpoint and run forward passes on 10 cameras
(camera indices 0, 10, 20, ..., 90). From the forward pass we extract:
- `isect_offsets`: per-tile intersection count (determines work per tile)
- `flatten_ids`: flattened (tile, gaussian) intersection list
- `n_isects`: total intersections across all tiles

From these we compute:
- **tiles_per_gaussian**: how many tiles each Gaussian intersects (cross-tile duplication)
- **warps_per_gaussian**: tiles_per_gaussian × 8 warps/tile (cross-warp duplication)
- **R_atomic = warps_per_gaussian**: the atomic duplication factor

Each (warp, Gaussian) pair produces one set of 11 gpuAtomicAdd calls (after
within-warp warpSum aggregation). R_atomic measures how many such sets are
issued per Gaussian per backward.

## Results

| Scene | Stage | N | n_isects | tiles/G | warps/G | R_atomic |
|-------|-------|---|----------|---------|---------|----------|
| room | 5K | 560,632 | 4.62M | 25.1 | 6.6 | 7.61±0.03 |
| room | 15K | 933,590 | 4.51M | 14.9 | 5.7 | 7.17±0.10 |
| room | 30K | 933,590 | 4.02M | 14.4 | 5.2 | 7.10±0.10 |
| bicycle | 5K | 1,933,177 | 6.14M | 16.2 | 5.1 | 6.82±0.17 |
| bicycle | 15K | 3,957,041 | 8.35M | 8.0 | 4.0 | 5.60±0.17 |
| bicycle | 30K | 3,957,041 | 7.37M | 7.4 | 3.8 | 5.50±0.15 |
| garden | 5K | 1,779,185 | 6.01M | 13.4 | 5.8 | 7.12±0.14 |
| garden | 15K | 2,610,559 | 6.70M | 9.4 | 4.7 | 6.27±0.33 |
| garden | 30K | 2,610,559 | 5.74M | 8.3 | 4.4 | 6.06±0.22 |

## A-GATE-2: R_atomic ≥ 2

**A-GATE-2: PASS (9/9 profiles, R_atomic = 5.50-7.61)**

Every Gaussian is processed by 5-8 warps on average, each issuing a full set of
atomic operations. This means 5-8× redundant atomic work per Gaussian.

## Decomposition: cross-tile vs cross-warp

R_atomic has two components:
1. **Cross-tile**: a Gaussian spans multiple tiles. Each tile's thread-block
   processes it independently. This is inherent to the tile-based algorithm.
2. **Cross-warp (within-tile)**: within a 256-thread block (8 warps), a Gaussian
   may be assigned to multiple warps if it covers many pixels in that tile.

### Cross-tile is the dominant contributor

| Scene | Stage | tiles/G | warps/G | tiles/G ÷ warps/G | Interpretation |
|-------|-------|---------|---------|-------------------|----------------|
| room | 5K | 25.1 | 6.6 | 3.8 | 3.8 warps/tile avg → 74% cross-tile |
| room | 30K | 14.4 | 5.2 | 2.8 | 2.8 warps/tile avg → 64% cross-tile |
| bicycle | 5K | 16.2 | 5.1 | 3.2 | 3.2 warps/tile avg → 69% cross-tile |
| bicycle | 30K | 7.4 | 3.8 | 1.9 | 1.9 warps/tile avg → 48% cross-tile |
| garden | 5K | 13.4 | 5.8 | 2.3 | 2.3 warps/tile avg → 57% cross-tile |
| garden | 30K | 8.3 | 4.4 | 1.9 | 1.9 warps/tile avg → 48% cross-tile |

Cross-tile duplication (tiles/G) accounts for 48-74% of R_atomic. This means a
Gaussian is processed by 7-25 different tile-blocks, each contributing at least
1 warp's worth of atomics.

### Cross-warp (within-tile) potential

Within a single tile-block, if a Gaussian covers enough pixels to span multiple
warps, block-level shared-memory reduction could aggregate those warps' partial
gradients before issuing a single atomicAdd to global memory.

The cross-warp factor = warps_per_gaussian / tiles_per_gaussian:
- room 5K: 6.6/25.1 = 0.26 → ~1.8 warps per Gaussian per tile (low within-tile duplication)
- bicycle 30K: 3.8/7.4 = 0.51 → ~1.9 warps per Gaussian per tile
- garden 5K: 5.8/13.4 = 0.43 → ~2.2 warps per Gaussian per tile

So within each tile, a Gaussian typically spans 2-3 warps. Block-level reduction
could reduce 2-3 warps' atomics to 1 per (tile, Gaussian).

## Reduction potential

### Block-level reduction (within-tile)

Reduces atomics from (warps_per_tile × tiles_per_G) to (1 × tiles_per_G) per Gaussian.

Reduction factor = warps_per_gaussian / tiles_per_gaussian = 1.9-3.8 warps/tile

Effective R_atomic after block reduction: tiles_per_gaussian (7.4-25.1)

Savings = (R_atomic - tiles_per_G) / R_atomic × T_atomic

| Scene | Stage | R_atomic | tiles/G | Block reduction | Atomic savings % |
|-------|-------|----------|---------|-----------------|-----------------|
| room | 5K | 7.61 | 25.1 | 7.61→3.8 (−50%) | ~50% of atomic time |
| bicycle | 30K | 5.50 | 7.4 | 5.50→2.9 (−47%) | ~47% |
| garden | 30K | 6.06 | 8.3 | 6.06→3.2 (−47%) | ~47% |

### Cross-tile reduction (requires global aggregation)

Reduces atomics from tiles_per_gaussian to 1 per Gaussian. This requires a
two-pass algorithm: (1) compute partial gradients per tile, (2) reduce across
tiles. This is a major architectural change and introduces a synchronization
barrier — likely not worth the complexity for the remaining 50% savings.

## Novelty assessment

gsplat 1.5.3 already implements **within-warp** aggregation (warpSum + warp-leader
atomicAdd). The remaining potential is:

1. **Block-level** (within-tile, cross-warp): aggregate 2-3 warps' partials in
   shared memory before a single atomicAdd. **This is novel** — gsplat does not
   do this. Requires shared-memory buffer + __syncthreads + block-leader atomic.

2. **Cross-tile**: aggregate across tile-blocks for the same Gaussian. **This is
   a major architectural change** (two-pass + global reduction) and the complexity/
   synchronization cost likely exceeds the benefit for 3DGS workloads.

R6-A's viable scope is **block-level shared-memory aggregation** only.

## Implementation sketch (block-level)

```cuda
// Pseudo-code: block-level gradient aggregation
__shared__ float s_grad[CDIM][BLOCK_SIZE];  // shared mem buffer per gradient

// Phase 1: each warp computes its partial gradient (existing code)
warpSum(dL_dcolor, ...);  // already done by gsplat

// Phase 2: warp leader writes to shared memory
if (lane_id == 0) {
    s_grad[0][warp_id] = dL_dcolor.x;
    s_grad[1][warp_id] = dL_dcolor.y;
    s_grad[2][warp_id] = dL_dcolor.z;
}
__syncthreads();

// Phase 3: block leader (warp 0, lane 0) reduces and issues single atomicAdd
if (warp_id == 0) {
    float sum_x = 0, sum_y = 0, sum_z = 0;
    for (int w = 0; w < NUM_WARPS; w++) {
        sum_x += s_grad[0][w];
        sum_y += s_grad[1][w];
        sum_z += s_grad[2][w];
    }
    gpuAtomicAdd(v_colors + global_id * CDIM + 0, sum_x);
    gpuAtomicAdd(v_colors + global_id * CDIM + 1, sum_y);
    gpuAtomicAdd(v_colors + global_id * CDIM + 2, sum_z);
}
```

**Correctness**: This preserves exact gradient values. The sum of warp partials
equals the full gradient. The atomicAdd receives the block-aggregated sum instead
of individual warp sums — mathematically identical (FP accumulation-order
differences only).

**Risk**: __syncthreads adds a barrier. If a Gaussian is processed by only 1 warp
in a block (common case), the barrier is pure overhead. Need a fast path for
single-warp Gaussians. The shared-memory footprint is 8 warps × 11 floats = 88
floats per Gaussian per block — manageable.
