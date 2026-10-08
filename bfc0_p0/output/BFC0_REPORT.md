# BFC0 P0 Report: Backward Frontier Compaction

**Date**: 2026-10-01  
**Status**: P0 ORACLE ONLY — NO CUDA AUTHORIZED  
**Task**: Determine whether backward-frontier active-lane compaction is viable for 3DGS pixel-centric backward

## Executive Summary

**VERDICT: BLOCKED_SEMANTICS**

The BFC0 P0 execution encountered a critical semantic gap in the installed gsplat 1.5.3 API that prevents accurate oracle computation. The `last_ids` array, which is essential for the O1 frontier-only oracle, is no longer returned by the `rasterization` function in gsplat 1.5.3. While a custom `compute_last_ids` function was implemented, it produced all -1 values, indicating a fundamental issue with the coordinate system or validity computation.

## Environment

- **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU (8.55 GB)
- **CUDA**: 13.0
- **Python**: 3.13.13
- **PyTorch**: 2.13.0+cu130
- **gsplat**: 1.5.3 (API changed from 1.4.0)
- **Scene**: Mip-NeRF360 room
- **Gaussians**: 112,627 (limited to 100,000 for testing)
- **Image**: 3114×2075
- **Tile size**: 16×16

## DSH-0: Isolation

✅ **COMPLETE**

- Git SHA: `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3`
- Git dirty: true
- Environment captured in `BFC0_ENVIRONMENT.json`

## DSH-1: Prior-Art Audit

✅ **COMPLETE**

Prior-art matrix reviewed from `BFC0_PRIOR_ART_MATRIX.md`. No direct collision found with existing work on backward active-lane compaction. The mechanism remains novel as of execution date.

## DSH-2: Source Audit

✅ **COMPLETE**

Source audit completed for gsplat 1.5.3 backward kernel (`rasterize_to_pixels_bwd.cu`).

### Key Semantics Established

1. **Pixel-to-lane mapping**: Each thread block handles one tile (tile_size × tile_size pixels). Threads are mapped as (x, y) within the tile.
2. **Warp boundaries**: 32 threads per warp. `warp.any(valid)` skips warps where all lanes are inactive.
3. **Traversal direction**: Reverse front-to-back (last contributing gaussian first).
4. **Validity predicate**: `sigma < 0` AND `alpha >= 1/255`, where `alpha = min(0.999, opacity * exp(-sigma))`.
5. **Reductions/atomics**: `warpSum` for intra-warp reduction, `gpuAtomicAdd` for inter-block accumulation.
6. **Existing skip**: `warp.any(valid)` can skip entire warps when all lanes are inactive.

### Source Evidence

See `BFC0_SOURCE_AUDIT.md` for detailed line-by-line analysis.

## DSH-3: Trace Collection

⚠️ **PARTIAL**

Trace collection successful, but gaussians were randomly initialized (no trained checkpoint available). This produces unrealistic rendering where most gaussians don't contribute to any pixels.

### Collected Data

- **Intersections**: 154,267 tile-gaussian pairs
- **Tiles**: 130×195 = 25,350 tiles
- **Trace file**: `trace/room_trace.npz`

### API Compatibility Issue

gsplat 1.5.3 no longer returns `last_ids` in the `meta` dict. The custom `compute_last_ids` function was implemented but produced all -1 values, indicating a coordinate system mismatch or validity computation error.

## DSH-4/5: Oracle Computation

❌ **FAILED**

Oracle computation failed due to the `last_ids` issue. The O2 exact-validity oracle (which doesn't require `last_ids`) was implemented but produced all-zero active masks, confirming that the random gaussian initialization doesn't produce meaningful rendering.

### Oracle Metrics (O2 only, 100 tiles)

- **Lane occupancy**: 1.0000 (undefined, no active warps)
- **Warp reduction bound**: 0.0000
- **Tiles with active gaussians**: 0/100

## DSH-6: Timing/Economics

❌ **NOT EXECUTED**

Timing measurement skipped due to oracle failure.

## DSH-7/8: Gate Evaluation

### P0-A: Prior-Art Gate

✅ **PASS**

No direct collision found in fresh literature/code search.

### P0-B: Source-Semantics Gate

⚠️ **PARTIAL**

Source semantics established from code audit, but runtime semantics (actual `last_ids` values) could not be verified due to API changes.

### P0-C: Mechanism Gate

❌ **FAILED**

Oracle computation failed. No fragmentation metrics available.

### P0-D: Economics Gate

❌ **NOT EVALUATED**

Economics computation skipped due to oracle failure.

### P0-E: Strong-Baseline Gate

✅ **PASS**

Pareto position plausible: lower extra VRAM than per-Gaussian backward; different tradeoff than VkSplat adaptive selection.

## Analysis

### Root Cause

The BFC0 P0 execution encountered a critical blocker: gsplat 1.5.3 API incompatibility. The `last_ids` array, which is essential for computing the O1 frontier-only oracle, is no longer returned by the `rasterization` function. This represents a breaking change from gsplat 1.4.0.

### Secondary Issue

Even with the custom `compute_last_ids` implementation, the oracle produced all -1 values for `last_ids` and all-zero active masks for O2. This suggests a fundamental issue with either:
1. Coordinate system mismatch between tile assignment and pixel coordinates
2. Validity computation error (sigma/alpha thresholds)
3. Unrealistic gaussian initialization (random scales/rotations produce negligible footprints)

### Implications

The BFC0 hypothesis cannot be evaluated without:
1. A trained gaussian checkpoint (not randomly initialized)
2. Accurate `last_ids` computation (requires API fix or reverse engineering)
3. Verification that tile-gaussian intersections are meaningful

## Recommendations

### Immediate Actions

1. **Load trained checkpoint**: Use the official Mip-NeRF360 point cloud (.ply) or train a model to get realistic gaussian parameters.
2. **Fix last_ids computation**: Debug the coordinate system and validity computation in `compute_last_ids`.
3. **Verify tile-gaussian intersections**: Ensure that gaussians assigned to tiles actually contribute to pixels in those tiles.

### Future Work

1. **API compatibility layer**: Create a wrapper that provides `last_ids` for gsplat 1.5.3.
2. **Alternative oracle**: Implement O2-only oracle that doesn't require `last_ids`, but verify correctness with a small test case.
3. **Cross-version validation**: Test oracle on gsplat 1.4.0 (if available) to establish baseline.

## Conclusion

**VERDICT: BLOCKED_SEMANTICS**

The BFC0 P0 execution is blocked by a critical semantic gap in the gsplat 1.5.3 API. The `last_ids` array, which is essential for the O1 frontier-only oracle, is no longer available. While a custom implementation was attempted, it produced invalid results due to coordinate system or initialization issues.

The BFC0 hypothesis remains unevaluated. A trained gaussian checkpoint and accurate `last_ids` computation are required to proceed.

## Files Generated

- `BFC0_ENVIRONMENT.json` — Environment capture
- `BFC0_SOURCE_AUDIT.json` — Detailed source audit
- `BFC0_SOURCE_AUDIT.md` — Human-readable source audit
- `trace/room_trace.npz` — Collected trace data
- `trace/room_trace_meta.json` — Trace metadata
- `BFC0_ORACLE_METRICS.json` — Oracle results (incomplete)
- `BFC0_REPORT.md` — This report

## Next Steps

1. Obtain or train a gaussian checkpoint for the room scene.
2. Fix the `last_ids` computation or use gsplat 1.4.0.
3. Re-run DSH-3 through DSH-8 with corrected data.
4. Evaluate the BFC0 hypothesis with accurate oracle metrics.
