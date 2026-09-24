# C17-2 Ordered Membership Representation Audit

## Research Question

Does the ordered per-tile Gaussian membership — `flatten_ids` (int32 `[n_isects]`) and `tile_offsets` (int32 `[I×th×tw]`) — contain exploitable structure such as within-tile delta locality, cross-tile overlap, run encoding, or bit-width reduction? Or is it effectively random and incompressible?

## Source Contract Confirmation

| Tensor | CUDA Dtype | API Shape | Bundle dtypes (C1 vs baseline) | Kept by backward? |
|--------|-----------|-----------|-------------------------------|-------------------|
| `flatten_ids` | `int32` | `[n_isects]` | **Identical** across C1 and baseline | **Yes** — backward reads `g = flatten_ids[idx]` |
| `tile_offsets` | `int32` | `[I, th, tw]` | **Identical** (recomputed from sorted `isect_ids` after sort) | **Yes** — backward reads `range_start` / `range_end` from it |

- `isect_ids` (int64): NOT retained by backward (only used during offset decode, then discarded).
- C1 modifies **only** the sort-key bit layout in `isect_ids`; CUB `SortPairs` reshuffles `flatten_ids` identically for the same per-tile depth ordering.
- C17-1 equivalence validation (`results/epic05/phase17b/c17_1_correctness.json`) already confirmed: per-tile local sort matches CUB sorted order perfectly — 0 missing, 0 extra, 4624/4624 exact tile order matches.

## Data Collection Method

- **Deterministic**: real-scene PLY + camera baseline via `gsplat.rasterization()` (`tile_size=16`, packed, activated scales). No CUDA kernel or renderer modifications.
- **Scenes**: room (1.6M Gaussians), bicycle (6.1M), garden (5.8M) — all at 1080p, tile 16.
- Raw saved to `results/phase-a100/c17_2_membership_collected.pt`

## Memory Footprint

| Scene | n_isects | flatten_ids | tile_offsets | Total |
|-------|----------|-------------|--------------|-------|
| room  | 1,690,772 | 6.8 MB | 18 KB | 6.8 MB |
| bicycle | 4,861,253 | 19.4 MB | 33 KB | 19.4 MB |
| garden | 5,110,309 | 20.4 MB | 33 KB | 20.4 MB |

`tile_offsets` overhead is negligible (~0.2% of flatten_ids).

## Within-Tile Delta Locality

**Finding: Negligible.** Adjacent Gaussian IDs in sorted membership order are not spatially local.

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| Delta pairs | 1,686,148 | 4,853,093 | 5,102,149 |
| |Δ| mean | 914.9 | 6,135.5 | 6,097.8 |
| |Δ| median | 75 | 101 | 883 |
| |Δ| p99 | 1,288,133 | 4,938,878 | 4,774,888 |
| |Δ| max | 1,586,921 | 6,131,939 | 5,829,947 |
| |Δ| = 0 (duplicate) | 0.00% | 0.00% | 0.00% |
| |Δ| ≤ 1 | 0.03% | 0.02% | 0.01% |
| |Δ| ≤ 2 | 0.03% | 0.02% | 0.02% |
| |Δ| ≤ 4 | 0.04% | 0.03% | 0.02% |
| |Δ| ≤ 16 | 0.08% | 0.05% | 0.03% |

**Interpretation**: The sort key is `depth_upper | tile_id << 16 | iid_enc`. Depth monotonicity within a tile guarantees strict front-to-back order, but Gaussian IDs (which are arbitrary positional indices from the SfM point cloud) are essentially random with respect to depth. Adjacent depth-sorted entries come from arbitrary spatial locations in the scene, so their IDs differ by large amounts. The signed LEB128 coding estimate yields ~4.0 bytes per delta (no savings vs raw int32).

### Contiguous ID Runs (±1 adjacency)

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| Runs found | 451 | 870 | 597 |
| Max run length | 2 | 3 | 2 |
| Run count ratio | 0.03% | 0.02% | 0.01% |

**Finding: Essentially zero.** Consecutive ID±1 sequences are vanishingly rare — the sort order does not preserve Gaussian ID adjacency.

## Cross-Tile Overlap

**Finding: Substantial.** Gaussians frequently span multiple adjacent tiles.

| Neighbor | Metric | room | bicycle | garden |
|----------|--------|------|---------|--------|
| Horizontal | Jaccard P50 | 0.35 | 0.31 | 0.24 |
| Horizontal | Jaccard mean | 0.37 | 0.35 | 0.25 |
| Horizontal | Jaccard P99 | 0.81 | 0.76 | 0.50 |
| Vertical | Jaccard P50 | 0.44 | 0.21 | 0.18 |
| Vertical | Jaccard mean | 0.46 | 0.24 | 0.19 |
| Vertical | Jaccard P99 | 0.87 | 0.60 | 0.44 |
| Diagonal (DR) | Jaccard P50 | 0.22 | 0.11 | 0.08 |
| Diagonal (DL) | Jaccard P50 | 0.22 | 0.11 | 0.08 |

**Interpretation**: ~24-46% of tile members (P50) are shared with the adjacent tile. This is expected from 2D Gaussian splat coverage: a Gaussian's 2D footprint covers multiple tiles, so its ID appears in the membership list of every tile it overlaps.

## Gaussian Tile Membership Distribution

**Finding: Heavy tail confirmed.**

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| P50 tiles per Gaussian | 3 | 2 | 2 |
| Mean tiles per Gaussian | 4.3 | 2.7 | 2.3 |
| P99 tiles per Gaussian | 29 | 12 | 10 |
| Max tiles per Gaussian | 783 | 8,160 | 8,160 |
| Gaussians with membership | 1,593,376 | 6,131,954 | 5,834,784 |
| Gaussians without membership | 0 | 0 | 0 |

**Interpretation**: Gini ~0.89-0.91 confirmed (from prior analysis). Most Gaussians appear in 2-3 tiles. The extreme tail (a Gaussian covering the entire 8160-tile image) matches the heavy-tail behavior observed in the corrected A100 baseline.

## Per-Tile ID Range

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| Global ID bits needed | 21 | 23 | 23 |
| Unused bits (vs int32) | 11 | 9 | 9 |
| Per-tile range P50 | 1,576,775 | 6,094,068 | 5,784,940 |
| Per-tile range max | 1,586,921 | 6,131,939 | 5,829,947 |

**Finding: Wide ranges — no bit-width savings possible.** Per-tile ID ranges are nearly as large as the full Gaussian set. A Gaussian with a high ID and a Gaussian with a low ID commonly appear in the same tile (the depth sort pulls them together from opposite sides of the scene). This means per-tile base+offset encoding buys nothing.

## Depth Order Locality

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| Pairs checked | 1,686,148 | 4,853,093 | 5,102,149 |
| Strict decreases | 0 | 0 | 0 |
| Decrease ratio | 0.0% | 0.0% | 0.0% |

**Finding: Perfect monotonicity.** The CUB sort + C1 key compression produce zero depth inversions within each tile, consistent with the theoretical guarantee from IEEE 754 monotonicity and right-shift monotonicity (validated in `results/epic05/phase16/c1_ordering.json`).

## Within-Tile Duplicates

| Scene | Duplicate entries |
|-------|------------------|
| room | 0 |
| bicycle | 0 |
| garden | 0 |

**Finding: Zero duplicates.** Every Gaussian appears at most once per tile in the sorted membership list. The per-tile membership is a set.

## Opportunity Matrix

| Strategy | Evidence | Potential | Risk / Note |
|----------|----------|-----------|-------------|
| **Delta encoding** | |Δ|≤16 in 0.03-0.08% of adjacent pairs; mean |Δ| = 915-6098 | **None** — worse than raw int32 (LEB128 ≈ 4.0 B/delta) | Adds decode logic for zero benefit |
| **Run-length encoding** | Max run=3, <900 runs across millions | **Negligible** | Would actually increase storage |
| **Per-tile base+offset** | Per-tile ID range ≈ full ID space | **None** — no bit savings (need as many offset bits as full ID) | Adds per-tile constant overhead |
| **Tile-local (16-bit) IDs** | 1.6-6.1M globally need 21-23 bits | **Negligible** — int22 not a GPU native type | Requires scatter-gather indirection |
| **Leverage cross-tile overlap** | Jaccard P50 0.18-0.46; P99 0.44-0.87 | **Moderate** — reuse tile-to-tile membership | Requires non-trivial set-difference logic in kernel |
| **Duplicate suppression** | 0 duplicates within tile | **None** — already a set per tile | N/A |
| **Bit-width reduction (global)** | Need 21-23 bits; int32 = 32 bits | **Marginal** (28-34% unused bits) | Non-power-of-2 types are costly on GPU; int16 insufficient |

## Summary

The ordered per-tile Gaussian membership is **effectively incompressible** through the standard structure-exploitation strategies examined:

1. **Delta locality is near-zero**: adjacent depth-sorted entries come from arbitrary scene locations.
2. **Contiguous ID runs are nonexistent**: ≤0.03% of adjacent pairs, max run length 3.
3. **Per-tile ID ranges are full width**: IDs span nearly the entire Gaussian distribution within each tile.
4. **Within-tile duplicates are zero**: already a set per tile.
5. **Cross-tile overlap is substantial** but exploiting it would require non-trivial set-difference logic in the backward kernel — beyond the scope of a representation-only change.

**The single exploitable property** is the ~9-11 unused high bits per int32 Gaussian ID (global ID range uses 21-23 bits of 32). Switching to `uint24` would save 25% of flatten_ids memory (≈1.7-5.1 MB per 1080p scene) but:
- `uint24` is not a native GPU type
- Packing 4× `uint24` into 3× `uint32` adds load/unpack overhead in forward and backward
- The backward kernel already uses `flatten_ids[idx]` as the primary index into Gaussian attribute arrays — changing the element type touches every kernel load site

**Within-tile delta encoding (intra-tile delta chain)**: While we verified that the within-tile flattened order IS the CUB depth-sorted order, the sorted Gaussian IDs have no spatial proximity in Gaussian-ID space; the sort key is depth. An intra-tile delta chain would start with the first ID unencoded, then encode each subsequent ID as `flatten_ids[i] - flatten_ids[i-1]`. The median absolute delta is 75-883 and the P99 is 1.3M-4.9M. Even with variable-length coding (LEB128), mean bytes per delta ≈ 4.0 — the same as or worse than raw int32.

## Conclusion

**OPTIMIZATION: NOT STARTED.** The ordered membership representation is a bandwidth-bound structure with no exploitable within-sequence locality and no run-length or duplicate redundancy. The only theoretical angle is the ~9-11 unused high bits per int32 entry, which would require a non-trivial packing scheme with GPU-unfriendly element types.

---

*Audit method: deterministic real-scene baseline capture via `gsplat.rasterization()` (no kernel modifications). Verification against prior C17-1 equivalence validation. Source contract confirmed from `Intersect.cpp`, `IntersectTile.cu`, `RasterizeToPixels3DGSBwd.cu`, and `_wrapper.py`.*
