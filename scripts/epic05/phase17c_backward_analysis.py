"""
Backward kernel data-flow analysis for differential membership.
Read-only: traces what the backward kernel actually does with flatten_ids.
"""
print("=" * 70)
print("BACKWARD KERNEL DATA-FLOW: flatten_ids ACCESS PATTERN")
print("=" * 70)

print("""
Source: RasterizeToPixels3DGSBwd.cu

Key observation: the backward kernel does NOT traverse flatten_ids linearly.
It processes per-pixel, not per-intersection.

Current backward kernel flow:

for each pixel (i, j):
    tile_id = tile_grid_position(i, j)
    range_start = tile_offsets[tile_id]
    range_end = tile_offsets[tile_id + 1]  or  n_isects  for final tile
    bin_final = last_ids[pixel]
    
    # Reverse iteration over gaussians in this tile
    for batch in range(num_batches):
        batch_end = range_end - 1 - block_size * batch
        batch_size = min(block_size, batch_end + 1 - range_start)
        idx = batch_end - thread_rank
        if idx >= range_start:
            g = flatten_ids[idx]        # THE PRIMARY ACCESS
            # Attribute loads using g
            means2d[g], conics[g], colors[g], opacities[g]
            # Gradient scatter-add
            gpuAtomicAdd(&d_means2d[g], ...)
            gpuAtomicAdd(&d_conics[g], ...)
            gpuAtomicAdd(&d_colors[g], ...)
            gpuAtomicAdd(&d_opacities[g], ...)

CRITICAL: The access is:
1. INDEX-BASED: flatten_ids[idx] where idx is computed from batch_end - thread_rank
2. PER-TILE: each tile has [range_start, range_end) range into flatten_ids
3. REVERSE ORDER: batches go from back to front
4. RANDOM GAUSSIAN ACCESS: once g is loaded, it's used as index into attribute arrays

For differential representation:
- Q1: Can flatten_ids[idx] be recovered in O(1)?
  NO — if flatten_ids is differential (reference + delta), every idx access
  requires checking whether the position falls in the reference range or
  requires the current-tile delta. This adds a branch per access.

- Q2: Can we scan tile_offsets[t] → tile_offsets[t+1]?
  YES — tile_offsets is unchanged. But the flatten_ids in that range would
  need on-the-fly reconstruction (reference + delta merge).

- Q3: Reverse traversal (back to front)?
  NO — the backward kernel iterates range_end-1 down to range_start.
  With differential encoding, the last element is easy (it's the last
  element of either reference or delta). But traversing backward while
  merging two lists requires either:
  (a) materializing the full merged list first, or
  (b) a two-pointer backward merge

- Q4: Can we avoid materializing the full tile list?
  YES — IN THEORY. A two-pointer merge of reference + delta works:
    - Maintain cursor on reference tile's IDs
    - Maintain cursor on insertions (sorted by depth)
    - At each step, emit the one with larger depth (reverse traversal)
  HOWEVER: this adds per-step comparison logic to the backward kernel,
  which currently is a simple flatten_ids[idx] lookup.

KEY INSIGHT: the backward kernel accesses flatten_ids heavily but
STRICTLY SEQUENTIALLY per tile. Each tile's range is traversed in
reverse order, fully, and never revisited. This means:

1. Materializing one tile at a time is acceptable
2. But materializing requires merging reference + delta
3. The merge is a linear scan over both lists (O(|S| + |I| + |R|))
4. This is asymptotically O(|M_B|) — same as current flatten_ids scan
5. Constants are higher due to merge overhead

For a chain of dependencies (T0 = full, T1 = delta(T0), T2 = delta(T1)):
- Reconstructing T2 requires first reconstructing T1
- This is a chain of length: materialize_count
- For worst-case access (a tile at depth D), you need D merges
- Cumulative cost: O(sum of all tiles in chain)

CONCLUSION:
- Storage saving: possible (delta is ~40-60% of full list by element count)
- Reconstruction cost: O(|M_B|) per tile — same asymptotic as current
  (current: 1 read per element; differential: 1 merge operation per element)
- Chain dependency: unacceptable for chains > 1
- Random access: not compatible without full materialization
- Per-tile reconstruction (no chain, reference = adjacent tile stored separately):
  Feasible. Each tile's membership = merge(sorted(ref_tile), sorted(insertions))

RECOMMENDED MODEL for further analysis:
- Store EVERY N-th tile as full reference
- Store deltas for intermediate tiles referencing the nearest full tile
- This bounds reconstruction depth to N
- Choose N so that reference overhead + delta overhead < full storage
""")
