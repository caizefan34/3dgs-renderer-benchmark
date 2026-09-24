# C17-1 Source Analysis — Phase A1

## gsplat 1.5.3 Tile Sort Pipeline: Complete Code Walkthrough

---

## 1. Pipeline Overview

The gsplat 3DGS rendering pipeline has the following stages:

```
Projection (fully_fused_projection)
  ↓ means2d, radii, depths, conics
Pass 1: intersect_tile_kernel (count tiles per Gaussian)
  ↓ tiles_per_gauss
cumsum (PyTorch at::cumsum)
  ↓ cum_tiles_per_gauss, n_isects
Pass 2: intersect_tile_kernel (write isect_ids + flatten_ids)
  ↓ isect_ids[N_isects] (64-bit), flatten_ids[N_isects] (32-bit)
Global Sort: CUB DeviceRadixSort (46-bit keys, 6 passes)
  ↓ sorted isect_ids, sorted flatten_ids
Offset Kernel: intersect_offset_kernel
  ↓ tile_offsets[I, tile_h, tile_w]
Rasterize Forward: rasterize_to_pixels_3dgs_fwd_kernel
  ↓ render_colors, render_alphas, last_ids
Rasterize Backward: rasterize_to_pixels_3dgs_bwd_kernel
  ↓ gradients
```

---

## 2. Where Are Unsorted Intersections Created?

**File**: `IntersectTile.cu`, `intersect_tile_kernel()`, lines 24–114
**Host dispatch**: `Intersect.cpp`, `intersect_tile()`, lines 99–116

### Pass 1 (lines 79–85):
Each Gaussian computes its tile bounding box and writes the count:
```cpp
tiles_per_gauss[idx] = (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x);
```

### Pass 2 (lines 87–114):
Each Gaussian writes one entry per intersected tile:
```cpp
int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
for (i = tile_min.y; i < tile_max.y; ++i) {
    for (j = tile_min.x; j < tile_max.x; ++j) {
        int64_t tile_id = i * tile_width + j;
        isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
        flatten_ids[cur_idx] = static_cast<int32_t>(idx);
        ++cur_idx;
    }
}
```

**Key 64-bit encoding** (line 108):
```
isect_ids = [image_id (upper bits) | tile_id (middle 32 bits) | depth (lower 32 bits)]
```
- `image_id` occupies bits [32+tile_n_bits, ...]
- `tile_id` occupies bits [32, 32+tile_n_bits)
- `depth` occupies bits [0, 32) (bit-reinterpret of float32)

**Data ordering after Pass 2**: Intersections are ordered by **Gaussian index**, not by tile. Within each Gaussian, tiles are written in row-major order. This means intersections for a given tile are **scattered** across the entire `isect_ids` array.

---

## 3. Where Does Global Ordering Become Necessary?

**File**: `IntersectTile.cu`, `radix_sort_double_buffer()`, lines 296–339
**Host dispatch**: `Intersect.cpp`, lines 135–143

```cpp
CUB_WRAPPER(
    cub::DeviceRadixSort::SortPairs,
    d_keys,       // isect_ids (64-bit)
    d_values,     // flatten_ids (32-bit)
    n_isects,
    0,            // begin_bit
    32 + tile_n_bits + image_n_bits,  // end_bit
    stream
);
```

**For 1920×1080, tile_size=16, I=1**:
- n_tiles = 120 × 68 = 8160
- tile_n_bits = floor(log2(8160)) + 1 = 13
- image_n_bits = floor(log2(1)) + 1 = 1
- end_bit = 32 + 13 + 1 = 46
- CUB radix = 8 bits → **6 passes** over the full array

**Why global sort is needed**: The rasterizer (forward kernel) reads intersections as contiguous ranges per tile:
```cpp
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = tile_offsets[tile_id + 1];  // or n_isects for last tile
```
Without sorting, intersections for a given tile are scattered across the array, making contiguous range access impossible.

**Global sort achieves two things simultaneously**:
1. **Grouping by tile**: All intersections of the same tile become contiguous
2. **Depth ordering within tile**: Intersections within each tile are sorted by depth (front-to-back)

---

## 4. Can tile_offsets Be Generated Without Separate Kernel?

**Current offset kernel**: `IntersectTile.cu`, `intersect_offset_kernel()`, lines 209–257

```cpp
// Each thread checks if (image_id, tile_id) changes between consecutive entries
int64_t isect_id_curr = isect_ids[idx] >> 32;  // strip depth
int64_t isect_id_prev = isect_ids[idx - 1] >> 32;
if (isect_id_prev != isect_id_curr) {
    // Fill offsets between prev and curr tile
    for (i = id_prev + 1; i <= id_curr; i++)
        offsets[i] = idx;
}
```

This kernel is O(n_isects) with 256 threads — for 8M intersections, ~31K thread blocks. It's relatively cheap (~0.05ms estimated) but still a separate kernel launch.

**Can it be fused?**

**Yes, if using tile-local sort**: If each CTA processes one tile and sorts its intersections in shared memory, it inherently knows:
- Its own tile_id
- The number of intersections it has
- The starting offset (from a prefix sum over per-tile counts)

The offset can be written directly: `tile_offsets[image_id * n_tiles + tile_id] = my_offset`

**No, if keeping the global sort**: The offset kernel depends on the sorted output. It must run after the sort completes. Fusing would require the sort kernel to also write offsets, which CUB doesn't support.

---

## 5. What Interface Must Remain Unchanged for Rasterizer?

The forward rasterizer (`RasterizeToPixels3DGSFwd.cu`, lines 85–89) reads:
```cpp
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = (image_id == I-1) && (tile_id == last) ? n_isects : tile_offsets[tile_id + 1];
```

The backward rasterizer (`RasterizeToPixels3DGSBwd.cu`, lines 88–92) reads the same:
```cpp
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = (image_id == I-1) && (tile_id == last) ? n_isects : tile_offsets[tile_id + 1];
```

Both also read `flatten_ids[idx]` to get the Gaussian index:
```cpp
int32_t g = flatten_ids[idx];  // index into [I*N] or [nnz]
```

The backward pass also reads `last_ids[pix_id]` (stored by forward pass):
```cpp
const int32_t bin_final = inside ? last_ids[pix_id] : 0;
```
`last_ids[pix_id]` is the index into the sorted `flatten_ids` array — the position of the last Gaussian that contributed to this pixel. **This means the sorted order must be identical between forward and backward.**

### Required interface (must not change):
1. **`tile_offsets`**: `[I, tile_height, tile_width]` int32 — starting position in flatten_ids for each tile. `tile_offsets[tile_id+1]` or `n_isects` (for last) gives the end.
2. **`flatten_ids`**: `[n_isects]` int32 — Gaussian indices, grouped by tile, sorted by depth within each tile.
3. **`n_isects`**: total number of intersections (unchanged by sort).
4. **`last_ids`**: `[I, H, W]` int32 — index into flatten_ids of last contributing Gaussian per pixel (written by forward, read by backward).

### What CAN change:
- The sort algorithm (as long as output is grouped by tile + depth-sorted within tile)
- The offset computation method
- Internal temporary buffers
- The number of kernel launches

### What CANNOT change:
- `flatten_ids` must contain the same Gaussian indices in the same tile-grouped, depth-sorted order
- `tile_offsets` must give correct contiguous ranges
- Tie-breaking for equal depths: the rasterizer's early termination (`done = true` when T < 1e-4) means that different tie-breaking can produce different `last_ids`, which would break backward pass consistency. **Forward and backward must use the same sorted order.**

---

## 6. C17-1 Design Space Analysis

### The fundamental challenge: Gaussian-parallel → tile-parallel transpose

After Pass 2, intersections are ordered by Gaussian. The rasterizer needs them ordered by tile. This transpose is what the global sort accomplishes.

### Option A: Pure tile-local sort (no global sort)

**Approach**: 
1. Build per-tile histogram (count intersections per tile)
2. Prefix sum to get per-tile offsets
3. Scatter intersections to tile-ordered positions
4. Each CTA loads its tile's intersections, sorts by depth in shared memory
5. Write sorted output + tile_offsets

**Problem**: Steps 1-3 are exactly what C17-0 attempted. The atomicAdd scatter (step 3) was the bottleneck — 2.5-4.6x slower than baseline. **This approach is already rejected.**

### Option B: Reduced-bit global sort + tile-local depth sort

**Approach**:
1. Global sort by tile_id only (14 bits → 2 passes instead of 46 bits → 6 passes)
2. Each CTA loads its tile's intersections (now contiguous after step 1)
3. Sort by depth in shared memory using `cub::BlockRadixSort` (32-bit keys, 4 passes)
4. Write sorted flatten_ids + tile_offsets

**Advantages**:
- Global sort: 6 passes → 2 passes = ~3x faster sort
- Tile-local depth sort: shared memory only, no global memory traffic
- Offset generation: fused with the grouping (offsets = prefix sum of per-tile counts, computed during sort)

**Disadvantages**:
- Still requires global sort (not eliminated, just reduced)
- Requires a separate kernel for the tile-local sort (new kernel launch)
- Per-tile sort has variable workload (some tiles have 100, others have 8000)

### Option C: Fused Pass2 + tile histogram + single-kernel sort

**Approach**:
1. Pass 2 writes isect_ids as before
2. New kernel: each CTA = one tile
   - Scan all isect_ids, extract those belonging to this tile (O(n_isects) per CTA)
   - Sort in shared memory by depth
   - Write to output

**Problem**: O(n_tiles × n_isects) = O(8160 × 8M) = 65 billion operations. Completely infeasible.

### Option D: Sort-free approach (change rasterizer to handle unsorted)

**Approach**: Modify the rasterizer to sort within shared memory at the start of each tile block, eliminating the separate sort entirely.

**Problem**: The rasterizer already uses shared memory for Gaussian batches (7168 bytes for tile16). Adding sort storage would require ~4096 × 12 bytes = 49KB additional shared memory, exceeding the 48KB limit. Also, the rasterizer's parallelization (1 block per tile) means each block would need to find its intersections in the unsorted array — the same scatter problem.

---

## 7. Recommended C17-1 Approach: Option B (Reduced-bit sort + BlockRadixSort)

### Rationale:
- C17-0 (Option A) already failed — scatter bottleneck is fundamental
- Option B reduces the global sort cost by 3x (6→2 passes) while adding cheap shared memory work
- `cub::BlockRadixSort` is a well-tested library kernel, low implementation risk
- Tile intersection distribution (from profiling) shows 99.77% of tiles fit in 48KB shared memory at tile16

### Pipeline comparison:

| Step | Baseline | C17-1 (Option B) |
|------|----------|------------------|
| Pass 2 | Write isect_ids + flatten_ids | Same |
| Sort | DeviceRadixSort 46-bit, 6 passes | DeviceRadixSort 14-bit, 2 passes |
| Tile sort | — | BlockRadixSort 32-bit, 4 passes (shared mem) |
| Offset | Separate kernel | Fused with tile sort |
| Total passes | 6 global | 2 global + 4 shared-mem per tile |

### Key question for Phase A2 microbenchmark:
Is `2-pass global sort + 4-pass block sort` faster than `6-pass global sort`?

The microbenchmark must measure:
1. **Method 1 (baseline)**: DeviceRadixSort on 8M 46-bit keys (6 passes)
2. **Method 2 (C17-1)**: DeviceRadixSort on 8M 14-bit keys (2 passes) + BlockRadixSort per tile on 32-bit keys (4 passes)

### Overflow handling:
- Tiles with >4096 intersections (0.23% for bicycle tile16): fall back to global sort for those tiles
- Implementation: after the 2-pass global sort, check per-tile count. If >4096, mark for global sort fallback.

---

## 8. Source File Summary

| File | Key Functions | Lines | Role |
|------|--------------|-------|------|
| `IntersectTile.cu` | `intersect_tile_kernel` | 24-114 | Pass 1+2: count tiles, write isect_ids |
| | `intersect_offset_kernel` | 209-257 | Compute tile_offsets from sorted ids |
| | `radix_sort_double_buffer` | 296-339 | CUB DeviceRadixSort (46-bit, 6 passes) |
| | `segmented_radix_sort_double_buffer` | 343-394 | CUB DeviceSegmentedRadixSort (per-image) |
| `Intersect.cpp` | `intersect_tile` | 15-149 | Host orchestration: pass1→cumsum→pass2→sort |
| | `intersect_offset` | 151-168 | Host wrapper for offset kernel |
| `Intersect.h` | declarations | 1-61 | Function signatures |
| `RasterizeToPixels3DGSFwd.cu` | `rasterize_to_pixels_3dgs_fwd_kernel` | 18-188 | Forward: reads tile_offsets + flatten_ids |
| `RasterizeToPixels3DGSBwd.cu` | `rasterize_to_pixels_3dgs_bwd_kernel` | 16-278 | Backward: reads tile_offsets + flatten_ids + last_ids |
| `Common.h` | `CUB_WRAPPER` macro | 23-30 | CUB temporary storage management |
| | `ALPHA_THRESHOLD` | 54 | 1/255 — minimum alpha for contribution |

---

## 9. Critical Interface Constraints

1. **`flatten_ids` must be depth-sorted within each tile**: The forward rasterizer processes front-to-back and uses early termination (T < 1e-4). Unsorted depths would produce incorrect alpha compositing.

2. **`last_ids` must be consistent**: The backward pass uses `last_ids[pix_id]` to know where to stop processing. If the sorted order changes between forward and backward, gradients will be wrong.

3. **`tile_offsets[tile_id+1]` must give the end of the range**: The rasterizer uses this as `range_end`. For the last tile of the last image, it uses `n_isects`.

4. **Tie-breaking for equal depths**: The current global radix sort produces a deterministic order for equal keys. CUB `BlockRadixSort` also produces deterministic order, but it may differ from `DeviceRadixSort` for equal keys. This is acceptable as long as forward and backward use the same sort (they share the same `flatten_ids`).

---

## 10. Phase A2 Microbenchmark Design

### Test: For a single tile with N intersections, compare:

**Method 1 (baseline share)**: Proportional cost of DeviceRadixSort on full array
- Sort 8M 46-bit keys, divide time by 8160 tiles → per-tile cost

**Method 2 (C17-1)**: BlockRadixSort on N 32-bit keys in shared memory
- Each CTA loads N keys from global memory
- Sorts using `cub::BlockRadixSort<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>`
- Writes sorted keys + values back to global memory

### Test sizes: 256, 512, 1024, 2048, 4096, 8192, 12000

### Metrics:
- Latency (CUDA events, median of 100 runs)
- Shared memory usage (from `cudaFuncGetAttributes`)
- Register usage (from `cudaFuncGetAttributes`)
- Occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`)
- Global memory read/write bytes

### Decision gate:
Continue to Phase A3 (integration) only if BlockRadixSort is **≥1.5x faster** than the proportional DeviceRadixSort cost for realistic tile sizes (1024-4096 intersections).
