# Track B — C17-1: Per-Tile Local Sort Pipeline Analysis

## Replacing Global CUB Radix Sort with Tile-Local Bounded Queue

---

## 1. Pipeline Overview

The gsplat sorting pipeline has four stages executed sequentially for every forward pass:

```
Stage 1 (Pass 1):  intersect_tile_kernel     — count tiles/gaussian       → tiles_per_gauss
Stage 2 (Host):     cumsum(tiles_per_gauss)   — compute output offsets     → cum_tiles_per_gauss, n_isects
Stage 3 (Pass 2):  intersect_tile_kernel     — write isect_ids + flatten   → isect_ids[n_isects], flatten_ids[n_isects]
Stage 4:            CUB radix sort            — sort by (depth, tile, img) → isect_ids_sorted, flatten_ids_sorted
Stage 5:            intersect_offset_kernel   — build tile offsets table    → tile_offsets[I×H_tile×W_tile]
```

C17-1 replaces **Stage 4 + Stage 5** with a single fused per-tile local sort. Stage 3 (Pass 2) is extended to write directly into per-tile bounded queues, each sorted locally by `(depth, gaussian_idx)`. The output format — `tile_offsets` and `flatten_ids` — remains identical.

---

## 2. Global Sort Input Structure

### 2.1 Tensors

| Tensor | Type | Shape | Semantic |
|--------|------|-------|----------|
| `isect_ids` | `int64_t` | `[n_isects]` | 64-bit sort key |
| `flatten_ids` | `int32_t` | `[n_isects]` | Gaussian index value |
| `tile_offsets` | `int32_t` | `[I, tile_height, tile_width]` | Per-tile start index |

All tensors live on GPU. `n_isects` is the **total number of tile–Gaussian intersections** across all images — typically **9–15 million** for a 1080p room scene with 1M Gaussians and tile16, or **2–3 million** with tile32.

### 2.2 64-bit Sort Key Layout

The `isect_ids` key is packed as follows (`IntersectTile.cu`, Pass 2 kernel, lines ~84–88):

```cpp
// iid_enc = iid << (32 + tile_n_bits)        // image_id at top
// tile_id << 32                                // tile_id in upper 32 bits
// depth_id_enc                                 // depth as raw int32 (zero-extended)
isect_ids[idx] = iid_enc | (tile_id << 32) | depth_id_enc;
```

Bit layout (MSB to LSB, 64-bit total):

```
Bit 63                   32+tile_n_bits          32                  0
 ┌──────────────────────┬──────────────────────┬──────────────────────┐
 │     image_id          │     tile_id          │     depth (int32)   │
 │  (image_n_bits bits)  │  (tile_n_bits bits)  │   (32 bits)         │
 └──────────────────────┴──────────────────────┴──────────────────────┘
```

| Field | Bits | Derivation |
|-------|------|------------|
| `depth` | `[0:32]` | `*(int32_t*)&depths[idx]` — raw IEEE 754 float bits, zero-extended to 64-bit |
| `tile_id` | `[32:32+tile_n_bits]` | `tile_y * tile_width + tile_x`, `tile_n_bits = floor(log2(n_tiles)) + 1` |
| `image_id` | `[32+tile_n_bits:63]` | Camera index, `image_n_bits = floor(log2(I)) + 1` |

**Constraints**: `image_n_bits + tile_n_bits ≤ 32` (verified by `assert` in `launch_intersect_tile_kernel`).

**Depth encoding**: The float is **reinterpreted as int32** (not cast): `int32_t depth_i32 = *(int32_t*)&(depths[idx])`. This preserves IEEE 754 ordering for **positive** floats — larger positive float values produce larger int32 bit patterns. Negative depths (theoretically possible near surfaces) would invert ordering; the code tolerates this by computing the `int64_t depth_id_enc = static_cast<uint32_t>(depth_i32)`, which zero-extends to 64 bits.

**`flatten_ids` value**: This is `int32_t` containing the **linear index** into the original Gaussian arrays (`means2d`, `radii`, `depths`). In non-packed mode: `flatten_ids[idx] = idx_in_pass2`. In packed mode: `flatten_ids[idx] = gaussian_ids[idx]` (the original packed index).

### 2.3 Why This Key Layout?

The key is designed so that CUB radix sort sorts by **depth first, then tile_id, then image_id**. Since CUB sorts from LSB to MSB, the final ordering is:

1. **Depth** (lowest 32 bits — sorted first by CUB)
2. **Tile ID** (next tile_n_bits — sorted second)
3. **Image ID** (top bits — sorted last)

This produces the correct rendering order: within each image+tile, Gaussians are sorted by depth (front-to-back for alpha blending), and tile boundaries are respected for the `tile_offsets` construction. The sort is:

```cpp
// Non-segmented: sorts all 64 bits (depth + tile + image)
cub::DeviceRadixSort::SortPairs(d_keys, d_values, n_isects,
    0, 32 + tile_n_bits + image_n_bits, stream);

// Segmented: sorts only depth + tile within each image segment
cub::DeviceSegmentedRadixSort::SortPairs(d_keys, d_values, n_isects,
    n_segments, offsets, offsets + 1,
    0, 32 + tile_n_bits, stream);
```

---

## 3. Downstream Dependencies

### 3.1 `tile_offsets` — The Per-Tile Index Table

**Producer**: `intersect_offset_kernel` (IntersectTile.cu, lines 208–257)

```cpp
__global__ void intersect_offset_kernel(
    const int64_t *__restrict__ isect_ids,   // [n_isects] — SORTED
    int32_t *__restrict__ offsets              // [I, n_tiles]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    int64_t isect_id_curr = isect_ids[idx] >> 32;  // extract (image_id | tile_id)
    int64_t iid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = iid_curr * n_tiles + tid_curr;

    // Thread 0: write offsets[0..id_curr] = 0
    // Thread n_isects-1: write offsets[id_curr+1..end] = n_isects
    // Other threads: detect tile changes and write boundary offsets
    if (idx > 0) {
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32;
        if (isect_id_prev == isect_id_curr) return;  // same tile → skip
        // Fill gap from prev tile's end to curr tile's start
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = idx;
    }
}
```

The kernel exploits the radix-sorted order: each thread `idx` inspects whether its intersection belongs to a different tile than the previous intersection. When a tile transition is detected, it writes the start index of the new tile. After the kernel completes:

```
offsets[tile_id]   = first intersection index belonging to tile_id
offsets[tile_id+1] = first intersection index of tile_id+1 (i.e., exclusive end)
```

**Range for a tile**: `tile_range = [tile_offsets[tile_id], tile_offsets[tile_id + 1])` — except for the last tile globally, where `tile_offsets[last] = n_isects`.

Empty tiles have `offsets[t] = offsets[t+1]` (both equal to the next non-empty tile's start).

### 3.2 `flatten_ids` — The Gaussian Index Indirection

**Producer**: Pass 2 of `intersect_tile_kernel` (written concurrently with `isect_ids`)

**Consumers** (forward and backward rasterization):

In `RasterizeToPixels3DGSFwd.cu`:

```cpp
// For each tile, the forward kernel:
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = /* tile_offsets[tile_id + 1] or n_isects for last tile */;

// All threads in tile collaboratively load gaussians in batches
for (uint32_t b = 0; b < num_batches; ++b) {
    int32_t batch_idx = thread_rank;  // 1:1 thread-to-intersection loading
    int32_t gaussian_idx = flatten_ids[range_start + b * block_size + batch_idx];
    // Use gaussian_idx to look up means2d, conics, colors, opacities
}
```

Same pattern in `RasterizeToPixels3DGSBwd.cu` and `RasterizeToIndices3DGS.cu`.

**Critical**: The backward kernel **iterates intersections in the same order** as the forward kernel, writing gradients into `v_means2d[flatten_ids[idx]]`, `v_conics[flatten_ids[idx]]`, etc. This means the `flatten_ids` ordering is the **sole ordering contract** between forward and backward passes.

### 3.3 `RasterizeToIndices3DGS.cu` — Helper for Adaptive Control

This kernel additionally consumes `tile_offsets` and `flatten_ids` to compute per-pixel transmittance-driven Gaussian selection. It also reads in the same batched pattern.

### 3.4 Dependency Summary

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────────┐
│ isect_ids   │ ──→ │ CUB radix sort│ ──→ │ intersect_offset     │ ──→ tile_offsets[I×H_tile×W_tile]
│ (unsorted)  │     │               │     │ (extract image+tile) │
└─────────────┘     └───────────────┘     └──────────────────────┘
       │                     │
       │                     └──→ isect_ids_sorted (not directly consumed)
       │
       └──→ flatten_ids ──────→ flatten_ids_sorted ──→ Fwd rasterization
                                                            │
                                                            └──→ Bwd rasterization
                                                                  (same order)
```

**Key invariant**: The backward kernel reads `flatten_ids` in the **exact same order** as the forward kernel. The sort must produce a deterministic, repeatable ordering that both passes agree on.

---

## 4. Replacement Points for C17-1 Per-Tile Sort

### 4.1 Where Exactly to Intervene

There are three natural interception points:

#### Option A: Replace Stage 3–5 (Recommended — C17-1)

Modify `IntersectTile.cu` and `Intersect.cpp`:

- **Pass 2** (`intersect_tile_kernel`): Instead of writing `isect_ids` and `flatten_ids` as unsorted parallel arrays, write directly into **per-tile queues** allocated in a single pre-allocated buffer.
- **Remove** the CUB radix sort call (`radix_sort_double_buffer` / `segmented_radix_sort_double_buffer`).
- **Remove** `intersect_offset_kernel` — queue headers provide tile offsets directly.
- Output `tile_offsets[I×H_tile×W_tile]` + `flatten_ids[n_isects]`.

The `Intersect.cpp` host code changes from:

```cpp
// Stage 2: cumsum tiles
cum_tiles_per_gauss = at::cumsum(tiles_per_gauss, 0);
n_isects = cum_tiles_per_gauss[-1].item<int64_t>();

// Stage 3: write isect_ids and flatten_ids
launch_intersect_tile_kernel(..., cum_tiles_per_gauss, nullopt, isect_ids, flatten_ids);

// Stage 4: CUB radix sort
radix_sort_double_buffer(n_isects, image_n_bits, tile_n_bits,
    isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted);

// Stage 5: build tile_offsets
offsets = intersect_offset(isect_ids_sorted, I, tile_width, tile_height);
```

To:

```cpp
// Stage 3+4 fused: write directly into per-tile queues, locally sorted
auto [tile_offsets, flatten_ids] = intersect_tile_local_queues(
    means2d, radii, depths, I, tile_size, tile_width, tile_height, cum_tiles_per_gauss
);
```

#### Option B: Replace only Stage 4 (Partial sort optimization)

Keep the two-pass structure and tile-offset computation. Replace CUB radix sort with a per-tile radix sort (e.g., CUB `DeviceSegmentedRadixSort` with `offsets`). This is simpler but keeps the two-pass structure and the separate offset kernel. Less potential savings.

#### Option C: Replace Stage 4–5 fused (not recommended)

Keep Pass 2 writing unsorted `isect_ids`. Replace CUB with a custom segmented sort that produces both sorted `flatten_ids` and `tile_offsets`. This avoids modifying the intersection kernel but still eliminates CUB.

**C17-1 implements Option A** for maximum speedup.

### 4.2 What Must NOT Change

The following interface contracts are **inviolable**:

1. **`flatten_ids` ordering**: Must produce the same depth-ordered sequence of Gaussian indices **within each tile** as the global CUB sort would. The Python correctness proof (Phase 17B) confirmed 100% order match across 34.8M intersections.

2. **`tile_offsets[tile_id]` semantics**: Must equal the count of intersections belonging to all previous tiles + previous images. `tile_offsets[tile_id + 1] - tile_offsets[tile_id]` = number of Gaussians in tile `tile_id`.

3. **Tiling geometry**: `tile_size` (16 or 32), `tile_width = ceil(image_width / tile_size)`, `tile_height = ceil(image_height / tile_size)` must remain unchanged.

4. **No autograd dependency**: The sort runs under `@torch.no_grad()` — this holds for both CUB and C17-1.

---

## 5. C17-1 CUDA Implementation Design

### 5.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Per-Tile Bounded Queue                      │
│                                                                │
│  Queue buffer: [ ██░░░░░░ │████████│██░░░░░░░│███████│... ]    │
│                 tile 0    tile 1   tile 2    tile 3            │
│                                                                │
│  Queue headers: [tile_0_start, tile_1_start, ...]              │
│                 └── this IS tile_offsets                       │
│                                                                │
│  Each tile: insertion-sorted by (depth, gaussian_idx)          │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Pass 2 Modification

The current Pass 2 kernel writes each intersection to the next available slot in `isect_ids` and `flatten_ids`, using `cum_tiles_per_gauss` to compute the write index:

```cpp
// Current code (non-queue):
int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
// ... loop over tiles ...
isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
flatten_ids[cur_idx] = packed ? gaussian_ids[idx] : idx;
cur_idx++;
```

**C17-1 modification**: Each thread (processing one Gaussian) instead pushes intersections into per-tile queues:

```cpp
// Pseudo-code for C17-1 Pass 2:
extern __shared__ int queue_tail[];  // per-tile tail pointers in shared memory
__shared__ bool initialized;

if (thread_rank == 0) {
    for (int t = 0; t < n_tiles_in_block; t++)
        queue_tail[t] = 0;  // tiles handled by this block
}
__syncthreads();

for each tile (i, j) that the Gaussian intersects {
    int tile_id = i * tile_width + j;
    int queue_idx = atomicAdd(&queue_tail[tile_id], 1);
    if (queue_idx < MAX_QUEUE_SIZE) {
        // Insertion-sort by (depth, gaussian_idx)
        insert_sorted(queue_buffer[tile_id], queue_idx, 
                      depth_value, gaussian_idx);
    } else {
        // Overflow handling
    }
}
```

### 5.3 Tile Queues — Allocation Strategy

**Problem**: We don't know `n_isects` in advance without the two-pass structure. C17-1 avoids this by:

1. **Pre-allocate maximum** using `cum_tiles_per_gauss` (already computed in Pass 1) as the per-Gaussian tile count. The total `n_isects` is `cum_tiles_per_gauss[-1]`, which we already compute atomically in the current two-pass design.

2. **Per-tile queue boundaries** in global memory: Since we know each Gaussian's tile count from Pass 1, we can pre-allocate a fixed-size buffer and compute queue boundaries.

**Queue buffer layout**:

```cpp
// Host code (Intersect.cpp):
auto queue_buffer = at::empty({n_isects}, opt.dtype(at::kInt));  // flatten_ids
auto tile_offsets = at::zeros({I * n_tiles + 1}, opt.dtype(at::kInt));

// Compute queue start positions — identical to current tile_offsets computation
// but using cum_tiles_per_gauss instead of sorted isect_ids
// queue_starts[t] = number of intersections in all tiles < t
// This is exact because it's derived from tile_count (same as current n_isects)
```

Actually, the cleanest approach is:

1. Keep Pass 1 (count tiles per Gaussian) — it's cheap (bitwise ops on 2D bounds).
2. Keep `cumsum(tiles_per_gauss)` to know total `n_isects`.
3. **Replace** Pass 2 + CUB sort + offset kernel with a **single kernel** that writes directly to per-tile queues.

### 5.4 Queue Structure

Each tile's queue header is stored in `tile_offsets[tile_id]` — the start pointer. The queue itself is a segment of `flatten_ids`:

```cpp
struct TileQueue {
    int32_t* base;        // &flatten_ids[queue_start]
    int32_t capacity;     // queue_end - queue_start
    uint32_t tail;        // atomic counter (next write position)
};

// Queue entries: stored as packed pair {depth_int32, gaussian_idx}
// Using int64_t: lower 32 bits = depth_int32, upper 32 bits = gaussian_idx
```

### 5.5 Insertion Sort Strategy

Within each tile, Gaussians must be ordered by `(depth, gaussian_idx)` for correct alpha blending. Since Gaussians arrive in arbitrary order (by their 2D position, not by depth), we need sorting.

**Option A: Atomic insertion sort** (for small queues, ≤ 256 intersections/tile)

Each GPU thread inserting into a tile performs a lock-free insertion:

```cpp
__device__ void insert_sorted(
    int64_t* queue,        // packed {depth_i32, gaussian_idx}
    uint32_t tail,         // current count in this tile
    int32_t depth_i32,     // new depth
    int32_t gaussian_idx    // new gaussian index
) {
    int64_t new_entry = ((int64_t)gaussian_idx << 32) | (uint32_t)depth_i32;
    
    // Find insertion point — scan from tail-1 backward
    for (int i = tail - 1; i >= 0; i--) {
        int64_t existing = queue[i];
        int32_t existing_depth = (int32_t)(existing & 0xFFFFFFFF);
        int32_t existing_gidx = (int32_t)(existing >> 32);
        
        if ((depth_i32 > existing_depth) || 
            (depth_i32 == existing_depth && gaussian_idx > existing_gidx)) {
            // Shift right: atomicCAS-based shift
            atomicExch(&queue[i + 1], existing);
        } else {
            queue[i + 1] = new_entry;  // Write in place
            return;
        }
    }
    queue[0] = new_entry;  // Smallest depth → front
}
```

This is O(n) per insertion. For mean queue depth of 75K (measured C17 benchmark), this is **prohibitively expensive** (O(n²) total = 2.8B comparisons per tile).

**Option B: Warp-level bitonic sort** (for moderate queues, ≤ 1024)

Batch insertions 32 at a time (one warp), perform warp-level bitonic merge:

```
1. Accumulate 32 entries in registers (no global memory)
2. Sort the 32 entries in-register using warp intrinsics
3. Merge-sort into the existing global queue (O(log(capacity)) binary search)
```

Still O(n²) for large queues if many batches.

**Option C: Per-tile radix sort** (Recommended — for all queue sizes)

Since C17 benchmarks showed **up to 102K intersections per tile** (mean 75K), we need a true O(n) sort:

1. **Stage A**: Allocate per-tile queues via atomic add.
2. **Stage B** (separate kernel): For each tile, launch a small CUB `DeviceRadixSort::SortPairs` on `(depth_i32, gaussian_idx)` pairs within the tile's queue segment.
   
   But launching CUB per-tile is wasteful. Instead:

3. **Stage B (fused)**: Use a **single segmented radix sort** kernel with per-tile boundaries (which we compute from queue tail pointers). CUB `DeviceSegmentedRadixSort` already supports this:

```cpp
// After Pass 2 with atomic queues:
// queue_tails[tile_id] = count of intersections in tile t

// Compute offsets for segmented sort:
// seg_offsets[t] = cumulative sum of queue_tails[0..t-1]

// One call to segmented radix sort:
cub::DeviceSegmentedRadixSort::SortPairs(
    d_keys, d_values, n_isects,
    n_segments, seg_offsets, seg_offsets + 1,
    0, 32,  // Sort only the depth field (lower 32 bits)
    stream
);
```

This is **still CUB-based**, so it doesn't eliminate CUB. However, it operates on **smaller segments** (tile-scoped, not image-scoped). The pure C17-1 approach goes further.

**Option D: Shared-memory bitonic sort per tile-block** (Ideal for C17-1)

This is the **true C17-1 approach** that eliminates CUB entirely:

1. **Grid configuration**: Each **CTA (thread block) processes one tile**. The grid is `[n_tiles_per_image × I]` CTAs.

2. **Shared memory queue**: Each CTA allocates a shared memory buffer:

```cpp
__shared__ int64_t queue[MAX_TILE_INTERSECTIONS];  // packed {depth_i32, gauss_idx}
__shared__ uint32_t tail;
```

3. **Global-to-shared loading**: First, all intersections for this tile are loaded from a temporary buffer (written by Pass 2) into shared memory. Since `tiles_per_gauss` is known, and `cum_tiles_per_gauss` gives the start offset, we can iterate all Gaussians that intersect this tile and load them.

But this requires **iterating all Gaussians per tile** — a global scatter-gather that defeats parallelism.

**Option E (C17-1 Recommended): Per-tile queue blocks with shared-memory sort**

The optimal approach:

**Pass 2+ sort fused kernel**: Each CTA is assigned to one tile. The CTA:

1. Receives intersections from **all Gaussians that overlap this tile** via global memory scatter
2. Processes intersections in batches, loading into shared memory
3. Sorts the shared memory buffer using a **block-wide radix sort** (not CUB)
4. Writes the sorted queue to global memory as `flatten_ids[tile_offsets[tile_id] + i]`

The key insight from the C17 benchmark: **tile-level parallelism** — tiles are independent (no cross-tile ordering needed), and each tile has enough work (mean 75K intersections) to keep a CTA busy.

### 5.6 Queue Overflow Strategy

Each tile's queue must have bounds. Since `tiles_per_gauss` is computed in Pass 1 and `cum_tiles_per_gauss` gives exact counts, **overflow is impossible** with correct pre-allocation:

| Check | Location | Action |
|-------|----------|--------|
| `tail < capacity` | Per-Gaussian push | Guaranteed by Pass 1 count |
| `shared_mem_used < shmem_limit` | At sort time | Tile capacity determined by SMEM (48KB→6000 entries with int64_t packing, use streaming to global) |

For tiles with >48KB worth of intersections (6000 int64_t entries), use a **two-level sort**: sort in shared memory in batches, then merge-sort to global memory.

Per the C17 benchmark:
- **Mean per-tile**: 75,026 intersections → 600 KB per tile → must use global memory, not SMEM
- **Max per-tile**: 102,394 intersections → 820 KB

**Conclusion**: Shared memory alone is insufficient for full per-tile buffers. Each tile's data must reside in global memory. The sort must be a per-tile **global memory radix sort**.

### 5.7 Revised C17-1 Plan

Given the memory constraints revealed by the C17 benchmark, the practical C17-1 implementation is:

```
┌──────────────────────────────────────────────────────────────────┐
│ C17-1: Fused Tile-Local Sort Pipeline                            │
│                                                                   │
│ Step 1 (same as before):                                          │
│   interset_tile_kernel (Pass 1) → tiles_per_gauss                 │
│   cumsum → cum_tiles_per_gauss, n_isects                          │
│                                                                   │
│ Step 2 (NEW — replaces Pass 2 + CUB + offset):                   │
│   intersect_tile_queues_kernel (Pass 2'):                         │
│   • Each Gaussian writes {depth_i32, gaussian_idx} pairs          │
│     into per-tile queue segments (allocated from n_isects)        │
│   • Atomic per-tile tail pointer (global memory)                  │
│   • No inter-tile ordering needed                                 │
│                                                                   │
│ Step 3 (NEW — tile-local sort):                                   │
│   tile_local_radix_sort_kernel:                                   │
│   • Each CTA processes one tile                                   │
│   • Uses its own segment of isect_ids [tail_start, tail_end)      │
│   • Block-wide radix sort (NOT CUB) on 64-bit keys                │
│   • Writes sorted flatten_ids                                     │
│                                                                   │
│ Step 4 (NEW — tile_offsets from queue headers):                  │
│   • Read from per-tile tail pointers → tile_offsets               │
│   • No separate kernel needed (can be produced by Step 3)         │
└──────────────────────────────────────────────────────────────────┘
```

### 5.8 Fused Kernel Pseudocode

```cpp
// ===============================================================
// Pass 2': Write intersections into per-tile queues
// ===============================================================
__global__ void intersect_tile_queues_kernel(
    const scalar_t* means2d,     // [N, 2]
    const int32_t* radii,        // [N, 2]  
    const scalar_t* depths,      // [N]
    const int64_t* cum_tiles,    // [N] — cumulative tile count
    const uint32_t tile_size, tile_width, tile_height,
    int64_t* queue_buffers,      // [n_tiles][capacity] — flattened per-tile queues
    int32_t* tile_queues_flat,   // [n_isects] — flattened depth+idx for sort
    uint32_t* tile_tails         // [n_tiles] — atomic tail pointers, pre-zeroed
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float rx = radii[idx*2], ry = radii[idx*2+1];
    if (rx <= 0 || ry <= 0) return;

    // Compute tile bounds (same as current kernel)
    vec2 mean2d = ...;
    uint2 tile_min, tile_max;  // tile intersection range

    int64_t queue_start = (idx == 0) ? 0 : cum_tiles[idx - 1];
    int32_t depth_i32 = *(int32_t*)&depths[idx];
    int32_t gauss_idx = idx;  // or packed gaussian_ids[idx]

    for (int ty = tile_min.y; ty < tile_max.y; ++ty) {
        for (int tx = tile_min.x; tx < tile_max.x; ++tx) {
            int tile_id = ty * tile_width + tx;
            
            // Atomic-claim a slot in this tile's queue
            uint32_t slot = atomicAdd(&tile_tails[tile_id], 1);
            
            // Pack {depth_i32, gaussian_idx} for sort
            tile_queues_flat[queue_start + slot] = depth_i32;
            // Also store gaussian_idx separately (or pack both)
        }
    }
}

// ===============================================================
// Step 3: Per-tile block-wide radix sort
// ===============================================================
__global__ void tile_local_sort_kernel(
    int32_t* depth_keys,      // [n_isects] — depth per intersection
    int32_t* flatten_ids,     // [n_isects] — output gaussian indices
    uint32_t* tile_offsets,   // [n_tiles + 1] — produced from tile_tails
    const uint32_t n_tiles
) {
    int tile_id = blockIdx.x;
    int range_start = tile_offsets[tile_id];
    int range_end = tile_offsets[tile_id + 1];
    int n_local = range_end - range_start;
    
    if (n_local <= 0) return;
    
    // Use shared memory for small tiles, global memory for large tiles
    if (n_local <= 1024) {
        __shared__ int32_t shm_keys[1024];
        __shared__ int32_t shm_vals[1024];
        // Load → shared memory bitonic sort → write back
    } else {
        // Global memory: Tile-wide radix sort using warp intrinsics
        // Each warp sorts a chunk, then warp-merge
        // Or: use CUB's BlockRadixSort in a cooperative fashion
    }
}
```

### 5.9 Backward Pass Considerations

The backward kernel (`RasterizeToPixels3DGSBwd.cu`) reads `tile_offsets` and `flatten_ids` in **exactly the same pattern** as the forward kernel. This means:

1. **`flatten_ids` must be deterministic** — given the same input Gaussians and camera parameters, the backward pass must see the same ordering as the forward pass.

2. **No sorting in backward** — the backward kernel does not sort. It only reads the pre-sorted `flatten_ids` and iterates intersections in that order.

3. **Memory management**: The backward kernel receives the same `tile_offsets` and `flatten_ids` tensors. These must be **persisted** from the forward pass (they are, since the forward pass returns `tile_offsets` and `flatten_ids_sorted` which PyTorch keeps alive through the autograd graph).

4. **C17-1 impact on backward**: The backward kernel needs **zero changes**. It reads `tile_offsets` and `flatten_ids` identically regardless of how they were produced. The ordering invariant (depth-sorted within each tile) is the same.

**C17-1 output contract for backward pass:**

```cpp
// Same as CUB-sorted output:
// For each intersection idx in [tile_offsets[t] .. tile_offsets[t+1]):
//    flatten_ids[idx] = gaussian index
//    depth ordering INCREASING within each tile

// Backward kernel invocation:
rasterize_to_pixels_3dgs_bwd_kernel<<<...>>>(
    ..., tile_offsets, flatten_ids, ...
);
```

No changes to the backward kernel signatures or semantics.

### 5.10 Compatibility with `RasterizeToIndices3DGS`

The `rasterize_to_indices_3dgs` kernel (adaptive control helper) also reads `tile_offsets` and `flatten_ids`. It is unchanged by C17-1.

### 5.11 Bit-Level Sort Key for Tile-Local Sort

Since each tile's sort is **independent**, the sort key simplifies:

| Field | Bits | Purpose |
|-------|------|---------|
| `depth_i32` | `[0:31]` | IEEE 754 float bits, reinterpeted as int32 |
| `gaussian_idx` | `[32:63]` | Linear index into Gaussian arrays |

**No tile_id or image_id needed** — those are implicit from the tile range. Each tile's sort only needs `(depth, gaussian_idx)`.

For the CUB-segmented approach:
```cpp
// Sort key = depth_i32 (lower 32 bits of isect_ids)
// Segment boundaries = tile_offsets[0..n_tiles]
CUB_WRAPPER(
    cub::DeviceSegmentedRadixSort::SortPairs,
    d_depth_keys, d_flatten_values, n_isects,
    n_tiles, tile_offsets, tile_offsets + 1,
    0, 32,  // Only sort the depth field
    stream
);
```

This is different from the current segmented sort which sorts on `(depth, tile_id)` within each image segment.

---

## 6. Performance Projection

Based on the C17 benchmark measurements:

| Metric | CUB Global Sort | C17-1 Per-Tile Sort | Improvement |
|--------|-----------------|---------------------|-------------|
| Sort time | 1.43 ms | 0.72–1.14 ms* | 50–80% |
| Sort + offset | ~1.5 ms | ~0.9 ms | 40% |
| End-to-end fwd | 4.44 ms | 3.33–3.72 ms | +16–25% |
| CUB temp storage | ~250 MB | 0 MB | No temporary alloc |

*Range depends on per-tile sort efficiency. Tile count: only 128/8160 non-empty tiles (C17 benchmark — but this is an artifact of all Gaussians covering every tile; typical workloads have more active tiles).

**Why per-tile sort is faster**:

1. **No cross-tile global barrier**: CUB sorts all 9M+ intersections globally. C17-1 sorts 128 independent segments of mean 75K each.

2. **Lower key bit width**: CUB sorts 54+ bits (image+ tile + depth). Per-tile sorts only 32 bits (depth).

3. **No `tile_offsets` kernel**: Queue headers directly provide offsets.

4. **CUB temporary allocation overhead**: `cub::DeviceRadixSort` allocates O(n) temporary storage. C17-1 uses pre-allocated buffers.

### 6.1 Scalability Considerations

**SM occupancy**: A100 has 108 SMs. At tile16 resolution (160×90 = 14,400 tiles per image), only 128 tiles are active per image. With I=1 camera, 128 tiles → ~1.2 tiles/SM. With I=311 cameras, 311 × 128 = 39,808 active tiles → ~369 tiles/SM, which is excellent utilization.

**Per-tile variance**: Tile 0 (top-left) has the most intersections (camera center). Small border tiles near image edges have few. Load balancing requires care:
- Use dynamic block scheduling (don't launch one block per tile with global barrier)
- Use a work queue: warp processes the next unassigned tile
- Tile groups: merge small adjacent tiles into a single CTA

---

## 7. Implementation Plan

### 7.1 Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `IntersectTile.cu` | **EXTEND** | Add `intersect_tile_queues_kernel` and `tile_local_sort_kernel` |
| `Intersect.cpp` | **MODIFY** | Replace sort + offset pipeline with C17-1 dispatch |
| `Intersect.h` | **MODIFY** | Add `launch_intersect_tile_queues_kernel` declaration |
| `Rasterization.h` | No change | Interface unchanged (`tile_offsets` + `flatten_ids`) |
| `RasterizeToPixels3DGSFwd.cu` | No change | Reads same `tile_offsets` + `flatten_ids` |
| `RasterizeToPixels3DGSBwd.cu` | No change | Reads same `tile_offsets` + `flatten_ids` |
| `RasterizeToIndices3DGS.cu` | No change | Reads same `tile_offsets` + `flatten_ids` |

### 7.2 Python API Changes

The Python-level `isect_tiles(sort=True)` already returns `(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted)` and `intersect_offset(isect_ids, ...)` returns `tile_offsets`. With C17-1:

- `sort=True` → use C17-1 per-tile local sort (fused with Pass 2)
- `sort=False` → use existing two-pass (no sort, no offset kernel needed)
- Or add a new parameter: `sort_mode="c17_1"` or `sort_mode="cub"`

### 7.3 Integration Steps

1. **Add queue allocation** to `Intersect.cpp` (using pre-computed `n_isects`)
2. **Replace `radix_sort_double_buffer` + `intersect_offset_kernel`** with `tile_local_sort` + queue dispatch
3. **Conditional compilation** with CUB fallback (keep CUB path as `sort_mode="cub"`)
4. **Test** with existing C17-1 Python correctness test (100% order match across 34.8M intersections)

---

## 8. Source Code Reference

### 8.1 IntersectTile.cu Key Sections

**Pass 1 kernel** (lines ~70–90): Computes `tiles_per_gauss[idx] = (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)`. No depth or sort key construction.

**Pass 2 kernel** (lines ~92-120): Constructs the 64-bit sort key and writes into the flat arrays:

```cpp
// depth reinterpet as int32
int32_t depth_i32 = *(int32_t*)&(depths[idx]);

// image_id at top of sort key
const int64_t iid_enc = iid << (32 + tile_n_bits);

// Packed key: depth (LSB) | tile_id | image_id (MSB)
isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;

// Gaussian index (value to sort alongside key)
flatten_ids[cur_idx] = packed ? gaussian_ids[idx] : idx;
```

**radix_sort_double_buffer** (lines ~340–394): CUB double-buffered sort:

```cpp
CUB_WRAPPER(
    cub::DeviceRadixSort::SortPairs,
    d_keys, d_values, n_isects,
    0, 32 + tile_n_bits + image_n_bits,  // sort depth + tile + image
    stream
);
```

**intersect_offset_kernel** (lines 208–257): Reads sorted `isect_ids`, extracts `(image_id, tile_id)` from upper 32 bits, writes `tile_offsets[]`.

### 8.2 Intersect.cpp Key Sections

**`intersect_tile` function** (two-pass + sort):

```cpp
// Pass 1
launch_intersect_tile_kernel(..., nullopt, tiles_per_gauss, nullopt, nullopt);
cum_tiles_per_gauss = cumsum(tiles_per_gauss);
n_isects = cum_tiles_per_gauss[-1];

// Pass 2
launch_intersect_tile_kernel(..., cum_tiles_per_gauss, nullopt, isect_ids, flatten_ids);

// CUB sort
if (segmented) {
    segmented_radix_sort_double_buffer(n_isects, I, image_n_bits, tile_n_bits,
        offsets, isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted);
} else {
    radix_sort_double_buffer(n_isects, image_n_bits, tile_n_bits,
        isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted);
}

// tile_offsets from sorted ids
offsets = intersect_offset(isect_ids_sorted, I, tile_width, tile_height);
```

### 8.3 Downstream Consumption (RasterizeToPixels3DGSFwd.cu)

```cpp
// Per-tile range extraction:
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = (image_id == I-1 && tile_id == last_tile)
    ? n_isects : tile_offsets[tile_id + 1];

// Batched intersection loading:
for (uint32_t b = 0; b < num_batches; ++b) {
    int32_t gaussian_idx = id_batch[thread_rank];
    // load means2d, conics, colors, opacities via gaussian_idx
    // ... alpha blending ...
}
```

---

## 9. Summary

| Aspect | Global CUB Sort | C17-1 Per-Tile Sort |
|--------|----------------|---------------------|
| Sort scope | Global across all tiles | Independent per tile |
| Key width | 54+ bits (depth + tile + image) | 32 bits (depth only) |
| Algorithm | CUB DeviceRadixSort (6+ passes) | Block-wide radix sort (2 passes) |
| Temp storage | O(n) temporary allocation | Pre-allocated, no temporary |
| `tile_offsets` | Separate kernel | Inferred from queue headers |
| Backward changes | None | None |
| Correctness proven | — | 100% order match, 34.8M intersections |
| Estimated speedup | baseline | +16–26% end-to-end |

The key insight: **global depth sorting creates unnecessary work**. Within each tile, Gaussians must be depth-sorted for correct alpha blending. But there is **no cross-tile ordering requirement**. Each tile's intersections are independent — a perfect candidate for tile-parallel local sort.

---

*Reference: C17-1 Phase 17B correctness proof, `reports/a100_validation/c17_queue_optimization.md`*
