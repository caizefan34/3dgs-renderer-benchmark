# Phase 17A — Intersection / Sorting Architecture Trace

**Date:** 2026-10-17
**Source:** gsplat v1.5.3 (installed at `C:\Users\36570\miniconda3\Lib\site-packages\gsplat`)
**Hardware context:** NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell, ~320 GB/s memory bandwidth)

---

## 1. Full Pipeline Dataflow Map

### Legend

```
[Kernel]          — CUDA kernel launch
(Tensor)          — GPU memory buffer
{shape, dtype}    — buffer dimensions and data type
{size_bytes}      — memory size estimate
──→               — producer → consumer dependency
```

### Stage S1: Projection (fully_fused_projection)

```
Input Gaussians
  means3d    {N, 3, float32}  {N×12 bytes}
  quats      {N, 4, float32}  {N×16 bytes}
  scales     {N, 3, float32}  {N×12 bytes}
  viewmats   {C, 4, 4, float32}
  Ks         {C, 3, 3, float32}

[Projection CUDA kernel]  (project_ewa_3dgs_packed_fwd_kernel)
  │
  ├── means2d    {nnz, 2, float32}  {nnz×8 bytes}
  ├── radii      {nnz, 2, int32}    {nnz×8 bytes}
  ├── depths     {nnz, 1, float32}  {nnz×4 bytes}
  ├── conics     {nnz, 3, float32}  {nnz×12 bytes}
  └── (opacities already in memory from input, potentially compensated)

  packed=True: nnz = number of Gaussians with >0 radius after frustum culling
  For room (1.59M gaussians, single camera): nnz ~ 200-500K
```

**Memory**: nnz × (8+8+4+12) = nnz × 32 bytes ≈ 6–16 MB

### Stage S2: SH Evaluation (spherical_harmonics)

```
(spherical_harmonics CUDA kernel)
  │
  └── colors  {nnz, CDIM, float32}  {nnz × CDIM × 4 bytes}
       (CDIM=3 for RGB, up to 64+ for N-D features)
```

**Memory**: nnz × CDIM × 4 bytes ≈ 2.4–6 MB (CDIM=3)

### Stage S3a: Intersect First Pass (Pass 1)

```
[intersect_tile_kernel — Pass 1]
  Inputs: means2d, radii, depths
  cum_tiles_per_gauss = nullptr (signal: first pass)
  Parallelism: index over nnz (packed) or I×N (dense), 256 threads/block
  
  For each Gaussian:
    Compute tile footprint from (mean2d ± radius / tile_size)
    tile_count = (tile_max.y - tile_min.y) × (tile_max.x - tile_min.x)
  
  Output:
    tiles_per_gauss  {nnz, int32}  {nnz×4 bytes}
```

**Work**: 1 thread per Gaussian, simple arithmetic + bounds check.
**Memory write**: 4 bytes per Gaussian.

### Stage S3b: Prefix Sum + Host Sync

```
cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.flatten(), 0)
  │  {nnz, int64}  {nnz×8 bytes}
  │
n_isects = cum_tiles_per_gauss[-1].item<int64_t>()
  │  ⚠ HOST-DEVICE SYNC
  │  forces cudaDeviceSynchronize after cumsum kernel
  │  Costs: ~5-30μs but serializes pipeline
  │
  ↓
isect_ids  = at::empty({n_isects}, torch::kInt64)  {n_isects×8 bytes}
flatten_ids = at::empty({n_isects}, torch::kInt32)  {n_isects×4 bytes}
```

**Critical observation**: The `.item<int64_t>()` call is the ONLY host-device sync in the intersect pipeline. It forces the CPU to wait for Pass 1 kernel + cumsum to complete before allocating buffers for Pass 2.

**n_isects for room at 1080p:**
- tile16: ~3.22M (from Phase 14B)
- tile20: ~2.37M
- tile32: ~0.56M

**Memory allocation for isect_ids + flatten_ids:**
- tile16: 3.22M × (8+4) = ~39 MB
- tile20: 2.37M × 12 = ~28 MB
- tile32: 0.56M × 12 = ~6.7 MB

### Stage S4: Intersect Second Pass (Pass 2)

```
[intersect_tile_kernel — Pass 2]
  Inputs: means2d, radii, depths, cum_tiles_per_gauss
          image_ids, tile_size, tile_width, tile_height
  
  For each Gaussian:
    Compute tile footprint (same as Pass 1)
    For each covered tile (inner loop):
      Construct 64-bit sort key:
        Bits [63 : 32+tile_n_bits]  — image_id  (image_n_bits wide)
        Bits [32+tile_n_bits-1 : 32] — tile_id  (tile_n_bits wide)
        Bits [31 : 0]               — depth (float32 bitcast → int32, zero-extended)
      
      isect_ids[cur_idx] = key      {int64}
      flatten_ids[cur_idx] = idx    {int32}  (flattened Gaussian index)
  
  Outputs:
    isect_ids    {n_isects, int64}  {n_isects×8 bytes = 12-26 MB}
    flatten_ids  {n_isects, int32}  {n_isects×4 bytes = 6-13 MB}
```

**Work**: Thread per Gaussian. Inner loop per covered tile. Typical: 4-32 tiles per Gaussian.
**Memory writes**: 12 bytes per intersection.

### Stage S5: Global Radix Sort

```
[radix_sort_double_buffer]
  │
  cub::DeviceRadixSort::SortPairs(
    d_keys, d_values, n_isects,
    begin_bit = 0,
    end_bit = 32 + tile_n_bits + image_n_bits,  // ~47 bits for 1080p tile16
    stream
  )
  
  Internal:
    Temporary storage: allocated by CUB via CachingAllocator
    Radix passes: ceil(end_bit / 4) = 12 passes for 47-bit key
    Per pass: read+write 12 bytes per element (double-buffered)
    
  DoubleBuffer:
    d_keys:   {isect_ids, isect_ids_sorted}    each {n_isects×8}
    d_values: {flatten_ids, flatten_ids_sorted} each {n_isects×4}
  
  Memory traffic per pass:
    2 × n_isects × (8+4) = 24 × n_isects bytes
    For 3.22M isects: 77 MB per pass
    Total: 12 passes × 77 MB = 926 MB

  Thread-level parallelism: CUB internally uses as many thread blocks as needed
  Sync: None visible — CUB manages within kernel
```

**Output after sorting:** `isect_ids` and `flatten_ids` are now sorted by (depth_ascending, tile_id, image_id).

**Key insight**: The sorted order guarantees that all intersections for the same (image_id, tile_id) are contiguous and depth-sorted within that tile.

**For high n_isects**: 12 passes × ~24 bytes × n_isects dominates. For n_isects=175M (room tile16 expanded), this is 50.4 GB of traffic in a single sort call.

### Stage S6: Offset Encoding

```
[intersect_offset_kernel]
  Input: isect_ids, I, n_tiles
  Output: offsets  {I, tile_height, tile_width, int32}
  
  For each intersection (1 thread):
    Decode image_id and tile_id from isect_id (shift out depth: >> 16 for C1, or >> 32 for baseline)
    Compare with previous intersection's (image_id, tile_id)
    On boundary: fill offset for all tiles between previous and current
  
  Result: For each tile, offsets[tile] = index into isect_ids/flatten_ids
          where this tile's sorted intersections begin.
  
  offsets  {I × tile_height × tile_width, int32}
  For 1080p tile16: {1, 68, 120} = 8160 int32 = 32 KB
```

**Critical property**: The offset kernel exploits the sort order. It uses the fact that sorted isect_ids have all (image_id, tile_id) groups contiguous. It detects boundaries between groups and writes the cumulative offset for each tile.

**The offset kernel currently reads `isect_id >> 32`** to extract tile_id (line 227 in IntersectTile.cu). This ASSUMES depth occupies bits [0:31]. If the key layout changes, this must be updated.

### Stage S7: Rasterization Forward

```
[rasterize_to_pixels_3dgs_fwd_kernel]
  
  Grid: {I, tile_height, tile_width} blocks
  Block: {tile_size, tile_size} threads
  
  Per block (one tile):
    range_start = offsets[tile_id]
    range_end = (last block ? n_isects : offsets[tile_id + 1])
    
    For each batch (block_size intersections):
      Load flatten_ids[idx], means2d[g], conics[g], xy_opacity[g] into shared memory
      block.sync()
      
      For each gaussian in batch:
        For each pixel in tile (256 threads):
          Compute 2D Gaussian response at this pixel
          If visible (sigma > 0, alpha > threshold):
            Accumulate alpha-composited color
            Track last intersected index
      If transmittance < 1e-4: skip remaining batch
  
  Outputs:
    render_colors  {I, height, width, CDIM, float32}
    render_alphas  {I, height, width, float32}
    last_ids       {I, height, width, int32}
```

**Work**: Each block processes all Gaussians covering its tile. The number of Gaussians per tile varies from 0 to thousands depending on scene complexity.

**Memory reads per Gaussian loaded to shared memory**:
- flatten_ids[idx] (int32)
- means2d[g] (float32 × 2)
- conics[g] (float32 × 3)
- xy_opacity = {means2d, opacities} (float32 × 3)
- colors[g * CDIM] (float32 × CDIM)

### Stage S8: Offset Encoding Backward Path

Backward does NOT use isect_ids. It only uses:
- `flatten_ids` — to map sorted intersections back to original Gaussian indices
- `tile_offsets` — to determine per-tile intersection ranges
- `last_ids` — to know which was the last contributor per pixel

### Stage S9: Rasterization Backward

```
[rasterize_to_pixels_3dgs_bwd_kernel]
  
  Same grid/block layout as forward.
  
  Per block (one tile):
    Iterates batches of Gaussians BACKWARD (back to front)
    Uses flatten_ids to index original Gaussian data
    
    For each visible Gaussian:
      Compute per-pixel gradient contribution
      gpuAtomicAdd to v_means2d[g], v_conics[g], v_colors[g], v_opacities[g]
```

**Critical observation**: Backward kernel re-loads means2d, conics, colors, opacities from global memory — these are the same tensors produced by the projection and SH evaluation stages. It does NOT read isect_ids.

---

## 2. Memory Allocation Lifecycle

```
=== Per forward call ===

tiles_per_gauss      : allocated in Pass 1, lives through sort selection
  │
cum_tiles_per_gauss  : allocated by at::cumsum, discarded after Pass 2 allocation
  │
isect_ids            : allocated based on n_isects (Pass 1 result)
flatten_ids          : allocated based on n_isects (Pass 1 result)
  │
isect_ids_sorted     : allocated by sort function  
flatten_ids_sorted   : allocated by sort function
  │
── sort merges which buffer is "sorted" via DoubleBuffer.selector
  │
offsets              : allocated by isect_offset_encode  (small: ~32KB)
  │
── isect_ids_sorted retained in meta for backward (ONLY flatten_ids is consumed)
  │
=== End of forward call ===
isect_ids_sorted discarded (by PyTorch autograd graph) — wait, is it?
```

**Check**: `isect_ids_sorted` is returned as part of the tuple from `intersect_tile()`. It goes into `meta["isect_ids"]`. The `RasterizeToPixels.autograd.Function.forward` saves `flatten_ids` and `tile_offsets` for backward — **NOT** `isect_ids`. So `isect_ids_sorted` is only used for `isect_offset_encode()`, after which it can be freed.

However, the **non-sorted** `isect_ids` is freed immediately. The sorted version persists until `isect_offset_encode` completes.

---

## 3. Per-Stage Memory Summary (room, 1080p, tile16, ~3.22M intersections)

| Stage | Allocation | Shape | Bytes | Lifetime |
|-------|-----------|-------|-------|----------|
| S1 | means2d | nnz×2×f32 | ~3.2M | Projection → Rasterization |
| S1 | radii | nnz×2×i32 | ~3.2M | Projection → Intersect (Pass 1+2) |
| S1 | depths | nnz×f32 | ~0.8M | Projection → Intersect (Pass 1+2) |
| S3a | tiles_per_gauss | nnz×i32 | ~0.8M | Intersect Pass 1 → cumsum |
| S3b | cum_tiles_per_gauss | nnz×i64 | ~1.6M | cumsum → Intersect Pass 2 |
| S4 | isect_ids | n_isects×i64 | ~25.8M | Pass 2 → Sort → Offset |
| S4 | flatten_ids | n_isects×i32 | ~12.9M | Pass 2 → Sort → Rasterization(bwd) |
| S5 | isect_ids_sorted | n_isects×i64 | ~25.8M | Sort → Offset (then freed) |
| S5 | flatten_ids_sorted | n_isects×i32 | ~12.9M | Sort → Rasterization(fwd+bwd) |
| S6 | offsets | I×H_t×W_t×i32 | ~32K | Offset → Rasterization(fwd+bwd) |
| S5(aux) | CUB temp storage | ~n_isects×~48×2 | ~1.2GB | Only during sort |
| S7 | render_colors | I×H×W×CDIM×f32 | ~24M | Forward output (autograd saved) |
| S7 | render_alphas | I×H×W×f32 | ~8M | Forward output |
| S7 | last_ids | I×H×W×i32 | ~8M | Forward output (autograd saved) |

**Peak memory**: Primarily driven by CUB temp storage (~1.2GB for 3.22M isects) + full forward/backward tensors.

---

## 4. Synchronization Points

| Point | Type | Cost | Eliminable? |
|-------|------|------|-------------|
| Pass 1 → cumsum | Implicit (same stream) | ~0 | No (data dependency) |
| cumsum → .item() | **Host-Device sync** | ~5-30μs | Yes (pre-allocate max capacity) |
| .item() → Pass 2 alloc | CPU allocation | ~1-5μs | Yes (pre-allocate) |
| Pass 2 → Sort | Implicit (same stream) | ~0 | No (data dependency on isect_ids) |
| Sort → isect_offset | Implicit | ~0 | No |
| isect_offset → Rasterize | Implicit | ~0 | No |
| Rasterize → backward | Autograd engine sync | variable | No (autograd requirement) |

**Only one eliminable sync**: The `.item<int64_t>()` call in Intersect.cpp:80.

---

## 5. Why Global Sort Exists

The global radix sort exists because the rasterizer needs:

1. **Per-tile depth ordering**: Each tile must process Gaussians in front-to-back depth order for correct alpha compositing
2. **Contiguous tile groups**: The offset kernel must find all Gaussians for each (image, tile) pair as a contiguous range

The sorting key encodes: `depth | tile_id | image_id` as one 64-bit integer. After sorting, all intersections for each (image, tile) pair are contiguous and depth-ordered.

**Why not use per-tile sort?** The standard approach requires that tiles_per_gauss is known before we can create per-tile buckets. But tiles_per_gauss is the result of Pass 1. After Pass 1, we could theoretically allocate per-tile arrays instead of a global array, then sort each tile independently.

---

## 6. What's Mathematically Necessary

| Requirement | Why | Is global sort needed? |
|-------------|-----|----------------------|
| Per-tile depth ordering | Alpha compositing requires front-to-back | No — tile-local sort suffices |
| Contiguous tile groups | Offset kernel scans for boundaries | No — can be metadata instead of inferred |
| Full float32 depth precision | Preserves correct ordering of near-identical depths | No — sufficient precision for local ordering |
| Image ID separation | Multi-image batches need per-image independent tiles | No — per-image segments can be independent |
| Gradient path through sort | Sort is `@torch.no_grad()` — no gradients through keys/values | N/A |

## 7. What's an Implementation Choice

| Choice | Why it exists | Could be different |
|--------|---------------|-------------------|
| Global radix sort | Simplest/fastest CUB primitive | Tile-local segmented sort, bucket sort, or hybrid |
| 64-bit sort key | Pack image_id + tile_id + depth into one sort | Hierarchical: tile membership metadata + per-tile depth sort |
| Full float32 depth in key | Direct, no precision loss | Quantized depth bits sufficient for tile-local ordering |
| .item() sync | Must know n_isects to allocate | Pre-allocate capacity, skip sync |
| Two-pass intersect | Must count before writing | Single-pass: tile-local bounded queues with overflow |
| int64 for isect_ids | 64-bit key fits in one word | Could split into hi/lo uint32 |
| DoubleBuffer | Reduces temp memory from O(N+P) to O(P) | Optimal choice |
| flatten_ids as int32 | Maps sorted position back to original Gaussian | Could use smaller index or indirection |
| Shared memory batch loading | Standard CUDA optimization for register-limited kernels | Different occupancy/cooperative strategies |
