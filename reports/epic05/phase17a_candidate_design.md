# Phase 17A — Candidate Design

**Date:** 2026-10-17
**Phase:** 17A (source + literature + dataflow + candidate design only — NO implementation)

---

## Executive Summary

After exhaustive source tracing (IntersectTile.cu, Intersect.cpp, RasterizeToPixels3DGSFwd.cu, RasterizeToPixels3DGSBwd.cu, SegmentedSort.cu), literature reconnaissance (gsplat, diff-gaussian-rasterization, HiGS, TC-GS, FlashGS, Speedy-Splat), and past-phase results review (C1 depth compression: <1% benefit; segmented sort: 1.9-4.5× slower), three genuinely novel candidates are proposed.

**None is a simple parameter tuning.** Each requires structural changes to the intersection/sorting pipeline or the backward kernel.

---

## C17-1: Tile-Local Bounded Queues (Eliminate Global Intersection Materialization)

### Summary
Replace the two-pass global materialization + global radix sort with per-tile bounded queues written directly during intersection generation, followed by per-tile local sorting.

### source_location
- `IntersectTile.cu` — `intersect_tile_kernel`, `radix_sort_double_buffer`
- `Intersect.cpp` — `intersect_tile()`
- `RasterizeToPixels3DGSFwd.cu` — consumption of `flatten_ids`

### current_behavior
```
Pass 1: count tiles_per_gauss per Gaussian → tiles_per_gauss[nnz]
cumsum → n_isects
Allocate isect_ids[n_isects] + flatten_ids[n_isects]
Pass 2: write isect_ids (64-bit key: image_id | tile_id | depth) + flatten_ids
Global CUB radix sort over all n_isects (12 passes for 47-bit key)
Offset kernel: decode tile_id from sorted key, build per-tile range array
```

### proposed_behavior
```
Allocate per-tile bounded buffers: tile_buf[n_tiles][max_per_tile] each 8+4 bytes
max_per_tile = estimated from worst-case scene statistics (e.g., 4096)
Total memory: n_tiles × max_per_tile × 12 bytes
  For 1080p tile16 (8160 tiles), max_per_tile=4096: 8160×4096×12 = 401 MB (pre-allocated)

Single pass: for each Gaussian, compute tile footprint, for each covered tile:
  atomicAdd to tile counter (shared memory per SM → global flush on overflow)
  Write {gaussian_idx, float_depth} to tile's buffer position
  If buffer full → spill to global overflow array (fallback global sort)

Per-tile sort: independent sort of each tile's entries by depth
  Small tiles (≤ 256 entries): block-level bitonic sort in shared memory
  Medium tiles (≤ 4096 entries): shared-memory radix sort (4-bit or 8-bit)
  Large tiles (overflow): fallback to global CUB sort

Offset construction: implicit — per-tile buffer start pointer IS the offset
  No offset kernel needed

Rasterization: per-tile depth-sorted buffer consumed directly
  No flatten_ids indirection needed
  Tile's Gaussian list read from pre-allocated buffer
```

### changed_data_structure
- **REMOVED**: `isect_ids` (global 64-bit key array)
- **REMOVED**: `flatten_ids` (global int32 indirection array)
- **REMOVED**: `isect_ids_sorted`, `flatten_ids_sorted` (double-buffer copies)
- **REMOVED**: CUB temporary storage for global radix sort
- **REMOVED**: `intersect_offset_kernel` (tile ranges are implicit in buffer pointers)
- **ADDED**: `tile_buffers[n_tiles][max_per_tile][2]` — per-tile Gaussian index + depth
- **ADDED**: `tile_counts[n_tiles]` — per-tile fill level (atomic counter)
- **ADDED**: Per-tile local sort kernels (block-level)

### expected_work_reduction
- **Intersection write**: same number of writes (same n_isects), but no 64-bit key encoding
- **Global sort**: ELIMINATED entirely
- **Per-tile sort**: O(n_tiles × mean_items_per_tile × log²(mean)) for bitonic
  - For tile16 at 1080p, n_tiles = 8160, mean_isects_per_tile ≈ 3.22M/8160 = 395
  - Bitonic sort of 395 items: ~395 × log²(395) ≈ 395 × 87 ≈ 34K comparisons per tile
  - Total: 8160 × 34K = 278M comparisons vs CUB global sort of 3.22M × 12 passes = 38.6M element-pass operations
  - BUT: per-tile sort is in shared memory (much higher throughput) vs global sort (global memory bandwidth bound)
- **Offset kernel**: ELIMINATED

### expected_memory_reduction
- Global intersection buffers (isect_ids, flatten_ids, 2 sorted copies): 4 × n_isects × (8+4) = ~155 MB for 3.22M isects
  → **Eliminated**
- Per-tile buffers: pre-allocated 401 MB (could be reduced with smarter allocation)
- CUB temp storage (~1.2 GB) → **Eliminated**
- **Net**: ~1.4 GB peak reduction at expense of ~400 MB pre-allocated per-tile buffers

### expected_sync_reduction
- The `.item()` host sync (Intersect.cpp:80) becomes OPTIONAL — because tile buffers are pre-allocated, we don't need to know n_isects before Pass 2. However, we still need to know if overflow occurred (to trigger fallback sort).

### forward_risk
- **MEDIUM**. Per-tile depth ordering must match current global sort ordering exactly. For bitonic sort, correctness depends on comparators. Slight numerical differences are possible if depth comparison uses IEEE 754 total ordering vs float comparison.
- **Overflow handling is the main risk**: if per-tile buffers overflow, falling back to global sort introduces a code path that must be tested.
- Shared memory atomic counters may have bank conflicts that limit throughput.

### backward_risk
- **LOW-MEDIUM**. Backward currently reads `flatten_ids` to map sorted intersection → Gaussian index. Under C17-1, the Gaussian index is directly stored in the per-tile buffer. Format changes but semantics are identical.
- No gradient path through sorting (sort is `@torch.no_grad()`).

### gradient_risk
- **NONE**. Same as baseline — sort is non-differentiable.

### training_risk
- **LOW**. No changes to Gaussian parameter gradients. Training dynamics unchanged.

### measurement_plan
1. Implement per-tile buffer allocation in Intersect.cpp (pre-allocated, fixed max_per_tile)
2. Modify intersect_tile_kernel to write per-tile buffer (atomicAdd per tile counter, write index+depth)
3. Implement per-tile block-level bitonic sort kernel
4. Implement overflow path → global sort (for tiles exceeding max_per_tile)
5. Forward-only benchmark: compare timing with baseline on room, bicycle, garden
6. Forward+backward benchmark: verify gradient norms match baseline (max_rel_diff < 1e-5)
7. Pixel equivalence: `allclose(baseline, candidate, atol=1e-4, rtol=1e-4)`
8. If all gates pass: 500-step training sanity

### existing_reference
**NOT_EXISTING** in any differentiable 3DGS renderer. HiGS uses per-macro-tile sorting but is inference-only and uses a fundamentally different macro-tile approach (8×8 Gaussian groups, not pixel tiles). The bounded queue + per-tile local sort for differentiable rendering is novel.

---

## C17-2: Two-Phase Sorting (Tile-Metadata-First then Depth-Local)

### Summary
Break the single global radix sort into two phases: Phase A sorts by (image_id | tile_id) only (narrow key, few CUB passes) to establish contiguous tile groups. Phase B sorts each tile group by depth using lightweight local sorting.

### source_location
- `IntersectTile.cu` — `radix_sort_double_buffer` (lines 296-339)
- `Intersect.cpp` — sort selection (lines 118-144)

### current_behavior
```
Single global radix sort on 47-bit key:
  depth(32) | tile_id(14) | image_id(1)
  12 CUB radix passes
  Result: all tiles contiguous AND depth-ordered simultaneously
```

### proposed_behavior
```
Phase A: Sort by (image_id | tile_id) only — 15-bit key
  Sort key: tile_id (14 bits) | image_id (1 bit)
  begin_bit = 0, end_bit = 15
  CUB passes: ceil(15/4) = 4 passes
  Result: all items for same (image, tile) are contiguous
  Phase A sort key does NOT include depth

Phase B: Per-tile depth sort
  After Phase A, the offset kernel can identify tile ranges (same as current)
  But items within each tile are NOT depth-ordered
  
  New kernel: per-tile depth sort within each tile's range
  For each tile with K isects:
    Load isect_ids range into shared memory (each contains depth in lower 32 bits)
    Sort by depth using block-level bitonic sort (shared memory)
    Write sorted isect_ids and flatten_ids back to global memory

  After Phase B: all tiles are depth-ordered (same as baseline)
  
  Offset kernel: unchanged (same as current, reads sorted isect_ids)
```

### changed_data_structure
- **SAME** `isect_ids`, `flatten_ids` (same format and size)
- **SAME** `isect_ids_sorted`, `flatten_ids_sorted` (double-buffer)
- **NEW** Phase A sort kernel (narrow key range)
- **NEW** Phase B kernel (per-tile depth sort)
- **SAME** `intersect_offset_kernel` (unchanged)

### expected_work_reduction
- Phase A: 4 CUB passes × 24 bytes per element = 96 bytes/element traffic
- Baseline: 12 CUB passes × 24 bytes = 288 bytes/element traffic
- Phase B: per-tile sort overhead
  - For mean 395 items per tile: bitonic sort ≈ 395 × log²(395) ≈ 34K comparisons
  - Each comparison is float32 depth compare (shared memory)
  - Each tile also needs to sort flatten_ids as values (4-byte swap per comparison)
  - For 8160 tiles: ~280M operations in shared memory (fast, ~0.1-0.3ms)
- **Net sorting memory traffic**: 96 vs 288 bytes/element → **67% reduction in global memory sort traffic**
- Plus local sort overhead (~0.2ms estimated)

### expected_memory_reduction
- **SAME** peak memory (same buffer sizes as baseline)
- CUB temp storage reduced proportionally to fewer passes
- But Phase B adds no new allocations (operates in-place on sorted buffers)

### expected_sync_reduction
- One additional kernel launch (Phase B) between Phase A sort and offset kernel
- No new host-device synchronization

### forward_risk
- **LOW-MEDIUM**. Phase B must produce identical depth ordering within each tile. Bitonic sort with full float32 comparison is exact (same IEEE 754 ordering). However, the tile boundary detection (offset kernel) runs after Phase A but before Phase B — this is correct because Phase A already establishes tile contiguity.
- **Critical**: The offset kernel currently reads `isect_id >> 32` to extract tile_id. After Phase A, `isect_id` contains... what? The current key format has depth in lower 32 bits. If we change Phase A to sort by tile_id only, the key must be different. Options:
  - **Option 1**: Restructure key as `depth(32) | tile_id(14) | image_id(1)`, sort Phase A only on upper bits (begin_bit = 15, end_bit = 47). CUB can do this. But then depth occupies bits [0:31] which are NOT sorted in Phase A — meaning tile_id groups are depth-unsorted within the group. Then Phase B sorts within each group using lower 32 bits.
  - **Option 2**: Use separate arrays: tile_id array for Phase A sort, depth array for Phase B. But this changes the data structure.

### backward_risk
- **NONE**. Backward reads flatten_ids which are reordered by both Phase A and Phase B. The final order is identical to baseline (tile groups contiguously, depth-sorted within tile). flatten_ids maps to the same Gaussian indices.

### gradient_risk
- **NONE**. Same as baseline.

### training_risk
- **NONE** (if pixel equivalence is maintained).

### measurement_plan
1. Modify sort dispatch in Intersect.cpp to implement two-phase flow
2. Phase A: narrow-key CUB sort on (tile_id | image_id) portion
3. Write Phase B kernel: per-tile block-level bitonic depth sort
4. Forward-only benchmark: compare Phase A+B timing vs baseline global sort
5. Verify pixel equivalence with `allclose(baseline, candidate)`
6. If successful: forward+backward benchmark, gradient norm comparison
7. If all gates pass: 500-step training sanity

### existing_reference
**NOT_EXISTING** in any 3DGS renderer. The standard approach (both gsplat and diff-gaussian-rasterization) uses a single global radix sort. The two-phase approach (contiguity sort + local depth sort) is a novel adaptation of hierarchical sorting ideas from database research to Gaussian splatting.

**DERIVED_FROM**: General hierarchical sorting literature (sorting by group key, then by sort key within groups). The specific adaptation to tile-grouped Gaussian intersections is novel.

---

## C17-3: Forward-Metadata Cache for Backward Coalesced Loading

### Summary
The backward kernel independently loads the same per-Gaussian attributes (means2d, conics, colors, opacities) that the forward kernel already loaded. For Gaussians spanning many tiles (e.g., 8×8=64 tiles), this means each attribute is loaded ~64 times — once per tile block. C17-3 restructures the backward kernel to pre-load Gaussian attributes into a persistent cache indexed by Gaussian ID, reducing redundant global memory traffic.

### source_location
- `RasterizeToPixels3DGSBwd.cu` — `rasterize_to_pixels_3dgs_bwd_kernel` (lines 140-151)
- Specifically lines 141-151: per-thread loading of Gaussian attributes via `flatten_ids[idx]` indirection

### current_behavior
```cuda
// For each batch in each tile (backward direction):
int32_t g = flatten_ids[idx];  // random access Gaussian index
id_batch[tr] = g;
const vec2 xy = means2d[g];                    // global memory load
const float opac = opacities[g];               // global memory load
xy_opacity_batch[tr] = {xy.x, xy.y, opac};
conic_batch[tr] = conics[g];                   // global memory load
for (uint32_t k = 0; k < CDIM; ++k)
    rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];  // global memory load

// Later in same kernel:
gpuAtomicAdd(v_means2d + 2*g, ...);   // global memory atomic write
gpuAtomicAdd(v_conics + 3*g, ...);    // global memory atomic write
gpuAtomicAdd(v_colors + CDIM*g, ...); // global memory atomic write
gpuAtomicAdd(v_opacities + g, ...);   // global memory atomic write
```

Each backward kernel block (one tile) independently loads the Gaussian attributes it needs. A Gaussian visible in T tiles has its attributes loaded T times.

### proposed_behavior
```cuda
// Pre-pass: collect unique Gaussian IDs needed by this tile block
// Load Gaussian attributes once into a block-level cache

// Phase 1 (before main backward loop):
uint32_t n_unique = 0;
for each isect in tile:
    g = flatten_ids[isect];
    if (g not already cached):
        cached_g[n_unique] = g;
        cached_means2d[n_unique] = means2d[g];
        cached_conics[n_unique] = conics[g];
        cached_rgbs[n_unique * CDIM + k] = colors[g * CDIM + k];
        cached_opacities[n_unique] = opacities[g];
        g_to_cache_slot[g] = n_unique;
        n_unique++;

// Phase 2 (main backward loop — same as current but using cache):
for each batch:
    for each gaussian in batch:
        g = flatten_ids[idx];
        cache_slot = g_to_cache_slot[g];  // shared memory lookup
        vec2 xy = cached_means2d[cache_slot];
        ...

// Phase 3 (atomic write — same as current):
gpuAtomicAdd(v_means2d + 2*g, ...);
```

**Alternatively (simpler):** Create a persistent GPU-side hash table or index array mapping Gaussian ID → pre-cached attribute offset. This is built once per forward pass and reused across all tile blocks in backward.

### changed_data_structure
- **NEW**: `gaussian_attr_cache[nnz][4+CDIM]` — compact cache of means2d.xy, conics.xyz, opacity, colors[CDIM]
- **NEW**: `gaussian_to_cache[nnz]` — mapping from Gaussian index to cache slot (or identity mapping if cache mirrors Gaussian array)
- **No changes to sorting pipeline** — this is a pure backward kernel optimization

### expected_work_reduction
- **Eliminates redundant attribute loads**: Each Gaussian's attributes loaded once per forward pass, not T times (where T = tile count for that Gaussian)
- For large Gaussians covering 16-64 tiles: 16-64× fewer global memory reads of the same attributes
- **Total traffic saved**: Σ_tiles Σ_g_in_tile sizeof(attributes) = total_isects × sizeof(attributes)
  - For room: ~3.22M isects × 28 bytes = ~90 MB per backward call
  - But these are global memory loads, and the GPU L2 cache already provides some coalescing
  - Real savings depends on L2 hit rate for current access pattern

### expected_memory_reduction
- **NEGATIVE**: Adds cache array: nnz × (8+12+4+CDIM×4) bytes
  - For nnz=200K, CDIM=3: 200K × 36 = 7.2 MB additional memory
  - This is a small increase (~0.5% of peak memory)

### expected_sync_reduction
- **NONE**. No new synchronization points.

### forward_risk
- **NONE**. Forward unchanged.

### backward_risk
- **LOW**. The cache must be populated before the backward kernel launches. If the cache is built during the forward pass (as part of `meta`), the autograd graph must save it. This is safe because the saved tensors are accessed only during backward.
- The cache is a computation speedup, not a mathematical change. Gradients are identical (same arithmetic, same atomics).

### gradient_risk
- **NONE**. Same atomicAdd pattern, same numeric computation. Only memory access pattern changes.

### training_risk
- **NONE**. No mathematical changes.

### measurement_plan
1. Add `gaussian_attr_cache` tensor construction to `rasterize_to_pixels()` forward (in meta dict, saved for backward)
2. Modify backward kernel to read from cache instead of indirection via flatten_ids + means2d/conics/colors/opacities global arrays
3. Benchmark backward-only timing: compare baseline vs cached
4. Verify gradient norms: `torch.allclose(grad_baseline, grad_cached, atol=1e-5, rtol=1e-5)`
5. If successful: forward+backward benchmark, 500-step training sanity

### existing_reference
**NOT_EXISTING** in any 3DGS renderer. The current backward kernel uses the same per-tile indirection pattern as the original Inria implementation. Pre-loading Gaussian attributes into a linear cache for backward is a novel adaptation of standard "data layout optimization" techniques from GPU computing.

**NOTE**: This is closely related to "texture caching" and "read-only cache hints" (`__ldg()`), which are existing hardware features. The proposed software cache differs by being explicit and avoiding redundant global reads entirely.

---

## Candidate Ranking

### Scoring (1-5, higher = better)

| Dimension | C17-1 (Tile Queues) | C17-2 (Two-Phase Sort) | C17-3 (Bwd Cache) |
|-----------|:-------------------:|:---------------------:|:-----------------:|
| 1. Evidence strength | 4 | 4 | 3 |
| 2. Bottleneck relevance | 5 | 4 | 3 |
| 3. Potential workload reduction | 5 | 3 | 2 |
| 4. Potential memory reduction | 5 | 1 | -1 (adds memory) |
| 5. Potential E2E impact | 4 | 2 | 2 |
| 6. Implementation feasibility | 2 | 3 | 4 |
| 7. Forward correctness risk | 3 | 2 | 1 |
| 8. Backward risk | 2 | 1 | 2 |
| 9. Training risk | 2 | 1 | 1 |
| 10. Research novelty | 5 | 4 | 3 |

### Analysis

**C17-1** (Tile-Local Bounded Queues) has the highest potential impact but the highest implementation risk and the most code changes. It eliminates the entire global sort + offset kernel pipeline, replacing it with per-tile local operations. The main risk is overflow handling and ensuring per-tile depth sorting is correct and efficient.

**C17-2** (Two-Phase Sorting) is more conservative — it keeps the existing buffer structure and only changes the sorting algorithm. The potential benefit is smaller (sort-only improvement), but the implementation risk is lower. The pixel equivalence check is straightforward.

**C17-3** (Backward Metadata Cache) has the lowest potential impact (backward-only, and L2 cache already provides partial benefit) but also the lowest risk. It's the most straightforward to implement and verify.

---

## TOP 3 Recommendation

### 🥇 C17-1 (Tile-Local Bounded Queues) — First Implementation Candidate

**Rationale**: Highest potential impact across all dimensions. Directly addresses the core problem: global intersection materialization is the source of both memory pressure and sorting cost. Per-tile local sort avoids the CUB segmented sort performance issue (Phase 14B finding) because:
- CUB's segmented sort creates I segments (one per image) — each segment is large (millions of items)
- Per-tile local sort creates n_tiles segments (thousands) — each segment is small (tens to hundreds of items)
- Small segments can be sorted in shared memory (block-level), which is much faster than global-memory CUB
- The existing segmented sort failure (1.9-4.5× slower) is NOT predictive of per-tile local sort performance

**Implementation order**: 
1. Pre-allocate per-tile buffers
2. Modify intersect pass to write per-tile (single-pass, no cumsum needed)
3. Implement per-tile block-level sort
4. Overflow fallback path
5. Modify rasterization to consume per-tile sorted data
6. Benchmark

### 🥈 C17-2 (Two-Phase Sorting)

**Rationale**: Preserves existing data structures, lower risk, measurable intermediate milestone. Phase A (narrow-key CUB sort) can be tested independently. Phase B (per-tile local sort) can reuse some of the same per-tile sorting infrastructure from C17-1.

**Implementation order**: 
1. Implement Phase A: narrow-key CUB sort (15-bit) separately
2. Measure Phase A timing vs baseline (expect ~3× faster for the sort phase)
3. Implement Phase B: per-tile bitonic depth sort
4. Measure combined Phase A+B timing
5. Verify pixel equivalence

### 🥉 C17-3 (Backward Metadata Cache)

**Rationale**: Lowest risk, most incremental. Even if the benefit is small (1-5% backward), it's a clean optimization with zero correctness risk. It can also be combined with C17-1 or C17-2.

**Implementation order**:
1. Add Gaussian attribute cache to forward metadata (saved for backward)
2. Modify backward kernel to use cache
3. Benchmark backward-only timing
4. Verify gradient equivalence

---

## Risk Summary Table

| Candidate | Highest Risk | Mitigation |
|-----------|-------------|------------|
| C17-1 | Overflow handling + per-tile sort correctness | Conservative max_per_tile estimation, unit tests for small/medium/large tiles |
| C17-2 | Tile boundary detection between Phase A and Phase B | Offset kernel runs after Phase A (contiguous groups exist), Phase B only sorts within groups |
| C17-3 | Cache memory overhead + added complexity | Cache size is small (~7MB), can be gated behind `use_cache=True` flag |

---

## Decision: Phase 17B Scope

**Recommended**: Implement **C17-1 (Tile-Local Bounded Queues)** as Phase 17B.

**Contingency**: If C17-1 proves infeasible (overflow handling too complex, per-tile sort performance regression), fall back to **C17-2 (Two-Phase Sorting)** as Phase 17C.

**Composability**: C17-3 can be implemented independently after either C17-1 or C17-2 is stable, or as a parallel effort.
