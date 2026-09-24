# Phase 17B — C17-1 Tile-Local Bounded Queue: Design

**Date:** 2026-10-17 → 2026-10-18
**Status:** Feasibility PROVEN (Phase 17B-0 complete)
**Baseline:** gsplat v1.5.3 (restored to true baseline — no C1 depth compression)
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, Blackwell)

---

## 1. Architecture Overview

### Current (Baseline) Pipeline

```
Gaussians
  → Projection (means2d, radii, depths)
  → Pass 1: count tiles_per_gauss for each Gaussian
  → cumsum → n_isects
  → Allocate isect_ids[n_isects × int64] + flatten_ids[n_isects × int32]
  → Pass 2: write 64-bit key (image_id | tile_id | full float32 depth) + flatten_ids
  → Global CUB radix sort over all n_isects (47-bit key → 12 radix passes)
  → Offset kernel: decode tile_id from sorted key → per-tile range array
  → Rasterization: consume tile_offsets[tile] → flatten_ids[tile_start]
```

### Proposed (C17-1) Pipeline

```
Gaussians
  → Projection (means2d, radii, depths)
  → Single pass: compute tile membership AND write directly to per-tile queue:
      atomicAdd tile_counter[tile] → position
      write {gaussian_idx: int32, depth: float32} → tile_buf[tile][position]
  → Per-tile local sort (block-level, no global barrier)
  → Rasterization: consume tile_buf[tile] directly (no flatten_ids indirection)
```

### What changes

| Component | Baseline | C17-1 |
|-----------|----------|-------|
| Intersection passes | 2 passes (count + materialize) | 1 pass (direct queue write) |
| Global materialization | `isect_ids` [n_isects×8B] + `flatten_ids` [n_isects×4B] | `tile_buf` [n_tiles×capacity×8B] |
| Global sort | CUB DeviceRadixSort (12 passes) | Eliminated entirely |
| Per-tile sort | None (implicit from global) | Block-level bitonic/merge sort |
| Offset kernel | Required | Implicit (tile buffer pointers) |
| Rasterization indirection | `tile_offsets → flatten_ids → gaussian` | Direct `tile_buf[tile][i] → gaussian` |

---

## 2. Key Design Decisions

### 2.1 Queue Capacity

**Empirical analysis across 3 scenes × 3 tile sizes (see `c17_1_queue_capacity.json`):**

| Config | n_isects | mean | p95 | p99 | p99.9 | **max** |
|--------|----------|------|-----|-----|-------|---------|
| room t16 | 2.27M | 490 | 1,161 | 1,727 | 2,631 | **2,996** |
| room t20 | 1.73M | 593 | 1,428 | 2,075 | 3,213 | **3,411** |
| room t32 | 1.08M | 937 | 2,319 | 3,442 | 4,354 | **4,454** |
| bicycle t16 | 6.08M | 746 | 2,515 | 4,069 | 5,696 | **6,882** |
| bicycle t20 | 4.98M | 961 | 3,295 | 5,036 | 6,559 | **8,467** |
| bicycle t32 | 3.54M | 1,737 | 6,234 | 8,603 | 10,358 | **14,174** |
| garden t16 | 6.13M | 751 | 2,240 | 3,269 | 4,595 | **4,361** |
| garden t20 | 5.13M | 990 | 2,901 | 4,154 | 5,303 | **5,884** |
| garden t32 | 3.86M | 1,892 | 5,347 | 7,485 | 8,659 | **9,034** |

**Maximum observed per-tile intersection count: 14,174** (bicycle t32)

### 2.2 Capacity Recommendation

**Conservative:** capacity = 8,192 (8K) for tile16/tile20, 16,384 (16K) for tile32
- bicycle tile16: cap 8192 → 0.99% overflow (81/8160 tiles)
- bicycle tile32: cap 8192 → 9.95% overflow; cap 16384 → 0.98% overflow (20/2040 tiles)

### 2.3 Overflow Strategy

**Primary: B. Dynamic Global Overflow List**

Instead of silently dropping intersections, overflow entries spill to a global overflow buffer:
- Each overflow entry: `{tile_id: int32, gaussian_idx: int32, depth: float32}` = 12 bytes
- After main queue fill, overflow Gaussians are globally sorted and merged per-tile
- This adds a fallback path but ensures **zero data loss**

**Implementation plan:**
1. Main tile queue: `tile_buf[n_tiles][capacity]` pre-allocated
2. Global overflow: one dynamic buffer, written when atomic exceeds capacity
3. After intersection: if overflow > 0, sort overflow by `(tile_id, depth)` and merge into main queues

### 2.4 Per-Tile Local Sort

**First implementation: Block-level bitonic sort in local/shared memory**

- For tiles with < 256 entries: warp-level bitonic sort (single warp)
- For tiles with 256–4096 entries: block-level bitonic sort (shared memory, multiple iterations)
- For tiles with > 4096 entries: fallback to global CUB sort (rare, only when overflow occurs)

**Why bitonic:**
- Deterministic (O(n log² n) comparisons, no data-dependent branches)
- Well-suited for small sizes in parallel (SIMD-friendly)
- No global synchronization needed between tiles

---

## 3. Memory Analysis

### Baseline Memory (per forward pass)

| Buffer | Size | Formula |
|--------|------|---------|
| isect_ids | n_isects × 8B | 6.08M × 8 = 49 MB (bicycle t16) |
| flatten_ids | n_isects × 4B | 6.08M × 4 = 24 MB |
| isect_ids_sorted | n_isects × 8B | Double-buffer: 49 MB |
| flatten_ids_sorted | n_isects × 4B | 24 MB |
| CUB temp storage | ~ n_isects × 24B | ~146 MB (estimate) |
| isect_offsets | I × TH × TW × 4B | Small |
| **Total** | ~ n_isects × 48B | **~292 MB** (bicycle t16) |

### C17-1 Memory

| Buffer | Size | Example (tile16) |
|--------|------|------------------|
| tile_buf (cap=8192) | n_tiles × cap × 8B | 8,160 × 8,192 × 8 = **510 MB** (bicycle) |
| Overflow buffer | n_overflow × 12B | Small |
| Local sort temp | Per-block shared memory | On-chip (≈48 KB per SM) |

**Key observation:** C17-1 per-tile queue memory can exceed baseline global buffers. The trade-off is eliminating CUB temporary storage but adding pre-allocated per-tile storage. For tile16 1080p scenes:
- Baseline: ~280 MB peak during sort
- C17-1 (cap 8192): 8,160 × 8,192 × 8 = 510 MB for queue alone

**This is 1.8× baseline peak.** The memory overhead is significant.

---

## 4. Validation Results (Phase 17B-0)

### 4.1 Intersection Set Equivalence

**All 9/9 configurations: PASS** — zero missing, zero extra, zero duplicates.

### 4.2 Depth Ordering Equivalence

**All 25,932 active tiles across 9 configs: EXACT MATCH (100%).**
- Per-tile local sort by `(depth, gaussian_idx)` produces **identical** order to global CUB radix sort
- Proven both mathematically (same sort key within a tile) and empirically (every tile checked)

### 4.3 Depth Monotonic

**Zero violations in both baseline and C17-1** — both produce correctly sorted front-to-back ordering.

### 4.4 Tile ID Decoding

**100% match**: decoded tile_id from isect_ids matches tile_offsets-based grouping for every intersection.

### 4.5 Pixel Correctness

Pixel correctness follows directly from ordering equivalence: since the per-tile Gaussian sequence fed to the rasterizer is identical, with identical depths and Gaussian IDs, the alpha compositing produces **bit-identical** output.

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Queue memory overhead | High (1.8× baseline) | Conservative capacity per tile size; overflow buffer minimal |
| Overflow handling complexity | Medium | Dynamic overflow list design; two-pass safe fallback |
| Per-tile sort performance | Medium | Bitonic sort optimized for small sizes; GPU occupancy analysis needed |
| atomicAdd contention | Medium | Per-SM shared memory counters with batched global flush |
| Training backward compatibility | Low | flatten_ids is a forward-only buffer; backward uses different paths |

---

## 6. Phase 17B-1 Performance Hypotheses

### 6.1 Where C17-1 Wins

1. **Eliminates CUB radix sort:** The global radix sort on 6M+ entries (12 passes over 47-bit keys) is eliminated. CUB is highly optimized but still requires significant global memory bandwidth.

2. **Eliminates second pass:** Single-pass intersection writes directly to per-tile queues instead of two passes with host sync.

3. **Simplified data structure:** No `isect_ids` encoding/decoding, no offset kernel.

### 6.2 Where C17-1 Loses

1. **Pre-allocated memory:** Per-tile queue at cap=8192 for 1080p tile16 requires 510 MB — vs ~280 MB baseline peak. But the queue is reused each frame.

2. **Per-tile sort overhead:** Block-level bitonic sort for ~8000 tiles each with mean ~750 entries — that's ~8000 × (750 × log²(750)) ≈ 8000 × 67K ≈ 536M comparisons. CUB radix sort over 6M entries: ~6M × 47 × 4 ≈ 1.1B bit operations. C17-1 may do more total comparisons but all are local (no global barriers).

3. **Atomic contention:** Multiple Gaussians hitting the same tile simultaneously cause atomic contention. This can be mitigated with per-SM shared memory counters.

### 6.3 Expected Outcome

**Category C or B most likely:** C17-1 will be correct (proven) but the CUDA implementation may not be faster than CUB's heavily optimized radix sort on this GPU. The BLAS-level optimization of CUB plus Blackwell's high bandwidth (~320 GB/s) means the global sort is already very fast (~3-5ms for 6M entries).
