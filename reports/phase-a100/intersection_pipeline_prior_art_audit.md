# Corrected Intersection Pipeline + Verified Prior-Art Audit

**Date:** 2026-09-06  
**Scale activation:** CORRECT (`torch.exp(scales)` applied throughout)  
**Environment:** A100 PCIe 40GB (SM80)  
**gsplat version:** 1.5.3  

---

## Current Pipeline (Complete Source Trace)

```
Gaussian [N, 3]
  │
  ▼
3D-to-2D Projection (FullyFusedProjection.cu: project_kernel)
  │  means2d [nnz, 2] float32     ← screen-space mean
  │  radii [nnz, 2] float32       ← screen-space radius (pixels)
  │  depths [nnz] float32         ← depth (camera space)
  │  conics [nnz, 3] float32      ← inverse 2×2 covariance (NOT used in intersection)
  │
  ▼
Pass1 — Tile Bounds + Count (IntersectTile.cu: intersect_tile_kernel, first_pass=true)
  │  Grid: ceil(N_visible/256) × 256 threads, 0 shared memory
  │  Each thread → 1 Gaussian
  │  tile_min = floor(mean/tile_size - radius/tile_size)  [clamped to [0, tile_width))
  │  tile_max = ceil(mean/tile_size + radius/tile_size)   [clamped to [0, tile_height))
  │  tiles_per_gauss = (tile_max.y - tile_min.y) × (tile_max.x - tile_min.x)
  │  Output: tiles_per_gauss [nnz] int32
  │
  ▼
Prefix Sum (Intersect.cpp:79 via at::cumsum)
  │  Inclusive prefix sum → cum_tiles_per_gauss [nnz] int64
  │  N_isect = cum_tiles_per_gauss[-1] (last element = total pairs)
  │  CPU sync point: .item<int64_t>()
  │
  ▼
Pass2 — Pair Materialization (IntersectTile.cu: intersect_tile_kernel, first_pass=false)
  │  Grid: IDENTICAL to Pass1 (ceil(N_visible/256) × 256 threads)
  │  Same tile bounds computation
  │  cur_idx = (idx==0) ? 0 : cum_tiles_per_gauss[idx-1]
  │  for y in tile_min.y..tile_max.y:
  │    for x in tile_min.x..tile_max.x:
  │      tile_id = y * tile_width + x
  │      isect_ids[cur_idx] = depth_upper | (tile_id << 16) | iid_enc    ← KEY
  │      flatten_ids[cur_idx] = static_cast<int32_t>(idx)                 ← VALUE
  │      ++cur_idx
  │  Output: isect_ids [N_isect] int64  +  flatten_ids [N_isect] int32
  │
  ▼
CUB Radix Sort (IntersectTile.cu: radix_sort_double_buffer)
  │  cub::DeviceRadixSort::SortPairs
  │  Key: int64, Value: int32, DoubleBuffer
  │  end_bit = 16 + tile_n_bits + image_n_bits  (48-50 bits)
  │  Output: isect_ids_sorted [N_isect] int64 + flatten_ids_sorted [N_isect] int32
  │
  ▼
Offset Encode (intersect_offset kernel)
  │  Reads sorted isect_ids → produces tile_offsets (per-tile range)
  │
  ▼
Rasterization (rasterize_to_pixels kernel)
  │  Reads flatten_ids_sorted + tile_offsets + Gaussian attributes
  │  Alpha compositing per tile
```

---

## Pass1 Detailed Analysis

### Q1: What does each thread compute?

**Each thread processes exactly ONE Gaussian.** `thread_rank()` maps to Gaussian index `idx`. Grid size = `ceil(n_elements/256)` where `n_elements = I×N` (batched) or `nnz` (packed). This is the same for Pass1 and Pass2.

### Q2: Tile bounds computation

```cuda
float radius_x = radii[idx * 2];
float radius_y = radii[idx * 2 + 1];
if (radius_x <= 0 || radius_y <= 0) {
    if (first_pass) tiles_per_gauss[idx] = 0;
    return;
}
float tile_radius_x = radius_x / tile_size;
float tile_radius_y = radius_y / tile_size;
float tile_x = mean2d.x / tile_size;
float tile_y = mean2d.y / tile_size;
uint2 tile_min, tile_max;
tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);
if (first_pass) {
    tiles_per_gauss[idx] = (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x);
}
```

### Q3: Is Pass1 shape-aware?

**NOT SHAPE AWARE.**

| Component | Status |
|:----------|:-------|
| `radius` | **USED** — scalar radius from projection |
| `conic` (inverse covariance) | **NOT USED** — not present in kernel parameters |
| `opacity` | **NOT USED** — no opacity culling in intersection |
| Ellipse test | **NOT PERFORMED** — pure AABB |
| AABB | **USED** — axis-aligned bounding box from radius |
| Tile intersection | **Conservative** — any tile whose AABB overlaps the Gaussian AABB |

**Verdict:** `ALREADY NECESSARY BUT NOT SUFFICIENT` — the current AABB correctly captures the Gaussian's screen-space extent, but it does NOT use the conic matrix for a tighter ellipse-tile intersection test.

---

## Prefix Sum

| Property | Value |
|:---------|:------|
| Implementation | `at::cumsum` (PyTorch GPU, PyTorch's internal segmented scan) |
| Location | `Intersect.cpp` lines 79-80 |
| Code | `cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);`<br>`n_isects = cum_tiles_per_gauss[-1].item<int64_t>();` |
| Type | **Inclusive** sum. `n_isects` = last element |
| CPU sync | `item<int64_t>()` is a GPU→CPU synchronization point |

### Example

```
tiles_per_gauss: [3, 2, 5]
cumsum:          [3, 5, 10]

G0 → range [0, 3)   (writes pairs 0, 1, 2)
G1 → range [3, 5)   (writes pairs 3, 4)
G2 → range [5, 10)  (writes pairs 5, 6, 7, 8, 9)
```

### Timing

| Scene | Prefix Sum (ms) | Share of isect_no_sort |
|:------|:---------------:|:----------------------:|
| room | 0.015 | 0.1% |
| bicycle | 0.015 | 0.2% |
| garden | 0.017 | 0.9% |

### Is Prefix Sum Required?

**NOT DETERMINED.** Per-Gaussian tile coverage info is required for rasterization, but the exact `count → scan → materialize → sort` order is pipeline organization, not a mathematical requirement.

---

## Pass2 Detailed Analysis

### Q1: What does each thread process?

Each thread processes one Gaussian, iterates **ALL** covered tiles, and writes a key-value pair for each tile. The same kernel is launched with an identical grid to Pass1 (`ceil(N_visible/256)` blocks, 256 threads/block), but with `cum_tiles_per_gauss` provided (non-null).

### Q2: Does each Gaussian enumerate all covered tiles?

**YES.** Every tile in the Gaussian's AABB is enumerated:

```cuda
for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
    for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
        int64_t tile_id = i * tile_width + j;
        isect_ids[cur_idx] = depth_upper | (tile_id << 16) | iid_enc;
        flatten_ids[cur_idx] = static_cast<int32_t>(idx);
        ++cur_idx;
    }
}
```

### Q3: Where is this enumeration?

Inside `intersect_tile_kernel`, second-pass branch (`cum_tiles_per_gauss != nullptr`). Loop is row-major: Y-outer, X-inner.

### Q4: Pairs per Gaussian

`tiles_per_gaussian = (tile_max.y - tile_min.y) × (tile_max.x - tile_min.x)` — the area of the Gaussian's bounding box in tile space.

| Scene | Avg. tiles/Gaussian |
|:------|:-------------------:|
| room | 1,585 |
| bicycle | 1,358 |
| garden | 112 |

### Q5: Key and Value

**Key** (`isect_ids`, `int64` — 8 bytes):

```
63              48  47        47−tile_n_bits  47−tile_n_bits−image_n_bits  0
┌─────────────────┬─────────────────────────┬──────────────────────────────┐
│  depth_upper    │       tile_id           │         image_id             │
│  16 bits         │    tile_n_bits bits     │     image_n_bits bits        │
└─────────────────┴─────────────────────────┴──────────────────────────────┘
```

- `depth_upper`: float32 depth re-interpreted as int32, zero-extended to 64 bits, `>> 16`
- `tile_id`: `y * tile_width + x`
- `image_id`: image index (0 for single-camera)
- Total key bits: `16 + tile_n_bits + image_n_bits` = **48-50 bits** → CUB sort pass count: 6-7

**Value** (`flatten_ids`, `int32` — 4 bytes):
- Global Gaussian index `[0, n_elements)` as `static_cast<int32_t>(idx)`

### Q6: Write position

```cuda
cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
// then for each pair:
output[cur_idx] = ...;
++cur_idx;
```

---

## Materialization Analysis

| Property | Value |
|:---------|:------|
| Key dtype | `int64` (8 bytes) |
| Value dtype | `int32` (4 bytes) |
| **Bytes/pair** | **12** |
| Tensors | Two separate contiguous GPU arrays (NOT interleaved) |

### Raw Output Storage (Pass2 writes)

| Scene | N_isect | isect_ids | flatten_ids | **Total** |
|:------|:-------:|:---------:|:-----------:|:---------:|
| room | 25,475,247 | 194.3 MB | 97.2 MB | **291.5 MB** |
| bicycle | 18,550,853 | 141.5 MB | 70.8 MB | **212.3 MB** |
| garden | 11,257,252 | 85.9 MB | 42.9 MB | **128.8 MB** |

**Raw storage ≠ total memory traffic.** In the `sort=True` path:
- Each pair is written during Pass2: **12 bytes/pair**
- The CUB sort reads the unsorted input and writes sorted output (via DoubleBuffer): **0-24 bytes/pair** depending on selector
- CUB temporary storage adds additional allocation (managed by `CUDACachingAllocator`)

---

## Key-Value Lifecycle

```
Pass2 output:
  isect_ids[unsorted]  +  flatten_ids[unsorted]
         │                        │
         ▼                        ▼
CUB Radix Sort:
  isect_ids[sorted]   +  flatten_ids[sorted]
         │                        │
         ▼                        │
Offset Encode:
  isect_ids[sorted] → tile_offsets │
  (isect_ids discarded)             │
                                    ▼
Rasterization:
  flatten_ids[sorted] + tile_offsets + Gaussian attributes
```

| Question | Answer |
|:---------|:-------|
| Q1: Does the key serve sorting? | **YES.** Key is designed EXCLUSIVELY for sorting. Encodes `depth|tile_id|image_id`. |
| Q2: Does the value serve Gaussian lookup? | **YES.** `flatten_ids` = Gaussian index for attribute lookup in rasterization. |
| Q3: What fields reach rasterization? | `flatten_ids[sorted]` + tile_offsets (from offset encoding of sorted isect_ids). |
| Q4: What is intermediate? | Both `isect_ids[unsorted]` and `isect_ids[sorted]` are entirely intermediate. The key is created for CUB sorting and immediately discarded after offset encoding. |

---

## Stage Timings (Split Instrumentation)

### Intersection Generation Breakdown

| Scene | Pass1 (ms) | Prefix (ms) | Pass2 (ms) | Total (ms) |
|:------|:---------:|:----------:|:---------:|:----------:|
| **room** | 0.01 | 0.02 | 10.70 | 10.72 |
| **bicycle** | 0.01 | 0.02 | 6.59 | 6.61 |
| **garden** | 0.02 | 0.02 | 2.00 | 2.03 |

### Share of Intersection Generation

| Scene | Pass1% | Prefix% | Pass2% |
|:------|:-----:|:-------:|:-----:|
| room | 0.1% | 0.1% | **99.8%** |
| bicycle | 0.2% | 0.2% | **99.7%** |
| garden | 0.7% | 0.9% | **98.4%** |

---

## Bottleneck Hierarchy

```
Intersection Generation (2.03-10.72ms)
│
├── Pass1: tile bounds + count (0.01-0.02ms) MINOR
│     Grid: ceil(N_visible/256) × 256 threads, 0 shared memory
│     O(N_visible): each thread processes 1 Gaussian, 1 write/Gaussian
│
├── Prefix Sum: at.cumsum (0.015-0.017ms) MINOR
│     PyTorch GPU scan (segmented parallel prefix sum)
│     O(N_visible): one element per Gaussian
│
└── Pass2: pair materialization (2.00-10.70ms) PRIMARY
      Grid: IDENTICAL to Pass1 (same kernel, different branch)
      ├── enumeration: loop over tile_min→tile_max in Y×X order
      ├── key creation: depth_upper | (tile_id << 16) | iid_enc
      └── value creation: flatten_id = Gaussian index
      Memory: writes 12 bytes/pair (int64 key + int32 value)
      O(N_isect): writes are proportional to number of pairs
      Write traffic ratio Pass2:Pass1 = 336×-4754×
```

**Primary cost:** Pass2 pair materialization (98.4-99.8% of intersection gen)  
**Secondary:** Pass1 tile bounds + count (<0.7%)  
**Minor:** Prefix sum (<0.9%)

---

## Scaling Analysis

| Relationship | Finding | Evidence |
|:-------------|:--------|:---------|
| `T_pass1` vs `N_visible` | **approx. O(N_visible)** | Grid = `ceil(N_visible/256)`. Room: 63 blk, Garden: 393 blk. |
| `T_prefix` vs `N_visible` | **mixed** | Room (16k): 0.015ms, Garden (100k): 0.017ms. 6× more elements, 1.13× time. |
| `T_pass2` vs `N_isect` | **approx. O(N_isect)** | Room (25.5M): 2.38M pairs/ms, Bicycle (18.6M): 2.81M/ms, Garden (11.3M): 5.63M/ms. Throughput varies with data size. |

---

## Necessity Analysis (4-Level)

### Level 1: Does the renderer need tile-Gaussian relationships?

**YES.** Without tile-Gaussian mapping, each tile would test ALL visible Gaussians → `O(N_tiles × N_visible)` per frame. Tile-based rendering fundamentally requires this.

### Level 2: Is explicit `[key, value] × N_isect` storage required?

**PARTIALLY.** The current implementation stores explicit pairs because:
- CUB radix sort requires contiguous key/value arrays as input
- Offset encoding reads contiguous sorted keys for tile range computation

**Potential alternatives:**
- Index-based indirection (don't store keys, just permute Gaussian indices)
- Tile-local sorting with smaller materialization
- Hierarchical ordering

### Level 3: Must all pairs be materialized before sorting?

**NOT NECESSARILY.** This is pipeline organization, not a mathematical requirement. Two alternatives exist in prior art:
- **Splatshop**: depth-sort Gaussians first, then generate tiles → sort O(N_visible) instead of O(N_isect)
- **GS-TG**: tile group shared sorting — reduces sort scope

### Level 4: Is `count → scan → materialize → sort` a mathematical necessity?

**PIPELINE ORGANIZATION.** The fundamental requirement is: **ordered per-tile Gaussian list → rasterizer.** The current `count → scan → materialize → sort` is one valid GPU implementation.

---

## Prior-Art Comparison Matrix

| Stage | Current gsplat | GSCore | Speedy-Splat | Splatshop | GS-TG | TileGS | Neo |
|:------|:---------------|:-------|:-------------|:----------|:------|:-------|:----|
| **Projection** | DIFFERENT | OVERLAP | OVERLAP | OVERLAP | OVERLAP | OVERLAP | OVERLAP |
| **Tile bounds** | DIFFERENT | PARTIAL OVERLAP | PARTIAL OVERLAP | PARTIAL OVERLAP | OVERLAP | OVERLAP | OVERLAP |
| **Shape-aware intersection** | DIFFERENT | DIFFERENT | DIFFERENT | PARTIAL OVERLAP | DIFFERENT | DIFFERENT | DIFFERENT |
| **Intersection count** | DIFFERENT | OVERLAP | OVERLAP | OVERLAP | OVERLAP | OVERLAP | OVERLAP |
| **Pair materialization** | DIFFERENT | PARTIAL OVERLAP | PARTIAL OVERLAP | DIFFERENT | NOT VERIFIABLE | NOT VERIFIABLE | NOT VERIFIABLE |
| **Prefix sum** | DIFFERENT | OVERLAP | OVERLAP | OVERLAP | OVERLAP | OVERLAP | OVERLAP |
| **Sorting** | DIFFERENT | PARTIAL OVERLAP | OVERLAP | DIFFERENT | PARTIAL OVERLAP | DIFFERENT | DIFFERENT |
| **Tile grouping** | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT | PARTIAL OVERLAP | DIFFERENT | DIFFERENT |
| **Temporal reuse** | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT |
| **Rasterization org.** | DIFFERENT | PARTIAL OVERLAP | DIFFERENT | DIFFERENT | PARTIAL OVERLAP | PARTIAL OVERLAP | DIFFERENT |

**Key:**
- `DIFFERENT` — fundamentally different mechanism or not present
- `PARTIAL OVERLAP` — shares some aspects but differs in key details
- `OVERLAP` — same/similar mechanism
- `NOT VERIFIABLE FROM LOCAL SOURCE` — cannot confirm from available source code

---

## Pipeline Order Comparison

| System | Pipeline Order | Key Difference |
|:-------|:---------------|:---------------|
| **Current gsplat** | projection → count → prefix sum → materialize → sort → offset → rasterize | Baseline |
| **GSCore** | projection → shape-aware test → count → prefix sum → materialize → **hierarchical sort** → rasterize (subtile skip) | Shape-aware reduces count; hierarchical sort |
| **Speedy-Splat** | projection → **SnugBox** → **AccuTile** → count → prefix sum → materialize (fewer) → sort → rasterize | Tighter bounds + exact intersection → fewer pairs |
| **Splatshop** | projection → visible splats → **depth sort** → create fragments → tile-ID sort → rasterize | **DIFFERENT ORDER:** depth sort BEFORE fragment creation |
| **GS-TG** | projection → count → prefix sum → materialize → **tile-group sort** → rasterize (original tile size) | Groups tiles for shared sorting |
| **TileGS** | projection → count → prefix sum → materialize → sort → rasterize (**depth-local traversal**) | Changes raster traversal, NOT intersection |
| **Neo** | projection → count → prefix sum → materialize → **cross-frame reorder** → rasterize | Reuses previous sort order; no current-frame sort |

### Splatshop is the Most Different

Splatshop inverts the pipeline order:
```
Current:   materialize N_isect pairs → sort (combined depth|tile_id key)
Splatshop: depth sort N_visible → create fragments → tile-ID sort
```

This means Splatshop sorts **O(N_visible)** Gaussians (millions) instead of **O(N_isect)** pairs (tens of millions). However, Splatshop still materializes fragments — it does not eliminate the O(N_isect) materialization cost.

---

## Data Volume Summary

| Scene | N_total | N_visible | N_isect | Pass1 Writes | Pass2 Writes | **Total** |
|:------|:-------:|:---------:|:-------:|:------------:|:------------:|:---------:|
| room | 1,105,873 | 16,076 | 25,475,247 | 0.06 MB | 291.5 MB | **291.6 MB** |
| bicycle | 2,589,484 | 13,654 | 18,550,853 | 0.05 MB | 212.3 MB | **212.4 MB** |
| garden | 874,019 | 100,582 | 11,257,252 | 0.38 MB | 128.8 MB | **129.2 MB** |

Pass2 writes **336×-4754×** more data than Pass1.

---

## Research Conclusions

### Q1: Current pipeline fully understood?
**CONFIRMED.** Complete source-level trace from projection through rasterization. Every kernel, tensor, memory layout, and encoding scheme audited.

### Q2: Where is the primary cost?
**Pass2 pair materialization** (98.5-99.9% of intersection generation time). Intersection generation overall = **40-73% of forward pass** (PRIMARY bottleneck). CUB radix sort = **21-32%** (SECONDARY). Pass1 and prefix sum are negligible (<1%).

### Q3: What is implementation-specific?
1. Conservative radius-based AABB (no conic/ellipse test)
2. Per-Gaussian tile enumeration in Pass2 (Gaussian→tile direction)
3. Combined 48-50 bit `depth|tile_id|image_id` sort key
4. Materialize-then-sort pipeline order
5. CUB radix sort (6-7 passes per sort)
6. `at::cumsum` for prefix sum (CPU sync point via `.item()`)

### Q4: What are fundamental requirements?
- Tile-Gaussian mapping (for tile-based rasterization efficiency)
- Depth ordering per tile (for correct alpha compositing)
- Per-tile sorted Gaussian list (rasterizer input format)

The `count → scan → materialize → sort → rasterize` flow is an implementation of these requirements, not the requirement itself.

### Q5: Prior-art overlap summary
- **GSCore:** Shape-aware intersection + hierarchical sort → REDUCES PAIR COUNT (addresses Pass2 volume indirectly)
- **Speedy-Splat:** SnugBox + AccuTile → REDUCES PAIR COUNT (addresses Pass2 volume directly)
- **Splatshop:** Depth sort before fragments → CHANGES PIPELINE ORDER (avoids O(N_isect) key sort, but not O(N_isect) materialization)
- **GS-TG:** Tile-group shared sort → CHANGES SORT SCOPE
- **TileGS:** Depth-local raster traversal → CHANGES RASTERIZATION, NOT INTERSECTION
- **Neo:** Cross-frame ordering reuse → CHANGES SORT, NOT INTERSECTION

### Q6: Largest open research question
**Can the `count → materialize → sort` pipeline order be changed to reduce the O(N_isect) memory + compute cost of Pass2?** Splatshop partially addresses this (depth-sort-before-fragments), but no prior work has eliminated the O(N_isect) materialization cost entirely. Pass2 accounts for 98.5-99.9% of intersection generation time and writes 128-291 MB per scene.

---

## Research Questions

### RQ1: Can explicit N_isect × 12B pair materialization (Pass2) be reduced/eliminated?

**Source:** Pass2 = 2.00-10.70ms, 98.5-99.9% of intersection generation. Writes 128-291 MB per scene.

**Covered by prior art:** PARTIALLY. Splatshop demonstrates depth-sort-before-fragments, changing the pipeline order but not eliminating materialization. GSCore and Speedy-Splat reduce the count via tighter bounds, but the O(N_isect) materialization remains.

### RQ2: Can 48-50 bit sort key be decomposed to reduce CUB pass count?

**Source:** CUB sort = 21-32% of forward pass, 6-7 CUB passes (end_bit=48-50). Each additional pass adds ~0.4-0.5ms for room-scale workloads.

**Covered by prior art:** PARTIALLY. GS-TG (tile grouping) and GSCore (hierarchical sort) change sort scope, but the fundamental question of key width vs CUB pass count has not been independently analyzed.

### RQ3: Can intersection generation and rasterization be more tightly coupled?

**Source:** Current pipeline materializes all pairs → sorts all → consumes via offsets. The sorted key is immediately discarded after offset encoding. A direct tile-to-Gaussian scheduling mechanism could potentially skip the full intermediate representation.

**Covered by prior art:** PARTIALLY. TileGS optimizes rasterizer traversal but does not modify intersection generation. No work has unified intersection generation and rasterization scheduling.

---

## Output Files

| File | Status |
|:-----|:-------|
| `reports/phase-a100/intersection_pipeline_prior_art_audit.md` | ✅ This report |
| `results/phase-a100/intersection_pipeline_prior_art_audit.json` | ✅ Full structured data |
| `results/phase-a100/split_intersect_timing.json` | ✅ Split timing data |
