# Phase 17B — C17-1 Tile-Local Bounded Queue: Failure Analysis

**Date:** 2026-10-18
**Phase:** 17B (Feasibility ✅ + Performance ⚠️)
**Classification:** **B — Correct, but requires CUDA implementation for speed validation**

---

## 1. What Was Proven

### ✅ Intersection Set Equivalence: 34,800,973 intersections, 0 missing, 0 extra, 0 duplicates

The per-tile local queue architecture **correctly represents** every tile→Gaussian relationship. No data is lost, duplicated, or mis-attributed.

### ✅ Depth Ordering Equivalence: 25,932/25,932 tiles, 100% exact order match

Per-tile local sort by `(depth, gaussian_idx)` produces **identical ordering** to global CUB radix sort.

### ✅ Tile Membership Decoding: 100% accurate

Decoded `tile_id` from `isect_ids` matches `tile_offsets`-based grouping for every intersection.

---

## 2. What Was NOT Proven (Performance)

### Cannot Measure Without CUDA Implementation

The Phase 17B-1 performance ablation **requires modifying gsplat's CUDA kernels** — specifically:
1. **`IntersectTile.cu`**: Replace two-pass global materialization with single-pass per-tile queue writes
2. **New CUDA kernel(s)**: Per-tile block-level sort
3. **`Intersect.cpp`**: New C++ bindings for C17-1 entry points
4. **`_wrapper.py`**: Python API for new intersection/tile-queue path
5. **`RasterizeToPixels3DGSFwd.cu`**: Optional: consume tile buffers directly (avoids flatten_ids indirection)
6. **Backward kernel**: Verify gradients flow through new data path

This is a **multi-file CUDA refactoring** requiring ~500+ lines of new CUDA code across 6 files. A single-prototype session cannot complete this.

---

## 3. Key Findings That Impact Feasibility

### 3.1 Memory Overhead (SIGNIFICANT)

**Uniform per-tile queue allocation wastes memory:**

| Config | Baseline Sort Mem | C17-1 Queue (cap=8192) | Delta |
|--------|:-----------------:|:---------------------:|:-----:|
| room t16 | ~114 MB | **290 MB** | **+176 MB** |
| bicycle t16 | ~306 MB | **510 MB** | **+204 MB** |

On an 8 GB GPU, adding **200+ MB** of pre-allocated queue memory is problematic — especially when training simultaneously holds scene data, gradients, and optimizer states.

**Mitigation available:** Two-pass estimation (count per-tile first, allocate exact capacities) drops bicycle t16 queue memory from 510 MB to ~59 MB. However, this re-introduces the two-pass host sync that C17-1 was supposed to eliminate.

**Recommendation:** Two-pass with device-side prefix sum (thrust::exclusive_scan) avoids host sync but adds complexity.

### 3.2 atomicAdd Contention (MODERATE)

Each intersection does an `atomicAdd` on the tile counter. For a Gaussian covering 400 tiles, this creates **400 atomic operations** — highly contended if multiple Gaussians hit the same tile simultaneously.

On RTX 5070 with ~30 SMs, global atomic throughput is ~2-5 billion/second. With 6M intersections, this is ~1-3 ms of atomic overhead — acceptable but non-trivial.

**Mitigation:** Per-block shared memory counters that flush to global once per block.

### 3.3 Per-Tile Sort for Large Tiles (MODERATE)

Bicycle t32 has tiles with up to **14,174 intersections** — well beyond shared memory capacity (typically 48 KB = 6,144 float32 values). For such tiles:
- The sort must use global memory (slower)
- Or use a multi-stage approach (more complex)
- Or fall back to CUB sort (re-introduces global sort)

**Mitigation:** Capacity check with CUB fallback for tiles > 4096 entries. These large tiles are rare (< 1% of tiles) so the performance impact is bounded.

### 3.4 Block Scheduling Inefficiency (MINOR)

RTX 5070 has ~30 SMs × 4 concurrent blocks/SM ≈ 120 blocks in flight. With 8,160 tiles, the sort launches in **~70 waves** of 120 blocks each. Wave scheduling overhead is small but measurable.

---

## 4. Comparison with Phase 14B (Segmented Sort: 1.9–4.5× Slower)

**C17-1 is architecturally different from Phase 14B:**

| Aspect | Phase 14B (`segmented=True`) | C17-1 (tile-local queue) |
|--------|-----------------------------|--------------------------|
| Global materialization | **Yes** (same isect_ids + flatten_ids) | **No** (per-tile queue directly) |
| Sort data | 6M entries in global memory | Mean 746 entries per tile in shared memory |
| Global memory traffic | 1.17 GB (12 CUB passes) | **Zero** (shared-memory sort) |
| CUB call | Full segmented sort | **Eliminated entirely** |
| Memory for sort | Same as baseline + offset arrays | **Separate per-tile buffers** |

**C17-1 is NOT "segmented sort on already-materialized data"** — it avoids materialization from the start. This is why C17-1 has potential to be faster while Phase 14B was 1.9–4.5× slower.

---

## 5. Is C17-1 Worth Pursuing?

### Yes, WITH modifications

| Barrier | Severity | Solution |
|---------|:--------:|----------|
| Memory overhead (uniform queue) | HIGH | Two-pass estimation OR dynamic allocation |
| CUDA implementation complexity | HIGH | Phased: first pass = queue fill only, second = sort |
| atomicAdd contention | MEDIUM | Per-block shared memory counters |
| Large tiles > shared memory | MEDIUM | CUB fallback path (rare) |
| Block scheduling depth | LOW | Acceptable at ~70 waves |

### Recommended Path Forward

1. **Implement two-pass estimation**: Device-side prefix sum to allocate exact per-tile capacities. This resolves the memory concern entirely.
2. **Single-pass queue fill with per-SM counters**: Warp-level shared memory atomic counters, flushed to global once per block.
3. **Block-level bitonic sort**: Use shared memory for tiles ≤ 4096 entries, CUB fallback for larger tiles.
4. **Direct tile buffer consumption**: Modify rasterization kernel to read from tile buffers instead of flatten_ids.

---

## 6. Final Classification

| Criterion | Result |
|-----------|:------:|
| Correct representation of intersection workload | ✅ **PASS** |
| No data loss (overflow) | ⚠️ **Conditional** — requires overflow strategy |
| Intersection set equivalence | ✅ **PROVEN** |
| Depth ordering equivalence | ✅ **PROVEN** |
| Pixel equivalence | ✅ **Proven by ordering identity** |
| Memory efficiency | ⚠️ **Worse** with uniform queue, **better** with two-pass |
| Forward speed | 🟡 **Estimated 1.1–1.5×** (needs CUDA measurement) |
| Backward compatibility | 🟡 **Likely unaffected** (backward reads forward output, not intermediate) |
| Training speed | 🟡 **Unknown** |
| E2E speedup | 🟡 **Unknown** |

### Bottom Line

**C17-1 is CORRECT but the memory overhead and CUDA complexity make it a Category B/C candidate:**

- **Category B (Renderer-level optimization only)** if the two-pass mitigation resolves memory concerns and CUDA shows 1.1–1.5× speedup
- **Category C (Correct but slower)** if the memory overhead + atomic contention + sort overhead exceed CUB's baseline

The primary value of C17-1 is **not speed** but **architectural simplification** — eliminating the 64-bit key encoding, the double-buffer global sort, and the offset kernel. Whether this translates to real speedup depends on the CUDA implementation quality and the specific GPU architecture.

### Decision for Phase 17C

**DO proceed to C17-2 (Two-Phase Sort) as a simpler fallback** that preserves the global materialization pattern but reduces sort key width. C17-2 requires only modest changes to the existing IntersectTile.cu, making it implementable in a single session. C17-1 should be revisited after C17-2 as a more ambitious follow-up.
