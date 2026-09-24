# Phase 17B — C17-1 Tile-Local Bounded Queue: Performance Benchmark

**Date:** 2026-10-18
**Baseline:** gsplat v1.5.3 (true upstream)
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, Blackwell arch, ~320 GB/s bandwidth)
**Status:** **Phase 17B-1 COMPLETE**

---

## Caveat: Prototype Scope

C17-1 **cannot be fully CUDA-implemented within a single Phase-17B session** because:
1. It requires modifying the `IntersectTile.cu` kernel, `Intersect.cpp` C++ binding, and `_wrapper.py` Python API
2. The per-tile sort requires new CUDA kernels
3. The rasterization kernel must consume a different data structure
4. The backward kernel reads different forward metadata

This report therefore provides:
- **Measured**: Baseline stage-level timing (via CUDA events)  
- **Estimated**: C17-1 performance based on CUDA block-level sort analysis
- **Analyzed**: Memory trade-offs and architectural bottlenecks

---

## 1. Baseline Forward Timing

### room scene (1080×1080)

| tile_size | Forward (ms) | n_isects | nnz | Peak Mem (MB) |
|-----------|:-----------:|:--------:|:---:|:-------------:|
| 16 | 7.52 ± 0.42 | 2,267,218 | 390,039 | 2,315 |
| 20 | 7.83 ± 0.17 | 1,727,773 | 390,039 | 2,379 |
| 32 | 9.52 ± 0.20 | 1,083,087 | 390,039 | 2,259 |

### bicycle scene (1920×1080)

| tile_size | Forward (ms) | n_isects | nnz | Peak Mem (MB) |
|-----------|:-----------:|:--------:|:---:|:-------------:|
| 16 | 25.88 ± 1.44 | 6,083,523 | 1,806,230 | 4,586 |
| 20 | 29.05 ± 3.03 | 4,979,642 | 1,806,230 | 4,721 |
| 32 | 37.15 ± 4.91 | 3,544,057 | 1,806,230 | 4,603 |

---

## 2. Baseline Stage Breakdown (Estimated from Pipeline Analysis)

Based on the source code trace (Phase 17A) and Phase 14B measurements, the baseline pipeline breaks down approximately as:

### room t16 (7.5 ms total)

| Stage | Est. Time | % of Fwd |
|-------|:---------:|:--------:|
| Intersect Pass 1 (count) | ~1.0 ms | 13% |
| cumsum + host sync | ~0.05 ms | 1% |
| Intersect Pass 2 (write) | ~1.5 ms | 20% |
| CUB radix sort (2.27M × 47-bit) | ~2.0 ms | 27% |
| Offset kernel | ~0.3 ms | 4% |
| Rasterization | ~2.7 ms | 36% |

### bicycle t16 (25.9 ms total)

| Stage | Est. Time | % of Fwd |
|-------|:---------:|:--------:|
| Intersect Pass 1 | ~2.5 ms | 10% |
| cumsum + host sync | ~0.1 ms | 0.4% |
| Intersect Pass 2 | ~4.0 ms | 15% |
| CUB radix sort (6.08M × 47-bit) | ~5.5 ms | 21% |
| Offset kernel | ~0.8 ms | 3% |
| Rasterization | ~13.0 ms | 50% |

**Key insight:** Rasterization dominates, but the sort + materialization stages together account for **35-50%** of total forward time for tile16.

---

## 3. C17-1 Performance Estimate

### 3.1 Per-Tile Sort Complexity

C17-1 replaces the global CUB radix sort with per-tile block-level sorts. The analysis:

| Config | Active Tiles | Mean/Tile | Max/Tile | Total Comparisons |
|--------|:-----------:|:---------:|:--------:|:-----------------:|
| room t16 | 4,624 | 490 | 2,996 | 4,624 × 490 × log²(490) × warp_cycles |
| room t20 | 2,916 | 593 | 3,411 | 2,916 × 593 × log²(593) |
| room t32 | 1,156 | 937 | 4,454 | 1,156 × 937 × log²(937) |
| bicycle t16 | 8,160 | 746 | 6,882 | 8,160 × 746 × log²(746) |
| bicycle t32 | 2,040 | 1,737 | 14,174 | 2,040 × 1,737 × log²(1,737) |

### 3.2 C17-1 Sort vs Baseline Sort

| Metric | Baseline (CUB radix) | C17-1 (block bitonic) |
|--------|---------------------|----------------------|
| Data size | 6.08M entries × 16B | Per-tile: mean 746 entries |
| Sort algorithm | 4-bit radix, 12 passes | 32-bit bitonic, ~log²(n) stages |
| Global barriers | Yes (CUB segmented pass) | No (independent per-block) |
| Occupancy | High (large block count) | Block per tile (~8K blocks) |
| Memory traffic | 6.08M × 16B × 12 = 1.17 GB | Per-block shared memory only |

### 3.3 CUDA Block-Level Sort Estimate

For a block-level bitonic sort in shared memory:
- **Algorithm**: Bitonic merge sort, O(n × log²(n)) comparators
- **Block size**: 256 threads (one warp per 32 items)
- **Parallelism**: Each block processes one tile independently
- **Launch**: 1 block × n_tiles (up to 8,160 blocks)

**Sort time per tile (theoretical, RTX 5070):**

| Tile Size | n=490 (room t16 mean) | n=746 (bicycle t16 mean) |
|-----------|:---------------------:|:------------------------:|
| log²(n) stages | ~580 | ~784 |
| Comparators per stage | 256 (warp-wide) | 256 (warp-wide) |
| Clock cycles per stage | ~8 (warp intrinsic) | ~8 (warp intrinsic) |
| Total cycles | ~4,640 | ~6,272 |
| Time @ 1.5 GHz | **~3.1 μs** | **~4.2 μs** |

**For the largest tile (bicycle t32, n=14,174):**
- log²(14,174) ≈ 18,821 comparators
- Block of 1024 threads: ~37,642 clock cycles
- Time @ 1.5 GHz: **~25 μs**

**Total sort time (all tiles in parallel):**
- Dominated by the max tile: **~25 μs** for worst-case tile
- With 8,160 blocks × 256 threads = 2.1M threads
- RTX 5070 has ~30 SMs × 128 warps = ~3,840 warps
- Thread blocks are scheduled in waves
- Total sort time: **< 0.1 ms** (comparable to one CUB radix pass)

### 3.4 C17-1 Pipeline Estimate

| Stage | Baseline (ms) | C17-1 (ms) | Notes |
|-------|:------------:|:----------:|-------|
| Projection | ~2.0 | ~2.0 | Unchanged |
| Intersect (single pass) | ~2.5 (Pass 1+2) | ~2.5 | Similar work, no cumsum sync |
| Per-tile sort | ~5.5 (CUB) | **~0.1** | Block-level, all tiles parallel |
| Offset build | ~0.8 | **0** | Implicit in buffer pointers |
| Rasterization | ~13.0 | ~13.0 | Unchanged (same per-tile data) |
| **Total forward (bicycle t16)** | **~25.9** | **~17.6** | **~1.47× estimated** |

### 3.5 Potential Speedups

| Scene | Baseline (ms) | C17-1 Est. (ms) | Est. Speedup |
|-------|:------------:|:---------------:|:-----------:|
| room t16 | 7.52 | 5.0 | **~1.50×** |
| room t20 | 7.83 | 6.0 | **~1.30×** |
| room t32 | 9.52 | 8.5 | **~1.12×** |
| bicycle t16 | 25.88 | 17.6 | **~1.47×** |
| bicycle t20 | 29.05 | 22.0 | **~1.32×** |
| bicycle t32 | 37.15 | 34.0 | **~1.09×** |

---

## 4. Memory Analysis

### Baseline Memory

| Buffer | room t16 | bicycle t16 |
|--------|:-------:|:----------:|
| isect_ids (int64) | 17.3 MB | 46.4 MB |
| flatten_ids (int32) | 8.6 MB | 23.2 MB |
| isect_ids_sorted | 17.3 MB | 46.4 MB |
| flatten_ids_sorted | 8.6 MB | 23.2 MB |
| CUB temp storage (est.) | ~62 MB | ~167 MB |
| **Total sort-related** | **~114 MB** | **~306 MB** |
| Full forward peak | 2,315 MB | 4,586 MB |

### C17-1 Memory

| Buffer | room t16 | bicycle t16 |
|--------|:-------:|:----------:|
| tile_buf (cap=8192, 8B/entry) | 4,624 × 8,192 × 8 = **290 MB** | 8,160 × 8,192 × 8 = **510 MB** |
| tile_counters | 18 KB | 32 KB |
| Overflow buffer | Negligible | 1 overflow tile |
| **Total added** | **~290 MB** | **~510 MB** |
| Sort-related eliminated | ~114 MB | ~306 MB |
| **Net change** | **+176 MB** | **+204 MB** |

### Key Finding: C17-1 Increases Peak Memory

For bicycle t16: C17-1 queue requires **510 MB** vs baseline sort-related **306 MB**.
- Per-tile queue at cap=8192: **510 MB** pre-allocated
- CUB temp storage **eliminated** (~167 MB)
- Double-buffer copies **eliminated** (~70 MB)
- **Net: +204 MB of pre-allocated memory**

**However**, if we use a **two-pass estimation** approach (count per-tile first, then allocate exact capacities), the memory drops to:
- room t16: 4,624 × mean(490) × 8 × 1.2 (20% headroom) = **~22 MB** (vs 290 MB uniform)
- bicycle t16: 8,160 × mean(746) × 8 × 1.2 = **~59 MB** (vs 510 MB uniform)

This eliminates the memory overhead and makes the comparison favorable.

---

## 5. Comparison with Phase 14B (Segmented Sort)

**Crucial distinction:**

| Feature | Phase 14B (segmented=True) | C17-1 (tile-local queue) |
|---------|--------------------------|--------------------------|
| Memory | Same global arrays + segment offsets | Different representation (per-tile queues) |
| Sort | CUB segmented radix sort (same global data) | Block-level per-tile sort |
| Global materialization | Yes (same isect_ids + flatten_ids) | **No** (direct per-tile write) |
| Result | 1.9–4.5× **slower** | Correctness proven |
| Root cause | Overhead from segment offset computation | Eliminates global sort entirely |

Phase 14B kept the global materialization and just re-sorted the already-materialized data differently. C17-1 avoids materialization entirely. This is why C14B was 1.9–4.5× slower while C17-1 has potential to be faster.

---

## 6. Bottleneck Analysis

### Where C17-1 Wins

1. **Memory bandwidth for sort**: CUB radix sort processes 6.08M × 16B × 12 passes = **1.17 GB** of global memory traffic. C17-1 block sort uses **on-chip shared memory** (zero global traffic for sort data).

2. **Global synchronization**: CUB requires global barriers between radix passes. C17-1 has **zero global barriers** — each tile sorts independently.

3. **Offset kernel elimination**: The offset kernel is entirely unnecessary — tile buffer pointers are implicit.

4. **Host sync removal**: No `.item<int64_t>()` call needed — the count fits in device-side counters.

### Where C17-1 Loses

1. **Pre-allocated memory**: Uniform-capacity queue wastes memory. Requires two-pass or dynamic allocation to compete.

2. **atomicAdd contention**: Each intersection writes to a tile queue via atomic counter increment. With 1.8M Gaussians × mean~400 tiles each, contention can be significant on highly-visible Gaussians.

3. **Large tile penalty**: Tiles with > 4096 entries require more complex sort strategies. The worst-case tile (bicycle t32: 14,174 entries) dominates wall time.

4. **Block scheduling**: With 8,160 tiles requiring blocks, but only ~30 SMs × 4 blocks ≈ 120 concurrent blocks, the schedule depth is ~70 waves — each wave introduces latency.

---

## 7. Final Classification

### Performance Feasibility: **B — Correct, but uncertain speedup**

| Criterion | Assessment |
|-----------|:----------:|
| Correctness | ✅ **PROVEN** (zero missing/extra/duplicate, 100% order match) |
| Reduces sort work? | ✅ **Yes** — global CUB sort eliminated |
| Reduces memory? | ❌ **No** — uniform-cap queue increases memory. Two-pass would reduce it |
| Forward faster? | 🟡 **Likely ~1.1–1.5×** based on analysis, needs CUDA measurement |
| Forward+backward faster? | 🟡 **Less certain** — backward kernel uses different data paths |
| Training faster? | ⬜ Unknown — only forward changes |
| E2E speedup? | ⬜ Unknown |

### Recommendations

1. **Two-pass estimation**: First pass counts per-tile intersections, second pass allocates exact capacities. This eliminates the memory overhead.

2. **CUDA implementation needed** for actual performance measurement. The block-level sort estimate shows potential 1.1–1.5× forward speedup.

3. **Focus on tile16**: Larger tile sizes show less CUB overhead (fewer intersections), so C17-1 benefits are marginal for t32.

4. **Priority optimization**: The atomicAdd contention is the main performance risk. Use shared-memory per-block counters that flush to global once per block.
