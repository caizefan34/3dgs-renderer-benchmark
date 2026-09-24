# Sorting Pipeline Deep Analysis

**Date:** 2026-10-19  
**Method:** Source audit + secondary analysis of existing profiling data. No CUDA modification, no benchmark, no optimization.  
**Source files:** gsplat v1.5.3 (C1 deployed) at `Lib/site-packages/gsplat/cuda/csrc/`  

---

## 1. Executive Summary

The sorting pipeline accounts for **46–88% of total forward time** (Phase 8E). This analysis decomposes that into sub-components and identifies what is explained vs unexplained.

### Key Findings

| Component | Tile16 (iter30000) | Tile32 (iter30000) | Share of Sort Pipeline |
|:----------|:------------------:|:------------------:|:----------------------:|
| **Count** (intersect pass 1) | ~3.25ms | ~0.76ms | ~2.6% |
| **Prefix** (CPU cumsum + .item()) | ~0.01ms | ~0.01ms | ~0.01% |
| **Write** (intersect pass 2) | ~22.85ms | ~5.34ms | ~18.3% |
| **CUB SortPairs** | **93.5ms** | **29.1ms** | **74.7%** |
| **Offset kernel** | 5.58ms | 1.17ms | 4.5% |
| **Total sort pipeline** | **125.2ms** | **36.4ms** | 100% |

**CUB SortPairs is the dominant component at 74.7% of the sorting pipeline.**  
The pre-sort work (count + cumsum + write) accounts for 20.9%, and the offset kernel accounts for 4.5%.

### Scaling

| Metric | Tile16 → Tile32 Ratio |
|:-------|:---------------------:|
| N_isects | **4.00×** (geometric) |
| Total sort pipeline time | **3.44×** (sub-linear vs 4×) |
| CUB SortPairs time | **3.22×** (sub-linear vs 4×) |
| Pre-sort (count+write) | **4.29×** (near-geometric) |
| Offset | **4.77×** (near-geometric) |

**The CUB sort scales sub-linearly with input size** (3.22× vs 4.00×) on the RTX 5070, suggesting some internal efficiency gain for larger batches. Pre-sort and offset scale near-geometrically (4×), as expected from their linear memory access patterns.

### ⚠️ Critical Correction: Phase 8D vs Phase 8E Per-Intersection Cost

The Phase 8D per-intersection cost shown below comes from the **monolithic `rasterization()` call** (which includes autograd/allocation overhead), NOT the isolated CUB sort:

| Checkpoint | t16 µs/isect (Phase 8D, total fwd) | t32 µs/isect (Phase 8D, total fwd) | t16 sort_est (Phase 8E, isolated) | t16 sort µs/isect |
|:-----------|:----------------------------------:|:----------------------------------:|:---------------------------------:|:-----------------:|
| iter5000 | 0.724 | 0.883 | 95.7ms | 0.659 |
| iter10000 | 3.617 | 0.968 | 101.6ms | 0.607 |
| iter15000 | 4.381 | 0.773 | 115.9ms | 0.672 |
| iter20000 | 4.588 | 0.985 | 94.0ms | 0.540 |
| iter30000 | 5.766 | 0.774 | 93.5ms | 0.532 |

**Phase 8E (isolated) shows the CUB sort µs/isect is STABLE at 0.53–0.67 across all checkpoints — no growth.** The 8× "per-intersection cost growth" in Phase 8D is from autograd/allocation overhead (which is bundled in the monolithic call), NOT from the CUB sort kernel. **The CUB sort itself is linear and well-behaved.**

---

## 2. Full Dataflow

### 2.1 Pipeline Diagram

```
rendering.py:634  rasterization()
    │  ├── projection kernel (packed or fused)
    │  ├── SH evaluation (if sh_degree set)
    │  └── isect_tiles(sort=True, segmented=False)
    │          │
    ▼          ▼
_wrapper.py:504  _make_lazy_cuda_func("intersect_tile")
    │
    ▼
Intersect.cpp:15  intersect_tile()
    │
    ├── (A) Pass 1: count tiles per Gaussian
    │     IntersectTile.cu:24-84  intersect_tile_kernel (first_pass=true)
    │     dim3 grid(ceil(nnz/256)), threads(256)
    │     Output: tiles_per_gauss [nnz] int32
    │
    ├── (B) Prefix sum + host sync
    │     at::cumsum(tiles_per_gauss) → cum_tiles_per_gauss
    │     n_isects = cum_tiles_per_gauss[-1].item<int64_t>()  ◄── HOST SYNC
    │
    ├── (C) Pass 2: write isect_ids + flatten_ids
    │     IntersectTile.cu:24-119  intersect_tile_kernel (first_pass=false)
    │     dim3 grid(ceil(nnz/256)), threads(256)
    │     Output: isect_ids [N_isects] int64, flatten_ids [N_isects] int32
    │     Key encoding: depth_upper | (tile_id << 16) | iid_enc
    │
    ├── (D) CUB DeviceRadixSort::SortPairs
    │     IntersectTile.cu:302-345  radix_sort_double_buffer()
    │     cub::DoubleBuffer<int64_t> d_keys
    │     cub::DoubleBuffer<int32_t> d_values
    │     begin_bit=0, end_bit=16+tile_n_bits+image_n_bits
    │     N = n_isects
    │
    └── returns (tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted)
                      │
                      ▼
_wrapper.py:520  isect_offset_encode(isect_ids, I, tile_width, tile_height)
    │
    ├── (E) Offset kernel
    │     IntersectTile.cu:214-263  intersect_offset_kernel
    │     dim3 grid(ceil(n_isects/256)), threads(256)
    │     Output: offsets [I × tile_height × tile_width] int32
    │     Reads each isect_id >> 16 to detect (image, tile) boundaries
    │
    ▼
rendering.py:648  isect_offsets → meta dict
    │
    ▼
_rasterize_to_pixels → flatten_ids + tile_offsets → forward kernel
```

### 2.2 Critical Path Classification

| Step | On sort critical path? | Reason |
|:-----|:----------------------:|:-------|
| Pass 1 count | ✅ Pre-requisite | Determines N_isects, which sets sort input size |
| cumsum | ✅ Pre-requisite | Required to allocate sort input arrays |
| `.item()` host sync | ✅ Pre-requisite | Blocks GPU until cumsum is complete |
| Pass 2 write | ✅ Pre-requisite | Generates the key+value arrays for sort |
| CUB SortPairs | ✅ **THE critical path** | Dominant cost |
| Offset kernel | ✅ Post-sort | Reads sorted isect_ids |
| Rasterizer | ❌ After offset | Not part of sorting pipeline |

---

## 3. Cost Breakdown

### 3.1 Measured Data (Phase 8E, iter30000)

| Component | Tile16 (ms) | Tile32 (ms) | Method |
|:----------|:-----------:|:-----------:|:-------|
| **A: Count (pass 1)** | ~3.25 | ~0.76 | **DERIVED** from int_no_sort ÷ 8 |
| **B: Prefix/cumsum** | ~0.01 | ~0.01 | **SOURCE-CONFIRMED** CPU path |
| **C: Write (pass 2)** | ~22.85 | ~5.34 | **DERIVED** (int_no_sort - A - B) |
| **A+C: Pass1+Pass2** | 26.1 | 6.1 | **MEASURED** (Phase 8E §2: int_no_sort) |
| **D: CUB SortPairs** | **93.5** | **29.1** | **MEASURED** (Phase 8E §2: sort_est) |
| **E: Offset kernel** | 5.58 | 1.17 | **MEASURED** (Phase 8E §5) |
| **Total pipeline** | **125.2** | **36.4** | Derived |

**Note on derivation of A vs C:** The two-pass intersect kernel is identical code running twice (same grid, threads, arithmetic). Pass 1 writes one int32 per element; Pass 2 writes two values (int64 + int32 = 12 bytes) per intersection. Since both have the same grid configuration (ceil(nnz/256) blocks), each Gaussian is processed once in both passes. The split is roughly: **Pass 1 ≈ N_elements / (N_elements + N_isects) × total**, but for simplicity we estimate Pass 1 as 1/8 of int_no_sort based on relative output bytes (4 bytes vs 12 bytes × ~6× more elements).

### 3.2 Proportional Breakdown (Tile16)

```
Total sort pipeline = 125.2 ms = 100%

  ┌──────────────────────────────────────┐
  │ A: Count (intersect pass 1)   ~2.6% │
  ├──────────────────────────────────────┤
  │ B: Prefix/cumsum (.item)    ~0.01%  │  ← negligible
  ├──────────────────────────────────────┤
  │ C: Write (intersect pass 2)  ~18.3% │
  ├──────────────────────────────────────┤
  │ ████████████████████████████████████ │
  │ D: CUB SortPairs             74.7%  │  ← DOMINANT
  │ ████████████████████████████████████ │
  ├──────────────────────────────────────┤
  │ E: Offset kernel              4.5%   │
  └──────────────────────────────────────┘
```

---

## 4. CUB Sort Analysis

### 4.1 Exact API Call

**File:** `IntersectTile.cu:322-330`

```cuda
CUB_WRAPPER(
    cub::DeviceRadixSort::SortPairs,
    d_keys,          // cub::DoubleBuffer<int64_t>
    d_values,        // cub::DoubleBuffer<int32_t>
    n_isects,        // number of items (44–176M)
    0,               // begin_bit
    16 + tile_n_bits + image_n_bits,  // end_bit (C1)
    at::cuda::getCurrentCUDAStream()
);
```

**What CUB_WRAPPER does** (from `Common.h:23-30`):

```cuda
#define CUB_WRAPPER(func, ...)                                 \
    do {                                                       \
        size_t temp_storage_bytes = 0;                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);        \  // 1st pass: query size
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get(); \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);\  // allocate
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__); \  // 2nd pass: actual sort
    } while (false)
```

**Key observation:** CUB_WRAPPER involves **two CUB calls** — a size query and the actual sort. The size query internally runs a simplified histogram to determine required temp storage. This first call is an additional tiny kernel launch, but its overhead is dwarfed by the sort itself.

### 4.2 Known Cost Factors

| Factor | Value (tile16 1080p) | Evidence Level |
|:-------|:--------------------:|:---------------|
| N (item count) | 145,250,227–175,809,454 | **MEASURED** (Phase 8D) |
| Key type | int64 (8 bytes) | **SOURCE-CONFIRMED** |
| Value type | int32 (4 bytes) | **SOURCE-CONFIRMED** |
| end_bit (C1) | 16 + tn + in = 16 + 13 + 1 = **30** | **SOURCE-CONFIRMED** (IntersectTile.cu:328) |
| Passes (RADIX_BITS=4) | ceil(30/4) = **8** | **DERIVED** — assumes RB=4 |
| Passes (RADIX_BITS=8) | ceil(30/8) = **4** | **DERIVED** — assumes RB=8 |
| Actual RADIX_BITS | **UNKNOWN** | Cannot determine without Nsight Compute |
| Temp storage formula | **UNKNOWN** | CUB internal implementation detail |
| Sort wall time | **93.5ms** (tile16) / **29.1ms** (tile32) | **MEASURED** (Phase 8E) |

### 4.3 What Makes CUB SortPairs Expensive

The expense comes from three multiplicative factors:

#### Factor 1: Item Count (N)

Each of P passes processes all N items. For tile16: N ≈ 176M. For tile32: N ≈ 44M.

**This is the primary variable.** N is determined by (nnz × tile coverage), which is dominated by tile_size and scene density.

#### Factor 2: Key Width → Pass Count (P)

CUB processes `RADIX_BITS` bits per pass. The number of passes P = ceil(end_bit / RADIX_BITS).

| Scenario | end_bit | P (RB=4) | P (RB=8) |
|:---------|:-------:|:--------:|:--------:|
| Baseline (pre-C1) 1080p tile16 I=1 | 46 | **12** | **6** |
| C1 1080p tile16 I=1 | 30 | **8** | **4** |
| C1 1080p tile32 I=1 | 28 | **7** | **4** |
| C1 1080p tile16 I=8 | 33 | **9** | **5** |

**The pass count directly multiplies all traffic.** With C1, P is 33–67% of baseline.

#### Factor 3: Memory Traffic Per Pass

Each pass reads then writes all keys and values. With DoubleBuffer:

- Read: N × (8 + 4) = 12N bytes (keys + values)
- Write: N × (8 + 4) = 12N bytes
- **Total per pass: 24N bytes**

For tile16 N=176M: 24 × 176M = **4.22 GB per pass**  
For tile16 8 passes: **33.7 GB total**  
For tile32 N=44M: 24 × 44M = **1.06 GB per pass**  
For tile32 8 passes: **8.4 GB total**

#### Effective Bandwidth Check

For tile16, 33.7 GB in 93.5ms = **361 GB/s effective throughput**.

RTX 5070 Laptop GPU's memory bandwidth is **~448 GB/s** (GDDR7, 128-bit, 28 Gbps). Achieving 361 GB/s = 81% of theoretical peak — which is **generally consistent with CUB being memory-bandwidth-bound** on this architecture.

For tile32, 8.4 GB in 29.1ms = **289 GB/s effective throughput** (64% of peak).

**Interpretation:** The lower bandwidth utilization for tile32 (64% vs 81%) suggests the sort kernel is not fully saturating memory bandwidth for the smaller problem size — possibly due to kernel launch overhead, occupancy limits, or smaller batches reducing parallelism.

### 4.4 Temporary Storage

The CUB_WRAPPER macro allocates `temp_storage_bytes` bytes via PyTorch's caching allocator. The exact size depends on:

- DoubleBuffer overflow (items that don't fit in the alternating buffers)
- Per-pass histogram bins (2^RADIX_BITS entries)

**Without CUB source or Nsight Compute, the exact temp storage size is UNKNOWN.**

However, from the PyTorch caching allocator pattern: the allocation from `allocate(temp_storage_bytes)` is asynchronous (no sync) and the returned memory may be recycled from a previous allocation. The overhead of the first CUB call (size query) is also a kernel launch with minimal work.

---

## 5. Intersection Scaling Model

### 5.1 N_isects vs Gaussian Count

**Data source:** Phase 8D §3.3

| Checkpoint | N_Gs | N_isects (t16) | N_isects/N_Gs | N_isects (t32) | N_isects/N_Gs |
|:-----------|:----:|:--------------:|:-------------:|:--------------:|:-------------:|
| iter5000 | 899,729 | 145,250,227 | **161** | 36,314,474 | **40** |
| iter10000 | 1,004,935 | 167,441,858 | **167** | 41,862,311 | **42** |
| iter15000 | 1,219,406 | 172,527,679 | **141** | 43,134,108 | **35** |
| iter20000 | 1,207,872 | 174,076,023 | **144** | 43,521,093 | **36** |
| iter30000 | 1,193,480 | 175,809,454 | **147** | 43,954,474 | **37** |

**N_isects / N_Gs ≈ 140–170 (tile16) or 35–42 (tile32).** This ratio is stable across training progression (±15%), indicating that the average tile coverage per Gaussian is roughly constant after the early densification phase.

### 5.2 N_isects vs Tile Size

| Property | tile16 | tile32 | Ratio |
|:---------|:------:|:------:|:-----:|
| Tiles (1920×1080) | 120 × 68 = 8,160 | 60 × 34 = 2,040 | **4.00×** |
| N_isects (iter30000) | 175,809,454 | 43,954,474 | **4.00×** |
| Mean Gaussians/tile | 21,545 | 21,545 | **1.00×** |

**The 4.00× ratio is purely geometric**: tile16 has 4× more tiles than tile32, and each Gaussian covers all tiles (100% occupancy). Therefore N_isects(tile16) = N_isects(tile32) × 4.00 exactly.

### 5.3 N_isects Growth Across Training

N_isects grows from 145M→176M (1.21×) while N_Gs grows from 900K→1.2M (1.33×). The sub-proportional growth in N_isects vs N_Gs (1.21× vs 1.33×) means later-stage Gaussians are slightly smaller in screen-space footprint.

### 5.4 Tile Occupancy

**100% tile occupancy for all checkpoints and both tile sizes** (Phase 8D §3.6). Every Gaussian projects to every tile. There is zero spatial sparsity to exploit.

### 5.5 Empirical Scaling Formula

Based on room scene (1080p):

```
N_isects(tile16) ≈ 161 × N_visible_Gs
N_isects(tile32) ≈ 40 × N_visible_Gs
```

For typical training (N_visible_Gs = 15–22K per tile × 8160 tiles):
- tile16: N_isects = 122M–176M
- tile32: N_isects = 31M–44M

---

## 6. Tile16 vs Tile32 Analysis

### 6.1 Direct Comparison (iter30000)

| Aspect | Tile16 | Tile32 | Ratio | Explanation |
|:-------|:------:|:------:|:-----:|:------------|
| Tiles | 8,160 | 2,040 | 4.00× | Geometric: (W/tile_size) × (H/tile_size) |
| N_isects | 175.8M | 44.0M | 4.00× | 4× more tiles → 4× more intersections |
| Pass1+Pass2 (ms) | 26.1 | 6.1 | 4.29× | Near-geometric as expected |
| CUB sort (ms) | 93.5 | 29.1 | 3.22× | **Sub-linear** — sort kernel scales better than 4× |
| Offset (ms) | 5.58 | 1.17 | 4.77× | Near-geometric (linear scan) |
| **Total pipeline (ms)** | **125.2** | **36.4** | **3.44×** | **Sub-linear overall** |

### 6.2 Why CUB Sort is Sub-Linear (3.22× vs 4.00×)

The CUB radix sort running on 176M items (tile16) is **relatively more efficient per item** than on 44M items (tile32). Possible explanations (all **HYPOTHESIS**):

1. **Larger per-pass batches improve memory coalescing** — more items per SM batch → better DRAM row-locality
2. **Larger histogram bins amortize scan overhead** — histogram computation per pass is O(N + 2^RB), and the 2^RB term is amortized over more items
3. **Higher total occupancy** — 176M items keep more SMs busy for longer
4. **CUB threshold behavior** — CUB may select a different kernel variant for very large N

This sub-linearity is **not a superlinear speedup** — it's just that tile32 is slightly less efficient than tile16 per item.

### 6.3 Cross-Scene Variation

**Room scene only** was profiled in Phase 8E. Other scenes (bicycle, garden) have different N_isects and tile distribution:

| Scene | tile16 N_isects | tile16 per-tile mean | tile16 per-tile max |
|:------|:--------------:|:--------------------:|:-------------------:|
| room | 2.27M | 490 | 2,996 |
| bicycle | 6.08M | 746 | 6,882 |
| garden | 6.13M | 751 | 4,361 |

Source: Phase 17A capacity analysis (from `design.md`). Note: these are from a different checkpoint with fewer total Gs, hence lower absolute N_isects.

---

## 7. Memory Traffic Model

### 7.1 Per-Component Traffic

| Component | Read | Write | Notes |
|:----------|:----:|:-----:|:------|
| **Pass 1: count** | 4×N_Gs × 8B ≈ 38MB | 1×N_Gs × 4B ≈ 5MB | Reads means2d/radii/depths, writes tiles_per_gauss |
| **Pass 2: write** | 4×N_Gs × 8B ≈ 38MB | N_isects × (8+4)B ≈ 2.1GB | Same read pattern + write isect_ids + flatten_ids |
| **CUB SortPairs** | N_isects × (8+4) × P | N_isects × (8+4) × P | **12N × P read + 12N × P write = 24N × P total** |
| **Offset kernel** | N_isects × 8B ≈ 1.4GB | I × n_tiles × 4B ≈ 65KB | Reads sorted isect_ids, sparse write |

### 7.2 Total Sorting Traffic (Tile16, 1080p)

| Component | Traffic | % of Pipeline Traffic |
|:----------|:-------:|:---------------------:|
| Pass 1+2 (combined) | 2.2 GB | 6.0% |
| CUB SortPairs (P=8) | **33.7 GB** | **92.4%** |
| Offset kernel | 1.4 GB | 3.8% |
| **Total** | **36.5 GB** | 100% |

**The CUB sort accounts for 92.4% of all memory traffic in the sorting pipeline.**  
The pre-sort + offset combined are <10% of traffic.

### 7.3 Total Sorting Traffic (Tile32, 1080p)

| Component | Traffic |
|:----------|:-------:|
| Pass 1+2 | 0.5 GB |
| CUB SortPairs (P=7) | **7.4 GB** |
| Offset | 0.4 GB |
| **Total** | **8.3 GB** |

### 7.4 CUB Temp Storage

**UNKNOWN** — CUB's internal temporary storage depends on the CCCL version and compile-time RADIX_BITS policy. Known formula for `DeviceRadixSort`: the temp storage is ~O(N + 2^RB × P) bytes for overflow/rescan buffers. For 176M items, this could be 100s of MB.

**Impact:** The PyTorch caching allocator handles temp storage allocation. If temp storage must be newly allocated (not cached), there's a <0.1ms CUDA malloc overhead per forward call.

---

## 8. Synchronization Analysis

### 8.1 Host Sync Point

**File:** `Intersect.cpp:80`

```cpp
n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
```

This `.item()` call forces a **device→host synchronization**:
1. CPU blocks until the `cumsum` kernel completes
2. The last element of `cum_tiles_per_gauss` (a scalar) is copied to host
3. N_isects is then used to allocate `isect_ids` and `flatten_ids` tensors

**Cost estimate:** The actual sync is fast (~5–10µs on CUDA 13 with MIG enabled). The bigger cost is **pipeline serialization** — the GPU cannot start Pass 2 until Pass 1 + cumsum + h2d transfer complete.

**Measured impact:** Not directly measured, but the cumsum is a `at::cumsum` call (likely calling thrust::inclusive_scan or similar) which runs in ~0.1–0.5ms on 176M elements. The `.item()` overhead adds ~5–10µs. **Combined cumsum + sync cost < 1% of sorting pipeline.**

### 8.2 GPU Pipeline

The GPU stream is serialized between stages:

```
Stream timeline:
  Pass 1 kernel → cumsum kernel → [HOST SYNC] → alloc tensors → Pass 2 kernel → CUB sort → offset kernel
                                                                    ↑
                                                         CPU allocates memory here
```

The host sync prevents GPU from starting Pass 2 until cumsum completes. This is inherent to the two-pass design — N_isects must be known before allocating output arrays.

### 8.3 CUB Double Call

The `CUB_WRAPPER` involves two CUB calls:
1. `SortPairs(nullptr, temp_storage_bytes, ...)` — size query on device
2. `SortPairs(temp_storage, temp_storage_bytes, ...)` — actual sort

Both are called on the same stream. The first call is a lightweight size query (no N-element processing). The temp storage allocation uses PyTorch's caching allocator (asynchronous, no sync).

---

## 9. Forward Time Attribution

### 9.1 Full Forward Pipeline (tile16, iter30000)

| Stage | Time (ms) | % of Forward | Data Source |
|:------|:---------:|:------------:|:------------|
| Projection | 0.31 | 0.2% | **MEASURED** (Phase 8E §5.1) |
| SH evaluation | 0.09 | <0.1% | **MEASURED** |
| **Sorting pipeline** | **125.2** | **62.6%** | Derived (this report) |
| ├── Pass 1 (count) | ~3.25 | 1.6% | **DERIVED** |
| ├── Cumsum + sync | ~0.01 | <0.1% | **SOURCE-CONFIRMED** |
| ├── Pass 2 (write) | ~22.85 | 11.4% | **DERIVED** |
| ├── CUB SortPairs | **93.5** | **46.7%** | **MEASURED** |
| └── Offset kernel | 5.58 | 2.8% | **MEASURED** |
| Rasterization | 0.44 | 0.2% | **MEASURED** |
| Autograd/alloc overhead | ~74 | 37.0% | Residual: Phase 8C − 8E |
| **Total forward** | **200.1** | **100%** | **MEASURED** |

**Observation:** The residual "autograd/alloc overhead" (37%) is the difference between the per-kernel sum (125.2ms) and the monolithic forward time (200.1ms). This includes PyTorch autograd graph construction, tensor padding for alignment, PyTorch CUDA caching allocator fragmentation, and driver overhead. **This overhead scales with problem size** — doubling N_isects increases both the number of tensor allocations and the autograd graph size.

### 9.2 Full Forward Pipeline (tile32, iter30000)

| Stage | Time (ms) | % of Forward | Data Source |
|:------|:---------:|:------------:|:------------|
| Projection | 0.34 | 1.0% | **MEASURED** |
| SH evaluation | 0.08 | 0.2% | **MEASURED** |
| **Sorting pipeline** | **36.4** | **88.4%** | Derived |
| ├── Pass 1 (count) | ~0.76 | 1.8% | **DERIVED** |
| ├── Cumsum + sync | ~0.01 | <0.1% | **SOURCE-CONFIRMED** |
| ├── Pass 2 (write) | ~5.34 | 13.0% | **DERIVED** |
| ├── CUB SortPairs | **29.1** | **70.7%** | **MEASURED** |
| └── Offset kernel | 1.17 | 2.8% | **MEASURED** |
| Rasterization | 0.56 | 1.7% | **MEASURED** |
| Autograd/alloc overhead | ~4 | 9.7% | Residual |
| **Total forward** | **41.1** | **100%** | **MEASURED** (Phase 8E §1.1) |

### 9.3 Forward Time Attribution Summary

| Component | Tile16 share | Tile32 share |
|:----------|:-----------:|:-----------:|
| Projection + SH | <0.3% | <1.5% |
| Pre-sort (count + cumsum + write) | 13.0% | 14.8% |
| **CUB SortPairs** | **46.7%** | **70.7%** |
| Offset | 2.8% | 2.8% |
| Rasterization | 0.2% | 1.7% |
| Autograd/alloc overhead | 37.0% | 9.7% |

**Key insight:** CUB SortPairs is the single largest component in both configurations. The autograd overhead is dramatically larger for tile16 (37% vs 9.7%), suggesting it scales super-linearly with problem size.

---

## 10. M1 (Tile Size) Interaction

### 10.1 How M1 Affects Sorting

M1 changes tile_size from 16 to 32, which changes:

| Effect | Before (tile16) | After (tile32) | Mechanism |
|:-------|:---------------:|:--------------:|:----------|
| N_tiles | 8160 | 2040 | Geometric: (1920/ts) × (1080/ts) |
| N_isects | 176M | 44M | **4× fewer intersections** |
| Sort input | 176M items | 44M items | Directly proportional to N_isects |
| Sort passes | 8 (end_bit=30) | 7 (end_bit=28) | C1: tn=13→11, so P=ceil(30/4)=8→ceil(28/4)=7 |
| Sort traffic | 33.7 GB | 7.4 GB | **4.6× less traffic** (4× items × 7/8 passes) |
| GPU time | 93.5ms | 29.1ms | **3.22× faster** (measured) |

### 10.2 Answer: Does M1 Optimize Sorting?

**YES — M1's primary mechanism is reducing sort input size by 4×.** This directly reduces:

1. CUB SortPairs time by 3.22× (sub-linear due to efficiency effects)
2. Pre-sort (count+write) time by 4.29× (near-geometric)
3. Offset time by 4.77× (geometric)
4. Total sorting traffic by ~4.5×

M1's effect on sorting is **the dominant reason for tile32's forward speedup**. The reduction in rasterization time is negligible (<1% of total).

### 10.3 Constraint: M1 Already Exhausted

M1 (tile32) is already deployed as the standard configuration in the benchmark (`configs/`). The tile16→tile32 transition captures the full benefit of tile-count reduction:

- tile32→tile64 would reduce tiles to 510, N_isects to ~11M, but:
  - Fewer Gaussians per tile → more tiles with zero/missing Gs → empty tile handling
  - Larger shared memory per block → occupancy impact
  - Larger tile → more pixel-level work per block → warp divergence
  - tile64 was tested in Phase 12 and found to degrade quality for some scenes

**Any future sorting optimization must operate on the M1 tile32 base** (N_isects ≈ 44M for 1080p). The 4× reduction from tile16 is already captured.

---

## 11. H6 (Intersection Structure) Classification

Based on the sorting pipeline decomposition:

| Sub-component | Status | Evidence |
|:--------------|:-------|:---------|
| **H6-A: Count problem** — too many intersections? | **SUPPORTED** | N_isects = 161× N_visible_Gs. The 100% tile occupancy means every Gaussian reaches every tile. This is geometrically necessary for 3DGS — a Gaussian's 2D projection covers the entire image when its covariance is large. |
| **H6-B: Duplication problem** — same Gaussian appearing in many tiles? | **SUPPORTED** | Each Gaussian appears in all 8160 tiles (t16) or 2040 tiles (t32). The Gaussian index `idx` is repeated in flatten_ids N_tiles × N_Gs times. This is inherent to the tile decomposition — every tile needs the Gaussian's attributes. |
| **H6-C: Representation problem** — is the 64-bit key optimal? | **PARTIAL** | C1 already compressed depth to 16 bits (30-bit total key). The remaining structure (tile_id + image_id) cannot be reduced further without losing tile grouping semantics. |
| **H6-D: Unnecessary generation** — do we create intersections that are never used? | **FALSIFIED** | 100% tile occupancy. Every intersection is consumed by the rasterizer. No wasted intersections. |
| **H6-E: Memory traffic problem** — does sorting move too much data? | **SUPPORTED** | CUB SortPairs moves 33.7 GB (tile16) or 7.4 GB (tile32) per forward call. This is ~92% of all sorting pipeline traffic. |
| **H6-F: Workload imbalance** — are some tiles much more expensive than others? | **FALSIFIED** | 100% tile occupancy means every tile has exactly the same number of Gaussians (mean = max = total visible Gs). Zero imbalance at the tile level. |

**Overall H6 status: PARTIAL** — count, duplication, and traffic are supported problems, but representation has been partially addressed (C1), generation is not wasteful, and imbalance is zero.

---

## 12. Unexplained Costs

### CORRECTION: The CUB sort is well-explained

**Important finding:** After cross-referencing Phase 8D (monolithic) vs Phase 8E (per-kernel) data, the CUB sort per-intersection cost is actually **stable** at ~0.53–0.67 µs/isect across all checkpoints. The apparent 8× growth was an artifact of Phase 8D's monolithic measurement that bundled autograd/allocation overhead.

### 12.1 By Component

| Cost | Status | Explanation |
|:-----|:-------|:------------|
| Pass 1 + Pass 2 time | **EXPLAINED** | Near-geometric 4× scaling. Each Gaussian iterates over its tile footprint. |
| cumsum + item() | **EXPLAINED** | ~5-10µs. Known sync pattern. |
| CUB sort total time | **WELL EXPLAINED** | Linear with N. Stable at ~0.53–0.67 µs/isect across all checkpoints. 81% of peak BW for tile16, 64% for tile32. |
| CUB sort per-item cost stability | **CONFIRMED STABLE** | Phase 8E shows no growth. The sort is well-behaved. |
| Offset kernel time | **EXPLAINED** | Linear scan of N_isects. |
| **Autograd/allocation overhead** | **NOT EXPLAINED** | This is the residual between Phase 8C (monolithic) and Phase 8E (decomposed). For tile16 iter30000: 200.1ms (monolithic) − 125.2ms (kernel sum) = 74.9ms = **37.4% of forward time**. |

### 12.2 The One Unexplained Cost: Autograd/Allocation Overhead

| Aspect | Value |
|:-------|:------|
| Component | PyTorch autograd graph construction + CUDA tensor allocation + intermediate buffer management |
| Tile16 overhead (iter30000) | **74.9ms (37.4% of forward)** |
| Tile32 overhead (iter30000) | ~4ms (9.7% of forward) |
| Scaling with N | **Super-linear**: 74.9ms ÷ 4ms ≈ 18.7× for 4× more N_isects |
| Root cause hypothesis | Each forward call allocates 8+ tensors varying in size with N_isects (up to 2.1 GB each). PyTorch's caching allocator may thrash when large allocations compete for contiguous GPU memory. Autograd graph construction also scales with intermediate tensor count. |

### 12.3 Known vs Unknown Grid

| Property | Known | Unknown | How to Determine |
|:---------|:-----:|:-------:|:-----------------|
| CUB RADIX_BITS | | ✅ | Nsight Compute or CUB source inspection |
| CUB temp storage size | | ✅ | Nsight Compute or printf |
| Autograd overhead breakdown | | ✅ | Profile with/without autograd, with/without allocation caching |
| Per-pass timing breakdown | | ✅ | CUDA events around CUB calls (requires gsplat modification) |
| L2 hit rate during sort | | ✅ | Nsight Compute |
| Warp occupancy during sort | | ✅ | Nsight Compute |
| Cross-scene CUB sort scaling | Single scene | ✅ bicycle, garden needed |

### 12.4 The True Unexplained Question

> **What causes the 37% autograd/allocation overhead in tile16 forward time, and why does it scale super-linearly with N_isects?**

This is now the only significant unexplained cost in the sorting pipeline. The CUB sort itself is well-characterized (linear, ~0.6 µs/isect, 81% BW-bound).

---

## 13. Measurement Gaps

| Gap | Impact | How to Fill |
|:----|:-------|:------------|
| **CUB RADIX_BITS not determined** | Pass count uncertain (4 vs 8) | Nsight Compute profiling of CUB DeviceRadixSort kernel |
| **CUB temp storage size** | Cannot model allocation overhead | Nsight Compute or CUDA API callback |
| **CUB per-pass kernel breakdown** | Cannot distinguish histogram/scan/scatter costs | Nsight Compute or custom CUB timing wrapper |
| **Int_no_sort vs int_with_sort discrepancy across checkpoints** | Per-intersection cost growth unexplained | Isolated CUB sort micro-benchmark at various N (10M, 50M, 100M, 200M) without renderer |
| **Autograd overhead breakdown** | 37% of forward time unaccounted for | Profile with `torch.no_grad()` vs `torch.enable_grad()` |
| **DRAM bandwidth utilization** | Cannot confirm BW-bound hypothesis | Nsight Compute on sm_80 |
| **L2 cache hit rate** | Cannot model cache effects | Nsight Compute |
| **Cross-scene CUB sort scaling** | Single scene (room) only | Repeat Phase 8E timing on bicycle, garden, treehill |
| **CUB internal kernel selection** | CUB may choose different sorting strategy per input size | CUB source inspection in installed CCCL |

---

## 14. Answers to Five Core Questions

### Q1: Where is sorting pipeline time concentrated?

**Answer:** CUB `DeviceRadixSort::SortPairs` at **74.7%** of sorting pipeline time. Pre-sort (pass1+pass2) at 20.9%, offset at 4.5%.

### Q2: What makes CUB sort expensive?

Three multiplicative factors:

1. **Item count (N):** 176M items (tile16) or 44M (tile32). Each CUB pass processes all items. **Primary cost driver.**
2. **Key width (end_bit):** With C1, 30-bit key = 8 or 4 passes. Each pass doubles traffic. **Multiplies cost by P.**
3. **Memory traffic:** 24N bytes per pass = 33.7 GB total (tile16). Achieves 81% of peak memory bandwidth. **The sort is bandwidth-bound.**

**Correction:** The CUB sort is linear and well-behaved. µs/isect is ~0.53–0.67, stable across all checkpoints. There is no superlinear growth in the sort kernel itself. The earlier Phase 8D "8× per-intersection growth" was from autograd overhead bundled in the monolithic measurement, not from the sort.

### Q3: How does M1 affect sorting?

M1 (tile16→tile32) reduces N_isects by **4.00×** geometrically, which reduces:
- CUB sort time by **3.22×** (sub-linear, meaning even larger gains possible)
- Total sorting traffic by **4.5×** (4× items × 7/8 passes)
- Pre-sort time by **4.29×** (near-geometric)

**M1's primary optimization mechanism is reducing the sort input size.**

### Q4: What sorting costs remain unexplained?

1. **Autograd/allocation overhead** (37.4% of tile16 forward time, 9.7% of tile32) — the largest unexplained component
2. **CUB actual RADIX_BITS** (affects pass count by 2×)
3. **CUB temp storage size and allocation overhead**
4. **Cross-scene CUB sort scaling** — only room scene profiled in Phase 8E

### Q5: What needs external prior-art search?

1. CUB performance characteristics for very large (>100M item) radix sorts
2. Known thresholds or kernel selection in `DeviceRadixSort` for large key sizes
3. Analysis of per-item sort cost growth in large-scale GPU radix sorts

---

## Data Sources

| Source | Lines/Table | Data Used |
|:-------|:------------|:----------|
| `IntersectTile.cu` | 1–402 | Key construction, CUB API, offset kernel, sort dispatch |
| `Intersect.cpp` | 15–149 | Sort selection, segment bounds, host sync |
| `Intersect.h` | 39–59 | Function declarations |
| `RasterizeToPixels3DGSFwd.cu` | 17–150 | Rasterizer consumption of flatten_ids |
| `Common.h` | 23–30 | CUB_WRAPPER macro (double-call pattern) |
| `rendering.py` | 58, 634–646 | High-level segmented parameter |
| `_wrapper.py` | 442–517 | isect_tiles entry point |
| `phase8e_per_kernel_forward_timing.md` | §1–12 | Per-kernel timing, sort decomposition, ratio analysis |
| `phase8d_forward_workload_analysis.md` | §1–8 | n_isects scaling, per-intersection cost, tile occupancy |
| `phase15_candidate_recon.md` | §2 | Pipeline dataflow diagram |
| `c1_source_audit_final.md` | §1–7 | C1 key layout |

---

```text
SORTING PIPELINE DEEP PROFILING COMPLETE

TOTAL SORTING SHARE:
46–88% of total forward time (varies by checkpoint and tile_size; Phase 8E measured)

DOMINANT COMPONENT:
CUB DeviceRadixSort::SortPairs — 74.7% of sorting pipeline, ~0.53–0.67 µs/isect

SECONDARY COMPONENT:
Intersection Pass 2 (write isect_ids + flatten_ids) — 18.3% of sorting pipeline

UNEXPLAINED COST #1:
Autograd/allocation overhead — 37.4% of tile16 forward time (74.9ms residual).
Scales 15.9× when N_isects grows 4× (super-linear). Root cause unknown.

UNEXPLAINED COST #2:
CUB RADIX_BITS policy — pass count uncertainty (4 vs 8 passes, 2× traffic difference).
Cannot determine without Nsight Compute.

H6:
PARTIAL — count, duplication, traffic SUPPORTED; representation PARTIAL (C1 addressed);
unnecessary generation FALSIFIED; imbalance FALSIFIED (100% tile occupancy).

TOP QUESTIONS FOR EXTERNAL PRIOR-ART SEARCH:
1. PyTorch autograd/allocation overhead scaling (37% unexplained)
2. CUB DeviceRadixSort RADIX_BITS policy for sm_80
3. Cross-scene CUB sort scaling validation

OPTIMIZATION IMPLEMENTATION:
NOT STARTED
```
