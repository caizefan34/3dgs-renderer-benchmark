# CUB Sort Nsight Systems Timeline Analysis

**Date:** 2026-09-06  
**Status:** Nsight Systems NOT AVAILABLE on A100 server

---

## 1. Tool Availability

| Tool | Available | Reason |
|:----|:---------:|:-------|
| `nsys` (Nsight Systems) | �?| Not installed in PATH, CUDA toolkit, or /opt |
| `ncu` (Nsight Compute) | �?| Not installed |
| `nvprof` (deprecated) | �?| Not available |

---

## 2. Predicted Kernel Timeline (from CUB source analysis)

Since direct profiling is not possible, this report documents the **expected** kernel sequence based on CUB 1.5.10 source code analysis.

### 2.1 Kernel Sequence (room tile16 �?smallest workload, 1 part, 6 passes)

```
CUDA Stream Timeline:
─────────────────────────────────────────────────────────────────────────
[HistogramKernel]  �?reads all 174M keys, 108 blocks × 128 threads
    �?(synchronization via stream ordering �?no explicit sync)
[ExclusiveSumKernel]  �?6 blocks × 256 threads, scans histogram bins
    �?[Onesweep_pass0]  �?bits 0-7
    �?[Onesweep_pass1]  �?bits 8-15
    �?[Onesweep_pass2]  �?bits 16-23
    �?[Onesweep_pass3]  �?bits 24-31
    �?[Onesweep_pass4]  �?bits 32-39
    �?[Onesweep_pass5]  �?bits 40-46 (last pass, 7 bits only)
    �?Return to caller (isect_ids_sorted ready for offset encode)
─────────────────────────────────────────────────────────────────────────
Total kernel launches: 8
```

### 2.2 Kernel Sequence (bicycle tile16 �?largest, 3 parts, 6 passes)

```
CUDA Stream Timeline:
─────────────────────────────────────────────────────────────────────────
[HistogramKernel]  �?reads all 607M keys
[ExclusiveSumKernel]
[Onesweep_pass0_part0]  �?part 0 of 3, bits 0-7
[Onesweep_pass0_part1]  �?part 1 of 3
[Onesweep_pass0_part2]  �?part 2 of 3
  �?d_keys.selector ^= 1, d_values.selector ^= 1
[Onesweep_pass1_part0]  �?bits 8-15
[Onesweep_pass1_part1]
[Onesweep_pass1_part2]
  �?...
[Onesweep_pass2_part0..2]  �?bits 16-23
[Onesweep_pass3_part0..2]  �?bits 24-31
[Onesweep_pass4_part0..2]  �?bits 32-39
[Onesweep_pass5_part0..2]  �?bits 40-46
─────────────────────────────────────────────────────────────────────────
Total kernel launches: 20
```

### 2.3 Expected Overlap

Within each pass's parts:
- Parts are **sequential** (same stream), not overlapping
- Between passes: **sequential** (each pass depends on previous pass's output)

No CUDA graph or multi-stream parallelism is used by CUB's `SortPairs` DoubleBuffer interface.

### 2.4 Synchronization

- All kernels launch on a single CUDA stream (the default stream from `at::cuda::getCurrentCUDAStream()`)
- No explicit `cudaDeviceSynchronize()` between CUB kernels �?stream ordering ensures correctness
- The gsplat caller (`radix_sort_double_buffer()`) does NOT synchronize between sort calls
- The `torch.cuda.Event` in profiling wraps the entire `isect_tiles(sort=True)` call including all CUB kernels

---

## 3. Memory Allocation

Temporary storage is allocated by the `CUB_WRAPPER` macro which calls `cudaMallocAsync` for the `d_temp_storage` if needed. On first call, CUB queries required size (passing `d_temp_storage=NULL`), then the caller allocates.

In gsplat, the `CUB_WRAPPER` macro handles this automatically:
```cpp
#define CUB_WRAPPER(func, ...)                                              \
    size_t temp_size = 0;                                                   \
    func(nullptr, temp_size, __VA_ARGS__);                                  \
    /* ... allocate temp storage via caching allocator ... */               \
    func(ptr, temp_size, __VA_ARGS__);
```

This means **two passes** through CUB dispatch for each sort call:
1. Query (no kernels launched, just size calculation)
2. Execute (kernels launched with actual storage)

Each call also involves a temporary storage allocation from CUDA's caching allocator.

---

## 4. Adjacent Kernel Context

Before `SortPairs`:
```
[intersect pass 1 kernel] �?writes isect_ids (int64_t), flatten_ids (int32_t)
  �?(synchronization for prefix sum on CPU)
[CPU prefix sum]          �?writes cum_tiles_per_gauss (int64_t)
  �?[intersect pass 2 kernel] �?writes isect_ids (int64_t, full sort keys), flatten_ids (int32_t)
  �?[SortPairs]               �?reads isect_ids, flatten_ids
                          �?writes sorted isect_ids, sorted flatten_ids
```

After `SortPairs`:
```
[offset encode kernel]    �?reads sorted isect_ids, writes offsets
[rasterization kernel]    �?reads sorted flatten_ids, offsets, colors, etc.
```

---

## 5. Kernel Duration Estimates (from total sort time ÷ kernel count)

These are **rough estimates** assuming equal kernel duration �?the actual distribution is likely uneven:

| Workload | Sort(ms) | Total Launches | **Avg per kernel(ms)** |
|----------|:--------:|:--------------:|:----------------------:|
| room t16 | 20.6 | 8 | ~2.6 ms |
| room t24 | 9.2 | 8 | ~1.2 ms |
| bicycle t16 | 72.9 | 20 | ~3.6 ms |
| bicycle t24 | 32.6 | 14 | ~2.3 ms |
| garden t16 | 82.5 | 20 | ~4.1 ms |
| garden t24 | 35.1 | 14 | ~2.5 ms |

**Note:** `ncu` is required to measure individual kernel durations accurately.

---

## 6. Conclusion

```
NSIGHT SYSTEMS TIMELINE STATUS:
UNKNOWN �?Nsight Systems not available on A100 server

PREDICTED TIMELINE:
Based on CUB 1.5.10 source (Policy800, ONESWEEP):
  1. HistogramKernel (1 launch, scans all keys)
  2. ExclusiveSumKernel (1 launch, global scan of bin counts)
  3. OnesweepKernel × num_passes × num_parts (e.g., 6 × 3 = 18 for bicycle t16)
  Total: 8-20 kernel launches per sort call

CONFIRMED BY SOURCE ONLY:
  �?Onesweep kernel sequence
  �?Single stream, sequential passes
  �?No overlap between passes
  �?No overlap between parts within a pass
```
