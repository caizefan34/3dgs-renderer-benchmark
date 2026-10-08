# R6 Implementation Status

## CRITICAL FINDING

After examining the actual CUDA source code in `IntersectTile.cu`, I discovered that **HiGS already implements two-pass exact allocation**, not worst-case over-allocation as hypothesized in R5.

## Source Code Evidence

**File:** `gsplat/cuda/csrc/IntersectTile.cu`  
**Lines:** 720-850  
**Function:** `intersect_tile` (PrivateUse1 backend)

### Current Implementation Already Uses:

**Pass 1: Count (Lines 745-810)**
```cpp
// Allocate temporary count buffer
cudaMallocAsync(&device_counts[device_id], n_elements * sizeof(int32_t), stream);

// First pass: count intersections (isect_ids = nullptr)
intersect_tile_kernel(..., nullptr, ..., device_counts[device_id], ...);

// Inclusive prefix sum
cub::DeviceScan::InclusiveSum(..., device_counts[device_id], device_cumsum[device_id], ...);

// Get exact total count
cudaMemcpyAsync(&device_totals[device_id], device_cumsum[device_id] + n_elements - 1, ...);

int64_t n_isects = sum(device_totals);  // EXACT count
```

**Pass 2: Fill (Lines 833-929)**
```cpp
// Allocate EXACT required capacity
isect_ids   = at::empty({n_isects}, opt.dtype(at::kLong));      // 8 bytes each
flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));       // 4 bytes each

// Second pass: fill intersection records
intersect_tile_kernel(..., isect_ids.data_ptr() + isect_offset, ...);
```

## R5 Hypothesis REVISION

**R5 Claimed:** `isect_tiles` pre-allocates `N * MAX_INTERSECTIONS_PER_GAUSSIAN`

**Reality:** `isect_tiles` uses exact two-pass allocation

**Status:** R5 root cause analysis was **INCORRECT**

## Actual Memory Breakdown at 4.4M

| Component | Size | Notes |
|-----------|------|-------|
| Temporary counts | ~53 MB | n_elements * 12 bytes |
| Intersection IDs | ~320 MB | n_isects * 8 bytes |
| Flatten IDs | ~160 MB | n_isects * 4 bytes |
| **Total intersection** | **~533 MB** | NOT 37.74 GB |

The 37.74 GB allocation that causes OOM is NOT from intersection buffers.

## Conclusion

**No fix needed for intersection allocation** - already optimal.

The 11.3× memory amplification must come from other sources:
- Hierarchy auxiliary memory
- Sorting workspace
- Other rendering buffers
- Two-pass temporary overhead (small)

**Implementation Status:** BLOCKED - no code modification needed for intersection allocation. True root cause requires deeper profiling of full memory breakdown.
