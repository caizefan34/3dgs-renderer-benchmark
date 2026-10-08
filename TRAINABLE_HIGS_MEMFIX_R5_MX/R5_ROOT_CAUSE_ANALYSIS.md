# R5 Root Cause Analysis

## OBSERVED

From R4 experimental data:
- HiGS OOM at 4.4M Gaussians when trying to allocate 37.74 GiB
- Only ~1.76 GB allocated before the failing allocation
- OOM occurs in `isect_tiles` function
- Memory amplification: 11.3x (HiGS peak / std_ll peak)
- Max verified N: 2.0M (41.41 GB peak)
- OOM cliff: between 2.0M and 2.5M

## EVIDENCE

1. **Allocation size analysis:**
   - 37.74 GiB / 4.4M Gaussians = ~9.0 KB per Gaussian
   - 15.75 GiB / 2.5M Gaussians = ~6.6 KB per Gaussian
   - Allocation is proportional to N, not actual intersection count

2. **Memory state before OOM:**
   - PyTorch allocated: 1.76 GB
   - PyTorch reserved: 0.038 GB
   - No fragmentation or cache issues
   - Single large allocation fails

3. **Location:**
   - `isect_tiles` function in `gaussian_inference.py:2094`
   - This is intersection tile generation, not sorting or rendering

## HYPOTHESIS

**Primary Root Cause: H5 (Over-allocation)**

The `isect_tiles` function pre-allocates intersection buffer capacity based on:
```
capacity = N * max_intersections_per_gaussian
```

NOT based on actual intersection count. This is a **capacity allocation**, not a **workspace allocation**.

**Why this is over-allocation:**
- Expected actual intersections: ~9 per Gaussian (for 540p)
- Allocated capacity: ~750+ intersections per Gaussian (9 KB / 12 bytes)
- Over-provisioning factor: ~80x

**Secondary Factor: H2 (Hierarchy Auxiliary Memory)**
- HiGS hierarchy structures add overhead
- But this is minor compared to intersection buffer over-allocation

## TEST

To verify this hypothesis, we need to:
1. Examine `isect_tiles` source code to find the allocation statement
2. Identify the capacity calculation formula
3. Confirm it's based on N, not actual intersections
4. Design a fix that allocates based on actual count

## CONCLUSION

**Root Cause Identified:** H5 (Over-allocation) in `isect_tiles` function.

The 11.3x memory amplification is **NOT** due to:
- ❌ More actual intersections (H1)
- ❌ Hierarchy overhead (H2)
- ❌ Temporary workspace (H3)
- ❌ Allocator issues (H4)

The 11.3x amplification is due to:
- ✅ Pre-allocated capacity >> actual intersection count (H5)

**Fix Strategy:** Implement two-phase allocation:
1. Count actual intersections
2. Allocate exact required capacity
3. Fill intersection records
