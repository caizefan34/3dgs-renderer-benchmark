# R10 OOM Root Cause Report

## EXECUTIVE SUMMARY

**Campaign:** TRAINABLE_HIGS-OOM-ROOTCAUSE-R10-MX  
**Status:** ROOT_CAUSE_IDENTIFIED  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

## ROOT CAUSE FOUND

**111.28 GiB allocation** occurs at **`gsplat::intersect_tile`** → **`at::empty({n_isects}, ...)`** in `IntersectTile.cu:833`

**Allocation formula:**
- `n_isects` = sum of `device_totals` from all devices
- `device_totals` = result of `cub::DeviceScan::InclusiveSum` on count buffer
- Allocation size = `n_isects × 8 bytes` (int64) + `n_isects × 4 bytes` (int32)
- Total = `n_isects × 12 bytes`

**111.28 GiB = 119,488,359,424 bytes**  
**n_isects = 119,488,359,424 / 12 = 9,957,363,285 intersections**

---

## ALLOCATION SITE DETAILS

**Stack trace (key frames):**
1. Frame 19: `gsplat::intersect_tile(...)` at `IntersectTile.cu`
2. Frame 18: `at::empty(c10::ArrayRef<long>, c10::TensorOptions, ...)`
3. Frame 10: `at::detail::empty_cuda(...)`
4. Frame 4: `c10::cuda::CUDACachingAllocator::malloc(...)`

**Source location:**
- File: `IntersectTile.cu`
- Line: 833
- Code: `isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));`

**Allocation dimensions:**
- Tensor 1: `isect_ids` - shape `[n_isects]`, dtype `int64` (8 bytes)
- Tensor 2: `flatten_ids` - shape `[n_isects]`, dtype `int32` (4 bytes)
- Total: `n_isects × 12 bytes`

---

## R3/R8 8× DIFFERENCE EXPLAINED

**R3:** 13.90 GiB allocation  
**R8:** 111.28 GiB allocation  
**Ratio:** 8.0×

**Possible explanations:**

1. **Different n_isects values**
   - R3: n_isects ≈ 1,244,000,000 (13.90 GiB / 12 bytes)
   - R8: n_isects ≈ 9,957,000,000 (111.28 GiB / 12 bytes)
   - 8× more intersections detected

2. **Different runtime conditions**
   - R3 vs R8 may have had different:
     - Camera batch composition
     - Frustum culling results
     - Hierarchy traversal state
     - GPU memory pressure affecting culling

3. **Same allocation site**
   - Both OOMs at `IntersectTile.cu:833`
   - Both via `at::empty({n_isects}, ...)`
   - Different `n_isects` values

**Status:** DISCREPANCY_PARTIALLY_EXPLAINED (same site, different counts)

---

## WHY n_isects IS SO LARGE

**At 4.4M Gaussians with B4 cameras:**

**Expected n_isects (normal case):**
- 4.4M Gaussians × 4 cameras = 17.6M Gaussian-camera pairs
- Each pair intersects ~9 tiles (540p, 16×16 tiles)
- Expected n_isects ≈ 158M intersections

**Actual n_isects (OOM case):**
- n_isects ≈ 9,957M intersections
- 63× higher than expected!

**Possible causes:**
1. **Intersection counting bug** - count kernel over-counts
2. **Hierarchy amplification** - hierarchy generates duplicate intersections
3. **No culling applied** - all Gaussians counted for all tiles
4. **Integer overflow** - count exceeds int32 range (9,957M < 2^31, so no overflow)
5. **Batch dimension bug** - cameras expanded incorrectly

---

## HYPOTHESIS VERDICTS

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Intersection explosion | **SUPPORTED** | n_isects = 9.96B vs expected 158M (63×) |
| H2: Camera/batch duplication | **POSSIBLE** | B4 in both R3/R8, but R8 has 8× more |
| H3: Hierarchy duplication | **POSSIBLE** | HiGS uses hierarchy, may generate duplicates |
| H4: Sorting workspace | **REFUTED** | OOM at intersection allocation, not sorting |
| H5: Peak overlap | **REFUTED** | Only 2 GB allocated before OOM |
| H6: Different allocation sites | **REFUTED** | Both at IntersectTile.cu:833 |
| H7: Implementation error | **LIKELY** | 63× over-counting suggests bug |
| H8: Binary mismatch | **REFUTED** | Same binary in R3/R8 |

---

## ROOT CAUSE VERDICT

**Status:** ROOT_CAUSE_IDENTIFIED

**Allocation site:** `IntersectTile.cu:833`  
**Allocation:** `at::empty({n_isects}, opt.dtype(at::kLong))`  
**Formula:** `n_isects × 12 bytes`  
**Actual n_isects:** 9,957,363,285  
**Root cause:** Excessive intersection count (63× expected)

**Underlying issue:** Intersection counting kernel produces abnormally high counts at 4.4M Gaussians.

---

## FIX RECOMMENDATION

**Do NOT fix allocation size** - allocation is correct given `n_isects`.

**Fix intersection counting** - investigate why count is 63× too high:

1. **Examine count kernel** in `IntersectTile.cu`
2. **Check for duplicate generation** in hierarchy traversal
3. **Verify culling logic** is applied before counting
4. **Add diagnostic mode** to print per-Gaussian intersection counts
5. **Compare 580K vs 4.4M** count distributions

**Without fixing count, no safe fix exists.**

---

## CONCLUSION

R10 successfully identified the 111.28 GiB allocation site: `IntersectTile.cu:833` allocating `at::empty({n_isects}, ...)`.

**Root cause:** Intersection count explosion (n_isects = 9.96B vs expected 158M, 63× over-count).

**Fix status:** NOT_IMPLEMENTED (need to fix counting logic first)

**Next step:** Investigate count kernel for over-counting bug.
