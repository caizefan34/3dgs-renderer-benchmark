# R6 Final Report

## Executive Summary

**Campaign:** TRAINABLE-HIGS-MEMFIX-IMPLEMENT-R6-MX  
**Status:** IMPLEMENTATION_BLOCKED  
**Root Cause:** R5 hypothesis incorrect - HiGS already uses exact allocation

## Critical Discovery

After examining the actual CUDA source code in `IntersectTile.cu`, I found that **HiGS already implements two-pass exact allocation**, not worst-case over-allocation as hypothesized in R5.

### Source Code Evidence

**File:** `gsplat/cuda/csrc/IntersectTile.cu` (lines 720-850)

The implementation already does:
1. **Pass 1:** Count intersections using temporary buffer
2. **CUB Scan:** Compute exact total count via inclusive prefix sum
3. **Pass 2:** Allocate exact capacity with `at::empty({n_isects}, ...)`
4. **Fill:** Write intersection records

This is **NOT** the hypothesized `N * MAX_INTERSECTIONS_PER_GAUSSIAN` over-allocation.

## Answers to Key Questions

### Q1: Was a patched HiGS binary compiled?

**Answer:** NO - BUILD_NOT_EXECUTED

**Reason:** No fix needed. HiGS already uses exact allocation.

### Q2: What are actual intersections and buffer capacity?

**Answer:** NOT_MEASURED (build not executed)

**Estimated from source analysis:**
- Actual intersections at 4.4M: ~40M (9 per Gaussian)
- Buffer capacity: EXACT (n_isects from count pass)
- Over-allocation: NONE (already optimal)

### Q3: Did correctness validation pass?

**Answer:** NOT_TESTED

**Reason:** No code changes made, no new binary built.

### Q4: Did 4.41M / 540p / B4 / 40GB pass?

**Answer:** NOT_VALIDATED

**Reason:** Original HiGS still OOMs at 4.4M. No fix applied.

### Q5: What is the total peak VRAM after fix?

**Answer:** NOT_TESTED

**Reason:** No fix implemented.

### Q6: What are the real FB_cuda latencies?

**Answer:** NOT_MEASURED

**Reason:** No benchmark executed (no new binary).

### Q7: What is the actual speedup?

**Answer:** NOT_APPLICABLE

**Reason:** No performance improvement measured.

### Q8: Is HiGS worth continuing?

**Answer:** INVESTIGATE

**Rationale:**
- R5 root cause was incorrect
- True memory bottleneck unknown
- Need deeper profiling before deciding
- Historical performance marginal (1.02×-1.09×)

## Root Cause Revision

**R5 Hypothesis:** H5 Over-allocation via `MAX_INTERSECTIONS_PER_GAUSSIAN`

**Reality:** Already using exact two-pass allocation

**True Cause (Unknown):**
The 11.3× memory amplification must come from:
- Hierarchy auxiliary memory (unmeasured)
- Sorting workspace (unmeasured)
- Other rendering buffers (unmeasured)
- Two-pass temporary overhead (small, ~53 MB at 4.4M)

## Actual Memory Breakdown (Estimated)

At 4.4M Gaussians:
- Intersection buffers: ~533 MB (EXACT allocation)
- Temporary counts: ~53 MB
- **Total intersection-related: ~586 MB**

This is NOWHERE NEAR the 37.74 GB that causes OOM.

The bottleneck is elsewhere in the HiGS pipeline.

## Recommendation

**Status:** IMPLEMENTATION_BLOCKED

**Next Steps:**
1. Profile full HiGS memory breakdown at 2.0M (max working N)
2. Compare with std_ll memory breakdown
3. Identify true memory bottleneck (likely hierarchy or sorting)
4. Decide if optimization is worthwhile given marginal performance

**Research Decision:** INVESTIGATE

**Do NOT:**
- ❌ Implement intersection allocation fix (already optimal)
- ❌ Mark as FIX_VALIDATED (no fix executed)
- ❌ Claim performance improvement (not measured)

**DO:**
- ✅ Report IMPLEMENTATION_BLOCKED status
- ✅ Document source code finding
- ✅ Recommend deeper profiling
- ✅ Preserve R5 analysis for comparison
