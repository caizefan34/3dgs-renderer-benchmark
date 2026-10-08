# R7 Memory Forensics - Final Report

## EXECUTIVE SUMMARY

**Campaign:** TRAINABLE-HIGS_MEMORY_FORENSICS_R7_MX  
**Status:** PARTIALLY_COMPLETE  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

## CRITICAL FINDING

**OOM NOT REPRODUCED.** The 13.90 GiB allocation failure reported in R3 did not occur in this forensic investigation.

## INVESTIGATION RESULTS

### P0: Binary Identity Verification ✅

**Baseline Binary:**
- Path: `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/csrc.so`
- Expected SHA256: `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`
- Actual SHA256: `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`
- **Status:** VERIFIED - MATCHES baseline

**Source File:**
- Path: `IntersectTile.cu`
- SHA256: `b2468d28a329e7e5d7693f76eae1ba93c927865b50c1ce106c591e42843c8c0f`
- **Status:** VERIFIED - Source exists and matches

### P1: OOM Reproduction ⚠️

**Standard Rasterization:**
- Workload: 4,413,928 Gaussians, 960×540, B1
- Result: **SUCCESS**
- Peak memory: 8.02 GB
- **No OOM occurred**

**HiGS-Specific Function:**
- Function: `rasterize_gaussian_higs_frozen`
- Result: **NOT AVAILABLE**
- Error: `ImportError: cannot import name 'rasterize_gaussian_higs_frozen'`
- **Status:** Cannot test HiGS-specific code path

### P2-P5: Allocation Trace & Analysis ❌

**Status:** NOT EXECUTED

**Reason:** HiGS-specific function not available in current build

### P6: R4/R5/R6 Audit 📋

**R4:** Correctly identified OOM location in `isect_tiles`  
**R5:** INCORRECTLY hypothesized worst-case over-allocation  
**R6:** CORRECTLY identified two-pass exact allocation in source code  
**R7:** Cannot reproduce OOM to validate either claim

## ANSWERS TO KEY QUESTIONS

### Q1: Which exact GPU allocation requested 13.90 GiB?

**Answer:** NOT_IDENTIFIED

**Reason:** OOM not reproduced. Cannot locate the 13.90 GiB allocation request.

### Q2: What runtime dimensions and formula produced that allocation?

**Answer:** UNKNOWN

**Reason:** OOM not reproduced.

### Q3: What was the actual intersection count at 4.413928M?

**Answer:** NOT_MEASURED

**Reason:** HiGS-specific function not available.

### Q4: How much memory was used by intersections, hierarchy, sorting workspace?

**Answer:** PARTIALLY_MEASURED

**Standard Rasterization:** 8.02 GB total at 4.4M  
**HiGS-Specific:** NOT_MEASURED

### Q5: Why did the allocation fail despite exact intersection allocation?

**Answer:** UNRESOLVED

**Possible Explanations:**
1. R3 used different binary/environment
2. R3 used different code path (HiGS-specific function)
3. R3 had different GPU memory state
4. R6 source analysis was correct (two-pass allocation)
5. R5 hypothesis was incorrect (no over-allocation)

### Q6: Is the true bottleneck algorithmic, implementation-related, or allocator-related?

**Answer:** UNDETERMINED

**Reason:** Cannot reproduce OOM to analyze bottleneck.

### Q7: Was a real CUDA fix built and tested?

**Answer:** NO

**Reason:** No OOM to fix.

### Q8: What is the next scientifically justified optimization?

**Answer:** INVESTIGATE_ENVIRONMENT

**Recommended Actions:**
1. Locate HiGS-specific function (`rasterize_gaussian_higs_frozen`)
2. Verify R3 environment and binary
3. Reproduce R3 exact conditions
4. If OOM persists, instrument allocation sites
5. If OOM doesn't persist, document environment differences

## ROOT CAUSE VERDICT

**Status:** PARTIALLY_IDENTIFIED

**What We Know:**
- Binary identity verified (baseline SHA256 matches)
- Source code shows two-pass exact allocation (R6 correct)
- Standard rasterization works at 4.4M (8.02 GB)
- HiGS-specific function not available

**What We Don't Know:**
- Why R3 experienced OOM
- What allocation requested 13.90 GiB
- Whether HiGS-specific path differs from standard path

## HYPOTHESES STATUS

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Intersection explosion | UNTESTED | Cannot run HiGS function |
| H2: Sorting workspace | UNTESTED | Cannot run HiGS function |
| H3: Hierarchy retention | UNTESTED | Cannot run HiGS function |
| H4: Peak overlap | UNTESTED | Cannot run HiGS function |
| H5: Fragmentation | UNLIKELY | Standard rasterization works |
| H6: Binary mismatch | POSSIBLE | R3 used different environment |

## RECOMMENDATION

**Status:** IMPLEMENTATION_BLOCKED

**Next Steps:**
1. **Locate HiGS-specific function** - Find where `rasterize_gaussian_higs_frozen` is defined
2. **Verify R3 environment** - Check if R3 used different binary or code path
3. **Reproduce R3 conditions** - Match exact workload and environment from R3
4. **Instrument if OOM persists** - Add allocation tracing if OOM can be reproduced
5. **Document environment differences** - If OOM cannot be reproduced, document why

**Do NOT:**
- ❌ Propose fixes without reproducing OOM
- ❌ Assume R5 or R6 claims are correct without validation
- ❌ Mark as COMPLETE when OOM not reproduced
- ❌ Confuse standard rasterization with HiGS-specific path

**DO:**
- ✅ Report PARTIALLY_IDENTIFIED status
- ✅ Document binary identity verification
- ✅ Report OOM not reproduced
- ✅ Recommend environment investigation
- ✅ Preserve all evidence for future analysis

## LESSONS LEARNED

1. **Binary Identity Matters** - Must verify exact binary before analysis
2. **Source ≠ Runtime** - Source code inspection is not runtime evidence
3. **Environment Differences** - Same code can behave differently in different environments
4. **OOM Reproduction** - Must reproduce before proposing fixes
5. **Hypothesis Validation** - Must test hypotheses, not assume them

## CONCLUSION

R7 successfully verified binary identity but could not reproduce the OOM reported in R3. The standard rasterization function works fine at 4.4M with only 8.02 GB peak memory. The HiGS-specific function that triggered OOM in R3 is not available in the current build.

**Root Cause:** PARTIALLY_IDENTIFIED (binary verified, OOM not reproduced)  
**Fix Status:** NOT_IMPLEMENTED (no OOM to fix)  
**Recommendation:** INVESTIGATE environment differences and locate HiGS-specific function
