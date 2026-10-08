# R9 Allocation Trace - Final Report

## EXECUTIVE SUMMARY

**Campaign:** TRAINABLE_HIGS_ALLOCATION_TRACE_R9_MX  
**Status:** FORENSICS_INCOMPLETE  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

## CRITICAL FINDING

**111.28 GiB allocation site NOT IDENTIFIED.** The allocation occurs inside CUDA kernels, not via PyTorch's torch.empty, making Python-level tracing ineffective.

---

## R3/R8 COMPARISON

### Configuration Comparison

| Parameter | R3 | R8 | Match? |
|-----------|----|----|--------|
| Function | rasterize_gaussian_higs_frozen | rasterize_gaussian_higs_frozen | ✓ |
| Binary | b64d02c1... | b64d02c1... | ✓ |
| Checkpoint | bicycle/iter_30000 | bicycle/iter_30000 | ✓ |
| N | 4,413,928 | 4,413,928 | ✓ |
| Resolution | 960×540 | 960×540 | ✓ |
| Batch | 4 | 4 | ✓ |
| **Allocation** | **13.90 GiB** | **111.28 GiB** | **✗ 8× DIFF** |

### Possible Explanations for 8× Difference

1. **Different GPU memory state**
   - R3: May have had more free memory available
   - R8: 23.45 GiB used by other processes

2. **Different allocation site**
   - R3: 13.90 GiB allocation (site unknown)
   - R8: 111.28 GiB allocation (site unknown)
   - May be different data structures

3. **Different camera configuration**
   - Both use B4, but camera poses may differ
   - R3 used circular camera path
   - R8 used single identity camera (test configuration)

4. **Different runtime dispatch**
   - Both use same binary
   - But may hit different code paths based on inputs

---

## ALLOCATION TRACING RESULTS

### P1: Python-level Tracing

**Method:** Patched torch.empty to log large allocations

**Result:** INEFFECTIVE
- 0 allocations >1M elements logged at 580K
- All allocations happen inside CUDA kernels
- Cannot trace via Python hooks

**Conclusion:** Need CUDA-level instrumentation (requires recompilation)

### P2: Count-only Diagnostic

**Status:** NOT_IMPLEMENTED

**Reason:** Cannot intercept allocation size calculation without modifying CUDA source code

### P3: Source Code Analysis

**R6 Finding:** HiGS uses two-pass exact allocation in IntersectTile.cu

**R5 Hypothesis:** Over-allocation via MAX_INTERSECTIONS_PER_GAUSSIAN

**R6 Correction:** Already using exact allocation (n_isects from count pass)

**Status:** Both R5 and R6 analyses are based on source inspection, not runtime evidence

---

## HYPOTHESIS STATUS

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Intersection explosion | UNTESTED | Cannot measure without CUDA instrumentation |
| H2: Camera/batch duplication | UNLIKELY | Both use B4 |
| H3: Hierarchy duplication | UNTESTED | Cannot measure |
| H4: Sorting workspace | UNTESTED | Cannot measure |
| H5: Peak overlap | UNTESTED | Cannot measure |
| H6: Different allocation sites | POSSIBLE | 13.90 vs 111.28 GiB |
| H7: Implementation error | UNTESTED | Cannot verify without runtime data |
| H8: Source/binary mismatch | REFUTED | Binary SHA256 verified |

---

## ROOT CAUSE VERDICT

**Status:** FORENSICS_INCOMPLETE

**What We Know:**
- OOM reproduced at 4.4M (111.28 GiB allocation)
- 580K works fine (12.07 GB peak)
- Function and binary verified
- Python-level tracing ineffective

**What We Don't Know:**
- Which allocation line causes 111.28 GiB
- What runtime dimensions produce that size
- Why R3 saw 13.90 GiB vs R8's 111.28 GiB
- Actual intersection count at 4.4M
- Whether this is algorithmic or implementation issue

---

## RECOMMENDATION

**Next Step:** CUDA-level allocation instrumentation

**Required Actions:**
1. Modify IntersectTile.cu to add allocation logging
2. Rebuild HiGS with instrumentation
3. Run 580K and 4.4M tests
4. Capture actual allocation sizes and stack traces
5. Identify exact allocation site

**Do NOT:**
- ❌ Propose fixes without allocation-level evidence
- ❌ Assume allocation size is proportional to N
- ❌ Compare R3/R8 sizes without understanding sites
- ❌ Mark as COMPLETE when allocation site unknown

**DO:**
- ✅ Instrument CUDA allocations
- ✅ Log allocation sizes and timing
- ✅ Capture full call stacks
- ✅ Measure intersection counts
- ✅ Report FORENSICS_INCOMPLETE status

---

## CONCLUSION

R9 successfully reproduced OOM but could not identify the exact allocation site due to CUDA-level allocation bypassing Python tracing. The 8× difference between R3 (13.90 GiB) and R8 (111.28 GiB) remains unexplained and may indicate different allocation sites or GPU memory states.

**Root Cause:** FORENSICS_INCOMPLETE  
**Fix Status:** NOT_IMPLEMENTED  
**Next Action:** CUDA_INSTRUMENTATION
