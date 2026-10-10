# R12 Count Debug - Final Report

## EXECUTIVE SUMMARY

**Campaign:** TRAINABLE-HIGS-COUNT-DEBUG-R12-MX  
**Status:** INCONCLUSIVE  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

## CRITICAL REVISION

R11's "COUNTING_BUG_CONFIRMED" conclusion was **INCORRECT** based on flawed physical bounds analysis.

---

## EVIDENCE AUDIT RESULTS

### P0: R10/R11 Evidence Verification ✅

**R11 Metadata Source:**
- n_gaussians: 580,000
- n_visible: 273,234
- n_isects: 490,387,290 (from HiGS metadata)
- culling_ratio: 0.5289

**Corrected Physical Bounds:**
- Image: 960×540
- Tile size: 16×16
- **Tile grid: 60×34 = 2,040 tiles** (NOT 561)
- Max per Gaussian: 2,040 tiles
- Max total: 273,234 × 2,040 = 557,397,360
- **Actual: 490,387,290 = 88% of maximum**

**R11 Error:**
- Claimed "320% of physical maximum" based on WRONG tile count (561)
- Actual: 88% of maximum is HIGH but PHYSICALLY POSSIBLE

### R10 OOM Allocation Analysis ✅

**OOM bytes:** 119,488,359,424

**Two interpretations:**
1. Single int64 array: 14,936,044,928 elements
2. Two arrays (int64 + int32): 9,957,363,285 elements

**Stack trace shows:** `at::empty({n_isects}, opt.dtype(at::kLong))` at line 833

**Conclusion:** Single int64 allocation → n_isects = 14.9B (not 9.96B)

**R11 Error:** Assumed two arrays without verifying allocation semantics

---

## KEY FINDINGS

### Finding 1: Tile Grid Was Wrong

**R11 claimed:** 561 tiles (33×17)  
**Actual:** 2,040 tiles (60×34)

This changes everything:
- 490M is 88% of max (not 320%)
- High coverage is possible with large Gaussians
- Physical impossibility claim was FALSE

### Finding 2: R10 Implies 14.9B Intersections

**R10 OOM:** 119.49 GB single int64 array  
**Implied n_isects:** 14,936,044,928  
**R11 n_isects:** 490,387,290  
**Ratio:** 30×

**Explanation:** Different N values:
- R11: 580K Gaussians → 490M intersections
- R10 OOM: Likely 4.4M → 14.9B intersections

### Finding 3: No Proven Bug Yet

**Evidence against bug:**
- 88% of physical maximum is possible
- No independent oracle verification performed
- Per-Gaussian analysis not done
- First mismatch not identified

**Evidence for concern:**
- 199× higher than "expected 9 tiles"
- But "expected" is assumption, not measurement

---

## HYPOTHESIS STATUS

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Counting bug | **UNPROVEN** | Physical max analysis was wrong |
| Large radius anomaly | **POSSIBLE** | Need per-Gaussian analysis |
| Hierarchy duplication | **POSSIBLE** | Need source audit |
| R10 allocation error | **POSSIBLE** | 14.9B vs 9.96B discrepancy |
| Different N values | **LIKELY** | 580K vs 4.4M explains ratio |

---

## WHAT WAS NOT DONE

### P1: Independent Oracle ❌
- Not implemented
- Cannot verify without per-Gaussian projected means/radii
- HiGS doesn't expose these in metadata

### P2: Per-Gaussian Comparison ❌
- Not performed
- Cannot identify specific mismatches
- Cannot locate exact bug

### P3: CUDA Error Location ❌
- Not attempted
- No mismatch samples identified
- Cannot pinpoint bug without oracle

### P4: 4.4M Safe Diagnostic ❌
- Not executed
- Would require source modification
- Count-only mode not implemented

### P5-P7: Fix, Validation, Performance ❌
- Not applicable (no proven bug)

---

## ROOT CAUSE VERDICT

**Status:** INCONCLUSIVE

**What we know:**
- R11's physical impossibility claim was WRONG
- 490M at 580K is 88% of max (possible)
- R10 OOM implies 14.9B at 4.4M (30× higher)
- No independent verification performed

**What we don't know:**
- Whether counting is actually wrong
- Which specific Gaussians have issues
- Whether hierarchy causes duplication
- Whether R10 allocation calculation was correct

---

## RECOMMENDATION

**Next Steps:**

1. **Implement per-Gaussian oracle**
   - Extract projected means/radii from HiGS
   - Compute AABB tile coverage independently
   - Compare with HiGS count kernel output

2. **Audit hierarchy semantics**
   - Understand HiGS hierarchy record semantics
   - Determine if duplicates are expected
   - Verify count vs fill consistency

3. **Verify R10 allocation**
   - Check if line 833 is single or dual allocation
   - Verify n_isects source (CPU vs GPU)
   - Confirm dtype and shape

4. **Safe 4.4M diagnostic**
   - Implement count-only mode
   - Measure actual n_isects without OOM
   - Compare with extrapolation

**Do NOT:**
- ❌ Claim COUNTING_BUG without oracle proof
- ❌ Use average tiles/Gaussian as evidence
- ❌ Assume physical bounds without verification
- ❌ Modify code without proven bug

**DO:**
- ✅ Implement true independent oracle
- ✅ Per-Gaussian comparison
- ✅ Locate specific mismatch samples
- ✅ Verify allocation semantics

---

## CONCLUSION

R12 successfully identified that **R11's bug confirmation was premature**:
- Physical bounds analysis was incorrect (561 vs 2040 tiles)
- 490M is 88% of max, not 320%
- No independent oracle was implemented
- No specific mismatch was located

**Status:** INCONCLUSIVE - Need per-Gaussian oracle and source audit  
**Fix Status:** NOT_IMPLEMENTED - No proven bug  
**Next Action:** IMPLEMENT_INDEPENDENT_ORACLE
