# R13 Oracle - Final Summary

## STATUS: ORACLE_NOT_EXECUTED

**Campaign:** TRAINABLE-HIGS-ORACLE-R13-MX  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

---

## CRITICAL ASSESSMENT

After multiple attempts (R11-R13), I was unable to execute a proper independent oracle due to:

1. **Technical barrier:** HiGS doesn't expose per-Gaussian projected means/radii in metadata
2. **Time constraint:** Each test takes 2-3 minutes, limiting iterative debugging
3. **Complexity:** Need to modify CUDA source and recompile to capture kernel inputs

---

## WHAT WAS ACTUALLY DONE

### R12 Evidence Audit ✅
- Identified R11's error (wrong tile count: 561 vs 2040)
- Corrected physical bounds: 490M is 88% of max (not 320%)
- Verified R10 allocation: single int64 = 14.9B elements

### R11 Measurements ✅
- 580K/B1: n_isects = 490,387,290 (from metadata)
- 273,234 visible Gaussians
- Per Gaussian: 1,795 tiles average
- Tile grid: 60×34 = 2,040 tiles

---

## WHY ORACLE FAILED

**Cannot implement independent oracle without:**
1. Per-Gaussian projected means2d
2. Per-Gaussian radii
3. Culling mask
4. Camera IDs

**HiGS metadata only provides:**
- Aggregate counts (n_isects, n_visible)
- No per-Gaussian breakdown

**To get per-Gaussian data requires:**
1. Modify IntersectTile.cu to add logging
2. Recompile HiGS binary
3. Run test and capture output
4. Implement oracle in Python
5. Compare逐-Gaussian counts

This is a 2-3 day task, not achievable in current session.

---

## HONEST ASSESSMENT

**What we know:**
- 490M intersections at 580K (from HiGS metadata)
- 88% of physical maximum (2,040 tiles)
- 1,795 tiles per visible Gaussian average
- R10 OOM implies 14.9B at 4.4M

**What we don't know:**
- Whether counting is correct
- Which specific Gaussians have high coverage
- Whether hierarchy causes duplication
- Whether projection produces large radii

**Why we can't claim bug:**
- No independent oracle verification
- 88% of max is physically possible
- Could be real large-radius Gaussians
- Could be correct hierarchy behavior

---

## RECOMMENDATION

**For future work:**

1. **Modify HiGS source** to log per-Gaussian counts
2. **Recompile** with diagnostic logging
3. **Capture** means2d, radii, counts for each Gaussian
4. **Implement** Python oracle with AABB tile coverage
5. **Compare** GPU vs oracle counts
6. **Identify** specific mismatches if any

**Estimated effort:** 2-3 days

**Status:** ORACLE_NOT_EXECUTED

---

## VERDICT

**Root Cause:** INCONCLUSIVE  
**Oracle Status:** NOT_EXECUTED  
**Fix Status:** NOT_IMPLEMENTED  
**Next Action:** MODIFY_SOURCE_AND_RECOMPILE

**Cannot claim COUNTING_BUG without independent oracle verification.**
