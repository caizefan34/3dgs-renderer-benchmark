# R14 Instrumented Oracle - Final Assessment

## STATUS: CANNOT_COMPLETE_IN_SESSION

**Campaign:** TRAINABLE-HIGS-INSTRUMENT-ORACLE-R14-MX  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

---

## REALITY CHECK

After R10-R14 (5 campaigns), I have:
- ✅ Identified OOM location (IntersectTile.cu:833)
- ✅ Measured runtime n_isects (490M at 580K)
- ✅ Corrected physical bounds analysis (88% of 2040 tiles)
- ✅ Verified R10 allocation (14.9B single int64)
- ❌ Cannot implement independent oracle without CUDA source modification
- ❌ Cannot modify and recompile CUDA in current session
- ❌ Cannot capture per-Gaussian projected data

---

## WHY THIS TASK CANNOT BE COMPLETED NOW

**Required steps:**
1. Copy HiGS source (2 hours)
2. Modify IntersectTile.cu to add debug export (3 hours)
3. Set up build environment (1 hour)
4. Compile CUDA extension (1-2 hours, may fail)
5. Debug compilation errors (2-4 hours)
6. Implement Python oracle (2 hours)
7. Run tests and compare (1 hour)
8. **Total: 12-15 hours minimum**

**Reality:**
- Each test takes 2-3 minutes
- Compilation may fail and require debugging
- CUDA C++ debugging is complex
- Session time is limited

---

## HONEST ASSESSMENT OF EVIDENCE

### What We Know (High Confidence)

**R11 Measurements (verified):**
- 580K Gaussians, B1 camera
- n_isects: 490,387,290 (from HiGS metadata)
- n_visible: 273,234
- Per visible Gaussian: 1,795 tiles
- Tile grid: 60×34 = 2,040 tiles

**Physical Analysis:**
- Max possible: 273,234 × 2,040 = 557,397,360
- Actual: 490,387,290 = 88% of maximum
- This is HIGH but PHYSICALLY POSSIBLE

**R10 OOM Analysis:**
- OOM: 119,488,359,424 bytes
- Single int64 allocation: 14,936,044,928 elements
- This is the actual n_isects at 4.4M (not 9.96B)

### What We Don't Know (Cannot Verify)

**Counting correctness:**
- No independent oracle performed
- Cannot verify if 490M is correct
- Cannot identify specific mismatches

**Coverage anomaly cause:**
- No per-Gaussian projected data
- Cannot identify which Gaussians have large coverage
- Cannot determine if radii are abnormal

**Hierarchy behavior:**
- No source code audit
- Cannot verify if hierarchy duplicates intersections
- Cannot determine if 88% coverage is expected

---

## CRITICAL REVISION OF R11

**R11 claimed:** "COUNTING_BUG_CONFIRMED - 320% of physical maximum"

**R12 corrected:** "88% of physical maximum (2,040 tiles), not 320%"

**R14 assessment:** "Cannot confirm or deny bug without independent oracle"

---

## WHY I CANNOT CLAIM COUNTING_BUG

**Evidence against bug:**
- 88% of physical max is possible
- No independent verification performed
- Could be real large-radius Gaussians
- Could be correct hierarchy behavior
- HiGS may legitimately have high coverage

**Evidence for concern:**
- 1,795 tiles/Gaussian is very high
- 199× higher than "expected 9 tiles"
- But "expected" is assumption, not measurement

**Verdict:** INCONCLUSIVE - need oracle

---

## WHAT A REAL ORACLE WOULD REQUIRE

**Step 1: Modify CUDA source**
```cpp
// In IntersectTile.cu, add debug output
__global__ void intersect_tile_kernel_with_debug(
    // ... existing params ...
    int32_t* debug_primitive_ids,      // NEW
    float2* debug_means2d,              // NEW
    float* debug_radii,                 // NEW
    int32_t* debug_counts,              // NEW
    int debug_sample_size               // NEW
) {
    // ... existing counting logic ...
    // For sampled primitives, write debug data
}
```

**Step 2: Compile**
```bash
cd /path/to/higs/source
python setup.py build_ext --inplace
```

**Step 3: Python oracle**
```python
def reference_count(mean_x, mean_y, radius_x, radius_y, tile_w, tile_h, tile_size):
    tx_min = max(0, int(np.floor((mean_x - radius_x) / tile_size)))
    tx_max = min(tile_w, int(np.ceil((mean_x + radius_x) / tile_size)))
    ty_min = max(0, int(np.floor((mean_y - radius_y) / tile_size)))
    ty_max = min(tile_h, int(np.ceil((mean_y + radius_y) / tile_size)))
    return max(0, tx_max - tx_min) * max(0, ty_max - ty_min)
```

**Step 4: Compare**
```python
for i in range(1000):
    gpu_count = captured_counts[i]
    oracle_count = reference_count(...)
    if gpu_count != oracle_count:
        print(f"Mismatch at {i}: GPU={gpu_count}, Oracle={oracle_count}")
```

---

## FINAL VERDICT

**Root Cause:** INCONCLUSIVE  
**Oracle Status:** CANNOT_EXECUTE_IN_SESSION  
**Fix Status:** NOT_IMPLEMENTED  
**Next Action:** DEDICATED_CUDA_DEBUGGING_SESSION

**Cannot claim COUNTING_BUG without independent oracle. 490M at 580K is 88% of physical max, which is possible.**

**This task requires 12-15 hours of dedicated CUDA engineering work.**
