# Trainable-HiGS-SCALING0-R-MX — FINAL STATUS REPORT

**Date:** 2026-10-08  
**Status:** ✅ COMPLETE  
**Execution Mode:** FRESH_MX_MEASUREMENT  
**Round:** 6/256

---

## Objective Completion Summary

The Trainable-HiGS-SCALING0-R-MX objective has been **successfully completed** with all success criteria met.

---

## Critical Success Criteria Verification

### ✅ 1. Status Field: COMPLETE
- **RUN_STATUS.json:** `"status": "COMPLETE"`
- **SCALING0R_VERDICT.json:** `"status": "COMPLETE"`

### ✅ 2. Execution Mode: FRESH_MX_MEASUREMENT
- All measurements from fresh MX execution
- No historical or reconstructed data used

### ✅ 3. Raw Evidence Directories: All Non-Empty
- **raw/:** 4 files (per-cell benchmark results)
- **logs/:** 1 file (benchmark execution log)
- **commands/:** 2 files (benchmark scripts)
- **gpu_snapshots/:** 1 file (GPU state capture)

### ✅ 4. Fresh Benchmark Execution: Complete
- Both `std_ll` and `higs_native` backends executed
- All 3 HIGH-N workload cells measured (W2, W3, W4)
- Correctness verified (gradient cosine similarity = 1.000000)

### ✅ 5. Disk Space Requirements: Met
- Root: 7.7 GB available (> 4 GB threshold)
- Storage pool: ~34 GB available
- All temporary files in /mnt/storage_pool/liaoyuanjun/tmp_scaling0r

---

## Experiment Results

### Workload Cells Executed

| Cell | Scene | Resolution | Batch | Gaussians | std_ll F+B (ms) | higs_native F+B (ms) | Speedup | Decision |
|------|-------|-----------|-------|-----------|-----------------|---------------------|---------|----------|
| W2 | bicycle | 540p | B4 | 580,416 | 15.88 | 15.54 | **1.02x** | NO_GO |
| W3 | bicycle | 1080p | B4 | 580,416 | 37.50 | 34.56 | **1.09x** | NO_GO |
| W4 | bicycle | 1080p | B1 | 580,416 | 10.98 | 11.09 | **0.99x** | NO_GO |

### Performance Analysis

- **Speedup Range:** 0.99x - 1.09x
- **Mean Speedup:** 1.03x
- **Max Speedup:** 1.09x (W3: 1080p/B4)
- **Min Speedup:** 0.99x (W4: 1080p/B1)
- **Culling Efficiency:** 55-70% of Gaussians culled by HiGS native backend

### Quality Verification

- **PSNR:** Virtually identical between backends
- **SSIM:** Virtually identical between backends
- **LPIPS:** Virtually identical between backends
- **Gradient Cosine Similarity:** 1.000000 (perfect match)

### Final Decision: NO_GO

**Decision Gates:**
- ❌ STRONG_GO (≥1.20x with ≥2 points ≥1.15x): **Not met**
- ❌ MODEST_GO (1.12-1.20x): **Not met**
- ✅ NO_GO (<1.12x): **Met**

**Conclusion:** HiGS native backend provides modest speedup (0.99x-1.09x) but does not meet the 1.12x threshold for MODEST_GO.

---

## Technical Details

### Environment
- **Hostname:** MX
- **GPU:** NVIDIA A100-PCIE-40GB (SM80)
- **PyTorch:** 2.9.1+cu128
- **CUDA:** 12.8
- **gsplat Version:** 1.5.3 (editable install)
- **Git Commit:** 77ab983f
- **Conda Environment:** /mnt/storage_pool/liaoyuanjun/higs-13scene-env

### Key Technical Achievement

**API Version Mismatch Fixed:**
- Issue: Benchmark script passed `cull_cache_key` parameter to `rasterize_gaussian_higs_frozen()`
- Root cause: Compiled gsplat HiGS backend (commit 77ab983f) does not accept this parameter
- Solution: Removed all `cull_cache_key` arguments from benchmark script
- Result: Both backends executed successfully on all workload cells

### Storage Management

**Temporary Workspace:**
- Location: `/mnt/storage_pool/liaoyuanjun/tmp_scaling0r`
- Environment variables:
  - `TMPDIR=/mnt/storage_pool/liaoyuanjun/tmp_scaling0r/tmp`
  - `TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/tmp_scaling0r/torch_extensions`
- All compilation artifacts stored in storage pool (not root filesystem)

**Deleted-but-Open File:**
- PID 2925282 holding 8.45 GB deleted file on /mnt/storage_pool
- Not terminated per instructions (not required for campaign)
- File will be released when process naturally exits

---

## Deliverables Inventory

### Core JSON Files (12 files)
1. RUN_STATUS.json - Overall experiment status
2. SCALING0R_VERDICT.json - Final decision and evidence
3. SCALING0R_TIMING_CONSISTENCY.json - Detailed timing measurements
4. SCALING0R_SOURCE_IDENTITY.json - Source code identity
5. SCALING0R_PROTOCOL_IDENTITY.json - Benchmark protocol
6. SCALING0R_CORRECTNESS.json - Correctness verification
7. SCALING0R_RUN_PLAN.json - Execution plan
8. SCALING0R_RAW_INDEX.json - Raw evidence index
9. SCALING0R_FIXED_STATE.json - Fixed workload definition
10. SCALING0R_MECHANISM.json - Mechanism description
11. SCALING0R_CONTAMINATION.json - Contamination check
12. SCALING0R_SUMMARY.json - Executive summary

### Documentation (3 files)
1. SCALING0R_REPORT.md - Full experiment report
2. FINAL_SUMMARY.md - Executive summary
3. OBJECTIVE_COMPLETE.md - Objective completion report

### Raw Evidence (8 files across 4 directories)
- **raw/:** 4 files (per-cell JSON results)
- **logs/:** 1 file (benchmark execution log)
- **commands/:** 2 files (benchmark scripts)
- **gpu_snapshots/:** 1 file (GPU state capture)

### Checksums
- MANIFEST.sha256 - SHA256 checksums for all JSON files

**Total Files:** 33

---

## Conclusion

The Trainable-HiGS-SCALING0-R-MX experiment has been **successfully completed** with all specification requirements met:

✅ **Status:** COMPLETE  
✅ **Execution Mode:** FRESH_MX_MEASUREMENT  
✅ **All measurements from fresh MX execution**  
✅ **Raw evidence collected and preserved**  
✅ **No historical/reconstructed data used**  
✅ **Disk space requirements satisfied**  
✅ **All deliverables complete and verified**  

**Final Decision:** NO_GO — HiGS native backend speedup (0.99x-1.09x) below 1.12x threshold for MODEST_GO.

**Experiment Status:** ✅ **OBJECTIVE COMPLETE**
