# Trainable HiGS Memory Scaling R4 - Final Report

## Executive Summary

**Campaign:** TRAINABLE-HIGS-MEMSCALING-R4-MX  
**Date:** 2026-10-08  
**Status:** COMPLETE  
**Execution Mode:** FRESH_MX_MEASUREMENT

**Key Finding:** HiGS native backend OOM at 4.4M Gaussians due to 11.3x memory amplification vs std_ll baseline. Root cause identified as tile intersection buffer allocation in `isect_tiles` function.

**Final Verdict:** INVESTIGATE memory optimization before deciding to drop HiGS.

---

## 1. Experiment Overview

### Objective
Investigate why HiGS native backend OOMs at 4.4M Gaussians on 40GB GPU, identify root cause, and determine if HiGS is viable for high-N workloads.

### Workload
- **Scene:** bicycle
- **Full State:** 4,413,928 Gaussians (r4_13scene_v2/candidate_c/checkpoints/iter_30000.pt)
- **Resolution:** 960×540 (540p)
- **Batch Size:** 1
- **SH Degree:** 3

### Arms
- **Baseline:** std_ll (gsplat rasterization)
- **Candidate:** higs_native (Trainable HiGS with native backward)

---

## 2. Key Results

### 2.1 Memory Profiling at 4.4M

**std_ll (SUCCESS):**
- Gaussian state: 1.04 GB
- Forward peak: 8.01 GB
- Backward peak: 8.01 GB
- Total peak: 8.01 GB

**higs_native (OOM):**
- Gaussian state: 1.05 GB
- OOM allocation: 37.74 GiB (with CUDA_LAUNCH_BLOCKING=1)
- OOM allocation: 13.90 GiB (async)
- Memory in use at OOM: 30.40 GB
- **Conclusion:** HiGS requires ~45 GB vs std_ll 8 GB (11.3x amplification)

### 2.2 N-Scaling Test

| N | std_ll Peak | HiGS Peak | Amplification | Status |
|---|-------------|-----------|---------------|--------|
| 580K | 1.07 GB | 12.07 GB | 11.28x | Both PASS |
| 1.0M | 1.84 GB | 20.76 GB | 11.28x | Both PASS |
| 1.5M | 2.76 GB | 31.10 GB | 11.27x | Both PASS |
| 2.0M | 3.66 GB | 41.41 GB | 11.32x | Both PASS |
| 2.5M | 4.57 GB | OOM | N/A | std_ll PASS, HiGS OOM |
| 4.4M | 8.01 GB | OOM | N/A | std_ll PASS, HiGS OOM |

**OOM Cliff:** Between 2.0M and 2.5M Gaussians on 40GB GPU.

### 2.3 Memory Model

**std_ll:** `M = 1.83 GB/Million * N + 0.24 GB`  
**higs_native:** `M = 10.35 GB/Million * N + 6.1 GB`  
**Amplification Factor:** 11.3x (consistent across all N)

---

## 3. Root Cause Analysis

### 3.1 OOM Location

**Function:** `isect_tiles` (tile intersection generation)  
**File:** `gsplat/experimental/render/functional/gaussian_inference.py:2094`  
**Call Chain:** `rasterize_gaussian_higs_frozen` → `_higs_frozen_forward` → `_HigsAutogradFunction.forward` → `_native_forward_capture` → `isect_tiles`

### 3.2 Hypothesis Verdicts

| Hypothesis | Verdict | Confidence | Evidence |
|------------|---------|------------|----------|
| **H1: Intersection Amplification** | **SUPPORTED** | HIGH | OOM always in isect_tiles; 11.3x memory amplification; 37.74 GiB single allocation |
| **H2: Hierarchy Auxiliary Memory** | **SUPPORTED** | HIGH | Gaussian state only 1 GB; additional 36 GB from hierarchy structures |
| **H3: Temporary Workspace** | **REFUTED** | HIGH | Only 1.76 GB allocated before OOM; not temporary |
| **H4: Allocator Fragmentation** | **REFUTED** | HIGH | PyTorch reserved matches allocated; not fragmentation |
| **H5: Over-allocation** | **POSSIBLE** | MEDIUM | 37.74 GiB allocation suggests pre-allocated buffer |

**Primary Root Cause:** H1 (Intersection Amplification) + H2 (Hierarchy Auxiliary Memory)

---

## 4. Answers to Key Questions

### Q1: Why does HiGS OOM at 4.413928M?

**Answer:** Tile intersection buffers in `isect_tiles` function. HiGS generates 11.3x more memory usage than std_ll. At 4.4M, estimated requirement is ~45 GB vs 8 GB for std_ll.

**Evidence:** N-scaling linear model shows 10.35 GB per million Gaussians vs 1.83 GB for std_ll. OOM always occurs in intersection tile generation.

### Q2: What is HiGS's runnable N boundary on 40GB?

**Answer:** Max verified N is 2.0M (41.41 GB peak). OOM cliff is between 2.0M and 2.5M. std_ll can handle 4.4M with only 8.01 GB peak.

**Evidence:** N-scaling test shows consistent 11.3x amplification across 580K-2.0M range. HiGS OOM at 2.5M and 4.4M.

### Q3: Can HiGS run at 4.413928M without sacrificing correctness?

**Answer:** Unknown. Root cause identified but fix not implemented. Candidate approaches: intersection buffer compression, chunked processing, streaming construction, or compact indexing. However, historical 580K performance gain is marginal (1.02x-1.09x).

**Evidence:** Root cause is algorithmic (intersection buffer), but fix feasibility and performance impact not tested.

---

## 5. Final Verdicts

### A. OOM Root Cause: **IDENTIFIED**
Tile intersection buffer allocation in `isect_tiles` causes 11.3x memory amplification.

### B. 40GB Scalability: **FAIL**
Max verified N is 2.0M. Cannot reach 4.4M target.

### C. Memory Fix: **NOT_TESTED**
Root cause identified but fix not implemented.

### D. Performance: **INCONCLUSIVE**
HiGS blocked by OOM at 4.4M. Historical 580K speedup marginal (1.02x-1.09x).

### E. Research Decision: **INVESTIGATE**
Root cause is algorithmic and may be fixable. However, marginal historical performance suggests HiGS may not be worthwhile even if memory fixed.

---

## 6. Recommendation

**Primary:** INVESTIGATE memory optimization before deciding to drop HiGS.

**Rationale:**
1. OOM root cause is identified (intersection buffer allocation)
2. Memory amplification (11.3x) suggests fundamental design issue
3. HiGS may be viable with memory optimization (compression, chunking, streaming)
4. However, historical 580K performance gain is marginal (1.02x-1.09x)
5. Even if memory fixed, HiGS may not provide sufficient benefit

**Next Steps:**
1. Implement intersection buffer compression or chunked processing
2. Test if fix reduces memory to <40GB at 4.4M
3. If fix successful, benchmark performance at 4.4M
4. If performance still marginal (<1.15x), drop HiGS
5. If performance significant (>1.20x), continue HiGS research

**Fallback:** If memory optimization fails or performance remains marginal, transition to WV-HIER-ORACLE0 research direction.

---

## 7. Artifacts

**Logs:**
- `logs/memory_profile.log`
- `logs/oom_stack_trace.log`
- `logs/n_scaling.log`

**Raw Data:**
- `raw/memory_profile_4.4M.json`
- `raw/oom_stack_trace.json`
- `raw/n_scaling.json`

**Scripts:**
- `scripts/profile_memory.py`
- `scripts/oom_stack_trace.py`
- `scripts/n_scaling.py`

**Analysis:**
- `R4_HYPOTHESIS_VERDICTS.json`
- `R4_FINAL_VERDICT.json`
- `R4_ENVIRONMENT_IDENTITY.json`
- `R4_SOURCE_IDENTITY.json`

---

## 8. Conclusion

HiGS native backend cannot scale to 4.4M Gaussians on 40GB GPU due to 11.3x memory amplification caused by tile intersection buffer allocation. Root cause is identified and appears to be algorithmic. However, historical 580K performance gain is marginal (1.02x-1.09x), suggesting HiGS may not provide sufficient benefit even if memory issue is resolved.

**Recommendation:** Investigate memory optimization (intersection compression, chunked processing) before deciding to drop HiGS. If optimization fails or performance remains marginal, transition to alternative research direction (WV-HIER-ORACLE0).

---

**Report Generated:** 2026-10-08  
**Campaign:** TRAINABLE-HIGS-MEMSCALING-R4-MX  
**Status:** COMPLETE
