# Phase 11 — Candidate Screening Report

**Date:** 2026-09-21  
**Author:** DSH coding agent  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU  
**Status:** COMPLETE

---

## 1. Overview

Phase 11 completes the evidence chain for all candidate optimization modules (M1–M5) that were left at varying stages of completion from previous phases. The evaluation follows a strict evidence chain:

```
Forward Correctness → Gradient Correctness → GT Quality → Training → Composability
```

**Server status:** EPIC-05 unreachable (`SERVER_BLOCKED`). Cross-scene validation (bicycle 30K, garden 30K) for M1 remains `PENDING_CROSS_SCENE`.

---

## 2. M1 — Tile Size (tile16 vs tile32)

### Results from Phase 7/8

| Evidence | Status |
|:---------|:------:|
| **Forward Correctness** | ✅ PASS (pixel-identical on room) |
| **Gradient Correctness** | ✅ PASS (all params finite, correct) |
| **GT Quality** | ✅ PASS (3 scenes, pixel-identical) |
| **Training (room 30K)** | ✅ PASS (tile32 faster, same quality) |
| **Cross-scene (bicycle/garden)** | ⏳ **PENDING_CROSS_SCENE** (requires EPIC-05) |

### Classification
- **Evidence status:** STRONG (room only)
- **Performance value:** POSITIVE on room (3.4×–13.5× forward speedup on real checkpoints)
- **Research value:** HIGH — tiles_per_gauss ≈ total tiles for real scenes (100% tile occupancy)
- **Composability eligibility:** `true` for room; `PENDING_CROSS_SCENE` overall

### Limitation
Results are validated on room scene only. The 3.4×–13.5× speedup on real checkpoints is dramatic, but cross-scene generalization on bicycle (~6.1M initial Gaussians, fundamentally different workload regime) remains unverified.

---

## 3. M2 — Packed/Dense

### Results from Phase 9A/10A

| Evidence | Status |
|:---------|:------:|
| **Source Audit** | ✅ COMPLETE — packed path compacts visible Gaussians, dense processes all |
| **Forward Correctness** | ✅ PASS (pixel-identical, verified) |
| **Gradient Correctness** | ✅ PASS (gradcheck passed) |
| **Training (room 30K)** | ✅ PASS |
| **E2E Benefit** | **NOT MEANINGFUL** (packed=62.2min, dense=64.3min, ratio ≈1.03×) |

### Classification
- **Evidence status:** PASS
- **Performance value:** NEUTRAL (1.03× difference is not meaningful for training)
- **Research value:** USEFUL NEGATIVE — explains that dense mode's larger memory footprint and simpler indexing don't translate to meaningful training speedup
- **Composability eligibility:** `false` (no benefit to compose)

---

## 4. M3 — SH Degree

### Results from Phase 9B/9C

| Evidence | Status |
|:---------|:------:|
| **Source Audit** | ✅ COMPLETE |
| **Forward Correctness** | ✅ PASS (expected degree-dependent differences) |
| **Gradient Correctness** | ✅ PASS |
| **GT Quality** | ✅ PASS (degree-dependent quality trade-off documented) |
| **Training (room 30K)** | ✅ PASS (all 3 degrees: SH0, SH1, SH3) |

### Classification
- **Evidence status:** PASS
- **Performance value:** DEGREE-DEPENDENT — SH0 is fastest but lowest quality, SH3 is highest quality but slower
- **Research value:** HIGH — clear representation/compute trade-off with practical implications
- **Composability eligibility:** `true`

### SH Degree Trade-off
| SH Degree | Quality | Speed | Use Case |
|:---------:|:-------:|:-----:|:---------|
| 0 | Lowest | Fastest | Minimal viable quality, real-time |
| 1 | Medium | Medium | Balanced |
| 3 | Highest | Slowest | Production quality |

---

## 5. M4 — `radius_clip`

### Results (Phase 11, this report)

| Evidence | Status |
|:---------|:------:|
| **Source Audit** | ✅ COMPLETE — filters Gaussians with both radii ≤ threshold from projection output |
| **Forward Correctness** | ✅ PASS (bounded differences, mechanism verified) |
| **Gradient Correctness** | ✅ PASS (all params finite, correct semantics) |
| **GT Quality** | ✅ PASS (preserved for clip ≤ 2.0) |
| **Workload Reduction** | ❌ **FALSIFIED** (< 1% intersection reduction at quality-preserving thresholds) |
| **Training Benefit** | ❌ **FALSIFIED** (no workload to convert) |

### Classification
- **Evidence status:** PASS (mechanism works correctly)
- **Performance value:** **NOT BENEFICIAL** for indoor scenes
- **Research value:** POSITIVE NEGATIVE — explains why radius_clip doesn't help indoors: removed Gaussians are inherently tiny (both radii ≤ threshold), contributing negligible intersection workload
- **Composability eligibility:** `false`

### Why it Fails for Indoor Scenes
The Gaussians whose both radii fall below a threshold are inherently **tiny** — they span very few tiles. Removing them saves a disproportionately small fraction of the intersection workload (the expensive part of rendering). For outdoor scenes with many far-away tiny Gaussians (e.g., bicycle), the mechanism might still help, but this is untestable without EPIC-05.

---

## 6. M5 — `eps2d`

### Results (Phase 11, this report)

| Evidence | Status |
|:---------|:------:|
| **Source Audit** | ✅ COMPLETE — adds eps2d to covariance diagonal, affecting conic, radii, tile coverage |
| **Forward Correctness** | ✅ PASS (expected output differences) |
| **Gradient Correctness** | ✅ PASS (all params finite, smooth gradient change) |
| **GT Quality** | ✅ PASS — **default eps2d=0.3 is optimal** (best PSNR/SSIM) |
| **Workload Savings** | ❌ **FALSIFIED** (workload INCREASES with eps2d) |
| **Training Benefit** | ❌ **FALSIFIED** (no quality-preserving speedup possible) |

### Classification
- **Evidence status:** PASS (mechanism and gradients correct)
- **Performance value:** **NOT BENEFICIAL** — default is optimal
- **Research value:** POSITIVE — confirms gsplat's default eps2d=0.3 is well-chosen; explains the quality-regularizing role
- **Composability eligibility:** `false`

### Why it's Not Tunable
eps2d=0.0 gives PSNR 30.01 (−1.61dB vs default) with only 9% faster forward pass. The quality cost far outweighs the speed gain.

---

## 7. Candidate Summary Matrix

| Module | Evidence | Performance | Research Value | Eligible for Composability |
|:-------|:--------:|:-----------:|:--------------:|:--------------------------:|
| **M1 tile16** | ✅ PASS | ⏺ NEUTRAL (baseline) | N/A | `true` (baseline) |
| **M1 tile32** | ⏳ PENDING_CROSS_SCENE | ✅ POSITIVE (room) | HIGH | `true` (room) / `PENDING` (general) |
| **M2 packed/dense** | ✅ PASS | ⏺ NEUTRAL | USEFUL NEGATIVE | `false` |
| **M3 SH degree** | ✅ PASS | 📊 DEGREE-DEPENDENT | HIGH | `true` |
| **M4 radius_clip** | ✅ PASS | ❌ NOT BENEFICIAL | POSITIVE NEGATIVE | `false` |
| **M5 eps2d** | ✅ PASS | ❌ NOT BENEFICIAL | POSITIVE | `false` |

---

## 8. Composability Candidates

Only modules with `eligible_for_composability = true` can proceed to composability testing.

### Currently Eligible
| Module | Scene Restriction | Notes |
|:-------|:-----------------|:------|
| **M1 tile32** | Room only | Cross-scene validation needed (requires EPIC-05) |
| **M3 SH0** | None | Replace SH3 with SH0 for speed, accept quality trade-off |
| **M3 SH1** | None | Replace SH3 with SH1 for balanced trade-off |

### Composability Ordering (When EPIC-05 Recovers)
1. M1 tile32 × M3 SH0 (maximum speed, room scene)
2. M1 tile32 × M3 SH1 (balanced, room scene)
3. M1 tile32 × M3 SH0/1 on bicycle (cross-scene composability)
4. M1 tile32 × M3 SH0/1 on garden (cross-scene composability)

### Modules Not Eligible
M2, M4, and M5 have been conclusively evaluated as non-beneficial for E2E training on the room scene. No further work is warranted.

---

## 9. Server Blocked Items

```
SERVER_BLOCKED
```

The following experiments are ready and waiting for EPIC-05 recovery:
1. M1 tile32 — bicycle 30K full training
2. M1 tile32 — garden 30K full training
3. M1 tile32 × M3 composability (if cross-scene validated)
4. Any A100/TCC-specific testing
5. Nsight Compute profiling for kernel-level analysis

These are the **only** items blocked by EPIC-05. All other research (M2–M5) is complete on the local machine.
