# Phase 8C — Hypothesis Matrix

**Date:** 2026-09-20  
**Status:** ✅ COMPLETED  

---

## Overview

Six competing hypotheses (H7-A through H7-F) evaluated against fresh measurement data
from Phase 8C, combined with source-code trace analysis and workload quantification.

---

## Hypothesis Assessment

### H7-A: Actual backward work quantity differs significantly

| Aspect | Value |
|:-------|:------|
| **Question** | Does tile16 actually perform more backward work than tile32? |
| **Prediction** | Total tile-Gaussian intersections (n_isects) should be significantly higher for tile16 |
| **Observed** | tile16: 145-176M intersections; tile32: 36-44M intersections |
| **Ratio** | **Exactly 4.00×** for all 6 checkpoints |
| **OBSERVED**| YES — 4× more work |
| **EVIDENCE** | Workload quantification on all 6 checkpoints shows precisely 4× ratio determined by tile geometry: 8160 tiles (t16) / 2040 tiles (t32) = 4 |
| **STATUS** | **SUPPORTED** — but 4× work alone CANNOT explain the claimed 137-234× backward ratio |

### H7-B: Same work magnitude but tile16 per-unit work is less efficient

| Aspect | Value |
|:-------|:------|
| **Question** | Given similar work per unit, is tile16's execution inherently slower? |
| **Prediction** | microseconds per intersection (μs/isect) should be significantly higher for tile16 |
| **Observed** | t16: 0.21-0.35 μs/isect; t32: 0.20-0.34 μs/isect |
| **Efficiency ratio** | **0.66-1.58×** (fluctuates around 1.0×, within measurement noise) |
| **OBSERVED** | NO — per-intersection efficiency is very similar |
| **EVIDENCE** | Normalized timing on cold GPU (2 checkpoints) shows per-association cost ratio ~1× |
| **STATUS** | **FALSIFIED** — tile16 is NOT inherently less efficient per unit work |

### H7-C: Tile16-specific backward code path

| Aspect | Value |
|:-------|:------|
| **Question** | Does tile16 trigger a different backward kernel or path? |
| **Prediction** | Different kernel launches or different kernel arguments |
| **Observed** | Same binary (cuobjdump VERIFIED: REG=40, SHARED=1024, STACK=0 for all tile sizes). Same 4 kernel sequence. Only grid/block dimensions differ |
| **OBSERVED** | NO — exact same code path |
| **EVIDENCE** | Full source trace in `phase8c_backward_path_trace.md`. All 4 backward kernels are identical for both tile sizes |
| **STATUS** | **FALSIFIED** — no tile16-specific path exists |

### H7-D: Nonlinear growth from tile-Gaussian intersections

| Aspect | Value |
|:-------|:------|
| **Question** | Does workload grow nonlinearly, explaining the extreme ratio? |
| **Prediction** | t16 should show superlinear scaling of backward time vs intersection count |
| **Observed** | t16 backward: 45ms → 49ms (1.07×) for isects increasing 1.21× |
| **Scaling** | **LINEAR or sub-linear** for t16 backward |
| **OBSERVED** | NO — t16 backward scales near-linearly with work |
| **EVIDENCE** | From iter5000 to iter30000: Gs +33%, isects +21%, t16 bwd +7%. No superlinearity |
| **STATUS** | **FALSIFIED** — the scaling is actually linear or sublinear |

### H7-E: Memory / atomic contention

| Aspect | Value |
|:-------|:------|
| **Question** | Does tile16 suffer from worse memory access patterns? |
| **Prediction** | Higher atomic contention, lower cache hit rates for tile16 |
| **Observed** | Nsight Compute BLOCKED by WDDM on this RTX 5070 laptop |
| **OBSERVED** | NO hardware counter evidence available |
| **EVIDENCE** | Source code shows `gpuAtomicAdd` for each valid Gaussian-pixel pair. Tile size does not change the number of atoms per pair, only the mapping of warps to memory addresses |
| **STATUS** | **BLOCKED** — Nsight Compute unavailable; hardware counters not measurable |

### H7-F: Timing artifact / synchronization issue

| Aspect | Value |
|:-------|:------|
| **Question** | Does the extreme ratio come from a measurement artifact? |
| **Prediction** | Phase 8B ratios should be reproducible under identical conditions |
| **Observed** | **NOT reproducible.** Phase 8C cold GPU → 3-6× ratio. Phase 8B → 137-234× ratio |
| **Root cause** | GPU thermal throttling. Phase 8B tile16 fwd+bwd raw samples show clear bimodal pattern (~875ms ↔ ~2528ms) consistent with power/thermal throttling |
| **STATUS** | **SUPPORTED** — the extreme ratio IS a timing artifact |

---

## Summary Matrix

| Hypothesis | Status | Conclusion |
|:-----------|:------:|:-----------|
| **H7-A**: More backward work | **SUPPORTED** | 4× more intersections, but only explains ~4× ratio |
| **H7-B**: Per-unit inefficiency | **FALSIFIED** | Per-intersection cost is similar (~1× ratio) |
| **H7-C**: Different code path | **FALSIFIED** | Same binary, same 4 kernels |
| **H7-D**: Nonlinear scaling | **FALSIFIED** | t16 backward scales near-linearly with work |
| **H7-E**: Memory contention | **BLOCKED** | Nsight Compute unavailable |
| **H7-F**: Timing artifact | **SUPPORTED** | Phase 8B ratios caused by GPU thermal throttling |

---

## Final Conclusion

**The genuine backward advantage of tile32 over tile16 is ~4×, fully explained by the 4× fewer tile-Gaussian intersections.**

The claimed 137-234× backward ratio from Phase 8B is **not reproducible** and is attributable to GPU thermal throttling from running tile16's heavy forward+backward workload without adequate cooldown between measurements.

**H7 (as originally defined: "backward sensitivity is 10-30× larger than forward") is FALSIFIED.** The backward ratio (2.6-6.3×) is comparable to or smaller than the forward ratio (3.7-15.6×) when measured correctly.

---

## What Remains OPEN

1. **Memory contention quantification (H7-E):** BLOCKED by Nsight Compute accessibility on WDDM. Source code analysis shows gpuAtomicAdd in critical path, but hardware-level measurement is needed to quantify its actual impact.

2. **Why forward scales superlinearly while backward does not:** t16 forward time increases 4.66× for 1.21× work increase, while t16 backward increases only 1.07×. This asymmetry is not yet explained and may indicate the forward kernel has a different computational complexity than the backward kernel with respect to tile density.

3. **Cross-scene validation:** Room only. Bicycle/garden untested (server BLOCKED).
