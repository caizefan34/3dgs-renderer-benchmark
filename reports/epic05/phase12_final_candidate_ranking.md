# Phase 12 — Final Candidate Ranking

**Date:** 2026-09-21  
**Author:** DSH coding agent  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM)  
**Status:** COMPLETE

---

## 1. Candidate Summary

All modules (M1–M5) have been evaluated through the complete evidence chain:

```
Forward Correctness → Gradient Correctness → GT Quality → Training → Cross-Scene
```

---

## 2. Tier Classification

### Tier A — Strong Performance Candidates

> Must satisfy: correctness + gradient + quality + full training + meaningful E2E benefit + cross-scene evidence

**No module qualifies for Tier A.**

**M1 tile32** fails cross-scene: it is slower than tile16 on bicycle/garden (0.69×). The room advantage (1.58×) does not generalize.

**M2 packed/dense** has only 1.03× E2E benefit — not meaningful.

**M3 SH degree** has <4% forward timing impact — not a performance knob.

---

### Tier B — Validated but Task-Specific

#### M1 tile32 (Room-specific)
| Evidence | Status |
|:---------|:-------|
| Forward Correctness | ✅ PASS (pixel-identical) |
| Gradient Correctness | ✅ PASS (torch.autograd.gradcheck PASS) |
| GT Quality | ✅ PASS (PSNR/SSIM/LPIPS delta = 0.0 across 3 scenes) |
| Full Training (room 30K) | ✅ PASS (1.58× faster, equivalent quality) |
| Cross-Scene (bicycle/garden) | ❌ **FALSIFIED** (tile32 is 1.44× slower) |

**Verdict:** tile32 is beneficial for **dense indoor scenes with fewer Gaussians** (room: 1.6M). It is detrimental for **sparse outdoor scenes** (bicycle/garden: 5.8–6.1M). The optimal tile size is **scene-dependent and hardware-dependent**.

**Recommendation:** Include as conditional recommendation — "use tile32 for indoor scenes with <2M Gaussians, tile16 otherwise."

#### M3 SH Degree (Quality-Compute Trade-off)
| Evidence | Status |
|:---------|:-------|
| Forward Correctness | ✅ PASS (expected degree-dependent differences) |
| Gradient Correctness | ✅ PASS |
| GT Quality | ✅ PASS (degree-dependent quality, consistent cross-scene) |
| Full Training (room 30K) | ✅ PASS (all 3 degrees stable) |
| Cross-Scene SH quality | ✅ PASS (consistent SH0 < SH1 < SH3 ordering) |

**Verdict:** SH degree is a **valid quality-compute trade-off** but not a meaningful performance knob (forward timing impact <4%). Useful for quality-aware deployment decisions, not for performance optimization.

**Recommendation:** SH0 for real-time/low-VRAM scenarios where some quality loss is acceptable; SH3 for production quality.

---

### Tier C — Validated but Performance-Neutral

#### M2 Packed/Dense
| Evidence | Status |
|:---------|:-------|
| Forward Correctness | ✅ PASS (bit-exact) |
| Gradient Correctness | ✅ PASS (max relative diff 5.5e-6) |
| GT Quality | ✅ PASS (identical pixel output) |
| Full Training (room 30K) | ✅ PASS |
| E2E Benefit | **1.03×** — NOT meaningful |

**Verdict:** M2 is functionally a **NO-OP for single-camera training**. The packed inference advantage (2.02×) does not translate to training because packed's compaction overhead and dense mode's simpler memory access cancel out.

**Recommendation:** Documentation only. Hard-drop packed/dense from optimization scope.

---

### Tier D — Validated but Non-Beneficial

#### M4 radius_clip
**Why it fails:** Removes inherently tiny Gaussians (both radii ≤ threshold), but these contribute negligible intersection workload. At quality-preserving thresholds (rclip ≤ 2.0), intersection reduction is <1%. **Performance benefit FALSIFIED.**

**Research value:** Positive negative result — explains why radius_clip doesn't help indoors.

#### M5 eps2d
**Why it fails:** Default eps2d=0.3 is at quality optimum. Reducing eps2d saves <2% intersections but costs up to 1.61 dB PSNR. **No quality-preserving configuration change possible.**

**Research value:** Positive result — confirms gsplat's defaults are well-chosen.

---

## 3. Complete Module Matrix

| Module | Evidence | Performance | Research Value | Composability |
|:-------|:---------|:-----------:|:--------------:|:-------------:|
| **M1 tile16** | ✅ PASS | ⏺ Baseline (reference) | N/A | Reference |
| **M1 tile32** | ✅ PASS (room) ❌ (cross-scene) | 📊 Scene-dependent (room: +58%, outdoor: −44%) | HIGH — tile-size generalization is fundamental | Conditional (indoor only) |
| **M2 packed/dense** | ✅ PASS | ⏺ NEUTRAL (1.03×) | USEFUL NEGATIVE | Not for performance |
| **M3 SH degree** | ✅ PASS | ⏺ NEUTRAL for performance (<4%) | HIGH — quality trade-off | Not for performance |
| **M4 radius_clip** | ✅ PASS | ❌ NOT BENEFICIAL | POSITIVE NEGATIVE | No |
| **M5 eps2d** | ✅ PASS | ❌ NOT BENEFICIAL | POSITIVE — confirms defaults | No |

---

## 4. Composability Assessment

### Question: Are there ≥2 meaningful candidates to compose?

**Answer: No.**

- **M1 tile32:** Only beneficial on room-like scenes. Not composable as a universal improvement.
- **M3 SH degree:** Not a performance knob. Composing SH0 with tile32 would compound scene-dependent effects without evidence of synergy.
- **M2 packed/dense:** Already a NO-OP. Composing adds complexity for zero benefit.
- **M4, M5:** Already non-beneficial.

### Recommendation:
Do not proceed to composability testing. The study does not have ≥2 modules with universal, meaningful, composable performance benefits.

---

## 5. Research Conclusion

### What the study found:

1. **Scene-dependent tile-size effect** (M1): tile32 helps indoor scenes (1.58×) but hurts outdoor scenes (0.69×). The mechanism is rooted in intersection sort workload vs rasterization granularity trade-off.

2. **Packed/dense NO-OP** (M2): The 2.02× inference advantage of packed mode does not carry over to training due to compaction overhead and optimizer state management.

3. **SH degree quality trade-off** (M3): Consistent across scenes but <4% performance impact — it's a quality knob, not a performance knob.

4. **Two negative results** (M4, M5): radius_clip and eps2d are not performance knobs at quality-preserving thresholds. Both are well-designed defaults in gsplat.

### What the study did NOT find:
- No module passed all gates as a universal performance candidate.
- No composable combination of ≥2 modules shows promise for additive or superadditive speedup.
- The search space of differentiable rendering optimizations within gsplat's public API is substantially characterized.

### Open question:
Could server-grade hardware (A100) change the tile-size preference? Phase 2 synthetic data (A100) showed tile32 was 1.42–3.93× faster on synthetic scenes. The RTX 5070 shows mixed results. **Hardware dependency** remains an open research dimension.

---

## 6. Data Files

- `results/epic05/phase12/phase12_candidate_ranking.json` — machine-readable ranking
- `reports/epic05/phase12_m1_cross_scene.md` — detailed M1 cross-scene analysis
- `reports/epic05/phase12_m3_cross_scene.md` — detailed M3 cross-scene analysis
- `results/epic05/phase12/phase12_bicycle.json` — bicycle raw data
- `results/epic05/phase12/phase12_garden.json` — garden raw data
