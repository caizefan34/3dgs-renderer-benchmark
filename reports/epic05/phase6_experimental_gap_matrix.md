# Phase 6+ Experimental Research Gap Matrix

**Date:** 2026-09-01 (updated)
**Project:** 3DGS Renderer Optimization and Differentiable Training Research Project
**Previous Phase:** Phase 6 Research Alignment Audit (2026-08-27)
**Phase:** 6+ Experimental Research

---

## I. Executive Summary

This document is the **living evidence matrix** for the Phase 6+ experimental research program. It supersedes the Phase 6 audit's static analysis with an **actionable, updatable experiment roadmap**.

**Current state:** The audit identified that **zero gradient correctness evidence exists** across all 9 optimization modules (M0–M9). Quality Layer B (renderer output vs real ground truth) is **blocked** — GT images not on local disk. Full training with densification/pruning/real GT is **not tested**.

**Current direct action:** Gradient correctness for tile_size (M1) is **VERIFIED SUPPORTED** on the local RTX 5070 Laptop GPU. EPIC-05 (A100) SSH connectivity is blocked (connection timeout). Next priority: GT quality (Layer B) and full training pipeline.

---

## II. Module Inventory

| ID | Name | Mechanism | Switch Type | Diff? | Forward Evidence | Backward Evidence | Gradient Evidence |
|:--:|------|-----------|:-----------:|:-----:|:----------------:|:-----------------:|:----------------:|
| M0 | Baseline (gsplat default) | Full differentiable pipeline | Reference | ✅ | SUPPORTED | SUPPORTED | SUPPORTED |
| M1 | tile_size | CUDA grid launch param | Runtime param | ✅ | SUPPORTED | SUPPORTED | **SUPPORTED** |
| M2 | packed/dense | Compaction kernel | Boolean | ✅ | INCONCLUSIVE | **SUPPORTED** | **SUPPORTED** |
| M3 | SH degree | SH coeff count | Integer | ✅ | PARTIAL | **SUPPORTED** | **SUPPORTED** |
| M4 | radius_clip | Screen-space radius filter | Float | ✅ | INCONCLUSIVE | **SUPPORTED** | **SUPPORTED** |
| M5 | eps2d | 2D covariance epsilon | Float | ✅ | INCONCLUSIVE | **SUPPORTED** | **SUPPORTED** |
| M6 | HiGS tile_size | Separate inference renderer | Separate class | ❌ | PARTIAL | N/A (inference) | N/A |
| M7 | HiGS SH compression | FP16/FP8 packing | Separate param | ❌ | PARTIAL | N/A (inference) | N/A |
| M8 | HiGS auto adapter | Scale-aware heuristic | Derived | ❌ | PARTIAL | N/A (inference) | N/A |
| M9 | TC-GS | Tensor Core alpha | External repo | ? | PARTIAL | NOT_TESTED | NOT_TESTED |

---

## III. Evidence Matrix (Per Module)

### M0 — Baseline (gsplat default)

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| Forward correctness | SUPPORTED | Reference configuration |
| Backward correctness | SUPPORTED | gsplat native backward kernels exist |
| Gradient correctness | **NOT_TESTED** | No gradcheck or finite-difference ever run |
| Quality Layer A | SUPPORTED | Self-comparison = bit-exact |
| Quality Layer B (vs GT) | **INCONCLUSIVE** | validate_quality.py exists but not run for M0 baseline against GT |
| Training simplified | SUPPORTED | Used as baseline in phase4/phase5 (5000 steps, room, random GT) |
| Training full | **NOT_TESTED** | No densification/pruning/real GT |
| Composability | SUPPORTED | Reference in 10 interaction experiments |

### M1 — tile_size

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| Forward correctness | **SUPPORTED** | tile16 vs tile32 bit-exact bicycle/garden; max 0.002 on room (FP reduction order) |
| Backward correctness | **SUPPORTED** | backward executes and produces gradients for all 5 param groups; tile16/tile32 indistinguishable |
| Gradient correctness | **SUPPORTED** | VERIFIED 2026-09-01: gradcheck PASS all 5 params; FD max_abs=0.0; tile16/tile32 grad norms within 5e-7 rel diff |
| Quality Layer A | **SUPPORTED** | Pixel equivalence confirmed |
| Quality Layer B (vs GT) | **BLOCKED** | GT images not on disk locally; pipeline exists but not executed |
| Training simplified | PARTIAL | 5000 steps room (random GT) — both tile16/tile32 tested |
| Training full | **NOT_TESTED** | Never run with densification/pruning |
| Composability | INCONCLUSIVE | Performance interactions only (no gradient gate) |

### M2 — packed/dense

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| Forward correctness | INCONCLUSIVE | No pixel equivalence published; ±0.5% perf diff |
| Backward correctness | NOT_TESTED | Not compared |
| Gradient correctness | NOT_TESTED | Not tested |
| Quality Layer A | NOT_TESTED | Not evaluated |
| Performance | NOT_SUPPORTED | ±0.5% — negligible |

### M3 — SH degree

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| Forward correctness | PARTIAL | Performance measured but quality not compared |
| Backward correctness | NOT_TESTED | spherical_harmonics_bwd_kernel exists but not verified per degree |
| Gradient correctness | NOT_TESTED | Not tested |
| Performance | NOT_SUPPORTED | ±0.3% — negligible |

### M4 — radius_clip

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| All layers | NOT_TESTED / INCONCLUSIVE | ±0.1% performance effect |
| Performance | NOT_SUPPORTED | Negligible effect |

### M5 — eps2d

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| All layers | NOT_TESTED / INCONCLUSIVE | ±0.1% performance effect |
| Performance | NOT_SUPPORTED | Negligible effect |

### M6 — HiGS tile_size

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| All gradient layers | **N/A** | **Inference-only renderer** — no backward pass, no gradient, cannot participate in training |
| Performance | SUPPORTED | Significantly faster than gsplat at high Gaussian counts |
| Differentiability | **FALSIFIED** | No backward kernels available — non-differentiable |

### M7 — HiGS SH compression

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| All gradient layers | **N/A** | **Inference-only renderer** — non-differentiable |
| Quality (SH16) | **FALSIFIED** | Min 49.79 dB PSNR — unacceptable quality degradation |
| Quality (SH32) | ACCEPTABLE | Min 64.85 dB — viable inference optimization |

### M8 — HiGS auto adapter

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| All gradient layers | **N/A** | Derived from M6+M7, both inference-only |

### M9 — TC-GS

| Evidence Layer | Status | Note |
|:--------------|:------:|------|
| Forward | PARTIAL | 10-frame smoke test only |
| Backward | NOT_TESTED | Not verified in this repository |
| Gradient | NOT_TESTED | Not tested |
| Differentiability | **UNKNOWN** | External repo — backward path existence not validated |

---

## IV. Evidence Gaps by Priority

### P0 — Critical (blocks all training contribution claims)

| Gap | Module | Current Status | Action Required |
|:---|:------:|:--------------:|-----------------|
| G0: tile_size gradient correctness | M1 | **SUPPORTED (2026-09-01)** | ✅ VERIFIED: gradcheck PASS + FD max_abs=0 |
| G1: All module gradient correctness | M2-M5, M9 | **NOT_TESTED** | Systematic check per module |
| G2: Finite-difference verification | M2-M5, M9 | **NOT_TESTED** | Extend FD verification |
| G3: Autograd gradcheck | M2-M5, M9 | **NOT_TESTED** | Extend gradcheck |

### P1 — High (training validation)

| Gap | Module | Current Status | Action Required |
|:---|:------:|:--------------:|-----------------|
| G4: Full training with densification | M1 | **NOT_TESTED** | Implement + run full 3DGS training |
| G5: Training with real GT | M1 | **NOT_TESTED** | Replace `create_random_gt()` with real images |
| G6: Full loss (L1 + D-SSIM) | M1 | **NOT_TESTED** | Use original 3DGS training loss |
| G7: Multi-scene training | M1 | **NOT_TESTED** | bicycle, garden, room |
| G8: Training quality vs GT | M1 | **NOT_TESTED** | Evaluate trained model on held-out views |

### P2 — Medium (quality validation)

| Gap | Module | Current Status | Action Required |
|:---|:------:|:--------------:|-----------------|
| G9: Layer B GT quality for tile16 | M1 | **BLOCKED** | Download GT images, run quality pipeline |
| G10: Layer B GT quality for tile32 | M1 | **BLOCKED** | Same as G9 with tile32 |
| G11: Quality gates documentation | M1 | DONE | Thresholds defined |

### P3 — Low (composability)

| Gap | Module | Current Status | Action Required |
|:---|:------:|:--------------:|-----------------|
| G12: tile32 + packed gradient | M1+M2 | **NOT_TESTED** | After G0 passes |
| G13: tile32 + SH degree | M1+M3 | **NOT_TESTED** | After G0 passes |

---

## V. Executable Experiments (Local RTX 5070 Laptop)

### Available Now (no dependencies)

| Experiment | Command | Estimated Time |
|:-----------|---------|:--------------:|
| ✅ Gradient correctness tile16 vs tile32 | `python scripts/epic05/gradient_correctness.py` | **DONE** |
| ✅ Gradient correctness dense mode | `python scripts/epic05/gradient_correctness.py --dense` | **DONE** |
| GT quality for tile_size (needs GT images) | `python scripts/epic05/evaluate_official_quality.py --all` | ~30 min/GPU |

### Blocked Experiments

| Experiment | Blocker | Resolution |
|:-----------|:--------|:-----------|
| Layer B GT quality | GT images not on local disk | Download Mip-NeRF 360 dataset (~3 GB per scene) |
| Full training | No training pipeline with densification | Need to implement or adapt existing 3DGS training code |

### EPIC-05 (A100) Experiments

| Experiment | Status | Note |
|:-----------|:------:|------|
| Gradient correctness on A100 | PENDING | Needs EPIC-05 SSH connectivity |
| GT quality on A100 | PENDING | Needs GT images on EPIC-05 or transfer |
| Full training on A100 | PENDING | Needs pipeline + data |

---

## VI. Training Pipeline Audit

### Current: SimplifiedTraining

```
SimpleGaussianModel (no densification, no pruning, no adaptive control)
  ↓
rasterization() forward
  ↓
MSE loss (not L1 + D-SSIM)
  ↓
loss.backward()
  ↓
Adam optimizer step
```

**Missing for "Full Training":**
- ❌ Densification (clone/split based on gradient magnitude)
- ❌ Pruning (remove low-opacity Gaussians)
- ❌ Adaptive Gaussian control (max Gaussians, reset interval)
- ❌ SH degree progression (start at SH0, increase during training)
- ❌ L1 + D-SSIM loss (MSE alone ≠ 3DGS paper)
- ❌ Real GT images (random tensor — PSNR meaningless)
- ❌ Validation split (train vs test cameras)
- ❌ Checkpointing
- ❌ Learning rate scheduling

### What Full Training Needs

A proper 3DGS training loop includes:
1. SfM initialization or random point cloud
2. Progressive training with densification every 100-1500 steps
3. Pruning based on opacity threshold (< 0.005)
4. Clone/split based on view-space gradient magnitude
5. SH degree 0→1→3 progression
6. Loss: L1 (λ=0.8) + D-SSIM (λ=0.2)
7. Adam with parameter-specific learning rates
8. Validation on held-out camera views
9. Full checkpointing

---

## VII. Research Questions and Hypotheses

| RQ | Question | Hypothesis | Experiment | Current Status |
|:---|:---------|:-----------|:-----------|:--------------:|
| RQ1 | Does tile32 preserve gradient correctness? | tile32 gradient matches baseline tile16 | finite diff + gradcheck | **SUPPORTED ✅** |
| RQ2 | Does tile32 preserve rendering quality vs GT? | tile32 PSNR/SSIM/LPIPS ≈ tile16 vs GT | GT quality pipeline | BLOCKED (no GT) |
| RQ3 | Does tile32 produce same training convergence as tile16? | Loss curves and final PSNR match | Simplified training with real GT | NOT TESTED |
| RQ4 | Do tile32 benefits survive full training with densification? | End-to-end training speedup matches simplified | Full training pipeline | NOT TESTED |
| RQ5 | Is tile32 composable with packed mode? | tile32+packed gradient matches tile16+packed | Pairwise composability | NOT TESTED |

---

## VIII. Decision Log

| Date | Decision | Rationale |
|:----|:---------|:----------|
| 2026-09-01 | Execute gradient correctness (M1) on local RTX 5070 first | P0 gap, highest information value, executable now |
| 2026-09-01 | Attempt EPIC-05 SSH connection for parallel A100 experiments | A100 has 80GB VRAM — can run larger scenes |
| 2026-09-01 | Prioritize gradient check over quality/training | If gradients fail, all other experiments for M1 are moot |

---

## IX. Output Directory Structure

```
results/epic05/
├── gradient/              ← P0: gradient correctness results (current)
├── quality/               ← P2: GT quality evaluation (future)
├── training/              ← P1: full training results (future)
├── composability/         ← P3: pairwise module tests (future)
└── research_alignment_matrix.json  ← Machine-readable evidence status
```

---

*This document should be updated after every experimental round.*
*Next scheduled update: after gradient correctness results are collected.*
