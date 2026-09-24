# Phase R2.1A — Geometry-Focused Oracle Validation

**Semantic label**: `REFERENCE_V1_ABSGRAD`  
**Date**: 2026-09-14  
**Scene**: mipnerf360/room  
**Renderer**: gsplat 1.5.3 (R2.1 patched, per-branch attribute masks)  
**Checkpoint**: iter_10000 (N=780,884)  
**GPU**: NVIDIA A100-PCIE-40GB  
**CUDA**: 11.5  
**Git**: 32ab80e

---

## Decision

```
Architecture:      ARCH_INVALIDATED
Dominant design:   FULL_ATTRIBUTE_REQUIRED
```

```
FULL E2E:                      54.60 ms
FULL_ATTRIBUTE_95 E2E:         54.07 ms
FULL_ATTRIBUTE_95 E2E gain:    0.97%

CURRENT-camera utility (pooled):
  geo = 0.950
  app = 0.950
  opa = 0.950

Does appearance gating add >=1% absolute E2E?
YES (+1.79% beyond GEO_OPACITY)
```

---

## Key Finding

R2.1A invalidates R2.1's `B_ARCH_STRONG` conclusion. When using **current-camera oracle masks** at **exactly 95% positive utility** (the correct oracle ceiling), the net E2E gain drops from 15.6% to 1.0%.

The bottleneck is not the utility-coverage tradeoff — it is the **mask mechanism overhead** (1.72ms, 7% of raster backward) from 3 shared-memory loads + 3 conditional branches per Gaussian, always incurred regardless of mask values.

Furthermore, oracle masks preferentially keep the **highest-utility Gaussians, which also have the most pixel intersections** (highest compute cost). Suppressing the low-utility remainder saves disproportionately little time — a fundamental correlation between utility and compute cost.

---

## 1. Baseline Recheck

| Metric | R2.1 | R2.1A | Delta |
|:-------|:----:|:-----:|:-----:|
| FULL raster backward | 24.69 ms | 24.71 ms | +0.1% |
| FULL E2E | 54.61 ms | 54.60 ms | -0.0% |
| ALL_ONES masked bwd | 26.51 ms | 26.43 ms | -0.3% |
| Mask overhead | 1.82 ms (7.4%) | 1.72 ms (7.0%) | -5.5% |

Result compatible (<5% deviation). No investigation needed.

---

## 2. CURRENT_CAMERA_ORACLE

### Per-Camera Keep Fractions (Minimum for >=95% Positive Utility)

| Camera | Vis Count | Geo Keep | App Keep | Opa Keep | Geo Util | App Util | Opa Util |
|:------:|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 10001 | 321,834 | 56.5% | 24.6% | 45.6% | 95.0% | 95.0% | 95.0% |
| 10002 | 228,237 | 53.5% | 22.4% | 42.9% | 95.0% | 95.0% | 95.0% |
| 10003 | 287,152 | 50.4% | 26.0% | 43.4% | 95.0% | 95.0% | 95.0% |
| 10004 | 115,664 | 56.4% | 30.8% | 46.0% | 95.0% | 95.0% | 95.0% |
| 10005 | 254,128 | 54.2% | 18.7% | 42.9% | 95.0% | 95.0% | 95.0% |
| **Mean** | **241,403** | **54.2%** | **24.5%** | **44.2%** | **95.0%** | **95.0%** | **95.0%** |

### Comparison with R2.1 Cross-Camera Masks

| Family | R2.1 Keep (cross-camera) | R2.1A Keep (current-camera) |
|:-------|:------------------------:|:---------------------------:|
| Geometry | 50% | 54.2% |
| Appearance | 20% | 24.5% |
| Opacity | 32% | 44.2% |

**Current-camera masks keep more Gaussians** because the utility estimation from the same camera is sharper and requires a larger top-k to reach 95%.

### Pooled Timing

| Metric | Value |
|:-------|:-----:|
| Raster backward | 24.08 ms |
| Raster speedup vs FULL | +2.6% |
| E2E | 53.72 ms |
| E2E gain vs FULL | +1.6% |

---

## 3. Main Comparison Table

| Configuration | Geo Util | App Util | Opa Util | Raster ms | Raster Gain | E2E ms | E2E Gain |
|:-------------|:--------:|:--------:|:--------:|:---------:|:-----------:|:------:|:--------:|
| **FULL** | 100% | 100% | 100% | 24.71 | — | 54.60 | — |
| **GEO_ONLY_95** | 95% | 100% | 100% | 25.37 | **–2.7%** | 55.23 | **–1.2%** |
| **GEO_OPACITY_95** | 95% | 100% | 95% | 25.01 | **–1.2%** | 55.04 | **–0.8%** |
| **FULL_ATTRIBUTE_95** | 95% | 95% | 95% | 24.17 | **+2.2%** | 54.07 | **+1.0%** |
| **CURRENT_CAMERA_ORACLE** | 95% | 95% | 95% | 24.08 | **+2.6%** | 53.72 | **+1.6%** |

**GEO_ONLY_95 and GEO_OPACITY_95 are SLOWER than FULL.** Only when ALL three branches are suppressed (FULL_ATTRIBUTE_95) do the combined savings barely exceed the fixed mask overhead.

---

## 4. Synthetic Geometry Scaling

| Configuration | Geo Keep | Raster Bwd | Δ from FULL |
|:-------------|:--------:|:----------:|:-----------:|
| FULL | 100% | 24.71 ms | — |
| GEO_SYNTH_K75 | 75% | 26.00 ms | +1.29 ms |
| GEO_SYNTH_K50 | 50% | 24.91 ms | +0.20 ms |
| GEO_SYNTH_K32 | 32% | 24.13 ms | –0.58 ms |
| GEO_SYNTH_K20 | 20% | 23.63 ms | –1.08 ms |

Note: Synthetic masks use **random** Gaussian selection (not oracle). They show a clearer speedup trend because they also suppress high-cost Gaussians. At K20 (80% suppressed), the random baseline saves only 1.08ms.

---

## 5. Mask Overhead Analysis

### Cost Breakdown

| Source | Overhead |
|:-------|:--------:|
| 3 shared-memory loads (app_mask_batch, geo_mask_batch, opacity_mask_batch) | ~1 load-cycle each per warp |
| 3 conditional branches per Gaussian per pixel | ~1-2 cycles each |
| **Total overhead** | **1.72 ms (7.0% of raster backward)** |

This overhead is **always incurred**, regardless of mask values. Even with all-1s masks, the kernel must load the masks from shared memory and evaluate the branches.

### Net Savings vs ALL_ONES Baseline

| Configuration | Bwd Δ from ALL_ONES | Interpreted as |
|:--------------|:-------------------:|:--------------|
| GEO_ONLY_95 | –1.06 ms | Saving from suppressing 46% of geo branch |
| GEO_OPACITY_95 | –1.43 ms | Saving from geo + opacity branch suppression |
| FULL_ATTRIBUTE_95 | –2.27 ms | Saving from all three branch suppressions |

The gross savings are meaningful (up to 2.27ms), but 1.72ms of that is eaten by the overhead.

### Optimization Opportunity

A single packed mask (3 bits in 1 byte) would replace 3 shared-memory loads + 3 branches with ~1 load + bitwise extract. Estimated overhead reduction: ~60% (from 1.72ms to ~0.7ms).

At 0.7ms overhead, GEO_ONLY_95 would net +0.4ms savings (~1.6% raster speedup). FULL_ATTRIBUTE_95 would net +1.6ms savings (~6.5% raster speedup, ~3% E2E).

---

## 6. Why R2.1's 15.6% Was Wrong

R2.1's `ORACLE_DECOUPLED_95` (15.6% raster speedup) used:

1. **Cross-camera masks** — 1 camera's utility used to construct masks for 5 cameras. Effective utility was only 89-93%.
2. **Lower keep fractions** — Because utility didn't transfer well, masks were more aggressive (geo=50%, app=20%, opa=32%).
3. **No utility constraint** — The masks operated at lower effective utility, which is not a fair comparison.

R2.1A fixes this by:
1. **Current-camera masks** — Each camera uses its own utility (95% exact).
2. **Higher keep fractions** — geo=54.2%, app=24.5%, opa=44.2% at exact 95%.
3. **Utility confirmed at exactly 95.0%** per family per camera.

The speedup drops from 15.6% to 1.0% because of the **correlation between utility and compute cost**: oracle masks keep the high-pixel-count Gaussians (which dominate compute time). The low-utility remainder contributes very little to total runtime. Suppressing 50% of Gaussians by utility rank saves much less time than suppressing 50% randomly.

---

## 7. Decision Logic Evaluation

### Architecture Gate

| Criterion | Required | Measured | Pass? |
|:----------|:--------:|:--------:|:-----:|
| Utility per family | ≥95% | 95.0% | ✅ |
| E2E gain | ≥5% | **1.0%** | ❌ |
| E2E gain (weakened) | ≥2% | **1.0%** | ❌ |

**Gate: ARCH_INVALIDATED** — E2E gain (1.0%) is below both the STRONG (5%) and WEAKENED (2%) thresholds.

### Dominant Design

| Condition | Met? |
|:----------|:----:|
| GEO_ONLY captures ≥80% of FULL_ATTR's gain? | No (GEO_ONLY is negative) |
| GEO_OPACITY captures ≥90% of FULL_ATTR's gain AND opacity adds ≥0.5%? | No (both negative) |
| Appearance gating adds ≥1% absolute E2E beyond GEO_OPACITY? | **Yes (+1.79%)** |

**Design: FULL_ATTRIBUTE_REQUIRED** — Only when all three branches are suppressed does the kernel overcome the mask overhead.

---

## 8. Important Limitation

> The mask mechanism overhead (1.72ms from 3 shared-memory loads + 3 conditional branches per Gaussian) dominates the savings from per-branch attribute gating. A single packed-mask implementation would substantially improve this, but was not tested — the current result measures the actual R2.1 kernel implementation, not a theoretical optimal design.

---

## Deliverables

| File | Description |
|------|-------------|
| `reports/phase-r2.1a-geometry-focused-oracle.md` | **Main report** |
| `results/reference_v1/r2.1a/current_camera_oracle.json` | Per-camera oracle analysis |
| `results/reference_v1/r2.1a/geo_only_oracle_95.json` | Geometry-only oracle benchmarking |
| `results/reference_v1/r2.1a/geo_opacity_oracle_95.json` | Geometry + opacity oracle benchmarking |
| `results/reference_v1/r2.1a/full_attribute_oracle_95.json` | Full three-family oracle benchmarking |
| `results/reference_v1/r2.1a/mask_overhead.json` | Mask mechanism overhead breakdown |
| `results/reference_v1/r2.1a/final_decision.json` | Final decision gates + verdict |
| `results/reference_v1/r2.1a/r21a_benchmark.json` | Raw benchmark data |
| `experiments/r2_profile_only/r21a_benchmark.py` | Benchmark script |
| `baseline/r2/r21a_analysis.py` | Analysis script |
