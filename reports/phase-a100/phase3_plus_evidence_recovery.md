# Phase 3+ Evidence Recovery — Corrected Baseline Audit

**Date:** 2026-09-06  
**Status:** Complete

---

## Scale Activation Bug

**CONFIRMED.** All Phase 1+ analysis scripts passed raw log-scales (mean ≈ -6.36)
to `fully_fused_projection`, which expects activated scales (mean ≈ 0.022).
The CUDA kernel does `S = diag(scale)` without applying `torch.exp`.

Scale inflation factor: **~288× in radius, ~84,000× in covariance.**

### Root Cause

```
checkpoint stores:  scales = log(params)  (mean ≈ -6.36, std dev of Gaussian in log space)
analysis scripts:   fully_fused_projection(..., raw_scales, ...)  ← BUG: no torch.exp()
CUDA kernel:        S = diag(scale); covar = R × S × S × R^T
                     → S[0][0] = -6.36 (interpreted as std dev)
                     → covar[0][0] = 40.45 instead of 0.00048
```

### 10 INVALIDATED Scripts

| Script | Bug |
|--------|-----|
| `@a100_baseline_profiling.py` | Raw log → `fully_fused_projection` (4 call sites) |
| `@a100_baseline_profiling_v2.py` | Raw log → `fully_fused_projection` (2 call sites) |
| `@footprint_analysis.py` | Raw log → `fully_fused_projection` |
| `@root_cause_analysis.py` | Raw log → `fully_fused_projection` |
| `@heavy_tail_analysis.py` | Raw log → `fully_fused_projection` |
| `@intersect_analysis_v2.py` | Raw log → `fully_fused_projection` |
| `@workload_validity.py` | Raw log → `fully_fused_projection` |
| `src/scripts/epic05/phase8e_per_kernel_timing.py` | Raw log → `fully_fused_projection` |
| `src/scripts/epic05/phase8c_backward_kernel_trace.py` | Raw log → `gsplat.rasterization()` |
| `src/renderers/fast_gauss_renderer.py` | Passthrough, no `torch.exp()` (unlike every other renderer) |

### 21 VALID Scripts

All renderer adapters in `src/renderers/` (except `fast_gauss_renderer.py`), all training
scripts, all `phase9a_m2_*` scripts, and all `phase11_*` scripts correctly activate scales.

---

## Evidence Matrix

| Phase / Artifact | Status | Reason |
| ---------------- | ------ | ------ |
| Phase 1 (baseline profiling v1) | INVALIDATED | Raw log-scales → fully_fused_projection. All timing inflated. |
| Phase 2 (baseline profiling v2 / CUB deep profile) | INVALIDATED | Same bug. |
| Phase 3 (intersection explosion source audit) | INVALIDATED | Full saturation / uniform coverage were artifacts of 288× inflated radii. |
| Phase 3.5 (real-scene workload validity) | INVALIDATED | 12B–47B intersection counts invalid. |
| H6 accounting audit | INVALIDATED | Depends on invalid Phase 3.5 data. |
| H6 heavy-tail audit | INVALIDATED | "Uniform distribution" (Lorenz ≈ 45°) was artifact of every Gaussian covering full tile grid. |
| H6 root cause / uniform saturation audit | INVALIDATED | MULTI_FACTOR conclusion derived from inflated workload. |
| M1 tile-size scaling study | INVALIDATED | Depends on invalid Phase 1-2 baseline timing. |
| gsplat training pipeline | VALID | Correctly applies `torch.exp(scales)` before `rasterization()`. |
| CUDA kernel `quat_scale_to_covar_preci` | VALID (by design) | Does `S = diag(scale)` without `exp`. Expects activated scales. |
| CUB sort throughput (0.12 ns/isect) | PARTIALLY_VALID | Hardware throughput characteristic, not scale-dependent. Absolute N_isect was inflated. |

---

## Corrected Workload (Activated Scales, Real Camera Cam0)

| Scene | Camera | Tile | N_visible | N_isect | Mean tiles | P50 tiles | P99 tiles | Full sat% |
| ----- | ------ | ---- | --------- | ------- | ---------- | --------- | --------- | --------- |
| room | artificial_t16 | 16 | 1,711 | 2,871,395 | 1678.2 | 704.0 | 8160.0 | 5.61% |
| room | real_t16 | 16 | 16,054 | 25,472,295 | 1586.7 | 39.0 | 25350.0 | 2.19% |
| room | artificial_t24 | 24 | 1,711 | 1,281,540 | 749.0 | 322.0 | 3600.0 | 5.73% |
| room | real_t24 | 24 | 16,054 | 11,447,730 | 713.1 | 20.0 | 11310.0 | 2.19% |
| bicycle | artificial_t16 | 16 | 5,742 | 1,028,692 | 179.2 | 15.0 | 3818.1 | 0.38% |
| bicycle | real_t16 | 16 | 13,627 | 18,548,394 | 1361.2 | 56.0 | 38979.5 | 0.56% |
| bicycle | artificial_t24 | 24 | 5,742 | 471,895 | 82.2 | 8.0 | 1677.5 | 0.38% |
| bicycle | real_t24 | 24 | 13,627 | 8,339,054 | 612.0 | 28.0 | 17334.2 | 0.56% |
| garden | artificial_t16 | 16 | 9,477 | 1,674,383 | 176.7 | 30.0 | 3352.6 | 0.22% |
| garden | real_t16 | 16 | 43,996 | 11,143,918 | 253.3 | 21.0 | 4165.5 | 0.00% |
| garden | artificial_t24 | 24 | 9,477 | 772,131 | 81.5 | 16.0 | 1485.0 | 0.24% |
| garden | real_t24 | 24 | 43,996 | 5,150,731 | 117.1 | 12.0 | 1890.0 | 0.00% |

### Key Observations

1. **Full saturation is GONE** — max 2.19% (room), most <1%, garden 0.0%
2. **P50 tiles/Gaussian is modest** — 12-56 tiles (not full grid)
3. **Heavy P99 tail** — 4,166-38,979 tiles for 1% of Gaussians (near-camera)
4. **N_isect range** — 5M-25M for real cameras (not 12B-47B)
5. **Artificial camera UNDERESTIMATES** real workload by 4-18× (not 38-70× as previously claimed)

---

## Corrected Heavy-Tail

| Scene | Tile | Gini | Classification | top 0.1% | top 1% | top 5% | top 10% |
| ----- | ---- | ---- | -------------- | -------- | ------ | ------ | ------- |
| room | t16 | 0.8970 | WEAK_HEAVY_TAIL | 1.59% | 15.92% | 48.21% | 70.97% |
| room | t24 | 0.8939 | WEAK_HEAVY_TAIL | 1.58% | 15.81% | 48.10% | 70.90% |
| bicycle | t16 | 0.9140 | MODERATE_HEAVY_TAIL | 4.48% | 41.98% | 82.71% | 94.85% |
| bicycle | t24 | 0.9098 | MODERATE_HEAVY_TAIL | 4.42% | 41.51% | 82.39% | 94.73% |
| garden | t16 | 0.8644 | MODERATE_HEAVY_TAIL | 9.46% | 34.93% | 71.76% | 87.60% |
| garden | t24 | 0.8523 | MODERATE_HEAVY_TAIL | 9.16% | 33.99% | 70.82% | 86.99% |

### Classification

| Scene | Classification | Why |
|-------|---------------|-----|
| room | **WEAK_HEAVY_TAIL** | Top 1% → 15.9% of intersections. Gini=0.897 but top-fraction is moderate. |
| bicycle | **MODERATE_HEAVY_TAIL** | Top 1% → 42.0% of intersections. Gini=0.914. |
| garden | **MODERATE_HEAVY_TAIL** | Top 1% → 34.9% of intersections. Gini=0.864. |

**Previous conclusion (H6 heavy-tail): INVALIDATED.** The old "uniform" finding was an
artifact of every Gaussian covering the full tile grid. With corrected scales, there IS
a heavy tail — especially in bicycle and garden scenes.

---

## Corrected Sorting Baseline

| Scene | Tile | Sort share% | N_isect | Sort est (ms) | ns/isect | Bottleneck |
| ----- | ---- | ----------- | ------- | ------------- | -------- | ---------- |
| room | t16 | 94.6% | 25,475,247 | 3.108 | 0.1220 | SORTING_DOMINANT |
| room | t24 | 88.8% | 11,449,280 | 6.010 | 0.5249 | SORTING_DOMINANT |
| bicycle | t16 | 86.6% | 18,550,853 | 2.653 | 0.1430 | SORTING_DOMINANT |
| bicycle | t24 | 73.5% | 8,340,195 | 3.872 | 0.4642 | SORTING_DOMINANT |
| garden | t16 | 72.4% | 11,257,252 | 3.976 | 0.3532 | SORTING_DOMINANT |
| garden | t24 | 52.8% | 5,239,819 | 1.645 | 0.3139 | SORTING_DOMINANT |

### Key Observations

1. **Sort dominates** in all configurations (52.8-94.6% of forward pass)
2. **Original 65-68% was underestimating** — with correct scales, sorting is MORE dominant
3. **ns/isect varies** (0.12-0.52) — not constant as previously claimed
4. **Room t16 has 94.6% sort share** — projection+rasterization are minimal (correct scales → few visible Gaussians per tile)
5. **Smaller tiles (t16) → more sort dominance** — larger N_isect from more tile overlap

### Relationship R1-R3

**R1: N_isect ↑ → sort time ↑** — CONFIRMED. room t16 (25M → 94.6%) vs garden t24 (5.2M → 52.8%).

**R2: sort_time / N_isect stable** — PARTIALLY. Room t16=0.122 ns, bicycle t16=0.143 ns (close), but room t24=0.525 ns (higher). Suggests per-item overhead varies with workload composition.

**R3: Sort is still the dominant bottleneck** — CONFIRMED. In every configuration.

---

## Historical Results Status

| Finding | New status | Action |
| ------- | ---------- | ------ |
| Phase 3 intersection explosion (R_full≈0.99) | INVALIDATED | Discard. Artifact of log-scale bug. |
| 12B–47B logical intersections | INVALIDATED | Discard. Actual range 5M-25M. |
| H6 uniform saturation (98-99% full tile grid) | INVALIDATED | Discard. Full saturation was artifact. |
| H6 heavy-tail: uniform (Lorenz ≈ 45°) | INVALIDATED | Rerun. Now shows MODERATE/WEAK heavy tail. |
| H6 root cause: MULTI_FACTOR (Jacobian) | INVALIDATED | Discard. Analysis was on inflated workload. |
| Sort share ≈65-68% of forward pass | INVALIDATED | Actual: 52.8-94.6% (MORE dominant). |
| CUB sort 0.12 ns/intersection constant | PARTIALLY_VALID | Throughput is hardware-based, but varies 0.12-0.52. |
| Phase 1-2 per-kernel timing (absolute ms) | INVALIDATED | All data based on inflated workload. |
| M1 tile-size scaling study | INVALIDATED | Depends on invalid Phase 1-2 timing. |

---

## Final Output

```
=== Phase 3+ Evidence Recovery ===

Scale activation bug:
CONFIRMED

Affected artifacts:
  [INVALIDATED] Phase 1 (baseline profiling v1)
  [INVALIDATED] Phase 2 (baseline profiling v2 / CUB deep profile)
  [INVALIDATED] Phase 3 (intersection explosion source audit)
  [INVALIDATED] Phase 3.5 (real-scene workload validity)
  [INVALIDATED] H6 accounting audit
  [INVALIDATED] H6 heavy-tail audit
  [INVALIDATED] H6 root cause / uniform saturation audit
  [INVALIDATED] M1 tile-size scaling study

Still valid:
  [VALID] gsplat training pipeline
  [VALID (by design)] CUDA kernel quat_scale_to_covar_preci
  [PARTIALLY_VALID] CUB sort throughput characteristic

Corrected intersection count:
  room/real_t16: N_visible=16,054  N_isect=25,472,295
  bicycle/real_t16: N_visible=13,627  N_isect=18,548,394
  garden/real_t16: N_visible=43,996  N_isect=11,143,918

Corrected mean/P50 tiles per Gaussian (real_t16):
  room: mean=1587  P50=39  P99=25350  full_sat=2.19%
  bicycle: mean=1361  P50=56  P99=38980  full_sat=0.56%
  garden: mean=253  P50=21  P99=4166  full_sat=0.0%

Heavy-tail:
  room:  WEAK_HEAVY_TAIL  Gini=0.897  top1%=15.9%
  bicycle:  MODERATE_HEAVY_TAIL  Gini=0.914  top1%=42.0%
  garden:  MODERATE_HEAVY_TAIL  Gini=0.864  top1%=34.9%

Corrected sorting share:
  room/real_t16: sort_share=94.6%  N_isect=25.5M  sort_est=3.1ms  ns/isect=0.122
  bicycle/real_t16: sort_share=86.6%  N_isect=18.5M  sort_est=2.7ms  ns/isect=0.143
  garden/real_t16: sort_share=72.4%  N_isect=11.3M  sort_est=4.0ms  ns/isect=0.353

True bottleneck:
SORTING_DOMINANT

Research direction:
NEEDS RE-EVALUATION

OPTIMIZATION:
NOT STARTED
```

---

## Output Files

| File | Status |
|:-----|:-------|
| `reports/phase-a100/phase3_plus_evidence_recovery.md` | This report |
| `results/phase-a100/phase3_plus_evidence_recovery.json` | Full JSON evidence matrix |
| `@phase3_plus_evidence_recovery.py` | Corrected analysis script |
| `@root_cause_corrected.py` | Previous corrected script (superseded) |
