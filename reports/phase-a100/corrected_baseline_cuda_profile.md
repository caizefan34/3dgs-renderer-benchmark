# Corrected A100 Baseline CUDA Profile

**Date:** 2026-09-06  
**Scale activation:** CORRECT (`torch.exp(scales)` applied throughout)  
**Environment:** A100 PCIe 40GB (SM80)

---

## Scale Activation Verification

The profiling pipeline uses activated scales throughout:

```
checkpoint → scales (log-space, mean ≈ -6.36)
           → torch.exp(scales) → activated (mean ≈ 0.022)
           → fully_fused_projection(..., activated_scales, ...)
           → quat_scale_to_covar_preci: S = diag(activated_scale)
           → covar = R × S × S × R^T
```

**CUDA kernel does NOT apply `torch.exp` internally** — confirmed by reading
`quat_scale_to_covar_preci` source in `Utils.cuh`.

---

## Profiling Protocol

| Parameter | Value |
|:----------|:------|
| Tile size | 16 |
| Warmup rounds | 20 |
| Measured rounds | 50 |
| Timing method | `torch.cuda.Event(enable_timing=True)` |
| Sort isolation | `isect_tiles(sort=True)` − `isect_tiles(sort=False)` |
| Repetitions | 2 independent batches (A, B) |
| GPU | A100 PCIe 40GB, SM80 |
| Camera | Real scene camera (cam0) |
| Scenes | room, bicycle, garden (Mip-NeRF 360) |

---

## Summary Table

| Scene | N_total | N_visible | N_isect | Sort(ms) | Isect Gen(ms) | Fwd(ms) | Sort% | Isect Gen% | Raster% |
|:------|:-------:|:---------:|:-------:|:--------:|:-------------:|:-------:|:-----:|:----------:|:-------:|
| room | 1,105,873 | 16,076 | 25,475,247 | 3.096 | 10.726 | 14.609 | 21.2% | 73.4% | 2.1% |
| bicycle | 2,589,484 | 13,654 | 18,550,853 | 2.654 | 6.667 | 10.764 | 24.7% | 61.9% | 8.2% |
| garden | 874,019 | 100,582 | 11,257,252 | 1.643 | 2.044 | 5.096 | 32.2% | 40.1% | 21.1% |

**Isect Gen = pass1 + cumsum + pass2 (intersection without sorting)**  
**Sort = CUB radix sort only (isect_with_sort − isect_no_sort)**  
**Fwd = projection + SH + isect_with_sort + offset + rasterization**

---

## Per-Stage Breakdown

### room

| Stage | Median (ms) | Share | Description |
|:------|:-----------:|:-----:|:------------|
| projection | 0.193 | 1.3% | Fused covar + projection |
| SH evaluation | 0.096 | 0.7% | Spherical harmonics |
| Intersect no sort | 10.726 | 73.4% | pass1 + cumsum + pass2 |
| CUB radix sort | 3.096 | 21.2% | `radix_sort_double_buffer` |
| Intersect with sort | 13.822 | 94.6% | Total intersect pipeline |
| Offset encode | 0.193 | 1.3% | Tile id → offset |
| Rasterization | 0.306 | 2.1% | Alpha compositing |
| **Total forward** | **14.609** | **100%** | |

### bicycle

| Stage | Median (ms) | Share | Description |
|:------|:-----------:|:-----:|:------------|
| projection | 0.314 | 2.9% | |
| SH evaluation | 0.096 | 0.9% | |
| Intersect no sort | 6.667 | 61.9% | pass1 + cumsum + pass2 |
| CUB radix sort | 2.654 | 24.7% | |
| Intersect with sort | 9.321 | 86.6% | |
| Offset encode | 0.152 | 1.4% | |
| Rasterization | 0.881 | 8.2% | |
| **Total forward** | **10.764** | **100%** | |

### garden

| Stage | Median (ms) | Share | Description |
|:------|:-----------:|:-----:|:------------|
| projection | 0.175 | 3.4% | |
| SH evaluation | 0.052 | 1.0% | |
| Intersect no sort | 2.044 | 40.1% | pass1 + cumsum + pass2 |
| CUB radix sort | 1.643 | 32.2% | |
| Intersect with sort | 3.686 | 72.3% | |
| Offset encode | 0.108 | 2.1% | |
| Rasterization | 1.075 | 21.1% | |
| **Total forward** | **5.096** | **100%** | |

---

## CUB Radix Sort Details

| Property | Value |
|:---------|:------|
| Call site | `radix_sort_double_buffer` in `IntersectTile.cu` (line ~10567) |
| CUB API | `cub::DeviceRadixSort::SortPairs` |
| Key type | `int64_t` (isect_ids: `image_id|tile_id|depth`) |
| Value type | `int32_t` (flatten_ids) |
| end_bit | `32 + tile_n_bits + image_n_bits` |
| begin_bit | 0 |
| Double buffer | `cub::DoubleBuffer<int64_t>`, `cub::DoubleBuffer<int32_t>` |
| Temporary storage | `c10::cuda::CUDACachingAllocator` |
| CUB version | `CUB_VERSION=101500` (CCCL 1.5.10) |
| Policy | Policy800 (confirmed in Phase 2) |
| Radix bits/pass | 8 |
| Pass count | room: **6** (end_bit=48), bicycle: **7** (end_bit=49), garden: **7** (end_bit=50) |
| Measurement | `isect_tiles(sort=True) − isect_tiles(sort=False)`, median of 50 rounds |

### Sort Analysis by Scene

| Scene | N_isect | end_bit | Passes | Sort(ms) | ns/isect |
|:------|:-------:|:------:|:------:|:--------:|:--------:|
| room | 25,475,247 | 48 | 6 | 3.096 | **0.122** |
| bicycle | 18,550,853 | 49 | 7 | 2.654 | **0.143** |
| garden | 11,257,252 | 50 | 7 | 1.643 | **0.146** |

**ns/isect varies** (0.12-0.15) — not constant as previously claimed. Variation is partly
due to different pass counts (room=6 passes, bicycle/garden=7 passes).

---

## Timing Stability

| Scene | Sort diff (A vs B) | Fwd diff (A vs B) | Status |
|:------|:------------------:|:-----------------:|:-------|
| room | 0.0% | 0.0% | ✅ STABLE |
| bicycle | 0.3% | 0.1% | ✅ STABLE |
| garden | 0.1% | 0.1% | ✅ STABLE |

**Overall: PASS** — timing is highly reproducible.

---

## N_isect Validation

All three scenes:

- `N_isect` = sum of `tiles_per_gaussian` from GPU intersect kernel
- Verified by single-camera setup (I=1, packed mode)
- **PASS** for all scenes

---

## Bottleneck Classification

| Scene | Sort% | Isect Gen% | Combined% | Bottleneck |
|:------|:----:|:----------:|:---------:|:-----------|
| room | 21.2% | **73.4%** | 94.6% | **INTERSECTION_GENERATION_DOMINANT** |
| bicycle | 24.7% | **61.9%** | 86.6% | **INTERSECTION_GENERATION_DOMINANT** |
| garden | 32.2% | **40.1%** | 72.3% | **INTERSECTION_GENERATION_DOMINANT** |

**True bottleneck: INTERSECTION_GENERATION_DOMINANT**

The combined intersection pipeline (pass1 + cumsum + pass2 + CUB sort) accounts for
72-95% of forward pass. However, within this pipeline, **intersection generation**
(pass1 + cumsum + pass2) is the dominant component at 40-73%, while **CUB sort**
is 21-32%.

---

## Key Differences from Invalidated Phase 1-2

| Metric | Phase 1-2 (WRONG) | Corrected | Change |
|:-------|:-----------------:|:---------:|:------:|
| N_isect (room) | ~12B | 25.5M | **471× smaller** |
| N_isect (bicycle) | ~47B | 18.6M | **2,527× smaller** |
| N_isect (garden) | ~3B | 11.3M | **265× smaller** |
| Sort share | 65-68% | **21-32%** | Sort is NOT dominant |
| Intersection gen share | Not measured | **40-73%** | THIS is the bottleneck |
| Rasterization share | Not isolated | 2-21% | Minor for room/bicycle |
| ns/isect | 0.12 (constant) | **0.12-0.15** | Varies with pass count |

---

## Final Output

```
=== Corrected A100 Baseline CUDA Profile ===

Scale activation:
CORRECT

Scene          N_visible   N_isect          Sort(ms)   Forward(ms)    Sort Share
room           16,076      25,475,247       3.096      14.609         21.2
bicycle        13,654      18,550,853       2.654      10.764         24.7
garden         100,582     11,257,252       1.643      5.096          32.2

CUB radix sort:
REAL CUDA EVENT TIMING = YES
  Method: sort=True - sort=False (CUB isolation)
  6-7 passes (end_bit=48-50), 8 bits/pass
  ns/isect: 0.122-0.146 (varies with pass count)

Sort bottleneck:
INTERSECTION_GENERATION_DOMINANT
  Intersection generation (pass1+cumsum+pass2): 40-73% of forward
  CUB radix sort: 21-32% of forward
  Combined intersect pipeline: 72-95% of forward
  Rasterization: 2-21% (minor for high-isect scenes)

N_isect validation:
PASS
  room = 25,475,247
  bicycle = 18,550,853
  garden = 11,257,252

Timing stability:
PASS (A vs B < 0.3%)

Cross-hardware comparison:
NOT PERFORMED

Research status:
BASELINE_REESTABLISHED

OPTIMIZATION:
NOT STARTED
```

---

## Output Files

| File | Status |
|:-----|:-------|
| `reports/phase-a100/corrected_baseline_cuda_profile.md` | ✅ This report |
| `results/phase-a100/corrected_baseline_cuda_profile.json` | ✅ Full JSON data |
| `@corrected_baseline_cuda_profile.py` | ✅ Profiling script |
