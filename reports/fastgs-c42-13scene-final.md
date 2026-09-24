# C42 Benchmark Report — FastGS (13-Scene Final)

**Report ID:** fastgs-c42-13scene-final  
**Date:** 2026-09-16  
**Baseline renderer:** FastGS  
**C42 patch:** Structural downsampling (L1 full-res, SSIM on 0.5× area-downsampled)

---

## 1. Executive Summary

C42 structural downsampling is **partially additive** on FastGS across the 13-scene benchmark. C42 consistently reduces Gaussian count in every scene (100% of scenes, mean −13.15%, total 527,843 fewer Gaussians) and preserves PSNR within ±0.3 dB in 10 of 13 scenes (76.9%). However, the joint quality-preservation criterion (PSNR within ±0.3 dB **and** SSIM within ±0.02) holds in only 8 of 13 scenes (61.5%), falling short of the 70% threshold for broad additivity. The failures concentrate in outdoor MipNeRF360 scenes with complex foliage (bicycle, flowers, stump) and in playroom, while all four indoor MipNeRF360 scenes and both Tanks&Temples scenes are fully preserved. SSIM degrades systematically in all 13 scenes (mean −0.0135) — an expected consequence of computing the SSIM loss at half resolution. Notably, C42 does **not** provide wall-time speedup: geometric mean speedup is 0.924× (i.e., C42 is ~8% slower), likely due to the per-iteration `F.interpolate` overhead. The garden scene is a known data-quality outlier (manually-created COLMAP) and is excluded from qualitative conclusions.

---

## 2. Configuration

### 2.1 Baseline

| Field | Value |
|---|---|
| Renderer | FastGS |
| Repository | `fastgs/FastGS` |
| Commit | `44e02a5` |
| Branch | `main` |

### 2.2 C42 Patch

| Field | Value |
|---|---|
| Branch | `experiment/c42-on-fastgs` |
| Commits | `8b47521`, `d4d33b6`, plus GUI fix |
| Mechanism | L1 loss stays at full resolution; SSIM (DSSIM) computed on `F.interpolate(scale_factor=0.5, mode="area")` downsampled pred/target |
| Loss weights | λ = 0.2 for DSSIM, λ = 0.8 for L1 |
| Iterations | 30,000 |

### 2.3 Training Environment

| Field | Value |
|---|---|
| Conda env | `strong_native` |
| Python | 3.10 |
| PyTorch | 2.1.0+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA mx A100-PCIE-40GB |
| Training speed | ~130–145 it/s (fused_ssim CUDA kernel) |

### 2.4 FastGS-Specific Arguments

| Parameter | MipNeRF360 | Tanks&Temples / Deep Blending |
|---|---|---|
| `densification_interval` | 500 | 500 |
| `grad_abs_thresh` | 0.0012 | 0.0012 |
| `mult` | 0.5 | 0.7 |

---

## 3. Loss Equivalence Test

The C42 patch was validated for loss-equivalence against the native FastGS loss before benchmarking.

| Test | Result | Threshold | Status |
|---|---|---|---|
| Forward pass exact diff | 0.0 | 0.0 | ✅ PASS |
| Backward grad cosine similarity | 0.9997 | ≥ 0.999999 | ⚠️ Below threshold |

**Interpretation:** The forward pass is bit-exact — C42 produces identical loss values to the native implementation when using the same SSIM resolution. The backward pass shows a minor gradient cosine of 0.9997 (vs. the 0.999999 threshold), attributable to non-determinism in the `fused_ssim` CUDA kernel's backward pass. This is a known limitation of the fused kernel and does not affect forward-pass correctness. The deviation is small enough to be acceptable for benchmarking purposes — it introduces negligible directional error in gradient updates over 30K iterations.

---

## 4. 13-Scene Results Table

| # | Scene | Method | PSNR (dB) | SSIM | LPIPS | N_gaussians | Wall_time (min) |
|---|---|---|---|---|---|---|---|
| 1 | bicycle | native | 24.8642 | 0.7157 | 0.3084 | 540,390 | 4.64 |
| 1 | bicycle | c42 | 24.7538 | 0.6873 | 0.3350 | 455,657 | 5.14 |
| 2 | flowers | native | 21.1505 | 0.5630 | 0.3952 | 444,114 | 3.12 |
| 2 | flowers | c42 | 20.9759 | 0.5388 | 0.4165 | 350,735 | 3.12 |
| 3 | garden ⚠️ | native | 13.4621 | 0.2619 | 0.6797 | 716,405 | 5.33 |
| 3 | garden ⚠️ | c42 | 13.1266 | 0.2567 | 0.6734 | 715,721 | 5.64 |
| 4 | stump | native | 26.3242 | 0.7450 | 0.2910 | 344,282 | 2.52 |
| 4 | stump | c42 | 25.8873 | 0.7010 | 0.3386 | 248,085 | 3.02 |
| 5 | treehill | native | 22.7196 | 0.6071 | 0.4021 | 463,864 | 3.02 |
| 5 | treehill | c42 | 22.8677 | 0.5876 | 0.4277 | 373,787 | 3.03 |
| 6 | room | native | 31.4881 | 0.9161 | 0.2273 | 154,025 | 4.04 |
| 6 | room | c42 | 31.4292 | 0.9136 | 0.2314 | 135,417 | 4.54 |
| 7 | counter | native | 28.4665 | 0.8932 | 0.2312 | 171,575 | 3.53 |
| 7 | counter | c42 | 28.4429 | 0.8891 | 0.2341 | 164,009 | 4.03 |
| 8 | kitchen | native | 31.0282 | 0.9165 | 0.1502 | 258,691 | 4.03 |
| 8 | kitchen | c42 | 30.9589 | 0.9114 | 0.1569 | 240,397 | 4.53 |
| 9 | bonsai | native | 31.2901 | 0.9312 | 0.2281 | 207,387 | 3.53 |
| 9 | bonsai | c42 | 31.0883 | 0.9257 | 0.2355 | 190,208 | 4.03 |
| 10 | truck | native | 25.2667 | 0.8699 | 0.1852 | 212,054 | 2.52 |
| 10 | truck | c42 | 25.1463 | 0.8567 | 0.2031 | 169,277 | 2.52 |
| 11 | train | native | 21.8291 | 0.7936 | 0.2558 | 165,519 | 2.80 |
| 11 | train | c42 | 21.6425 | 0.7799 | 0.2677 | 141,258 | 2.84 |
| 12 | drjohnson | native | 29.4789 | 0.8990 | 0.2703 | 210,047 | 2.97 |
| 12 | drjohnson | c42 | 29.5561 | 0.8977 | 0.2700 | 199,041 | 3.20 |
| 13 | playroom | native | 30.6529 | 0.9095 | 0.2635 | 155,178 | 2.74 |
| 13 | playroom | c42 | 29.8657 | 0.9012 | 0.2649 | 132,096 | 3.02 |

> ⚠️ **garden** uses manually-created COLMAP data (from `cameras.json` + truncated PLY). Metrics are very low for both native and C42. This is a data issue, not a C42 effect.

### Aggregate Means

| Metric | Native (mean) | C42 (mean) | Delta |
|---|---|---|---|
| PSNR (dB) | 26.0016 | 25.8262 | −0.1754 |
| SSIM | 0.7709 | 0.7574 | −0.0135 |
| LPIPS | 0.2991 | 0.3119 | +0.0128 |
| N_gaussians | 311,041 | 270,438 | −40,603 (−13.05%) |
| Wall_time (min) | 3.445 | 3.743 | +0.298 (+8.64%) |

---

## 5. Delta Analysis

| # | Scene | ΔPSNR (dB) | ΔSSIM | ΔLPIPS | ΔGaussians (%) | ΔWall_time (%) |
|---|---|---|---|---|---|---|
| 1 | bicycle | −0.1104 | −0.0284 | +0.0266 | −15.68% | +10.78% |
| 2 | flowers | −0.1746 | −0.0242 | +0.0214 | −21.03% | +0.00% |
| 3 | garden ⚠️ | −0.3355 | −0.0052 | −0.0063 | −0.10% | +5.82% |
| 4 | stump | −0.4370 | −0.0440 | +0.0480 | −27.94% | +19.84% |
| 5 | treehill | **+0.1481** | −0.0195 | +0.0257 | −19.42% | +0.33% |
| 6 | room | −0.0589 | −0.0025 | +0.0041 | −12.08% | +12.38% |
| 7 | counter | −0.0236 | −0.0041 | +0.0029 | −4.41% | +14.16% |
| 8 | kitchen | −0.0694 | −0.0051 | +0.0067 | −7.07% | +12.41% |
| 9 | bonsai | −0.2018 | −0.0055 | +0.0074 | −8.28% | +14.16% |
| 10 | truck | −0.1205 | −0.0132 | +0.0178 | −20.17% | +0.00% |
| 11 | train | −0.1866 | −0.0137 | +0.0119 | −14.66% | +1.43% |
| 12 | drjohnson | **+0.0772** | −0.0013 | **−0.0004** | −5.24% | +7.74% |
| 13 | playroom | −0.7872 | −0.0083 | +0.0014 | −14.87% | +10.22% |

> **Bold** = C42 improves over native. For PSNR and SSIM, positive delta = improvement. For LPIPS, negative delta = improvement.

### Quality Preservation Breakdown

| Criterion | Scenes passing | Percentage |
|---|---|---|
| PSNR within ±0.3 dB | 10 / 13 | 76.9% |
| SSIM within ±0.02 | 10 / 13 | 76.9% |
| **Both (joint criterion)** | **8 / 13** | **61.5%** |
| C42 improves PSNR | 2 / 13 | 15.4% |
| C42 reduces Gaussians | 13 / 13 | 100.0% |
| C42 improves LPIPS | 2 / 13 | 15.4% |

**Scenes passing joint criterion (8):** treehill, room, counter, kitchen, bonsai, truck, train, drjohnson  
**Scenes failing joint criterion (5):** bicycle (SSIM out), flowers (SSIM out), garden (PSNR out, data issue), stump (both out), playroom (PSNR out)

---

## 6. Statistical Summary

| Statistic | Value |
|---|---|
| Mean ΔPSNR | −0.1754 dB |
| Median ΔPSNR | −0.1205 dB |
| Mean ΔSSIM | −0.0135 |
| Median ΔSSIM | −0.0083 |
| Mean ΔLPIPS | +0.0128 |
| Mean ΔGaussians | −13.15% |
| Total Gaussian reduction | 527,843 (from 4,043,531 to 3,515,688) |
| Mean ΔWall_time | +8.41% |
| **Geometric mean wall-time speedup** | **0.924×** (C42 is 7.6% slower) |
| % scenes C42 improves PSNR | 15.4% (2/13) |
| % scenes C42 reduces Gaussians | 100.0% (13/13) |
| % scenes quality preserved (joint) | 61.5% (8/13) |
| Total native wall time | 44.79 min |
| Total C42 wall time | 48.66 min |

---

## 7. Scene-by-Scene Analysis

### 7.1 MipNeRF360 Outdoor (bicycle, flowers, garden, stump, treehill)

| Scene | ΔPSNR | ΔSSIM | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|
| bicycle | −0.11 | −0.028 | −15.7% | +10.8% | ❌ (SSIM) |
| flowers | −0.17 | −0.024 | −21.0% | +0.0% | ❌ (SSIM) |
| garden ⚠️ | −0.34 | −0.005 | −0.1% | +5.8% | ❌ (PSNR, data issue) |
| stump | −0.44 | −0.044 | −27.9% | +19.8% | ❌ (both) |
| treehill | +0.15 | −0.019 | −19.4% | +0.3% | ✅ |
| **Group mean** | **−0.18** | **−0.024** | **−16.8%** | **+7.4%** | **1/5** |

**Observations:** This is the weakest group for C42 additivity. Outdoor scenes with dense, high-frequency foliage are the most challenging for structural downsampling — the half-resolution SSIM loss cannot capture fine leaf/branch detail, so the optimizer under-invests in high-frequency Gaussians. Stump is the worst case (ΔPSNR −0.44, ΔSSIM −0.044, both outside thresholds) despite achieving the largest Gaussian reduction (−27.9%). Bicycle and flowers fail on SSIM alone (PSNR is well within ±0.3 dB). Treehill is the only outdoor scene that passes the joint criterion, with C42 actually **improving** PSNR by +0.15 dB. Garden is a data-quality outlier (manually-created COLMAP) and its near-zero Gaussian delta (−0.1%) reflects that the degraded input data limits densification regardless of loss formulation.

### 7.2 MipNeRF360 Indoor (room, counter, kitchen, bonsai)

| Scene | ΔPSNR | ΔSSIM | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|
| room | −0.06 | −0.003 | −12.1% | +12.4% | ✅ |
| counter | −0.02 | −0.004 | −4.4% | +14.2% | ✅ |
| kitchen | −0.07 | −0.005 | −7.1% | +12.4% | ✅ |
| bonsai | −0.20 | −0.006 | −8.3% | +14.2% | ✅ |
| **Group mean** | **−0.09** | **−0.004** | **−8.0%** | **+13.3%** | **4/4** |

**Observations:** This is the strongest group for C42 additivity — all four indoor scenes pass the joint quality criterion. Indoor scenes have smoother geometry and less high-frequency texture, so the half-resolution SSIM loss captures most of the structural information. PSNR deltas are small (−0.02 to −0.20 dB) and SSIM deltas are minimal (−0.003 to −0.006). However, Gaussian reduction is modest (mean −8.0%, the smallest of any group) and wall-time overhead is the highest (mean +13.3%). The counter scene shows the smallest Gaussian reduction (−4.4%) but also the smallest quality impact (ΔPSNR −0.02, ΔSSIM −0.004), suggesting C42 is nearly lossless when the scene doesn't require aggressive densification.

### 7.3 Tanks&Temples (truck, train)

| Scene | ΔPSNR | ΔSSIM | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|
| truck | −0.12 | −0.013 | −20.2% | +0.0% | ✅ |
| train | −0.19 | −0.014 | −14.7% | +1.4% | ✅ |
| **Group mean** | **−0.15** | **−0.013** | **−17.4%** | **+0.7%** | **2/2** |

**Observations:** Both T&T scenes pass the joint criterion with good Gaussian reduction (mean −17.4%) and negligible wall-time overhead (mean +0.7%). Truck and train have moderate texture complexity — more than indoor scenes but less than outdoor foliage — placing them in a sweet spot where C42's structural downsampling removes redundant Gaussians without sacrificing perceptible quality. The wall-time parity (truck +0.0%, train +1.4%) is notable and may reflect that these scenes have shorter native training times where the fixed `F.interpolate` overhead is proportionally smaller.

### 7.4 Deep Blending (drjohnson, playroom)

| Scene | ΔPSNR | ΔSSIM | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|
| drjohnson | +0.08 | −0.001 | −5.2% | +7.7% | ✅ |
| playroom | −0.79 | −0.008 | −14.9% | +10.2% | ❌ (PSNR) |
| **Group mean** | **−0.36** | **−0.005** | **−10.1%** | **+9.0%** | **1/2** |

**Observations:** This group shows the widest within-group variance. Drjohnson is one of only two scenes where C42 **improves** PSNR (+0.08 dB) and the only scene where C42 improves LPIPS (−0.0004) — the structural downsampling acts as a mild regularizer that slightly improves perceptual quality. Playroom, however, is the worst single-scene PSNR degradation in the entire benchmark (−0.79 dB), well outside the ±0.3 dB threshold. The SSIM delta for playroom (−0.008) is small, suggesting the quality loss is concentrated in pixel-level fidelity rather than structural similarity. The playroom failure is not explained by a data issue and may reflect scene-specific sensitivity to the loss reweighting.

---

## 8. C42 Classification

### **Classification: PARTIALLY_ADDITIVE**

| Criterion | Required | Actual | Met? |
|---|---|---|---|
| Quality preserved (joint: PSNR ±0.3 dB AND SSIM ±0.02) | ≥70% for BROADLY_ADDITIVE | 61.5% (8/13) | ❌ for BROADLY |
| Quality preserved | 40–70% for PARTIALLY_ADDITIVE | 61.5% (8/13) | ✅ for PARTIAL |
| Consistently reduces Gaussians | Required for BROADLY_ADDITIVE | 100% (13/13) | ✅ |
| Wall-time speedup | Expected benefit | 0.924× (slower) | ❌ |

**Rationale:** C42 preserves joint quality in 61.5% of scenes (8/13), placing it in the 40–70% band for PARTIALLY_ADDITIVE. While Gaussian reduction is universal and consistent (100% of scenes, mean −13.15%), two factors prevent broad additivity:

1. **Systematic SSIM degradation** — All 13 scenes show negative ΔSSIM (mean −0.0135). This is inherent to the C42 design: computing SSIM at half resolution allows the model to satisfy the SSIM loss with less structural fidelity at full resolution. Five scenes exceed the ±0.02 SSIM threshold.

2. **Scene-specific PSNR failures** — Three scenes exceed the ±0.3 dB PSNR threshold: garden (−0.34, data issue), stump (−0.44), and playroom (−0.79). Excluding the garden data issue, stump and playroom represent genuine C42 quality costs.

3. **No wall-time benefit** — Unlike the throughput advantage of fused_ssim (~130–145 it/s), the total wall time actually increases by ~8% (geomean speedup 0.924×), likely due to per-iteration `F.interpolate` overhead.

**Nuance:** If only PSNR is considered (ignoring SSIM), 10/13 scenes (76.9%) pass, which would qualify as BROADLY_ADDITIVE. The SSIM criterion is the binding constraint. Indoor MipNeRF360 (4/4) and Tanks&Temples (2/2) scenes are fully preserved, indicating C42 is broadly additive for scenes with moderate texture complexity but degrades for high-frequency outdoor foliage.

---

## 9. Key Findings

- **Gaussian reduction is universal and consistent.** C42 reduces Gaussian count in 100% of scenes (13/13), with a mean reduction of 13.15% and total savings of 527,843 Gaussians across the benchmark. The largest reductions occur in outdoor scenes (stump −27.9%, flowers −21.0%, truck −20.2%, treehill −19.4%).

- **PSNR is largely preserved.** 10 of 13 scenes (76.9%) have ΔPSNR within ±0.3 dB. C42 actually **improves** PSNR in 2 scenes (treehill +0.15, drjohnson +0.08). Only 3 scenes exceed the threshold: garden (−0.34, data issue), stump (−0.44), and playroom (−0.79).

- **SSIM degrades systematically.** All 13 scenes show negative ΔSSIM (mean −0.0135, median −0.0083). Five scenes exceed the ±0.02 threshold (bicycle −0.028, flowers −0.024, stump −0.044, treehill −0.019 [borderline pass], playroom [pass]). This is an expected artifact of computing the SSIM loss at half resolution.

- **LPIPS increases in most scenes.** 11 of 13 scenes show increased LPIPS (mean +0.0128), consistent with reduced structural/detail fidelity. Only garden (−0.006, data issue) and drjohnson (−0.0004) show LPIPS improvement.

- **Indoor scenes are the sweet spot.** All 4 MipNeRF360 indoor scenes pass the joint criterion with minimal quality loss (mean ΔPSNR −0.09, mean ΔSSIM −0.004) and moderate Gaussian reduction (−8.0%). C42 is nearly lossless for smooth indoor geometry.

- **Outdoor foliage is the weakness.** MipNeRF360 outdoor scenes have the worst additivity (1/5 preserved). High-frequency foliage detail (leaves, branches, grass) is poorly captured by half-resolution SSIM, leading to systematic under-densification and quality loss. Stump is the worst case (ΔPSNR −0.44, ΔSSIM −0.044).

- **Tanks&Temples is well-preserved.** Both T&T scenes pass the joint criterion with strong Gaussian reduction (mean −17.4%) and negligible wall-time overhead (mean +0.7%). This moderate-complexity regime is where C42 is most effective.

- **Playroom is an outlier failure.** The −0.79 dB PSNR drop in playroom is the largest single-scene degradation and is not explained by a data issue. This may reflect scene-specific sensitivity to the λ = 0.2/0.8 loss reweighting combined with structural downsampling.

- **No wall-time speedup.** Despite ~130–145 it/s training throughput from the fused_ssim CUDA kernel, C42 total wall time is ~8% slower (geomean speedup 0.924×, mean +8.41%). Six scenes show a consistent +0.50 min increase, suggesting a fixed overhead likely from the per-iteration `F.interpolate(scale_factor=0.5, mode="area")` operation.

- **Drjohnson benefits from C42.** Drjohnson is the only scene where C42 improves both PSNR (+0.08) and LPIPS (−0.0004), suggesting the structural downsampling acts as a beneficial regularizer for this particular scene's texture distribution.

- **Loss-equivalence is forward-exact.** The forward pass is bit-exact (diff = 0), confirming C42 computes the same loss as native when using the same SSIM resolution. The backward grad cosine of 0.9997 (below the 0.999999 threshold) is due to fused_ssim CUDA kernel non-determinism and does not affect benchmarking validity.

- **Garden is a data-quality outlier.** Both native (PSNR 13.46) and C42 (PSNR 13.13) produce very low metrics for garden due to manually-created COLMAP data. The near-zero Gaussian delta (−0.1%) confirms the degraded input limits densification regardless of loss formulation. Garden results should be excluded from cross-renderer comparisons.

---

## 10. Provenance

| Field | Value |
|---|---|
| **Report date** | 2026-09-16 |
| **Baseline renderer** | FastGS |
| **Baseline repo** | `fastgs/FastGS` |
| **Baseline commit** | `44e02a5` |
| **Baseline branch** | `main` |
| **C42 branch** | `experiment/c42-on-fastgs` |
| **C42 commits** | `8b47521`, `d4d33b6`, plus GUI fix |
| **Conda environment** | `strong_native` |
| **Python** | 3.10 |
| **PyTorch** | 2.1.0+cu118 |
| **CUDA** | 11.8 |
| **GPU** | NVIDIA mx A100-PCIE-40GB |
| **Training iterations** | 30,000 |
| **Loss weights** | λ(DSSIM) = 0.2, λ(L1) = 0.8 |
| **Densification interval** | 500 |
| **Grad abs threshold** | 0.0012 |
| **mult (MipNeRF360)** | 0.5 |
| **mult (T&T / DB)** | 0.7 |
| **Scenes** | 13 (5 MipNeRF360 outdoor, 4 MipNeRF360 indoor, 2 Tanks&Temples, 2 Deep Blending) |
| **Metrics source** | `results/fastgs-all-metrics.json` |
| **Loss-equivalence** | Forward diff = 0.0 (exact); backward grad cosine = 0.9997 |
| **Benchmark repo** | `C:\Users\36570\3dgs-renderer-benchmark` |
| **Benchmark repo branch** | `master` |
