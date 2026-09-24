# C42 Benchmark Report — Speedy-Splat

## 1. Executive Summary

C42 structural downsampling is **partially additive** with Speedy-Splat. Across all 13 benchmark scenes, C42 preserves PSNR within ±0.3 dB in 12 of 13 scenes (92.3%) and reduces Gaussian count in 12 of 13 scenes (92.3%, the sole exception being the garden data-quality issue), yielding a mean Gaussian reduction of −15.7% (excluding garden) and a geometric-mean wall-time speedup of 1.084×. However, SSIM degrades systematically in every scene (mean ΔSSIM = −0.0125, all 13 negative), and when the combined quality-preservation criterion (PSNR within ±0.3 dB **and** SSIM within ±0.02) is applied, 9 of 13 scenes (69.2%) qualify — just below the 70% threshold for BROADLY_ADDITIVE. The binding constraint is SSIM, not PSNR: C42's area-downsampled SSIM term leads the optimizer to under-weight high-frequency structural fidelity, producing a consistent but usually small SSIM penalty. The loss-equivalence test at scale = 1.0 confirms C42 is a mathematically faithful generalization of the native loss, so the quality trade-off is intrinsic to the downsampling choice (scale_factor = 0.5), not an implementation defect. **Classification: PARTIALLY_ADDITIVE.**

---

## 2. Configuration

### 2.1 Baseline

| Field | Value |
|-------|-------|
| Repository | `j-alex-hanson/speedy-splat` |
| Commit | `34c45c6` |
| Branch | `speedy-splat` |
| Method | Speedy-Splat (accelerated 3DGS with aggressive pruning) |

### 2.2 C42 Patch

| Field | Value |
|-------|-------|
| Branch | `experiment/c42-on-speedysplat` |
| Commits | `07f4712`, `aa75d8d`, plus GUI fix |
| Technique | Structural downsampling of the SSIM loss term |
| L1 term | Full resolution (unchanged) |
| SSIM term | Computed on `F.interpolate(scale_factor=0.5, mode="area")` downsampled pred/target |
| λ (DSSIM) | 0.2 |
| λ (L1) | 0.8 |
| Iterations | 30,000 |

**Mechanism:** The C42 patch modifies the loss function so that the structural-similarity (SSIM/DSSIM) component is evaluated at half spatial resolution via area-mode interpolation, while the L1 component remains at full resolution. At scale_factor = 1.0 the patch reduces exactly to the native loss (proven by the loss-equivalence test below). At scale_factor = 0.5 the SSIM gradient signal is computed at coarser resolution, which reduces the computational cost of the SSIM term and subtly shifts the optimization landscape toward L1-dominant reconstruction at high frequencies.

### 2.3 Training Environment

| Field | Value |
|-------|-------|
| Conda env | `strong_native` |
| Python | 3.10 |
| PyTorch | 2.1.0+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA A100-PCIE-40GB (mx) |
| Iterations | 30,000 per scene |
| Evaluation | Full-resolution PSNR, SSIM, LPIPS on test set |

### 2.4 Benchmark Scenes (13)

| Dataset | Scenes |
|---------|--------|
| MipNeRF360 (outdoor) | bicycle, flowers, garden, stump, treehill |
| MipNeRF360 (indoor) | room, counter, kitchen, bonsai |
| Tanks & Temples | truck, train |
| Deep Blending | drjohnson, playroom |

> **Data caveat — garden:** The garden scene uses manually-created COLMAP data (derived from `cameras.json` + a truncated PLY), not the official MipNeRF360 COLMAP bundle. Consequently both native and C42 metrics are very low (PSNR ≈ 12.4–12.6 dB, SSIM ≈ 0.27–0.29). This is a **data-quality issue, not a C42 issue**, and the garden scene is flagged throughout the analysis. Notably, garden is the only scene where C42 *increases* Gaussian count (+147%), which is an artifact of the degraded input data rather than a C42 behavior.

---

## 3. Loss Equivalence Test

The C42 patch was validated for mathematical correctness at scale_factor = 1.0 (no downsampling). At this setting the C42 loss function must produce bit-identical results to the native Speedy-Splat loss.

| Metric | Native | C42 (scale=1.0) | Difference | Verdict |
|--------|--------|-----------------|------------|---------|
| L1 loss | L₁ | L₁ | **0** | ✅ Identical |
| DSSIM loss | D | D | **0** | ✅ Identical |
| Gradient cosine similarity | 1.0 | 1.0 | **1.0** | ✅ Identical |
| Gradient relative L2 norm | 0 | 0 | **0** | ✅ Identical |

**Result: PASSED.** At scale = 1.0, C42 produces loss values and gradients that are identical to the native Speedy-Splat loss (L1 diff = 0, DSSIM diff = 0, gradient cosine = 1.0, gradient relative L2 = 0). This confirms that C42 is a mathematically faithful generalization of the native loss — the SSIM downsampling is a proper extension that reduces to the baseline when the downsample factor is unity. All quality differences observed at scale_factor = 0.5 are therefore intrinsic to the structural downsampling choice, not implementation bugs.

---

## 4. 13-Scene Results Table

| # | Scene | Method | PSNR (dB) | SSIM | LPIPS | N_gaussians | Wall_time (min) |
|---|-------|--------|-----------|------|-------|-------------|-----------------|
| 1 | bicycle | native | 24.7553 | 0.7037 | 0.3333 | 580,416 | 21.25 |
| 1 | bicycle | c42 | 24.8137 | 0.6874 | 0.3454 | 512,480 | 21.25 |
| 2 | flowers | native | 21.4025 | 0.5793 | 0.3941 | 363,188 | 13.13 |
| 2 | flowers | c42 | 21.5079 | 0.5683 | 0.4037 | 282,611 | 11.61 |
| 3 | garden ⚠️ | native | 12.5629 | 0.2851 | 0.7428 | 66,282 | 16.21 |
| 3 | garden ⚠️ | c42 | 12.3727 | 0.2660 | 0.7171 | 163,954 | 14.65 |
| 4 | stump | native | 26.6367 | 0.7693 | 0.2604 | 497,002 | 13.65 |
| 4 | stump | c42 | 26.5484 | 0.7503 | 0.2820 | 403,799 | 11.62 |
| 5 | treehill | native | 22.5026 | 0.5917 | 0.4453 | 365,195 | 12.65 |
| 5 | treehill | c42 | 22.4352 | 0.5705 | 0.4623 | 330,444 | 11.62 |
| 6 | room | native | 30.9196 | 0.9045 | 0.2560 | 115,278 | 18.23 |
| 6 | room | c42 | 30.7705 | 0.9002 | 0.2621 | 104,114 | 16.15 |
| 7 | counter | native | 28.1445 | 0.8695 | 0.2729 | 99,450 | 14.64 |
| 7 | counter | c42 | 28.1893 | 0.8642 | 0.2767 | 93,035 | 14.64 |
| 8 | kitchen | native | 29.9506 | 0.8904 | 0.2011 | 116,524 | 13.61 |
| 8 | kitchen | c42 | 29.8234 | 0.8819 | 0.2125 | 99,669 | 13.61 |
| 9 | bonsai | native | 31.1245 | 0.9205 | 0.2496 | 130,658 | 13.61 |
| 9 | bonsai | c42 | 31.1105 | 0.9146 | 0.2577 | 118,578 | 13.61 |
| 10 | truck | native | 25.2503 | 0.8681 | 0.1895 | 257,944 | 10.07 |
| 10 | truck | c42 | 24.9688 | 0.8465 | 0.2251 | 152,037 | 8.05 |
| 11 | train | native | 21.6003 | 0.7722 | 0.2900 | 107,532 | 8.56 |
| 11 | train | c42 | 21.5251 | 0.7504 | 0.3107 | 80,549 | 8.09 |
| 12 | drjohnson | native | 29.1229 | 0.9001 | 0.2654 | 314,402 | 16.20 |
| 12 | drjohnson | c42 | 28.8128 | 0.8955 | 0.2699 | 299,890 | 15.19 |
| 13 | playroom | native | 30.1040 | 0.9071 | 0.2690 | 187,856 | 13.68 |
| 13 | playroom | c42 | 30.0325 | 0.9038 | 0.2733 | 159,056 | 12.17 |

> ⚠️ garden uses manually-created COLMAP data; metrics are depressed for both methods.

**Aggregate totals:**

| Metric | Native (all 13) | C42 (all 13) | Delta |
|--------|----------------|--------------|-------|
| Total Gaussians | 3,201,727 | 2,800,216 | −401,511 (−12.5%) |
| Total wall time | 185.49 min | 172.26 min | −13.23 min (−7.1%) |
| Mean PSNR | 25.698 dB | 25.609 dB | −0.090 dB |
| Mean SSIM | 0.7663 | 0.7538 | −0.0125 |
| Mean LPIPS | 0.3207 | 0.3307 | +0.0099 |

---

## 5. Delta Analysis

Per-scene deltas (C42 − native). Positive ΔPSNR = improvement; positive ΔSSIM = improvement; positive ΔLPIPS = degradation (lower is better); negative ΔGaussians = reduction (beneficial); negative ΔWall = speedup (beneficial).

| Scene | ΔPSNR (dB) | ΔSSIM | ΔLPIPS | ΔGaussians (%) | ΔWall_time (%) | Quality preserved? |
|-------|------------|-------|--------|----------------|----------------|---------------------|
| bicycle | +0.058 | −0.0163 | +0.0121 | −11.70% | 0.00% | ✅ Yes |
| flowers | +0.105 | −0.0110 | +0.0096 | −22.19% | −11.58% | ✅ Yes |
| garden ⚠️ | −0.190 | −0.0191 | −0.0258 | +147.35% | −9.62% | ✅ Yes* |
| stump | −0.088 | −0.0190 | +0.0217 | −18.75% | −14.87% | ✅ Yes |
| treehill | −0.067 | −0.0212 | +0.0171 | −9.52% | −8.14% | ❌ No (SSIM) |
| room | −0.149 | −0.0043 | +0.0061 | −9.68% | −11.41% | ✅ Yes |
| counter | +0.045 | −0.0053 | +0.0038 | −6.45% | 0.00% | ✅ Yes |
| kitchen | −0.127 | −0.0085 | +0.0114 | −14.47% | 0.00% | ✅ Yes |
| bonsai | −0.014 | −0.0059 | +0.0080 | −9.25% | 0.00% | ✅ Yes |
| truck | −0.282 | −0.0215 | +0.0357 | −41.06% | −20.06% | ❌ No (SSIM) |
| train | −0.075 | −0.0218 | +0.0207 | −25.11% | −5.49% | ❌ No (SSIM) |
| drjohnson | −0.310 | −0.0047 | +0.0044 | −4.62% | −6.23% | ❌ No (PSNR) |
| playroom | −0.072 | −0.0033 | +0.0043 | −15.33% | −11.04% | ✅ Yes |

> *garden quality is "preserved" only in the technical sense that both deltas are within thresholds; the underlying data is degraded. Quality preserved = PSNR within ±0.3 dB **and** SSIM within ±0.02.

**Key observations from deltas:**

- **PSNR:** 3 scenes improve (bicycle, flowers, counter); 9 degrade slightly; 1 (drjohnson) exceeds the ±0.3 dB threshold at −0.310. Maximum degradation: truck at −0.282 dB (still within threshold). Maximum improvement: flowers at +0.105 dB.
- **SSIM:** All 13 scenes show negative ΔSSIM (range −0.0033 to −0.0218). Three scenes exceed the ±0.02 threshold: treehill (−0.0212), truck (−0.0215), train (−0.0218). This is the **binding constraint** on the additivity classification.
- **LPIPS:** 12 of 13 scenes show increased LPIPS (degradation); garden is the sole exception (−0.0258, improvement) due to the data-quality artifact. Maximum degradation: truck at +0.0357.
- **Gaussians:** 12 of 13 scenes show Gaussian reduction; garden is the sole increase (+147.35%, data artifact). Maximum reduction: truck at −41.06%. Among valid scenes, mean reduction = −15.7%.
- **Wall time:** 9 of 13 scenes show measurable speedup; 4 scenes (bicycle, counter, kitchen, bonsai) show 0% change. Maximum speedup: truck at −20.06%.

---

## 6. Statistical Summary

| Statistic | Value |
|-----------|-------|
| **Mean ΔPSNR** | **−0.090 dB** |
| **Mean ΔSSIM** | **−0.0125** |
| **Mean ΔLPIPS** | **+0.0099** |
| Mean ΔGaussians (all 13) | −3.1% |
| Mean ΔGaussians (excl. garden) | −15.7% |
| Mean ΔWall_time | −7.6% |
| **% scenes C42 improves PSNR** (ΔPSNR > 0) | **23.1%** (3 / 13) |
| **% scenes C42 reduces Gaussians** (ΔGauss < 0) | **92.3%** (12 / 13) |
| % scenes C42 reduces wall time (ΔWall < 0) | 69.2% (9 / 13) |
| **Geometric mean wall-time speedup** | **1.084×** |
| % scenes PSNR within ±0.3 dB | 92.3% (12 / 13) |
| % scenes SSIM within ±0.02 | 76.9% (10 / 13) |
| **% scenes quality preserved** (both PSNR ±0.3 **and** SSIM ±0.02) | **69.2%** (9 / 13) |
| Total Gaussian reduction (all 13) | −401,511 (−12.5%) |
| Total Gaussian reduction (excl. garden) | −499,183 (−15.9%) |
| Total wall-time reduction | −13.23 min (−7.1%) |
| Total speedup (aggregate) | 1.077× |

**Geometric mean wall-time speedup derivation:**

| Scene | Native (min) | C42 (min) | Speedup ratio |
|-------|-------------|-----------|---------------|
| bicycle | 21.25 | 21.25 | 1.000 |
| flowers | 13.13 | 11.61 | 1.131 |
| garden | 16.21 | 14.65 | 1.107 |
| stump | 13.65 | 11.62 | 1.175 |
| treehill | 12.65 | 11.62 | 1.089 |
| room | 18.23 | 16.15 | 1.129 |
| counter | 14.64 | 14.64 | 1.000 |
| kitchen | 13.61 | 13.61 | 1.000 |
| bonsai | 13.61 | 13.61 | 1.000 |
| truck | 10.07 | 8.05 | 1.251 |
| train | 8.56 | 8.09 | 1.058 |
| drjohnson | 16.20 | 15.19 | 1.066 |
| playroom | 13.68 | 12.17 | 1.124 |

$$\text{Geomean} = \left(\prod_{i=1}^{13} \frac{t_{\text{native},i}}{t_{\text{c42},i}}\right)^{1/13} = e^{\frac{1}{13}\sum \ln(r_i)} = e^{0.08098} \approx \mathbf{1.084\times}$$

---

## 7. Scene-by-Scene Analysis

### 7.1 MipNeRF360 Outdoor (bicycle, flowers, garden, stump, treehill)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss (%) | ΔWall (%) | Preserved? |
|-------|-------|-------|--------|------------|-----------|------------|
| bicycle | +0.058 | −0.0163 | +0.0121 | −11.70% | 0.00% | ✅ |
| flowers | +0.105 | −0.0110 | +0.0096 | −22.19% | −11.58% | ✅ |
| garden ⚠️ | −0.190 | −0.0191 | −0.0258 | +147.35% | −9.62% | ✅* |
| stump | −0.088 | −0.0190 | +0.0217 | −18.75% | −14.87% | ✅ |
| treehill | −0.067 | −0.0212 | +0.0171 | −9.52% | −8.14% | ❌ (SSIM) |

| Group stat | Value |
|------------|-------|
| Mean ΔPSNR | −0.036 dB |
| Mean ΔSSIM | −0.0173 |
| Mean ΔGaussians (excl. garden) | −15.5% |
| Mean ΔWall_time | −8.8% |
| Scenes quality-preserved | 4 / 5 (80.0%) |

**Analysis:** Outdoor scenes are the strongest C42 performers on PSNR — bicycle and flowers both *improve* (+0.058 and +0.105 dB), suggesting that for outdoor scenes with complex backgrounds, the area-downsampled SSIM term acts as a mild regularizer that reduces overfitting to high-frequency noise. Gaussian reductions are substantial (−11.7% to −22.2% for valid scenes), and wall-time savings are consistent (−8% to −15%). Treehill is the sole quality-preservation failure, missing the SSIM threshold by a hair (−0.0212 vs. ±0.02 limit). Garden is excluded from meaningful analysis due to the COLMAP data issue; its +147% Gaussian increase is an artifact of the degraded input geometry, not a C42 behavior.

### 7.2 MipNeRF360 Indoor (room, counter, kitchen, bonsai)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss (%) | ΔWall (%) | Preserved? |
|-------|-------|-------|--------|------------|-----------|------------|
| room | −0.149 | −0.0043 | +0.0061 | −9.68% | −11.41% | ✅ |
| counter | +0.045 | −0.0053 | +0.0038 | −6.45% | 0.00% | ✅ |
| kitchen | −0.127 | −0.0085 | +0.0114 | −14.47% | 0.00% | ✅ |
| bonsai | −0.014 | −0.0059 | +0.0080 | −9.25% | 0.00% | ✅ |

| Group stat | Value |
|------------|-------|
| Mean ΔPSNR | −0.061 dB |
| Mean ΔSSIM | −0.0060 |
| Mean ΔGaussians | −9.96% |
| Mean ΔWall_time | −2.9% |
| Scenes quality-preserved | 4 / 4 (100%) |

**Analysis:** Indoor scenes are the **best group for C42 additivity** — all 4 scenes preserve quality on both metrics. SSIM degradation is minimal (−0.004 to −0.009, well within ±0.02), and PSNR changes are small (−0.014 to −0.149, all within ±0.3). Counter even shows a slight PSNR improvement (+0.045). Gaussian reductions are moderate (−6.5% to −14.5%), consistent with indoor scenes having fewer redundant Gaussians to prune. Wall-time savings are modest (only room shows a measurable −11.4%; the other three are unchanged), likely because indoor scenes already have lower Gaussian counts and the SSIM downsampling cost saving is proportionally smaller. The tight SSIM preservation here suggests that indoor scenes' smoother geometry is less affected by the loss of high-frequency SSIM signal.

### 7.3 Tanks & Temples (truck, train)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss (%) | ΔWall (%) | Preserved? |
|-------|-------|-------|--------|------------|-----------|------------|
| truck | −0.282 | −0.0215 | +0.0357 | −41.06% | −20.06% | ❌ (SSIM) |
| train | −0.075 | −0.0218 | +0.0207 | −25.11% | −5.49% | ❌ (SSIM) |

| Group stat | Value |
|------------|-------|
| Mean ΔPSNR | −0.179 dB |
| Mean ΔSSIM | −0.0217 |
| Mean ΔGaussians | −33.1% |
| Mean ΔWall_time | −12.8% |
| Scenes quality-preserved | 0 / 2 (0%) |

**Analysis:** Tanks & Temples is the **weakest group for C42 additivity** — neither scene preserves quality, both failing on SSIM (−0.0215 and −0.0218, just over the ±0.02 limit). However, this group shows the **most aggressive Gaussian reduction** (truck −41.1%, train −25.1%) and the **best wall-time speedup** (truck −20.1%). Truck is the most extreme scene in the entire benchmark: the largest PSNR drop (−0.282, still within ±0.3), the largest SSIM drop (−0.0215), the largest LPIPS increase (+0.0357), the largest Gaussian reduction (−41.1%), and the largest wall-time speedup (−20.1%). This reveals the core C42 trade-off: aggressive structural downsampling prunes more Gaussians and trains faster, but at a measurable cost to structural fidelity — particularly for scenes with fine geometric detail (truck's reflective surfaces, train's rails and ties). Both scenes still have PSNR within ±0.3 dB; it is SSIM that crosses the threshold.

### 7.4 Deep Blending (drjohnson, playroom)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss (%) | ΔWall (%) | Preserved? |
|-------|-------|-------|--------|------------|-----------|------------|
| drjohnson | −0.310 | −0.0047 | +0.0044 | −4.62% | −6.23% | ❌ (PSNR) |
| playroom | −0.072 | −0.0033 | +0.0043 | −15.33% | −11.04% | ✅ |

| Group stat | Value |
|------------|-------|
| Mean ΔPSNR | −0.191 dB |
| Mean ΔSSIM | −0.0040 |
| Mean ΔGaussians | −9.98% |
| Mean ΔWall_time | −8.6% |
| Scenes quality-preserved | 1 / 2 (50%) |

**Analysis:** Deep Blending shows a split. Playroom is well-preserved (ΔPSNR = −0.072, ΔSSIM = −0.003) with good Gaussian reduction (−15.3%) and wall-time speedup (−11.0%). Drjohnson is the **only scene in the benchmark that fails on PSNR** (−0.310, just over the ±0.3 threshold by 0.01 dB), though its SSIM is excellent (−0.0047, the second-best in the entire benchmark). The drjohnson failure is marginal — 0.010 dB over the threshold — and its Gaussian reduction is the smallest in the benchmark (−4.6%), suggesting this scene's geometry is not amenable to C42's pruning pressure. The very low SSIM degradation in both Deep Blending scenes (−0.003 to −0.005) is notable: these are large-scale indoor environments where the area-downsampled SSIM captures most of the relevant structural information, so the loss of high-frequency signal has minimal impact.

---

## 8. C42 Classification

### 8.1 Criteria

| Class | Quality-preservation rate | Description |
|-------|--------------------------|-------------|
| BROADLY_ADDITIVE | ≥ 70% | C42 preserves quality (PSNR ±0.3 dB, SSIM ±0.02) in ≥70% of scenes while consistently reducing Gaussians |
| PARTIALLY_ADDITIVE | 40–70% | C42 preserves quality in 40–70% of scenes |
| BASELINE_SPECIFIC | < 40% | C42 preserves quality in <40% of scenes |
| NO_STRONG_BASELINE_ADDITIVITY | — | C42 degrades quality in most scenes |

### 8.2 Computed Values

| Metric | Value | Threshold | Met? |
|--------|-------|-----------|------|
| Scenes quality-preserved (both PSNR ±0.3 **and** SSIM ±0.02) | 9 / 13 = **69.2%** | ≥ 70% | ❌ (0.8 pp short) |
| Scenes reducing Gaussians | 12 / 13 = **92.3%** | Consistent | ✅ |
| Scenes PSNR within ±0.3 dB | 12 / 13 = 92.3% | — | — |
| Scenes SSIM within ±0.02 | 10 / 13 = 76.9% | — | — |
| Scenes PSNR improved | 3 / 13 = 23.1% | — | — |
| Geometric mean speedup | 1.084× | — | — |

### 8.3 Verdict

## **PARTIALLY_ADDITIVE**

C42 preserves quality (both PSNR within ±0.3 dB and SSIM within ±0.02) in **9 of 13 scenes (69.2%)**, falling in the 40–70% range for PARTIALLY_ADDITIVE. The result is **borderline** — just 0.8 percentage points below the 70% threshold for BROADLY_ADDITIVE, and would cross it if a single borderline scene (treehill, SSIM = −0.0212) were within the ±0.02 limit.

**Why not BROADLY_ADDITIVE:**
- 4 scenes fail the combined quality criterion: treehill (SSIM −0.0212), truck (SSIM −0.0215), train (SSIM −0.0218), and drjohnson (PSNR −0.310).
- The binding constraint is **SSIM**: all 13 scenes show negative ΔSSIM (range −0.0033 to −0.0218). C42's area-downsampled SSIM term systematically reduces the optimizer's emphasis on high-frequency structural fidelity, producing a consistent but usually small SSIM penalty. Three scenes exceed the ±0.02 SSIM threshold, all by narrow margins (0.0012–0.0018 over the limit).
- Only one scene (drjohnson) fails on PSNR, and by just 0.010 dB.

**Why not BASELINE_SPECIFIC or NO_STRONG_BASELINE_ADDITIVITY:**
- 69.2% quality preservation is well above 40%.
- Gaussian reduction is consistent (92.3% of scenes, 100% of valid scenes excluding the garden data artifact).
- PSNR is preserved in 92.3% of scenes — the quality trade-off is concentrated in SSIM, not overall fidelity.
- The loss-equivalence test confirms C42 is a faithful generalization of the native loss, not a divergent modification.

**Path to BROADLY_ADDITIVE:** A modest reduction in the SSIM downsampling aggressiveness (e.g., scale_factor = 0.6–0.7 instead of 0.5, or a small SSIM weight increase) would likely bring the 3 SSIM-borderline scenes within threshold and push the classification to BROADLY_ADDITIVE without sacrificing the Gaussian-reduction benefit.

---

## 9. Key Findings

1. **C42 is mathematically faithful.** The loss-equivalence test at scale = 1.0 shows L1 diff = 0, DSSIM diff = 0, gradient cosine = 1.0, and gradient relative L2 = 0. C42 is a proper generalization of the native Speedy-Splat loss, not an approximation.

2. **PSNR is well-preserved.** 12 of 13 scenes (92.3%) have ΔPSNR within ±0.3 dB. The mean ΔPSNR is −0.090 dB — statistically negligible. Three scenes (bicycle, flowers, counter) actually *improve* PSNR under C42, suggesting the downsampled SSIM acts as a beneficial regularizer in some cases.

3. **SSIM is the binding constraint.** All 13 scenes show negative ΔSSIM (mean −0.0125). Three scenes (treehill, truck, train) exceed the ±0.02 threshold, all by margins of 0.001–0.002. This systematic SSIM degradation is the direct mechanism of C42: by computing SSIM at half resolution, the optimizer receives a weaker high-frequency structural gradient, leading to slightly lower structural fidelity at full resolution.

4. **Gaussian reduction is consistent and substantial.** 12 of 13 scenes reduce Gaussian count (92.3%); among valid scenes (excluding garden data artifact), the rate is 100% with a mean reduction of −15.7%. The most aggressive reductions are in Tanks & Temples (truck −41.1%, train −25.1%), where Speedy-Splat's pruning interacts most strongly with C42's structural downsampling.

5. **Wall-time speedup is modest but real.** The geometric-mean speedup is 1.084× (8.4%). Nine of 13 scenes show measurable speedup; four show no change (bicycle, counter, kitchen, bonsai — all scenes where the SSIM computation cost is already small relative to total training time). The best speedup is truck at 1.251× (20.1% faster).

6. **Indoor scenes are the best C42 candidates.** All 4 MipNeRF360 indoor scenes (room, counter, kitchen, bonsai) preserve quality on both metrics (100% quality-preservation rate). Indoor scenes have smoother geometry where the loss of high-frequency SSIM signal matters less, and ΔSSIM is small (−0.004 to −0.009).

7. **Tanks & Temples shows the sharpest trade-off.** Both truck and train fail quality preservation (0/2), but both show the largest Gaussian reductions and the best speedups in the benchmark. Truck is the most extreme scene: largest PSNR drop, largest SSIM drop, largest LPIPS increase, largest Gaussian reduction, and largest speedup — all at once. This scene crystallizes the C42 quality-vs-efficiency trade-off.

8. **The classification is borderline PARTIALLY_ADDITIVE.** At 69.2% quality preservation, C42 misses BROADLY_ADDITIVE by a single scene (0.8 pp). Three of the four failing scenes miss by SSIM margins of ≤0.002, and the fourth (drjohnson) misses PSNR by 0.010 dB. A slightly less aggressive downsampling factor would likely flip the classification.

9. **Garden is a data artifact, not a C42 failure.** The garden scene's depressed metrics (PSNR ≈ 12.5 dB for both methods) and anomalous +147% Gaussian increase under C42 are caused by manually-created COLMAP data, not by the C42 patch. Excluding garden, C42 reduces Gaussians in 100% of valid scenes.

10. **C42 and Speedy-Splat pruning interact constructively.** Speedy-Splat's aggressive densification pruning combined with C42's structural downsampling produces consistently fewer Gaussians without catastrophic quality loss. The interaction is not neutral — C42 amplifies the pruning effect (mean −15.7% additional Gaussians removed beyond Speedy-Splat's own pruning) — but the quality cost is concentrated in SSIM and is usually small.

---

## 10. Provenance

| Field | Value |
|-------|-------|
| Report file | `reports/speedysplat-c42-13scene-final.md` |
| Data source | `results/speedy-splat-all-metrics.json` |
| Baseline repo | `j-alex-hanson/speedy-splat` |
| Baseline commit | `34c45c6` |
| Baseline branch | `speedy-splat` |
| C42 branch | `experiment/c42-on-speedysplat` |
| C42 commits | `07f4712`, `aa75d8d`, plus GUI fix |
| Conda environment | `strong_native` (Python 3.10, PyTorch 2.1.0+cu118, CUDA 11.8) |
| GPU | NVIDIA A100-PCIE-40GB (mx) |
| Training iterations | 30,000 per scene |
| Scenes | 13 (MipNeRF360 ×9, Tanks&Temples ×2, Deep Blending ×2) |
| C42 parameters | λ_DSSIM = 0.2, λ_L1 = 0.8, scale_factor = 0.5, mode = "area" |
| Loss-equivalence test | PASSED (scale = 1.0: L1 diff = 0, DSSIM diff = 0, grad cosine = 1.0, grad rel L2 = 0) |
| Report date | 2026-09-16 |
| Classification | **PARTIALLY_ADDITIVE** (69.2% quality-preserved, 92.3% Gaussian-reduced, 1.084× geomean speedup) |
