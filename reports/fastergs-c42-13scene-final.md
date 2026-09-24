# C42 Benchmark Report — Faster-GS (13-Scene Final)

**Report ID:** fastergs-c42-13scene-final  
**Date:** 2026-09-17  
**Baseline renderer:** Faster-GS (NeRFICG framework)  
**C42 patch:** Structural downsampling (L1 full-res, DSSIM on 0.5× area-downsampled)

---

## 1. Executive Summary

C42 structural downsampling is **partially additive** on Faster-GS across the 13-scene benchmark, but in a fundamentally different manner than on FastGS or Speedy-Splat. Under the strict quality-preservation criterion (PSNR within ±0.3 dB **and** SSIM within ±0.02), C42 preserves quality in 8 of 13 scenes (61.5%), placing it in the PARTIALLY_ADDITIVE band. However, this headline number is misleading: 3 of the 5 "failures" (bicycle, room, truck) are scenes where C42 **improves** quality beyond the preservation band, not scenes where it degrades quality. Only stump (ΔPSNR −0.94, ΔSSIM −0.040) represents a genuine quality degradation, and garden is a known data-quality outlier. Under a non-degradation criterion (C42 does not make quality worse), 11 of 13 scenes (84.6%) pass — which would qualify as BROADLY_ADDITIVE. C42 improves mean PSNR by +0.19 dB, leaves mean SSIM unchanged (0.000), and improves mean LPIPS by −0.006 — the first baseline where C42 produces a net quality **improvement** rather than degradation. The loss-equivalence test is exact (all zeros, gradient cosine = 1.0), confirming C42 is a mathematically faithful generalization. However, Gaussian reduction is inconsistent: C42 reduces Gaussians in only 9 of 13 scenes (69.2%), with 4 scenes showing Gaussian **increases**, and wall-time speedup is negligible (geometric mean 0.998×, essentially parity). The classification remains PARTIALLY_ADDITIVE due to the strict 61.5% preservation rate and the inconsistent Gaussian reduction, but the underlying dynamic is quality-improving rather than quality-degrading. **Classification: PARTIALLY_ADDITIVE.**

---

## 2. Configuration

### 2.1 Baseline

| Field | Value |
|---|---|
| Renderer | Faster-GS |
| Repository | `nerficg-project/faster-gaussian-splatting` |
| Commit | `3cb0b75` |
| Branch | `main` |
| Framework | NeRFICG (config-driven via YAML, launched with `scripts/train.py -c config.yaml`) |

### 2.2 C42 Patch

| Field | Value |
|---|---|
| Mechanism | L1 loss stays at full resolution; DSSIM computed on `F.interpolate(scale_factor=0.5, mode="area")` downsampled pred/target |
| Loss weights | λ = 0.2 for DSSIM, λ = 0.8 for L1 |
| Iterations | 30,000 |
| Patch files | `Loss.py` (C42_SCALE attribute + conditional downsample in forward), `Trainer.py` (added C42_SCALE=1.0 to LOSS ConfigParameterList) |

**Mechanism:** The C42 patch modifies the NeRFICG loss function so that the structural-similarity (DSSIM) component is evaluated at half spatial resolution via area-mode interpolation, while the L1 component remains at full resolution. At C42_SCALE = 1.0 the patch reduces exactly to the native Faster-GS loss (proven by the loss-equivalence test in Section 3). At C42_SCALE = 0.5 the DSSIM gradient signal is computed at coarser resolution, which reduces the optimizer's emphasis on high-frequency structural fidelity and shifts the loss landscape toward L1-dominant reconstruction.

**Patch details:**
- **`Loss.py`**: Added `import torch.nn.functional as F`, `c42_scale` attribute in `__init__`, modified `forward()` to conditionally downsample pred/target with `F.interpolate(scale_factor=c42_scale, mode='area')` for DSSIM_Color term only. L1_Color uses full-resolution input/target.
- **`Trainer.py`**: Added `C42_SCALE=1.0` to the `LOSS` ConfigParameterList default configuration.

### 2.3 Training Environment

| Field | Value |
|---|---|
| Conda env | `nerficg` |
| Python | 3.11 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA mx A100-PCIE-40GB |
| Training speed | ~70–140 it/s (scene-dependent) |

### 2.4 NeRFICG Densification & Framework Parameters

| Parameter | Value |
|---|---|
| DENSIFICATION_INTERVAL | 100 |
| DENSIFICATION_GRAD_THRESHOLD | 0.0002 |
| Iterations | 30,000 |
| PPISP | Disabled (USE: false) |
| IMAGE_SCALE_FACTOR | 0.25 (MipNeRF360), 1.0 (T&T/DB, using MipNeRF360 loader) |

### 2.5 Benchmark Scenes (13)

| Dataset | Scenes |
|---------|--------|
| MipNeRF360 (outdoor) | bicycle, flowers, garden, stump, treehill |
| MipNeRF360 (indoor) | room, counter, kitchen, bonsai |
| Tanks & Temples | truck, train |
| Deep Blending | drjohnson, playroom |

> **Data caveat — garden:** The garden scene uses manually-created COLMAP data (derived from `cameras.json` + a truncated PLY), not the official MipNeRF360 COLMAP bundle. Consequently both native and C42 metrics are very low (PSNR ≈ 13.4–13.9 dB, SSIM ≈ 0.22–0.24). This is a **data-quality issue, not a C42 issue**, and the garden scene is flagged throughout the analysis.

---

## 3. Loss Equivalence Test

The C42 patch was validated for mathematical correctness at C42_SCALE = 1.0 (no downsampling). At this setting the C42 loss function must produce bit-identical results to the native Faster-GS loss.

| Metric | Native | C42 (scale=1.0) | Difference | Verdict |
|--------|--------|-----------------|------------|---------|
| Loss (total) | L | L | **0** | ✅ Identical |
| L1 loss | L₁ | L₁ | **0** | ✅ Identical |
| DSSIM loss | D | D | **0** | ✅ Identical |
| Gradient cosine similarity | 1.0 | 1.0 | **1.0** | ✅ Identical |
| Gradient relative L2 norm | 0 | 0 | **0** | ✅ Identical |

**Result: PASSED.** At scale = 1.0, C42 produces loss values and gradients that are **exactly** identical to the native Faster-GS loss (loss diff = 0, L1 diff = 0, DSSIM diff = 0, gradient cosine = 1.0, gradient relative L2 = 0). This confirms that C42 is a mathematically faithful generalization of the native loss — the DSSIM downsampling is a proper extension that reduces to the baseline when the downsample factor is unity. Unlike FastGS (which showed gradient cosine = 0.9997 due to fused_ssim CUDA kernel non-determinism), the Faster-GS/NeRFICG implementation uses `fused_dssim` from NeRFICG's `Optim.Losses.DSSIM` which is fully deterministic in both forward and backward passes, achieving **exact** equivalence with no numerical deviation. All quality differences observed at C42_SCALE = 0.5 are therefore intrinsic to the structural downsampling choice, not implementation bugs.

---

## 4. 13-Scene Results Table

| # | Scene | Method | PSNR (dB) | SSIM | LPIPS | N_gaussians | Wall_time (min) |
|---|---|---|---|---|---|---|---|
| 1 | bicycle | native | 23.84 | 0.661 | 0.366 | 2,846,244 | 11.46 |
| 1 | bicycle | c42 | 23.75 | 0.687 | 0.329 | 2,796,456 | 11.15 |
| 2 | flowers | native | 20.46 | 0.520 | 0.420 | 2,092,110 | 8.35 |
| 2 | flowers | c42 | 20.60 | 0.522 | 0.410 | 2,226,512 | 8.14 |
| 3 | garden ⚠️ | native | 13.40 | 0.240 | 0.676 | 2,951,732 | 9.97 |
| 3 | garden ⚠️ | c42 | 13.90 | 0.217 | 0.650 | 3,291,078 | 6.69 |
| 4 | stump | native | 23.42 | 0.643 | 0.369 | 2,618,281 | 4.68 |
| 4 | stump | c42 | 22.48 | 0.603 | 0.400 | 2,311,449 | 4.76 |
| 5 | treehill | native | 20.90 | 0.574 | 0.449 | 2,479,792 | 4.74 |
| 5 | treehill | c42 | 21.14 | 0.580 | 0.441 | 2,362,574 | 5.10 |
| 6 | room | native | 27.74 | 0.876 | 0.323 | 955,542 | 4.96 |
| 6 | room | c42 | 29.73 | 0.904 | 0.295 | 953,389 | 5.61 |
| 7 | counter | native | 27.16 | 0.866 | 0.313 | 969,498 | 4.98 |
| 7 | counter | c42 | 26.98 | 0.869 | 0.305 | 984,464 | 5.45 |
| 8 | kitchen | native | 28.82 | 0.897 | 0.188 | 1,188,079 | 6.03 |
| 8 | kitchen | c42 | 28.99 | 0.900 | 0.192 | 1,119,283 | 6.42 |
| 9 | bonsai | native | 29.61 | 0.912 | 0.292 | 1,183,977 | 4.71 |
| 9 | bonsai | c42 | 29.44 | 0.904 | 0.295 | 1,294,185 | 5.69 |
| 10 | truck | native | 22.50 | 0.804 | 0.249 | 1,718,002 | 5.10 |
| 10 | truck | c42 | 23.14 | 0.814 | 0.247 | 1,387,922 | 4.37 |
| 11 | train | native | 20.77 | 0.760 | 0.296 | 899,292 | 3.78 |
| 11 | train | c42 | 20.82 | 0.754 | 0.303 | 767,043 | 4.04 |
| 12 | drjohnson | native | 29.24 | 0.899 | 0.318 | 2,578,978 | 5.76 |
| 12 | drjohnson | c42 | 29.09 | 0.901 | 0.312 | 2,573,644 | 6.24 |
| 13 | playroom | native | 29.54 | 0.905 | 0.311 | 1,614,673 | 4.73 |
| 13 | playroom | c42 | 29.81 | 0.902 | 0.311 | 1,457,076 | 4.40 |

> ⚠️ **garden** uses manually-created COLMAP data (from `cameras.json` + truncated PLY). Metrics are very low for both native and C42. This is a data issue, not a C42 effect.

### Aggregate Means

| Metric | Native (mean) | C42 (mean) | Delta |
|---|---|---|---|
| PSNR (dB) | 24.4154 | 24.6054 | **+0.1900** |
| SSIM | 0.7352 | 0.7352 | **0.0000** |
| LPIPS | 0.3515 | 0.3454 | **−0.0062** |
| N_gaussians | 1,853,554 | 1,809,621 | −43,933 (−2.37%) |
| Wall_time (min) | 6.096 | 6.005 | −0.092 (−1.50%) |

> **Notable:** This is the first C42 benchmark where mean PSNR **improves** (+0.19 dB), mean SSIM is **unchanged** (0.000), and mean LPIPS **improves** (−0.006). On FastGS, all three means degraded (ΔPSNR −0.18, ΔSSIM −0.014, ΔLPIPS +0.013). On Speedy-Splat, all three degraded (ΔPSNR −0.09, ΔSSIM −0.013, ΔLPIPS +0.010).

---

## 5. Delta Analysis

Per-scene deltas (C42 − native). Positive ΔPSNR = improvement; positive ΔSSIM = improvement; positive ΔLPIPS = degradation (lower is better); negative ΔGaussians = reduction (beneficial); negative ΔWall = speedup (beneficial).

| # | Scene | ΔPSNR (dB) | ΔSSIM | ΔLPIPS | ΔGaussians (%) | ΔWall_time (%) | Preserved? |
|---|---|---|---|---|---|---|---|
| 1 | bicycle | −0.09 | **+0.026** | **−0.037** | −1.75% | −2.71% | ❌ (SSIM improved ↑) |
| 2 | flowers | **+0.14** | **+0.002** | **−0.010** | +6.42% | −2.51% | ✅ Yes |
| 3 | garden ⚠️ | **+0.50** | −0.023 | **−0.026** | +11.50% | −32.90% | ❌ (data issue) |
| 4 | stump | −0.94 | −0.040 | +0.031 | −11.72% | +1.71% | ❌ No (both degraded) |
| 5 | treehill | **+0.24** | **+0.006** | **−0.008** | −4.73% | +7.59% | ✅ Yes |
| 6 | room | **+1.99** | **+0.028** | **−0.028** | −0.23% | +13.10% | ❌ (both improved ↑) |
| 7 | counter | −0.18 | **+0.003** | **−0.008** | +1.54% | +9.44% | ✅ Yes |
| 8 | kitchen | **+0.17** | **+0.003** | +0.004 | −5.79% | +6.47% | ✅ Yes |
| 9 | bonsai | −0.17 | −0.008 | +0.003 | +9.31% | +20.81% | ✅ Yes |
| 10 | truck | **+0.64** | **+0.010** | **−0.002** | −19.21% | −14.31% | ❌ (PSNR improved ↑) |
| 11 | train | **+0.05** | −0.006 | +0.007 | −14.71% | +6.88% | ✅ Yes |
| 12 | drjohnson | −0.15 | **+0.002** | **−0.006** | −0.21% | +8.33% | ✅ Yes |
| 13 | playroom | **+0.27** | −0.003 | 0.000 | −9.76% | −6.98% | ✅ Yes |

> **Bold** = C42 improves over native. For PSNR and SSIM, positive delta = improvement. For LPIPS, negative delta = improvement.  
> **↑** = fails strict preservation because C42 **improves** quality beyond the ±band (not a degradation).

### Quality Preservation Breakdown

| Criterion | Scenes passing | Percentage |
|---|---|---|
| PSNR within ±0.3 dB | 9 / 13 | 69.2% |
| SSIM within ±0.02 | 9 / 13 | 69.2% |
| **Both (joint criterion, strict)** | **8 / 13** | **61.5%** |
| Non-degradation (PSNR ≥ −0.3 AND SSIM ≥ −0.02) | 11 / 13 | 84.6% |
| C42 improves PSNR | 8 / 13 | 61.5% |
| C42 reduces Gaussians | 9 / 13 | 69.2% |
| C42 improves SSIM | 8 / 13 | 61.5% |
| C42 improves LPIPS | 8 / 13 | 61.5% |

**Scenes passing joint criterion (8):** flowers, treehill, counter, kitchen, bonsai, train, drjohnson, playroom  
**Scenes failing joint criterion (5):**
- **bicycle** — SSIM improved (+0.026, exceeds ±0.02 band upward) — **not a degradation**
- **garden** — data-quality outlier (manually-created COLMAP)
- **stump** — genuine quality degradation (ΔPSNR −0.94, ΔSSIM −0.040)
- **room** — both PSNR (+1.99) and SSIM (+0.028) improved beyond band — **not a degradation**
- **truck** — PSNR improved (+0.64, exceeds ±0.3 band upward) — **not a degradation**

> **Critical nuance:** Of the 5 scenes failing the strict joint criterion, only **stump** represents a genuine quality degradation. Garden is a data issue. Bicycle, room, and truck all fail because C42 **improves** quality beyond the preservation band — the deltas are positive, not negative. Under a non-degradation criterion (C42 does not make quality worse), 11/13 scenes (84.6%) pass, exceeding the 70% threshold for BROADLY_ADDITIVE.

---

## 6. Statistical Summary

| Statistic | Value |
|---|---|
| **Mean ΔPSNR** | **+0.1900 dB** |
| Median ΔPSNR | +0.14 dB |
| **Mean ΔSSIM** | **0.0000** |
| Median ΔSSIM | +0.002 |
| **Mean ΔLPIPS** | **−0.0062** |
| Median ΔLPIPS | −0.006 |
| Mean ΔGaussians | −3.02% |
| Total Gaussian reduction | −571,125 (from 24,096,200 to 23,525,075, −2.37%) |
| Mean ΔWall_time | +1.15% |
| **Geometric mean wall-time speedup** | **0.998×** (near parity) |
| Geomean speedup (excl. garden) | 0.965× (3.5% slower) |
| **% scenes C42 improves PSNR** | **61.5%** (8/13) |
| **% scenes C42 reduces Gaussians** | **69.2%** (9/13) |
| % scenes C42 improves SSIM | 61.5% (8/13) |
| % scenes C42 improves LPIPS | 61.5% (8/13) |
| % scenes PSNR within ±0.3 dB | 69.2% (9/13) |
| % scenes SSIM within ±0.02 | 69.2% (9/13) |
| **% scenes quality preserved (strict)** | **61.5%** (8/13) |
| % scenes quality not degraded | 84.6% (11/13) |
| Total native wall time | 79.25 min |
| Total C42 wall time | 78.06 min |
| Total wall-time change | −1.19 min (−1.50%) |

**Geometric mean wall-time speedup derivation:**

| Scene | Native (min) | C42 (min) | Speedup ratio |
|-------|-------------|-----------|---------------|
| bicycle | 11.46 | 11.15 | 1.0278 |
| flowers | 8.35 | 8.14 | 1.0258 |
| garden | 9.97 | 6.69 | 1.4903 |
| stump | 4.68 | 4.76 | 0.9832 |
| treehill | 4.74 | 5.10 | 0.9294 |
| room | 4.96 | 5.61 | 0.8841 |
| counter | 4.98 | 5.45 | 0.9138 |
| kitchen | 6.03 | 6.42 | 0.9393 |
| bonsai | 4.71 | 5.69 | 0.8278 |
| truck | 5.10 | 4.37 | 1.1670 |
| train | 3.78 | 4.04 | 0.9356 |
| drjohnson | 5.76 | 6.24 | 0.9231 |
| playroom | 4.73 | 4.40 | 1.0750 |

$$\text{Geomean} = \left(\prod_{i=1}^{13} \frac{t_{\text{native},i}}{t_{\text{c42},i}}\right)^{1/13} = e^{\frac{1}{13}\sum \ln(r_i)} = e^{-0.00174} \approx \mathbf{0.998\times}$$

---

## 7. Scene-by-Scene Analysis

### 7.1 MipNeRF360 Outdoor (bicycle, flowers, garden, stump, treehill)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|---|
| bicycle | −0.09 | +0.026 | −0.037 | −1.75% | −2.71% | ❌ (SSIM improved ↑) |
| flowers | +0.14 | +0.002 | −0.010 | +6.42% | −2.51% | ✅ |
| garden ⚠️ | +0.50 | −0.023 | −0.026 | +11.50% | −32.90% | ❌ (data issue) |
| stump | −0.94 | −0.040 | +0.031 | −11.72% | +1.71% | ❌ (both degraded) |
| treehill | +0.24 | +0.006 | −0.008 | −4.73% | +7.59% | ✅ |
| **Group mean** | **−0.03** | **−0.006** | **−0.010** | **−0.06%** | **−5.76%** | **2/5** |

**Observations:** This group shows the widest within-group variance and the weakest strict preservation (2/5). Stump is the sole genuine quality degradation in the entire benchmark — C42 drops PSNR by 0.94 dB and SSIM by 0.040, both well outside thresholds, while achieving the largest Gaussian reduction in the group (−11.72%). This is the classic C42 trade-off seen on FastGS and Speedy-Splat: aggressive structural downsampling on high-frequency outdoor foliage sacrifices structural fidelity. However, the other outdoor scenes tell a very different story. Bicycle **improves** SSIM (+0.026) and LPIPS (−0.037) while keeping PSNR essentially unchanged (−0.09), though the SSIM improvement exceeds the ±0.02 band, causing a strict-preservation "failure" that is actually a quality gain. Treehill improves on all three quality metrics (ΔPSNR +0.24, ΔSSIM +0.006, ΔLPIPS −0.008) and passes the joint criterion. Flowers improves PSNR (+0.14) and LPIPS (−0.010) but increases Gaussian count (+6.42%). Garden is a data-quality outlier with anomalous wall-time speedup (−32.90%) driven by the degraded COLMAP input. The group's near-zero mean Gaussian change (−0.06%) reflects the canceling effect of stump's reduction and garden/flowers' increases.

### 7.2 MipNeRF360 Indoor (room, counter, kitchen, bonsai)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|---|
| room | +1.99 | +0.028 | −0.028 | −0.23% | +13.10% | ❌ (both improved ↑) |
| counter | −0.18 | +0.003 | −0.008 | +1.54% | +9.44% | ✅ |
| kitchen | +0.17 | +0.003 | +0.004 | −5.79% | +6.47% | ✅ |
| bonsai | −0.17 | −0.008 | +0.003 | +9.31% | +20.81% | ✅ |
| **Group mean** | **+0.45** | **+0.007** | **−0.007** | **+1.21%** | **+12.45%** | **3/4** |

**Observations:** This is the most **quality-improving** group for C42 — the group mean PSNR improves by +0.45 dB and group mean SSIM by +0.007. The standout is **room**, where C42 produces a massive +1.99 dB PSNR improvement and +0.028 SSIM improvement, the largest single-scene quality gain in any C42 benchmark. This suggests the native Faster-GS loss was poorly optimized for room's geometry, and C42's downsampled DSSIM acts as a beneficial regularizer that corrects the optimization trajectory — the half-resolution SSIM shifts optimization weight toward full-resolution L1, directly benefiting pixel-level fidelity for the relatively simple indoor geometry. However, room "fails" the strict preservation criterion because both deltas exceed the preservation band — a failure that is entirely positive. Counter, kitchen, and bonsai all pass the joint criterion with small quality changes. Notably, this group has the **highest wall-time overhead** (mean +12.45%) and **slightly increases Gaussian count** (mean +1.21%), driven by bonsai's +9.31% Gaussian increase and +20.81% wall-time increase. The indoor scene behavior is qualitatively different from FastGS and Speedy-Splat, where indoor scenes showed the best Gaussian reduction with minimal quality loss — here, C42 trades Gaussian efficiency for quality improvement.

### 7.3 Tanks & Temples (truck, train)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|---|
| truck | +0.64 | +0.010 | −0.002 | −19.21% | −14.31% | ❌ (PSNR improved ↑) |
| train | +0.05 | −0.006 | +0.007 | −14.71% | +6.88% | ✅ |
| **Group mean** | **+0.35** | **+0.002** | **+0.003** | **−16.96%** | **−3.72%** | **1/2** |

**Observations:** Tanks & Temples shows the **best Gaussian reduction** of any group (mean −16.96%) combined with **quality improvement** (mean ΔPSNR +0.35). Truck is the most striking scene: C42 improves PSNR by +0.64 dB, improves SSIM by +0.010, improves LPIPS by −0.002, reduces Gaussians by −19.21%, and speeds up wall time by −14.31% — an across-the-board improvement. Yet truck "fails" the strict preservation criterion solely because the +0.64 dB PSNR gain exceeds the ±0.3 dB band upward. This is a clear case where the strict criterion penalizes C42 for being too beneficial. Train passes the joint criterion with a smaller PSNR gain (+0.05) and good Gaussian reduction (−14.71%), though with a slight wall-time increase (+6.88%). This group demonstrates that C42 can simultaneously improve quality, reduce Gaussians, and speed up training when the baseline's native optimization is suboptimal — a profile not seen on FastGS or Speedy-Splat, where T&T showed either the worst quality preservation (Speedy-Splat: 0/2 preserved) or moderate preservation (FastGS: 2/2 preserved but with systematic SSIM degradation).

### 7.4 Deep Blending (drjohnson, playroom)

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGauss | ΔWall | Preserved? |
|---|---|---|---|---|---|---|
| drjohnson | −0.15 | +0.002 | −0.006 | −0.21% | +8.33% | ✅ |
| playroom | +0.27 | −0.003 | 0.000 | −9.76% | −6.98% | ✅ |
| **Group mean** | **+0.06** | **−0.001** | **−0.003** | **−4.99%** | **+0.67%** | **2/2** |

**Observations:** Deep Blending is the **only group with 100% strict preservation** (2/2). Both scenes pass the joint criterion with minimal quality changes. Drjohnson shows a small PSNR decrease (−0.15, within threshold) but improves SSIM (+0.002) and LPIPS (−0.006), with negligible Gaussian change (−0.21%). Playroom improves PSNR (+0.27, within threshold) and reduces Gaussians by −9.76% with a wall-time speedup of −6.98%. The group's near-zero mean wall-time change (+0.67%) and moderate Gaussian reduction (−4.99%) reflect a balanced profile. These large-scale indoor environments benefit from C42's structural downsampling without the high-frequency foliage challenges of outdoor scenes. Notably, on FastGS, Deep Blending was the most volatile group (playroom had the worst single-scene PSNR drop at −0.79 dB); on Faster-GS, the same group is the most stable, highlighting how C42's effect is baseline-dependent.

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
| Scenes quality-preserved (strict: both PSNR ±0.3 **and** SSIM ±0.02) | 8 / 13 = **61.5%** | ≥ 70% for BROADLY | ❌ |
| Scenes quality not degraded (PSNR ≥ −0.3 **and** SSIM ≥ −0.02) | 11 / 13 = **84.6%** | ≥ 70% for BROADLY | ✅ |
| Scenes reducing Gaussians | 9 / 13 = **69.2%** | Consistent | ⚠️ Inconsistent |
| Scenes PSNR within ±0.3 dB | 9 / 13 = 69.2% | — | — |
| Scenes SSIM within ±0.02 | 9 / 13 = 69.2% | — | — |
| Scenes PSNR improved | 8 / 13 = 61.5% | — | — |
| Mean ΔPSNR | +0.19 dB | — | — |
| Mean ΔSSIM | 0.000 | — | — |
| Geometric mean speedup | 0.998× | — | — |

### 8.3 Verdict

## **PARTIALLY_ADDITIVE**

C42 preserves quality (strict: both PSNR within ±0.3 dB and SSIM within ±0.02) in **8 of 13 scenes (61.5%)**, placing it in the 40–70% range for PARTIALLY_ADDITIVE.

**Why not BROADLY_ADDITIVE:**

1. **Strict preservation is 61.5%** — below the 70% threshold. Five scenes fail the joint criterion: bicycle, garden, stump, room, and truck.

2. **Gaussian reduction is inconsistent** — C42 reduces Gaussian count in only 9 of 13 scenes (69.2%). Four scenes show Gaussian **increases** (flowers +6.42%, garden +11.50%, counter +1.54%, bonsai +9.31%). This does not meet the "consistently reducing Gaussians" requirement for BROADLY_ADDITIVE, unlike FastGS (100%) and Speedy-Splat (92.3%).

3. **No wall-time speedup** — the geometric mean speedup is 0.998× (near parity), and excluding the garden data outlier, C42 is 3.5% slower (0.965×). Eight of 13 scenes show wall-time increases, with bonsai (+20.81%) and room (+13.10%) the worst. The NeRFICG framework's overhead (config loading, data preloading, checkpoint management) and Faster-GS's frequent densification (interval=100) mask any training-speed benefit from fewer Gaussians.

**Critical nuance — this PARTIALLY_ADDITIVE is qualitatively different from FastGS and Speedy-Splat:**

On FastGS and Speedy-Splat, the strict-preservation "failures" were **genuine quality degradations** — C42 made quality worse. On Faster-GS, the picture is inverted:

| Scene | Fails because… | Actual effect |
|---|---|---|
| bicycle | SSIM +0.026 exceeds ±0.02 | **SSIM improved** |
| garden | Both exceed bands | **Data-quality outlier** |
| stump | Both exceed bands | **Genuine degradation** (sole real failure) |
| room | Both exceed bands | **Both improved** (PSNR +1.99, SSIM +0.028) |
| truck | PSNR +0.64 exceeds ±0.3 | **PSNR improved** |

Under a **non-degradation** criterion (C42 does not make quality worse: ΔPSNR ≥ −0.3 dB and ΔSSIM ≥ −0.02), 11/13 scenes (84.6%) pass — well above the 70% BROADLY_ADDITIVE threshold. The sole genuine quality degradation is stump; garden is a data issue. C42 improves mean PSNR (+0.19 dB), maintains mean SSIM (0.000), and improves mean LPIPS (−0.006) — the first baseline where C42 produces a net quality improvement.

The classification remains PARTIALLY_ADDITIVE because (a) the strict preservation criterion is the standard used across all C42 benchmarks for consistency, and (b) the inconsistent Gaussian reduction (69.2%, with 4 scenes increasing) does not meet the "consistently reducing Gaussians" bar. However, the underlying dynamic is **quality-improving**, not quality-degrading — a fundamentally different failure mode than on FastGS or Speedy-Splat.

**Path to BROADLY_ADDITIVE:** Two changes would flip the classification: (1) The Faster-GS native baseline within NeRFICG appears underperforming (native PSNR values are 1–4 dB below FastGS and Speedy-Splat on most scenes), and C42's regularization corrects this; if the native baseline were better optimized, the quality changes would be smaller and more would fall within the preservation band. (2) The inconsistent Gaussian reduction suggests C42's interaction with NeRFICG's densification (interval=100, threshold=0.0002) is scene-dependent — a densification schedule tuned for C42 could improve consistency.

---

## 9. Key Findings

1. **C42 improves mean quality — a first across baselines.** Mean PSNR improves by +0.19 dB, mean SSIM is unchanged (0.000), and mean LPIPS improves by −0.006. On FastGS, all three means degraded (−0.18, −0.014, +0.013). On Speedy-Splat, all three degraded (−0.09, −0.013, +0.010). Faster-GS is the first baseline where C42 produces a net quality improvement, suggesting the native Faster-GS loss within NeRFICG is suboptimally tuned and C42's downsampled DSSIM acts as a beneficial regularizer.

2. **C42 improves PSNR in 61.5% of scenes (8/13).** This is dramatically higher than FastGS (15.4%, 2/13) and Speedy-Splat (23.1%, 3/13). The largest improvements are room (+1.99 dB), truck (+0.64 dB), and garden (+0.50 dB, data issue). C42 also improves SSIM in 8/13 scenes and LPIPS in 8/13 scenes — the half-resolution SSIM shifts optimization weight toward full-resolution L1, directly benefiting pixel-level fidelity.

3. **Stump is the sole genuine quality degradation.** Among the 5 scenes failing the strict joint criterion, only stump (ΔPSNR −0.94, ΔSSIM −0.040) represents C42 making quality worse. The other failures — bicycle (SSIM improved), room (both improved), truck (PSNR improved) — fail because C42 **improves** quality beyond the preservation band. Garden is a data-quality outlier. Under a non-degradation criterion, 84.6% of scenes pass.

4. **Loss-equivalence is exact.** The forward pass and backward pass are both bit-exact (loss diff = 0, L1 diff = 0, DSSIM diff = 0, gradient cosine = 1.0, gradient relative L2 = 0). This is stronger than FastGS, which showed gradient cosine = 0.9997 due to fused_ssim CUDA kernel non-determinism. The Faster-GS/NeRFICG implementation uses deterministic `fused_dssim` from `Optim.Losses.DSSIM`, achieving perfect mathematical equivalence at scale = 1.0.

5. **Gaussian reduction is inconsistent.** C42 reduces Gaussians in only 9/13 scenes (69.2%), with 4 scenes showing increases (flowers +6.42%, garden +11.50%, counter +1.54%, bonsai +9.31%). The total reduction is −571,125 Gaussians (−2.37%), much smaller than FastGS (−13.05%) or Speedy-Splat (−12.5%). The inconsistency likely reflects C42's interaction with NeRFICG's aggressive densification (interval=100, threshold=0.0002): the downsampled DSSIM gradient may allow more Gaussians to pass the densification threshold in some scenes, as the coarser SSIM signal has less pruning pressure on high-frequency regions.

6. **No wall-time speedup.** The geometric mean speedup is 0.998× (near parity). Excluding the garden data outlier, C42 is 3.5% slower (0.965×). Eight of 13 scenes show wall-time increases, with bonsai (+20.81%) and room (+13.10%) the worst. The per-iteration `F.interpolate(scale_factor=0.5, mode="area")` overhead is not offset by Gaussian reduction savings, since the reduction is small and inconsistent. The NeRFICG framework's overhead and Faster-GS's frequent densification (every 100 iterations) further mask any training-speed benefit.

7. **Room is a dramatic C42 success story.** The +1.99 dB PSNR improvement in room is the largest single-scene quality gain in any C42 benchmark. SSIM also improves by +0.028 and LPIPS by −0.028. This suggests the native Faster-GS loss was poorly calibrated for room's indoor geometry, and C42's structural downsampling corrects the optimization trajectory. Room "fails" the strict criterion only because the improvement exceeds the ±0.3 dB / ±0.02 band.

8. **Truck is an across-the-board improvement.** C42 improves all three quality metrics (ΔPSNR +0.64, ΔSSIM +0.010, ΔLPIPS −0.002), reduces Gaussians by −19.21%, and speeds up wall time by −14.31%. This is the ideal C42 profile — better quality, fewer Gaussians, faster training — yet it "fails" strict preservation because the PSNR gain exceeds ±0.3 dB.

9. **Indoor scenes show quality improvement, not just preservation.** The MipNeRF360 indoor group has the highest group mean ΔPSNR (+0.45 dB) and ΔSSIM (+0.007). On FastGS and Speedy-Splat, indoor scenes were the "sweet spot" for preservation (minimal quality loss, moderate Gaussian reduction). On Faster-GS, C42 goes beyond preservation to active improvement, but at the cost of increased wall time (+12.45%) and slightly increased Gaussian count (+1.21%).

10. **Deep Blending is the most stable group.** Both drjohnson and playroom pass the strict joint criterion (2/2, 100%), with small quality changes, moderate Gaussian reduction (mean −4.99%), and near-zero wall-time change (mean +0.67%). On FastGS, this group was the most volatile (playroom −0.79 dB); on Faster-GS, it is the most stable, highlighting C42's baseline-dependent behavior.

11. **Stump is the worst-case scene — consistent across baselines.** On all three baselines (FastGS, Speedy-Splat, Faster-GS), stump is among the worst C42 performers. On Faster-GS it is the sole genuine degradation (ΔPSNR −0.94, ΔSSIM −0.040). Stump's dense high-frequency foliage is inherently challenging for structural downsampling — the half-resolution DSSIM cannot capture fine branch/leaf detail, causing systematic quality loss. This is a C42 mechanism limitation, not a baseline-specific issue.

12. **SSIM does not systematically degrade.** Unlike FastGS (all 13 scenes showed negative ΔSSIM, mean −0.014) and Speedy-Splat (all 13 negative, mean −0.013), Faster-GS shows **mixed** SSIM deltas: 8 scenes improve, 5 degrade, with a mean of exactly 0.000. This is because the native Faster-GS SSIM values are already depressed by suboptimal training, and C42's regularization recovers structural fidelity in many scenes rather than sacrificing it. Faster-GS's frequent densification (every 100 iters) with low gradient threshold means the coarser SSIM signal has minimal impact on the densification-driven training dynamics.

13. **Garden is a data-quality outlier.** Both native (PSNR 13.40) and C42 (PSNR 13.90) produce very low metrics for garden due to manually-created COLMAP data. The +11.50% Gaussian increase and anomalous −32.90% wall-time speedup are artifacts of the degraded input data, not C42 behavior. Garden results should be excluded from cross-renderer comparisons.

---

## 10. Provenance

| Field | Value |
|---|---|
| **Report date** | 2026-09-17 |
| **Report file** | `reports/fastergs-c42-13scene-final.md` |
| **Data source** | `results/faster-gs-all-metrics.json` |
| **Baseline renderer** | Faster-GS |
| **Baseline repo** | `nerficg-project/faster-gaussian-splatting` |
| **Baseline commit** | `3cb0b75` |
| **Baseline branch** | `main` |
| **Framework** | NeRFICG (config-driven via YAML, `scripts/train.py -c config.yaml`) |
| **C42 patch files** | `Loss.py` (C42_SCALE attribute + conditional downsample in forward), `Trainer.py` (C42_SCALE=1.0 in LOSS ConfigParameterList) |
| **Conda environment** | `nerficg` (Python 3.11, PyTorch 2.7.1+cu118, CUDA 11.8) |
| **GPU** | NVIDIA mx A100-PCIE-40GB |
| **Training iterations** | 30,000 per scene |
| **Training speed** | ~70–140 it/s (scene-dependent) |
| **Loss weights** | λ(DSSIM) = 0.2, λ(L1) = 0.8 |
| **C42 parameters** | scale_factor = 0.5, mode = "area" |
| **Densification interval** | 100 |
| **Densification grad threshold** | 0.0002 |
| **PPISP** | Disabled (USE: false) |
| **IMAGE_SCALE_FACTOR** | 0.25 (MipNeRF360), 1.0 (T&T/DB) |
| **Scenes** | 13 (5 MipNeRF360 outdoor, 4 MipNeRF360 indoor, 2 Tanks&Temples, 2 Deep Blending) |
| **Evaluation** | NeRFICG built-in (renders test views, computes PSNR/SSIM/LPIPS via torchmetrics) |
| **Gaussian count** | Parsed from train.log "final number of Gaussians" line |
| **Loss-equivalence test** | PASSED (scale = 1.0: loss diff = 0, L1 diff = 0, DSSIM diff = 0, grad cosine = 1.0, grad rel L2 = 0) — **exact** |
| **Classification** | **PARTIALLY_ADDITIVE** (61.5% strict quality-preserved, 84.6% non-degraded, 69.2% Gaussian-reduced, 0.998× geomean speedup, mean ΔPSNR +0.19 dB) |
| **Benchmark repo** | `C:\Users\36570\3dgs-renderer-benchmark` |
