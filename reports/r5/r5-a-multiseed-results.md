# R5-A: Multi-Seed Results

## Complete Paired Results (2 scenes × 3 seeds × 2 methods = 12 runs)

### Final 30K Metrics

| Scene | Seed | Method | PSNR (dB) | SSIM | L1 | N_Gaussians |
|-------|------|--------|-----------|------|----|-------------|
| train | 0 | B | 21.86 | 0.8313 | 0.0462 | 419,478 |
| train | 0 | C | 22.38 | 0.8183 | 0.0447 | 476,167 |
| train | 1 | B | 22.42 | 0.8328 | 0.0464 | 413,455 |
| train | 1 | C | 21.77 | 0.8135 | — | 446,052 |
| train | 2 | B | 23.32 | 0.8417 | 0.0424 | 409,375 |
| train | 2 | C | 22.16 | 0.8194 | — | 471,798 |
| truck | 0 | B | 22.33 | 0.8520 | 0.0356 | 1,074,296 |
| truck | 0 | C | 24.28 | 0.8479 | — | 1,181,561 |
| truck | 1 | B | 24.58 | 0.8650 | 0.0354 | 1,089,627 |
| truck | 1 | C | 24.58 | 0.8491 | — | 1,194,540 |
| truck | 2 | B | 22.71 | 0.8568 | 0.0445 | 1,068,903 |
| truck | 2 | C | 23.53 | 0.8480 | — | 1,217,421 |

### Paired Deltas (ΔPSNR = C − B)

| Scene | Seed | B_PSNR | C_PSNR | ΔPSNR | ΔSSIM | R_GS (C/B) | Positive? |
|-------|------|--------|--------|-------|-------|------------|-----------|
| train | 0 | 21.86 | 22.38 | **+0.52** | −0.0130 | 1.135 | ✅ YES |
| train | 1 | 22.42 | 21.77 | **−0.64** | −0.0192 | 1.079 | ❌ NO |
| train | 2 | 23.32 | 22.16 | **−1.16** | −0.0224 | 1.152 | ❌ NO |
| truck | 0 | 22.33 | 24.28 | **+1.94** | −0.0042 | 1.100 | ✅ YES |
| truck | 1 | 24.58 | 24.58 | **−0.00** | −0.0159 | 1.096 | ❌ NO (tie) |
| truck | 2 | 22.71 | 23.53 | **+0.81** | −0.0088 | 1.139 | ✅ YES |

### Per-Scene Summary

| Scene | Seeds | Mean ΔPSNR | Median ΔPSNR | Std ΔPSNR | Min | Max | Positive | 95% CI |
|-------|-------|------------|--------------|-----------|-----|-----|----------|--------|
| train | 3 | **−0.43 dB** | −0.64 | 0.86 | −1.16 | +0.52 | 1/3 | [−2.57, +1.71] |
| truck | 3 | **+0.92 dB** | +0.81 | 0.97 | −0.00 | +1.94 | 2/3 | [−1.50, +3.34] |

### Combined T&T

| Metric | Value |
|--------|-------|
| Total paired observations | 6 |
| Combined mean ΔPSNR | +0.24 dB |
| Positive observations | 3/6 (50%) |
| Train positive | 1/3 |
| Truck positive | 2/3 |
| Mean ΔSSIM (all) | −0.0139 |
| Mean R_GS (all) | 1.117 |

## Key Observations

1. **R4 train gain is NOT reproducible**: Seed 0 showed +0.52 dB, but seeds 1 and 2 both showed negative ΔPSNR (−0.64 and −1.16). The seed 0 positive was a chance occurrence.

2. **Truck shows partial reproducibility**: 2/3 seeds positive (seed 0: +1.94, seed 2: +0.81), but seed 1 was exactly 0.00. However, 2/3 positive does not meet the PASS_STRONG threshold of ≥2/3 AND both scenes positive.

3. **SSIM always degrades**: Every single paired comparison shows ΔSSIM < 0 (mean −0.0139). Candidate C consistently hurts structural similarity even when PSNR improves.

4. **Gaussian count always increases**: R_GS > 1.0 in all 6 pairs (mean 1.117). Candidate C produces 7-15% more Gaussians than baseline, suggesting the skip mechanism affects densification dynamics.

5. **High variance**: Std ΔPSNR is 0.86 (train) and 0.97 (truck), comparable to or larger than the mean effects. The 95% CIs for both scenes cross zero.

## Data Provenance

- Seed 0: R4 original runs (provenance-verified, see r5-a0-r4-seed-config-audit.md)
- Seeds 1-2: R5-A v2 runs (launched 2026-09-19 01:19, completed 2026-09-19 ~03:00)
- All runs: 30K iterations, identical config, A100-PCIE-40GB, gsplat 1.5.3, PyTorch 2.4.1+cu124
- No checkpoint saving (disk full mitigation); PSNR extracted from training logs and training_metrics.json
