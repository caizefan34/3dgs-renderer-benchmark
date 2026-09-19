# R5-A: Multi-Seed Results

## Complete Paired Results (2 scenes × 3 seeds × 2 methods = 12 runs)

### Final 30K Metrics

| Scene | Seed | Method | PSNR (dB) | SSIM | N_Gaussians | Source |
|-------|------|--------|-----------|------|-------------|--------|
| train | 0 | B | 21.8608 | 0.8313 | 419,478 | final_results.json (R4 original) |
| train | 0 | C | 22.3361 | 0.8166 | 472,289 | final_results.json (R4 original) |
| train | 1 | B | 22.4175 | 0.8328 | 413,455 | R5-A training_metrics.json |
| train | 1 | C | 21.7741 | 0.8135 | 446,052 | R5-A training_metrics.json |
| train | 2 | B | 23.3203 | 0.8417 | 409,375 | R5-A training_metrics.json |
| train | 2 | C | 22.1556 | 0.8194 | 471,798 | R5-A training_metrics.json |
| truck | 0 | B | 22.3348 | 0.8520 | 1,074,296 | final_results.json (R4 original) |
| truck | 0 | C | 24.4609 | 0.8506 | 1,185,928 | final_results.json (R4 original) |
| truck | 1 | B | 24.5792 | 0.8650 | 1,089,627 | R5-A training_metrics.json |
| truck | 1 | C | 24.5789 | 0.8491 | 1,194,540 | R5-A training_metrics.json |
| truck | 2 | B | 22.7138 | 0.8568 | 1,068,903 | R5-A training_metrics.json |
| truck | 2 | C | 23.5266 | 0.8480 | 1,217,421 | R5-A training_metrics.json |

> **Note**: Seed-0 candidate_c `training_metrics.json` in the R4 directory was overwritten by an
> unidentified re-run (see `r5-a-provenance-reconciliation.md`). Seed-0 numbers are taken from
> `final_results.json` (mtime 2026-09-18 23:58:04, sha256 prefix `ea645ac3a0f61f1b`), which is the
> authoritative R4 original artifact. Seeds 1–2 are from fresh R5-A runs
> (`/mnt/storage_pool/liaoyuanjun/r5_a/{scene}_seed{seed}/{method}/training_metrics.json`).

### Paired Deltas (ΔPSNR = C − B)

| Scene | Seed | B_PSNR | C_PSNR | ΔPSNR | ΔSSIM | R_GS (C/B) | Positive? |
|-------|------|--------|--------|-------|-------|------------|-----------|
| train | 0 | 21.8608 | 22.3361 | **+0.4752** | −0.0146 | 1.126 | ✅ YES |
| train | 1 | 22.4175 | 21.7741 | **−0.6433** | −0.0192 | 1.079 | ❌ NO |
| train | 2 | 23.3203 | 22.1556 | **−1.1648** | −0.0224 | 1.152 | ❌ NO |
| truck | 0 | 22.3348 | 24.4609 | **+2.1261** | −0.0014 | 1.103 | ✅ YES |
| truck | 1 | 24.5792 | 24.5789 | **−0.0003** | −0.0159 | 1.096 | ❌ NO (tie) |
| truck | 2 | 22.7138 | 23.5266 | **+0.8128** | −0.0088 | 1.139 | ✅ YES |

### Per-Scene Summary

| Scene | Seeds | Mean ΔPSNR | Median ΔPSNR | Std ΔPSNR | Min | Max | Positive | 95% CI |
|-------|-------|------------|--------------|-----------|-----|-----|----------|--------|
| train | 3 | **−0.44 dB** | −0.64 | 0.86 | −1.16 | +0.48 | 1/3 | [−2.59, +1.71] |
| truck | 3 | **+0.98 dB** | +0.81 | 1.05 | −0.00 | +2.13 | 2/3 | [−1.62, +3.57] |

### Combined T&T

| Metric | Value |
|--------|-------|
| Total paired observations | 6 |
| Combined mean ΔPSNR | +0.27 dB |
| Positive observations | 3/6 (50%) |
| Train positive | 1/3 |
| Truck positive | 2/3 |
| Mean ΔSSIM (all) | −0.0137 |
| Mean R_GS (all) | 1.116 |

## Key Observations

1. **R4 train gain is NOT reproducible**: Seed 0 showed +0.48 dB, but seeds 1 and 2 both showed negative ΔPSNR (−0.64 and −1.16). The seed 0 positive was a chance occurrence.

2. **Truck shows partial reproducibility**: 2/3 seeds positive (seed 0: +2.13, seed 2: +0.81), but seed 1 was −0.0003 (effectively zero). However, 2/3 positive does not meet the PASS_STRONG threshold of ≥2/3 AND both scenes positive.

3. **SSIM always degrades**: Every single paired comparison shows ΔSSIM < 0 (mean −0.0137). Candidate C consistently hurts structural similarity even when PSNR improves. (See statistical analysis for exact binomial test.)

4. **Gaussian count always increases**: R_GS > 1.0 in all 6 pairs (mean 1.116). Candidate C produces 8–15% more Gaussians than baseline, suggesting the skip mechanism affects densification dynamics.

5. **High variance**: Std ΔPSNR is 0.86 (train) and 1.05 (truck), comparable to or larger than the mean effects. The 95% CIs for both scenes cross zero.

## Data Provenance

- **Seed 0**: `final_results.json` (R4 original run, mtime 2026-09-18 23:58:04, sha256 prefix `ea645ac3a0f61f1b`).
  Path: `/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/final_results.json`.
  Provenance verified: see `r5-a0-r4-seed-config-audit.md` and `r5-a-provenance-reconciliation.md`.
- **Seeds 1–2**: R5-A v2 runs (launched 2026-09-19 01:19, completed 2026-09-19 ~03:00).
  Path: `/mnt/storage_pool/liaoyuanjun/r5_a/{scene}_seed{seed}/{method}/training_metrics.json`.
- All runs: 30K iterations, identical config, A100-PCIE-40GB, gsplat 1.5.3, PyTorch 2.4.1+cu124.
- No checkpoint saving for seeds 1–2 (disk full mitigation; `checkpoint_iterations = ()`).
