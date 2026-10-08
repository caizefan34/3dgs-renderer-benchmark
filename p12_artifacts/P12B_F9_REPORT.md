# P12B F9 Report — Strong-Baseline Revalidation (A0 vs A1 (F9))

**Question**: Does the original F9 materialization-free producer retain a reproducible full-training benefit after the common SepSSIM bottleneck is removed?
**Common loss**: P11-validated fused-SSIM (padding="same"), L = 0.8*L1 + 0.2*(1-SSIM); eval SSIM metric = SepSSIM (frozen).

## Per-scene wall-clock (R_F9 = A0_FUSED/A1_FUSED)

| scene | A0_FUSED wall (min) | A1_FUSED wall (min) | R_F9 | ΔPSNR | ΔSSIM | ΔLPIPS | N-ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 14.5 | 14.2 | **1.0165** | -0.10 | -0.0009 | +0.0006 | 1.005 |
| bonsai | 8.2 | 8.1 | **1.0125** | +0.13 | +0.0003 | -0.0008 | 0.982 |
| counter | 8.8 | 8.6 | **1.0262** | -0.03 | -0.0004 | +0.0005 | 0.992 |
| drjohnson | 8.5 | 8.3 | **1.0276** | -0.41 | -0.0065 | +0.0103 | 1.028 |
| flowers | 12.2 | 12.0 | **1.0172** | +0.00 | -0.0005 | +0.0001 | 1.004 |
| garden | 13.7 | 13.3 | **1.0365** | +0.45 | +0.0104 | -0.0086 | 1.077 |
| kitchen | 10.8 | 10.6 | **1.0184** | +0.00 | -0.0002 | -0.0002 | 1.005 |
| playroom | 7.8 | 7.6 | **1.0135** | +0.05 | -0.0001 | +0.0003 | 1.004 |
| room | 8.6 | 8.5 | **1.0028** | -0.05 | -0.0013 | +0.0018 | 0.993 |
| stump | 11.0 | 10.9 | **1.0069** | -0.07 | +0.0002 | -0.0010 | 0.980 |
| train | 10.4 | 10.3 | **1.0112** | +0.03 | -0.0014 | +0.0016 | 1.013 |
| treehill | 12.9 | 12.7 | **1.0160** | +0.08 | +0.0011 | -0.0012 | 0.999 |
| truck | 9.7 | 9.6 | **1.0188** | +0.02 | -0.0004 | +0.0010 | 1.012 |

- geometric mean **1.0172**; median 1.0165; min 1.0028; max 1.0365
- 13/13 scenes with R_F9 > 1
- 95% CI on GM (log-ratio t, df=12): [1.0118, 1.0226] (entirely > 1)
- equivalent A1 wall-time reduction: 1.69%
- quality: mean ΔPSNR +0.008 dB (median +0.004), mean ΔSSIM +0.00004, mean ΔLPIPS +0.00033; preregistered quality-neutral band: PASS
- population: final-N ratio spans [0.980, 1.077]; corr(log R_F9, log N-ratio) = 0.765

## Old reference (descriptive only, never pooled)
- P2 historical SXM (SepSSIM common loss): GM 1.0111, positive 13/13
- new fused-loss GM 1.0172; Δlog-ratio +0.0060; relative change +0.60%

## Verdict
**F9_REVALIDATED_STRONG**
