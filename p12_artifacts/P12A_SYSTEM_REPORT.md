# P12A SYSTEM Report — Strong-Baseline Revalidation (current gsplat vs C0)

**Question**: Under the P11-validated fused-loss baseline, is the frozen C0 system still faster than current gsplat main under the strict matched 13-scene 30K protocol?
**Common loss**: P11-validated fused-SSIM (padding="same"), L = 0.8*L1 + 0.2*(1-SSIM); eval SSIM metric = SepSSIM (frozen).

## Per-scene wall-clock (R_sys = GMAIN_FUSED/C0_FUSED)

| scene | GMAIN_FUSED wall (min) | C0_FUSED wall (min) | R_sys | ΔPSNR | ΔSSIM | ΔLPIPS | N-ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 16.1 | 14.3 | **1.1296** | -0.10 | +0.0000 | -0.0005 | 1.012 |
| bonsai | 9.6 | 8.1 | **1.1826** | +0.10 | +0.0004 | -0.0000 | 1.058 |
| counter | 9.8 | 8.9 | **1.1098** | +0.00 | +0.0005 | -0.0012 | 1.012 |
| drjohnson | 9.6 | 8.3 | **1.1553** | -1.03 | -0.0120 | +0.0199 | 0.871 |
| flowers | 13.2 | 12.0 | **1.0931** | +0.09 | +0.0033 | -0.0023 | 1.020 |
| garden | 14.7 | 13.3 | **1.1063** | +0.18 | +0.0109 | -0.0129 | 0.902 |
| kitchen | 11.8 | 10.6 | **1.1212** | -0.18 | +0.0001 | -0.0010 | 1.027 |
| playroom | 9.1 | 7.7 | **1.1849** | -0.09 | +0.0006 | -0.0012 | 1.029 |
| room | 9.7 | 8.5 | **1.1482** | +0.01 | +0.0004 | -0.0004 | 0.994 |
| stump | 12.7 | 10.9 | **1.1675** | +0.12 | +0.0031 | -0.0054 | 1.101 |
| train | 11.9 | 10.2 | **1.1636** | -0.04 | +0.0006 | -0.0008 | 1.013 |
| treehill | 14.5 | 12.7 | **1.1432** | +0.12 | +0.0030 | -0.0033 | 1.037 |
| truck | 11.0 | 9.5 | **1.1632** | +0.16 | +0.0025 | -0.0036 | 1.035 |

- geometric mean **1.1434**; median 1.1482; min 1.0931; max 1.1849
- 13/13 scenes with R_sys > 1
- 95% CI on GM (log-ratio t, df=12): [1.1256, 1.1615] (entirely > 1)
- equivalent C0 wall-time reduction: 12.54%
- quality: mean ΔPSNR -0.051 dB (median +0.011), mean ΔSSIM +0.00105, mean ΔLPIPS -0.00098; preregistered quality-neutral band: PASS
- population: final-N ratio spans [0.871, 1.101]; corr(log R_sys, log N-ratio) = 0.299

## Old reference (descriptive only, never pooled)
- P8 strict matched (SepSSIM common loss): GM 1.0979, positive 13/13
- new fused-loss GM 1.1434; Δlog-ratio +0.0406; relative change +4.14%

## Verdict
**SYSTEM_REVALIDATED_STRONG**
