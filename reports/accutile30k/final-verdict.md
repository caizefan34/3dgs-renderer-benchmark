# Final Verdict — B1A 13-Scene 30K Full-Training Validation

## Gate: `B1A_STRONG_PASS`

## Promotion: YES — B1A = FROZEN ENHANCED PAPER BASELINE

## Gate criteria evaluation

| Criterion | Required | Observed | Pass |
|-----------|----------|----------|:----:|
| 13/13 runs complete | 13/13 | 13/13 | ✅ |
| geomean full-training speedup > 1.0 | >1.0 | 1.1039× | ✅ |
| majority scenes faster | ≥7/13 | 13/13 | ✅ |
| no systematic quality degradation | mean ΔPSNR > -0.2 | 0.110 dB | ✅ |
| no large convergence failure | ≤1 scene ΔPSNR < -0.5 | 1 | ✅ |

## Worst quality-delta scene

- garden: ΔPSNR = -0.714 dB

## Scenes with runtime regression (speedup ≤ 1.0)

- none

## Classification

AccuTile is **prior art** (Speedy-Splat). Correct wording on success: `AccuTile-enhanced gsplat baseline` / `strong prior-art baseline`. AccuTile is NOT our contribution.

## Final paper-ready table

| Scene | Dataset | B1 PSNR | B1A PSNR | ΔPSNR | B1 SSIM | B1A SSIM | B1 Time (min) | B1A Time (min) | Speedup | B1 N_GS | B1A N_GS |
|-------|---------|--------:|---------:|------:|--------:|---------:|--------------:|---------------:|--------:|--------:|---------:|
| bicycle | Mip-NeRF360 | 26.36 | 26.35 | -0.008 | 0.8363 | 0.8370 | 42.5 | 38.3 | 1.109× | 3924004 | 3935720 |
| bonsai | Mip-NeRF360 | 30.14 | 31.35 | 1.210 | 0.9383 | 0.9413 | 30.9 | 27.4 | 1.127× | 862423 | 863106 |
| counter | Mip-NeRF360 | 29.48 | 29.82 | 0.345 | 0.9092 | 0.9101 | 31.6 | 28.3 | 1.117× | 729159 | 704544 |
| flowers | Mip-NeRF360 | 24.37 | 24.34 | -0.030 | 0.7620 | 0.7628 | 37.9 | 33.9 | 1.118× | 2682713 | 2691413 |
| garden | Mip-NeRF360 | 24.85 | 24.13 | -0.714 | 0.7550 | 0.7296 | 38.7 | 34.8 | 1.113× | 2666504 | 2516606 |
| kitchen | Mip-NeRF360 | 31.57 | 31.63 | 0.059 | 0.9304 | 0.9312 | 34.3 | 30.9 | 1.110× | 936492 | 917918 |
| room | Mip-NeRF360 | 32.00 | 32.04 | 0.045 | 0.9270 | 0.9256 | 31.2 | 28.3 | 1.103× | 933243 | 937768 |
| stump | Mip-NeRF360 | 30.62 | 30.60 | -0.021 | 0.8924 | 0.8930 | 36.1 | 33.1 | 1.090× | 2809518 | 2850251 |
| treehill | Mip-NeRF360 | 25.73 | 25.76 | 0.023 | 0.8277 | 0.8280 | 38.7 | 35.5 | 1.091× | 3007385 | 2983953 |
| train | Tanks & Temples | 19.45 | 19.71 | 0.255 | 0.7504 | 0.7556 | 32.3 | 29.4 | 1.099× | 439499 | 444544 |
| truck | Tanks & Temples | 24.95 | 24.77 | -0.189 | 0.8726 | 0.8707 | 32.7 | 29.7 | 1.098× | 1154151 | 1122997 |
| drjohnson | Deep Blending | 25.04 | 25.30 | 0.253 | 0.8632 | 0.8707 | 29.3 | 27.1 | 1.084× | 673784 | 683304 |
| playroom | Deep Blending | 23.09 | 23.29 | 0.195 | 0.9061 | 0.9116 | 31.1 | 28.5 | 1.092× | 958420 | 948581 |

| **All-13 geomean** | | | | **0.110** | | | | | **1.1039×** | | |
