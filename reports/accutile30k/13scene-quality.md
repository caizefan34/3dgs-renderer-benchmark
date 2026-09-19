# 13-Scene Quality — B1A vs B1 (30K)

Quality metrics from final evaluation (all cameras) at 30,000 iterations.

| Scene | Dataset | B1 PSNR | B1A PSNR | ΔPSNR | B1 SSIM | B1A SSIM | ΔSSIM | B1 LPIPS | B1A LPIPS | ΔLPIPS |
|-------|---------|--------:|---------:|------:|--------:|---------:|------:|---------:|----------:|-------:|
| bicycle | Mip-NeRF360 | 26.36 | 26.35 | -0.008 | 0.8363 | 0.8370 | 0.0007 | 0.2462 | 0.2453 | -0.0010 |
| bonsai | Mip-NeRF360 | 30.14 | 31.35 | 1.210 | 0.9383 | 0.9413 | 0.0030 | 0.2761 | 0.2712 | -0.0049 |
| counter | Mip-NeRF360 | 29.48 | 29.82 | 0.345 | 0.9092 | 0.9101 | 0.0009 | 0.2853 | 0.2827 | -0.0027 |
| flowers | Mip-NeRF360 | 24.37 | 24.34 | -0.030 | 0.7620 | 0.7628 | 0.0007 | 0.2929 | 0.2927 | -0.0003 |
| garden | Mip-NeRF360 | 24.85 | 24.13 | -0.714 | 0.7550 | 0.7296 | -0.0254 | 0.2963 | 0.3231 | 0.0267 |
| kitchen | Mip-NeRF360 | 31.57 | 31.63 | 0.059 | 0.9304 | 0.9312 | 0.0008 | 0.1745 | 0.1729 | -0.0016 |
| room | Mip-NeRF360 | 32.00 | 32.04 | 0.045 | 0.9270 | 0.9256 | -0.0014 | 0.3010 | 0.3011 | 0.0001 |
| stump | Mip-NeRF360 | 30.62 | 30.60 | -0.021 | 0.8924 | 0.8930 | 0.0006 | 0.2250 | 0.2244 | -0.0007 |
| treehill | Mip-NeRF360 | 25.73 | 25.76 | 0.023 | 0.8277 | 0.8280 | 0.0003 | 0.2962 | 0.2958 | -0.0004 |
| train | Tanks & Temples | 19.45 | 19.71 | 0.255 | 0.7504 | 0.7556 | 0.0051 | 0.4517 | 0.4483 | -0.0034 |
| truck | Tanks & Temples | 24.95 | 24.77 | -0.189 | 0.8726 | 0.8707 | -0.0019 | 0.2931 | 0.2958 | 0.0027 |
| drjohnson | Deep Blending | 25.04 | 25.30 | 0.253 | 0.8632 | 0.8707 | 0.0074 | 0.4655 | 0.4525 | -0.0130 |
| playroom | Deep Blending | 23.09 | 23.29 | 0.195 | 0.9061 | 0.9116 | 0.0055 | 0.4244 | 0.4109 | -0.0135 |

## Dataset aggregates

| Dataset | n | mean ΔPSNR | mean ΔSSIM | mean ΔLPIPS |
|---------|---|-----------:|-----------:|------------:|
| Mip-NeRF360 | 9 | 0.101 | -0.0022 | 0.0017 |
| Tanks & Temples | 2 | 0.033 | 0.0016 | -0.0003 |
| Deep Blending | 2 | 0.224 | 0.0065 | -0.0132 |
| **All 13** | 13 | **0.110** | **-0.0003** | **-0.0009** |

## Interpretation

AccuTile is an exact conservative workload optimization. The expected result is quality equivalence within normal FP/stochastic training variation, NOT PSNR improvement. Small positive/negative ΔPSNR values (< 0.2 dB) are within normal trajectory sensitivity from floating-point accumulation-order differences amplified by densification.
