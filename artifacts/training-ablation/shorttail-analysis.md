# HiGS Short-Tail Exploration Matrix (4 methods x 11 scenes x seed 0) (paired vs gsplat)

- baseline: `gsplat`; configs: gsplat_25k, higs_visible_24k, higs_visible_25k, higs_visible_27k
- exploration jobs: 44; control (gsplat) cells: 33

## Paired NI + acceleration gates (scene-block bootstrap 95% CI)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_25k | psnr_db | -0.1283 | [-0.3352, 0.0553] | >= -0.10 | False |
| gsplat_25k | ssim | -0.0022 | [-0.0050, 0.0001] | >= -0.003 | False |
| gsplat_25k | lpips | 0.0047 | [0.0009, 0.0094] | <= +0.005 | False |
| gsplat_25k | time_to_quality_seconds | -156.1176 | [-233.0352, -84.1792] | <= 0 | True |
| gsplat_25k | speedup ratio | 1.215 | CI lo 1.176 | mean>=1.111 & lo>1.0 | True |
| gsplat_25k | TTQ delta (s) | -156.12 | [-233.04, -84.18] | <= 0 | True |
| gsplat_25k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3511 | - | descriptive | - |
| gsplat_25k | energy_joules (gsplat / cand mean) | 222903 / 179196 | - | descriptive | - |
| gsplat_25k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2268333 | - | descriptive | - |

| higs_visible_24k | psnr_db | -0.2622 | [-0.3844, -0.1168] | >= -0.10 | False |
| higs_visible_24k | ssim | -0.0047 | [-0.0086, -0.0010] | >= -0.003 | False |
| higs_visible_24k | lpips | 0.0090 | [0.0037, 0.0161] | <= +0.005 | False |
| higs_visible_24k | time_to_quality_seconds | -129.6143 | [-238.7745, -22.1907] | <= 0 | True |
| higs_visible_24k | speedup ratio | 1.263 | CI lo 1.042 | mean>=1.111 & lo>1.0 | True |
| higs_visible_24k | TTQ delta (s) | -129.61 | [-238.77, -22.19] | <= 0 | True |
| higs_visible_24k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3195 | - | descriptive | - |
| higs_visible_24k | energy_joules (gsplat / cand mean) | 222903 / 159454 | - | descriptive | - |
| higs_visible_24k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2091808 | - | descriptive | - |

| higs_visible_25k | psnr_db | -0.1976 | [-0.3519, -0.0286] | >= -0.10 | False |
| higs_visible_25k | ssim | -0.0046 | [-0.0085, -0.0008] | >= -0.003 | False |
| higs_visible_25k | lpips | 0.0085 | [0.0036, 0.0148] | <= +0.005 | False |
| higs_visible_25k | time_to_quality_seconds | -126.5422 | [-213.6125, -48.9969] | <= 0 | True |
| higs_visible_25k | speedup ratio | 1.213 | CI lo 1.002 | mean>=1.111 & lo>1.0 | True |
| higs_visible_25k | TTQ delta (s) | -126.54 | [-213.61, -49.00] | <= 0 | True |
| higs_visible_25k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3255 | - | descriptive | - |
| higs_visible_25k | energy_joules (gsplat / cand mean) | 222903 / 170548 | - | descriptive | - |
| higs_visible_25k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2131226 | - | descriptive | - |

| higs_visible_27k | psnr_db | -0.1838 | [-0.4084, 0.0222] | >= -0.10 | False |
| higs_visible_27k | ssim | -0.0038 | [-0.0076, -0.0002] | >= -0.003 | False |
| higs_visible_27k | lpips | 0.0076 | [0.0029, 0.0133] | <= +0.005 | False |
| higs_visible_27k | time_to_quality_seconds | -100.1232 | [-189.9954, -10.7313] | <= 0 | True |
| higs_visible_27k | speedup ratio | 1.123 | CI lo 0.920 | mean>=1.111 & lo>1.0 | False |
| higs_visible_27k | TTQ delta (s) | -100.12 | [-190.00, -10.73] | <= 0 | True |
| higs_visible_27k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3291 | - | descriptive | - |
| higs_visible_27k | energy_joules (gsplat / cand mean) | 222903 / 184554 | - | descriptive | - |
| higs_visible_27k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2157020 | - | descriptive | - |

## Per-scene speedup ratio (gsplat wall / candidate wall, seed 0)

| config | scene | speedup | gsplat s | cand s |
|---|---|---|---|---|
| gsplat_25k | deep_blending/drjohnson | 1.241 | 623 | 502 |
| gsplat_25k | deep_blending/playroom | 1.225 | 489 | 399 |
| gsplat_25k | mipnerf360/bicycle | 1.230 | 1737 | 1412 |
| gsplat_25k | mipnerf360/bonsai | 1.176 | 729 | 619 |
| gsplat_25k | mipnerf360/counter | 1.192 | 740 | 621 |
| gsplat_25k | mipnerf360/garden | 1.219 | 1748 | 1435 |
| gsplat_25k | mipnerf360/kitchen | 1.191 | 801 | 672 |
| gsplat_25k | mipnerf360/room | 1.236 | 766 | 620 |
| gsplat_25k | mipnerf360/stump | 1.212 | 1574 | 1300 |
| gsplat_25k | tanks_and_temples/train | 1.184 | 516 | 436 |
| gsplat_25k | tanks_and_temples/truck | 1.264 | 498 | 394 |
| higs_visible_24k | deep_blending/drjohnson | 1.315 | 623 | 473 |
| higs_visible_24k | deep_blending/playroom | 1.288 | 489 | 380 |
| higs_visible_24k | mipnerf360/bicycle | 1.388 | 1737 | 1251 |
| higs_visible_24k | mipnerf360/bonsai | 1.228 | 729 | 593 |
| higs_visible_24k | mipnerf360/counter | 1.232 | 740 | 601 |
| higs_visible_24k | mipnerf360/garden | 1.347 | 1748 | 1298 |
| higs_visible_24k | mipnerf360/kitchen | 1.207 | 801 | 663 |
| higs_visible_24k | mipnerf360/room | 1.290 | 766 | 594 |
| higs_visible_24k | mipnerf360/stump | 1.354 | 1574 | 1163 |
| higs_visible_24k | tanks_and_temples/train | 1.042 | 516 | 495 |
| higs_visible_24k | tanks_and_temples/truck | 1.198 | 498 | 415 |
| higs_visible_25k | deep_blending/drjohnson | 1.252 | 623 | 498 |
| higs_visible_25k | deep_blending/playroom | 1.229 | 489 | 398 |
| higs_visible_25k | mipnerf360/bicycle | 1.332 | 1737 | 1304 |
| higs_visible_25k | mipnerf360/bonsai | 1.184 | 729 | 615 |
| higs_visible_25k | mipnerf360/counter | 1.187 | 740 | 623 |
| higs_visible_25k | mipnerf360/garden | 1.291 | 1748 | 1354 |
| higs_visible_25k | mipnerf360/kitchen | 1.150 | 801 | 696 |
| higs_visible_25k | mipnerf360/room | 1.264 | 766 | 606 |
| higs_visible_25k | mipnerf360/stump | 1.297 | 1574 | 1214 |
| higs_visible_25k | tanks_and_temples/train | 1.002 | 516 | 515 |
| higs_visible_25k | tanks_and_temples/truck | 1.151 | 498 | 433 |
| higs_visible_27k | deep_blending/drjohnson | 1.149 | 623 | 542 |
| higs_visible_27k | deep_blending/playroom | 1.132 | 489 | 432 |
| higs_visible_27k | mipnerf360/bicycle | 1.227 | 1737 | 1415 |
| higs_visible_27k | mipnerf360/bonsai | 1.095 | 729 | 666 |
| higs_visible_27k | mipnerf360/counter | 1.103 | 740 | 670 |
| higs_visible_27k | mipnerf360/garden | 1.200 | 1748 | 1457 |
| higs_visible_27k | mipnerf360/kitchen | 1.069 | 801 | 749 |
| higs_visible_27k | mipnerf360/room | 1.176 | 766 | 652 |
| higs_visible_27k | mipnerf360/stump | 1.205 | 1574 | 1306 |
| higs_visible_27k | tanks_and_temples/train | 0.920 | 516 | 561 |
| higs_visible_27k | tanks_and_temples/truck | 1.072 | 498 | 464 |