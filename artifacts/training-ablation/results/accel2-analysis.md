# HiGS Accel2 Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.1428 | [-0.4059, 0.0667] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0033 | [-0.0092, 0.0003] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0064 | [0.0007, 0.0161] | <= +0.005 | False |
| gsplat_27k | time_to_quality_seconds | -110.2749 | [-179.6937, -53.6721] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.126 | CI lo 1.107 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3605 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 198051 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2331742 | - | descriptive | - |

| higs_tilesamp_27k_r05 | psnr_db | -1.2005 | [-2.1469, -0.6064] | >= -0.10 | False |
| higs_tilesamp_27k_r05 | ssim | -0.0291 | [-0.0438, -0.0159] | >= -0.003 | False |
| higs_tilesamp_27k_r05 | lpips | 0.0459 | [0.0212, 0.0751] | <= +0.005 | False |
| higs_tilesamp_27k_r05 | time_to_quality_seconds | -139.9396 | [-253.7316, -3.7418] | <= 0 | True |
| higs_tilesamp_27k_r05 | speedup ratio | 1.313 | CI lo 1.188 | mean>=1.111 & lo>1.0 | True |
| higs_tilesamp_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 1799 | - | descriptive | - |
| higs_tilesamp_27k_r05 | energy_joules (ctrl / cand mean) | 222903 / 150669 | - | descriptive | - |
| higs_tilesamp_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1148313 | - | descriptive | - |

| higs_tilesamp_27k_r07 | psnr_db | -0.0518 | [-0.1774, 0.0571] | >= -0.10 | False |
| higs_tilesamp_27k_r07 | ssim | -0.0018 | [-0.0055, 0.0007] | >= -0.003 | False |
| higs_tilesamp_27k_r07 | lpips | 0.0036 | [0.0002, 0.0089] | <= +0.005 | False |
| higs_tilesamp_27k_r07 | time_to_quality_seconds | -37.0954 | [-104.9828, 45.1294] | <= 0 | False |
| higs_tilesamp_27k_r07 | speedup ratio | 1.056 | CI lo 0.859 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_27k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3561 | - | descriptive | - |
| higs_tilesamp_27k_r07 | energy_joules (ctrl / cand mean) | 222903 / 210461 | - | descriptive | - |
| higs_tilesamp_27k_r07 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2299910 | - | descriptive | - |

| higs_tilesamp_30k_r07 | psnr_db | -0.1459 | [-0.3623, 0.0155] | >= -0.10 | False |
| higs_tilesamp_30k_r07 | ssim | -0.0011 | [-0.0032, 0.0006] | >= -0.003 | False |
| higs_tilesamp_30k_r07 | lpips | 0.0025 | [-0.0001, 0.0062] | <= +0.005 | False |
| higs_tilesamp_30k_r07 | time_to_quality_seconds | -15.9321 | [-77.8612, 29.5873] | <= 0 | False |
| higs_tilesamp_30k_r07 | speedup ratio | 0.962 | CI lo 0.831 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_30k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3666 | - | descriptive | - |
| higs_tilesamp_30k_r07 | energy_joules (ctrl / cand mean) | 222903 / 222232 | - | descriptive | - |
| higs_tilesamp_30k_r07 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2380325 | - | descriptive | - |

| higs_tilesamp_refine_27k_r05 | psnr_db | -1.0219 | [-1.8050, -0.5660] | >= -0.10 | False |
| higs_tilesamp_refine_27k_r05 | ssim | -0.0266 | [-0.0419, -0.0134] | >= -0.003 | False |
| higs_tilesamp_refine_27k_r05 | lpips | 0.0437 | [0.0185, 0.0737] | <= +0.005 | False |
| higs_tilesamp_refine_27k_r05 | time_to_quality_seconds | -144.5977 | [-254.2846, -18.1154] | <= 0 | True |
| higs_tilesamp_refine_27k_r05 | speedup ratio | 1.320 | CI lo 1.180 | mean>=1.111 & lo>1.0 | True |
| higs_tilesamp_refine_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 1789 | - | descriptive | - |
| higs_tilesamp_refine_27k_r05 | energy_joules (ctrl / cand mean) | 222903 / 149329 | - | descriptive | - |
| higs_tilesamp_refine_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1137781 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.135 | 623 | 549 |
| gsplat_27k | deep_blending/playroom | 1.112 | 489 | 440 |
| gsplat_27k | mipnerf360/bicycle | 1.129 | 1737 | 1539 |
| gsplat_27k | mipnerf360/bonsai | 1.110 | 729 | 656 |
| gsplat_27k | mipnerf360/counter | 1.107 | 740 | 668 |
| gsplat_27k | mipnerf360/garden | 1.129 | 1748 | 1548 |
| gsplat_27k | mipnerf360/kitchen | 1.113 | 801 | 720 |
| gsplat_27k | mipnerf360/room | 1.164 | 766 | 658 |
| gsplat_27k | mipnerf360/stump | 1.116 | 1574 | 1410 |
| gsplat_27k | tanks_and_temples/train | 1.107 | 516 | 467 |
| gsplat_27k | tanks_and_temples/truck | 1.168 | 498 | 426 |
| higs_tilesamp_27k_r05 | deep_blending/drjohnson | 1.416 | 623 | 440 |
| higs_tilesamp_27k_r05 | deep_blending/playroom | 1.214 | 489 | 403 |
| higs_tilesamp_27k_r05 | mipnerf360/bicycle | 1.414 | 1737 | 1228 |
| higs_tilesamp_27k_r05 | mipnerf360/bonsai | 1.191 | 729 | 612 |
| higs_tilesamp_27k_r05 | mipnerf360/counter | 1.188 | 740 | 623 |
| higs_tilesamp_27k_r05 | mipnerf360/garden | 1.401 | 1748 | 1248 |
| higs_tilesamp_27k_r05 | mipnerf360/kitchen | 1.262 | 801 | 634 |
| higs_tilesamp_27k_r05 | mipnerf360/room | 1.245 | 766 | 615 |
| higs_tilesamp_27k_r05 | mipnerf360/stump | 1.299 | 1574 | 1212 |
| higs_tilesamp_27k_r05 | tanks_and_temples/train | 1.386 | 516 | 373 |
| higs_tilesamp_27k_r05 | tanks_and_temples/truck | 1.426 | 498 | 349 |
| higs_tilesamp_27k_r07 | deep_blending/drjohnson | 1.075 | 623 | 579 |
| higs_tilesamp_27k_r07 | deep_blending/playroom | 1.075 | 489 | 455 |
| higs_tilesamp_27k_r07 | mipnerf360/bicycle | 1.140 | 1737 | 1524 |
| higs_tilesamp_27k_r07 | mipnerf360/bonsai | 1.079 | 729 | 675 |
| higs_tilesamp_27k_r07 | mipnerf360/counter | 1.069 | 740 | 692 |
| higs_tilesamp_27k_r07 | mipnerf360/garden | 0.859 | 1748 | 2034 |
| higs_tilesamp_27k_r07 | mipnerf360/kitchen | 1.065 | 801 | 752 |
| higs_tilesamp_27k_r07 | mipnerf360/room | 1.140 | 766 | 672 |
| higs_tilesamp_27k_r07 | mipnerf360/stump | 1.139 | 1574 | 1382 |
| higs_tilesamp_27k_r07 | tanks_and_temples/train | 0.936 | 516 | 552 |
| higs_tilesamp_27k_r07 | tanks_and_temples/truck | 1.035 | 498 | 481 |
| higs_tilesamp_30k_r07 | deep_blending/drjohnson | 0.964 | 623 | 646 |
| higs_tilesamp_30k_r07 | deep_blending/playroom | 0.948 | 489 | 516 |
| higs_tilesamp_30k_r07 | mipnerf360/bicycle | 1.023 | 1737 | 1699 |
| higs_tilesamp_30k_r07 | mipnerf360/bonsai | 0.974 | 729 | 748 |
| higs_tilesamp_30k_r07 | mipnerf360/counter | 0.962 | 740 | 769 |
| higs_tilesamp_30k_r07 | mipnerf360/garden | 1.008 | 1748 | 1734 |
| higs_tilesamp_30k_r07 | mipnerf360/kitchen | 0.960 | 801 | 834 |
| higs_tilesamp_30k_r07 | mipnerf360/room | 1.016 | 766 | 754 |
| higs_tilesamp_30k_r07 | mipnerf360/stump | 0.959 | 1574 | 1641 |
| higs_tilesamp_30k_r07 | tanks_and_temples/train | 0.831 | 516 | 621 |
| higs_tilesamp_30k_r07 | tanks_and_temples/truck | 0.939 | 498 | 530 |
| higs_tilesamp_refine_27k_r05 | deep_blending/drjohnson | 1.424 | 623 | 437 |
| higs_tilesamp_refine_27k_r05 | deep_blending/playroom | 1.228 | 489 | 398 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/bicycle | 1.415 | 1737 | 1227 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/bonsai | 1.190 | 729 | 612 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/counter | 1.180 | 740 | 627 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/garden | 1.409 | 1748 | 1241 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/kitchen | 1.270 | 801 | 631 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/room | 1.266 | 766 | 605 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/stump | 1.311 | 1574 | 1201 |
| higs_tilesamp_refine_27k_r05 | tanks_and_temples/train | 1.380 | 516 | 374 |
| higs_tilesamp_refine_27k_r05 | tanks_and_temples/truck | 1.452 | 498 | 343 |