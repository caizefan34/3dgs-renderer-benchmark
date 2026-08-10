# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0115 | [-0.1011, 0.0865] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0005 | [-0.0019, 0.0008] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0016 | [-0.0002, 0.0036] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | -96.6402 | [-143.3656, -53.4066] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.123 | CI lo 1.098 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3603 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 195840 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2331469 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish100_cal | psnr_db | -0.0443 | [-0.1399, 0.0581] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish100_cal | ssim | -0.0009 | [-0.0019, 0.0002] | >= -0.003 | True |
| higs_eg_sparse_phase_27k_r07_polish100_cal | lpips | 0.0011 | [-0.0003, 0.0029] | <= +0.005 | True |
| higs_eg_sparse_phase_27k_r07_polish100_cal | time_to_quality_seconds | -64.0487 | [-110.9174, -23.3004] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish100_cal | speedup ratio | 1.086 | CI lo 0.936 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish100_cal | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3566 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_cal | energy_joules (ctrl / cand mean) | 222903 / 196132 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_cal | final_gaussian_count (ctrl / cand mean) | 2400650 / 2310718 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish25_cal | psnr_db | -0.0779 | [-0.2856, 0.1041] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | ssim | -0.0014 | [-0.0042, 0.0007] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | lpips | 0.0029 | [0.0001, 0.0064] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | time_to_quality_seconds | -97.4885 | [-162.0655, -41.2378] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish25_cal | speedup ratio | 1.087 | CI lo 0.928 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3560 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_cal | energy_joules (ctrl / cand mean) | 222903 / 194823 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_cal | final_gaussian_count (ctrl / cand mean) | 2400650 / 2306417 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50 | psnr_db | -0.0871 | [-0.2706, 0.1079] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | ssim | -0.0080 | [-0.0106, -0.0053] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | lpips | 0.0064 | [0.0000, 0.0124] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | time_to_quality_seconds | -142.8820 | [-256.5375, -36.7873] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish50 | speedup ratio | 1.254 | CI lo 0.929 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3578 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | energy_joules (ctrl / cand mean) | 222903 / 160225 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2318299 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50_cal | psnr_db | -0.1082 | [-0.2301, 0.0045] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50_cal | ssim | -0.0007 | [-0.0016, 0.0003] | >= -0.003 | True |
| higs_eg_sparse_phase_27k_r07_polish50_cal | lpips | 0.0020 | [0.0007, 0.0035] | <= +0.005 | True |
| higs_eg_sparse_phase_27k_r07_polish50_cal | time_to_quality_seconds | -76.5264 | [-118.6773, -41.3316] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish50_cal | speedup ratio | 1.081 | CI lo 0.926 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50_cal | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3581 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_cal | energy_joules (ctrl / cand mean) | 222903 / 196045 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_cal | final_gaussian_count (ctrl / cand mean) | 2400650 / 2321398 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.139 | 623 | 547 |
| gsplat_27k | deep_blending/playroom | 1.120 | 489 | 437 |
| gsplat_27k | mipnerf360/bicycle | 1.128 | 1737 | 1540 |
| gsplat_27k | mipnerf360/bonsai | 1.101 | 729 | 662 |
| gsplat_27k | mipnerf360/counter | 1.098 | 740 | 674 |
| gsplat_27k | mipnerf360/garden | 1.128 | 1748 | 1550 |
| gsplat_27k | mipnerf360/kitchen | 1.112 | 801 | 720 |
| gsplat_27k | mipnerf360/room | 1.158 | 766 | 662 |
| gsplat_27k | mipnerf360/stump | 1.116 | 1574 | 1410 |
| gsplat_27k | tanks_and_temples/train | 1.101 | 516 | 469 |
| gsplat_27k | tanks_and_temples/truck | 1.152 | 498 | 432 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | deep_blending/drjohnson | 1.094 | 623 | 569 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | deep_blending/playroom | 1.078 | 489 | 454 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/bicycle | 1.142 | 1737 | 1521 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/bonsai | 1.088 | 729 | 670 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/counter | 1.078 | 740 | 686 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/garden | 1.135 | 1748 | 1540 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/kitchen | 1.074 | 801 | 746 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/room | 1.144 | 766 | 670 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/stump | 1.137 | 1574 | 1384 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | tanks_and_temples/train | 0.936 | 516 | 552 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | tanks_and_temples/truck | 1.041 | 498 | 478 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | deep_blending/drjohnson | 1.105 | 623 | 564 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | deep_blending/playroom | 1.084 | 489 | 451 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/bicycle | 1.144 | 1737 | 1519 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/bonsai | 1.085 | 729 | 671 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/counter | 1.075 | 740 | 688 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/garden | 1.135 | 1748 | 1540 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/kitchen | 1.075 | 801 | 744 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/room | 1.142 | 766 | 671 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/stump | 1.136 | 1574 | 1386 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | tanks_and_temples/train | 0.928 | 516 | 556 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | tanks_and_temples/truck | 1.045 | 498 | 476 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/drjohnson | 1.145 | 623 | 544 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/playroom | 1.171 | 489 | 417 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bicycle | 1.404 | 1737 | 1237 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bonsai | 1.300 | 729 | 560 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/counter | 1.282 | 740 | 577 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/garden | 1.436 | 1748 | 1218 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/kitchen | 1.250 | 801 | 641 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/room | 1.374 | 766 | 558 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/stump | 1.469 | 1574 | 1071 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/train | 0.929 | 516 | 556 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/truck | 1.037 | 498 | 480 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | deep_blending/drjohnson | 1.096 | 623 | 568 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | deep_blending/playroom | 1.072 | 489 | 456 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/bicycle | 1.146 | 1737 | 1516 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/bonsai | 1.081 | 729 | 674 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/counter | 1.073 | 740 | 690 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/garden | 1.137 | 1748 | 1538 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/kitchen | 1.069 | 801 | 749 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/room | 1.127 | 766 | 680 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/stump | 1.132 | 1574 | 1391 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | tanks_and_temples/train | 0.926 | 516 | 557 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | tanks_and_temples/truck | 1.038 | 498 | 479 |