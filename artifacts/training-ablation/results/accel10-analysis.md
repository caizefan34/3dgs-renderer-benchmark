# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.1014 | [-0.3516, 0.0902] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0018 | [-0.0054, 0.0007] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0032 | [0.0001, 0.0079] | <= +0.005 | False |
| gsplat_27k | time_to_quality_seconds | -102.5148 | [-171.5905, -40.7947] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.108 | CI lo 0.959 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3593 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 197960 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2323263 | - | descriptive | - |

| gsplat_30k_ssim05 | psnr_db | -0.0115 | [-0.1589, 0.1616] | >= -0.10 | False |
| gsplat_30k_ssim05 | ssim | -0.0083 | [-0.0113, -0.0056] | >= -0.003 | False |
| gsplat_30k_ssim05 | lpips | 0.0015 | [-0.0037, 0.0057] | <= +0.005 | False |
| gsplat_30k_ssim05 | time_to_quality_seconds | -244.1981 | [-414.5077, -103.9751] | <= 0 | True |
| gsplat_30k_ssim05 | speedup ratio | 1.407 | CI lo 1.085 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3007 | - | descriptive | - |
| gsplat_30k_ssim05 | energy_joules (ctrl / cand mean) | 222903 / 141596 | - | descriptive | - |
| gsplat_30k_ssim05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1947540 | - | descriptive | - |

| higs_skipbwd_27k | psnr_db | -0.2518 | [-0.4157, -0.1009] | >= -0.10 | False |
| higs_skipbwd_27k | ssim | -0.0034 | [-0.0070, -0.0008] | >= -0.003 | False |
| higs_skipbwd_27k | lpips | 0.0033 | [0.0004, 0.0073] | <= +0.005 | False |
| higs_skipbwd_27k | time_to_quality_seconds | -188.6721 | [-357.3308, -67.0405] | <= 0 | True |
| higs_skipbwd_27k | speedup ratio | 1.213 | CI lo 1.080 | mean>=1.111 & lo>1.0 | True |
| higs_skipbwd_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3572 | - | descriptive | - |
| higs_skipbwd_27k | energy_joules (ctrl / cand mean) | 222903 / 172214 | - | descriptive | - |
| higs_skipbwd_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2305200 | - | descriptive | - |

| higs_skipbwd_30k | psnr_db | -0.1702 | [-0.3069, -0.0355] | >= -0.10 | False |
| higs_skipbwd_30k | ssim | -0.0021 | [-0.0047, -0.0002] | >= -0.003 | False |
| higs_skipbwd_30k | lpips | 0.0017 | [0.0000, 0.0036] | <= +0.005 | True |
| higs_skipbwd_30k | time_to_quality_seconds | -139.2525 | [-295.3763, -35.8459] | <= 0 | True |
| higs_skipbwd_30k | speedup ratio | 1.099 | CI lo 0.974 | mean>=1.111 & lo>1.0 | False |
| higs_skipbwd_30k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3694 | - | descriptive | - |
| higs_skipbwd_30k | energy_joules (ctrl / cand mean) | 222903 / 193309 | - | descriptive | - |
| higs_skipbwd_30k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2388409 | - | descriptive | - |

| higs_skipbwd_30k_agg | psnr_db | -0.1410 | [-0.3237, 0.0342] | >= -0.10 | False |
| higs_skipbwd_30k_agg | ssim | -0.0021 | [-0.0050, 0.0003] | >= -0.003 | False |
| higs_skipbwd_30k_agg | lpips | 0.0021 | [0.0001, 0.0046] | <= +0.005 | True |
| higs_skipbwd_30k_agg | time_to_quality_seconds | -81.7016 | [-171.3748, -12.9433] | <= 0 | True |
| higs_skipbwd_30k_agg | speedup ratio | 1.041 | CI lo 0.944 | mean>=1.111 & lo>1.0 | False |
| higs_skipbwd_30k_agg | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3709 | - | descriptive | - |
| higs_skipbwd_30k_agg | energy_joules (ctrl / cand mean) | 222903 / 204783 | - | descriptive | - |
| higs_skipbwd_30k_agg | final_gaussian_count (ctrl / cand mean) | 2400650 / 2398217 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.133 | 623 | 549 |
| gsplat_27k | deep_blending/playroom | 1.122 | 489 | 436 |
| gsplat_27k | mipnerf360/bicycle | 1.129 | 1737 | 1538 |
| gsplat_27k | mipnerf360/bonsai | 1.105 | 729 | 659 |
| gsplat_27k | mipnerf360/counter | 1.110 | 740 | 666 |
| gsplat_27k | mipnerf360/garden | 1.131 | 1748 | 1546 |
| gsplat_27k | mipnerf360/kitchen | 1.118 | 801 | 716 |
| gsplat_27k | mipnerf360/room | 1.152 | 766 | 665 |
| gsplat_27k | mipnerf360/stump | 1.038 | 1574 | 1517 |
| gsplat_27k | tanks_and_temples/train | 0.959 | 516 | 539 |
| gsplat_27k | tanks_and_temples/truck | 1.186 | 498 | 420 |
| gsplat_30k_ssim05 | deep_blending/drjohnson | 1.462 | 623 | 426 |
| gsplat_30k_ssim05 | deep_blending/playroom | 1.507 | 489 | 324 |
| gsplat_30k_ssim05 | mipnerf360/bicycle | 1.825 | 1737 | 952 |
| gsplat_30k_ssim05 | mipnerf360/bonsai | 1.191 | 729 | 611 |
| gsplat_30k_ssim05 | mipnerf360/counter | 1.151 | 740 | 643 |
| gsplat_30k_ssim05 | mipnerf360/garden | 2.032 | 1748 | 860 |
| gsplat_30k_ssim05 | mipnerf360/kitchen | 1.149 | 801 | 697 |
| gsplat_30k_ssim05 | mipnerf360/room | 1.236 | 766 | 620 |
| gsplat_30k_ssim05 | mipnerf360/stump | 1.596 | 1574 | 986 |
| gsplat_30k_ssim05 | tanks_and_temples/train | 1.085 | 516 | 476 |
| gsplat_30k_ssim05 | tanks_and_temples/truck | 1.239 | 498 | 402 |
| higs_skipbwd_27k | deep_blending/drjohnson | 1.187 | 623 | 525 |
| higs_skipbwd_27k | deep_blending/playroom | 1.183 | 489 | 413 |
| higs_skipbwd_27k | mipnerf360/bicycle | 1.327 | 1737 | 1309 |
| higs_skipbwd_27k | mipnerf360/bonsai | 1.285 | 729 | 567 |
| higs_skipbwd_27k | mipnerf360/counter | 1.198 | 740 | 618 |
| higs_skipbwd_27k | mipnerf360/garden | 1.284 | 1748 | 1361 |
| higs_skipbwd_27k | mipnerf360/kitchen | 1.254 | 801 | 639 |
| higs_skipbwd_27k | mipnerf360/room | 1.242 | 766 | 617 |
| higs_skipbwd_27k | mipnerf360/stump | 1.148 | 1574 | 1371 |
| higs_skipbwd_27k | tanks_and_temples/train | 1.080 | 516 | 478 |
| higs_skipbwd_27k | tanks_and_temples/truck | 1.152 | 498 | 432 |
| higs_skipbwd_30k | deep_blending/drjohnson | 1.068 | 623 | 583 |
| higs_skipbwd_30k | deep_blending/playroom | 1.064 | 489 | 460 |
| higs_skipbwd_30k | mipnerf360/bicycle | 1.200 | 1737 | 1448 |
| higs_skipbwd_30k | mipnerf360/bonsai | 1.201 | 729 | 607 |
| higs_skipbwd_30k | mipnerf360/counter | 1.089 | 740 | 679 |
| higs_skipbwd_30k | mipnerf360/garden | 1.157 | 1748 | 1511 |
| higs_skipbwd_30k | mipnerf360/kitchen | 1.138 | 801 | 704 |
| higs_skipbwd_30k | mipnerf360/room | 1.107 | 766 | 692 |
| higs_skipbwd_30k | mipnerf360/stump | 1.061 | 1574 | 1484 |
| higs_skipbwd_30k | tanks_and_temples/train | 0.974 | 516 | 530 |
| higs_skipbwd_30k | tanks_and_temples/truck | 1.035 | 498 | 481 |
| higs_skipbwd_30k_agg | deep_blending/drjohnson | 1.040 | 623 | 599 |
| higs_skipbwd_30k_agg | deep_blending/playroom | 1.036 | 489 | 472 |
| higs_skipbwd_30k_agg | mipnerf360/bicycle | 1.132 | 1737 | 1534 |
| higs_skipbwd_30k_agg | mipnerf360/bonsai | 1.119 | 729 | 651 |
| higs_skipbwd_30k_agg | mipnerf360/counter | 1.030 | 740 | 718 |
| higs_skipbwd_30k_agg | mipnerf360/garden | 1.079 | 1748 | 1620 |
| higs_skipbwd_30k_agg | mipnerf360/kitchen | 1.068 | 801 | 749 |
| higs_skipbwd_30k_agg | mipnerf360/room | 0.981 | 766 | 781 |
| higs_skipbwd_30k_agg | mipnerf360/stump | 1.028 | 1574 | 1532 |
| higs_skipbwd_30k_agg | tanks_and_temples/train | 0.944 | 516 | 547 |
| higs_skipbwd_30k_agg | tanks_and_temples/truck | 0.988 | 498 | 504 |