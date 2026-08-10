# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.1992 | [-0.4503, -0.0029] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0023 | [-0.0051, -0.0000] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0045 | [0.0011, 0.0090] | <= +0.005 | False |
| gsplat_27k | time_to_quality_seconds | -107.1318 | [-165.0047, -59.6953] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.113 | CI lo 1.073 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3593 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 199812 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2324813 | - | descriptive | - |

| gsplat_27k_preload_accum8 | psnr_db | -0.6022 | [-0.7687, -0.3977] | >= -0.10 | False |
| gsplat_27k_preload_accum8 | ssim | -0.0181 | [-0.0331, -0.0057] | >= -0.003 | False |
| gsplat_27k_preload_accum8 | lpips | 0.0305 | [0.0115, 0.0550] | <= +0.005 | False |
| gsplat_27k_preload_accum8 | time_to_quality_seconds | -82.7739 | [-206.4738, 60.3463] | <= 0 | False |
| gsplat_27k_preload_accum8 | speedup ratio | 1.235 | CI lo 0.983 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_preload_accum8 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3505 | - | descriptive | - |
| gsplat_27k_preload_accum8 | energy_joules (ctrl / cand mean) | 222903 / 161413 | - | descriptive | - |
| gsplat_27k_preload_accum8 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1648942 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish100_acc7 | psnr_db | -0.7360 | [-1.0218, -0.4432] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | ssim | -0.0277 | [-0.0460, -0.0123] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | lpips | 0.0397 | [0.0150, 0.0713] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | time_to_quality_seconds | -211.1896 | [-370.7305, -65.1148] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | speedup ratio | 1.474 | CI lo 0.960 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3494 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | energy_joules (ctrl / cand mean) | 222903 / 125851 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1634467 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish25_acc7 | psnr_db | -0.7444 | [-0.9669, -0.4993] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | ssim | -0.0275 | [-0.0451, -0.0127] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | lpips | 0.0398 | [0.0161, 0.0701] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | time_to_quality_seconds | -199.5253 | [-359.0462, -50.7702] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | speedup ratio | 1.464 | CI lo 0.948 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3514 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | energy_joules (ctrl / cand mean) | 222903 / 127290 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1646862 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50_acc7 | psnr_db | -0.8086 | [-1.0603, -0.5353] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | ssim | -0.0278 | [-0.0459, -0.0127] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | lpips | 0.0400 | [0.0154, 0.0717] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | time_to_quality_seconds | -211.1673 | [-370.2641, -66.1510] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | speedup ratio | 1.472 | CI lo 0.940 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3499 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | energy_joules (ctrl / cand mean) | 222903 / 124362 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1634123 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.123 | 623 | 555 |
| gsplat_27k | deep_blending/playroom | 1.101 | 489 | 444 |
| gsplat_27k | mipnerf360/bicycle | 1.124 | 1737 | 1545 |
| gsplat_27k | mipnerf360/bonsai | 1.097 | 729 | 664 |
| gsplat_27k | mipnerf360/counter | 1.073 | 740 | 690 |
| gsplat_27k | mipnerf360/garden | 1.119 | 1748 | 1562 |
| gsplat_27k | mipnerf360/kitchen | 1.104 | 801 | 725 |
| gsplat_27k | mipnerf360/room | 1.138 | 766 | 673 |
| gsplat_27k | mipnerf360/stump | 1.114 | 1574 | 1414 |
| gsplat_27k | tanks_and_temples/train | 1.084 | 516 | 476 |
| gsplat_27k | tanks_and_temples/truck | 1.164 | 498 | 427 |
| gsplat_27k_preload_accum8 | deep_blending/drjohnson | 1.311 | 623 | 475 |
| gsplat_27k_preload_accum8 | deep_blending/playroom | 1.252 | 489 | 391 |
| gsplat_27k_preload_accum8 | mipnerf360/bicycle | 1.334 | 1737 | 1302 |
| gsplat_27k_preload_accum8 | mipnerf360/bonsai | 1.156 | 729 | 630 |
| gsplat_27k_preload_accum8 | mipnerf360/counter | 1.171 | 740 | 632 |
| gsplat_27k_preload_accum8 | mipnerf360/garden | 1.371 | 1748 | 1275 |
| gsplat_27k_preload_accum8 | mipnerf360/kitchen | 1.253 | 801 | 639 |
| gsplat_27k_preload_accum8 | mipnerf360/room | 1.264 | 766 | 606 |
| gsplat_27k_preload_accum8 | mipnerf360/stump | 1.262 | 1574 | 1248 |
| gsplat_27k_preload_accum8 | tanks_and_temples/train | 0.983 | 516 | 525 |
| gsplat_27k_preload_accum8 | tanks_and_temples/truck | 1.225 | 498 | 406 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | deep_blending/drjohnson | 1.417 | 623 | 439 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | deep_blending/playroom | 1.381 | 489 | 354 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/bicycle | 1.739 | 1737 | 999 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/bonsai | 1.384 | 729 | 526 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/counter | 1.458 | 740 | 507 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/garden | 1.859 | 1748 | 940 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/kitchen | 1.534 | 801 | 522 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/room | 1.563 | 766 | 490 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/stump | 1.714 | 1574 | 919 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | tanks_and_temples/train | 0.960 | 516 | 538 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | tanks_and_temples/truck | 1.202 | 498 | 414 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | deep_blending/drjohnson | 1.414 | 623 | 440 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | deep_blending/playroom | 1.446 | 489 | 338 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/bicycle | 1.718 | 1737 | 1011 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/bonsai | 1.374 | 729 | 530 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/counter | 1.423 | 740 | 520 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/garden | 1.837 | 1748 | 952 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/kitchen | 1.490 | 801 | 537 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/room | 1.560 | 766 | 491 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/stump | 1.691 | 1574 | 931 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | tanks_and_temples/train | 0.948 | 516 | 544 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | tanks_and_temples/truck | 1.201 | 498 | 415 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | deep_blending/drjohnson | 1.409 | 623 | 442 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | deep_blending/playroom | 1.434 | 489 | 341 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/bicycle | 1.741 | 1737 | 997 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/bonsai | 1.386 | 729 | 526 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/counter | 1.454 | 740 | 509 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/garden | 1.839 | 1748 | 951 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/kitchen | 1.522 | 801 | 526 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/room | 1.556 | 766 | 492 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/stump | 1.707 | 1574 | 922 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | tanks_and_temples/train | 0.940 | 516 | 549 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | tanks_and_temples/truck | 1.198 | 498 | 415 |