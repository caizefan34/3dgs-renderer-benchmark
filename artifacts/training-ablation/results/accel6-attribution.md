# HiGS accel6 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| higs_eg_sparse_phase_27k_r07_polish100_cal | psnr_db | -0.0328 | [-0.0820, 0.0204] | >= -0.10 | True |
| higs_eg_sparse_phase_27k_r07_polish100_cal | ssim | -0.0004 | [-0.0010, 0.0003] | >= -0.003 | True |
| higs_eg_sparse_phase_27k_r07_polish100_cal | lpips | -0.0005 | [-0.0021, 0.0005] | <= +0.005 | True |
| higs_eg_sparse_phase_27k_r07_polish100_cal | time_to_quality_seconds | 32.5914 | [1.7915, 80.8939] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish100_cal | speedup ratio | 0.967 | CI lo 0.850 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish100_cal | peak_gpu_memory_mib (ctrl / cand mean) | 3603 / 3566 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_cal | energy_joules (ctrl / cand mean) | 195840 / 196132 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_cal | final_gaussian_count (ctrl / cand mean) | 2331469 / 2310718 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish25_cal | psnr_db | -0.0664 | [-0.2275, 0.0588] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | ssim | -0.0008 | [-0.0024, 0.0001] | >= -0.003 | True |
| higs_eg_sparse_phase_27k_r07_polish25_cal | lpips | 0.0013 | [0.0001, 0.0030] | <= +0.005 | True |
| higs_eg_sparse_phase_27k_r07_polish25_cal | time_to_quality_seconds | -0.8483 | [-22.8254, 16.3052] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | speedup ratio | 0.968 | CI lo 0.843 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish25_cal | peak_gpu_memory_mib (ctrl / cand mean) | 3603 / 3560 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_cal | energy_joules (ctrl / cand mean) | 195840 / 194823 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_cal | final_gaussian_count (ctrl / cand mean) | 2331469 / 2306417 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50 | psnr_db | -0.0756 | [-0.2262, 0.1201] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | ssim | -0.0074 | [-0.0094, -0.0055] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | lpips | 0.0048 | [-0.0025, 0.0102] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | time_to_quality_seconds | -46.2419 | [-136.3234, 49.9218] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | speedup ratio | 1.117 | CI lo 0.843 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | peak_gpu_memory_mib (ctrl / cand mean) | 3603 / 3578 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | energy_joules (ctrl / cand mean) | 195840 / 160225 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | final_gaussian_count (ctrl / cand mean) | 2331469 / 2318299 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50_cal | psnr_db | -0.0967 | [-0.2550, 0.0079] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50_cal | ssim | -0.0002 | [-0.0010, 0.0006] | >= -0.003 | True |
| higs_eg_sparse_phase_27k_r07_polish50_cal | lpips | 0.0004 | [-0.0007, 0.0018] | <= +0.005 | True |
| higs_eg_sparse_phase_27k_r07_polish50_cal | time_to_quality_seconds | 20.1138 | [-8.7969, 58.1163] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish50_cal | speedup ratio | 0.963 | CI lo 0.841 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50_cal | peak_gpu_memory_mib (ctrl / cand mean) | 3603 / 3581 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_cal | energy_joules (ctrl / cand mean) | 195840 / 196045 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_cal | final_gaussian_count (ctrl / cand mean) | 2331469 / 2321398 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| higs_eg_sparse_phase_27k_r07_polish100_cal | deep_blending/drjohnson | 0.961 | 547 | 569 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | deep_blending/playroom | 0.962 | 437 | 454 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/bicycle | 1.012 | 1540 | 1521 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/bonsai | 0.988 | 662 | 670 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/counter | 0.982 | 674 | 686 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/garden | 1.007 | 1550 | 1540 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/kitchen | 0.965 | 720 | 746 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/room | 0.988 | 662 | 670 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | mipnerf360/stump | 1.019 | 1410 | 1384 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | tanks_and_temples/train | 0.850 | 469 | 552 |
| higs_eg_sparse_phase_27k_r07_polish100_cal | tanks_and_temples/truck | 0.904 | 432 | 478 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | deep_blending/drjohnson | 0.970 | 547 | 564 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | deep_blending/playroom | 0.968 | 437 | 451 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/bicycle | 1.014 | 1540 | 1519 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/bonsai | 0.986 | 662 | 671 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/counter | 0.979 | 674 | 688 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/garden | 1.007 | 1550 | 1540 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/kitchen | 0.967 | 720 | 744 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/room | 0.986 | 662 | 671 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | mipnerf360/stump | 1.017 | 1410 | 1386 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | tanks_and_temples/train | 0.843 | 469 | 556 |
| higs_eg_sparse_phase_27k_r07_polish25_cal | tanks_and_temples/truck | 0.908 | 432 | 476 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/drjohnson | 1.006 | 547 | 544 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/playroom | 1.046 | 437 | 417 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bicycle | 1.245 | 1540 | 1237 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bonsai | 1.181 | 662 | 560 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/counter | 1.167 | 674 | 577 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/garden | 1.273 | 1550 | 1218 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/kitchen | 1.123 | 720 | 641 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/room | 1.187 | 662 | 558 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/stump | 1.316 | 1410 | 1071 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/train | 0.843 | 469 | 556 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/truck | 0.901 | 432 | 480 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | deep_blending/drjohnson | 0.962 | 547 | 568 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | deep_blending/playroom | 0.957 | 437 | 456 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/bicycle | 1.016 | 1540 | 1516 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/bonsai | 0.982 | 662 | 674 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/counter | 0.977 | 674 | 690 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/garden | 1.008 | 1550 | 1538 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/kitchen | 0.961 | 720 | 749 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/room | 0.974 | 662 | 680 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | mipnerf360/stump | 1.014 | 1410 | 1391 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | tanks_and_temples/train | 0.841 | 469 | 557 |
| higs_eg_sparse_phase_27k_r07_polish50_cal | tanks_and_temples/truck | 0.902 | 432 | 479 |