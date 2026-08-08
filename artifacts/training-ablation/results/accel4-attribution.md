# HiGS Accel4 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| higs_eg_sparse_phase_27k_r05 | psnr_db | -0.0390 | [-0.2508, 0.2460] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r05 | ssim | -0.0076 | [-0.0107, -0.0043] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r05 | lpips | 0.0040 | [-0.0073, 0.0117] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r05 | time_to_quality_seconds | -22.0696 | [-133.2813, 102.3351] | <= 0 | False |
| higs_eg_sparse_phase_27k_r05 | speedup ratio | 1.089 | CI lo 0.684 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3558 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r05 | energy_joules (ctrl / cand mean) | 195585 / 172721 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2313752 / 2305222 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07 | psnr_db | 0.0375 | [-0.1731, 0.3420] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07 | ssim | -0.0067 | [-0.0094, -0.0038] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07 | lpips | 0.0031 | [-0.0080, 0.0105] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07 | time_to_quality_seconds | -33.9308 | [-132.7519, 79.5724] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07 | speedup ratio | 1.125 | CI lo 0.845 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3575 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | energy_joules (ctrl / cand mean) | 195585 / 159608 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | final_gaussian_count (ctrl / cand mean) | 2313752 / 2315975 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_mix05 | psnr_db | 0.0324 | [-0.1599, 0.3252] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | ssim | -0.0062 | [-0.0090, -0.0029] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | lpips | 0.0025 | [-0.0087, 0.0098] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | time_to_quality_seconds | -31.0597 | [-130.7752, 83.5714] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | speedup ratio | 1.118 | CI lo 0.840 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3557 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_mix05 | energy_joules (ctrl / cand mean) | 195585 / 161104 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_mix05 | final_gaussian_count (ctrl / cand mean) | 2313752 / 2304722 | - | descriptive | - |

| higs_eg_sparse_phase_30k_r07 | psnr_db | 0.0445 | [-0.1643, 0.3559] | >= -0.10 | False |
| higs_eg_sparse_phase_30k_r07 | ssim | -0.0054 | [-0.0081, -0.0021] | >= -0.003 | False |
| higs_eg_sparse_phase_30k_r07 | lpips | 0.0010 | [-0.0096, 0.0072] | <= +0.005 | False |
| higs_eg_sparse_phase_30k_r07 | time_to_quality_seconds | 16.9966 | [-69.2553, 124.9470] | <= 0 | False |
| higs_eg_sparse_phase_30k_r07 | speedup ratio | 1.031 | CI lo 0.762 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_30k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3675 | - | descriptive | - |
| higs_eg_sparse_phase_30k_r07 | energy_joules (ctrl / cand mean) | 195585 / 174930 | - | descriptive | - |
| higs_eg_sparse_phase_30k_r07 | final_gaussian_count (ctrl / cand mean) | 2313752 / 2385675 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| higs_eg_sparse_phase_27k_r05 | deep_blending/drjohnson | 1.043 | 559 | 536 |
| higs_eg_sparse_phase_27k_r05 | deep_blending/playroom | 1.072 | 444 | 414 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/bicycle | 1.244 | 1535 | 1234 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/bonsai | 1.185 | 656 | 554 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/counter | 1.191 | 675 | 567 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/garden | 1.278 | 1549 | 1212 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/kitchen | 0.684 | 722 | 1056 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/room | 1.193 | 661 | 554 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/stump | 1.318 | 1393 | 1057 |
| higs_eg_sparse_phase_27k_r05 | tanks_and_temples/train | 0.866 | 471 | 543 |
| higs_eg_sparse_phase_27k_r05 | tanks_and_temples/truck | 0.906 | 428 | 473 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/drjohnson | 1.040 | 559 | 538 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/playroom | 1.062 | 444 | 418 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bicycle | 1.250 | 1535 | 1228 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bonsai | 1.186 | 656 | 553 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/counter | 1.184 | 675 | 570 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/garden | 1.275 | 1549 | 1216 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/kitchen | 1.128 | 722 | 640 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/room | 1.191 | 661 | 555 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/stump | 1.303 | 1393 | 1069 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/train | 0.845 | 471 | 557 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/truck | 0.909 | 428 | 471 |
| higs_eg_sparse_phase_27k_r07_mix05 | deep_blending/drjohnson | 1.044 | 559 | 536 |
| higs_eg_sparse_phase_27k_r07_mix05 | deep_blending/playroom | 1.073 | 444 | 413 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/bicycle | 1.244 | 1535 | 1234 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/bonsai | 1.186 | 656 | 553 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/counter | 1.185 | 675 | 570 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/garden | 1.269 | 1549 | 1221 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/kitchen | 1.126 | 722 | 642 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/room | 1.108 | 661 | 596 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/stump | 1.318 | 1393 | 1057 |
| higs_eg_sparse_phase_27k_r07_mix05 | tanks_and_temples/train | 0.840 | 471 | 560 |
| higs_eg_sparse_phase_27k_r07_mix05 | tanks_and_temples/truck | 0.907 | 428 | 472 |
| higs_eg_sparse_phase_30k_r07 | deep_blending/drjohnson | 0.931 | 559 | 601 |
| higs_eg_sparse_phase_30k_r07 | deep_blending/playroom | 0.971 | 444 | 457 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/bicycle | 1.149 | 1535 | 1336 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/bonsai | 1.102 | 656 | 595 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/counter | 1.098 | 675 | 615 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/garden | 1.169 | 1549 | 1325 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/kitchen | 1.032 | 722 | 700 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/room | 1.109 | 661 | 596 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/stump | 1.212 | 1393 | 1149 |
| higs_eg_sparse_phase_30k_r07 | tanks_and_temples/train | 0.762 | 471 | 618 |
| higs_eg_sparse_phase_30k_r07 | tanks_and_temples/truck | 0.808 | 428 | 529 |