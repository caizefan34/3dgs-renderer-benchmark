# HiGS Accel4 Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.1330 | [-0.3236, 0.0280] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0022 | [-0.0058, 0.0003] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0043 | [0.0004, 0.0105] | <= +0.005 | False |
| gsplat_27k | time_to_quality_seconds | -106.8567 | [-165.9411, -55.6096] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.122 | CI lo 1.095 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3576 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 195585 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2313752 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r05 | psnr_db | -0.1720 | [-0.3657, 0.0240] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r05 | ssim | -0.0098 | [-0.0128, -0.0068] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r05 | lpips | 0.0084 | [0.0022, 0.0146] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r05 | time_to_quality_seconds | -128.9263 | [-254.7035, -6.9512] | <= 0 | True |
| higs_eg_sparse_phase_27k_r05 | speedup ratio | 1.222 | CI lo 0.758 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3558 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r05 | energy_joules (ctrl / cand mean) | 222903 / 172721 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2305222 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07 | psnr_db | -0.0955 | [-0.2696, 0.0931] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07 | ssim | -0.0089 | [-0.0118, -0.0059] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07 | lpips | 0.0074 | [0.0015, 0.0135] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07 | time_to_quality_seconds | -140.7875 | [-256.6115, -29.2602] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07 | speedup ratio | 1.262 | CI lo 0.927 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3575 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | energy_joules (ctrl / cand mean) | 222903 / 159608 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2315975 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_mix05 | psnr_db | -0.1006 | [-0.2719, 0.0822] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | ssim | -0.0084 | [-0.0110, -0.0055] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | lpips | 0.0068 | [0.0008, 0.0126] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | time_to_quality_seconds | -137.9164 | [-254.4396, -26.4003] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_mix05 | speedup ratio | 1.254 | CI lo 0.922 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_mix05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3557 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_mix05 | energy_joules (ctrl / cand mean) | 222903 / 161104 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_mix05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2304722 | - | descriptive | - |

| higs_eg_sparse_phase_30k_r07 | psnr_db | -0.0885 | [-0.2557, 0.0957] | >= -0.10 | False |
| higs_eg_sparse_phase_30k_r07 | ssim | -0.0075 | [-0.0099, -0.0050] | >= -0.003 | False |
| higs_eg_sparse_phase_30k_r07 | lpips | 0.0054 | [-0.0001, 0.0099] | <= +0.005 | False |
| higs_eg_sparse_phase_30k_r07 | time_to_quality_seconds | -89.8601 | [-188.6474, 6.4004] | <= 0 | False |
| higs_eg_sparse_phase_30k_r07 | speedup ratio | 1.157 | CI lo 0.836 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_30k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3675 | - | descriptive | - |
| higs_eg_sparse_phase_30k_r07 | energy_joules (ctrl / cand mean) | 222903 / 174930 | - | descriptive | - |
| higs_eg_sparse_phase_30k_r07 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2385675 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.113 | 623 | 559 |
| gsplat_27k | deep_blending/playroom | 1.102 | 489 | 444 |
| gsplat_27k | mipnerf360/bicycle | 1.132 | 1737 | 1535 |
| gsplat_27k | mipnerf360/bonsai | 1.110 | 729 | 656 |
| gsplat_27k | mipnerf360/counter | 1.095 | 740 | 675 |
| gsplat_27k | mipnerf360/garden | 1.128 | 1748 | 1549 |
| gsplat_27k | mipnerf360/kitchen | 1.108 | 801 | 722 |
| gsplat_27k | mipnerf360/room | 1.159 | 766 | 661 |
| gsplat_27k | mipnerf360/stump | 1.130 | 1574 | 1393 |
| gsplat_27k | tanks_and_temples/train | 1.097 | 516 | 471 |
| gsplat_27k | tanks_and_temples/truck | 1.163 | 498 | 428 |
| higs_eg_sparse_phase_27k_r05 | deep_blending/drjohnson | 1.161 | 623 | 536 |
| higs_eg_sparse_phase_27k_r05 | deep_blending/playroom | 1.182 | 489 | 414 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/bicycle | 1.408 | 1737 | 1234 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/bonsai | 1.316 | 729 | 554 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/counter | 1.305 | 740 | 567 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/garden | 1.442 | 1748 | 1212 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/kitchen | 0.758 | 801 | 1056 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/room | 1.383 | 766 | 554 |
| higs_eg_sparse_phase_27k_r05 | mipnerf360/stump | 1.489 | 1574 | 1057 |
| higs_eg_sparse_phase_27k_r05 | tanks_and_temples/train | 0.950 | 516 | 543 |
| higs_eg_sparse_phase_27k_r05 | tanks_and_temples/truck | 1.053 | 498 | 473 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/drjohnson | 1.157 | 623 | 538 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/playroom | 1.170 | 489 | 418 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bicycle | 1.415 | 1737 | 1228 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bonsai | 1.316 | 729 | 553 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/counter | 1.297 | 740 | 570 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/garden | 1.438 | 1748 | 1216 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/kitchen | 1.250 | 801 | 640 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/room | 1.381 | 766 | 555 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/stump | 1.472 | 1574 | 1069 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/train | 0.927 | 516 | 557 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/truck | 1.058 | 498 | 471 |
| higs_eg_sparse_phase_27k_r07_mix05 | deep_blending/drjohnson | 1.161 | 623 | 536 |
| higs_eg_sparse_phase_27k_r07_mix05 | deep_blending/playroom | 1.183 | 489 | 413 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/bicycle | 1.408 | 1737 | 1234 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/bonsai | 1.317 | 729 | 553 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/counter | 1.298 | 740 | 570 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/garden | 1.432 | 1748 | 1221 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/kitchen | 1.248 | 801 | 642 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/room | 1.285 | 766 | 596 |
| higs_eg_sparse_phase_27k_r07_mix05 | mipnerf360/stump | 1.490 | 1574 | 1057 |
| higs_eg_sparse_phase_27k_r07_mix05 | tanks_and_temples/train | 0.922 | 516 | 560 |
| higs_eg_sparse_phase_27k_r07_mix05 | tanks_and_temples/truck | 1.055 | 498 | 472 |
| higs_eg_sparse_phase_30k_r07 | deep_blending/drjohnson | 1.036 | 623 | 601 |
| higs_eg_sparse_phase_30k_r07 | deep_blending/playroom | 1.070 | 489 | 457 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/bicycle | 1.300 | 1737 | 1336 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/bonsai | 1.224 | 729 | 595 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/counter | 1.202 | 740 | 615 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/garden | 1.319 | 1748 | 1325 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/kitchen | 1.144 | 801 | 700 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/room | 1.286 | 766 | 596 |
| higs_eg_sparse_phase_30k_r07 | mipnerf360/stump | 1.370 | 1574 | 1149 |
| higs_eg_sparse_phase_30k_r07 | tanks_and_temples/train | 0.836 | 516 | 618 |
| higs_eg_sparse_phase_30k_r07 | tanks_and_temples/truck | 0.940 | 498 | 529 |