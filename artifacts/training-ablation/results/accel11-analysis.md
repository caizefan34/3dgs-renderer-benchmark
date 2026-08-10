# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0474 | [-0.1654, 0.0737] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0009 | [-0.0024, 0.0004] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0016 | [0.0000, 0.0034] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | -100.6178 | [-148.6332, -55.2384] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.129 | CI lo 1.098 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3581 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 198485 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2317871 | - | descriptive | - |

| gsplat_30k_ssim075 | psnr_db | 0.0116 | [-0.0806, 0.1127] | >= -0.10 | True |
| gsplat_30k_ssim075 | ssim | -0.0028 | [-0.0045, -0.0014] | >= -0.003 | False |
| gsplat_30k_ssim075 | lpips | -0.0009 | [-0.0048, 0.0021] | <= +0.005 | True |
| gsplat_30k_ssim075 | time_to_quality_seconds | -183.2510 | [-279.1190, -98.0585] | <= 0 | True |
| gsplat_30k_ssim075 | speedup ratio | 1.270 | CI lo 0.976 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3198 | - | descriptive | - |
| gsplat_30k_ssim075 | energy_joules (ctrl / cand mean) | 222903 / 162950 | - | descriptive | - |
| gsplat_30k_ssim075 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2072855 | - | descriptive | - |

| gsplat_30k_ssim_e2 | psnr_db | -0.1106 | [-0.2839, 0.0727] | >= -0.10 | False |
| gsplat_30k_ssim_e2 | ssim | -0.0111 | [-0.0204, -0.0035] | >= -0.003 | False |
| gsplat_30k_ssim_e2 | lpips | 0.0175 | [0.0037, 0.0359] | <= +0.005 | False |
| gsplat_30k_ssim_e2 | time_to_quality_seconds | -248.5087 | [-387.6999, -129.7096] | <= 0 | True |
| gsplat_30k_ssim_e2 | speedup ratio | 1.490 | CI lo 1.111 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim_e2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 2344 | - | descriptive | - |
| gsplat_30k_ssim_e2 | energy_joules (ctrl / cand mean) | 222903 / 134489 | - | descriptive | - |
| gsplat_30k_ssim_e2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1524581 | - | descriptive | - |

| gsplat_30k_ssim_e3 | psnr_db | -0.2305 | [-0.5022, 0.0671] | >= -0.10 | False |
| gsplat_30k_ssim_e3 | ssim | -0.0198 | [-0.0363, -0.0057] | >= -0.003 | False |
| gsplat_30k_ssim_e3 | lpips | 0.0306 | [0.0058, 0.0621] | <= +0.005 | False |
| gsplat_30k_ssim_e3 | time_to_quality_seconds | -342.1803 | [-522.3407, -190.5561] | <= 0 | True |
| gsplat_30k_ssim_e3 | speedup ratio | 1.834 | CI lo 1.259 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim_e3 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 1863 | - | descriptive | - |
| gsplat_30k_ssim_e3 | energy_joules (ctrl / cand mean) | 222903 / 100794 | - | descriptive | - |
| gsplat_30k_ssim_e3 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1201699 | - | descriptive | - |

| higs_skipbwd_30k_ssim_e2 | psnr_db | -0.3497 | [-0.5559, -0.1121] | >= -0.10 | False |
| higs_skipbwd_30k_ssim_e2 | ssim | -0.0122 | [-0.0223, -0.0037] | >= -0.003 | False |
| higs_skipbwd_30k_ssim_e2 | lpips | 0.0214 | [0.0070, 0.0402] | <= +0.005 | False |
| higs_skipbwd_30k_ssim_e2 | time_to_quality_seconds | -314.8277 | [-453.3139, -192.6174] | <= 0 | True |
| higs_skipbwd_30k_ssim_e2 | speedup ratio | 1.616 | CI lo 1.276 | mean>=1.111 & lo>1.0 | True |
| higs_skipbwd_30k_ssim_e2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 2356 | - | descriptive | - |
| higs_skipbwd_30k_ssim_e2 | energy_joules (ctrl / cand mean) | 222903 / 120054 | - | descriptive | - |
| higs_skipbwd_30k_ssim_e2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1524274 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.132 | 623 | 550 |
| gsplat_27k | deep_blending/playroom | 1.113 | 489 | 439 |
| gsplat_27k | mipnerf360/bicycle | 1.131 | 1737 | 1535 |
| gsplat_27k | mipnerf360/bonsai | 1.115 | 729 | 654 |
| gsplat_27k | mipnerf360/counter | 1.099 | 740 | 673 |
| gsplat_27k | mipnerf360/garden | 1.134 | 1748 | 1542 |
| gsplat_27k | mipnerf360/kitchen | 1.115 | 801 | 718 |
| gsplat_27k | mipnerf360/room | 1.164 | 766 | 658 |
| gsplat_27k | mipnerf360/stump | 1.123 | 1574 | 1402 |
| gsplat_27k | tanks_and_temples/train | 1.098 | 516 | 470 |
| gsplat_27k | tanks_and_temples/truck | 1.199 | 498 | 415 |
| gsplat_30k_ssim075 | deep_blending/drjohnson | 1.228 | 623 | 507 |
| gsplat_30k_ssim075 | deep_blending/playroom | 1.230 | 489 | 398 |
| gsplat_30k_ssim075 | mipnerf360/bicycle | 1.366 | 1737 | 1271 |
| gsplat_30k_ssim075 | mipnerf360/bonsai | 1.310 | 729 | 556 |
| gsplat_30k_ssim075 | mipnerf360/counter | 1.292 | 740 | 573 |
| gsplat_30k_ssim075 | mipnerf360/garden | 1.447 | 1748 | 1208 |
| gsplat_30k_ssim075 | mipnerf360/kitchen | 1.255 | 801 | 638 |
| gsplat_30k_ssim075 | mipnerf360/room | 1.380 | 766 | 555 |
| gsplat_30k_ssim075 | mipnerf360/stump | 1.379 | 1574 | 1142 |
| gsplat_30k_ssim075 | tanks_and_temples/train | 0.976 | 516 | 529 |
| gsplat_30k_ssim075 | tanks_and_temples/truck | 1.108 | 498 | 449 |
| gsplat_30k_ssim_e2 | deep_blending/drjohnson | 1.419 | 623 | 439 |
| gsplat_30k_ssim_e2 | deep_blending/playroom | 1.352 | 489 | 362 |
| gsplat_30k_ssim_e2 | mipnerf360/bicycle | 1.767 | 1737 | 983 |
| gsplat_30k_ssim_e2 | mipnerf360/bonsai | 1.511 | 729 | 482 |
| gsplat_30k_ssim_e2 | mipnerf360/counter | 1.474 | 740 | 502 |
| gsplat_30k_ssim_e2 | mipnerf360/garden | 1.726 | 1748 | 1013 |
| gsplat_30k_ssim_e2 | mipnerf360/kitchen | 1.488 | 801 | 538 |
| gsplat_30k_ssim_e2 | mipnerf360/room | 1.588 | 766 | 482 |
| gsplat_30k_ssim_e2 | mipnerf360/stump | 1.730 | 1574 | 910 |
| gsplat_30k_ssim_e2 | tanks_and_temples/train | 1.111 | 516 | 465 |
| gsplat_30k_ssim_e2 | tanks_and_temples/truck | 1.227 | 498 | 406 |
| gsplat_30k_ssim_e3 | deep_blending/drjohnson | 1.709 | 623 | 364 |
| gsplat_30k_ssim_e3 | deep_blending/playroom | 1.585 | 489 | 308 |
| gsplat_30k_ssim_e3 | mipnerf360/bicycle | 2.323 | 1737 | 748 |
| gsplat_30k_ssim_e3 | mipnerf360/bonsai | 1.845 | 729 | 395 |
| gsplat_30k_ssim_e3 | mipnerf360/counter | 1.796 | 740 | 412 |
| gsplat_30k_ssim_e3 | mipnerf360/garden | 2.268 | 1748 | 771 |
| gsplat_30k_ssim_e3 | mipnerf360/kitchen | 1.826 | 801 | 438 |
| gsplat_30k_ssim_e3 | mipnerf360/room | 1.910 | 766 | 401 |
| gsplat_30k_ssim_e3 | mipnerf360/stump | 2.281 | 1574 | 690 |
| gsplat_30k_ssim_e3 | tanks_and_temples/train | 1.259 | 516 | 410 |
| gsplat_30k_ssim_e3 | tanks_and_temples/truck | 1.369 | 498 | 364 |
| higs_skipbwd_30k_ssim_e2 | deep_blending/drjohnson | 1.562 | 623 | 399 |
| higs_skipbwd_30k_ssim_e2 | deep_blending/playroom | 1.496 | 489 | 327 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/bicycle | 1.890 | 1737 | 919 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/bonsai | 1.606 | 729 | 454 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/counter | 1.591 | 740 | 465 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/garden | 1.844 | 1748 | 948 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/kitchen | 1.607 | 801 | 498 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/room | 1.702 | 766 | 450 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/stump | 1.824 | 1574 | 863 |
| higs_skipbwd_30k_ssim_e2 | tanks_and_temples/train | 1.276 | 516 | 405 |
| higs_skipbwd_30k_ssim_e2 | tanks_and_temples/truck | 1.377 | 498 | 361 |