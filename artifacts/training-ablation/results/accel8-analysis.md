# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.1266 | [-0.3006, 0.0181] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0020 | [-0.0056, 0.0004] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0043 | [0.0003, 0.0104] | <= +0.005 | False |
| gsplat_27k | time_to_quality_seconds | -105.3982 | [-164.9882, -53.4462] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.127 | CI lo 1.105 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3579 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 195786 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2315025 | - | descriptive | - |

| gsplat_27k_dens600 | psnr_db | -0.0801 | [-0.2691, 0.0810] | >= -0.10 | False |
| gsplat_27k_dens600 | ssim | -0.0014 | [-0.0039, 0.0005] | >= -0.003 | False |
| gsplat_27k_dens600 | lpips | 0.0025 | [-0.0000, 0.0055] | <= +0.005 | False |
| gsplat_27k_dens600 | time_to_quality_seconds | -23.7759 | [-139.4229, 147.0365] | <= 0 | False |
| gsplat_27k_dens600 | speedup ratio | 1.025 | CI lo 0.557 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_dens600 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3573 | - | descriptive | - |
| gsplat_27k_dens600 | energy_joules (ctrl / cand mean) | 222903 / 232049 | - | descriptive | - |
| gsplat_27k_dens600 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2313788 | - | descriptive | - |

| gsplat_27k_preload_accum8 | psnr_db | -0.7120 | [-0.9377, -0.4563] | >= -0.10 | False |
| gsplat_27k_preload_accum8 | ssim | -0.0189 | [-0.0337, -0.0067] | >= -0.003 | False |
| gsplat_27k_preload_accum8 | lpips | 0.0309 | [0.0118, 0.0555] | <= +0.005 | False |
| gsplat_27k_preload_accum8 | time_to_quality_seconds | -9.2042 | [-175.8179, 178.5446] | <= 0 | False |
| gsplat_27k_preload_accum8 | speedup ratio | 1.163 | CI lo 0.443 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_preload_accum8 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3488 | - | descriptive | - |
| gsplat_27k_preload_accum8 | energy_joules (ctrl / cand mean) | 222903 / 185036 | - | descriptive | - |
| gsplat_27k_preload_accum8 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1638625 | - | descriptive | - |

| gsplat_27k_sh_fast | psnr_db | -0.1796 | [-0.4270, 0.0618] | >= -0.10 | False |
| gsplat_27k_sh_fast | ssim | -0.0020 | [-0.0063, 0.0015] | >= -0.003 | False |
| gsplat_27k_sh_fast | lpips | 0.0042 | [-0.0001, 0.0103] | <= +0.005 | False |
| gsplat_27k_sh_fast | time_to_quality_seconds | -0.3261 | [-100.4494, 141.7543] | <= 0 | False |
| gsplat_27k_sh_fast | speedup ratio | 1.064 | CI lo 0.926 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_sh_fast | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3587 | - | descriptive | - |
| gsplat_27k_sh_fast | energy_joules (ctrl / cand mean) | 222903 / 197021 | - | descriptive | - |
| gsplat_27k_sh_fast | final_gaussian_count (ctrl / cand mean) | 2400650 / 2326235 | - | descriptive | - |

| higs_sched_27k | psnr_db | -0.7713 | [-1.0668, -0.4475] | >= -0.10 | False |
| higs_sched_27k | ssim | -0.0196 | [-0.0351, -0.0067] | >= -0.003 | False |
| higs_sched_27k | lpips | 0.0327 | [0.0132, 0.0574] | <= +0.005 | False |
| higs_sched_27k | time_to_quality_seconds | -93.4840 | [-220.0384, 59.0216] | <= 0 | False |
| higs_sched_27k | speedup ratio | 1.237 | CI lo 1.005 | mean>=1.111 & lo>1.0 | True |
| higs_sched_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3477 | - | descriptive | - |
| higs_sched_27k | energy_joules (ctrl / cand mean) | 222903 / 161581 | - | descriptive | - |
| higs_sched_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 1630054 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.131 | 623 | 550 |
| gsplat_27k | deep_blending/playroom | 1.113 | 489 | 439 |
| gsplat_27k | mipnerf360/bicycle | 1.128 | 1737 | 1539 |
| gsplat_27k | mipnerf360/bonsai | 1.111 | 729 | 656 |
| gsplat_27k | mipnerf360/counter | 1.106 | 740 | 669 |
| gsplat_27k | mipnerf360/garden | 1.132 | 1748 | 1545 |
| gsplat_27k | mipnerf360/kitchen | 1.114 | 801 | 718 |
| gsplat_27k | mipnerf360/room | 1.155 | 766 | 663 |
| gsplat_27k | mipnerf360/stump | 1.115 | 1574 | 1412 |
| gsplat_27k | tanks_and_temples/train | 1.105 | 516 | 467 |
| gsplat_27k | tanks_and_temples/truck | 1.186 | 498 | 420 |
| gsplat_27k_dens600 | deep_blending/drjohnson | 1.046 | 623 | 595 |
| gsplat_27k_dens600 | deep_blending/playroom | 1.041 | 489 | 470 |
| gsplat_27k_dens600 | mipnerf360/bicycle | 1.141 | 1737 | 1523 |
| gsplat_27k_dens600 | mipnerf360/bonsai | 1.083 | 729 | 673 |
| gsplat_27k_dens600 | mipnerf360/counter | 1.076 | 740 | 688 |
| gsplat_27k_dens600 | mipnerf360/garden | 1.134 | 1748 | 1542 |
| gsplat_27k_dens600 | mipnerf360/kitchen | 1.069 | 801 | 749 |
| gsplat_27k_dens600 | mipnerf360/room | 1.127 | 766 | 680 |
| gsplat_27k_dens600 | mipnerf360/stump | 0.557 | 1574 | 2824 |
| gsplat_27k_dens600 | tanks_and_temples/train | 0.939 | 516 | 550 |
| gsplat_27k_dens600 | tanks_and_temples/truck | 1.059 | 498 | 470 |
| gsplat_27k_preload_accum8 | deep_blending/drjohnson | 1.333 | 623 | 467 |
| gsplat_27k_preload_accum8 | deep_blending/playroom | 1.270 | 489 | 385 |
| gsplat_27k_preload_accum8 | mipnerf360/bicycle | 1.335 | 1737 | 1301 |
| gsplat_27k_preload_accum8 | mipnerf360/bonsai | 1.156 | 729 | 630 |
| gsplat_27k_preload_accum8 | mipnerf360/counter | 1.206 | 740 | 613 |
| gsplat_27k_preload_accum8 | mipnerf360/garden | 1.377 | 1748 | 1269 |
| gsplat_27k_preload_accum8 | mipnerf360/kitchen | 1.273 | 801 | 629 |
| gsplat_27k_preload_accum8 | mipnerf360/room | 0.881 | 766 | 870 |
| gsplat_27k_preload_accum8 | mipnerf360/stump | 1.273 | 1574 | 1237 |
| gsplat_27k_preload_accum8 | tanks_and_temples/train | 0.443 | 516 | 1166 |
| gsplat_27k_preload_accum8 | tanks_and_temples/truck | 1.243 | 498 | 400 |
| gsplat_27k_sh_fast | deep_blending/drjohnson | 1.039 | 623 | 599 |
| gsplat_27k_sh_fast | deep_blending/playroom | 1.012 | 489 | 483 |
| gsplat_27k_sh_fast | mipnerf360/bicycle | 1.136 | 1737 | 1529 |
| gsplat_27k_sh_fast | mipnerf360/bonsai | 1.076 | 729 | 677 |
| gsplat_27k_sh_fast | mipnerf360/counter | 1.061 | 740 | 697 |
| gsplat_27k_sh_fast | mipnerf360/garden | 1.135 | 1748 | 1540 |
| gsplat_27k_sh_fast | mipnerf360/kitchen | 1.038 | 801 | 772 |
| gsplat_27k_sh_fast | mipnerf360/room | 1.111 | 766 | 690 |
| gsplat_27k_sh_fast | mipnerf360/stump | 1.123 | 1574 | 1403 |
| gsplat_27k_sh_fast | tanks_and_temples/train | 0.926 | 516 | 558 |
| gsplat_27k_sh_fast | tanks_and_temples/truck | 1.042 | 498 | 478 |
| higs_sched_27k | deep_blending/drjohnson | 1.307 | 623 | 476 |
| higs_sched_27k | deep_blending/playroom | 1.268 | 489 | 386 |
| higs_sched_27k | mipnerf360/bicycle | 1.335 | 1737 | 1301 |
| higs_sched_27k | mipnerf360/bonsai | 1.165 | 729 | 625 |
| higs_sched_27k | mipnerf360/counter | 1.192 | 740 | 621 |
| higs_sched_27k | mipnerf360/garden | 1.369 | 1748 | 1277 |
| higs_sched_27k | mipnerf360/kitchen | 1.258 | 801 | 636 |
| higs_sched_27k | mipnerf360/room | 1.218 | 766 | 629 |
| higs_sched_27k | mipnerf360/stump | 1.271 | 1574 | 1239 |
| higs_sched_27k | tanks_and_temples/train | 1.005 | 516 | 514 |
| higs_sched_27k | tanks_and_temples/truck | 1.220 | 498 | 408 |