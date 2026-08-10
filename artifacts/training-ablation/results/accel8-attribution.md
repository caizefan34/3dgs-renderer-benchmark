# HiGS accel8 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k_dens600 | psnr_db | 0.0465 | [0.0039, 0.0993] | >= -0.10 | True |
| gsplat_27k_dens600 | ssim | 0.0005 | [-0.0003, 0.0019] | >= -0.003 | True |
| gsplat_27k_dens600 | lpips | -0.0018 | [-0.0052, 0.0002] | <= +0.005 | True |
| gsplat_27k_dens600 | time_to_quality_seconds | 81.6222 | [11.0570, 209.7194] | <= 0 | False |
| gsplat_27k_dens600 | speedup ratio | 0.909 | CI lo 0.500 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_dens600 | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3573 | - | descriptive | - |
| gsplat_27k_dens600 | energy_joules (ctrl / cand mean) | 195786 / 232049 | - | descriptive | - |
| gsplat_27k_dens600 | final_gaussian_count (ctrl / cand mean) | 2315025 / 2313788 | - | descriptive | - |

| gsplat_27k_preload_accum8 | psnr_db | -0.5854 | [-0.7711, -0.3743] | >= -0.10 | False |
| gsplat_27k_preload_accum8 | ssim | -0.0170 | [-0.0314, -0.0049] | >= -0.003 | False |
| gsplat_27k_preload_accum8 | lpips | 0.0266 | [0.0074, 0.0512] | <= +0.005 | False |
| gsplat_27k_preload_accum8 | time_to_quality_seconds | 96.1940 | [-65.1120, 307.8724] | <= 0 | False |
| gsplat_27k_preload_accum8 | speedup ratio | 1.031 | CI lo 0.401 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_preload_accum8 | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3488 | - | descriptive | - |
| gsplat_27k_preload_accum8 | energy_joules (ctrl / cand mean) | 195786 / 185036 | - | descriptive | - |
| gsplat_27k_preload_accum8 | final_gaussian_count (ctrl / cand mean) | 2315025 / 1638625 | - | descriptive | - |

| gsplat_27k_sh_fast | psnr_db | -0.0530 | [-0.1691, 0.0629] | >= -0.10 | False |
| gsplat_27k_sh_fast | ssim | 0.0000 | [-0.0014, 0.0016] | >= -0.003 | True |
| gsplat_27k_sh_fast | lpips | -0.0001 | [-0.0014, 0.0012] | <= +0.005 | True |
| gsplat_27k_sh_fast | time_to_quality_seconds | 105.0721 | [48.3250, 206.6823] | <= 0 | False |
| gsplat_27k_sh_fast | speedup ratio | 0.944 | CI lo 0.838 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_sh_fast | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3587 | - | descriptive | - |
| gsplat_27k_sh_fast | energy_joules (ctrl / cand mean) | 195786 / 197021 | - | descriptive | - |
| gsplat_27k_sh_fast | final_gaussian_count (ctrl / cand mean) | 2315025 / 2326235 | - | descriptive | - |

| higs_sched_27k | psnr_db | -0.6447 | [-0.8545, -0.4084] | >= -0.10 | False |
| higs_sched_27k | ssim | -0.0176 | [-0.0324, -0.0054] | >= -0.003 | False |
| higs_sched_27k | lpips | 0.0284 | [0.0098, 0.0528] | <= +0.005 | False |
| higs_sched_27k | time_to_quality_seconds | 11.9142 | [-75.5553, 123.7396] | <= 0 | False |
| higs_sched_27k | speedup ratio | 1.098 | CI lo 0.910 | mean>=1.111 & lo>1.0 | False |
| higs_sched_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3477 | - | descriptive | - |
| higs_sched_27k | energy_joules (ctrl / cand mean) | 195786 / 161581 | - | descriptive | - |
| higs_sched_27k | final_gaussian_count (ctrl / cand mean) | 2315025 / 1630054 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k_dens600 | deep_blending/drjohnson | 0.925 | 550 | 595 |
| gsplat_27k_dens600 | deep_blending/playroom | 0.936 | 439 | 470 |
| gsplat_27k_dens600 | mipnerf360/bicycle | 1.011 | 1539 | 1523 |
| gsplat_27k_dens600 | mipnerf360/bonsai | 0.975 | 656 | 673 |
| gsplat_27k_dens600 | mipnerf360/counter | 0.973 | 669 | 688 |
| gsplat_27k_dens600 | mipnerf360/garden | 1.002 | 1545 | 1542 |
| gsplat_27k_dens600 | mipnerf360/kitchen | 0.959 | 718 | 749 |
| gsplat_27k_dens600 | mipnerf360/room | 0.976 | 663 | 680 |
| gsplat_27k_dens600 | mipnerf360/stump | 0.500 | 1412 | 2824 |
| gsplat_27k_dens600 | tanks_and_temples/train | 0.850 | 467 | 550 |
| gsplat_27k_dens600 | tanks_and_temples/truck | 0.893 | 420 | 470 |
| gsplat_27k_preload_accum8 | deep_blending/drjohnson | 1.178 | 550 | 467 |
| gsplat_27k_preload_accum8 | deep_blending/playroom | 1.141 | 439 | 385 |
| gsplat_27k_preload_accum8 | mipnerf360/bicycle | 1.183 | 1539 | 1301 |
| gsplat_27k_preload_accum8 | mipnerf360/bonsai | 1.040 | 656 | 630 |
| gsplat_27k_preload_accum8 | mipnerf360/counter | 1.091 | 669 | 613 |
| gsplat_27k_preload_accum8 | mipnerf360/garden | 1.217 | 1545 | 1269 |
| gsplat_27k_preload_accum8 | mipnerf360/kitchen | 1.142 | 718 | 629 |
| gsplat_27k_preload_accum8 | mipnerf360/room | 0.762 | 663 | 870 |
| gsplat_27k_preload_accum8 | mipnerf360/stump | 1.142 | 1412 | 1237 |
| gsplat_27k_preload_accum8 | tanks_and_temples/train | 0.401 | 467 | 1166 |
| gsplat_27k_preload_accum8 | tanks_and_temples/truck | 1.048 | 420 | 400 |
| gsplat_27k_sh_fast | deep_blending/drjohnson | 0.919 | 550 | 599 |
| gsplat_27k_sh_fast | deep_blending/playroom | 0.910 | 439 | 483 |
| gsplat_27k_sh_fast | mipnerf360/bicycle | 1.007 | 1539 | 1529 |
| gsplat_27k_sh_fast | mipnerf360/bonsai | 0.969 | 656 | 677 |
| gsplat_27k_sh_fast | mipnerf360/counter | 0.960 | 669 | 697 |
| gsplat_27k_sh_fast | mipnerf360/garden | 1.003 | 1545 | 1540 |
| gsplat_27k_sh_fast | mipnerf360/kitchen | 0.931 | 718 | 772 |
| gsplat_27k_sh_fast | mipnerf360/room | 0.961 | 663 | 690 |
| gsplat_27k_sh_fast | mipnerf360/stump | 1.007 | 1412 | 1403 |
| gsplat_27k_sh_fast | tanks_and_temples/train | 0.838 | 467 | 558 |
| gsplat_27k_sh_fast | tanks_and_temples/truck | 0.879 | 420 | 478 |
| higs_sched_27k | deep_blending/drjohnson | 1.155 | 550 | 476 |
| higs_sched_27k | deep_blending/playroom | 1.139 | 439 | 386 |
| higs_sched_27k | mipnerf360/bicycle | 1.183 | 1539 | 1301 |
| higs_sched_27k | mipnerf360/bonsai | 1.048 | 656 | 625 |
| higs_sched_27k | mipnerf360/counter | 1.078 | 669 | 621 |
| higs_sched_27k | mipnerf360/garden | 1.210 | 1545 | 1277 |
| higs_sched_27k | mipnerf360/kitchen | 1.129 | 718 | 636 |
| higs_sched_27k | mipnerf360/room | 1.054 | 663 | 629 |
| higs_sched_27k | mipnerf360/stump | 1.140 | 1412 | 1239 |
| higs_sched_27k | tanks_and_temples/train | 0.910 | 467 | 514 |
| higs_sched_27k | tanks_and_temples/truck | 1.029 | 420 | 408 |