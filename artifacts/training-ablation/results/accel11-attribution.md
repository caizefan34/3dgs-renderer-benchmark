# HiGS Accel Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_ssim075 | psnr_db | 0.0591 | [-0.0333, 0.1962] | >= -0.10 | True |
| gsplat_30k_ssim075 | ssim | -0.0019 | [-0.0037, 0.0001] | >= -0.003 | False |
| gsplat_30k_ssim075 | lpips | -0.0025 | [-0.0074, 0.0012] | <= +0.005 | True |
| gsplat_30k_ssim075 | time_to_quality_seconds | -82.6331 | [-163.3200, 8.1132] | <= 0 | False |
| gsplat_30k_ssim075 | speedup ratio | 1.125 | CI lo 0.889 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075 | peak_gpu_memory_mib (ctrl / cand mean) | 3581 / 3198 | - | descriptive | - |
| gsplat_30k_ssim075 | energy_joules (ctrl / cand mean) | 198485 / 162950 | - | descriptive | - |
| gsplat_30k_ssim075 | final_gaussian_count (ctrl / cand mean) | 2317871 / 2072855 | - | descriptive | - |

| gsplat_30k_ssim_e2 | psnr_db | -0.0632 | [-0.2153, 0.1203] | >= -0.10 | False |
| gsplat_30k_ssim_e2 | ssim | -0.0102 | [-0.0190, -0.0029] | >= -0.003 | False |
| gsplat_30k_ssim_e2 | lpips | 0.0159 | [0.0027, 0.0330] | <= +0.005 | False |
| gsplat_30k_ssim_e2 | time_to_quality_seconds | -147.8909 | [-258.7577, -41.4080] | <= 0 | True |
| gsplat_30k_ssim_e2 | speedup ratio | 1.320 | CI lo 1.012 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim_e2 | peak_gpu_memory_mib (ctrl / cand mean) | 3581 / 2344 | - | descriptive | - |
| gsplat_30k_ssim_e2 | energy_joules (ctrl / cand mean) | 198485 / 134489 | - | descriptive | - |
| gsplat_30k_ssim_e2 | final_gaussian_count (ctrl / cand mean) | 2317871 / 1524581 | - | descriptive | - |

| gsplat_30k_ssim_e3 | psnr_db | -0.1831 | [-0.4275, 0.0997] | >= -0.10 | False |
| gsplat_30k_ssim_e3 | ssim | -0.0189 | [-0.0351, -0.0051] | >= -0.003 | False |
| gsplat_30k_ssim_e3 | lpips | 0.0290 | [0.0048, 0.0596] | <= +0.005 | False |
| gsplat_30k_ssim_e3 | time_to_quality_seconds | -241.5625 | [-392.7101, -105.4202] | <= 0 | True |
| gsplat_30k_ssim_e3 | speedup ratio | 1.625 | CI lo 1.142 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim_e3 | peak_gpu_memory_mib (ctrl / cand mean) | 3581 / 1863 | - | descriptive | - |
| gsplat_30k_ssim_e3 | energy_joules (ctrl / cand mean) | 198485 / 100794 | - | descriptive | - |
| gsplat_30k_ssim_e3 | final_gaussian_count (ctrl / cand mean) | 2317871 / 1201699 | - | descriptive | - |

| higs_skipbwd_30k_ssim_e2 | psnr_db | -0.3022 | [-0.4085, -0.1485] | >= -0.10 | False |
| higs_skipbwd_30k_ssim_e2 | ssim | -0.0113 | [-0.0207, -0.0034] | >= -0.003 | False |
| higs_skipbwd_30k_ssim_e2 | lpips | 0.0198 | [0.0065, 0.0372] | <= +0.005 | False |
| higs_skipbwd_30k_ssim_e2 | time_to_quality_seconds | -214.2099 | [-321.9513, -123.2989] | <= 0 | True |
| higs_skipbwd_30k_ssim_e2 | speedup ratio | 1.432 | CI lo 1.148 | mean>=1.111 & lo>1.0 | True |
| higs_skipbwd_30k_ssim_e2 | peak_gpu_memory_mib (ctrl / cand mean) | 3581 / 2356 | - | descriptive | - |
| higs_skipbwd_30k_ssim_e2 | energy_joules (ctrl / cand mean) | 198485 / 120054 | - | descriptive | - |
| higs_skipbwd_30k_ssim_e2 | final_gaussian_count (ctrl / cand mean) | 2317871 / 1524274 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_ssim075 | deep_blending/drjohnson | 1.085 | 550 | 507 |
| gsplat_30k_ssim075 | deep_blending/playroom | 1.105 | 439 | 398 |
| gsplat_30k_ssim075 | mipnerf360/bicycle | 1.208 | 1535 | 1271 |
| gsplat_30k_ssim075 | mipnerf360/bonsai | 1.175 | 654 | 556 |
| gsplat_30k_ssim075 | mipnerf360/counter | 1.175 | 673 | 573 |
| gsplat_30k_ssim075 | mipnerf360/garden | 1.276 | 1542 | 1208 |
| gsplat_30k_ssim075 | mipnerf360/kitchen | 1.125 | 718 | 638 |
| gsplat_30k_ssim075 | mipnerf360/room | 1.186 | 658 | 555 |
| gsplat_30k_ssim075 | mipnerf360/stump | 1.228 | 1402 | 1142 |
| gsplat_30k_ssim075 | tanks_and_temples/train | 0.889 | 470 | 529 |
| gsplat_30k_ssim075 | tanks_and_temples/truck | 0.924 | 415 | 449 |
| gsplat_30k_ssim_e2 | deep_blending/drjohnson | 1.253 | 550 | 439 |
| gsplat_30k_ssim_e2 | deep_blending/playroom | 1.215 | 439 | 362 |
| gsplat_30k_ssim_e2 | mipnerf360/bicycle | 1.562 | 1535 | 983 |
| gsplat_30k_ssim_e2 | mipnerf360/bonsai | 1.355 | 654 | 482 |
| gsplat_30k_ssim_e2 | mipnerf360/counter | 1.341 | 673 | 502 |
| gsplat_30k_ssim_e2 | mipnerf360/garden | 1.523 | 1542 | 1013 |
| gsplat_30k_ssim_e2 | mipnerf360/kitchen | 1.334 | 718 | 538 |
| gsplat_30k_ssim_e2 | mipnerf360/room | 1.364 | 658 | 482 |
| gsplat_30k_ssim_e2 | mipnerf360/stump | 1.540 | 1402 | 910 |
| gsplat_30k_ssim_e2 | tanks_and_temples/train | 1.012 | 470 | 465 |
| gsplat_30k_ssim_e2 | tanks_and_temples/truck | 1.023 | 415 | 406 |
| gsplat_30k_ssim_e3 | deep_blending/drjohnson | 1.510 | 550 | 364 |
| gsplat_30k_ssim_e3 | deep_blending/playroom | 1.425 | 439 | 308 |
| gsplat_30k_ssim_e3 | mipnerf360/bicycle | 2.053 | 1535 | 748 |
| gsplat_30k_ssim_e3 | mipnerf360/bonsai | 1.655 | 654 | 395 |
| gsplat_30k_ssim_e3 | mipnerf360/counter | 1.634 | 673 | 412 |
| gsplat_30k_ssim_e3 | mipnerf360/garden | 2.000 | 1542 | 771 |
| gsplat_30k_ssim_e3 | mipnerf360/kitchen | 1.638 | 718 | 438 |
| gsplat_30k_ssim_e3 | mipnerf360/room | 1.641 | 658 | 401 |
| gsplat_30k_ssim_e3 | mipnerf360/stump | 2.031 | 1402 | 690 |
| gsplat_30k_ssim_e3 | tanks_and_temples/train | 1.147 | 470 | 410 |
| gsplat_30k_ssim_e3 | tanks_and_temples/truck | 1.142 | 415 | 364 |
| higs_skipbwd_30k_ssim_e2 | deep_blending/drjohnson | 1.380 | 550 | 399 |
| higs_skipbwd_30k_ssim_e2 | deep_blending/playroom | 1.345 | 439 | 327 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/bicycle | 1.671 | 1535 | 919 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/bonsai | 1.441 | 654 | 454 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/counter | 1.447 | 673 | 465 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/garden | 1.626 | 1542 | 948 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/kitchen | 1.441 | 718 | 498 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/room | 1.462 | 658 | 450 |
| higs_skipbwd_30k_ssim_e2 | mipnerf360/stump | 1.624 | 1402 | 863 |
| higs_skipbwd_30k_ssim_e2 | tanks_and_temples/train | 1.162 | 470 | 405 |
| higs_skipbwd_30k_ssim_e2 | tanks_and_temples/truck | 1.148 | 415 | 361 |