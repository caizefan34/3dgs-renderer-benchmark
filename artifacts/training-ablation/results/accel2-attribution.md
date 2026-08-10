# HiGS Accel2 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| higs_tilesamp_27k_r05 | psnr_db | -1.0577 | [-2.0524, -0.4768] | >= -0.10 | False |
| higs_tilesamp_27k_r05 | ssim | -0.0258 | [-0.0407, -0.0129] | >= -0.003 | False |
| higs_tilesamp_27k_r05 | lpips | 0.0395 | [0.0134, 0.0699] | <= +0.005 | False |
| higs_tilesamp_27k_r05 | time_to_quality_seconds | -29.6647 | [-128.2964, 87.3068] | <= 0 | False |
| higs_tilesamp_27k_r05 | speedup ratio | 1.165 | CI lo 1.070 | mean>=1.111 & lo>1.0 | True |
| higs_tilesamp_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 1799 | - | descriptive | - |
| higs_tilesamp_27k_r05 | energy_joules (ctrl / cand mean) | 198051 / 150669 | - | descriptive | - |
| higs_tilesamp_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2331742 / 1148313 | - | descriptive | - |

| higs_tilesamp_27k_r07 | psnr_db | 0.0909 | [-0.0369, 0.2444] | >= -0.10 | True |
| higs_tilesamp_27k_r07 | ssim | 0.0015 | [-0.0001, 0.0039] | >= -0.003 | True |
| higs_tilesamp_27k_r07 | lpips | -0.0028 | [-0.0076, 0.0001] | <= +0.005 | True |
| higs_tilesamp_27k_r07 | time_to_quality_seconds | 73.1795 | [8.8616, 165.1080] | <= 0 | False |
| higs_tilesamp_27k_r07 | speedup ratio | 0.937 | CI lo 0.761 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_27k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 3561 | - | descriptive | - |
| higs_tilesamp_27k_r07 | energy_joules (ctrl / cand mean) | 198051 / 210461 | - | descriptive | - |
| higs_tilesamp_27k_r07 | final_gaussian_count (ctrl / cand mean) | 2331742 / 2299910 | - | descriptive | - |

| higs_tilesamp_30k_r07 | psnr_db | -0.0031 | [-0.2791, 0.2054] | >= -0.10 | False |
| higs_tilesamp_30k_r07 | ssim | 0.0023 | [-0.0009, 0.0066] | >= -0.003 | True |
| higs_tilesamp_30k_r07 | lpips | -0.0040 | [-0.0111, 0.0012] | <= +0.005 | True |
| higs_tilesamp_30k_r07 | time_to_quality_seconds | 94.3428 | [55.1156, 125.3681] | <= 0 | False |
| higs_tilesamp_30k_r07 | speedup ratio | 0.854 | CI lo 0.751 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_30k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 3666 | - | descriptive | - |
| higs_tilesamp_30k_r07 | energy_joules (ctrl / cand mean) | 198051 / 222232 | - | descriptive | - |
| higs_tilesamp_30k_r07 | final_gaussian_count (ctrl / cand mean) | 2331742 / 2380325 | - | descriptive | - |

| higs_tilesamp_refine_27k_r05 | psnr_db | -0.8791 | [-1.7770, -0.3116] | >= -0.10 | False |
| higs_tilesamp_refine_27k_r05 | ssim | -0.0233 | [-0.0391, -0.0097] | >= -0.003 | False |
| higs_tilesamp_refine_27k_r05 | lpips | 0.0373 | [0.0102, 0.0692] | <= +0.005 | False |
| higs_tilesamp_refine_27k_r05 | time_to_quality_seconds | -34.3228 | [-128.3673, 75.3132] | <= 0 | False |
| higs_tilesamp_refine_27k_r05 | speedup ratio | 1.172 | CI lo 1.066 | mean>=1.111 & lo>1.0 | True |
| higs_tilesamp_refine_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 1789 | - | descriptive | - |
| higs_tilesamp_refine_27k_r05 | energy_joules (ctrl / cand mean) | 198051 / 149329 | - | descriptive | - |
| higs_tilesamp_refine_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2331742 / 1137781 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| higs_tilesamp_27k_r05 | deep_blending/drjohnson | 1.247 | 549 | 440 |
| higs_tilesamp_27k_r05 | deep_blending/playroom | 1.092 | 440 | 403 |
| higs_tilesamp_27k_r05 | mipnerf360/bicycle | 1.253 | 1539 | 1228 |
| higs_tilesamp_27k_r05 | mipnerf360/bonsai | 1.072 | 656 | 612 |
| higs_tilesamp_27k_r05 | mipnerf360/counter | 1.073 | 668 | 623 |
| higs_tilesamp_27k_r05 | mipnerf360/garden | 1.241 | 1548 | 1248 |
| higs_tilesamp_27k_r05 | mipnerf360/kitchen | 1.135 | 720 | 634 |
| higs_tilesamp_27k_r05 | mipnerf360/room | 1.070 | 658 | 615 |
| higs_tilesamp_27k_r05 | mipnerf360/stump | 1.163 | 1410 | 1212 |
| higs_tilesamp_27k_r05 | tanks_and_temples/train | 1.252 | 467 | 373 |
| higs_tilesamp_27k_r05 | tanks_and_temples/truck | 1.221 | 426 | 349 |
| higs_tilesamp_27k_r07 | deep_blending/drjohnson | 0.947 | 549 | 579 |
| higs_tilesamp_27k_r07 | deep_blending/playroom | 0.967 | 440 | 455 |
| higs_tilesamp_27k_r07 | mipnerf360/bicycle | 1.010 | 1539 | 1524 |
| higs_tilesamp_27k_r07 | mipnerf360/bonsai | 0.972 | 656 | 675 |
| higs_tilesamp_27k_r07 | mipnerf360/counter | 0.966 | 668 | 692 |
| higs_tilesamp_27k_r07 | mipnerf360/garden | 0.761 | 1548 | 2034 |
| higs_tilesamp_27k_r07 | mipnerf360/kitchen | 0.958 | 720 | 752 |
| higs_tilesamp_27k_r07 | mipnerf360/room | 0.979 | 658 | 672 |
| higs_tilesamp_27k_r07 | mipnerf360/stump | 1.020 | 1410 | 1382 |
| higs_tilesamp_27k_r07 | tanks_and_temples/train | 0.845 | 467 | 552 |
| higs_tilesamp_27k_r07 | tanks_and_temples/truck | 0.886 | 426 | 481 |
| higs_tilesamp_30k_r07 | deep_blending/drjohnson | 0.850 | 549 | 646 |
| higs_tilesamp_30k_r07 | deep_blending/playroom | 0.853 | 440 | 516 |
| higs_tilesamp_30k_r07 | mipnerf360/bicycle | 0.906 | 1539 | 1699 |
| higs_tilesamp_30k_r07 | mipnerf360/bonsai | 0.877 | 656 | 748 |
| higs_tilesamp_30k_r07 | mipnerf360/counter | 0.869 | 668 | 769 |
| higs_tilesamp_30k_r07 | mipnerf360/garden | 0.893 | 1548 | 1734 |
| higs_tilesamp_30k_r07 | mipnerf360/kitchen | 0.863 | 720 | 834 |
| higs_tilesamp_30k_r07 | mipnerf360/room | 0.873 | 658 | 754 |
| higs_tilesamp_30k_r07 | mipnerf360/stump | 0.860 | 1410 | 1641 |
| higs_tilesamp_30k_r07 | tanks_and_temples/train | 0.751 | 467 | 621 |
| higs_tilesamp_30k_r07 | tanks_and_temples/truck | 0.804 | 426 | 530 |
| higs_tilesamp_refine_27k_r05 | deep_blending/drjohnson | 1.255 | 549 | 437 |
| higs_tilesamp_refine_27k_r05 | deep_blending/playroom | 1.105 | 440 | 398 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/bicycle | 1.254 | 1539 | 1227 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/bonsai | 1.072 | 656 | 612 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/counter | 1.066 | 668 | 627 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/garden | 1.248 | 1548 | 1241 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/kitchen | 1.141 | 720 | 631 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/room | 1.087 | 658 | 605 |
| higs_tilesamp_refine_27k_r05 | mipnerf360/stump | 1.174 | 1410 | 1201 |
| higs_tilesamp_refine_27k_r05 | tanks_and_temples/train | 1.247 | 467 | 374 |
| higs_tilesamp_refine_27k_r05 | tanks_and_temples/truck | 1.243 | 426 | 343 |