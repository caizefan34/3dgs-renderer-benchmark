# HiGS accel12 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_res075 | psnr_db | -1.3322 | [-1.8903, -0.7435] | >= -0.10 | False |
| gsplat_30k_res075 | ssim | -0.0263 | [-0.0353, -0.0187] | >= -0.003 | False |
| gsplat_30k_res075 | lpips | 0.0084 | [-0.0014, 0.0148] | <= +0.005 | False |
| gsplat_30k_res075 | time_to_quality_seconds | -120.2120 | [-314.6536, 74.2750] | <= 0 | False |
| gsplat_30k_res075 | speedup ratio | 1.064 | CI lo 0.757 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_res075 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 3659 | - | descriptive | - |
| gsplat_30k_res075 | energy_joules (ctrl / cand mean) | 250827 / 207994 | - | descriptive | - |
| gsplat_30k_res075 | final_gaussian_count (ctrl / cand mean) | 2332395 / 2024849 | - | descriptive | - |

| gsplat_30k_res085 | psnr_db | -0.7643 | [-1.0917, -0.4953] | >= -0.10 | False |
| gsplat_30k_res085 | ssim | -0.0199 | [-0.0279, -0.0128] | >= -0.003 | False |
| gsplat_30k_res085 | lpips | 0.0124 | [0.0080, 0.0175] | <= +0.005 | False |
| gsplat_30k_res085 | time_to_quality_seconds | -83.2910 | [-175.0102, -2.4567] | <= 0 | True |
| gsplat_30k_res085 | speedup ratio | 1.001 | CI lo 0.800 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_res085 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 3801 | - | descriptive | - |
| gsplat_30k_res085 | energy_joules (ctrl / cand mean) | 250827 / 228974 | - | descriptive | - |
| gsplat_30k_res085 | final_gaussian_count (ctrl / cand mean) | 2332395 / 2023451 | - | descriptive | - |

| higs_accum2_res075 | psnr_db | -1.3349 | [-1.9209, -0.7280] | >= -0.10 | False |
| higs_accum2_res075 | ssim | -0.0256 | [-0.0359, -0.0162] | >= -0.003 | False |
| higs_accum2_res075 | lpips | 0.0103 | [-0.0017, 0.0179] | <= +0.005 | False |
| higs_accum2_res075 | time_to_quality_seconds | -145.0996 | [-339.3203, 53.2829] | <= 0 | False |
| higs_accum2_res075 | speedup ratio | 1.112 | CI lo 0.736 | mean>=1.111 & lo>1.0 | False |
| higs_accum2_res075 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 3498 | - | descriptive | - |
| higs_accum2_res075 | energy_joules (ctrl / cand mean) | 250827 / 194206 | - | descriptive | - |
| higs_accum2_res075 | final_gaussian_count (ctrl / cand mean) | 2332395 / 1905937 | - | descriptive | - |

| higs_anchor_res075 | psnr_db | -2.7159 | [-3.7606, -1.6730] | >= -0.10 | False |
| higs_anchor_res075 | ssim | -0.0554 | [-0.0806, -0.0345] | >= -0.003 | False |
| higs_anchor_res075 | lpips | 0.0590 | [0.0277, 0.0986] | <= +0.005 | False |
| higs_anchor_res075 | time_to_quality_seconds | -209.9795 | [-438.1970, 17.9976] | <= 0 | False |
| higs_anchor_res075 | speedup ratio | 1.371 | CI lo 0.754 | mean>=1.111 & lo>1.0 | False |
| higs_anchor_res075 | peak_gpu_memory_mib (ctrl / cand mean) | 3605 / 1581 | - | descriptive | - |
| higs_anchor_res075 | energy_joules (ctrl / cand mean) | 250827 / 156839 | - | descriptive | - |
| higs_anchor_res075 | final_gaussian_count (ctrl / cand mean) | 2332395 / 539001 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_res075 | deep_blending/drjohnson | 1.017 | 547 | 537 |
| gsplat_30k_res075 | deep_blending/playroom | 1.045 | 441 | 422 |
| gsplat_30k_res075 | mipnerf360/bicycle | 1.239 | 1830 | 1477 |
| gsplat_30k_res075 | mipnerf360/bonsai | 0.937 | 750 | 800 |
| gsplat_30k_res075 | mipnerf360/counter | 1.117 | 915 | 819 |
| gsplat_30k_res075 | mipnerf360/garden | 1.334 | 2167 | 1624 |
| gsplat_30k_res075 | mipnerf360/kitchen | 1.061 | 983 | 927 |
| gsplat_30k_res075 | mipnerf360/room | 1.108 | 886 | 799 |
| gsplat_30k_res075 | mipnerf360/stump | 1.271 | 1954 | 1538 |
| gsplat_30k_res075 | tanks_and_temples/train | 0.757 | 668 | 882 |
| gsplat_30k_res075 | tanks_and_temples/truck | 0.816 | 601 | 737 |
| gsplat_30k_res085 | deep_blending/drjohnson | 1.005 | 547 | 544 |
| gsplat_30k_res085 | deep_blending/playroom | 0.987 | 441 | 446 |
| gsplat_30k_res085 | mipnerf360/bicycle | 1.075 | 1830 | 1702 |
| gsplat_30k_res085 | mipnerf360/bonsai | 0.872 | 750 | 860 |
| gsplat_30k_res085 | mipnerf360/counter | 1.041 | 915 | 878 |
| gsplat_30k_res085 | mipnerf360/garden | 1.200 | 2167 | 1806 |
| gsplat_30k_res085 | mipnerf360/kitchen | 1.022 | 983 | 962 |
| gsplat_30k_res085 | mipnerf360/room | 1.060 | 886 | 836 |
| gsplat_30k_res085 | mipnerf360/stump | 1.093 | 1954 | 1788 |
| gsplat_30k_res085 | tanks_and_temples/train | 0.800 | 668 | 835 |
| gsplat_30k_res085 | tanks_and_temples/truck | 0.859 | 601 | 700 |
| higs_accum2_res075 | deep_blending/drjohnson | 1.067 | 547 | 513 |
| higs_accum2_res075 | deep_blending/playroom | 1.131 | 441 | 389 |
| higs_accum2_res075 | mipnerf360/bicycle | 1.260 | 1830 | 1453 |
| higs_accum2_res075 | mipnerf360/bonsai | 0.961 | 750 | 780 |
| higs_accum2_res075 | mipnerf360/counter | 1.133 | 915 | 807 |
| higs_accum2_res075 | mipnerf360/garden | 1.431 | 2167 | 1515 |
| higs_accum2_res075 | mipnerf360/kitchen | 1.140 | 983 | 862 |
| higs_accum2_res075 | mipnerf360/room | 1.195 | 886 | 741 |
| higs_accum2_res075 | mipnerf360/stump | 1.346 | 1954 | 1452 |
| higs_accum2_res075 | tanks_and_temples/train | 0.736 | 668 | 908 |
| higs_accum2_res075 | tanks_and_temples/truck | 0.829 | 601 | 725 |
| higs_anchor_res075 | deep_blending/drjohnson | 1.853 | 547 | 295 |
| higs_anchor_res075 | deep_blending/playroom | 1.610 | 441 | 274 |
| higs_anchor_res075 | mipnerf360/bicycle | 1.767 | 1830 | 1036 |
| higs_anchor_res075 | mipnerf360/bonsai | 0.955 | 750 | 786 |
| higs_anchor_res075 | mipnerf360/counter | 1.252 | 915 | 731 |
| higs_anchor_res075 | mipnerf360/garden | 1.721 | 2167 | 1259 |
| higs_anchor_res075 | mipnerf360/kitchen | 1.206 | 983 | 815 |
| higs_anchor_res075 | mipnerf360/room | 1.384 | 886 | 640 |
| higs_anchor_res075 | mipnerf360/stump | 1.817 | 1954 | 1076 |
| higs_anchor_res075 | tanks_and_temples/train | 0.754 | 668 | 886 |
| higs_anchor_res075 | tanks_and_temples/truck | 0.769 | 601 | 781 |