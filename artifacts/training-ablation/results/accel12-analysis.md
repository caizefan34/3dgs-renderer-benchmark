# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0787 | [-0.2876, 0.0732] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0027 | [-0.0076, 0.0004] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0044 | [-0.0000, 0.0119] | <= +0.005 | False |
| gsplat_27k | time_to_quality_seconds | 93.4969 | [-1.5543, 188.0556] | <= 0 | False |
| gsplat_27k | speedup ratio | 0.897 | CI lo 0.773 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3605 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 250827 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2332395 | - | descriptive | - |

| gsplat_30k_res075 | psnr_db | -1.4109 | [-1.9640, -0.9036] | >= -0.10 | False |
| gsplat_30k_res075 | ssim | -0.0290 | [-0.0377, -0.0215] | >= -0.003 | False |
| gsplat_30k_res075 | lpips | 0.0128 | [0.0100, 0.0156] | <= +0.005 | False |
| gsplat_30k_res075 | time_to_quality_seconds | -26.7151 | [-154.4987, 96.5795] | <= 0 | False |
| gsplat_30k_res075 | speedup ratio | 0.954 | CI lo 0.585 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_res075 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3659 | - | descriptive | - |
| gsplat_30k_res075 | energy_joules (ctrl / cand mean) | 222903 / 207994 | - | descriptive | - |
| gsplat_30k_res075 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2024849 | - | descriptive | - |

| gsplat_30k_res085 | psnr_db | -0.8430 | [-1.2053, -0.5193] | >= -0.10 | False |
| gsplat_30k_res085 | ssim | -0.0226 | [-0.0330, -0.0134] | >= -0.003 | False |
| gsplat_30k_res085 | lpips | 0.0168 | [0.0117, 0.0231] | <= +0.005 | False |
| gsplat_30k_res085 | time_to_quality_seconds | 10.2059 | [-79.2461, 90.0235] | <= 0 | False |
| gsplat_30k_res085 | speedup ratio | 0.898 | CI lo 0.618 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_res085 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3801 | - | descriptive | - |
| gsplat_30k_res085 | energy_joules (ctrl / cand mean) | 222903 / 228974 | - | descriptive | - |
| gsplat_30k_res085 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2023451 | - | descriptive | - |

| higs_accum2_res075 | psnr_db | -1.4136 | [-2.0034, -0.8771] | >= -0.10 | False |
| higs_accum2_res075 | ssim | -0.0282 | [-0.0381, -0.0202] | >= -0.003 | False |
| higs_accum2_res075 | lpips | 0.0147 | [0.0094, 0.0197] | <= +0.005 | False |
| higs_accum2_res075 | time_to_quality_seconds | -51.6027 | [-189.0426, 77.5648] | <= 0 | False |
| higs_accum2_res075 | speedup ratio | 0.997 | CI lo 0.569 | mean>=1.111 & lo>1.0 | False |
| higs_accum2_res075 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3498 | - | descriptive | - |
| higs_accum2_res075 | energy_joules (ctrl / cand mean) | 222903 / 194206 | - | descriptive | - |
| higs_accum2_res075 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1905937 | - | descriptive | - |

| higs_anchor_res075 | psnr_db | -2.7946 | [-3.8131, -1.7840] | >= -0.10 | False |
| higs_anchor_res075 | ssim | -0.0581 | [-0.0832, -0.0369] | >= -0.003 | False |
| higs_anchor_res075 | lpips | 0.0634 | [0.0331, 0.1021] | <= +0.005 | False |
| higs_anchor_res075 | time_to_quality_seconds | -116.4826 | [-293.3327, 53.3016] | <= 0 | False |
| higs_anchor_res075 | speedup ratio | 1.251 | CI lo 0.583 | mean>=1.111 & lo>1.0 | False |
| higs_anchor_res075 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 1581 | - | descriptive | - |
| higs_anchor_res075 | energy_joules (ctrl / cand mean) | 222903 / 156839 | - | descriptive | - |
| higs_anchor_res075 | final_gaussian_count (ctrl / cand mean) | 2400650 / 539001 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.139 | 623 | 547 |
| gsplat_27k | deep_blending/playroom | 1.110 | 489 | 441 |
| gsplat_27k | mipnerf360/bicycle | 0.949 | 1737 | 1830 |
| gsplat_27k | mipnerf360/bonsai | 0.972 | 729 | 750 |
| gsplat_27k | mipnerf360/counter | 0.809 | 740 | 915 |
| gsplat_27k | mipnerf360/garden | 0.807 | 1748 | 2167 |
| gsplat_27k | mipnerf360/kitchen | 0.814 | 801 | 983 |
| gsplat_27k | mipnerf360/room | 0.865 | 766 | 886 |
| gsplat_27k | mipnerf360/stump | 0.806 | 1574 | 1954 |
| gsplat_27k | tanks_and_temples/train | 0.773 | 516 | 668 |
| gsplat_27k | tanks_and_temples/truck | 0.829 | 498 | 601 |
| gsplat_30k_res075 | deep_blending/drjohnson | 1.158 | 623 | 537 |
| gsplat_30k_res075 | deep_blending/playroom | 1.160 | 489 | 422 |
| gsplat_30k_res075 | mipnerf360/bicycle | 1.176 | 1737 | 1477 |
| gsplat_30k_res075 | mipnerf360/bonsai | 0.911 | 729 | 800 |
| gsplat_30k_res075 | mipnerf360/counter | 0.903 | 740 | 819 |
| gsplat_30k_res075 | mipnerf360/garden | 1.077 | 1748 | 1624 |
| gsplat_30k_res075 | mipnerf360/kitchen | 0.864 | 801 | 927 |
| gsplat_30k_res075 | mipnerf360/room | 0.958 | 766 | 799 |
| gsplat_30k_res075 | mipnerf360/stump | 1.024 | 1574 | 1538 |
| gsplat_30k_res075 | tanks_and_temples/train | 0.585 | 516 | 882 |
| gsplat_30k_res075 | tanks_and_temples/truck | 0.676 | 498 | 737 |
| gsplat_30k_res085 | deep_blending/drjohnson | 1.144 | 623 | 544 |
| gsplat_30k_res085 | deep_blending/playroom | 1.096 | 489 | 446 |
| gsplat_30k_res085 | mipnerf360/bicycle | 1.020 | 1737 | 1702 |
| gsplat_30k_res085 | mipnerf360/bonsai | 0.847 | 729 | 860 |
| gsplat_30k_res085 | mipnerf360/counter | 0.842 | 740 | 878 |
| gsplat_30k_res085 | mipnerf360/garden | 0.968 | 1748 | 1806 |
| gsplat_30k_res085 | mipnerf360/kitchen | 0.832 | 801 | 962 |
| gsplat_30k_res085 | mipnerf360/room | 0.916 | 766 | 836 |
| gsplat_30k_res085 | mipnerf360/stump | 0.881 | 1574 | 1788 |
| gsplat_30k_res085 | tanks_and_temples/train | 0.618 | 516 | 835 |
| gsplat_30k_res085 | tanks_and_temples/truck | 0.712 | 498 | 700 |
| higs_accum2_res075 | deep_blending/drjohnson | 1.214 | 623 | 513 |
| higs_accum2_res075 | deep_blending/playroom | 1.255 | 489 | 389 |
| higs_accum2_res075 | mipnerf360/bicycle | 1.196 | 1737 | 1453 |
| higs_accum2_res075 | mipnerf360/bonsai | 0.933 | 729 | 780 |
| higs_accum2_res075 | mipnerf360/counter | 0.917 | 740 | 807 |
| higs_accum2_res075 | mipnerf360/garden | 1.154 | 1748 | 1515 |
| higs_accum2_res075 | mipnerf360/kitchen | 0.929 | 801 | 862 |
| higs_accum2_res075 | mipnerf360/room | 1.033 | 766 | 741 |
| higs_accum2_res075 | mipnerf360/stump | 1.085 | 1574 | 1452 |
| higs_accum2_res075 | tanks_and_temples/train | 0.569 | 516 | 908 |
| higs_accum2_res075 | tanks_and_temples/truck | 0.687 | 498 | 725 |
| higs_anchor_res075 | deep_blending/drjohnson | 2.110 | 623 | 295 |
| higs_anchor_res075 | deep_blending/playroom | 1.786 | 489 | 274 |
| higs_anchor_res075 | mipnerf360/bicycle | 1.677 | 1737 | 1036 |
| higs_anchor_res075 | mipnerf360/bonsai | 0.927 | 729 | 786 |
| higs_anchor_res075 | mipnerf360/counter | 1.012 | 740 | 731 |
| higs_anchor_res075 | mipnerf360/garden | 1.388 | 1748 | 1259 |
| higs_anchor_res075 | mipnerf360/kitchen | 0.982 | 801 | 815 |
| higs_anchor_res075 | mipnerf360/room | 1.196 | 766 | 640 |
| higs_anchor_res075 | mipnerf360/stump | 1.463 | 1574 | 1076 |
| higs_anchor_res075 | tanks_and_temples/train | 0.583 | 516 | 886 |
| higs_anchor_res075 | tanks_and_temples/truck | 0.637 | 498 | 781 |