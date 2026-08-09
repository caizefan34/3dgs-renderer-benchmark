# HiGS accel13 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_ssim075_gtc | psnr_db | 0.0819 | [-0.0628, 0.2960] | >= -0.10 | True |
| gsplat_30k_ssim075_gtc | ssim | -0.0019 | [-0.0039, 0.0009] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc | lpips | -0.0031 | [-0.0085, 0.0008] | <= +0.005 | True |
| gsplat_30k_ssim075_gtc | time_to_quality_seconds | -101.0186 | [-188.1819, -17.4231] | <= 0 | True |
| gsplat_30k_ssim075_gtc | speedup ratio | 1.088 | CI lo 0.864 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3742 | - | descriptive | - |
| gsplat_30k_ssim075_gtc | energy_joules (ctrl / cand mean) | 259437 / 224082 | - | descriptive | - |
| gsplat_30k_ssim075_gtc | final_gaussian_count (ctrl / cand mean) | 2325602 / 2079184 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2 | psnr_db | 0.0345 | [-0.0666, 0.1531] | >= -0.10 | True |
| gsplat_30k_ssim075_gtc_acc2 | ssim | -0.0034 | [-0.0061, -0.0007] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2 | lpips | 0.0011 | [-0.0034, 0.0048] | <= +0.005 | True |
| gsplat_30k_ssim075_gtc_acc2 | time_to_quality_seconds | -127.6824 | [-226.7798, -32.8618] | <= 0 | True |
| gsplat_30k_ssim075_gtc_acc2 | speedup ratio | 1.131 | CI lo 0.841 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3575 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2 | energy_joules (ctrl / cand mean) | 259437 / 208325 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2 | final_gaussian_count (ctrl / cand mean) | 2325602 / 1947223 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_polish24 | psnr_db | -0.0145 | [-0.0994, 0.0823] | >= -0.10 | True |
| gsplat_30k_ssim075_gtc_acc2_polish24 | ssim | -0.0018 | [-0.0037, 0.0001] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | lpips | 0.0026 | [0.0007, 0.0049] | <= +0.005 | True |
| gsplat_30k_ssim075_gtc_acc2_polish24 | time_to_quality_seconds | -99.9558 | [-183.8100, -21.8887] | <= 0 | True |
| gsplat_30k_ssim075_gtc_acc2_polish24 | speedup ratio | 1.083 | CI lo 0.837 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3599 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24 | energy_joules (ctrl / cand mean) | 259437 / 220433 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24 | final_gaussian_count (ctrl / cand mean) | 2325602 / 1957179 | - | descriptive | - |

| gsplat_30k_ssim085_gtc_acc2 | psnr_db | 0.0103 | [-0.0724, 0.1186] | >= -0.10 | True |
| gsplat_30k_ssim085_gtc_acc2 | ssim | -0.0026 | [-0.0053, 0.0001] | >= -0.003 | False |
| gsplat_30k_ssim085_gtc_acc2 | lpips | 0.0022 | [-0.0011, 0.0054] | <= +0.005 | False |
| gsplat_30k_ssim085_gtc_acc2 | time_to_quality_seconds | -62.7232 | [-129.7485, 1.4666] | <= 0 | False |
| gsplat_30k_ssim085_gtc_acc2 | speedup ratio | 1.045 | CI lo 0.824 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim085_gtc_acc2 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3764 | - | descriptive | - |
| gsplat_30k_ssim085_gtc_acc2 | energy_joules (ctrl / cand mean) | 259437 / 230564 | - | descriptive | - |
| gsplat_30k_ssim085_gtc_acc2 | final_gaussian_count (ctrl / cand mean) | 2325602 / 1974985 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_ssim075_gtc | deep_blending/drjohnson | 1.046 | 789 | 754 |
| gsplat_30k_ssim075_gtc | deep_blending/playroom | 1.041 | 627 | 602 |
| gsplat_30k_ssim075_gtc | mipnerf360/bicycle | 1.172 | 2067 | 1763 |
| gsplat_30k_ssim075_gtc | mipnerf360/bonsai | 1.125 | 887 | 788 |
| gsplat_30k_ssim075_gtc | mipnerf360/counter | 1.121 | 916 | 817 |
| gsplat_30k_ssim075_gtc | mipnerf360/garden | 1.261 | 2108 | 1672 |
| gsplat_30k_ssim075_gtc | mipnerf360/kitchen | 1.111 | 985 | 887 |
| gsplat_30k_ssim075_gtc | mipnerf360/room | 1.124 | 888 | 790 |
| gsplat_30k_ssim075_gtc | mipnerf360/stump | 1.200 | 1910 | 1592 |
| gsplat_30k_ssim075_gtc | tanks_and_temples/train | 0.864 | 671 | 777 |
| gsplat_30k_ssim075_gtc | tanks_and_temples/truck | 0.900 | 606 | 673 |
| gsplat_30k_ssim075_gtc_acc2 | deep_blending/drjohnson | 1.097 | 789 | 719 |
| gsplat_30k_ssim075_gtc_acc2 | deep_blending/playroom | 1.113 | 627 | 563 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/bicycle | 1.249 | 2067 | 1655 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/bonsai | 1.140 | 887 | 778 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/counter | 1.150 | 916 | 796 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/garden | 1.334 | 2108 | 1580 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/kitchen | 1.189 | 985 | 828 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/room | 1.153 | 888 | 770 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/stump | 1.257 | 1910 | 1520 |
| gsplat_30k_ssim075_gtc_acc2 | tanks_and_temples/train | 0.841 | 671 | 799 |
| gsplat_30k_ssim075_gtc_acc2 | tanks_and_temples/truck | 0.924 | 606 | 655 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | deep_blending/drjohnson | 1.065 | 789 | 741 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | deep_blending/playroom | 1.075 | 627 | 583 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/bicycle | 1.175 | 2067 | 1759 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/bonsai | 1.090 | 887 | 813 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/counter | 1.082 | 916 | 846 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/garden | 1.263 | 2108 | 1669 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/kitchen | 1.125 | 985 | 876 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/room | 1.102 | 888 | 806 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/stump | 1.185 | 1910 | 1612 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | tanks_and_temples/train | 0.837 | 671 | 802 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | tanks_and_temples/truck | 0.913 | 606 | 663 |
| gsplat_30k_ssim085_gtc_acc2 | deep_blending/drjohnson | 1.019 | 789 | 774 |
| gsplat_30k_ssim085_gtc_acc2 | deep_blending/playroom | 1.045 | 627 | 600 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/bicycle | 1.115 | 2067 | 1854 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/bonsai | 1.038 | 887 | 854 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/counter | 1.043 | 916 | 878 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/garden | 1.175 | 2108 | 1793 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/kitchen | 1.082 | 985 | 910 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/room | 1.050 | 888 | 846 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/stump | 1.119 | 1910 | 1707 |
| gsplat_30k_ssim085_gtc_acc2 | tanks_and_temples/train | 0.824 | 671 | 814 |
| gsplat_30k_ssim085_gtc_acc2 | tanks_and_temples/truck | 0.980 | 606 | 618 |