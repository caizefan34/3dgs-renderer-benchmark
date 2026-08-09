# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0227 | [-0.0830, 0.0397] | >= -0.10 | True |
| gsplat_27k | ssim | -0.0004 | [-0.0014, 0.0007] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0015 | [0.0004, 0.0032] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | 176.5158 | [131.0943, 222.7463] | <= 0 | False |
| gsplat_27k | speedup ratio | 0.815 | CI lo 0.769 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3593 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 259437 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2325602 | - | descriptive | - |

| gsplat_30k_ssim075_gtc | psnr_db | 0.0592 | [-0.1082, 0.2625] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc | ssim | -0.0022 | [-0.0045, 0.0004] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc | lpips | -0.0016 | [-0.0065, 0.0018] | <= +0.005 | True |
| gsplat_30k_ssim075_gtc | time_to_quality_seconds | 75.4972 | [28.3199, 126.8913] | <= 0 | False |
| gsplat_30k_ssim075_gtc | speedup ratio | 0.888 | CI lo 0.664 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3742 | - | descriptive | - |
| gsplat_30k_ssim075_gtc | energy_joules (ctrl / cand mean) | 222903 / 224082 | - | descriptive | - |
| gsplat_30k_ssim075_gtc | final_gaussian_count (ctrl / cand mean) | 2400650 / 2079184 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2 | psnr_db | 0.0117 | [-0.1164, 0.1535] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc_acc2 | ssim | -0.0037 | [-0.0070, -0.0006] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2 | lpips | 0.0026 | [-0.0020, 0.0069] | <= +0.005 | False |
| gsplat_30k_ssim075_gtc_acc2 | time_to_quality_seconds | 48.8334 | [-9.6318, 109.7700] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2 | speedup ratio | 0.923 | CI lo 0.646 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3575 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2 | energy_joules (ctrl / cand mean) | 222903 / 208325 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1947223 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_polish24 | psnr_db | -0.0372 | [-0.1722, 0.1126] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | ssim | -0.0022 | [-0.0048, 0.0005] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | lpips | 0.0041 | [0.0013, 0.0073] | <= +0.005 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | time_to_quality_seconds | 76.5599 | [10.2567, 148.8338] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | speedup ratio | 0.884 | CI lo 0.644 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3599 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24 | energy_joules (ctrl / cand mean) | 222903 / 220433 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1957179 | - | descriptive | - |

| gsplat_30k_ssim085_gtc_acc2 | psnr_db | -0.0124 | [-0.1388, 0.1371] | >= -0.10 | False |
| gsplat_30k_ssim085_gtc_acc2 | ssim | -0.0030 | [-0.0063, 0.0003] | >= -0.003 | False |
| gsplat_30k_ssim085_gtc_acc2 | lpips | 0.0037 | [-0.0003, 0.0080] | <= +0.005 | False |
| gsplat_30k_ssim085_gtc_acc2 | time_to_quality_seconds | 113.7926 | [52.1999, 185.5747] | <= 0 | False |
| gsplat_30k_ssim085_gtc_acc2 | speedup ratio | 0.852 | CI lo 0.634 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim085_gtc_acc2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3764 | - | descriptive | - |
| gsplat_30k_ssim085_gtc_acc2 | energy_joules (ctrl / cand mean) | 222903 / 230564 | - | descriptive | - |
| gsplat_30k_ssim085_gtc_acc2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1974985 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 0.789 | 623 | 789 |
| gsplat_27k | deep_blending/playroom | 0.780 | 489 | 627 |
| gsplat_27k | mipnerf360/bicycle | 0.840 | 1737 | 2067 |
| gsplat_27k | mipnerf360/bonsai | 0.822 | 729 | 887 |
| gsplat_27k | mipnerf360/counter | 0.808 | 740 | 916 |
| gsplat_27k | mipnerf360/garden | 0.829 | 1748 | 2108 |
| gsplat_27k | mipnerf360/kitchen | 0.813 | 801 | 985 |
| gsplat_27k | mipnerf360/room | 0.863 | 766 | 888 |
| gsplat_27k | mipnerf360/stump | 0.824 | 1574 | 1910 |
| gsplat_27k | tanks_and_temples/train | 0.769 | 516 | 671 |
| gsplat_27k | tanks_and_temples/truck | 0.822 | 498 | 606 |
| gsplat_30k_ssim075_gtc | deep_blending/drjohnson | 0.826 | 623 | 754 |
| gsplat_30k_ssim075_gtc | deep_blending/playroom | 0.812 | 489 | 602 |
| gsplat_30k_ssim075_gtc | mipnerf360/bicycle | 0.985 | 1737 | 1763 |
| gsplat_30k_ssim075_gtc | mipnerf360/bonsai | 0.924 | 729 | 788 |
| gsplat_30k_ssim075_gtc | mipnerf360/counter | 0.905 | 740 | 817 |
| gsplat_30k_ssim075_gtc | mipnerf360/garden | 1.046 | 1748 | 1672 |
| gsplat_30k_ssim075_gtc | mipnerf360/kitchen | 0.903 | 801 | 887 |
| gsplat_30k_ssim075_gtc | mipnerf360/room | 0.970 | 766 | 790 |
| gsplat_30k_ssim075_gtc | mipnerf360/stump | 0.989 | 1574 | 1592 |
| gsplat_30k_ssim075_gtc | tanks_and_temples/train | 0.664 | 516 | 777 |
| gsplat_30k_ssim075_gtc | tanks_and_temples/truck | 0.740 | 498 | 673 |
| gsplat_30k_ssim075_gtc_acc2 | deep_blending/drjohnson | 0.866 | 623 | 719 |
| gsplat_30k_ssim075_gtc_acc2 | deep_blending/playroom | 0.868 | 489 | 563 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/bicycle | 1.049 | 1737 | 1655 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/bonsai | 0.937 | 729 | 778 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/counter | 0.929 | 740 | 796 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/garden | 1.107 | 1748 | 1580 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/kitchen | 0.967 | 801 | 828 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/room | 0.995 | 766 | 770 |
| gsplat_30k_ssim075_gtc_acc2 | mipnerf360/stump | 1.036 | 1574 | 1520 |
| gsplat_30k_ssim075_gtc_acc2 | tanks_and_temples/train | 0.646 | 516 | 799 |
| gsplat_30k_ssim075_gtc_acc2 | tanks_and_temples/truck | 0.759 | 498 | 655 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | deep_blending/drjohnson | 0.840 | 623 | 741 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | deep_blending/playroom | 0.838 | 489 | 583 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/bicycle | 0.987 | 1737 | 1759 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/bonsai | 0.896 | 729 | 813 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/counter | 0.875 | 740 | 846 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/garden | 1.047 | 1748 | 1669 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/kitchen | 0.914 | 801 | 876 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/room | 0.951 | 766 | 806 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | mipnerf360/stump | 0.977 | 1574 | 1612 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | tanks_and_temples/train | 0.644 | 516 | 802 |
| gsplat_30k_ssim075_gtc_acc2_polish24 | tanks_and_temples/truck | 0.751 | 498 | 663 |
| gsplat_30k_ssim085_gtc_acc2 | deep_blending/drjohnson | 0.804 | 623 | 774 |
| gsplat_30k_ssim085_gtc_acc2 | deep_blending/playroom | 0.815 | 489 | 600 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/bicycle | 0.937 | 1737 | 1854 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/bonsai | 0.853 | 729 | 854 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/counter | 0.843 | 740 | 878 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/garden | 0.975 | 1748 | 1793 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/kitchen | 0.880 | 801 | 910 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/room | 0.906 | 766 | 846 |
| gsplat_30k_ssim085_gtc_acc2 | mipnerf360/stump | 0.922 | 1574 | 1707 |
| gsplat_30k_ssim085_gtc_acc2 | tanks_and_temples/train | 0.634 | 516 | 814 |
| gsplat_30k_ssim085_gtc_acc2 | tanks_and_temples/truck | 0.806 | 498 | 618 |