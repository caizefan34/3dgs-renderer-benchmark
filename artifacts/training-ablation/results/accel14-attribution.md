# HiGS accel14 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_prune16 | psnr_db | 0.0023 | [-0.1760, 0.1648] | >= -0.10 | False |
| gsplat_30k_prune16 | ssim | 0.0000 | [-0.0020, 0.0015] | >= -0.003 | True |
| gsplat_30k_prune16 | lpips | 0.0010 | [-0.0027, 0.0064] | <= +0.005 | False |
| gsplat_30k_prune16 | time_to_quality_seconds | 394.6526 | [-73.2681, 931.2125] | <= 0 | False |
| gsplat_30k_prune16 | speedup ratio | 0.844 | CI lo 0.299 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_prune16 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3667 | - | descriptive | - |
| gsplat_30k_prune16 | energy_joules (ctrl / cand mean) | 235454 / 382836 | - | descriptive | - |
| gsplat_30k_prune16 | final_gaussian_count (ctrl / cand mean) | 2311884 / 1680047 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | psnr_db | 0.0689 | [-0.0174, 0.1627] | >= -0.10 | True |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | ssim | -0.0004 | [-0.0034, 0.0030] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | lpips | 0.0009 | [-0.0032, 0.0045] | <= +0.005 | True |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | time_to_quality_seconds | -25.5778 | [-339.4264, 285.0619] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | speedup ratio | 1.206 | CI lo 0.416 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3601 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | energy_joules (ctrl / cand mean) | 235454 / 217813 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | final_gaussian_count (ctrl / cand mean) | 2311884 / 1524282 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_prune16 | psnr_db | 0.0305 | [-0.0736, 0.1509] | >= -0.10 | True |
| gsplat_30k_ssim075_gtc_acc2_prune16 | ssim | -0.0029 | [-0.0059, -0.0001] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | lpips | 0.0021 | [-0.0002, 0.0048] | <= +0.005 | True |
| gsplat_30k_ssim075_gtc_acc2_prune16 | time_to_quality_seconds | 239.1516 | [-167.8914, 622.2725] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | speedup ratio | 1.041 | CI lo 0.342 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3583 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune16 | energy_joules (ctrl / cand mean) | 235454 / 314246 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune16 | final_gaussian_count (ctrl / cand mean) | 2311884 / 1521322 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_prune18 | psnr_db | 0.0389 | [-0.1759, 0.2474] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | ssim | -0.0028 | [-0.0071, 0.0021] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | lpips | 0.0016 | [-0.0043, 0.0068] | <= +0.005 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | time_to_quality_seconds | 124.8762 | [-251.4301, 466.8156] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | speedup ratio | 1.117 | CI lo 0.458 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | peak_gpu_memory_mib (ctrl / cand mean) | 3576 / 3594 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune18 | energy_joules (ctrl / cand mean) | 235454 / 276144 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune18 | final_gaussian_count (ctrl / cand mean) | 2311884 / 1535960 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_prune16 | deep_blending/drjohnson | 0.982 | 901 | 918 |
| gsplat_30k_prune16 | deep_blending/playroom | 0.811 | 437 | 538 |
| gsplat_30k_prune16 | mipnerf360/bicycle | 0.762 | 1545 | 2028 |
| gsplat_30k_prune16 | mipnerf360/bonsai | 1.396 | 1031 | 739 |
| gsplat_30k_prune16 | mipnerf360/counter | 0.369 | 686 | 1858 |
| gsplat_30k_prune16 | mipnerf360/garden | 0.369 | 1557 | 4223 |
| gsplat_30k_prune16 | mipnerf360/kitchen | 2.184 | 1786 | 818 |
| gsplat_30k_prune16 | mipnerf360/room | 0.905 | 666 | 736 |
| gsplat_30k_prune16 | mipnerf360/stump | 0.360 | 1403 | 3894 |
| gsplat_30k_prune16 | tanks_and_temples/train | 0.299 | 473 | 1582 |
| gsplat_30k_prune16 | tanks_and_temples/truck | 0.853 | 427 | 501 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | deep_blending/drjohnson | 1.051 | 901 | 857 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | deep_blending/playroom | 1.015 | 437 | 430 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/bicycle | 0.635 | 1545 | 2433 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/bonsai | 1.765 | 1031 | 584 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/counter | 1.138 | 686 | 602 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/garden | 1.308 | 1557 | 1190 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/kitchen | 2.845 | 1786 | 628 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/room | 0.416 | 666 | 1600 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/stump | 1.224 | 1403 | 1147 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | tanks_and_temples/train | 0.895 | 473 | 528 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | tanks_and_temples/truck | 0.976 | 427 | 438 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | deep_blending/drjohnson | 2.000 | 901 | 451 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | deep_blending/playroom | 1.041 | 437 | 420 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/bicycle | 0.821 | 1545 | 1882 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/bonsai | 0.602 | 1031 | 1714 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/counter | 0.436 | 686 | 1571 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/garden | 0.530 | 1557 | 2940 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/kitchen | 3.005 | 1786 | 594 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/room | 1.237 | 666 | 539 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/stump | 0.525 | 1403 | 2672 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | tanks_and_temples/train | 0.911 | 473 | 519 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | tanks_and_temples/truck | 0.342 | 427 | 1247 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | deep_blending/drjohnson | 1.988 | 901 | 453 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | deep_blending/playroom | 0.782 | 437 | 558 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/bicycle | 1.320 | 1545 | 1171 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/bonsai | 0.606 | 1031 | 1701 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/counter | 1.194 | 686 | 574 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/garden | 0.608 | 1557 | 2560 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/kitchen | 2.981 | 1786 | 599 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/room | 0.458 | 666 | 1455 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/stump | 0.470 | 1403 | 2984 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | tanks_and_temples/train | 0.892 | 473 | 530 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | tanks_and_temples/truck | 0.988 | 427 | 432 |