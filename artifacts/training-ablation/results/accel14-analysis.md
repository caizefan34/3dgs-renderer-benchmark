# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.1028 | [-0.2413, 0.0312] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0015 | [-0.0039, 0.0002] | >= -0.003 | False |
| gsplat_27k | lpips | 0.0025 | [0.0006, 0.0048] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | 75.8633 | [-76.0402, 285.6984] | <= 0 | False |
| gsplat_27k | speedup ratio | 0.984 | CI lo 0.448 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3576 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 235454 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2311884 | - | descriptive | - |

| gsplat_30k_prune16 | psnr_db | -0.1004 | [-0.2981, 0.0556] | >= -0.10 | False |
| gsplat_30k_prune16 | ssim | -0.0015 | [-0.0056, 0.0009] | >= -0.003 | False |
| gsplat_30k_prune16 | lpips | 0.0035 | [-0.0003, 0.0103] | <= +0.005 | False |
| gsplat_30k_prune16 | time_to_quality_seconds | 470.5159 | [103.4024, 930.8967] | <= 0 | False |
| gsplat_30k_prune16 | speedup ratio | 0.726 | CI lo 0.326 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_prune16 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3667 | - | descriptive | - |
| gsplat_30k_prune16 | energy_joules (ctrl / cand mean) | 222903 / 382836 | - | descriptive | - |
| gsplat_30k_prune16 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1680047 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | psnr_db | -0.0339 | [-0.1941, 0.1428] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | ssim | -0.0019 | [-0.0047, 0.0008] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | lpips | 0.0034 | [0.0000, 0.0071] | <= +0.005 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | time_to_quality_seconds | 50.2855 | [-151.0647, 276.6690] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | speedup ratio | 1.069 | CI lo 0.479 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3601 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | energy_joules (ctrl / cand mean) | 222903 / 217813 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1524282 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_prune16 | psnr_db | -0.0722 | [-0.2313, 0.0969] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | ssim | -0.0045 | [-0.0077, -0.0013] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | lpips | 0.0045 | [0.0015, 0.0078] | <= +0.005 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | time_to_quality_seconds | 315.0149 | [14.3645, 621.9987] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | speedup ratio | 0.883 | CI lo 0.399 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune16 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3583 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune16 | energy_joules (ctrl / cand mean) | 222903 / 314246 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune16 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1521322 | - | descriptive | - |

| gsplat_30k_ssim075_gtc_acc2_prune18 | psnr_db | -0.0639 | [-0.2239, 0.1014] | >= -0.10 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | ssim | -0.0044 | [-0.0079, -0.0007] | >= -0.003 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | lpips | 0.0041 | [-0.0006, 0.0089] | <= +0.005 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | time_to_quality_seconds | 200.7396 | [-80.9493, 496.1187] | <= 0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | speedup ratio | 0.968 | CI lo 0.428 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim075_gtc_acc2_prune18 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3594 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune18 | energy_joules (ctrl / cand mean) | 222903 / 276144 | - | descriptive | - |
| gsplat_30k_ssim075_gtc_acc2_prune18 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1535960 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 0.691 | 623 | 901 |
| gsplat_27k | deep_blending/playroom | 1.120 | 489 | 437 |
| gsplat_27k | mipnerf360/bicycle | 1.124 | 1737 | 1545 |
| gsplat_27k | mipnerf360/bonsai | 0.707 | 729 | 1031 |
| gsplat_27k | mipnerf360/counter | 1.079 | 740 | 686 |
| gsplat_27k | mipnerf360/garden | 1.123 | 1748 | 1557 |
| gsplat_27k | mipnerf360/kitchen | 0.448 | 801 | 1786 |
| gsplat_27k | mipnerf360/room | 1.150 | 766 | 666 |
| gsplat_27k | mipnerf360/stump | 1.122 | 1574 | 1403 |
| gsplat_27k | tanks_and_temples/train | 1.093 | 516 | 473 |
| gsplat_27k | tanks_and_temples/truck | 1.166 | 498 | 427 |
| gsplat_30k_prune16 | deep_blending/drjohnson | 0.678 | 623 | 918 |
| gsplat_30k_prune16 | deep_blending/playroom | 0.908 | 489 | 538 |
| gsplat_30k_prune16 | mipnerf360/bicycle | 0.856 | 1737 | 2028 |
| gsplat_30k_prune16 | mipnerf360/bonsai | 0.986 | 729 | 739 |
| gsplat_30k_prune16 | mipnerf360/counter | 0.398 | 740 | 1858 |
| gsplat_30k_prune16 | mipnerf360/garden | 0.414 | 1748 | 4223 |
| gsplat_30k_prune16 | mipnerf360/kitchen | 0.979 | 801 | 818 |
| gsplat_30k_prune16 | mipnerf360/room | 1.041 | 766 | 736 |
| gsplat_30k_prune16 | mipnerf360/stump | 0.404 | 1574 | 3894 |
| gsplat_30k_prune16 | tanks_and_temples/train | 0.326 | 516 | 1582 |
| gsplat_30k_prune16 | tanks_and_temples/truck | 0.994 | 498 | 501 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | deep_blending/drjohnson | 0.726 | 623 | 857 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | deep_blending/playroom | 1.137 | 489 | 430 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/bicycle | 0.714 | 1737 | 2433 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/bonsai | 1.247 | 729 | 584 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/counter | 1.228 | 740 | 602 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/garden | 1.469 | 1748 | 1190 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/kitchen | 1.276 | 801 | 628 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/room | 0.479 | 766 | 1600 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | mipnerf360/stump | 1.373 | 1574 | 1147 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | tanks_and_temples/train | 0.977 | 516 | 528 |
| gsplat_30k_ssim075_gtc_acc2_polish24_prune16 | tanks_and_temples/truck | 1.138 | 498 | 438 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | deep_blending/drjohnson | 1.382 | 623 | 451 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | deep_blending/playroom | 1.165 | 489 | 420 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/bicycle | 0.923 | 1737 | 1882 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/bonsai | 0.425 | 729 | 1714 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/counter | 0.471 | 740 | 1571 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/garden | 0.595 | 1748 | 2940 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/kitchen | 1.347 | 801 | 594 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/room | 1.423 | 766 | 539 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | mipnerf360/stump | 0.589 | 1574 | 2672 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | tanks_and_temples/train | 0.995 | 516 | 519 |
| gsplat_30k_ssim075_gtc_acc2_prune16 | tanks_and_temples/truck | 0.399 | 498 | 1247 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | deep_blending/drjohnson | 1.374 | 623 | 453 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | deep_blending/playroom | 0.876 | 489 | 558 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/bicycle | 1.484 | 1737 | 1171 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/bonsai | 0.428 | 729 | 1701 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/counter | 1.289 | 740 | 574 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/garden | 0.683 | 1748 | 2560 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/kitchen | 1.336 | 801 | 599 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/room | 0.527 | 766 | 1455 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | mipnerf360/stump | 0.528 | 1574 | 2984 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | tanks_and_temples/train | 0.975 | 516 | 530 |
| gsplat_30k_ssim075_gtc_acc2_prune18 | tanks_and_temples/truck | 1.152 | 498 | 432 |