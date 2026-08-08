# HiGS Exploration Matrix Analysis (5 configs x 11 scenes x seed 0, paired vs gsplat)

- baseline: `gsplat`; configs: higs_switch_10k, higs_switch_12k_s65, higs_switch_12k_s70, higs_switch_6k, higs_switch_8k
- exploration jobs: 55; control (gsplat) cells: 33

## Paired NI + acceleration gates (scene-block bootstrap 95% CI)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| higs_switch_10k | psnr_db | -0.1813 | [-0.3631, -0.0162] | >= -0.10 | False |
| higs_switch_10k | ssim | -0.0035 | [-0.0064, -0.0009] | >= -0.003 | False |
| higs_switch_10k | lpips | 0.0037 | [0.0014, 0.0064] | <= +0.005 | False |
| higs_switch_10k | time_to_quality_seconds | -64.3623 | [-162.4406, 16.4437] | <= 0 | False |
| higs_switch_10k | speedup ratio | 1.089 | CI lo 0.809 | mean>=1.111 & lo>1.0 | False |
| higs_switch_10k | TTQ delta (s) | -64.36 | [-162.44, 16.44] | <= 0 | False |
| higs_switch_10k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3200 | - | descriptive | - |
| higs_switch_10k | energy_joules (gsplat / cand mean) | 222903 / 181911 | - | descriptive | - |
| higs_switch_10k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2109474 | - | descriptive | - |

| higs_switch_12k_s65 | psnr_db | -0.1829 | [-0.3186, -0.0526] | >= -0.10 | False |
| higs_switch_12k_s65 | ssim | -0.0034 | [-0.0066, -0.0007] | >= -0.003 | False |
| higs_switch_12k_s65 | lpips | 0.0047 | [0.0021, 0.0081] | <= +0.005 | False |
| higs_switch_12k_s65 | time_to_quality_seconds | -84.5878 | [-185.6760, 29.7666] | <= 0 | False |
| higs_switch_12k_s65 | speedup ratio | 1.131 | CI lo 0.833 | mean>=1.111 & lo>1.0 | False |
| higs_switch_12k_s65 | TTQ delta (s) | -84.59 | [-185.68, 29.77] | <= 0 | False |
| higs_switch_12k_s65 | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3125 | - | descriptive | - |
| higs_switch_12k_s65 | energy_joules (gsplat / cand mean) | 222903 / 173889 | - | descriptive | - |
| higs_switch_12k_s65 | final_gaussian_count (gsplat / cand mean) | 2400650 / 2060179 | - | descriptive | - |

| higs_switch_12k_s70 | psnr_db | -0.1458 | [-0.3304, 0.0176] | >= -0.10 | False |
| higs_switch_12k_s70 | ssim | -0.0038 | [-0.0075, -0.0008] | >= -0.003 | False |
| higs_switch_12k_s70 | lpips | 0.0044 | [0.0021, 0.0071] | <= +0.005 | False |
| higs_switch_12k_s70 | time_to_quality_seconds | -93.0494 | [-186.9650, 7.6235] | <= 0 | False |
| higs_switch_12k_s70 | speedup ratio | 1.127 | CI lo 0.849 | mean>=1.111 & lo>1.0 | False |
| higs_switch_12k_s70 | TTQ delta (s) | -93.05 | [-186.97, 7.62] | <= 0 | False |
| higs_switch_12k_s70 | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3094 | - | descriptive | - |
| higs_switch_12k_s70 | energy_joules (gsplat / cand mean) | 222903 / 177417 | - | descriptive | - |
| higs_switch_12k_s70 | final_gaussian_count (gsplat / cand mean) | 2400650 / 2039549 | - | descriptive | - |

| higs_switch_6k | psnr_db | -0.1252 | [-0.3389, 0.0514] | >= -0.10 | False |
| higs_switch_6k | ssim | -0.0026 | [-0.0061, 0.0002] | >= -0.003 | False |
| higs_switch_6k | lpips | 0.0030 | [0.0005, 0.0063] | <= +0.005 | False |
| higs_switch_6k | time_to_quality_seconds | -66.7535 | [-138.3474, -2.6019] | <= 0 | True |
| higs_switch_6k | speedup ratio | 1.055 | CI lo 0.805 | mean>=1.111 & lo>1.0 | False |
| higs_switch_6k | TTQ delta (s) | -66.75 | [-138.35, -2.60] | <= 0 | True |
| higs_switch_6k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3340 | - | descriptive | - |
| higs_switch_6k | energy_joules (gsplat / cand mean) | 222903 / 191236 | - | descriptive | - |
| higs_switch_6k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2199566 | - | descriptive | - |

| higs_switch_8k | psnr_db | -0.1833 | [-0.4331, 0.0315] | >= -0.10 | False |
| higs_switch_8k | ssim | -0.0050 | [-0.0109, -0.0002] | >= -0.003 | False |
| higs_switch_8k | lpips | 0.0060 | [0.0012, 0.0123] | <= +0.005 | False |
| higs_switch_8k | time_to_quality_seconds | -88.9144 | [-168.8677, -16.2640] | <= 0 | True |
| higs_switch_8k | speedup ratio | 1.070 | CI lo 0.811 | mean>=1.111 & lo>1.0 | False |
| higs_switch_8k | TTQ delta (s) | -88.91 | [-168.87, -16.26] | <= 0 | True |
| higs_switch_8k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3241 | - | descriptive | - |
| higs_switch_8k | energy_joules (gsplat / cand mean) | 222903 / 186254 | - | descriptive | - |
| higs_switch_8k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2135883 | - | descriptive | - |

## Per-scene speedup ratio (gsplat wall / candidate wall, seed 0)

| config | scene | speedup | gsplat s | cand s |
|---|---|---|---|---|
| higs_switch_10k | deep_blending/drjohnson | 1.115 | 623 | 558 |
| higs_switch_10k | deep_blending/playroom | 1.169 | 489 | 418 |
| higs_switch_10k | mipnerf360/bicycle | 1.296 | 1737 | 1340 |
| higs_switch_10k | mipnerf360/bonsai | 1.007 | 729 | 723 |
| higs_switch_10k | mipnerf360/counter | 1.003 | 740 | 738 |
| higs_switch_10k | mipnerf360/garden | 1.308 | 1748 | 1337 |
| higs_switch_10k | mipnerf360/kitchen | 0.905 | 801 | 885 |
| higs_switch_10k | mipnerf360/room | 1.129 | 766 | 679 |
| higs_switch_10k | mipnerf360/stump | 1.274 | 1574 | 1236 |
| higs_switch_10k | tanks_and_temples/train | 0.809 | 516 | 638 |
| higs_switch_10k | tanks_and_temples/truck | 0.962 | 498 | 518 |
| higs_switch_12k_s65 | deep_blending/drjohnson | 1.122 | 623 | 555 |
| higs_switch_12k_s65 | deep_blending/playroom | 1.131 | 489 | 432 |
| higs_switch_12k_s65 | mipnerf360/bicycle | 1.284 | 1737 | 1353 |
| higs_switch_12k_s65 | mipnerf360/bonsai | 1.116 | 729 | 653 |
| higs_switch_12k_s65 | mipnerf360/counter | 1.116 | 740 | 663 |
| higs_switch_12k_s65 | mipnerf360/garden | 1.302 | 1748 | 1343 |
| higs_switch_12k_s65 | mipnerf360/kitchen | 1.039 | 801 | 770 |
| higs_switch_12k_s65 | mipnerf360/room | 1.236 | 766 | 620 |
| higs_switch_12k_s65 | mipnerf360/stump | 1.265 | 1574 | 1245 |
| higs_switch_12k_s65 | tanks_and_temples/train | 0.833 | 516 | 620 |
| higs_switch_12k_s65 | tanks_and_temples/truck | 0.991 | 498 | 502 |
| higs_switch_12k_s70 | deep_blending/drjohnson | 1.136 | 623 | 548 |
| higs_switch_12k_s70 | deep_blending/playroom | 1.115 | 489 | 439 |
| higs_switch_12k_s70 | mipnerf360/bicycle | 1.265 | 1737 | 1373 |
| higs_switch_12k_s70 | mipnerf360/bonsai | 1.111 | 729 | 656 |
| higs_switch_12k_s70 | mipnerf360/counter | 1.108 | 740 | 668 |
| higs_switch_12k_s70 | mipnerf360/garden | 1.283 | 1748 | 1363 |
| higs_switch_12k_s70 | mipnerf360/kitchen | 1.067 | 801 | 750 |
| higs_switch_12k_s70 | mipnerf360/room | 1.219 | 766 | 628 |
| higs_switch_12k_s70 | mipnerf360/stump | 1.240 | 1574 | 1269 |
| higs_switch_12k_s70 | tanks_and_temples/train | 0.849 | 516 | 608 |
| higs_switch_12k_s70 | tanks_and_temples/truck | 1.007 | 498 | 494 |
| higs_switch_6k | deep_blending/drjohnson | 1.082 | 623 | 576 |
| higs_switch_6k | deep_blending/playroom | 1.098 | 489 | 445 |
| higs_switch_6k | mipnerf360/bicycle | 1.184 | 1737 | 1467 |
| higs_switch_6k | mipnerf360/bonsai | 1.002 | 729 | 727 |
| higs_switch_6k | mipnerf360/counter | 0.994 | 740 | 744 |
| higs_switch_6k | mipnerf360/garden | 1.211 | 1748 | 1444 |
| higs_switch_6k | mipnerf360/kitchen | 0.962 | 801 | 832 |
| higs_switch_6k | mipnerf360/room | 1.101 | 766 | 696 |
| higs_switch_6k | mipnerf360/stump | 1.178 | 1574 | 1337 |
| higs_switch_6k | tanks_and_temples/train | 0.805 | 516 | 641 |
| higs_switch_6k | tanks_and_temples/truck | 0.985 | 498 | 505 |
| higs_switch_8k | deep_blending/drjohnson | 1.100 | 623 | 566 |
| higs_switch_8k | deep_blending/playroom | 1.132 | 489 | 432 |
| higs_switch_8k | mipnerf360/bicycle | 1.241 | 1737 | 1400 |
| higs_switch_8k | mipnerf360/bonsai | 1.002 | 729 | 727 |
| higs_switch_8k | mipnerf360/counter | 0.993 | 740 | 745 |
| higs_switch_8k | mipnerf360/garden | 1.267 | 1748 | 1380 |
| higs_switch_8k | mipnerf360/kitchen | 0.930 | 801 | 861 |
| higs_switch_8k | mipnerf360/room | 1.119 | 766 | 685 |
| higs_switch_8k | mipnerf360/stump | 1.221 | 1574 | 1289 |
| higs_switch_8k | tanks_and_temples/train | 0.811 | 516 | 637 |
| higs_switch_8k | tanks_and_temples/truck | 0.959 | 498 | 519 |