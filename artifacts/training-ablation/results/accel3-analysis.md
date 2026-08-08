# HiGS Accel3 Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0473 | [-0.1655, 0.0736] | >= -0.10 | False |
| gsplat_27k | ssim | 0.0001 | [-0.0012, 0.0014] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0003 | [-0.0025, 0.0026] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | -86.7254 | [-122.6497, -55.1538] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.124 | CI lo 1.090 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3589 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 196037 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2323026 | - | descriptive | - |

| higs_tilesamp_phase_27k_r04 | psnr_db | -0.6750 | [-1.5165, -0.1221] | >= -0.10 | False |
| higs_tilesamp_phase_27k_r04 | ssim | -0.0105 | [-0.0209, -0.0027] | >= -0.003 | False |
| higs_tilesamp_phase_27k_r04 | lpips | 0.0161 | [0.0043, 0.0320] | <= +0.005 | False |
| higs_tilesamp_phase_27k_r04 | time_to_quality_seconds | -122.0049 | [-193.2050, -59.2860] | <= 0 | True |
| higs_tilesamp_phase_27k_r04 | speedup ratio | 1.099 | CI lo 0.986 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_phase_27k_r04 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3568 | - | descriptive | - |
| higs_tilesamp_phase_27k_r04 | energy_joules (ctrl / cand mean) | 222903 / 192031 | - | descriptive | - |
| higs_tilesamp_phase_27k_r04 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2308272 | - | descriptive | - |

| higs_tilesamp_phase_27k_r05 | psnr_db | -0.6922 | [-1.5423, -0.1494] | >= -0.10 | False |
| higs_tilesamp_phase_27k_r05 | ssim | -0.0094 | [-0.0181, -0.0029] | >= -0.003 | False |
| higs_tilesamp_phase_27k_r05 | lpips | 0.0167 | [0.0047, 0.0328] | <= +0.005 | False |
| higs_tilesamp_phase_27k_r05 | time_to_quality_seconds | -124.1914 | [-192.7404, -64.1578] | <= 0 | True |
| higs_tilesamp_phase_27k_r05 | speedup ratio | 1.104 | CI lo 0.988 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_phase_27k_r05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3562 | - | descriptive | - |
| higs_tilesamp_phase_27k_r05 | energy_joules (ctrl / cand mean) | 222903 / 188504 | - | descriptive | - |
| higs_tilesamp_phase_27k_r05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2307488 | - | descriptive | - |

| higs_tilesamp_phase_27k_r05_polish | psnr_db | -0.3805 | [-0.8409, -0.0851] | >= -0.10 | False |
| higs_tilesamp_phase_27k_r05_polish | ssim | -0.0043 | [-0.0088, -0.0007] | >= -0.003 | False |
| higs_tilesamp_phase_27k_r05_polish | lpips | 0.0088 | [0.0031, 0.0164] | <= +0.005 | False |
| higs_tilesamp_phase_27k_r05_polish | time_to_quality_seconds | -108.5110 | [-162.9068, -58.9369] | <= 0 | True |
| higs_tilesamp_phase_27k_r05_polish | speedup ratio | 1.100 | CI lo 0.974 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_phase_27k_r05_polish | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3563 | - | descriptive | - |
| higs_tilesamp_phase_27k_r05_polish | energy_joules (ctrl / cand mean) | 222903 / 192868 | - | descriptive | - |
| higs_tilesamp_phase_27k_r05_polish | final_gaussian_count (ctrl / cand mean) | 2400650 / 2304570 | - | descriptive | - |

| higs_tilesamp_phase_27k_r06 | psnr_db | -0.5349 | [-1.1663, -0.1308] | >= -0.10 | False |
| higs_tilesamp_phase_27k_r06 | ssim | -0.0066 | [-0.0118, -0.0026] | >= -0.003 | False |
| higs_tilesamp_phase_27k_r06 | lpips | 0.0110 | [0.0042, 0.0213] | <= +0.005 | False |
| higs_tilesamp_phase_27k_r06 | time_to_quality_seconds | -120.2058 | [-192.1576, -57.0786] | <= 0 | True |
| higs_tilesamp_phase_27k_r06 | speedup ratio | 1.103 | CI lo 0.987 | mean>=1.111 & lo>1.0 | False |
| higs_tilesamp_phase_27k_r06 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3543 | - | descriptive | - |
| higs_tilesamp_phase_27k_r06 | energy_joules (ctrl / cand mean) | 222903 / 190699 | - | descriptive | - |
| higs_tilesamp_phase_27k_r06 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2293924 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.132 | 623 | 550 |
| gsplat_27k | deep_blending/playroom | 1.116 | 489 | 438 |
| gsplat_27k | mipnerf360/bicycle | 1.127 | 1737 | 1542 |
| gsplat_27k | mipnerf360/bonsai | 1.100 | 729 | 662 |
| gsplat_27k | mipnerf360/counter | 1.090 | 740 | 678 |
| gsplat_27k | mipnerf360/garden | 1.129 | 1748 | 1549 |
| gsplat_27k | mipnerf360/kitchen | 1.114 | 801 | 719 |
| gsplat_27k | mipnerf360/room | 1.158 | 766 | 662 |
| gsplat_27k | mipnerf360/stump | 1.123 | 1574 | 1402 |
| gsplat_27k | tanks_and_temples/train | 1.092 | 516 | 473 |
| gsplat_27k | tanks_and_temples/truck | 1.184 | 498 | 420 |
| higs_tilesamp_phase_27k_r04 | deep_blending/drjohnson | 1.101 | 623 | 565 |
| higs_tilesamp_phase_27k_r04 | deep_blending/playroom | 1.087 | 489 | 450 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/bicycle | 1.160 | 1737 | 1497 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/bonsai | 1.081 | 729 | 674 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/counter | 1.067 | 740 | 694 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/garden | 1.155 | 1748 | 1514 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/kitchen | 1.088 | 801 | 736 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/room | 1.153 | 766 | 664 |
| higs_tilesamp_phase_27k_r04 | mipnerf360/stump | 1.149 | 1574 | 1370 |
| higs_tilesamp_phase_27k_r04 | tanks_and_temples/train | 0.986 | 516 | 524 |
| higs_tilesamp_phase_27k_r04 | tanks_and_temples/truck | 1.067 | 498 | 467 |
| higs_tilesamp_phase_27k_r05 | deep_blending/drjohnson | 1.104 | 623 | 564 |
| higs_tilesamp_phase_27k_r05 | deep_blending/playroom | 1.092 | 489 | 448 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/bicycle | 1.157 | 1737 | 1501 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/bonsai | 1.093 | 729 | 667 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/counter | 1.085 | 740 | 682 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/garden | 1.157 | 1748 | 1511 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/kitchen | 1.092 | 801 | 733 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/room | 1.151 | 766 | 665 |
| higs_tilesamp_phase_27k_r05 | mipnerf360/stump | 1.145 | 1574 | 1375 |
| higs_tilesamp_phase_27k_r05 | tanks_and_temples/train | 0.988 | 516 | 523 |
| higs_tilesamp_phase_27k_r05 | tanks_and_temples/truck | 1.075 | 498 | 463 |
| higs_tilesamp_phase_27k_r05_polish | deep_blending/drjohnson | 1.106 | 623 | 563 |
| higs_tilesamp_phase_27k_r05_polish | deep_blending/playroom | 1.074 | 489 | 455 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/bicycle | 1.153 | 1737 | 1506 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/bonsai | 1.089 | 729 | 669 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/counter | 1.088 | 740 | 680 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/garden | 1.153 | 1748 | 1517 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/kitchen | 1.085 | 801 | 738 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/room | 1.154 | 766 | 664 |
| higs_tilesamp_phase_27k_r05_polish | mipnerf360/stump | 1.148 | 1574 | 1371 |
| higs_tilesamp_phase_27k_r05_polish | tanks_and_temples/train | 0.974 | 516 | 530 |
| higs_tilesamp_phase_27k_r05_polish | tanks_and_temples/truck | 1.078 | 498 | 462 |
| higs_tilesamp_phase_27k_r06 | deep_blending/drjohnson | 1.105 | 623 | 563 |
| higs_tilesamp_phase_27k_r06 | deep_blending/playroom | 1.087 | 489 | 450 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/bicycle | 1.155 | 1737 | 1504 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/bonsai | 1.086 | 729 | 671 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/counter | 1.082 | 740 | 684 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/garden | 1.152 | 1748 | 1517 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/kitchen | 1.087 | 801 | 736 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/room | 1.159 | 766 | 661 |
| higs_tilesamp_phase_27k_r06 | mipnerf360/stump | 1.151 | 1574 | 1367 |
| higs_tilesamp_phase_27k_r06 | tanks_and_temples/train | 0.987 | 516 | 523 |
| higs_tilesamp_phase_27k_r06 | tanks_and_temples/truck | 1.082 | 498 | 460 |