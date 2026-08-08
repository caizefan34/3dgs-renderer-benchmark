# HiGS Topology Exploration (paired vs gsplat)

- baseline: `gsplat`; configs: gsplat_27k, higs_topology_27k, higs_topology_30k, higs_visible_27k
- exploration jobs: 44; control (gsplat) cells: 33

## Paired NI + acceleration gates (scene-block bootstrap 95% CI)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0125 | [-0.0917, 0.0729] | >= -0.10 | True |
| gsplat_27k | ssim | -0.0005 | [-0.0017, 0.0007] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0018 | [0.0001, 0.0039] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | -99.7929 | [-145.1330, -58.5941] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.120 | CI lo 1.067 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | TTQ delta (s) | -99.79 | [-145.13, -58.59] | <= 0 | True |
| gsplat_27k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3596 | - | descriptive | - |
| gsplat_27k | energy_joules (gsplat / cand mean) | 222903 / 198987 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2327127 | - | descriptive | - |

| higs_topology_27k | psnr_db | -0.5704 | [-1.4621, -0.0340] | >= -0.10 | False |
| higs_topology_27k | ssim | -0.0120 | [-0.0300, -0.0011] | >= -0.003 | False |
| higs_topology_27k | lpips | 0.0102 | [0.0042, 0.0172] | <= +0.005 | False |
| higs_topology_27k | time_to_quality_seconds | -102.1245 | [-179.0587, -29.9256] | <= 0 | True |
| higs_topology_27k | speedup ratio | 1.107 | CI lo 0.893 | mean>=1.111 & lo>1.0 | False |
| higs_topology_27k | TTQ delta (s) | -102.12 | [-179.06, -29.93] | <= 0 | True |
| higs_topology_27k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3315 | - | descriptive | - |
| higs_topology_27k | energy_joules (gsplat / cand mean) | 222903 / 182633 | - | descriptive | - |
| higs_topology_27k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2141025 | - | descriptive | - |

| higs_topology_30k | psnr_db | -0.4234 | [-0.9247, -0.0302] | >= -0.10 | False |
| higs_topology_30k | ssim | -0.0149 | [-0.0396, -0.0004] | >= -0.003 | False |
| higs_topology_30k | lpips | 0.0183 | [0.0010, 0.0490] | <= +0.005 | False |
| higs_topology_30k | time_to_quality_seconds | -99.6483 | [-281.6982, 26.4094] | <= 0 | False |
| higs_topology_30k | speedup ratio | 1.171 | CI lo 0.796 | mean>=1.111 & lo>1.0 | False |
| higs_topology_30k | TTQ delta (s) | -99.65 | [-281.70, 26.41] | <= 0 | False |
| higs_topology_30k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 2877 | - | descriptive | - |
| higs_topology_30k | energy_joules (gsplat / cand mean) | 222903 / 178429 | - | descriptive | - |
| higs_topology_30k | final_gaussian_count (gsplat / cand mean) | 2400650 / 1746465 | - | descriptive | - |

| higs_visible_27k | psnr_db | -0.1322 | [-0.2748, 0.0216] | >= -0.10 | False |
| higs_visible_27k | ssim | -0.0032 | [-0.0063, -0.0001] | >= -0.003 | False |
| higs_visible_27k | lpips | 0.0061 | [0.0022, 0.0115] | <= +0.005 | False |
| higs_visible_27k | time_to_quality_seconds | -93.3185 | [-157.5266, -37.6529] | <= 0 | True |
| higs_visible_27k | speedup ratio | 1.116 | CI lo 0.901 | mean>=1.111 & lo>1.0 | False |
| higs_visible_27k | TTQ delta (s) | -93.32 | [-157.53, -37.65] | <= 0 | True |
| higs_visible_27k | peak_gpu_memory_mib (gsplat / cand mean) | 3705 / 3337 | - | descriptive | - |
| higs_visible_27k | energy_joules (gsplat / cand mean) | 222903 / 183923 | - | descriptive | - |
| higs_visible_27k | final_gaussian_count (gsplat / cand mean) | 2400650 / 2187516 | - | descriptive | - |

## Per-scene speedup ratio (gsplat wall / candidate wall, seed 0)

| config | scene | speedup | gsplat s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.133 | 623 | 550 |
| gsplat_27k | deep_blending/playroom | 1.115 | 489 | 438 |
| gsplat_27k | mipnerf360/bicycle | 1.130 | 1737 | 1537 |
| gsplat_27k | mipnerf360/bonsai | 1.108 | 729 | 657 |
| gsplat_27k | mipnerf360/counter | 1.101 | 740 | 672 |
| gsplat_27k | mipnerf360/garden | 1.127 | 1748 | 1551 |
| gsplat_27k | mipnerf360/kitchen | 1.101 | 801 | 727 |
| gsplat_27k | mipnerf360/room | 1.153 | 766 | 664 |
| gsplat_27k | mipnerf360/stump | 1.112 | 1574 | 1416 |
| gsplat_27k | tanks_and_temples/train | 1.067 | 516 | 484 |
| gsplat_27k | tanks_and_temples/truck | 1.177 | 498 | 423 |
| higs_topology_27k | deep_blending/drjohnson | 1.143 | 623 | 545 |
| higs_topology_27k | deep_blending/playroom | 1.126 | 489 | 434 |
| higs_topology_27k | mipnerf360/bicycle | 1.235 | 1737 | 1406 |
| higs_topology_27k | mipnerf360/bonsai | 1.099 | 729 | 663 |
| higs_topology_27k | mipnerf360/counter | 1.113 | 740 | 664 |
| higs_topology_27k | mipnerf360/garden | 1.201 | 1748 | 1455 |
| higs_topology_27k | mipnerf360/kitchen | 1.055 | 801 | 759 |
| higs_topology_27k | mipnerf360/room | 1.166 | 766 | 657 |
| higs_topology_27k | mipnerf360/stump | 1.190 | 1574 | 1323 |
| higs_topology_27k | tanks_and_temples/train | 0.893 | 516 | 578 |
| higs_topology_27k | tanks_and_temples/truck | 0.955 | 498 | 521 |
| higs_topology_30k | deep_blending/drjohnson | 1.019 | 623 | 611 |
| higs_topology_30k | deep_blending/playroom | 1.015 | 489 | 482 |
| higs_topology_30k | mipnerf360/bicycle | 2.177 | 1737 | 798 |
| higs_topology_30k | mipnerf360/bonsai | 0.994 | 729 | 733 |
| higs_topology_30k | mipnerf360/counter | 1.000 | 740 | 740 |
| higs_topology_30k | mipnerf360/garden | 1.066 | 1748 | 1640 |
| higs_topology_30k | mipnerf360/kitchen | 0.946 | 801 | 847 |
| higs_topology_30k | mipnerf360/room | 1.857 | 766 | 413 |
| higs_topology_30k | mipnerf360/stump | 1.068 | 1574 | 1475 |
| higs_topology_30k | tanks_and_temples/train | 0.796 | 516 | 648 |
| higs_topology_30k | tanks_and_temples/truck | 0.950 | 498 | 524 |
| higs_visible_27k | deep_blending/drjohnson | 1.140 | 623 | 546 |
| higs_visible_27k | deep_blending/playroom | 1.127 | 489 | 434 |
| higs_visible_27k | mipnerf360/bicycle | 1.234 | 1737 | 1408 |
| higs_visible_27k | mipnerf360/bonsai | 1.101 | 729 | 662 |
| higs_visible_27k | mipnerf360/counter | 1.102 | 740 | 671 |
| higs_visible_27k | mipnerf360/garden | 1.197 | 1748 | 1461 |
| higs_visible_27k | mipnerf360/kitchen | 1.060 | 801 | 755 |
| higs_visible_27k | mipnerf360/room | 1.173 | 766 | 653 |
| higs_visible_27k | mipnerf360/stump | 1.188 | 1574 | 1325 |
| higs_visible_27k | tanks_and_temples/train | 0.901 | 516 | 573 |
| higs_visible_27k | tanks_and_temples/truck | 1.049 | 498 | 475 |