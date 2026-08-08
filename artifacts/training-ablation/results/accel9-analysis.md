# HiGS Accel Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0125 | [-0.0909, 0.0725] | >= -0.10 | True |
| gsplat_27k | ssim | -0.0000 | [-0.0013, 0.0014] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0007 | [-0.0014, 0.0029] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | -87.3144 | [-128.7330, -50.3155] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.128 | CI lo 1.085 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3607 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 195894 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2334098 | - | descriptive | - |

| gsplat_30k_ssim05 | psnr_db | 0.0290 | [-0.1402, 0.2238] | >= -0.10 | False |
| gsplat_30k_ssim05 | ssim | -0.0075 | [-0.0108, -0.0048] | >= -0.003 | False |
| gsplat_30k_ssim05 | lpips | 0.0005 | [-0.0052, 0.0048] | <= +0.005 | True |
| gsplat_30k_ssim05 | time_to_quality_seconds | -242.7404 | [-413.3808, -100.6556] | <= 0 | True |
| gsplat_30k_ssim05 | speedup ratio | 1.425 | CI lo 1.081 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 2980 | - | descriptive | - |
| gsplat_30k_ssim05 | energy_joules (ctrl / cand mean) | 222903 / 143108 | - | descriptive | - |
| gsplat_30k_ssim05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1925745 | - | descriptive | - |

| gsplat_30k_ssim05_ev2 | psnr_db | -0.0964 | [-0.3084, 0.1421] | >= -0.10 | False |
| gsplat_30k_ssim05_ev2 | ssim | -0.0156 | [-0.0246, -0.0080] | >= -0.003 | False |
| gsplat_30k_ssim05_ev2 | lpips | 0.0160 | [0.0032, 0.0308] | <= +0.005 | False |
| gsplat_30k_ssim05_ev2 | time_to_quality_seconds | -389.4705 | [-596.5193, -221.9158] | <= 0 | True |
| gsplat_30k_ssim05_ev2 | speedup ratio | 1.982 | CI lo 1.316 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim05_ev2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 2062 | - | descriptive | - |
| gsplat_30k_ssim05_ev2 | energy_joules (ctrl / cand mean) | 222903 / 95377 | - | descriptive | - |
| gsplat_30k_ssim05_ev2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1320141 | - | descriptive | - |

| higs_ssim_27k | psnr_db | -0.2817 | [-0.5377, -0.0007] | >= -0.10 | False |
| higs_ssim_27k | ssim | -0.0196 | [-0.0296, -0.0105] | >= -0.003 | False |
| higs_ssim_27k | lpips | 0.0199 | [0.0060, 0.0361] | <= +0.005 | False |
| higs_ssim_27k | time_to_quality_seconds | -397.4403 | [-616.6216, -212.2651] | <= 0 | True |
| higs_ssim_27k | speedup ratio | 2.205 | CI lo 1.457 | mean>=1.111 & lo>1.0 | True |
| higs_ssim_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 2943 | - | descriptive | - |
| higs_ssim_27k | energy_joules (ctrl / cand mean) | 222903 / 83701 | - | descriptive | - |
| higs_ssim_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 1285198 | - | descriptive | - |

| higs_ssim_30k | psnr_db | -0.1735 | [-0.4362, 0.1165] | >= -0.10 | False |
| higs_ssim_30k | ssim | -0.0169 | [-0.0260, -0.0088] | >= -0.003 | False |
| higs_ssim_30k | lpips | 0.0165 | [0.0039, 0.0310] | <= +0.005 | False |
| higs_ssim_30k | time_to_quality_seconds | -357.6417 | [-570.7937, -179.0284] | <= 0 | True |
| higs_ssim_30k | speedup ratio | 1.990 | CI lo 1.301 | mean>=1.111 & lo>1.0 | True |
| higs_ssim_30k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 2998 | - | descriptive | - |
| higs_ssim_30k | energy_joules (ctrl / cand mean) | 222903 / 94395 | - | descriptive | - |
| higs_ssim_30k | final_gaussian_count (ctrl / cand mean) | 2400650 / 1323141 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.134 | 623 | 549 |
| gsplat_27k | deep_blending/playroom | 1.129 | 489 | 433 |
| gsplat_27k | mipnerf360/bicycle | 1.133 | 1737 | 1533 |
| gsplat_27k | mipnerf360/bonsai | 1.104 | 729 | 660 |
| gsplat_27k | mipnerf360/counter | 1.106 | 740 | 669 |
| gsplat_27k | mipnerf360/garden | 1.126 | 1748 | 1553 |
| gsplat_27k | mipnerf360/kitchen | 1.111 | 801 | 721 |
| gsplat_27k | mipnerf360/room | 1.162 | 766 | 659 |
| gsplat_27k | mipnerf360/stump | 1.119 | 1574 | 1407 |
| gsplat_27k | tanks_and_temples/train | 1.085 | 516 | 476 |
| gsplat_27k | tanks_and_temples/truck | 1.195 | 498 | 416 |
| gsplat_30k_ssim05 | deep_blending/drjohnson | 1.391 | 623 | 448 |
| gsplat_30k_ssim05 | deep_blending/playroom | 1.481 | 489 | 330 |
| gsplat_30k_ssim05 | mipnerf360/bicycle | 1.825 | 1737 | 951 |
| gsplat_30k_ssim05 | mipnerf360/bonsai | 1.176 | 729 | 619 |
| gsplat_30k_ssim05 | mipnerf360/counter | 1.147 | 740 | 645 |
| gsplat_30k_ssim05 | mipnerf360/garden | 2.036 | 1748 | 859 |
| gsplat_30k_ssim05 | mipnerf360/kitchen | 1.148 | 801 | 698 |
| gsplat_30k_ssim05 | mipnerf360/room | 1.251 | 766 | 612 |
| gsplat_30k_ssim05 | mipnerf360/stump | 1.899 | 1574 | 829 |
| gsplat_30k_ssim05 | tanks_and_temples/train | 1.081 | 516 | 477 |
| gsplat_30k_ssim05 | tanks_and_temples/truck | 1.239 | 498 | 402 |
| gsplat_30k_ssim05_ev2 | deep_blending/drjohnson | 1.800 | 623 | 346 |
| gsplat_30k_ssim05_ev2 | deep_blending/playroom | 1.844 | 489 | 265 |
| gsplat_30k_ssim05_ev2 | mipnerf360/bicycle | 2.721 | 1737 | 638 |
| gsplat_30k_ssim05_ev2 | mipnerf360/bonsai | 1.744 | 729 | 418 |
| gsplat_30k_ssim05_ev2 | mipnerf360/counter | 1.688 | 740 | 438 |
| gsplat_30k_ssim05_ev2 | mipnerf360/garden | 2.988 | 1748 | 585 |
| gsplat_30k_ssim05_ev2 | mipnerf360/kitchen | 1.694 | 801 | 473 |
| gsplat_30k_ssim05_ev2 | mipnerf360/room | 1.822 | 766 | 421 |
| gsplat_30k_ssim05_ev2 | mipnerf360/stump | 2.730 | 1574 | 577 |
| gsplat_30k_ssim05_ev2 | tanks_and_temples/train | 1.316 | 516 | 392 |
| gsplat_30k_ssim05_ev2 | tanks_and_temples/truck | 1.459 | 498 | 341 |
| higs_ssim_27k | deep_blending/drjohnson | 2.007 | 623 | 310 |
| higs_ssim_27k | deep_blending/playroom | 2.025 | 489 | 241 |
| higs_ssim_27k | mipnerf360/bicycle | 3.059 | 1737 | 568 |
| higs_ssim_27k | mipnerf360/bonsai | 1.876 | 729 | 388 |
| higs_ssim_27k | mipnerf360/counter | 1.891 | 740 | 391 |
| higs_ssim_27k | mipnerf360/garden | 3.369 | 1748 | 519 |
| higs_ssim_27k | mipnerf360/kitchen | 1.871 | 801 | 428 |
| higs_ssim_27k | mipnerf360/room | 2.005 | 766 | 382 |
| higs_ssim_27k | mipnerf360/stump | 3.084 | 1574 | 510 |
| higs_ssim_27k | tanks_and_temples/train | 1.457 | 516 | 354 |
| higs_ssim_27k | tanks_and_temples/truck | 1.607 | 498 | 310 |
| higs_ssim_30k | deep_blending/drjohnson | 1.825 | 623 | 341 |
| higs_ssim_30k | deep_blending/playroom | 1.761 | 489 | 278 |
| higs_ssim_30k | mipnerf360/bicycle | 2.748 | 1737 | 632 |
| higs_ssim_30k | mipnerf360/bonsai | 1.751 | 729 | 416 |
| higs_ssim_30k | mipnerf360/counter | 1.713 | 740 | 432 |
| higs_ssim_30k | mipnerf360/garden | 3.013 | 1748 | 580 |
| higs_ssim_30k | mipnerf360/kitchen | 1.698 | 801 | 471 |
| higs_ssim_30k | mipnerf360/room | 1.829 | 766 | 419 |
| higs_ssim_30k | mipnerf360/stump | 2.780 | 1574 | 566 |
| higs_ssim_30k | tanks_and_temples/train | 1.301 | 516 | 397 |
| higs_ssim_30k | tanks_and_temples/truck | 1.470 | 498 | 339 |
## Cross-matrix non-determinism note (train scene)

The in-matrix `gsplat_27k` control PASSES all pre-registered gates in
accel9 (PSNR CI lo -0.091, SSIM CI lo -0.0013, LPIPS CI hi 0.0029,
speedup 1.128 CI lo 1.085) but FAILED the same gates in accel8
(PSNR CI lo -0.30). cfg.yml, source hashes, dataset inventory sha256,
and seed are identical between the two matrices; only result_dir differs.
Per-scene comparison shows a single scene, tanks_and_temples/train,
drives the discrepancy (gsplat_27k seed0 PSNR: accel8 20.947 vs accel9
21.824, +0.878 dB), with the scene itself bimodal across all exploration
matrices (20.59-21.82 dB). This is training-run non-determinism, not a
config difference. Conclusion: single-seed exploration conclusions are
not reliable for near-threshold candidates; only the 3-seed formal matrix
is authoritative. Also note gsplat_27k is ordinary early-stop and cannot
be attributed to any HiGS mechanism.

## accel9 substantive conclusion

All SSIM-amortization variants (0.5x SSIM and/or every-2 SSIM) achieve
large wall-clock speedups (1.4-2.2x, CI lower > 1.0) but violate the
quality gates: gsplat_30k_ssim05 SSIM -0.0075 (CI lo -0.0108),
gsplat_30k_ssim05_ev2 SSIM -0.0156 / LPIPS +0.016, higs_ssim_30k
SSIM -0.0169 / LPIPS +0.0165, higs_ssim_27k SSIM -0.0196 / LPIPS +0.0199.
The every-2 SSIM schedule is the main quality driver (sparser SSIM
signal degrades SSIM/LPIPS metrics); the 0.5x SSIM alone keeps
PSNR/LPIPS within margins and misses SSIM by only 0.0045 above the
-0.003 margin. No accel9 candidate qualifies; no candidate is frozen.
