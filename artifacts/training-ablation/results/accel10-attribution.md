# HiGS accel10 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_ssim05 | psnr_db | 0.0899 | [-0.1460, 0.4552] | >= -0.10 | False |
| gsplat_30k_ssim05 | ssim | -0.0065 | [-0.0104, -0.0018] | >= -0.003 | False |
| gsplat_30k_ssim05 | lpips | -0.0017 | [-0.0112, 0.0048] | <= +0.005 | True |
| gsplat_30k_ssim05 | time_to_quality_seconds | -141.6834 | [-297.3440, 10.3128] | <= 0 | False |
| gsplat_30k_ssim05 | speedup ratio | 1.271 | CI lo 1.027 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim05 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3007 | - | descriptive | - |
| gsplat_30k_ssim05 | energy_joules (ctrl / cand mean) | 197960 / 141596 | - | descriptive | - |
| gsplat_30k_ssim05 | final_gaussian_count (ctrl / cand mean) | 2323263 / 1947540 | - | descriptive | - |

| higs_skipbwd_27k | psnr_db | -0.1505 | [-0.3504, 0.0706] | >= -0.10 | False |
| higs_skipbwd_27k | ssim | -0.0016 | [-0.0055, 0.0023] | >= -0.003 | False |
| higs_skipbwd_27k | lpips | 0.0001 | [-0.0065, 0.0048] | <= +0.005 | True |
| higs_skipbwd_27k | time_to_quality_seconds | -86.1573 | [-239.5869, 32.7725] | <= 0 | False |
| higs_skipbwd_27k | speedup ratio | 1.096 | CI lo 0.972 | mean>=1.111 & lo>1.0 | False |
| higs_skipbwd_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3572 | - | descriptive | - |
| higs_skipbwd_27k | energy_joules (ctrl / cand mean) | 197960 / 172214 | - | descriptive | - |
| higs_skipbwd_27k | final_gaussian_count (ctrl / cand mean) | 2323263 / 2305200 | - | descriptive | - |

| higs_skipbwd_30k | psnr_db | -0.0689 | [-0.2491, 0.1736] | >= -0.10 | False |
| higs_skipbwd_30k | ssim | -0.0004 | [-0.0032, 0.0032] | >= -0.003 | False |
| higs_skipbwd_30k | lpips | -0.0016 | [-0.0069, 0.0016] | <= +0.005 | True |
| higs_skipbwd_30k | time_to_quality_seconds | -36.7377 | [-182.7152, 71.5554] | <= 0 | False |
| higs_skipbwd_30k | speedup ratio | 0.994 | CI lo 0.872 | mean>=1.111 & lo>1.0 | False |
| higs_skipbwd_30k | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3694 | - | descriptive | - |
| higs_skipbwd_30k | energy_joules (ctrl / cand mean) | 197960 / 193309 | - | descriptive | - |
| higs_skipbwd_30k | final_gaussian_count (ctrl / cand mean) | 2323263 / 2388409 | - | descriptive | - |

| higs_skipbwd_30k_agg | psnr_db | -0.0397 | [-0.1452, 0.0960] | >= -0.10 | False |
| higs_skipbwd_30k_agg | ssim | -0.0004 | [-0.0019, 0.0012] | >= -0.003 | True |
| higs_skipbwd_30k_agg | lpips | -0.0011 | [-0.0036, 0.0005] | <= +0.005 | True |
| higs_skipbwd_30k_agg | time_to_quality_seconds | 20.8132 | [-45.0365, 72.5123] | <= 0 | False |
| higs_skipbwd_30k_agg | speedup ratio | 0.941 | CI lo 0.833 | mean>=1.111 & lo>1.0 | False |
| higs_skipbwd_30k_agg | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3709 | - | descriptive | - |
| higs_skipbwd_30k_agg | energy_joules (ctrl / cand mean) | 197960 / 204783 | - | descriptive | - |
| higs_skipbwd_30k_agg | final_gaussian_count (ctrl / cand mean) | 2323263 / 2398217 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_ssim05 | deep_blending/drjohnson | 1.290 | 549 | 426 |
| gsplat_30k_ssim05 | deep_blending/playroom | 1.343 | 436 | 324 |
| gsplat_30k_ssim05 | mipnerf360/bicycle | 1.616 | 1538 | 952 |
| gsplat_30k_ssim05 | mipnerf360/bonsai | 1.078 | 659 | 611 |
| gsplat_30k_ssim05 | mipnerf360/counter | 1.036 | 666 | 643 |
| gsplat_30k_ssim05 | mipnerf360/garden | 1.798 | 1546 | 860 |
| gsplat_30k_ssim05 | mipnerf360/kitchen | 1.027 | 716 | 697 |
| gsplat_30k_ssim05 | mipnerf360/room | 1.073 | 665 | 620 |
| gsplat_30k_ssim05 | mipnerf360/stump | 1.538 | 1517 | 986 |
| gsplat_30k_ssim05 | tanks_and_temples/train | 1.132 | 539 | 476 |
| gsplat_30k_ssim05 | tanks_and_temples/truck | 1.045 | 420 | 402 |
| higs_skipbwd_27k | deep_blending/drjohnson | 1.047 | 549 | 525 |
| higs_skipbwd_27k | deep_blending/playroom | 1.054 | 436 | 413 |
| higs_skipbwd_27k | mipnerf360/bicycle | 1.175 | 1538 | 1309 |
| higs_skipbwd_27k | mipnerf360/bonsai | 1.162 | 659 | 567 |
| higs_skipbwd_27k | mipnerf360/counter | 1.079 | 666 | 618 |
| higs_skipbwd_27k | mipnerf360/garden | 1.136 | 1546 | 1361 |
| higs_skipbwd_27k | mipnerf360/kitchen | 1.121 | 716 | 639 |
| higs_skipbwd_27k | mipnerf360/room | 1.078 | 665 | 617 |
| higs_skipbwd_27k | mipnerf360/stump | 1.107 | 1517 | 1371 |
| higs_skipbwd_27k | tanks_and_temples/train | 1.127 | 539 | 478 |
| higs_skipbwd_27k | tanks_and_temples/truck | 0.972 | 420 | 432 |
| higs_skipbwd_30k | deep_blending/drjohnson | 0.942 | 549 | 583 |
| higs_skipbwd_30k | deep_blending/playroom | 0.948 | 436 | 460 |
| higs_skipbwd_30k | mipnerf360/bicycle | 1.063 | 1538 | 1448 |
| higs_skipbwd_30k | mipnerf360/bonsai | 1.086 | 659 | 607 |
| higs_skipbwd_30k | mipnerf360/counter | 0.981 | 666 | 679 |
| higs_skipbwd_30k | mipnerf360/garden | 1.023 | 1546 | 1511 |
| higs_skipbwd_30k | mipnerf360/kitchen | 1.017 | 716 | 704 |
| higs_skipbwd_30k | mipnerf360/room | 0.961 | 665 | 692 |
| higs_skipbwd_30k | mipnerf360/stump | 1.022 | 1517 | 1484 |
| higs_skipbwd_30k | tanks_and_temples/train | 1.016 | 539 | 530 |
| higs_skipbwd_30k | tanks_and_temples/truck | 0.872 | 420 | 481 |
| higs_skipbwd_30k_agg | deep_blending/drjohnson | 0.917 | 549 | 599 |
| higs_skipbwd_30k_agg | deep_blending/playroom | 0.923 | 436 | 472 |
| higs_skipbwd_30k_agg | mipnerf360/bicycle | 1.003 | 1538 | 1534 |
| higs_skipbwd_30k_agg | mipnerf360/bonsai | 1.013 | 659 | 651 |
| higs_skipbwd_30k_agg | mipnerf360/counter | 0.928 | 666 | 718 |
| higs_skipbwd_30k_agg | mipnerf360/garden | 0.954 | 1546 | 1620 |
| higs_skipbwd_30k_agg | mipnerf360/kitchen | 0.955 | 716 | 749 |
| higs_skipbwd_30k_agg | mipnerf360/room | 0.852 | 665 | 781 |
| higs_skipbwd_30k_agg | mipnerf360/stump | 0.990 | 1517 | 1532 |
| higs_skipbwd_30k_agg | tanks_and_temples/train | 0.985 | 539 | 547 |
| higs_skipbwd_30k_agg | tanks_and_temples/truck | 0.833 | 420 | 504 |