# HiGS accel9 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_ssim05 | psnr_db | 0.0415 | [-0.0788, 0.1967] | >= -0.10 | True |
| gsplat_30k_ssim05 | ssim | -0.0075 | [-0.0100, -0.0053] | >= -0.003 | False |
| gsplat_30k_ssim05 | lpips | -0.0003 | [-0.0045, 0.0035] | <= +0.005 | True |
| gsplat_30k_ssim05 | time_to_quality_seconds | -155.4260 | [-290.6831, -34.8019] | <= 0 | True |
| gsplat_30k_ssim05 | speedup ratio | 1.264 | CI lo 0.997 | mean>=1.111 & lo>1.0 | False |
| gsplat_30k_ssim05 | peak_gpu_memory_mib (ctrl / cand mean) | 3607 / 2980 | - | descriptive | - |
| gsplat_30k_ssim05 | energy_joules (ctrl / cand mean) | 195894 / 143108 | - | descriptive | - |
| gsplat_30k_ssim05 | final_gaussian_count (ctrl / cand mean) | 2334098 / 1925745 | - | descriptive | - |

| gsplat_30k_ssim05_ev2 | psnr_db | -0.0840 | [-0.2357, 0.0978] | >= -0.10 | False |
| gsplat_30k_ssim05_ev2 | ssim | -0.0156 | [-0.0236, -0.0087] | >= -0.003 | False |
| gsplat_30k_ssim05_ev2 | lpips | 0.0153 | [0.0040, 0.0281] | <= +0.005 | False |
| gsplat_30k_ssim05_ev2 | time_to_quality_seconds | -302.1561 | [-471.3586, -162.1969] | <= 0 | True |
| gsplat_30k_ssim05_ev2 | speedup ratio | 1.759 | CI lo 1.213 | mean>=1.111 & lo>1.0 | True |
| gsplat_30k_ssim05_ev2 | peak_gpu_memory_mib (ctrl / cand mean) | 3607 / 2062 | - | descriptive | - |
| gsplat_30k_ssim05_ev2 | energy_joules (ctrl / cand mean) | 195894 / 95377 | - | descriptive | - |
| gsplat_30k_ssim05_ev2 | final_gaussian_count (ctrl / cand mean) | 2334098 / 1320141 | - | descriptive | - |

| higs_ssim_27k | psnr_db | -0.2692 | [-0.4564, -0.0631] | >= -0.10 | False |
| higs_ssim_27k | ssim | -0.0196 | [-0.0288, -0.0111] | >= -0.003 | False |
| higs_ssim_27k | lpips | 0.0191 | [0.0069, 0.0334] | <= +0.005 | False |
| higs_ssim_27k | time_to_quality_seconds | -310.1259 | [-490.7308, -157.5097] | <= 0 | True |
| higs_ssim_27k | speedup ratio | 1.956 | CI lo 1.343 | mean>=1.111 & lo>1.0 | True |
| higs_ssim_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3607 / 2943 | - | descriptive | - |
| higs_ssim_27k | energy_joules (ctrl / cand mean) | 195894 / 83701 | - | descriptive | - |
| higs_ssim_27k | final_gaussian_count (ctrl / cand mean) | 2334098 / 1285198 | - | descriptive | - |

| higs_ssim_30k | psnr_db | -0.1610 | [-0.3590, 0.0660] | >= -0.10 | False |
| higs_ssim_30k | ssim | -0.0169 | [-0.0251, -0.0095] | >= -0.003 | False |
| higs_ssim_30k | lpips | 0.0158 | [0.0049, 0.0283] | <= +0.005 | False |
| higs_ssim_30k | time_to_quality_seconds | -270.3273 | [-445.1248, -123.7209] | <= 0 | True |
| higs_ssim_30k | speedup ratio | 1.766 | CI lo 1.199 | mean>=1.111 & lo>1.0 | True |
| higs_ssim_30k | peak_gpu_memory_mib (ctrl / cand mean) | 3607 / 2998 | - | descriptive | - |
| higs_ssim_30k | energy_joules (ctrl / cand mean) | 195894 / 94395 | - | descriptive | - |
| higs_ssim_30k | final_gaussian_count (ctrl / cand mean) | 2334098 / 1323141 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_ssim05 | deep_blending/drjohnson | 1.227 | 549 | 448 |
| gsplat_30k_ssim05 | deep_blending/playroom | 1.312 | 433 | 330 |
| gsplat_30k_ssim05 | mipnerf360/bicycle | 1.611 | 1533 | 951 |
| gsplat_30k_ssim05 | mipnerf360/bonsai | 1.066 | 660 | 619 |
| gsplat_30k_ssim05 | mipnerf360/counter | 1.037 | 669 | 645 |
| gsplat_30k_ssim05 | mipnerf360/garden | 1.809 | 1553 | 859 |
| gsplat_30k_ssim05 | mipnerf360/kitchen | 1.033 | 721 | 698 |
| gsplat_30k_ssim05 | mipnerf360/room | 1.077 | 659 | 612 |
| gsplat_30k_ssim05 | mipnerf360/stump | 1.696 | 1407 | 829 |
| gsplat_30k_ssim05 | tanks_and_temples/train | 0.997 | 476 | 477 |
| gsplat_30k_ssim05 | tanks_and_temples/truck | 1.037 | 416 | 402 |
| gsplat_30k_ssim05_ev2 | deep_blending/drjohnson | 1.587 | 549 | 346 |
| gsplat_30k_ssim05_ev2 | deep_blending/playroom | 1.634 | 433 | 265 |
| gsplat_30k_ssim05_ev2 | mipnerf360/bicycle | 2.402 | 1533 | 638 |
| gsplat_30k_ssim05_ev2 | mipnerf360/bonsai | 1.580 | 660 | 418 |
| gsplat_30k_ssim05_ev2 | mipnerf360/counter | 1.526 | 669 | 438 |
| gsplat_30k_ssim05_ev2 | mipnerf360/garden | 2.655 | 1553 | 585 |
| gsplat_30k_ssim05_ev2 | mipnerf360/kitchen | 1.525 | 721 | 473 |
| gsplat_30k_ssim05_ev2 | mipnerf360/room | 1.568 | 659 | 421 |
| gsplat_30k_ssim05_ev2 | mipnerf360/stump | 2.439 | 1407 | 577 |
| gsplat_30k_ssim05_ev2 | tanks_and_temples/train | 1.213 | 476 | 392 |
| gsplat_30k_ssim05_ev2 | tanks_and_temples/truck | 1.221 | 416 | 341 |
| higs_ssim_27k | deep_blending/drjohnson | 1.770 | 549 | 310 |
| higs_ssim_27k | deep_blending/playroom | 1.794 | 433 | 241 |
| higs_ssim_27k | mipnerf360/bicycle | 2.700 | 1533 | 568 |
| higs_ssim_27k | mipnerf360/bonsai | 1.700 | 660 | 388 |
| higs_ssim_27k | mipnerf360/counter | 1.709 | 669 | 391 |
| higs_ssim_27k | mipnerf360/garden | 2.993 | 1553 | 519 |
| higs_ssim_27k | mipnerf360/kitchen | 1.684 | 721 | 428 |
| higs_ssim_27k | mipnerf360/room | 1.726 | 659 | 382 |
| higs_ssim_27k | mipnerf360/stump | 2.756 | 1407 | 510 |
| higs_ssim_27k | tanks_and_temples/train | 1.343 | 476 | 354 |
| higs_ssim_27k | tanks_and_temples/truck | 1.344 | 416 | 310 |
| higs_ssim_30k | deep_blending/drjohnson | 1.609 | 549 | 341 |
| higs_ssim_30k | deep_blending/playroom | 1.560 | 433 | 278 |
| higs_ssim_30k | mipnerf360/bicycle | 2.426 | 1533 | 632 |
| higs_ssim_30k | mipnerf360/bonsai | 1.587 | 660 | 416 |
| higs_ssim_30k | mipnerf360/counter | 1.548 | 669 | 432 |
| higs_ssim_30k | mipnerf360/garden | 2.677 | 1553 | 580 |
| higs_ssim_30k | mipnerf360/kitchen | 1.529 | 721 | 471 |
| higs_ssim_30k | mipnerf360/room | 1.575 | 659 | 419 |
| higs_ssim_30k | mipnerf360/stump | 2.484 | 1407 | 566 |
| higs_ssim_30k | tanks_and_temples/train | 1.199 | 476 | 397 |
| higs_ssim_30k | tanks_and_temples/truck | 1.230 | 416 | 339 |