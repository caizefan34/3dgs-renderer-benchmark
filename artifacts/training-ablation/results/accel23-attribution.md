# HiGS accel23 Attribution (paired vs in-matrix gsplat) (paired vs gsplat (in-matrix))

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_fused | psnr_db | 0.0989 | [-0.0055, 0.2150] | >= -0.1 | True |
| gsplat_30k_fused | ssim | 0.0013 | [0.0002, 0.0029] | >= -0.003 | True |
| gsplat_30k_fused | lpips | -0.0008 | [-0.0023, 0.0004] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | 52.9988 | [1.2358, 106.0131] | <= 0.0 | False |
| gsplat_30k_fused | speedup ratio | 0.941 | CI lo 0.778 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3729 / 3679 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 144933 / 159066 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2412946 / 1681025 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | 0.0179 | [-0.1701, 0.1872] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0007 | [-0.0042, 0.0022] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0022 | [-0.0007, 0.0058] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -1.7218 | [-100.5317, 110.2519] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 1.076 | CI lo 0.646 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3729 / 3701 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 144933 / 133678 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2412946 / 1039684 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16 | psnr_db | -0.8873 | [-1.2124, -0.5155] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_accum16 | ssim | -0.0311 | [-0.0573, -0.0090] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_accum16 | lpips | 0.0515 | [0.0183, 0.0931] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_accum16 | time_to_quality_seconds | -51.9484 | [-187.2287, 82.5820] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05_accum16 | speedup ratio | 1.334 | CI lo 0.854 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_accum16 | peak_gpu_memory_mib (ctrl / cand mean) | 3729 / 2159 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16 | energy_joules (ctrl / cand mean) | 144933 / 97674 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16 | final_gaussian_count (ctrl / cand mean) | 2412946 / 646289 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | psnr_db | -1.6712 | [-2.8202, -0.7812] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | ssim | -0.0401 | [-0.0699, -0.0212] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | lpips | 0.0373 | [0.0171, 0.0668] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | time_to_quality_seconds | 61.8404 | [-118.0574, 245.0512] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | speedup ratio | 1.081 | CI lo 0.472 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | peak_gpu_memory_mib (ctrl / cand mean) | 3729 / 4136 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | energy_joules (ctrl / cand mean) | 144933 / 138292 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | final_gaussian_count (ctrl / cand mean) | 2412946 / 1816248 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | psnr_db | -1.8677 | [-3.0208, -0.9259] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | ssim | -0.0451 | [-0.0743, -0.0247] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | lpips | 0.0426 | [0.0211, 0.0717] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | time_to_quality_seconds | 42.1929 | [-124.9766, 208.1454] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | speedup ratio | 1.068 | CI lo 0.606 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | peak_gpu_memory_mib (ctrl / cand mean) | 3729 / 4191 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | energy_joules (ctrl / cand mean) | 144933 / 136050 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | final_gaussian_count (ctrl / cand mean) | 2412946 / 1845480 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_fused | deep_blending/drjohnson | 1.037 | 519 | 501 |
| gsplat_30k_fused | deep_blending/playroom | 1.018 | 346 | 340 |
| gsplat_30k_fused | mipnerf360/bicycle | 1.115 | 987 | 886 |
| gsplat_30k_fused | mipnerf360/bonsai | 0.942 | 329 | 350 |
| gsplat_30k_fused | mipnerf360/counter | 0.991 | 360 | 364 |
| gsplat_30k_fused | mipnerf360/garden | 1.061 | 791 | 745 |
| gsplat_30k_fused | mipnerf360/kitchen | 0.871 | 414 | 475 |
| gsplat_30k_fused | mipnerf360/room | 0.778 | 556 | 716 |
| gsplat_30k_fused | mipnerf360/stump | 0.861 | 1366 | 1587 |
| gsplat_30k_fused | tanks_and_temples/train | 0.859 | 791 | 922 |
| gsplat_30k_fused | tanks_and_temples/truck | 0.821 | 694 | 846 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.276 | 519 | 407 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 0.869 | 346 | 398 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 1.578 | 987 | 626 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 1.028 | 329 | 321 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 1.080 | 360 | 334 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 1.251 | 791 | 632 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 0.777 | 414 | 533 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 1.102 | 556 | 505 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 1.187 | 1366 | 1151 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 0.646 | 791 | 1225 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.038 | 694 | 669 |
| gsplat_30k_fused_prune10_rclip05_accum16 | deep_blending/drjohnson | 1.318 | 519 | 394 |
| gsplat_30k_fused_prune10_rclip05_accum16 | deep_blending/playroom | 1.058 | 346 | 327 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/bicycle | 2.356 | 987 | 419 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/bonsai | 1.237 | 329 | 266 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/counter | 1.356 | 360 | 266 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/garden | 1.842 | 791 | 429 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/kitchen | 0.854 | 414 | 484 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/room | 1.021 | 556 | 545 |
| gsplat_30k_fused_prune10_rclip05_accum16 | mipnerf360/stump | 1.344 | 1366 | 1016 |
| gsplat_30k_fused_prune10_rclip05_accum16 | tanks_and_temples/train | 1.111 | 791 | 712 |
| gsplat_30k_fused_prune10_rclip05_accum16 | tanks_and_temples/truck | 1.173 | 694 | 592 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | deep_blending/drjohnson | 1.383 | 519 | 375 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | deep_blending/playroom | 0.782 | 346 | 442 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/bicycle | 1.693 | 987 | 583 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/bonsai | 1.080 | 329 | 305 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/counter | 1.137 | 360 | 317 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/garden | 1.398 | 791 | 566 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/kitchen | 0.472 | 414 | 876 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/room | 1.074 | 556 | 518 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | mipnerf360/stump | 1.425 | 1366 | 959 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | tanks_and_temples/train | 0.580 | 791 | 1364 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4 | tanks_and_temples/truck | 0.869 | 694 | 799 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | deep_blending/drjohnson | 1.384 | 519 | 375 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | deep_blending/playroom | 0.884 | 346 | 391 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/bicycle | 1.690 | 987 | 584 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/bonsai | 1.067 | 329 | 309 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/counter | 1.131 | 360 | 318 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/garden | 1.402 | 791 | 564 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/kitchen | 0.606 | 414 | 683 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/room | 0.786 | 556 | 708 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | mipnerf360/stump | 1.026 | 1366 | 1332 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | tanks_and_temples/train | 0.894 | 791 | 885 |
| gsplat_30k_fused_prune10_rclip05_accum16_lr4_dens600 | tanks_and_temples/truck | 0.872 | 694 | 796 |