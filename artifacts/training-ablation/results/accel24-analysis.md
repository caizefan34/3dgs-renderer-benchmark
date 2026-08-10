# HiGS accel24 Exploration (paired vs in-matrix gsplat 30k) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat | psnr_db | 0.0000 | [0.0000, 0.0000] | >= -0.1 | True |
| gsplat | ssim | 0.0000 | [0.0000, 0.0000] | >= -0.003 | True |
| gsplat | lpips | 0.0000 | [0.0000, 0.0000] | <= 0.005 | True |
| gsplat | time_to_quality_seconds | 0.0000 | [0.0000, 0.0000] | <= 0.0 | True |
| gsplat | speedup ratio | 1.000 | CI lo 1.000 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat | peak_gpu_memory_mib (ctrl / cand mean) | 3700 / 3700 | - | descriptive | - |
| gsplat | energy_joules (ctrl / cand mean) | 119803 / 119803 | - | descriptive | - |
| gsplat | final_gaussian_count (ctrl / cand mean) | 2394158 / 2394158 | - | descriptive | - |
| gsplat_30k_fused | psnr_db | 0.0205 | [-0.0017, 0.0460] | >= -0.1 | True |
| gsplat_30k_fused | ssim | 0.0001 | [-0.0003, 0.0006] | >= -0.003 | True |
| gsplat_30k_fused | lpips | -0.0004 | [-0.0010, 0.0002] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | -0.2960 | [-19.5339, 17.6143] | <= 0.0 | False |
| gsplat_30k_fused | speedup ratio | 1.021 | CI lo 0.883 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3700 / 3708 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 119803 / 109179 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2394158 / 1695712 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | 0.0271 | [-0.0885, 0.1738] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0005 | [-0.0015, 0.0006] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0011 | [-0.0004, 0.0025] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -57.7623 | [-92.6315, -26.5166] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 1.190 | CI lo 1.069 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3700 / 3695 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 119803 / 91145 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2394158 / 1036560 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat | deep_blending/drjohnson | 1.000 | 520 | 520 |
| gsplat | deep_blending/playroom | 1.000 | 410 | 410 |
| gsplat | mipnerf360/bicycle | 1.000 | 925 | 925 |
| gsplat | mipnerf360/bonsai | 1.000 | 523 | 523 |
| gsplat | mipnerf360/counter | 1.000 | 365 | 365 |
| gsplat | mipnerf360/garden | 1.000 | 790 | 790 |
| gsplat | mipnerf360/kitchen | 1.000 | 417 | 417 |
| gsplat | mipnerf360/room | 1.000 | 344 | 344 |
| gsplat | mipnerf360/stump | 1.000 | 666 | 666 |
| gsplat | tanks_and_temples/train | 1.000 | 427 | 427 |
| gsplat | tanks_and_temples/truck | 1.000 | 375 | 375 |
| gsplat_30k_fused | deep_blending/drjohnson | 1.039 | 520 | 500 |
| gsplat_30k_fused | deep_blending/playroom | 1.216 | 410 | 338 |
| gsplat_30k_fused | mipnerf360/bicycle | 1.089 | 925 | 849 |
| gsplat_30k_fused | mipnerf360/bonsai | 1.001 | 523 | 522 |
| gsplat_30k_fused | mipnerf360/counter | 0.985 | 365 | 370 |
| gsplat_30k_fused | mipnerf360/garden | 1.069 | 790 | 739 |
| gsplat_30k_fused | mipnerf360/kitchen | 0.958 | 417 | 435 |
| gsplat_30k_fused | mipnerf360/room | 0.998 | 344 | 345 |
| gsplat_30k_fused | mipnerf360/stump | 1.061 | 666 | 628 |
| gsplat_30k_fused | tanks_and_temples/train | 0.883 | 427 | 484 |
| gsplat_30k_fused | tanks_and_temples/truck | 0.934 | 375 | 401 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.267 | 520 | 410 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 1.264 | 410 | 325 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 1.138 | 925 | 813 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 1.524 | 523 | 343 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 1.120 | 365 | 325 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 1.236 | 790 | 639 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 1.069 | 417 | 390 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 1.069 | 344 | 322 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 1.205 | 666 | 553 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 1.070 | 427 | 399 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.126 | 375 | 333 |