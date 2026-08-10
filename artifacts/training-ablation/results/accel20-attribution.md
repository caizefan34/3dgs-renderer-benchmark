# HiGS accel20 Attribution (paired vs in-matrix gsplat) (paired vs gsplat (in-matrix))

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_fused | psnr_db | 0.0276 | [-0.0255, 0.0967] | >= -0.1 | True |
| gsplat_30k_fused | ssim | 0.0002 | [-0.0001, 0.0006] | >= -0.003 | True |
| gsplat_30k_fused | lpips | 0.0001 | [-0.0004, 0.0007] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | 3.4786 | [-22.7542, 28.5627] | <= 0.0 | False |
| gsplat_30k_fused | speedup ratio | 1.010 | CI lo 0.883 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3732 / 3689 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 121253 / 111355 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2415324 / 1683152 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | -0.0002 | [-0.0805, 0.0925] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0003 | [-0.0013, 0.0008] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0017 | [0.0005, 0.0029] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -52.3144 | [-89.9357, -20.5620] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 1.170 | CI lo 1.045 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3732 / 3676 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 121253 / 93298 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2415324 / 1031895 | - | descriptive | - |
| gsplat_30k_fused_win07 | psnr_db | 0.0255 | [-0.0537, 0.1020] | >= -0.1 | True |
| gsplat_30k_fused_win07 | ssim | -0.0002 | [-0.0006, 0.0003] | >= -0.003 | True |
| gsplat_30k_fused_win07 | lpips | 0.0010 | [0.0005, 0.0016] | <= 0.005 | True |
| gsplat_30k_fused_win07 | time_to_quality_seconds | -2.7182 | [-34.1546, 25.4375] | <= 0.0 | False |
| gsplat_30k_fused_win07 | speedup ratio | 1.019 | CI lo 0.844 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_win07 | peak_gpu_memory_mib (ctrl / cand mean) | 3732 / 3726 | - | descriptive | - |
| gsplat_30k_fused_win07 | energy_joules (ctrl / cand mean) | 121253 / 106741 | - | descriptive | - |
| gsplat_30k_fused_win07 | final_gaussian_count (ctrl / cand mean) | 2415324 / 1512115 | - | descriptive | - |
| gsplat_30k_fused_win07_prune10 | psnr_db | 0.0033 | [-0.0743, 0.0987] | >= -0.1 | True |
| gsplat_30k_fused_win07_prune10 | ssim | -0.0011 | [-0.0022, -0.0000] | >= -0.003 | True |
| gsplat_30k_fused_win07_prune10 | lpips | 0.0027 | [0.0017, 0.0038] | <= 0.005 | True |
| gsplat_30k_fused_win07_prune10 | time_to_quality_seconds | -51.9688 | [-93.6487, -17.7896] | <= 0.0 | True |
| gsplat_30k_fused_win07_prune10 | speedup ratio | 1.169 | CI lo 0.994 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_win07_prune10 | peak_gpu_memory_mib (ctrl / cand mean) | 3732 / 3694 | - | descriptive | - |
| gsplat_30k_fused_win07_prune10 | energy_joules (ctrl / cand mean) | 121253 / 90437 | - | descriptive | - |
| gsplat_30k_fused_win07_prune10 | final_gaussian_count (ctrl / cand mean) | 2415324 / 894270 | - | descriptive | - |
| gsplat_30k_fused_win07_prune10_rclip05 | psnr_db | 0.0398 | [-0.0784, 0.1905] | >= -0.1 | True |
| gsplat_30k_fused_win07_prune10_rclip05 | ssim | -0.0008 | [-0.0018, 0.0003] | >= -0.003 | True |
| gsplat_30k_fused_win07_prune10_rclip05 | lpips | 0.0026 | [0.0014, 0.0040] | <= 0.005 | True |
| gsplat_30k_fused_win07_prune10_rclip05 | time_to_quality_seconds | -18.7807 | [-86.1000, 68.2152] | <= 0.0 | False |
| gsplat_30k_fused_win07_prune10_rclip05 | speedup ratio | 1.128 | CI lo 0.549 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_win07_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3732 / 3676 | - | descriptive | - |
| gsplat_30k_fused_win07_prune10_rclip05 | energy_joules (ctrl / cand mean) | 121253 / 98116 | - | descriptive | - |
| gsplat_30k_fused_win07_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2415324 / 887736 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_fused | deep_blending/drjohnson | 1.057 | 413 | 391 |
| gsplat_30k_fused | deep_blending/playroom | 1.020 | 299 | 293 |
| gsplat_30k_fused | mipnerf360/bicycle | 1.136 | 822 | 723 |
| gsplat_30k_fused | mipnerf360/bonsai | 0.977 | 325 | 333 |
| gsplat_30k_fused | mipnerf360/counter | 0.989 | 348 | 351 |
| gsplat_30k_fused | mipnerf360/garden | 1.071 | 776 | 724 |
| gsplat_30k_fused | mipnerf360/kitchen | 0.964 | 407 | 422 |
| gsplat_30k_fused | mipnerf360/room | 0.999 | 340 | 341 |
| gsplat_30k_fused | mipnerf360/stump | 1.119 | 670 | 599 |
| gsplat_30k_fused | tanks_and_temples/train | 0.883 | 413 | 467 |
| gsplat_30k_fused | tanks_and_temples/truck | 0.890 | 670 | 752 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.301 | 413 | 317 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 1.201 | 299 | 249 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 1.348 | 822 | 610 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 1.045 | 325 | 311 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 1.100 | 348 | 316 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 1.236 | 776 | 628 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 1.068 | 407 | 381 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 1.125 | 340 | 302 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 1.262 | 670 | 531 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 1.071 | 413 | 386 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.112 | 670 | 602 |
| gsplat_30k_fused_win07 | deep_blending/drjohnson | 1.008 | 413 | 410 |
| gsplat_30k_fused_win07 | deep_blending/playroom | 1.027 | 299 | 291 |
| gsplat_30k_fused_win07 | mipnerf360/bicycle | 1.181 | 822 | 696 |
| gsplat_30k_fused_win07 | mipnerf360/bonsai | 0.978 | 325 | 332 |
| gsplat_30k_fused_win07 | mipnerf360/counter | 0.990 | 348 | 351 |
| gsplat_30k_fused_win07 | mipnerf360/garden | 1.138 | 776 | 682 |
| gsplat_30k_fused_win07 | mipnerf360/kitchen | 0.953 | 407 | 427 |
| gsplat_30k_fused_win07 | mipnerf360/room | 1.028 | 340 | 331 |
| gsplat_30k_fused_win07 | mipnerf360/stump | 1.135 | 670 | 591 |
| gsplat_30k_fused_win07 | tanks_and_temples/train | 0.844 | 413 | 489 |
| gsplat_30k_fused_win07 | tanks_and_temples/truck | 0.931 | 670 | 719 |
| gsplat_30k_fused_win07_prune10 | deep_blending/drjohnson | 1.259 | 413 | 328 |
| gsplat_30k_fused_win07_prune10 | deep_blending/playroom | 1.233 | 299 | 242 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/bicycle | 1.401 | 822 | 587 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/bonsai | 0.994 | 325 | 327 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/counter | 1.065 | 348 | 326 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/garden | 1.271 | 776 | 610 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/kitchen | 1.100 | 407 | 370 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/room | 1.113 | 340 | 305 |
| gsplat_30k_fused_win07_prune10 | mipnerf360/stump | 1.290 | 670 | 520 |
| gsplat_30k_fused_win07_prune10 | tanks_and_temples/train | 1.038 | 413 | 398 |
| gsplat_30k_fused_win07_prune10 | tanks_and_temples/truck | 1.096 | 670 | 611 |
| gsplat_30k_fused_win07_prune10_rclip05 | deep_blending/drjohnson | 1.296 | 413 | 319 |
| gsplat_30k_fused_win07_prune10_rclip05 | deep_blending/playroom | 1.232 | 299 | 242 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/bicycle | 1.391 | 822 | 591 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/bonsai | 1.045 | 325 | 311 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/counter | 1.103 | 348 | 315 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/garden | 1.294 | 776 | 599 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/kitchen | 1.092 | 407 | 372 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/room | 1.102 | 340 | 308 |
| gsplat_30k_fused_win07_prune10_rclip05 | mipnerf360/stump | 1.289 | 670 | 520 |
| gsplat_30k_fused_win07_prune10_rclip05 | tanks_and_temples/train | 0.549 | 413 | 752 |
| gsplat_30k_fused_win07_prune10_rclip05 | tanks_and_temples/truck | 1.016 | 670 | 659 |