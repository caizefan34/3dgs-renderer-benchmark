# HiGS accel21 Exploration (paired vs in-matrix gsplat 30k) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat | psnr_db | 0.0000 | [0.0000, 0.0000] | >= -0.1 | True |
| gsplat | ssim | 0.0000 | [0.0000, 0.0000] | >= -0.003 | True |
| gsplat | lpips | 0.0000 | [0.0000, 0.0000] | <= 0.005 | True |
| gsplat | time_to_quality_seconds | 0.0000 | [0.0000, 0.0000] | <= 0.0 | True |
| gsplat | speedup ratio | 1.000 | CI lo 1.000 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat | peak_gpu_memory_mib (ctrl / cand mean) | 3712 / 3712 | - | descriptive | - |
| gsplat | energy_joules (ctrl / cand mean) | 116954 / 116954 | - | descriptive | - |
| gsplat | final_gaussian_count (ctrl / cand mean) | 2403320 / 2403320 | - | descriptive | - |
| gsplat_30k_fused | psnr_db | -0.0410 | [-0.1831, 0.0491] | >= -0.1 | False |
| gsplat_30k_fused | ssim | -0.0009 | [-0.0024, 0.0001] | >= -0.003 | True |
| gsplat_30k_fused | lpips | 0.0009 | [0.0002, 0.0017] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | -4.3561 | [-28.9529, 13.4221] | <= 0.0 | False |
| gsplat_30k_fused | speedup ratio | 1.017 | CI lo 0.910 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3712 / 3709 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 116954 / 107392 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2403320 / 1696024 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | -0.0077 | [-0.0816, 0.0675] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0011 | [-0.0022, 0.0002] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0021 | [0.0011, 0.0032] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -60.1588 | [-102.3788, -23.0009] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 1.182 | CI lo 0.967 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3712 / 3715 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 116954 / 90380 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2403320 / 1043728 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | psnr_db | -0.1464 | [-0.2448, -0.0383] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | ssim | -0.0017 | [-0.0033, 0.0001] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | lpips | 0.0041 | [0.0021, 0.0061] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | time_to_quality_seconds | -81.7886 | [-136.9951, -26.9305] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | speedup ratio | 1.305 | CI lo 1.021 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | peak_gpu_memory_mib (ctrl / cand mean) | 3712 / 3688 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | energy_joules (ctrl / cand mean) | 116954 / 79846 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | final_gaussian_count (ctrl / cand mean) | 2403320 / 1073995 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | psnr_db | -0.1753 | [-0.3421, -0.0223] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | ssim | -0.0029 | [-0.0060, -0.0001] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | lpips | 0.0050 | [0.0018, 0.0084] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | time_to_quality_seconds | -76.0698 | [-134.7173, -27.9513] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | speedup ratio | 1.282 | CI lo 0.987 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | peak_gpu_memory_mib (ctrl / cand mean) | 3712 / 3685 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | energy_joules (ctrl / cand mean) | 116954 / 83949 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | final_gaussian_count (ctrl / cand mean) | 2403320 / 1077616 | - | descriptive | - |
| gsplat_30k_fused_skipbwd_pv | psnr_db | -0.0696 | [-0.1693, 0.0211] | >= -0.1 | False |
| gsplat_30k_fused_skipbwd_pv | ssim | -0.0008 | [-0.0020, 0.0004] | >= -0.003 | True |
| gsplat_30k_fused_skipbwd_pv | lpips | 0.0024 | [0.0010, 0.0041] | <= 0.005 | True |
| gsplat_30k_fused_skipbwd_pv | time_to_quality_seconds | -52.2580 | [-102.2354, -7.1436] | <= 0.0 | True |
| gsplat_30k_fused_skipbwd_pv | speedup ratio | 1.144 | CI lo 0.855 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_skipbwd_pv | peak_gpu_memory_mib (ctrl / cand mean) | 3712 / 3692 | - | descriptive | - |
| gsplat_30k_fused_skipbwd_pv | energy_joules (ctrl / cand mean) | 116954 / 94323 | - | descriptive | - |
| gsplat_30k_fused_skipbwd_pv | final_gaussian_count (ctrl / cand mean) | 2403320 / 1794055 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat | deep_blending/drjohnson | 1.000 | 413 | 413 |
| gsplat | deep_blending/playroom | 1.000 | 300 | 300 |
| gsplat | mipnerf360/bicycle | 1.000 | 892 | 892 |
| gsplat | mipnerf360/bonsai | 1.000 | 326 | 326 |
| gsplat | mipnerf360/counter | 1.000 | 488 | 488 |
| gsplat | mipnerf360/garden | 1.000 | 861 | 861 |
| gsplat | mipnerf360/kitchen | 1.000 | 410 | 410 |
| gsplat | mipnerf360/room | 1.000 | 360 | 360 |
| gsplat | mipnerf360/stump | 1.000 | 665 | 665 |
| gsplat | tanks_and_temples/train | 1.000 | 418 | 418 |
| gsplat | tanks_and_temples/truck | 1.000 | 424 | 424 |
| gsplat_30k_fused | deep_blending/drjohnson | 1.026 | 413 | 402 |
| gsplat_30k_fused | deep_blending/playroom | 1.034 | 300 | 290 |
| gsplat_30k_fused | mipnerf360/bicycle | 1.160 | 892 | 769 |
| gsplat_30k_fused | mipnerf360/bonsai | 0.950 | 326 | 343 |
| gsplat_30k_fused | mipnerf360/counter | 1.020 | 488 | 478 |
| gsplat_30k_fused | mipnerf360/garden | 0.999 | 861 | 861 |
| gsplat_30k_fused | mipnerf360/kitchen | 0.971 | 410 | 422 |
| gsplat_30k_fused | mipnerf360/room | 1.010 | 360 | 357 |
| gsplat_30k_fused | mipnerf360/stump | 1.074 | 665 | 619 |
| gsplat_30k_fused | tanks_and_temples/train | 0.910 | 418 | 460 |
| gsplat_30k_fused | tanks_and_temples/truck | 1.030 | 424 | 411 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.294 | 413 | 319 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 1.015 | 300 | 295 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 1.323 | 892 | 674 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 0.967 | 326 | 337 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 1.527 | 488 | 319 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 1.062 | 861 | 810 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 1.067 | 410 | 384 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 1.142 | 360 | 315 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 1.217 | 665 | 546 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 1.029 | 418 | 406 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.355 | 424 | 313 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | deep_blending/drjohnson | 1.382 | 413 | 299 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | deep_blending/playroom | 1.246 | 300 | 240 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/bicycle | 1.435 | 892 | 621 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/bonsai | 1.021 | 326 | 319 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/counter | 1.660 | 488 | 294 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/garden | 1.273 | 861 | 676 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/kitchen | 1.173 | 410 | 350 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/room | 1.214 | 360 | 297 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | mipnerf360/stump | 1.345 | 665 | 494 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | tanks_and_temples/train | 1.133 | 418 | 369 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv | tanks_and_temples/truck | 1.467 | 424 | 289 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | deep_blending/drjohnson | 1.379 | 413 | 299 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | deep_blending/playroom | 1.335 | 300 | 224 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/bicycle | 1.586 | 892 | 562 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/bonsai | 0.987 | 326 | 330 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/counter | 1.578 | 488 | 309 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/garden | 1.047 | 861 | 822 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/kitchen | 1.187 | 410 | 346 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/room | 1.225 | 360 | 294 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | mipnerf360/stump | 1.334 | 665 | 498 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | tanks_and_temples/train | 1.163 | 418 | 360 |
| gsplat_30k_fused_prune10_rclip05_skipbwd_pv_agg | tanks_and_temples/truck | 1.281 | 424 | 331 |
| gsplat_30k_fused_skipbwd_pv | deep_blending/drjohnson | 1.159 | 413 | 356 |
| gsplat_30k_fused_skipbwd_pv | deep_blending/playroom | 1.132 | 300 | 265 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/bicycle | 1.357 | 892 | 657 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/bonsai | 0.855 | 326 | 381 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/counter | 1.376 | 488 | 354 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/garden | 1.279 | 861 | 673 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/kitchen | 1.060 | 410 | 387 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/room | 1.143 | 360 | 315 |
| gsplat_30k_fused_skipbwd_pv | mipnerf360/stump | 1.210 | 665 | 549 |
| gsplat_30k_fused_skipbwd_pv | tanks_and_temples/train | 1.006 | 418 | 416 |
| gsplat_30k_fused_skipbwd_pv | tanks_and_temples/truck | 1.002 | 424 | 423 |