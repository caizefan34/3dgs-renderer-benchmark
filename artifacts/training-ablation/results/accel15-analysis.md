# HiGS Accel15 Exploration (paired vs frozen gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat | psnr_db | 0.1217 | [-0.0294, 0.3469] | >= -0.1 | True |
| gsplat | ssim | 0.0021 | [-0.0001, 0.0058] | >= -0.003 | True |
| gsplat | lpips | -0.0031 | [-0.0089, 0.0003] | <= 0.005 | True |
| gsplat | time_to_quality_seconds | -375.6995 | [-537.9924, -230.4173] | <= 0.0 | True |
| gsplat | speedup ratio | 1.871 | CI lo 1.212 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3713 | - | descriptive | - |
| gsplat | energy_joules (ctrl / cand mean) | 222903 / 114525 | - | descriptive | - |
| gsplat | final_gaussian_count (ctrl / cand mean) | 2400650 / 2401800 | - | descriptive | - |
| gsplat_30k_fused | psnr_db | 0.1246 | [-0.0562, 0.3691] | >= -0.1 | True |
| gsplat_30k_fused | ssim | 0.0023 | [-0.0001, 0.0064] | >= -0.003 | True |
| gsplat_30k_fused | lpips | -0.0032 | [-0.0091, 0.0003] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | -380.8100 | [-561.4679, -219.5783] | <= 0.0 | True |
| gsplat_30k_fused | speedup ratio | 1.908 | CI lo 1.090 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3696 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 222903 / 105562 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2400650 / 1688201 | - | descriptive | - |
| gsplat_30k_fused_prune10 | psnr_db | 0.1136 | [-0.1349, 0.4390] | >= -0.1 | False |
| gsplat_30k_fused_prune10 | ssim | 0.0017 | [-0.0013, 0.0061] | >= -0.003 | True |
| gsplat_30k_fused_prune10 | lpips | -0.0019 | [-0.0084, 0.0021] | <= 0.005 | True |
| gsplat_30k_fused_prune10 | time_to_quality_seconds | -426.9775 | [-619.8068, -258.0580] | <= 0.0 | True |
| gsplat_30k_fused_prune10 | speedup ratio | 2.171 | CI lo 1.277 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3683 | - | descriptive | - |
| gsplat_30k_fused_prune10 | energy_joules (ctrl / cand mean) | 222903 / 88029 | - | descriptive | - |
| gsplat_30k_fused_prune10 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1033607 | - | descriptive | - |
| gsplat_30k_fused_prune10_accum2 | psnr_db | 0.0882 | [-0.1392, 0.3576] | >= -0.1 | False |
| gsplat_30k_fused_prune10_accum2 | ssim | 0.0006 | [-0.0035, 0.0055] | >= -0.003 | False |
| gsplat_30k_fused_prune10_accum2 | lpips | 0.0014 | [-0.0064, 0.0082] | <= 0.005 | False |
| gsplat_30k_fused_prune10_accum2 | time_to_quality_seconds | -447.9380 | [-653.9445, -268.4801] | <= 0.0 | True |
| gsplat_30k_fused_prune10_accum2 | speedup ratio | 2.329 | CI lo 1.287 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_accum2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3515 | - | descriptive | - |
| gsplat_30k_fused_prune10_accum2 | energy_joules (ctrl / cand mean) | 222903 / 78055 | - | descriptive | - |
| gsplat_30k_fused_prune10_accum2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1026642 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | 0.1169 | [-0.0940, 0.4320] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | 0.0017 | [-0.0011, 0.0060] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | -0.0018 | [-0.0078, 0.0018] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -426.3691 | [-619.3509, -258.8389] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 2.168 | CI lo 1.299 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3696 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 222903 / 88076 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1040168 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum2 | psnr_db | 0.0935 | [-0.1562, 0.4300] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_accum2 | ssim | 0.0006 | [-0.0039, 0.0063] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_accum2 | lpips | 0.0018 | [-0.0065, 0.0088] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_accum2 | time_to_quality_seconds | -440.4171 | [-646.4895, -262.8417] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05_accum2 | speedup ratio | 2.331 | CI lo 1.277 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05_accum2 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3518 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum2 | energy_joules (ctrl / cand mean) | 222903 / 78159 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum2 | final_gaussian_count (ctrl / cand mean) | 2400650 / 1026577 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat | deep_blending/drjohnson | 1.468 | 623 | 424 |
| gsplat | deep_blending/playroom | 1.595 | 489 | 306 |
| gsplat | mipnerf360/bicycle | 2.070 | 1737 | 839 |
| gsplat | mipnerf360/bonsai | 2.137 | 729 | 341 |
| gsplat | mipnerf360/counter | 2.021 | 740 | 366 |
| gsplat | mipnerf360/garden | 2.211 | 1748 | 791 |
| gsplat | mipnerf360/kitchen | 1.915 | 801 | 418 |
| gsplat | mipnerf360/room | 2.181 | 766 | 351 |
| gsplat | mipnerf360/stump | 2.365 | 1574 | 666 |
| gsplat | tanks_and_temples/train | 1.212 | 516 | 426 |
| gsplat | tanks_and_temples/truck | 1.405 | 498 | 354 |
| gsplat_30k_fused | deep_blending/drjohnson | 1.558 | 623 | 400 |
| gsplat_30k_fused | deep_blending/playroom | 1.637 | 489 | 299 |
| gsplat_30k_fused | mipnerf360/bicycle | 2.348 | 1737 | 740 |
| gsplat_30k_fused | mipnerf360/bonsai | 2.059 | 729 | 354 |
| gsplat_30k_fused | mipnerf360/counter | 2.010 | 740 | 368 |
| gsplat_30k_fused | mipnerf360/garden | 2.390 | 1748 | 732 |
| gsplat_30k_fused | mipnerf360/kitchen | 1.839 | 801 | 435 |
| gsplat_30k_fused | mipnerf360/room | 2.187 | 766 | 350 |
| gsplat_30k_fused | mipnerf360/stump | 2.547 | 1574 | 618 |
| gsplat_30k_fused | tanks_and_temples/train | 1.090 | 516 | 474 |
| gsplat_30k_fused | tanks_and_temples/truck | 1.327 | 498 | 375 |
| gsplat_30k_fused_prune10 | deep_blending/drjohnson | 1.864 | 623 | 334 |
| gsplat_30k_fused_prune10 | deep_blending/playroom | 1.864 | 489 | 262 |
| gsplat_30k_fused_prune10 | mipnerf360/bicycle | 2.746 | 1737 | 632 |
| gsplat_30k_fused_prune10 | mipnerf360/bonsai | 2.229 | 729 | 327 |
| gsplat_30k_fused_prune10 | mipnerf360/counter | 2.238 | 740 | 331 |
| gsplat_30k_fused_prune10 | mipnerf360/garden | 2.742 | 1748 | 638 |
| gsplat_30k_fused_prune10 | mipnerf360/kitchen | 2.054 | 801 | 390 |
| gsplat_30k_fused_prune10 | mipnerf360/room | 2.412 | 766 | 318 |
| gsplat_30k_fused_prune10 | mipnerf360/stump | 2.878 | 1574 | 547 |
| gsplat_30k_fused_prune10 | tanks_and_temples/train | 1.277 | 516 | 404 |
| gsplat_30k_fused_prune10 | tanks_and_temples/truck | 1.579 | 498 | 315 |
| gsplat_30k_fused_prune10_accum2 | deep_blending/drjohnson | 1.943 | 623 | 320 |
| gsplat_30k_fused_prune10_accum2 | deep_blending/playroom | 2.036 | 489 | 240 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/bicycle | 3.004 | 1737 | 578 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/bonsai | 2.330 | 729 | 313 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/counter | 2.318 | 740 | 319 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/garden | 3.152 | 1748 | 555 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/kitchen | 2.224 | 801 | 360 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/room | 2.526 | 766 | 303 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/stump | 3.146 | 1574 | 500 |
| gsplat_30k_fused_prune10_accum2 | tanks_and_temples/train | 1.287 | 516 | 401 |
| gsplat_30k_fused_prune10_accum2 | tanks_and_temples/truck | 1.651 | 498 | 302 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.808 | 623 | 344 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 1.927 | 489 | 254 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 2.767 | 1737 | 628 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 2.225 | 729 | 327 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 2.225 | 740 | 332 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 2.738 | 1748 | 639 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 2.050 | 801 | 390 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 2.388 | 766 | 321 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 2.817 | 1574 | 559 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 1.299 | 516 | 398 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.605 | 498 | 310 |
| gsplat_30k_fused_prune10_rclip05_accum2 | deep_blending/drjohnson | 1.952 | 623 | 319 |
| gsplat_30k_fused_prune10_rclip05_accum2 | deep_blending/playroom | 2.113 | 489 | 231 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/bicycle | 2.966 | 1737 | 586 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/bonsai | 2.366 | 729 | 308 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/counter | 2.311 | 740 | 320 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/garden | 3.144 | 1748 | 556 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/kitchen | 2.236 | 801 | 358 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/room | 2.504 | 766 | 306 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/stump | 3.109 | 1574 | 506 |
| gsplat_30k_fused_prune10_rclip05_accum2 | tanks_and_temples/train | 1.277 | 516 | 404 |
| gsplat_30k_fused_prune10_rclip05_accum2 | tanks_and_temples/truck | 1.663 | 498 | 299 |