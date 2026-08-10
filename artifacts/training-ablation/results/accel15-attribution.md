# HiGS Accel15 Attribution (paired vs in-matrix gsplat) (paired vs gsplat (in-matrix))

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_30k_fused | psnr_db | 0.0029 | [-0.0580, 0.0833] | >= -0.1 | True |
| gsplat_30k_fused | ssim | 0.0002 | [-0.0004, 0.0008] | >= -0.003 | True |
| gsplat_30k_fused | lpips | -0.0001 | [-0.0013, 0.0009] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | -5.1105 | [-27.8796, 14.2475] | <= 0.0 | False |
| gsplat_30k_fused | speedup ratio | 1.013 | CI lo 0.900 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3713 / 3696 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 114525 / 105562 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2401800 / 1688201 | - | descriptive | - |
| gsplat_30k_fused_prune10 | psnr_db | -0.0081 | [-0.1389, 0.1191] | >= -0.1 | False |
| gsplat_30k_fused_prune10 | ssim | -0.0004 | [-0.0017, 0.0008] | >= -0.003 | True |
| gsplat_30k_fused_prune10 | lpips | 0.0012 | [-0.0004, 0.0028] | <= 0.005 | True |
| gsplat_30k_fused_prune10 | time_to_quality_seconds | -51.2779 | [-88.1512, -21.1364] | <= 0.0 | True |
| gsplat_30k_fused_prune10 | speedup ratio | 1.157 | CI lo 1.043 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10 | peak_gpu_memory_mib (ctrl / cand mean) | 3713 / 3683 | - | descriptive | - |
| gsplat_30k_fused_prune10 | energy_joules (ctrl / cand mean) | 114525 / 88029 | - | descriptive | - |
| gsplat_30k_fused_prune10 | final_gaussian_count (ctrl / cand mean) | 2401800 / 1033607 | - | descriptive | - |
| gsplat_30k_fused_prune10_accum2 | psnr_db | -0.0335 | [-0.1594, 0.1284] | >= -0.1 | False |
| gsplat_30k_fused_prune10_accum2 | ssim | -0.0016 | [-0.0043, 0.0014] | >= -0.003 | False |
| gsplat_30k_fused_prune10_accum2 | lpips | 0.0045 | [0.0003, 0.0096] | <= 0.005 | False |
| gsplat_30k_fused_prune10_accum2 | time_to_quality_seconds | -72.2385 | [-121.4054, -32.0392] | <= 0.0 | True |
| gsplat_30k_fused_prune10_accum2 | speedup ratio | 1.236 | CI lo 1.062 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_accum2 | peak_gpu_memory_mib (ctrl / cand mean) | 3713 / 3515 | - | descriptive | - |
| gsplat_30k_fused_prune10_accum2 | energy_joules (ctrl / cand mean) | 114525 / 78055 | - | descriptive | - |
| gsplat_30k_fused_prune10_accum2 | final_gaussian_count (ctrl / cand mean) | 2401800 / 1026642 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | -0.0048 | [-0.0846, 0.0977] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0004 | [-0.0014, 0.0006] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0013 | [0.0001, 0.0026] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -50.6696 | [-87.6008, -20.9684] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 1.157 | CI lo 1.041 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3713 / 3696 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 114525 / 88076 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2401800 / 1040168 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum2 | psnr_db | -0.0282 | [-0.1719, 0.1454] | >= -0.1 | False |
| gsplat_30k_fused_prune10_rclip05_accum2 | ssim | -0.0016 | [-0.0046, 0.0016] | >= -0.003 | False |
| gsplat_30k_fused_prune10_rclip05_accum2 | lpips | 0.0049 | [0.0005, 0.0101] | <= 0.005 | False |
| gsplat_30k_fused_prune10_rclip05_accum2 | time_to_quality_seconds | -64.7176 | [-115.8495, -19.5782] | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05_accum2 | speedup ratio | 1.239 | CI lo 1.054 | mean>=1.1111111111111112 & lo>1.0 | True |
| gsplat_30k_fused_prune10_rclip05_accum2 | peak_gpu_memory_mib (ctrl / cand mean) | 3713 / 3518 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum2 | energy_joules (ctrl / cand mean) | 114525 / 78159 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_accum2 | final_gaussian_count (ctrl / cand mean) | 2401800 / 1026577 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_30k_fused | deep_blending/drjohnson | 1.061 | 424 | 400 |
| gsplat_30k_fused | deep_blending/playroom | 1.026 | 306 | 299 |
| gsplat_30k_fused | mipnerf360/bicycle | 1.134 | 839 | 740 |
| gsplat_30k_fused | mipnerf360/bonsai | 0.963 | 341 | 354 |
| gsplat_30k_fused | mipnerf360/counter | 0.994 | 366 | 368 |
| gsplat_30k_fused | mipnerf360/garden | 1.081 | 791 | 732 |
| gsplat_30k_fused | mipnerf360/kitchen | 0.960 | 418 | 435 |
| gsplat_30k_fused | mipnerf360/room | 1.003 | 351 | 350 |
| gsplat_30k_fused | mipnerf360/stump | 1.077 | 666 | 618 |
| gsplat_30k_fused | tanks_and_temples/train | 0.900 | 426 | 474 |
| gsplat_30k_fused | tanks_and_temples/truck | 0.944 | 354 | 375 |
| gsplat_30k_fused_prune10 | deep_blending/drjohnson | 1.270 | 424 | 334 |
| gsplat_30k_fused_prune10 | deep_blending/playroom | 1.169 | 306 | 262 |
| gsplat_30k_fused_prune10 | mipnerf360/bicycle | 1.327 | 839 | 632 |
| gsplat_30k_fused_prune10 | mipnerf360/bonsai | 1.043 | 341 | 327 |
| gsplat_30k_fused_prune10 | mipnerf360/counter | 1.107 | 366 | 331 |
| gsplat_30k_fused_prune10 | mipnerf360/garden | 1.240 | 791 | 638 |
| gsplat_30k_fused_prune10 | mipnerf360/kitchen | 1.073 | 418 | 390 |
| gsplat_30k_fused_prune10 | mipnerf360/room | 1.106 | 351 | 318 |
| gsplat_30k_fused_prune10 | mipnerf360/stump | 1.217 | 666 | 547 |
| gsplat_30k_fused_prune10 | tanks_and_temples/train | 1.054 | 426 | 404 |
| gsplat_30k_fused_prune10 | tanks_and_temples/truck | 1.123 | 354 | 315 |
| gsplat_30k_fused_prune10_accum2 | deep_blending/drjohnson | 1.324 | 424 | 320 |
| gsplat_30k_fused_prune10_accum2 | deep_blending/playroom | 1.276 | 306 | 240 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/bicycle | 1.451 | 839 | 578 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/bonsai | 1.091 | 341 | 313 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/counter | 1.147 | 366 | 319 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/garden | 1.426 | 791 | 555 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/kitchen | 1.162 | 418 | 360 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/room | 1.158 | 351 | 303 |
| gsplat_30k_fused_prune10_accum2 | mipnerf360/stump | 1.330 | 666 | 500 |
| gsplat_30k_fused_prune10_accum2 | tanks_and_temples/train | 1.062 | 426 | 401 |
| gsplat_30k_fused_prune10_accum2 | tanks_and_temples/truck | 1.174 | 354 | 302 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.232 | 424 | 344 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 1.208 | 306 | 254 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 1.337 | 839 | 628 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 1.041 | 341 | 327 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 1.101 | 366 | 332 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 1.239 | 791 | 639 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 1.071 | 418 | 390 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 1.095 | 351 | 321 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 1.191 | 666 | 559 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 1.072 | 426 | 398 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.142 | 354 | 310 |
| gsplat_30k_fused_prune10_rclip05_accum2 | deep_blending/drjohnson | 1.329 | 424 | 319 |
| gsplat_30k_fused_prune10_rclip05_accum2 | deep_blending/playroom | 1.324 | 306 | 231 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/bicycle | 1.433 | 839 | 586 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/bonsai | 1.107 | 341 | 308 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/counter | 1.144 | 366 | 320 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/garden | 1.422 | 791 | 556 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/kitchen | 1.168 | 418 | 358 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/room | 1.148 | 351 | 306 |
| gsplat_30k_fused_prune10_rclip05_accum2 | mipnerf360/stump | 1.315 | 666 | 506 |
| gsplat_30k_fused_prune10_rclip05_accum2 | tanks_and_temples/train | 1.054 | 426 | 404 |
| gsplat_30k_fused_prune10_rclip05_accum2 | tanks_and_temples/truck | 1.184 | 354 | 299 |