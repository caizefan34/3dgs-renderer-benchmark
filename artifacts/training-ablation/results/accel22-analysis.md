# HiGS accel22 Exploration (paired vs in-matrix gsplat 30k) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat | psnr_db | 0.0000 | [0.0000, 0.0000] | >= -0.1 | True |
| gsplat | ssim | 0.0000 | [0.0000, 0.0000] | >= -0.003 | True |
| gsplat | lpips | 0.0000 | [0.0000, 0.0000] | <= 0.005 | True |
| gsplat | time_to_quality_seconds | 0.0000 | [0.0000, 0.0000] | <= 0.0 | True |
| gsplat | speedup ratio | 1.000 | CI lo 1.000 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat | peak_gpu_memory_mib (ctrl / cand mean) | 3720 / 3720 | - | descriptive | - |
| gsplat | energy_joules (ctrl / cand mean) | 158263 / 158263 | - | descriptive | - |
| gsplat | final_gaussian_count (ctrl / cand mean) | 2406431 / 2406431 | - | descriptive | - |
| gsplat_30k_fused | psnr_db | -0.0084 | [-0.0859, 0.0650] | >= -0.1 | True |
| gsplat_30k_fused | ssim | 0.0004 | [-0.0002, 0.0011] | >= -0.003 | True |
| gsplat_30k_fused | lpips | 0.0002 | [-0.0004, 0.0009] | <= 0.005 | True |
| gsplat_30k_fused | time_to_quality_seconds | 71.7500 | [-16.4805, 186.9399] | <= 0.0 | False |
| gsplat_30k_fused | speedup ratio | 0.941 | CI lo 0.448 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused | peak_gpu_memory_mib (ctrl / cand mean) | 3720 / 3709 | - | descriptive | - |
| gsplat_30k_fused | energy_joules (ctrl / cand mean) | 158263 / 171057 | - | descriptive | - |
| gsplat_30k_fused | final_gaussian_count (ctrl / cand mean) | 2406431 / 1696285 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | 0.0378 | [-0.0691, 0.1834] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0005 | [-0.0015, 0.0004] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0020 | [0.0010, 0.0031] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -32.3064 | [-108.5429, 42.1073] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05 | speedup ratio | 1.137 | CI lo 0.786 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib (ctrl / cand mean) | 3720 / 3692 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | energy_joules (ctrl / cand mean) | 158263 / 133353 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count (ctrl / cand mean) | 2406431 / 1038129 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_dens600 | psnr_db | -0.0056 | [-0.0857, 0.0889] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05_dens600 | ssim | -0.0006 | [-0.0015, 0.0004] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05_dens600 | lpips | 0.0020 | [0.0010, 0.0028] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05_dens600 | time_to_quality_seconds | -0.4303 | [-86.1104, 82.0015] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05_dens600 | speedup ratio | 1.086 | CI lo 0.674 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_dens600 | peak_gpu_memory_mib (ctrl / cand mean) | 3720 / 3702 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_dens600 | energy_joules (ctrl / cand mean) | 158263 / 145781 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_dens600 | final_gaussian_count (ctrl / cand mean) | 2406431 / 1040567 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_shfp16 | psnr_db | 0.0384 | [-0.0481, 0.1536] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05_shfp16 | ssim | -0.0003 | [-0.0012, 0.0006] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05_shfp16 | lpips | 0.0016 | [0.0006, 0.0026] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05_shfp16 | time_to_quality_seconds | 25.8854 | [-66.5726, 136.7329] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05_shfp16 | speedup ratio | 1.041 | CI lo 0.490 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_shfp16 | peak_gpu_memory_mib (ctrl / cand mean) | 3720 / 3755 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_shfp16 | energy_joules (ctrl / cand mean) | 158263 / 148737 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_shfp16 | final_gaussian_count (ctrl / cand mean) | 2406431 / 1022698 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | psnr_db | 0.0516 | [-0.0636, 0.1975] | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | ssim | -0.0003 | [-0.0013, 0.0007] | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | lpips | 0.0017 | [0.0003, 0.0031] | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | time_to_quality_seconds | 20.1147 | [-57.4712, 98.8168] | <= 0.0 | False |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | speedup ratio | 1.020 | CI lo 0.600 | mean>=1.1111111111111112 & lo>1.0 | False |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | peak_gpu_memory_mib (ctrl / cand mean) | 3720 / 3735 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | energy_joules (ctrl / cand mean) | 158263 / 150882 | - | descriptive | - |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | final_gaussian_count (ctrl / cand mean) | 2406431 / 1015742 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat | deep_blending/drjohnson | 1.000 | 517 | 517 |
| gsplat | deep_blending/playroom | 1.000 | 428 | 428 |
| gsplat | mipnerf360/bicycle | 1.000 | 961 | 961 |
| gsplat | mipnerf360/bonsai | 1.000 | 328 | 328 |
| gsplat | mipnerf360/counter | 1.000 | 366 | 366 |
| gsplat | mipnerf360/garden | 1.000 | 790 | 790 |
| gsplat | mipnerf360/kitchen | 1.000 | 456 | 456 |
| gsplat | mipnerf360/room | 1.000 | 767 | 767 |
| gsplat | mipnerf360/stump | 1.000 | 1413 | 1413 |
| gsplat | tanks_and_temples/train | 1.000 | 798 | 798 |
| gsplat | tanks_and_temples/truck | 1.000 | 690 | 690 |
| gsplat_30k_fused | deep_blending/drjohnson | 1.032 | 517 | 500 |
| gsplat_30k_fused | deep_blending/playroom | 1.058 | 428 | 405 |
| gsplat_30k_fused | mipnerf360/bicycle | 1.079 | 961 | 891 |
| gsplat_30k_fused | mipnerf360/bonsai | 0.944 | 328 | 348 |
| gsplat_30k_fused | mipnerf360/counter | 0.995 | 366 | 368 |
| gsplat_30k_fused | mipnerf360/garden | 1.077 | 790 | 734 |
| gsplat_30k_fused | mipnerf360/kitchen | 0.448 | 456 | 1018 |
| gsplat_30k_fused | mipnerf360/room | 1.204 | 767 | 637 |
| gsplat_30k_fused | mipnerf360/stump | 0.872 | 1413 | 1621 |
| gsplat_30k_fused | tanks_and_temples/train | 0.855 | 798 | 934 |
| gsplat_30k_fused | tanks_and_temples/truck | 0.783 | 690 | 882 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/drjohnson | 1.276 | 517 | 405 |
| gsplat_30k_fused_prune10_rclip05 | deep_blending/playroom | 1.264 | 428 | 339 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bicycle | 1.546 | 961 | 622 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/bonsai | 1.017 | 328 | 323 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/counter | 1.112 | 366 | 329 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/garden | 1.223 | 790 | 647 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/kitchen | 0.856 | 456 | 532 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/room | 0.786 | 767 | 977 |
| gsplat_30k_fused_prune10_rclip05 | mipnerf360/stump | 1.451 | 1413 | 974 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/train | 0.954 | 798 | 836 |
| gsplat_30k_fused_prune10_rclip05 | tanks_and_temples/truck | 1.022 | 690 | 675 |
| gsplat_30k_fused_prune10_rclip05_dens600 | deep_blending/drjohnson | 1.240 | 517 | 417 |
| gsplat_30k_fused_prune10_rclip05_dens600 | deep_blending/playroom | 1.366 | 428 | 314 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/bicycle | 1.539 | 961 | 624 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/bonsai | 1.016 | 328 | 323 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/counter | 1.107 | 366 | 331 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/garden | 1.230 | 790 | 643 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/kitchen | 0.674 | 456 | 676 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/room | 0.995 | 767 | 772 |
| gsplat_30k_fused_prune10_rclip05_dens600 | mipnerf360/stump | 0.980 | 1413 | 1442 |
| gsplat_30k_fused_prune10_rclip05_dens600 | tanks_and_temples/train | 0.997 | 798 | 800 |
| gsplat_30k_fused_prune10_rclip05_dens600 | tanks_and_temples/truck | 0.797 | 690 | 866 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | deep_blending/drjohnson | 1.097 | 517 | 471 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | deep_blending/playroom | 1.120 | 428 | 383 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/bicycle | 1.438 | 961 | 668 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/bonsai | 0.987 | 328 | 333 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/counter | 1.043 | 366 | 351 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/garden | 1.073 | 790 | 737 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/kitchen | 0.490 | 456 | 929 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/room | 0.793 | 767 | 968 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | mipnerf360/stump | 1.406 | 1413 | 1006 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | tanks_and_temples/train | 1.002 | 798 | 797 |
| gsplat_30k_fused_prune10_rclip05_shfp16 | tanks_and_temples/truck | 1.000 | 690 | 690 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | deep_blending/drjohnson | 1.131 | 517 | 457 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | deep_blending/playroom | 1.112 | 428 | 385 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/bicycle | 1.455 | 961 | 660 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/bonsai | 0.986 | 328 | 333 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/counter | 1.057 | 366 | 347 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/garden | 0.971 | 790 | 814 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/kitchen | 0.600 | 456 | 759 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/room | 0.983 | 767 | 781 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | mipnerf360/stump | 0.948 | 1413 | 1491 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | tanks_and_temples/train | 0.978 | 798 | 816 |
| gsplat_30k_fused_prune10_rclip05_shfp16_dens600 | tanks_and_temples/truck | 1.003 | 690 | 688 |