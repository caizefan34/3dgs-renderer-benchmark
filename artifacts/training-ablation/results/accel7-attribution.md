# HiGS accel7 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k_preload_accum8 | psnr_db | -0.4030 | [-0.6911, -0.0422] | >= -0.10 | False |
| gsplat_27k_preload_accum8 | ssim | -0.0158 | [-0.0303, -0.0035] | >= -0.003 | False |
| gsplat_27k_preload_accum8 | lpips | 0.0261 | [0.0074, 0.0500] | <= +0.005 | False |
| gsplat_27k_preload_accum8 | time_to_quality_seconds | 24.3579 | [-86.2941, 150.7791] | <= 0 | False |
| gsplat_27k_preload_accum8 | speedup ratio | 1.109 | CI lo 0.907 | mean>=1.111 & lo>1.0 | False |
| gsplat_27k_preload_accum8 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3505 | - | descriptive | - |
| gsplat_27k_preload_accum8 | energy_joules (ctrl / cand mean) | 199812 / 161413 | - | descriptive | - |
| gsplat_27k_preload_accum8 | final_gaussian_count (ctrl / cand mean) | 2324813 / 1648942 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish100_acc7 | psnr_db | -0.5368 | [-0.9376, -0.0658] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | ssim | -0.0254 | [-0.0435, -0.0098] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | lpips | 0.0353 | [0.0097, 0.0670] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | time_to_quality_seconds | -104.0578 | [-248.3783, 42.2360] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | speedup ratio | 1.324 | CI lo 0.886 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3494 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | energy_joules (ctrl / cand mean) | 199812 / 125851 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | final_gaussian_count (ctrl / cand mean) | 2324813 / 1634467 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish25_acc7 | psnr_db | -0.5452 | [-0.8790, -0.1412] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | ssim | -0.0252 | [-0.0423, -0.0106] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | lpips | 0.0353 | [0.0112, 0.0657] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | time_to_quality_seconds | -92.3935 | [-238.5015, 56.1034] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | speedup ratio | 1.315 | CI lo 0.875 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3514 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | energy_joules (ctrl / cand mean) | 199812 / 127290 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | final_gaussian_count (ctrl / cand mean) | 2324813 / 1646862 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50_acc7 | psnr_db | -0.6094 | [-0.9952, -0.1298] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | ssim | -0.0255 | [-0.0435, -0.0103] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | lpips | 0.0356 | [0.0103, 0.0672] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | time_to_quality_seconds | -104.0355 | [-247.5351, 41.9367] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | speedup ratio | 1.322 | CI lo 0.868 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | peak_gpu_memory_mib (ctrl / cand mean) | 3593 / 3499 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | energy_joules (ctrl / cand mean) | 199812 / 124362 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | final_gaussian_count (ctrl / cand mean) | 2324813 / 1634123 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k_preload_accum8 | deep_blending/drjohnson | 1.168 | 555 | 475 |
| gsplat_27k_preload_accum8 | deep_blending/playroom | 1.137 | 444 | 391 |
| gsplat_27k_preload_accum8 | mipnerf360/bicycle | 1.187 | 1545 | 1302 |
| gsplat_27k_preload_accum8 | mipnerf360/bonsai | 1.053 | 664 | 630 |
| gsplat_27k_preload_accum8 | mipnerf360/counter | 1.091 | 690 | 632 |
| gsplat_27k_preload_accum8 | mipnerf360/garden | 1.225 | 1562 | 1275 |
| gsplat_27k_preload_accum8 | mipnerf360/kitchen | 1.135 | 725 | 639 |
| gsplat_27k_preload_accum8 | mipnerf360/room | 1.111 | 673 | 606 |
| gsplat_27k_preload_accum8 | mipnerf360/stump | 1.133 | 1414 | 1248 |
| gsplat_27k_preload_accum8 | tanks_and_temples/train | 0.907 | 476 | 525 |
| gsplat_27k_preload_accum8 | tanks_and_temples/truck | 1.052 | 427 | 406 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | deep_blending/drjohnson | 1.263 | 555 | 439 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | deep_blending/playroom | 1.254 | 444 | 354 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/bicycle | 1.547 | 1545 | 999 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/bonsai | 1.262 | 664 | 526 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/counter | 1.359 | 690 | 507 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/garden | 1.661 | 1562 | 940 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/kitchen | 1.390 | 725 | 522 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/room | 1.373 | 673 | 490 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | mipnerf360/stump | 1.539 | 1414 | 919 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | tanks_and_temples/train | 0.886 | 476 | 538 |
| higs_eg_sparse_phase_27k_r07_polish100_acc7 | tanks_and_temples/truck | 1.032 | 427 | 414 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | deep_blending/drjohnson | 1.259 | 555 | 440 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | deep_blending/playroom | 1.313 | 444 | 338 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/bicycle | 1.529 | 1545 | 1011 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/bonsai | 1.253 | 664 | 530 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/counter | 1.327 | 690 | 520 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/garden | 1.641 | 1562 | 952 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/kitchen | 1.350 | 725 | 537 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/room | 1.371 | 673 | 491 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | mipnerf360/stump | 1.519 | 1414 | 931 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | tanks_and_temples/train | 0.875 | 476 | 544 |
| higs_eg_sparse_phase_27k_r07_polish25_acc7 | tanks_and_temples/truck | 1.031 | 427 | 415 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | deep_blending/drjohnson | 1.255 | 555 | 442 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | deep_blending/playroom | 1.302 | 444 | 341 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/bicycle | 1.549 | 1545 | 997 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/bonsai | 1.263 | 664 | 526 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/counter | 1.356 | 690 | 509 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/garden | 1.643 | 1562 | 951 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/kitchen | 1.379 | 725 | 526 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/room | 1.368 | 673 | 492 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | mipnerf360/stump | 1.533 | 1414 | 922 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | tanks_and_temples/train | 0.868 | 476 | 549 |
| higs_eg_sparse_phase_27k_r07_polish50_acc7 | tanks_and_temples/truck | 1.029 | 427 | 415 |