# HiGS Accel5 Exploration (paired vs gsplat 30k control) (paired vs gsplat 30k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| gsplat_27k | psnr_db | -0.0762 | [-0.2123, 0.0723] | >= -0.10 | False |
| gsplat_27k | ssim | -0.0005 | [-0.0016, 0.0006] | >= -0.003 | True |
| gsplat_27k | lpips | 0.0018 | [0.0001, 0.0036] | <= +0.005 | True |
| gsplat_27k | time_to_quality_seconds | -94.0310 | [-144.4754, -45.8865] | <= 0 | True |
| gsplat_27k | speedup ratio | 1.124 | CI lo 1.089 | mean>=1.111 & lo>1.0 | True |
| gsplat_27k | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3579 | - | descriptive | - |
| gsplat_27k | energy_joules (ctrl / cand mean) | 222903 / 198653 | - | descriptive | - |
| gsplat_27k | final_gaussian_count (ctrl / cand mean) | 2400650 / 2314602 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07 | psnr_db | -0.1382 | [-0.2909, 0.0119] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07 | ssim | -0.0086 | [-0.0113, -0.0059] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07 | lpips | 0.0074 | [0.0015, 0.0132] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07 | time_to_quality_seconds | -149.4563 | [-260.0411, -48.2625] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07 | speedup ratio | 1.261 | CI lo 0.932 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3560 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | energy_joules (ctrl / cand mean) | 222903 / 157094 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2306465 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish100 | psnr_db | -0.1482 | [-0.2834, -0.0140] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | ssim | -0.0083 | [-0.0111, -0.0054] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | lpips | 0.0072 | [0.0016, 0.0128] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | time_to_quality_seconds | -145.7605 | [-261.6975, -36.5122] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish100 | speedup ratio | 1.262 | CI lo 0.932 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3566 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100 | energy_joules (ctrl / cand mean) | 222903 / 160350 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2309036 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish25 | psnr_db | -0.0808 | [-0.2606, 0.1217] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | ssim | -0.0074 | [-0.0099, -0.0049] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | lpips | 0.0064 | [0.0004, 0.0121] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | time_to_quality_seconds | -135.4543 | [-250.2422, -26.5826] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish25 | speedup ratio | 1.247 | CI lo 0.927 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3560 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25 | energy_joules (ctrl / cand mean) | 222903 / 164314 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2305255 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50 | psnr_db | -0.1160 | [-0.2827, 0.0577] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | ssim | -0.0081 | [-0.0108, -0.0054] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | lpips | 0.0070 | [0.0012, 0.0126] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | time_to_quality_seconds | -142.3371 | [-256.2686, -37.0617] | <= 0 | True |
| higs_eg_sparse_phase_27k_r07_polish50 | speedup ratio | 1.251 | CI lo 0.935 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | peak_gpu_memory_mib (ctrl / cand mean) | 3705 / 3570 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | energy_joules (ctrl / cand mean) | 222903 / 161342 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | final_gaussian_count (ctrl / cand mean) | 2400650 / 2312104 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| gsplat_27k | deep_blending/drjohnson | 1.133 | 623 | 549 |
| gsplat_27k | deep_blending/playroom | 1.125 | 489 | 435 |
| gsplat_27k | mipnerf360/bicycle | 1.127 | 1737 | 1541 |
| gsplat_27k | mipnerf360/bonsai | 1.094 | 729 | 666 |
| gsplat_27k | mipnerf360/counter | 1.089 | 740 | 679 |
| gsplat_27k | mipnerf360/garden | 1.128 | 1748 | 1550 |
| gsplat_27k | mipnerf360/kitchen | 1.110 | 801 | 721 |
| gsplat_27k | mipnerf360/room | 1.153 | 766 | 665 |
| gsplat_27k | mipnerf360/stump | 1.127 | 1574 | 1398 |
| gsplat_27k | tanks_and_temples/train | 1.103 | 516 | 468 |
| gsplat_27k | tanks_and_temples/truck | 1.174 | 498 | 424 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/drjohnson | 1.160 | 623 | 537 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/playroom | 1.185 | 489 | 413 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bicycle | 1.400 | 1737 | 1241 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bonsai | 1.315 | 729 | 554 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/counter | 1.289 | 740 | 574 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/garden | 1.431 | 1748 | 1222 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/kitchen | 1.255 | 801 | 638 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/room | 1.371 | 766 | 559 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/stump | 1.489 | 1574 | 1058 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/train | 0.932 | 516 | 554 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/truck | 1.041 | 498 | 478 |
| higs_eg_sparse_phase_27k_r07_polish100 | deep_blending/drjohnson | 1.160 | 623 | 537 |
| higs_eg_sparse_phase_27k_r07_polish100 | deep_blending/playroom | 1.191 | 489 | 410 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/bicycle | 1.395 | 1737 | 1245 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/bonsai | 1.316 | 729 | 553 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/counter | 1.287 | 740 | 575 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/garden | 1.438 | 1748 | 1216 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/kitchen | 1.248 | 801 | 641 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/room | 1.380 | 766 | 555 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/stump | 1.485 | 1574 | 1060 |
| higs_eg_sparse_phase_27k_r07_polish100 | tanks_and_temples/train | 0.932 | 516 | 554 |
| higs_eg_sparse_phase_27k_r07_polish100 | tanks_and_temples/truck | 1.051 | 498 | 474 |
| higs_eg_sparse_phase_27k_r07_polish25 | deep_blending/drjohnson | 1.160 | 623 | 537 |
| higs_eg_sparse_phase_27k_r07_polish25 | deep_blending/playroom | 1.157 | 489 | 423 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/bicycle | 1.394 | 1737 | 1246 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/bonsai | 1.280 | 729 | 569 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/counter | 1.274 | 740 | 581 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/garden | 1.420 | 1748 | 1231 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/kitchen | 1.247 | 801 | 642 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/room | 1.342 | 766 | 571 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/stump | 1.474 | 1574 | 1068 |
| higs_eg_sparse_phase_27k_r07_polish25 | tanks_and_temples/train | 0.927 | 516 | 557 |
| higs_eg_sparse_phase_27k_r07_polish25 | tanks_and_temples/truck | 1.046 | 498 | 476 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/drjohnson | 1.163 | 623 | 535 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/playroom | 1.160 | 489 | 422 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bicycle | 1.398 | 1737 | 1242 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bonsai | 1.273 | 729 | 572 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/counter | 1.279 | 740 | 578 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/garden | 1.434 | 1748 | 1219 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/kitchen | 1.250 | 801 | 641 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/room | 1.371 | 766 | 559 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/stump | 1.475 | 1574 | 1067 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/train | 0.935 | 516 | 552 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/truck | 1.021 | 498 | 488 |