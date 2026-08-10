# HiGS accel5 Attribution (paired vs in-matrix gsplat_27k) (paired vs gsplat_27k)

| config | metric | delta mean | 95% CI | gate | passed |
|---|---|---|---|---|---|
| higs_eg_sparse_phase_27k_r07 | psnr_db | -0.0620 | [-0.1670, 0.0808] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07 | ssim | -0.0081 | [-0.0100, -0.0063] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07 | lpips | 0.0056 | [-0.0006, 0.0104] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07 | time_to_quality_seconds | -55.4253 | [-142.3454, 40.6972] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07 | speedup ratio | 1.122 | CI lo 0.845 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07 | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3560 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | energy_joules (ctrl / cand mean) | 198653 / 157094 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07 | final_gaussian_count (ctrl / cand mean) | 2314602 / 2306465 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish100 | psnr_db | -0.0720 | [-0.1802, 0.0590] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | ssim | -0.0078 | [-0.0098, -0.0057] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | lpips | 0.0054 | [-0.0006, 0.0100] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | time_to_quality_seconds | -51.7294 | [-143.9544, 49.2689] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | speedup ratio | 1.124 | CI lo 0.845 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish100 | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3566 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100 | energy_joules (ctrl / cand mean) | 198653 / 160350 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish100 | final_gaussian_count (ctrl / cand mean) | 2314602 / 2309036 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish25 | psnr_db | -0.0046 | [-0.1521, 0.2047] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | ssim | -0.0069 | [-0.0086, -0.0052] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | lpips | 0.0046 | [-0.0018, 0.0094] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | time_to_quality_seconds | -41.4232 | [-131.7243, 57.4786] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | speedup ratio | 1.110 | CI lo 0.840 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish25 | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3560 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25 | energy_joules (ctrl / cand mean) | 198653 / 164314 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish25 | final_gaussian_count (ctrl / cand mean) | 2314602 / 2305255 | - | descriptive | - |

| higs_eg_sparse_phase_27k_r07_polish50 | psnr_db | -0.0398 | [-0.1680, 0.1297] | >= -0.10 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | ssim | -0.0076 | [-0.0095, -0.0057] | >= -0.003 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | lpips | 0.0052 | [-0.0009, 0.0098] | <= +0.005 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | time_to_quality_seconds | -48.3060 | [-136.8678, 48.1413] | <= 0 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | speedup ratio | 1.113 | CI lo 0.848 | mean>=1.111 & lo>1.0 | False |
| higs_eg_sparse_phase_27k_r07_polish50 | peak_gpu_memory_mib (ctrl / cand mean) | 3579 / 3570 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | energy_joules (ctrl / cand mean) | 198653 / 161342 | - | descriptive | - |
| higs_eg_sparse_phase_27k_r07_polish50 | final_gaussian_count (ctrl / cand mean) | 2314602 / 2312104 | - | descriptive | - |

## Per-scene speedup ratio (control wall / candidate wall, seed 0)

| config | scene | speedup | control s | cand s |
|---|---|---|---|---|
| higs_eg_sparse_phase_27k_r07 | deep_blending/drjohnson | 1.023 | 549 | 537 |
| higs_eg_sparse_phase_27k_r07 | deep_blending/playroom | 1.054 | 435 | 413 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bicycle | 1.242 | 1541 | 1241 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/bonsai | 1.202 | 666 | 554 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/counter | 1.183 | 679 | 574 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/garden | 1.269 | 1550 | 1222 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/kitchen | 1.131 | 721 | 638 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/room | 1.189 | 665 | 559 |
| higs_eg_sparse_phase_27k_r07 | mipnerf360/stump | 1.321 | 1398 | 1058 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/train | 0.845 | 468 | 554 |
| higs_eg_sparse_phase_27k_r07 | tanks_and_temples/truck | 0.886 | 424 | 478 |
| higs_eg_sparse_phase_27k_r07_polish100 | deep_blending/drjohnson | 1.023 | 549 | 537 |
| higs_eg_sparse_phase_27k_r07_polish100 | deep_blending/playroom | 1.059 | 435 | 410 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/bicycle | 1.238 | 1541 | 1245 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/bonsai | 1.203 | 666 | 553 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/counter | 1.181 | 679 | 575 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/garden | 1.274 | 1550 | 1216 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/kitchen | 1.125 | 721 | 641 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/room | 1.197 | 665 | 555 |
| higs_eg_sparse_phase_27k_r07_polish100 | mipnerf360/stump | 1.318 | 1398 | 1060 |
| higs_eg_sparse_phase_27k_r07_polish100 | tanks_and_temples/train | 0.845 | 468 | 554 |
| higs_eg_sparse_phase_27k_r07_polish100 | tanks_and_temples/truck | 0.895 | 424 | 474 |
| higs_eg_sparse_phase_27k_r07_polish25 | deep_blending/drjohnson | 1.023 | 549 | 537 |
| higs_eg_sparse_phase_27k_r07_polish25 | deep_blending/playroom | 1.029 | 435 | 423 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/bicycle | 1.237 | 1541 | 1246 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/bonsai | 1.170 | 666 | 569 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/counter | 1.170 | 679 | 581 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/garden | 1.259 | 1550 | 1231 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/kitchen | 1.124 | 721 | 642 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/room | 1.164 | 665 | 571 |
| higs_eg_sparse_phase_27k_r07_polish25 | mipnerf360/stump | 1.308 | 1398 | 1068 |
| higs_eg_sparse_phase_27k_r07_polish25 | tanks_and_temples/train | 0.840 | 468 | 557 |
| higs_eg_sparse_phase_27k_r07_polish25 | tanks_and_temples/truck | 0.891 | 424 | 476 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/drjohnson | 1.026 | 549 | 535 |
| higs_eg_sparse_phase_27k_r07_polish50 | deep_blending/playroom | 1.031 | 435 | 422 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bicycle | 1.241 | 1541 | 1242 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/bonsai | 1.164 | 666 | 572 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/counter | 1.175 | 679 | 578 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/garden | 1.271 | 1550 | 1219 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/kitchen | 1.126 | 721 | 641 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/room | 1.189 | 665 | 559 |
| higs_eg_sparse_phase_27k_r07_polish50 | mipnerf360/stump | 1.309 | 1398 | 1067 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/train | 0.848 | 468 | 552 |
| higs_eg_sparse_phase_27k_r07_polish50 | tanks_and_temples/truck | 0.869 | 424 | 488 |