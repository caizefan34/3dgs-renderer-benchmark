# HiGS Confirmatory Matrix Analysis (5 methods x 11 scenes x 3 seeds)

- baseline: `gsplat`; methods: higs_current, higs_full, higs_switch_12k, higs_switch_21k
- jobs analyzed: 165; paired (scene, seed) cells: 33

## Paired deltas vs gsplat (scene-block bootstrap 95% CI)

| method | metric | delta mean | 95% CI | cohen's dz | CV | NI margin | passed |
|---|---|---|---|---|---|---|---|
| higs_current | psnr_db | -0.3901 | [-0.5959, -0.1983] | -1.04 | 0.965 | >= -0.1 | False |
| higs_current | ssim | -0.0120 | [-0.0179, -0.0071] | -1.25 | 0.803 | >= -0.003 | False |
| higs_current | lpips | 0.0130 | [0.0082, 0.0184] | 1.33 | 0.752 | <= 0.005 | False |
| higs_current | wall_time_seconds | -172.0317 | [-383.5130, 21.3063] | -0.49 | 2.062 | - | None |
| higs_current | time_to_quality_seconds | -26.8591 | [-213.1402, 134.1846] | -0.09 | 11.349 | <= 0.0 | False |
| higs_current | peak_gpu_memory_mib | -844.0964 | [-1469.3710, -304.7125] | -0.83 | 1.204 | - | None |
| higs_current | energy_joules | -72247.9543 | [-127722.6261, -23233.4654] | -0.79 | 1.257 | - | None |
| higs_current | final_gaussian_count | -526170.0606 | [-939984.9697, -176305.3333] | -0.80 | 1.253 | - | None |
| higs_current | wall speedup ratio | 1.187 | CI lo 0.718 (need > 1.0) | - | - | mean >= 1.111 | False |
| higs_full | psnr_db | -0.0103 | [-0.0522, 0.0372] | -0.07 | 13.202 | >= -0.1 | True |
| higs_full | ssim | 0.0003 | [0.0000, 0.0006] | 0.17 | 5.768 | >= -0.003 | True |
| higs_full | lpips | -0.0001 | [-0.0006, 0.0002] | -0.05 | 18.605 | <= 0.005 | True |
| higs_full | wall_time_seconds | 19.8610 | [3.2607, 39.1243] | 0.57 | 1.738 | - | None |
| higs_full | time_to_quality_seconds | 6.3743 | [-5.8581, 20.1626] | 0.14 | 7.190 | <= 0.0 | False |
| higs_full | peak_gpu_memory_mib | -32.7897 | [-64.7032, -3.8319] | -0.55 | 1.838 | - | None |
| higs_full | energy_joules | -3548.9859 | [-7792.9294, 624.4087] | -0.38 | 2.659 | - | None |
| higs_full | final_gaussian_count | -13149.7273 | [-21015.7576, -6628.1818] | -0.55 | 1.829 | - | None |
| higs_full | wall speedup ratio | 0.965 | CI lo 0.845 (need > 1.0) | - | - | mean >= 1.111 | False |
| higs_switch_12k | psnr_db | -0.1959 | [-0.3514, -0.0544] | -0.62 | 1.618 | >= -0.1 | False |
| higs_switch_12k | ssim | -0.0053 | [-0.0083, -0.0027] | -0.97 | 1.034 | >= -0.003 | False |
| higs_switch_12k | lpips | 0.0048 | [0.0026, 0.0072] | 1.03 | 0.980 | <= 0.005 | False |
| higs_switch_12k | wall_time_seconds | -116.6137 | [-239.8037, -4.5878] | -0.57 | 1.764 | - | None |
| higs_switch_12k | time_to_quality_seconds | -64.0550 | [-172.8085, 27.3833] | -0.34 | 2.930 | <= 0.0 | False |
| higs_switch_12k | peak_gpu_memory_mib | -566.8954 | [-1046.3543, -152.0033] | -0.73 | 1.373 | - | None |
| higs_switch_12k | energy_joules | -47717.1015 | [-81941.6490, -16896.5386] | -0.84 | 1.190 | - | None |
| higs_switch_12k | final_gaussian_count | -336455.1818 | [-656307.8788, -70985.3030] | -0.67 | 1.502 | - | None |
| higs_switch_12k | wall speedup ratio | 1.102 | CI lo 0.826 (need > 1.0) | - | - | mean >= 1.111 | False |
| higs_switch_21k | psnr_db | -0.3826 | [-0.6181, -0.1838] | -0.97 | 1.035 | >= -0.1 | False |
| higs_switch_21k | ssim | -0.0111 | [-0.0179, -0.0062] | -1.06 | 0.943 | >= -0.003 | False |
| higs_switch_21k | lpips | 0.0127 | [0.0069, 0.0205] | 1.01 | 0.995 | <= 0.005 | False |
| higs_switch_21k | wall_time_seconds | -161.2478 | [-351.4020, 11.2827] | -0.51 | 1.972 | - | None |
| higs_switch_21k | time_to_quality_seconds | -16.6874 | [-186.6261, 130.8199] | -0.06 | 16.710 | <= 0.0 | False |
| higs_switch_21k | peak_gpu_memory_mib | -847.0534 | [-1461.3022, -311.2664] | -0.84 | 1.182 | - | None |
| higs_switch_21k | energy_joules | -67417.3854 | [-118061.4941, -22450.5609] | -0.81 | 1.236 | - | None |
| higs_switch_21k | final_gaussian_count | -527375.9394 | [-934357.3939, -180619.7273] | -0.81 | 1.232 | - | None |
| higs_switch_21k | wall speedup ratio | 1.164 | CI lo 0.753 (need > 1.0) | - | - | mean >= 1.111 | False |
