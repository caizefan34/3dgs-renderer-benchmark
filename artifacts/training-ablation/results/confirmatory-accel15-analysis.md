# HiGS Confirmatory Matrix Analysis (3 methods x 11 scenes)

- baseline: `gsplat`; methods: gsplat_25k, gsplat_30k_fused_prune10_rclip05, higs_visible_only
- jobs analyzed: 132; paired (scene, seed) cells: 33

## Paired deltas vs gsplat (scene-block bootstrap 95% CI)

| method | metric | delta mean | 95% CI | cohen's dz | CV | NI margin | passed |
|---|---|---|---|---|---|---|---|
| gsplat_25k | psnr_db | -0.0567 | [-0.1215, 0.0163] | -0.34 | 2.921 | >= -0.1 | False |
| gsplat_25k | ssim | -0.0013 | [-0.0025, 0.0000] | -0.52 | 1.929 | >= -0.003 | True |
| gsplat_25k | lpips | 0.0030 | [0.0013, 0.0053] | 0.81 | 1.226 | <= 0.005 | False |
| gsplat_25k | wall_time_seconds | -87.0230 | [-111.5604, -65.2521] | -2.13 | 0.470 | - | None |
| gsplat_25k | time_to_quality_seconds | -67.5240 | [-96.2146, -40.9687] | -1.40 | 0.713 | <= 0.0 | True |
| gsplat_25k | peak_gpu_memory_mib | -186.3489 | [-297.2148, -93.1366] | -0.97 | 1.035 | - | None |
| gsplat_25k | energy_joules | -24945.5456 | [-34940.8195, -16368.9459] | -1.50 | 0.666 | - | None |
| gsplat_25k | final_gaussian_count | -128854.5455 | [-201099.3636, -67102.5152] | -1.00 | 1.004 | - | None |
| gsplat_25k | wall speedup ratio | 1.220 | CI lo 1.178 (need > 1.0) | - | - | mean >= 1.111 | True |
| gsplat_30k_fused_prune10_rclip05 | psnr_db | 0.0577 | [-0.0222, 0.1505] | 0.35 | 2.889 | >= -0.1 | True |
| gsplat_30k_fused_prune10_rclip05 | ssim | -0.0001 | [-0.0011, 0.0010] | -0.04 | 28.485 | >= -0.003 | True |
| gsplat_30k_fused_prune10_rclip05 | lpips | 0.0014 | [0.0005, 0.0025] | 0.74 | 1.353 | <= 0.005 | True |
| gsplat_30k_fused_prune10_rclip05 | wall_time_seconds | -71.8087 | [-109.5208, -39.3271] | -1.18 | 0.850 | - | None |
| gsplat_30k_fused_prune10_rclip05 | time_to_quality_seconds | -49.4536 | [-85.4530, -18.9753] | -0.85 | 1.172 | <= 0.0 | True |
| gsplat_30k_fused_prune10_rclip05 | peak_gpu_memory_mib | -23.2976 | [-58.2217, 6.7372] | -0.22 | 4.618 | - | None |
| gsplat_30k_fused_prune10_rclip05 | energy_joules | -28663.0439 | [-42662.6827, -17001.8452] | -1.28 | 0.781 | - | None |
| gsplat_30k_fused_prune10_rclip05 | final_gaussian_count | -1362966.8788 | [-1950623.9697, -864135.1212] | -1.43 | 0.699 | - | None |
| gsplat_30k_fused_prune10_rclip05 | wall speedup ratio | 1.164 | CI lo 1.034 (need > 1.0) | - | - | mean >= 1.111 | True |
| higs_visible_only | psnr_db | -0.0483 | [-0.1716, 0.0948] | -0.17 | 5.866 | >= -0.1 | False |
| higs_visible_only | ssim | -0.0022 | [-0.0045, 0.0005] | -0.46 | 2.171 | >= -0.003 | False |
| higs_visible_only | lpips | 0.0043 | [0.0015, 0.0078] | 0.75 | 1.343 | <= 0.005 | False |
| higs_visible_only | wall_time_seconds | -14.1235 | [-58.0409, 28.2563] | -0.19 | 5.326 | - | None |
| higs_visible_only | time_to_quality_seconds | 7.0996 | [-31.8764, 43.6869] | 0.10 | 9.969 | <= 0.0 | False |
| higs_visible_only | peak_gpu_memory_mib | -291.0397 | [-619.4784, -21.8240] | -0.56 | 1.783 | - | None |
| higs_visible_only | energy_joules | -15324.9980 | [-29229.3894, -2626.5209] | -0.66 | 1.507 | - | None |
| higs_visible_only | final_gaussian_count | -169244.3636 | [-383590.4242, 2894.6364] | -0.50 | 1.990 | - | None |
| higs_visible_only | wall speedup ratio | 1.012 | CI lo 0.794 (need > 1.0) | - | - | mean >= 1.111 | False |
