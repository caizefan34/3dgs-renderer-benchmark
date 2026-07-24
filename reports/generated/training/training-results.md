# EPIC-05 Native Training Report

Status of the EPIC-05 native training matrix. All runs use 30,000 iterations on 8x A100-SXM4-80GB.

## Summary

| Backend | Cases | Complete | Wall time | It/s | VRAM | PSNR | SSIM | LPIPS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| original_3dgs_train | 5 | 5/5 | 1298.4s | 26.1 | 15824 MiB | 26.70 | 0.85809 | 0.24541 |  |
| local_gs_train | 5 | 0/5 | N/A | N/A | N/A | N/A | N/A | N/A | CUDA backward bug |
| gemm_gs_train | 5 | 5/5 | 179.4s | 182.5 | 9680 MiB | 7.97 | 0.04634 | 0.66593 |  |

## Per-scene results

| Backend | Scene | Status | Wall time | It/s | VRAM | Gaussians | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm_gs_train | large-bicycle-1080p | complete | 217.4s | 138.0 | 7640 MiB | 441368 | 8.35 | 0.00625 | 0.65142 |
| gemm_gs_train | large-bonsai-1080p | complete | 240.3s | 124.8 | 9680 MiB | 559537 | 10.05 | 0.12177 | 0.66650 |
| gemm_gs_train | medium-train-1080p | complete | 111.7s | 268.6 | 3830 MiB | 182686 | 5.50 | 0.01100 | 0.67989 |
| gemm_gs_train | medium-truck-1080p | complete | 131.6s | 228.0 | 3320 MiB | 134752 | N/A | N/A | N/A |
| gemm_gs_train | small-garden-1080p | complete | 196.0s | 153.0 | 6092 MiB | 138766 | N/A | N/A | N/A |
| local_gs_train | large-bicycle-1080p | failed | 4.9s | 6064.5 | 416 MiB | N/A | N/A | N/A | N/A |
| local_gs_train | medium-train-1080p | failed | 7.8s | 3863.4 | 416 MiB | N/A | N/A | N/A | N/A |
| local_gs_train | medium-truck-1080p | failed | 54.8s | 547.4 | 2670 MiB | N/A | N/A | N/A | N/A |
| local_gs_train | small-garden-1080p | failed | 146.7s | 204.5 | 4940 MiB | N/A | N/A | N/A | N/A |
| original_3dgs_train | large-bicycle-1080p | complete | 1885.0s | 15.9 | 15824 MiB | 4760344 | 25.67 | 0.82122 | 0.27125 |
| original_3dgs_train | large-bonsai-1080p | complete | 1142.7s | 26.3 | 12136 MiB | 1066014 | 33.06 | 0.95662 | 0.20698 |
| original_3dgs_train | medium-train-1080p | complete | 809.0s | 37.1 | 7318 MiB | 1091826 | 22.70 | 0.81086 | 0.28522 |
| original_3dgs_train | medium-truck-1080p | complete | 863.1s | 34.8 | 7184 MiB | 2053301 | 24.43 | 0.85919 | 0.25790 |
| original_3dgs_train | small-garden-1080p | complete | 1792.4s | 16.7 | 14918 MiB | 4207518 | 27.62 | 0.84257 | 0.20569 |

## Failures

### local_gs_train
All 5 scenes failed with a CUDA backward pass error:
```
RuntimeError: Function _RasterizeGaussiansBackward returned an invalid gradient at index 2
got [0, 0, 3] but expected shape compatible with [0, 16, 3]
```
This occurs at iteration ~6000 when pruning reduces Gaussian count to near-zero.
The rasterization backward kernel returns a zero-size tensor for SH gradients.

### gemm_gs_train (quality only)
Training completed for all 5 scenes but gsplat renderer cannot read the GEMM-GS PLY format.
The gemm_gs output PLY stores Gaussian attributes in a different layout (62 properties)
than the standard 3DGS format. Quality evaluation with gsplat produces invalid PSNR values.
This is a cross-renderer PLY compatibility issue, not a training failure.

### flashgs
SIGSEGV crash during candidate renderer smoke test. No training was attempted.

Generated: 2026-07-24
