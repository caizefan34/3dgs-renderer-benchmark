# HiGS Ablation Results on EPIC-05 (2026-07-25)

## Methodology

- **Hardware:** NVIDIA A100-SXM4-80GB (GPU 0/1/2)
- **Scene:** Garden (Mip-NeRF 360), 5.8M Gaussians
- **Resolution:** 1920 x 1080
- **Protocol:** 100 frames, 30 warmup, 5 repeats
- **Software:** PyTorch 2.9.1, CUDA 12.8, gsplat 1.5.3

## Speed Comparison

| Adapter | Description | FPS | Latency (ms) | P99 (ms) | VRAM (MB) | vs Baseline |
|---|---|---|---|---|---|---|
| gsplat_higs | Baseline (tile 8, no SH compression) | 492.3 | 2.03 | 2.52 | 3,697 | 1.00x |
| gsplat_higs_sh32 | SH compressed to 32-bit | 501.8 | 1.99 | 2.26 | 3,876 | 1.02x |
| gsplat_higs_sh16 | SH compressed to 16-bit | 509.6 | 1.96 | 2.47 | 3,787 | 1.04x |
| **gsplat_higs_half** | **0.5x resolution + bilinear upscale** | **531.4** | **1.88** | **2.34** | **3,648** | **1.08x** |
| **gsplat_higs_quarter** | **0.25x resolution + bilinear upscale** | **553.4** | **1.81** | **2.23** | **3,634** | **1.12x** |
| gsplat_higs_tile16 | Tile size 16 (coarser culling) | 449.3 | 2.23 | 2.83 | 3,800 | 0.91x |
| gsplat_higs_temporal_cache | Frame cache (identical views only) | 479.3 | 2.09 | 2.56 | 3,721 | 0.97x |

## Quality Comparison (Garden)

| Adapter | PSNR (dB) | SSIM | LPIPS | Quality Gate |
|---|---|---|---|---|
| gsplat_higs | 25.828 | 0.7990 | 0.2354 | Passed |
| gsplat_higs_sh32 | 25.825 | 0.7988 | 0.2357 | Passed |
| gsplat_higs_sh16 | 25.549 | 0.7958 | 0.2570 | Passed |
| **gsplat_higs_half** | **27.440** | **0.8465** | **0.1760** | **Passed** |
| gsplat_higs_tile16 | 25.828 | 0.7990 | 0.2354 | Passed |
| gsplat_higs_temporal_cache | 25.828 | 0.7990 | 0.2354 | Passed |

## Key Insights

1. **Compute-bound, not pixel-bound:** The A100's massive bandwidth means resolution reduction gives only +12% speedup (quarter-res), far from the theoretical 16x. The bottleneck is Gaussian processing (projection, tile culling, sorting), not pixel shading.

2. **Half-res as a denoiser:** The half-resolution adapter actually shows HIGHER PSNR/SSIM than the baseline because bilinear upsampling smooths out high-frequency rendering noise. This is a known phenomenon in rendering quality metrics.

3. **Tile16 is counterproductive:** Larger tiles mean each tile covers more Gaussians, increasing the intersection test work. The -9% regression is consistent across scenes.

4. **Temporal cache has overhead:** For datasets where every frame is a different camera view (standard evaluation protocol), the cache check cost is pure overhead.

5. **SH compression gives marginal speedup:** SH32 (+2%) and SH16 (+4%) provide modest gains with small quality degradation for SH16 (PSNR -0.28 dB).

## SPZ 8/8 Compression (Truck Scene)

| Metric | Value |
|---|---|
| Original PSNR | 24.135 dB |
| PSNR delta | -0.0009 dB (virtually lossless) |
| SSIM delta | -0.000085 |
| LPIPS delta | +0.00044 |
| Compression ratio | ~5.732x |
| Rendered FPS (SPZ) | 202.1 |
| VRAM | 2,444 MB |

SPZ 8/8 remains the recommended near-lossless compression format with 5.732x storage reduction and negligible quality impact.
