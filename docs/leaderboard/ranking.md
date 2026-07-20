# Renderer Rankings

Evidence tiers are intentionally separate. Empty tables mean that no complete, comparable matrix has been submitted.

## Tier A: Measured

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |

### Real-time ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |

### Quality ranking - PSNR

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |

### Quality ranking - SSIM

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |

### Quality ranking - LPIPS

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |

### Efficiency ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |

### Memory ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat packed/dense (gsplat) | 1.966 | 241.60 | 240.62-242.58 | 4.139 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015499 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.88 | 120.70-125.11 | 8.138 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004218 |

### Combined Pareto ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS variants (gsplat_higs) | 5.671 | 696.91 | 656.72-739.89 | 1.435 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028444 |
| Speedy-Splat (speedy_splat) | 2.385 | 293.03 | 283.98-302.40 | 3.413 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019373 |
| 3DGSTensorCore / TC-GS (tcgs) | 2.048 | 251.62 | 153.09-395.78 | 3.974 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.016455 |

## Tier B: Reproduced

No renderer has complete suite coverage.

## Tier C: Paper Reported

No renderer has complete suite coverage.
