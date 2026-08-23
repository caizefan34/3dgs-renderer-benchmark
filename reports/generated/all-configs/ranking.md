# Renderer Rankings

Evidence tiers are intentionally separate. Empty tables mean that no complete, comparable matrix has been submitted.

## Tier A: Measured

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |

### Real-time ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |

### Quality ranking - PSNR

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |

### Quality ranking - SSIM

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |

### Quality ranking - LPIPS

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |

### Efficiency ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |

### Memory ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat packed/dense (gsplat_dense) | 2.146 | 262.24 | 259.85-264.66 | 3.813 | 25.834 | 0.8372 | 0.2653 | 3818 | 0.018533 |
| gsplat packed/dense (gsplat) | 1.966 | 240.27 | 238.28-242.28 | 4.162 | 25.834 | 0.8372 | 0.2653 | 4206 | 0.015414 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| Original 3DGS rasterizer (original_3dgs) | 1.000 | 122.22 | 118.82-125.78 | 8.182 | 26.120 | 0.8403 | 0.2637 | 8234 | 0.004195 |
| gsplat HiGS variants (gsplat_higs_tile16_sh16) | 5.269 | 643.97 | 613.13-676.65 | 1.553 | 25.455 | 0.8345 | 0.2778 | 8726 | 0.018372 |
| gsplat HiGS variants (gsplat_higs_tile16_sh32) | 5.212 | 636.95 | 610.72-664.63 | 1.570 | 25.828 | 0.8370 | 0.2652 | 8726 | 0.019649 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_sh32) | 5.698 | 696.44 | 670.11-724.13 | 1.436 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021475 |

### Combined Pareto ranking

| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS variants (gsplat_higs) | 5.713 | 698.19 | 662.64-735.94 | 1.432 | 25.834 | 0.8372 | 0.2649 | 6616 | 0.028496 |
| gsplat HiGS variants (gsplat_higs_auto) | 5.723 | 699.50 | 675.01-725.13 | 1.430 | 25.828 | 0.8370 | 0.2653 | 8728 | 0.021570 |
| gsplat HiGS variants (gsplat_higs_sh16) | 5.802 | 709.09 | 676.61-743.36 | 1.410 | 25.455 | 0.8345 | 0.2780 | 8728 | 0.020220 |
| gsplat HiGS variants (gsplat_higs_tile16) | 5.180 | 633.13 | 594.08-675.16 | 1.579 | 25.834 | 0.8372 | 0.2647 | 6614 | 0.025853 |
| Speedy-Splat (speedy_splat) | 2.402 | 293.59 | 287.18-300.16 | 3.406 | 26.121 | 0.8404 | 0.2637 | 4276 | 0.019411 |
| 3DGSTensorCore / TC-GS (tcgs) | 1.865 | 227.97 | 135.81-376.38 | 4.387 | 26.130 | 0.8401 | 0.2633 | 4322 | 0.014908 |

## Tier B: Reproduced

No renderer has complete suite coverage.

## Tier C: Paper Reported

No renderer has complete suite coverage.
