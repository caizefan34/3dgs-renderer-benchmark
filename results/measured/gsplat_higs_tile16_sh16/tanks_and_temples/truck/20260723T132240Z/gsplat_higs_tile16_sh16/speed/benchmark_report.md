# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh16 | real_scene_speed | 653.5 | N/A | 1.50 | 1.91 | 0.785 | 1804 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh16 (fastest)
- **FPS**: mean=653.5, P5=552.5, P95=752.2
- **Latency**: mean=1.53016ms, median=1.5ms, P99=1.9101ms
- **Jitter**: 9.2%
- **Stability**: CV=0.0925, score=0.7853
- **VRAM**: peak=1804MB, avg=1715MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3016.0ms
