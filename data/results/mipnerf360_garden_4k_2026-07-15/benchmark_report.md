# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 139.9 | N/A | 7.11 | 8.10 | 0.877 | 2759 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 38.7 | N/A | 25.37 | 30.79 | 0.824 | 3707 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=139.9, P5=126.8, P95=155.0
- **Latency**: mean=7.146777777777779ms, median=7.105ms, P99=8.0977ms
- **Jitter**: 6.4%
- **Stability**: CV=0.0635, score=0.8774
- **VRAM**: peak=2759MB, avg=2712MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5303.0ms

### speedy_splat
- **FPS**: mean=38.7, P5=33.2, P95=42.9
- **Latency**: mean=25.84ms, median=25.365000000000002ms, P99=30.7899ms
- **Jitter**: 8.0%
- **Stability**: CV=0.0802, score=0.8238
- **VRAM**: peak=3707MB, avg=2728MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5303.0ms
