# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 489.8 | N/A | 2.05 | 2.80 | 0.732 | 786 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 176.4 | N/A | 5.44 | 9.05 | 0.601 | 1047 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=489.8, P5=422.7, P95=675.4
- **Latency**: mean=2.041555555555555ms, median=2.05ms, P99=2.801ms
- **Jitter**: 13.5%
- **Stability**: CV=0.1351, score=0.7319
- **VRAM**: peak=786MB, avg=761MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,593,376 gaussians, 376.9MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1299.9ms

### speedy_splat
- **FPS**: mean=176.4, P5=120.5, P95=302.9
- **Latency**: mean=5.668777777777779ms, median=5.4399999999999995ms, P99=9.0472ms
- **Jitter**: 25.0%
- **Stability**: CV=0.2502, score=0.6013
- **VRAM**: peak=1047MB, avg=769MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,593,376 gaussians, 376.9MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1299.9ms
