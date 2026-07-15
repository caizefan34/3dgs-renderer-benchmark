# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 318.9 | N/A | 3.14 | 3.85 | 0.817 | 916 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 89.7 | N/A | 11.27 | 12.80 | 0.881 | 1350 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=318.9, P5=291.5, P95=399.3
- **Latency**: mean=3.1356666666666673ms, median=3.14ms, P99=3.8455ms
- **Jitter**: 8.8%
- **Stability**: CV=0.0880, score=0.8165
- **VRAM**: peak=916MB, avg=868MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,593,376 gaussians, 376.9MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1291.4ms

### speedy_splat
- **FPS**: mean=89.7, P5=79.3, P95=122.4
- **Latency**: mean=11.142111111111111ms, median=11.27ms, P99=12.7965ms
- **Jitter**: 11.2%
- **Stability**: CV=0.1118, score=0.8807
- **VRAM**: peak=1350MB, avg=769MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,593,376 gaussians, 376.9MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1291.4ms
