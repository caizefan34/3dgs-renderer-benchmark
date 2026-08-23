# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16 | real_scene_speed | 507.9 | N/A | 1.95 | 2.79 | 0.699 | 3960 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16 (fastest)
- **FPS**: mean=507.9, P5=409.8, P95=649.6
- **Latency**: mean=1.9687599999999998ms, median=1.95ms, P99=2.7901ms
- **Jitter**: 13.1%
- **Stability**: CV=0.1307, score=0.6989
- **VRAM**: peak=3960MB, avg=3913MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6745.1ms
