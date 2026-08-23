# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh16 | real_scene_speed | 457.5 | N/A | 2.17 | 2.51 | 0.865 | 3890 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh16 (fastest)
- **FPS**: mean=457.5, P5=414.9, P95=497.5
- **Latency**: mean=2.186ms, median=2.17ms, P99=2.5100999999999996ms
- **Jitter**: 6.1%
- **Stability**: CV=0.0606, score=0.8645
- **VRAM**: peak=3890MB, avg=3842MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6296.0ms
