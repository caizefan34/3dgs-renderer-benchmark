# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat | real_scene_speed | 173.4 | N/A | 5.64 | 7.14 | 0.790 | 3253 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat (fastest)
- **FPS**: mean=173.4, P5=147.9, P95=195.7
- **Latency**: mean=5.76728ms, median=5.64ms, P99=7.142599999999997ms
- **Jitter**: 8.6%
- **Stability**: CV=0.0861, score=0.7896
- **VRAM**: peak=3253MB, avg=2637MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 7042.0ms
