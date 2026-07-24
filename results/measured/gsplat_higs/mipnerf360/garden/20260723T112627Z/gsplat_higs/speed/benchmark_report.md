# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs | real_scene_speed | 498.4 | N/A | 1.97 | 2.53 | 0.779 | 3697 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs (fastest)
- **FPS**: mean=498.4, P5=421.9, P95=549.5
- **Latency**: mean=2.00644ms, median=1.97ms, P99=2.53ms
- **Jitter**: 8.2%
- **Stability**: CV=0.0819, score=0.7787
- **VRAM**: peak=3697MB, avg=3649MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6343.4ms
