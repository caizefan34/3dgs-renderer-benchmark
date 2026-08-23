# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_dense | real_scene_speed | 191.4 | N/A | 5.12 | 6.40 | 0.800 | 3073 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_dense (fastest)
- **FPS**: mean=191.4, P5=170.1, P95=208.8
- **Latency**: mean=5.223319999999999ms, median=5.12ms, P99=6.4001ms
- **Jitter**: 6.7%
- **Stability**: CV=0.0669, score=0.8000
- **VRAM**: peak=3073MB, avg=2637MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6347.3ms
