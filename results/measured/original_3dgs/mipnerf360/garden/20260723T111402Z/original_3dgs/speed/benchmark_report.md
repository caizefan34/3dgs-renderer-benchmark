# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | original_3dgs | real_scene_speed | 109.5 | N/A | 9.10 | 10.46 | 0.870 | 3657 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### original_3dgs (fastest)
- **FPS**: mean=109.5, P5=97.1, P95=120.5
- **Latency**: mean=9.1322ms, median=9.1ms, P99=10.4603ms
- **Jitter**: 6.8%
- **Stability**: CV=0.0676, score=0.8700
- **VRAM**: peak=3657MB, avg=2681MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6686.6ms
