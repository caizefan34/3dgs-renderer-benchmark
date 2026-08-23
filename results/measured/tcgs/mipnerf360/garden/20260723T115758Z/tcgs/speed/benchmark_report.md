# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | tcgs | real_scene_speed | 170.9 | N/A | 3.79 | 14.37 | 0.264 | 3406 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### tcgs (fastest)
- **FPS**: mean=170.9, P5=105.6, P95=335.6
- **Latency**: mean=5.85202ms, median=3.79ms, P99=14.370199999999999ms
- **Jitter**: 363.9%
- **Stability**: CV=3.6385, score=0.2637
- **VRAM**: peak=3406MB, avg=2704MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 7094.4ms
