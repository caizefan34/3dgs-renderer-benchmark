# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | original_3dgs | real_scene_speed | 87.6 | N/A | 11.27 | 15.93 | 0.707 | 4054 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### original_3dgs (fastest)
- **FPS**: mean=87.6, P5=65.0, P95=146.8
- **Latency**: mean=11.409920000000001ms, median=11.265ms, P99=15.9302ms
- **Jitter**: 20.2%
- **Stability**: CV=0.2019, score=0.7071
- **VRAM**: peak=4054MB, avg=2817MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6635.9ms
