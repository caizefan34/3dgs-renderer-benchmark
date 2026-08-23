# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_dense | real_scene_speed | 190.2 | N/A | 5.20 | 6.91 | 0.752 | 3227 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_dense (fastest)
- **FPS**: mean=190.2, P5=154.6, P95=226.8
- **Latency**: mean=5.25782ms, median=5.2ms, P99=6.911799999999999ms
- **Jitter**: 10.6%
- **Stability**: CV=0.1060, score=0.7523
- **VRAM**: peak=3227MB, avg=2771MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 7849.0ms
