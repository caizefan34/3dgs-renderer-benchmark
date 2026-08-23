# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs | real_scene_speed | 578.6 | N/A | 1.72 | 2.21 | 0.778 | 3872 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs (fastest)
- **FPS**: mean=578.6, P5=473.8, P95=709.2
- **Latency**: mean=1.72844ms, median=1.72ms, P99=2.210899999999999ms
- **Jitter**: 10.7%
- **Stability**: CV=0.1070, score=0.7780
- **VRAM**: peak=3872MB, avg=3824MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 7377.3ms
