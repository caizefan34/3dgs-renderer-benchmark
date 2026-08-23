# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_dense | real_scene_speed | 423.3 | N/A | 2.32 | 2.97 | 0.781 | 691 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_dense (fastest)
- **FPS**: mean=423.3, P5=346.0, P95=467.3
- **Latency**: mean=2.3625ms, median=2.32ms, P99=2.97ms
- **Jitter**: 8.2%
- **Stability**: CV=0.0821, score=0.7811
- **VRAM**: peak=691MB, avg=569MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1750.0ms
