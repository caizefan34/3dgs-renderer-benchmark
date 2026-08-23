# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | speedy_splat | real_scene_speed | 453.6 | N/A | 2.16 | 2.90 | 0.743 | 811 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### speedy_splat (fastest)
- **FPS**: mean=453.6, P5=378.7, P95=510.2
- **Latency**: mean=2.2045ms, median=2.1550000000000002ms, P99=2.900999999999999ms
- **Jitter**: 11.1%
- **Stability**: CV=0.1107, score=0.7428
- **VRAM**: peak=811MB, avg=583MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1863.5ms
