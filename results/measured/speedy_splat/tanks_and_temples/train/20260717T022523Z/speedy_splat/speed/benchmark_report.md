# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | speedy_splat | real_scene_speed | 175.7 | N/A | 5.69 | 7.33 | 0.776 | 772 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### speedy_splat (fastest)
- **FPS**: mean=175.7, P5=145.6, P95=219.8
- **Latency**: mean=5.6923ms, median=5.69ms, P99=7.3301ms
- **Jitter**: 12.8%
- **Stability**: CV=0.1282, score=0.7763
- **VRAM**: peak=772MB, avg=508MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 874.5ms
