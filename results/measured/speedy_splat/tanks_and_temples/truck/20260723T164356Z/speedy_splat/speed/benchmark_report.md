# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | speedy_splat | real_scene_speed | 306.0 | N/A | 3.23 | 4.45 | 0.726 | 1585 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### speedy_splat (fastest)
- **FPS**: mean=306.0, P5=250.0, P95=355.9
- **Latency**: mean=3.2681000000000004ms, median=3.23ms, P99=4.4502ms
- **Jitter**: 10.7%
- **Stability**: CV=0.1073, score=0.7258
- **VRAM**: peak=1585MB, avg=1188MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2922.5ms
