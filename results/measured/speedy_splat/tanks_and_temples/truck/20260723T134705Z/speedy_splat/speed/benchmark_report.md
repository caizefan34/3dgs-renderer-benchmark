# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | speedy_splat | real_scene_speed | 306.0 | N/A | 3.20 | 4.50 | 0.711 | 1585 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### speedy_splat (fastest)
- **FPS**: mean=306.0, P5=245.7, P95=357.1
- **Latency**: mean=3.2679ms, median=3.2ms, P99=4.5005999999999995ms
- **Jitter**: 12.4%
- **Stability**: CV=0.1241, score=0.7110
- **VRAM**: peak=1585MB, avg=1188MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2974.7ms
