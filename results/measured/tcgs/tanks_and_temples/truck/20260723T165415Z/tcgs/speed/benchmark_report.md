# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | tcgs | real_scene_speed | 302.4 | N/A | 2.22 | 13.02 | 0.170 | 1593 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### tcgs (fastest)
- **FPS**: mean=302.4, P5=139.0, P95=500.0
- **Latency**: mean=3.3063800000000003ms, median=2.22ms, P99=13.02139999999999ms
- **Jitter**: 113.6%
- **Stability**: CV=1.1359, score=0.1705
- **VRAM**: peak=1593MB, avg=1188MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3326.7ms
