# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | tcgs | real_scene_speed | 229.6 | N/A | 4.40 | 5.26 | 0.836 | 780 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### tcgs (fastest)
- **FPS**: mean=229.6, P5=201.6, P95=275.5
- **Latency**: mean=4.35626ms, median=4.4ms, P99=5.260199999999999ms
- **Jitter**: 9.3%
- **Stability**: CV=0.0932, score=0.8365
- **VRAM**: peak=780MB, avg=508MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 885.5ms
