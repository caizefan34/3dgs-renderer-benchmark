# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | tcgs | real_scene_speed | 358.3 | N/A | 1.53 | 14.25 | 0.107 | 818 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### tcgs (fastest)
- **FPS**: mean=358.3, P5=112.7, P95=729.9
- **Latency**: mean=2.79078ms, median=1.53ms, P99=14.252599999999989ms
- **Jitter**: 94.7%
- **Stability**: CV=0.9467, score=0.1073
- **VRAM**: peak=818MB, avg=583MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1628.0ms
