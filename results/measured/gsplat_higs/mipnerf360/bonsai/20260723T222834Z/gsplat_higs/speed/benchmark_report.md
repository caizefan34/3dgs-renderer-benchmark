# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs | real_scene_speed | 1101.3 | N/A | 0.89 | 1.16 | 0.767 | 860 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs (fastest)
- **FPS**: mean=1101.3, P5=934.6, P95=1250.0
- **Latency**: mean=0.90804ms, median=0.89ms, P99=1.16ms
- **Jitter**: 9.4%
- **Stability**: CV=0.0940, score=0.7672
- **VRAM**: peak=860MB, avg=813MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2030.1ms
