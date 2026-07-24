# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat | real_scene_speed | 190.3 | N/A | 5.17 | 7.42 | 0.697 | 3309 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat (fastest)
- **FPS**: mean=190.3, P5=140.6, P95=243.3
- **Latency**: mean=5.25584ms, median=5.175ms, P99=7.424099999999997ms
- **Jitter**: 15.3%
- **Stability**: CV=0.1531, score=0.6971
- **VRAM**: peak=3309MB, avg=2771MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6972.4ms
