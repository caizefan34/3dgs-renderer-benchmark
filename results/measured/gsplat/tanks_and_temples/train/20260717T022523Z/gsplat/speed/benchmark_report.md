# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat | real_scene_speed | 153.5 | N/A | 6.64 | 8.18 | 0.812 | 728 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat (fastest)
- **FPS**: mean=153.5, P5=126.9, P95=199.7
- **Latency**: mean=6.51666ms, median=6.64ms, P99=8.18ms
- **Jitter**: 13.5%
- **Stability**: CV=0.1355, score=0.8117
- **VRAM**: peak=728MB, avg=496MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 961.1ms
