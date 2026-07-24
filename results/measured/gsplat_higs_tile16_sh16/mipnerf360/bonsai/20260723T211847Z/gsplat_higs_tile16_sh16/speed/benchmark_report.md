# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh16 | real_scene_speed | 1007.9 | N/A | 0.98 | 1.25 | 0.784 | 899 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh16 (fastest)
- **FPS**: mean=1007.9, P5=833.3, P95=1176.5
- **Latency**: mean=0.9921800000000001ms, median=0.98ms, P99=1.2501ms
- **Jitter**: 10.2%
- **Stability**: CV=0.1017, score=0.7839
- **VRAM**: peak=899MB, avg=851MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1879.3ms
