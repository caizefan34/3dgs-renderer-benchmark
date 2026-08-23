# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16 | real_scene_speed | 645.8 | N/A | 1.52 | 1.96 | 0.775 | 1766 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16 (fastest)
- **FPS**: mean=645.8, P5=546.4, P95=740.7
- **Latency**: mean=1.5485ms, median=1.52ms, P99=1.9601999999999997ms
- **Jitter**: 9.3%
- **Stability**: CV=0.0934, score=0.7754
- **VRAM**: peak=1766MB, avg=1676MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2894.7ms
