# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh32 | real_scene_speed | 658.5 | N/A | 1.50 | 1.89 | 0.794 | 1844 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh32 (fastest)
- **FPS**: mean=658.5, P5=578.0, P95=740.7
- **Latency**: mean=1.5185799999999998ms, median=1.5ms, P99=1.8901999999999997ms
- **Jitter**: 7.5%
- **Stability**: CV=0.0748, score=0.7936
- **VRAM**: peak=1844MB, avg=1754MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3063.7ms
