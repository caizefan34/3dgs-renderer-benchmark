# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh32 | real_scene_speed | 724.4 | N/A | 1.38 | 1.53 | 0.902 | 1754 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh32 (fastest)
- **FPS**: mean=724.4, P5=662.3, P95=806.5
- **Latency**: mean=1.3805399999999999ms, median=1.38ms, P99=1.5301ms
- **Jitter**: 5.7%
- **Stability**: CV=0.0569, score=0.9019
- **VRAM**: peak=1754MB, avg=1706MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3135.7ms
