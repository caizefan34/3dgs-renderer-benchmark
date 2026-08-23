# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh16 | real_scene_speed | 453.7 | N/A | 2.16 | 2.83 | 0.763 | 3890 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh16 (fastest)
- **FPS**: mean=453.7, P5=377.3, P95=502.5
- **Latency**: mean=2.2040200000000003ms, median=2.16ms, P99=2.83ms
- **Jitter**: 8.7%
- **Stability**: CV=0.0871, score=0.7633
- **VRAM**: peak=3890MB, avg=3842MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6689.9ms
