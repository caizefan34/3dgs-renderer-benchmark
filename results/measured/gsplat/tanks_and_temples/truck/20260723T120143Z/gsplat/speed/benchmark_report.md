# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat | real_scene_speed | 245.7 | N/A | 4.03 | 5.39 | 0.748 | 1449 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat (fastest)
- **FPS**: mean=245.7, P5=202.0, P95=292.4
- **Latency**: mean=4.069559999999999ms, median=4.03ms, P99=5.390199999999999ms
- **Jitter**: 11.4%
- **Stability**: CV=0.1137, score=0.7477
- **VRAM**: peak=1449MB, avg=1157MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3652.4ms
