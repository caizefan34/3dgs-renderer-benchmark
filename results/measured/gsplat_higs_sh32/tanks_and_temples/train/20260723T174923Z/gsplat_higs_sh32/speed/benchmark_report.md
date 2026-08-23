# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh32 | real_scene_speed | 827.8 | N/A | 1.20 | 1.60 | 0.750 | 782 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh32 (fastest)
- **FPS**: mean=827.8, P5=680.0, P95=1010.1
- **Latency**: mean=1.208ms, median=1.2ms, P99=1.6ms
- **Jitter**: 12.1%
- **Stability**: CV=0.1210, score=0.7500
- **VRAM**: peak=782MB, avg=733MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1289.0ms
