# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh32 | real_scene_speed | 999.6 | N/A | 0.99 | 1.26 | 0.786 | 919 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh32 (fastest)
- **FPS**: mean=999.6, P5=833.3, P95=1162.8
- **Latency**: mean=1.00044ms, median=0.99ms, P99=1.26ms
- **Jitter**: 9.9%
- **Stability**: CV=0.0986, score=0.7857
- **VRAM**: peak=919MB, avg=870MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2889.9ms
