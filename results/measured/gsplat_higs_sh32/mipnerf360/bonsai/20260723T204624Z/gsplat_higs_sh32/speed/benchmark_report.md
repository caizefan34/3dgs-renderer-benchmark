# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh32 | real_scene_speed | 1077.1 | N/A | 0.90 | 1.18 | 0.763 | 897 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh32 (fastest)
- **FPS**: mean=1077.1, P5=909.1, P95=1234.6
- **Latency**: mean=0.92838ms, median=0.9ms, P99=1.1802999999999997ms
- **Jitter**: 10.4%
- **Stability**: CV=0.1037, score=0.7625
- **VRAM**: peak=897MB, avg=850MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1645.4ms
