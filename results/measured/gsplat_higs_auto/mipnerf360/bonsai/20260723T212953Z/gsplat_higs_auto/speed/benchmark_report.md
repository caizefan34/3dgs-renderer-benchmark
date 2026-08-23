# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 1090.9 | N/A | 0.90 | 1.17 | 0.765 | 897 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=1090.9, P5=917.0, P95=1234.6
- **Latency**: mean=0.9166400000000001ms, median=0.895ms, P99=1.1701ms
- **Jitter**: 9.8%
- **Stability**: CV=0.0982, score=0.7649
- **VRAM**: peak=897MB, avg=850MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2015.7ms
