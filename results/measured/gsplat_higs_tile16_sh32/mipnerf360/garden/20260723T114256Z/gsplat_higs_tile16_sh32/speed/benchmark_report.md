# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh32 | real_scene_speed | 449.1 | N/A | 2.21 | 2.53 | 0.875 | 3979 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh32 (fastest)
- **FPS**: mean=449.1, P5=408.1, P95=487.8
- **Latency**: mean=2.22684ms, median=2.215ms, P99=2.5300999999999996ms
- **Jitter**: 5.8%
- **Stability**: CV=0.0581, score=0.8755
- **VRAM**: peak=3979MB, avg=3931MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6429.0ms
