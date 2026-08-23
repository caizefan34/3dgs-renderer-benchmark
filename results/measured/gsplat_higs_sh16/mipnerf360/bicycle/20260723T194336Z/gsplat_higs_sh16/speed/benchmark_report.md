# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh16 | real_scene_speed | 578.8 | N/A | 1.72 | 2.27 | 0.758 | 3965 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh16 (fastest)
- **FPS**: mean=578.8, P5=480.7, P95=709.2
- **Latency**: mean=1.7277799999999999ms, median=1.72ms, P99=2.27ms
- **Jitter**: 10.8%
- **Stability**: CV=0.1084, score=0.7577
- **VRAM**: peak=3965MB, avg=3918MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 8615.2ms
