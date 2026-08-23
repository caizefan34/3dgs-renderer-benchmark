# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 502.4 | N/A | 1.98 | 2.29 | 0.865 | 3876 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=502.4, P5=458.7, P95=543.5
- **Latency**: mean=1.99044ms, median=1.98ms, P99=2.29ms
- **Jitter**: 5.9%
- **Stability**: CV=0.0589, score=0.8646
- **VRAM**: peak=3876MB, avg=3828MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6665.9ms
