# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh16 | real_scene_speed | 715.5 | N/A | 1.37 | 1.76 | 0.778 | 1717 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh16 (fastest)
- **FPS**: mean=715.5, P5=602.4, P95=819.7
- **Latency**: mean=1.3976ms, median=1.37ms, P99=1.76ms
- **Jitter**: 9.2%
- **Stability**: CV=0.0917, score=0.7784
- **VRAM**: peak=1717MB, avg=1668MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2921.8ms
