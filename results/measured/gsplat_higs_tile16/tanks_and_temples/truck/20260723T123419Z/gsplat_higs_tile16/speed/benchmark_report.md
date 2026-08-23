# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16 | real_scene_speed | 652.1 | N/A | 1.51 | 1.91 | 0.791 | 1766 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16 (fastest)
- **FPS**: mean=652.1, P5=552.5, P95=751.9
- **Latency**: mean=1.5334199999999998ms, median=1.51ms, P99=1.9101ms
- **Jitter**: 9.1%
- **Stability**: CV=0.0906, score=0.7905
- **VRAM**: peak=1766MB, avg=1676MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2842.1ms
