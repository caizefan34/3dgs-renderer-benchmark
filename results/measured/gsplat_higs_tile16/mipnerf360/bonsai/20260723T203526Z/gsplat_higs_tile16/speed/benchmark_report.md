# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16 | real_scene_speed | 1007.2 | N/A | 0.99 | 1.25 | 0.792 | 880 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16 (fastest)
- **FPS**: mean=1007.2, P5=826.4, P95=1162.8
- **Latency**: mean=0.99286ms, median=0.99ms, P99=1.25ms
- **Jitter**: 10.0%
- **Stability**: CV=0.0997, score=0.7920
- **VRAM**: peak=880MB, avg=832MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1714.2ms
