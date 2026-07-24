# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_dense | real_scene_speed | 309.4 | N/A | 3.06 | 4.71 | 0.650 | 615 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_dense (fastest)
- **FPS**: mean=309.4, P5=229.1, P95=386.2
- **Latency**: mean=3.23154ms, median=3.06ms, P99=4.7101ms
- **Jitter**: 17.0%
- **Stability**: CV=0.1695, score=0.6497
- **VRAM**: peak=615MB, avg=472MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1822.6ms
