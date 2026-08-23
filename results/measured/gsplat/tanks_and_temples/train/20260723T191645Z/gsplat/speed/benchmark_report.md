# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat | real_scene_speed | 268.8 | N/A | 3.60 | 5.18 | 0.696 | 704 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat (fastest)
- **FPS**: mean=268.8, P5=201.2, P95=350.9
- **Latency**: mean=3.7206000000000006ms, median=3.605ms, P99=5.18ms
- **Jitter**: 16.6%
- **Stability**: CV=0.1662, score=0.6959
- **VRAM**: peak=704MB, avg=472MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1534.5ms
