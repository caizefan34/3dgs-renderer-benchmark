# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 870.6 | N/A | 1.17 | 1.45 | 0.805 | 758 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 277.5 | N/A | 3.70 | 4.52 | 0.817 | 984 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=870.6, P5=731.8, P95=1282.1
- **Latency**: mean=1.1486666666666667ms, median=1.17ms, P99=1.4533ms
- **Jitter**: 14.3%
- **Stability**: CV=0.1427, score=0.8051
- **VRAM**: peak=758MB, avg=741MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,593,376 gaussians, 376.9MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1321.0ms

### speedy_splat
- **FPS**: mean=277.5, P5=234.1, P95=414.6
- **Latency**: mean=3.603777777777778ms, median=3.6950000000000003ms, P99=4.5211ms
- **Jitter**: 15.2%
- **Stability**: CV=0.1520, score=0.8173
- **VRAM**: peak=984MB, avg=769MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,593,376 gaussians, 376.9MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1321.0ms
