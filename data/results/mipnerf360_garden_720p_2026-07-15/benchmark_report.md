# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 258.7 | N/A | 3.81 | 4.65 | 0.820 | 2591 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 71.2 | N/A | 13.69 | 16.63 | 0.823 | 3363 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=258.7, P5=223.1, P95=297.8
- **Latency**: mean=3.8656666666666655ms, median=3.815ms, P99=4.6497ms
- **Jitter**: 9.1%
- **Stability**: CV=0.0911, score=0.8205
- **VRAM**: peak=2591MB, avg=2586MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5704.4ms

### speedy_splat
- **FPS**: mean=71.2, P5=61.3, P95=81.5
- **Latency**: mean=14.04088888888889ms, median=13.69ms, P99=16.6307ms
- **Jitter**: 9.0%
- **Stability**: CV=0.0901, score=0.8232
- **VRAM**: peak=3363MB, avg=2728MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5704.4ms
