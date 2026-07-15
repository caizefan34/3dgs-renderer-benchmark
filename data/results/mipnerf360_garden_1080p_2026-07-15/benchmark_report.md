# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 236.5 | N/A | 4.00 | 7.01 | 0.571 | 2620 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 61.9 | N/A | 16.01 | 22.24 | 0.720 | 3423 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=236.5, P5=182.8, P95=282.2
- **Latency**: mean=4.227555555555554ms, median=4.005ms, P99=7.0128999999999975ms
- **Jitter**: 21.5%
- **Stability**: CV=0.2153, score=0.5711
- **VRAM**: peak=2620MB, avg=2608MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5826.7ms

### speedy_splat
- **FPS**: mean=61.9, P5=48.5, P95=87.0
- **Latency**: mean=16.15611111111111ms, median=16.009999999999998ms, P99=22.2444ms
- **Jitter**: 16.9%
- **Stability**: CV=0.1694, score=0.7197
- **VRAM**: peak=3423MB, avg=2728MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5826.7ms
