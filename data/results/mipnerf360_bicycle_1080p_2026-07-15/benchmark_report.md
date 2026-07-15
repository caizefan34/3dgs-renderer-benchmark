# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 245.9 | N/A | 3.77 | 7.70 | 0.489 | 2776 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 56.3 | N/A | 17.20 | 24.27 | 0.709 | 3589 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=245.9, P5=165.3, P95=331.7
- **Latency**: mean=4.067111111111111ms, median=3.77ms, P99=7.704899999999998ms
- **Jitter**: 28.3%
- **Stability**: CV=0.2829, score=0.4893
- **VRAM**: peak=2776MB, avg=2724MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6054.4ms

### speedy_splat
- **FPS**: mean=56.3, P5=43.7, P95=68.6
- **Latency**: mean=17.752333333333333ms, median=17.205ms, P99=24.2665ms
- **Jitter**: 14.1%
- **Stability**: CV=0.1411, score=0.7090
- **VRAM**: peak=3589MB, avg=2866MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6054.4ms
