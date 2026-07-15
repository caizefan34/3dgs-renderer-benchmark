# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 255.6 | N/A | 3.86 | 5.00 | 0.772 | 2744 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 76.2 | N/A | 12.72 | 16.56 | 0.768 | 3522 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=255.6, P5=214.3, P95=299.0
- **Latency**: mean=3.9128888888888893ms, median=3.86ms, P99=4.998799999999999ms
- **Jitter**: 12.0%
- **Stability**: CV=0.1201, score=0.7722
- **VRAM**: peak=2744MB, avg=2701MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5632.9ms

### speedy_splat
- **FPS**: mean=76.2, P5=62.4, P95=91.7
- **Latency**: mean=13.131222222222224ms, median=12.72ms, P99=16.5639ms
- **Jitter**: 12.3%
- **Stability**: CV=0.1232, score=0.7679
- **VRAM**: peak=3522MB, avg=2866MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5632.9ms
