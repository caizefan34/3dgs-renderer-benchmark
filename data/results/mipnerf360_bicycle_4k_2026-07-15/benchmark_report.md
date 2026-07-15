# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_auto | real_scene_speed | 124.8 | N/A | 5.22 | 98.82 | 0.053 | 2933 | N/A | N/A | N/A | N/A |
| 2 | speedy_splat | real_scene_speed | 39.7 | N/A | 24.34 | 33.28 | 0.731 | 3885 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_auto (fastest)
- **FPS**: mean=124.8, P5=128.6, P95=219.8
- **Latency**: mean=8.013666666666666ms, median=5.220000000000001ms, P99=98.81599999999999ms
- **Jitter**: 189.5%
- **Stability**: CV=1.8945, score=0.0528
- **VRAM**: peak=2933MB, avg=2837MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5675.7ms

### speedy_splat
- **FPS**: mean=39.7, P5=32.6, P95=46.5
- **Latency**: mean=25.217555555555556ms, median=24.345ms, P99=33.2842ms
- **Jitter**: 11.8%
- **Stability**: CV=0.1177, score=0.7314
- **VRAM**: peak=3885MB, avg=2866MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 5675.7ms
