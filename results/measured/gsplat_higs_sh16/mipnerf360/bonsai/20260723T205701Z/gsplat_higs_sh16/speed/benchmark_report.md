# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh16 | real_scene_speed | 1102.8 | N/A | 0.88 | 1.22 | 0.721 | 878 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh16 (fastest)
- **FPS**: mean=1102.8, P5=934.6, P95=1250.0
- **Latency**: mean=0.90676ms, median=0.88ms, P99=1.22ms
- **Jitter**: 10.4%
- **Stability**: CV=0.1035, score=0.7213
- **VRAM**: peak=878MB, avg=831MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1700.8ms
