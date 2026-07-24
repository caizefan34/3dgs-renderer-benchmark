# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh32 | real_scene_speed | 565.2 | N/A | 1.75 | 2.51 | 0.697 | 4059 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh32 (fastest)
- **FPS**: mean=565.2, P5=465.0, P95=689.7
- **Latency**: mean=1.7693ms, median=1.75ms, P99=2.51ms
- **Jitter**: 11.3%
- **Stability**: CV=0.1128, score=0.6972
- **VRAM**: peak=4059MB, avg=4011MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 6,131,954 gaussians, 1450.3MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6654.1ms
