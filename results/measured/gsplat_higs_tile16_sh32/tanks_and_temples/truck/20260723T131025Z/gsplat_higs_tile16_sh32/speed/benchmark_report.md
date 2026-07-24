# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16_sh32 | real_scene_speed | 650.8 | N/A | 1.51 | 1.94 | 0.778 | 1844 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16_sh32 (fastest)
- **FPS**: mean=650.8, P5=546.3, P95=741.0
- **Latency**: mean=1.53668ms, median=1.51ms, P99=1.9401999999999997ms
- **Jitter**: 9.0%
- **Stability**: CV=0.0900, score=0.7783
- **VRAM**: peak=1844MB, avg=1754MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3364.8ms
