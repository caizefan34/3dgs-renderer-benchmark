# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh32 | real_scene_speed | 727.9 | N/A | 1.37 | 1.54 | 0.890 | 1754 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh32 (fastest)
- **FPS**: mean=727.9, P5=666.7, P95=813.0
- **Latency**: mean=1.37378ms, median=1.37ms, P99=1.54ms
- **Jitter**: 5.8%
- **Stability**: CV=0.0578, score=0.8896
- **VRAM**: peak=1754MB, avg=1706MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3714.0ms
