# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh16 | real_scene_speed | 727.7 | N/A | 1.36 | 1.72 | 0.791 | 1717 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh16 (fastest)
- **FPS**: mean=727.7, P5=649.4, P95=826.4
- **Latency**: mean=1.37422ms, median=1.36ms, P99=1.72ms
- **Jitter**: 7.4%
- **Stability**: CV=0.0739, score=0.7907
- **VRAM**: peak=1717MB, avg=1668MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 2844.1ms
