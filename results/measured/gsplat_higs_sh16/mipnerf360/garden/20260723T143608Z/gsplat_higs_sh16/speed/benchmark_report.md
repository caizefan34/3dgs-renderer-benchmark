# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_sh16 | real_scene_speed | 509.6 | N/A | 1.93 | 2.47 | 0.781 | 3787 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_sh16 (fastest)
- **FPS**: mean=509.6, P5=429.1, P95=558.7
- **Latency**: mean=1.9624799999999998ms, median=1.93ms, P99=2.4702ms
- **Jitter**: 8.1%
- **Stability**: CV=0.0808, score=0.7813
- **VRAM**: peak=3787MB, avg=3739MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6146.1ms
