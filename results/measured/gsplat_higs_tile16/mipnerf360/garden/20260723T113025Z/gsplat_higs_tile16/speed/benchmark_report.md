# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs_tile16 | real_scene_speed | 447.9 | N/A | 2.20 | 2.85 | 0.772 | 3800 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs_tile16 (fastest)
- **FPS**: mean=447.9, P5=386.1, P95=495.0
- **Latency**: mean=2.23246ms, median=2.2ms, P99=2.8501ms
- **Jitter**: 8.2%
- **Stability**: CV=0.0823, score=0.7719
- **VRAM**: peak=3800MB, avg=3752MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 6472.9ms
