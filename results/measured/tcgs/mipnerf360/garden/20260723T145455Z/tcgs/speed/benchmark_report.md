# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | tcgs | real_scene_speed | 249.7 | N/A | 3.22 | 11.32 | 0.284 | 3406 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### tcgs (fastest)
- **FPS**: mean=249.7, P5=143.9, P95=342.5
- **Latency**: mean=4.00468ms, median=3.22ms, P99=11.322999999999988ms
- **Jitter**: 42.6%
- **Stability**: CV=0.4256, score=0.2844
- **VRAM**: peak=3406MB, avg=2704MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 5,834,784 gaussians, 1380.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 7583.5ms
