# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | original_3dgs | real_scene_speed | 57.9 | N/A | 16.92 | 22.89 | 0.739 | 1254 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### original_3dgs (fastest)
- **FPS**: mean=57.9, P5=45.8, P95=69.8
- **Latency**: mean=17.271279999999997ms, median=16.92ms, P99=22.8901ms
- **Jitter**: 14.6%
- **Stability**: CV=0.1465, score=0.7392
- **VRAM**: peak=1254MB, avg=504MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 837.7ms
