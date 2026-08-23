# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | original_3dgs | real_scene_speed | 211.1 | N/A | 4.71 | 5.71 | 0.825 | 967 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### original_3dgs (fastest)
- **FPS**: mean=211.1, P5=183.1, P95=242.7
- **Latency**: mean=4.73634ms, median=4.71ms, P99=5.711999999999998ms
- **Jitter**: 8.7%
- **Stability**: CV=0.0867, score=0.8246
- **VRAM**: peak=967MB, avg=578MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,244,819 gaussians, 294.4MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1506.1ms
