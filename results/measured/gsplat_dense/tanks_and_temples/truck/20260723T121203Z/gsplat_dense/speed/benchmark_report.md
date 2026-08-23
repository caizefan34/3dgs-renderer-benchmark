# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_dense | real_scene_speed | 273.1 | N/A | 3.62 | 4.76 | 0.759 | 1394 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_dense (fastest)
- **FPS**: mean=273.1, P5=229.9, P95=312.5
- **Latency**: mean=3.6621200000000003ms, median=3.615ms, P99=4.7600999999999996ms
- **Jitter**: 10.6%
- **Stability**: CV=0.1061, score=0.7594
- **VRAM**: peak=1394MB, avg=1157MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 2,541,226 gaussians, 601.0MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 3046.7ms
