# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | speedy_splat | real_scene_speed | 352.5 | N/A | 2.68 | 4.44 | 0.604 | 748 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### speedy_splat (fastest)
- **FPS**: mean=352.5, P5=251.2, P95=471.7
- **Latency**: mean=2.83708ms, median=2.68ms, P99=4.4403ms
- **Jitter**: 20.5%
- **Stability**: CV=0.2049, score=0.6036
- **VRAM**: peak=748MB, avg=484MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1283.1ms
