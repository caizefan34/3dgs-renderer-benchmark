# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs | real_scene_speed | 832.5 | N/A | 1.21 | 1.54 | 0.786 | 750 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs (fastest)
- **FPS**: mean=832.5, P5=699.3, P95=1010.1
- **Latency**: mean=1.20122ms, median=1.21ms, P99=1.5401ms
- **Jitter**: 11.3%
- **Stability**: CV=0.1133, score=0.7857
- **VRAM**: peak=750MB, avg=703MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 1454.7ms
