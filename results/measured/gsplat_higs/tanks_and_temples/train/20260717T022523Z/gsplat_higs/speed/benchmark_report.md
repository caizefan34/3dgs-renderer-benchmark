# 3DGS Renderer Benchmark Report

## Summary

| Rank | Renderer | Type | Mean FPS | Effective FPS* | Median (ms) | P99 (ms) | Stability | VRAM(MB) | Difficulty | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|------|----------|------|:--------:|:--------------:|:-----------:|:--------:|:---------:|:--------:|:----------:|:----------:|:----------:|:-----------:|
| 1 (fastest) | gsplat_higs | real_scene_speed | 530.0 | N/A | 1.92 | 2.26 | 0.849 | 553 | N/A | N/A | N/A | N/A |

*Effective FPS is experimental and remains N/A without GT metrics.*
Synthetic Stress results must not be interpreted as quality equivalence.

## Per-Renderer Details

### gsplat_higs (fastest)
- **FPS**: mean=530.0, P5=469.5, P95=632.9
- **Latency**: mean=1.88666ms, median=1.92ms, P99=2.2603999999999993ms
- **Jitter**: 9.8%
- **Stability**: CV=0.0975, score=0.8494
- **VRAM**: peak=553MB, avg=505MB
- **Quality vs ground truth**: PSNR=N/A, SSIM=N/A, LPIPS=N/A (not_measured)
- **Scene**: 1,026,508 gaussians, 242.8MB
- **Difficulty**: N/A (not measured)
- **Load Time**: 962.9ms
