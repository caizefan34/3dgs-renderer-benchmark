# Local Validation: RTX 5070 Laptop

Date: 2026-07-13

## Environment

- NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB, compute capability 12.0
- Driver 592.01; driver CUDA 13.1; toolkit CUDA 13.3
- Windows WDDM; GPU clocks unlocked
- Python 3.10.20; PyTorch 2.12.1+cu130
- gsplat source commit `77ab983ffe43420b2131669cb35776b883ca4c3c`
- Runtime gsplat version 1.5.3
- GPU timing: CUDA events with per-frame synchronization
- End-to-end timing: wall clock from frame call through event completion

## Validated results

| Gaussians | Configuration | Mean ms | Median ms | P99 ms | FPS from mean | Peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 50K | HiGS tile16 | 1.99 | 1.9 | 2.45 | 502.7 | 147 MB |
| 50K | Speedy-Splat | 12.56 | 12.5 | 13.82 | 79.6 | 584 MB |
| 50K | gsplat dense | 12.25 | 12.2 | 13.76 | 81.6 | 368 MB |
| 200K | HiGS tile16 | 6.34 | 6.3 | 7.23 | 157.8 | 391 MB |
| 200K | Speedy-Splat | 145.75 | 38.8 | 1934.10 | 6.9 | 2183 MB |
| 200K | gsplat dense | 383.10 | 50.0 | 876.45 | 2.6 | 1450 MB |
| 400K | HiGS tile8 | 15.96 | 15.8 | 23.22 | 62.7 | 1057 MB |
| 400K | Speedy-Splat | 1705.36 | 1608.9 | 4776.47 | 0.6 | 4276 MB |

The 200K/400K standard-renderer mean/median gap is caused by repeatable,
view-dependent overdraw. It is not removed as an outlier because those views
are part of the fixed trajectory.

## Quality

- Speedy vs gsplat dense, 50K: minimum 111.96 dB, SSIM 1.0.
- HiGS vs gsplat dense, 50K: minimum 59.37 dB, SSIM 0.9997.
- HiGS vs gsplat dense, 200K: minimum 58.80 dB, SSIM 0.9997.
- HiGS vs gsplat dense, 400K: minimum 59.45 dB, SSIM 0.9997.
- Scale-aware HiGS auto vs gsplat dense, 400K: minimum 58.88 dB.

## Optimization ablation

| Scene | Change | Effect |
|---:|---|---|
| 50K | tile8 -> tile16 | 2.34 -> 1.99 ms, about 15% faster |
| 200K | tile8 -> tile16 | 7.83 -> 6.34 ms, about 19% faster |
| 400K | tile16 -> tile8 | 19.37 -> 15.96 ms, about 21% faster |
| 50K | tile16 + SH32 | slower: 1.99 -> 2.95 ms |
| 400K | tile8 + SH32 | similar mean; observed P99 24.31 -> 17.31 ms |

## Required fixes before results were trusted

- Real gsplat API instead of diff-gaussian fallback.
- Standard/legacy SH PLY parsing to `[N, K, 3]`.
- Correct scale, opacity, and quaternion activation.
- Correct camera forward direction and projection multiplication order.
- CUDA event and end-to-end timing split.
- Runtime renderer version/source metadata.
- Windows CUDA13 gsplat build fixes saved in
  `third_party_patches/gsplat-windows-cuda13.patch`.
