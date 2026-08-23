# Local Validation: RTX 5070 Laptop

> Historical exploratory validation. The current publishable comparison is the
> complete EPIC-05 A100 Tier A matrix in
> [`comparison-analysis.md`](comparison-analysis.md).

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

| Gaussians | Configuration | Mean ms | Median ms | P99 ms | FPS | VRAM | PSNR vs GT | SSIM vs GT | LPIPS vs GT |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50K | HiGS tile16 | 1.99 | 1.9 | 2.45 | 502.7 | 147 MB | N/A | N/A | N/A |
| 50K | Speedy-Splat | 12.56 | 12.5 | 13.82 | 79.6 | 584 MB | N/A | N/A | N/A |
| 50K | gsplat dense | 12.25 | 12.2 | 13.76 | 81.6 | 368 MB | N/A | N/A | N/A |
| 200K | HiGS tile16 | 6.34 | 6.3 | 7.23 | 157.8 | 391 MB | N/A | N/A | N/A |
| 200K | Speedy-Splat | 145.75 | 38.8 | 1934.10 | 6.9 | 2183 MB | N/A | N/A | N/A |
| 200K | gsplat dense | 383.10 | 50.0 | 876.45 | 2.6 | 1450 MB | N/A | N/A | N/A |
| 400K | HiGS tile8 | 15.96 | 15.8 | 23.22 | 62.7 | 1057 MB | N/A | N/A | N/A |
| 400K | Speedy-Splat | 1705.36 | 1608.9 | 4776.47 | 0.6 | 4276 MB | N/A | N/A | N/A |

These generated stress scenes have no source photographs. Their GT-relative
PSNR/SSIM/LPIPS are therefore not measurable, and the timing rows alone do not
establish that HiGS preserves trained-scene reconstruction quality.

The 200K/400K standard-renderer mean/median gap is caused by repeatable,
view-dependent overdraw. It is not removed as an outlier because those views
are part of the fixed trajectory.

## Renderer consistency diagnostics (not GT quality)

- Speedy vs gsplat dense, 50K: minimum 111.96 dB, SSIM 1.0.
- HiGS vs gsplat dense, 50K: minimum 59.37 dB, SSIM 0.9997.
- HiGS vs gsplat dense, 200K: minimum 58.80 dB, SSIM 0.9997.
- HiGS vs gsplat dense, 400K: minimum 59.45 dB, SSIM 0.9997.
- Scale-aware HiGS auto vs gsplat dense, 400K: minimum 58.88 dB.

These numbers compare rasterizers to one another. Reference quality instead
compares every renderer independently against the corresponding photograph.

## Follow-up paired-reference audit (2026-07-14)

The official pretrained Train model was evaluated against 38 paired official
photographs at 980x545. The pretrained archive does not establish that those
photographs were excluded from training, so this is a renderer-fidelity audit,
not a held-out reconstruction leaderboard.

| Renderer | PSNR vs GT | SSIM vs GT | LPIPS vs GT | Mean delta vs original |
|---|---:|---:|---:|---|
| original 3DGS | 24.9319 | 0.865773 | 0.223592 | reference |
| gsplat dense | 24.3061 | 0.858717 | 0.226278 | -0.6257 dB / -0.007056 / +0.002686 |
| TC-GS | 24.9138 | 0.865044 | 0.222874 | -0.0180 dB / -0.000729 / -0.000718 |

SSIM uses Graphdeco's zero-padded 11x11 Gaussian implementation; LPIPS uses
PyPI LPIPS VGG with `[-1,1]` inputs. Dense reproduces the earlier HiGS-sized
drop, so that drop is not specific evidence against HiGS. TC-GS passes the
configured 0.1 dB / 0.001 / 0.001 equivalence thresholds.

The corresponding native-camera 1959x1090 speed smoke test used 10 measured
frames after 5 warmups, one repeat, and unlocked clocks:

| Renderer | Mean GPU | P99 | FPS | Peak VRAM |
|---|---:|---:|---:|---:|
| gsplat dense | 6.125 ms | 6.916 ms | 163.3 | 653 MB |
| TC-GS | 4.614 ms | 5.237 ms | 216.7 | 796 MB |

TC-GS was 1.33x faster by mean GPU latency in this short run. See
`data/results/rtx5070_train_reference_summary_2026-07-14.json` for hashes and
machine-readable results.

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
- Windows CUDA13/SM120 TC-GS build fixes saved in
  `third_party_patches/tcgs-windows-cuda13-sm120.patch`.
