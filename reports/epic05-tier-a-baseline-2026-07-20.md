# EPIC-05 Tier A baseline (2026-07-20)

This is the first complete Matrix v3.1 `common_representation` cohort. Every
row was measured on EPIC-05 from the same five immutable checkpoints, camera
manifests, resolution, protocol, software cohort, and physical GPU class.

## Evidence

- Evidence tier: **Tier A (repository run)**.
- Host: native Ubuntu 22.04, NVIDIA A100-SXM4-80GB, driver 580.105.08.
- Runtime cohort: Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8 wheel.
- Suite: Matrix v3.1.0, 1920x1080, Garden, Truck, Train, Bicycle, Bonsai.
- Sampling: 30 warmups, 5 repeats x 100 timed frames, 100 GT views per row.
- Completeness: 5 renderers x 5 cases = 25 complete `metrics.json` records.
- Integrity: all 25 `raw_samples.json` SHA-256 values match their provenance.
- Primary memory: positive absolute per-process NVML peak for every row.

The generated machine-readable authority is
[`docs/leaderboard/ranking.json`](../docs/leaderboard/ranking.json). Raw
rendered PNGs are intentionally retained on EPIC-05 rather than committed;
the compact metrics, raw timing samples, NVML observations, quality summaries,
and hashes are committed under `results/measured/`.

## Overall baseline

Geometric-mean FPS and frame time are used across the five cases. Peak VRAM is
the maximum case value. Quality is the arithmetic mean of the five case means.

| Renderer | FPS | Speed index | P99 ms | Peak VRAM MiB | PSNR dB | SSIM | LPIPS | Delta PSNR dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsplat HiGS | 696.91 | 5.671x | 2.55 | 6616 | 25.8345 | 0.837222 | 0.264909 | -0.2853 |
| Speedy-Splat | 293.03 | 2.385x | 6.30 | 4276 | 26.1208 | 0.840367 | 0.263736 | +0.0009 |
| TC-GS | 251.62 | 2.048x | 16.28 | 4322 | 26.1298 | 0.840146 | 0.263274 | +0.0100 |
| gsplat packed | 241.60 | 1.966x | 7.49 | 4206 | 25.8341 | 0.837249 | 0.265305 | -0.2857 |
| Original 3DGS | 122.88 | 1.000x | 16.17 | 8234 | 26.1198 | 0.840346 | 0.263738 | 0.0000 |

## Decisions supported by this cohort

- Highest measured throughput: **HiGS**, at 5.67x original 3DGS. It trades
  0.285 dB PSNR and 2.34 GiB more peak VRAM than Speedy-Splat for throughput.
- Near-lossless speed choice: **Speedy-Splat**. Its aggregate PSNR, SSIM, and
  LPIPS are effectively equal to original 3DGS in this cohort while FPS is
  2.38x higher and peak VRAM is 3.87 GiB lower.
- Lowest measured memory: **gsplat packed**, 4206 MiB, but its quality follows
  the same approximately 0.285 dB delta as HiGS on these checkpoint mappings.
- TC-GS is quality-preserving and fast on average, but its wide FPS confidence
  interval and 16.28 ms P99 make tail-latency investigation a prerequisite for
  a production recommendation.

These conclusions apply only to the declared A100/Linux/common-representation
cohort. They do not establish training speed, compressed-model quality,
temporal stability, mobile performance, or 4K scaling.

## Reproduce and validate

```bash
ssh EPIC-05
cd /root/3dgs-renderer-benchmark
/root/miniforge3/envs/gsplat/bin/python src/scripts/generate_matrix_rankings.py \
  --suite benchmark/suite.json \
  --results-root results \
  --output-dir docs/leaderboard
/root/miniforge3/envs/gsplat/bin/python -m pytest -q
```

Long-form renderer, compression, HiGS, integration, and publication priorities
are in [`docs/research-roadmap-2026.md`](../docs/research-roadmap-2026.md).
