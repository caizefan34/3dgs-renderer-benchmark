# EPIC-05 compression artifact encoding (2026-07-23)

Evidence tier: **A / measured on EPIC-05**. Scope: compressed artifact bytes
and CPU encode/decode validation only. These are not renderer-quality or
near-lossless claims.

| Case | Codec | Source MB | Compressed MB | Ratio | Encode s | Decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small-garden-1080p | block-float | 1447.0 | 674.3 | 2.146x | 36.20 | 9.20 |
| small-garden-1080p | tile-codebook | 1447.0 | 380.5 | 3.803x | 28.23 | 12.45 |
| medium-truck-1080p | block-float | 630.2 | 287.0 | 2.196x | 18.51 | 5.73 |
| medium-truck-1080p | tile-codebook | 630.2 | 163.4 | 3.856x | 17.32 | 6.27 |
| medium-train-1080p | block-float | 254.6 | 114.7 | 2.219x | 7.29 | 2.14 |
| medium-train-1080p | tile-codebook | 254.6 | 64.9 | 3.924x | 7.39 | 2.59 |
| large-bicycle-1080p | block-float | 1520.7 | 705.5 | 2.156x | 44.93 | 13.02 |
| large-bicycle-1080p | tile-codebook | 1520.7 | 395.5 | 3.845x | 44.08 | 16.70 |
| large-bonsai-1080p | block-float | 308.7 | 136.2 | 2.267x | 8.80 | 2.59 |
| large-bonsai-1080p | tile-codebook | 308.7 | 79.4 | 3.890x | 7.42 | 2.79 |

Across the five canonical checkpoints, block-float reduces 4.161 GB to
1.918 GB (2.170x aggregate), while tile-codebook reduces the same inputs to
1.084 GB (3.840x aggregate). Tile-codebook has a slightly higher mean CPU
decode time (8.16 s versus 6.54 s).

All ten source and compressed artifact hashes are recorded in
[`results/measured-compression/artifact-encoding-2026-07-23.json`](../results/measured-compression/artifact-encoding-2026-07-23.json).
The ZIP artifacts and decoded PLY files are intentionally not committed.

## Remaining promotion gates

Each codec still needs a same-GPU reference row and decoded-Ply row for every
case, including FPS/P95/P99, NVML peak memory, PSNR/SSIM/LPIPS, ordered raw
samples, and visual audit. Until those 15 GPU rows complete, the status remains
`artifact_ready` and no codec is described as near-lossless.
