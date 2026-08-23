# Confirmatory tables: 720p resolution leg (EPIC-05 A100, 960x540)

- Protocol: `paper/confirmatory-protocol.md` (2026-08-06).
- Raw per-run values: `paper/tables/confirmatory-consumer-720p-per-seed.json`; bootstrap artifacts: `paper/tables/confirmatory-consumer-720p-<metric>-bootstrap.json`.
- Arm `ctrl` = full-resolution frozen baseline; arm `pd` = progressive-resolution + masked-Adam union-decay cell.
- Every cell below traces to the JSON artifacts above.

## Per-scene means (3 seeds)

| scene | arm | train_ms (ms/step) | total wall (s) | PSNR (dB) | SSIM | LPIPS | final_n | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|
| bicycle | ctrl | 7.136 | 215.379 | 12.283 | 0.291 | 0.589 | 3117307 | 12.55 |
| bicycle | pd | 8.957 | 270.125 | 12.820 | 0.291 | 0.548 | 436761 | 12.48 |
| bonsai | ctrl | 4.296 | 130.158 | 20.077 | 0.739 | 0.222 | 633198 | 3.09 |
| bonsai | pd | 4.529 | 137.306 | 20.144 | 0.741 | 0.228 | 80470 | 2.69 |
| garden | ctrl | 6.526 | 197.087 | 15.349 | 0.407 | 0.425 | 2548343 | 12.26 |
| garden | pd | 7.481 | 225.840 | 16.042 | 0.402 | 0.402 | 888044 | 12.17 |
| train | ctrl | 7.169 | 216.319 | 10.465 | 0.431 | 0.548 | 320386 | 2.81 |
| train | pd | 7.920 | 239.041 | 11.825 | 0.436 | 0.538 | 213926 | 2.45 |
| truck | ctrl | 6.826 | 206.048 | 15.605 | 0.555 | 0.353 | 1080276 | 5.17 |
| truck | pd | 7.574 | 228.656 | 15.603 | 0.534 | 0.355 | 289347 | 5.03 |

## Paired deltas (pd - ctrl)

Lower is better for `train_ms`, `total_wall_s`, `lpips`; higher is better for `psnr`, `ssim`.
Per-scene rows are the mean of the three paired deltas for that scene; the `all` row is the
scene-level block-bootstrap mean with a 95% percentile interval over the whole matrix.
Strict dominance (per protocol section 5) requires a negative paired train_ms delta on every
seed for that scene and the quality guardrail (mean PSNR delta >= -0.05 dB, mean LPIPS delta <= 0.005).

| scene | delta train_ms (ms/step) | delta total wall (s) | delta PSNR (dB) | delta SSIM | delta LPIPS | strict dominance (train_ms) |
|---|---|---|---|---|---|---|
| bicycle | 1.821 | 54.746 | 0.538 | 0.001 | -0.041 | no |
| bonsai | 0.233 | 7.149 | 0.067 | 0.001 | 0.006 | no |
| garden | 0.954 | 28.753 | 0.694 | -0.005 | -0.023 | no |
| train | 0.752 | 22.722 | 1.360 | 0.006 | -0.011 | no |
| truck | 0.748 | 22.607 | -0.002 | -0.021 | 0.002 | no |
| all | +0.902 [+0.481, +1.433] | +27.195 [+14.584, +43.119] | +0.531 [+0.134, +0.968] | -0.004 [-0.013, +0.003] | -0.013 [-0.029, +0.000] | 0 of 5 |

## Time to target quality (pd vs paired ctrl final PSNR)

Pre-registered rule: pd wall_s of the first eval point whose PSNR reaches the paired ctrl arm's final (30k-step) PSNR.

| scene | pd runs | min step | max step | median wall_s |
|---|--:|--:|--:|--:|
| bicycle | 3 | 300 | 300 | 6.07 |
| bonsai | 3 | 300 | 300 | 3.32 |
| garden | 3 | 300 | 300 | 6.27 |
| train | 3 | 300 | 300 | 5.62 |
| truck | 3 | 300 | 300 | 4.48 |
