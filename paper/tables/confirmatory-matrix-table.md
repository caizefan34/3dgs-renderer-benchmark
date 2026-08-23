# Confirmatory tables: canonical five scenes (A100, 1080p)

- Protocol: `paper/confirmatory-protocol.md` (2026-08-06).
- Raw per-run values: `paper/tables/confirmatory-matrix-per-seed.json`; bootstrap artifacts: `paper/tables/confirmatory-matrix-<metric>-bootstrap.json`.
- Arm `ctrl` = full-resolution frozen baseline; arm `pd` = progressive-resolution + masked-Adam union-decay cell.
- Every cell below traces to the JSON artifacts above.

## Per-scene means (3 seeds)

| scene | arm | train_ms (ms/step) | total wall (s) | PSNR (dB) | SSIM | LPIPS | final_n | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|
| bicycle | ctrl | 12.942 | 389.831 | 11.592 | 0.348 | 0.620 | 3120378 | 14.43 |
| bicycle | pd | 14.967 | 450.713 | 12.963 | 0.356 | 0.584 | 398454 | 12.72 |
| bonsai | ctrl | 8.654 | 261.143 | 19.734 | 0.754 | 0.292 | 638549 | 5.55 |
| bonsai | pd | 8.753 | 264.309 | 20.191 | 0.761 | 0.286 | 85470 | 4.38 |
| garden | ctrl | 11.603 | 349.682 | 15.609 | 0.455 | 0.449 | 2568478 | 14.14 |
| garden | pd | 12.410 | 374.024 | 16.175 | 0.454 | 0.428 | 872617 | 12.81 |
| train | ctrl | 10.946 | 329.926 | 11.027 | 0.493 | 0.613 | 319788 | 5.24 |
| train | pd | 10.972 | 330.848 | 11.594 | 0.495 | 0.602 | 206899 | 4.47 |
| truck | ctrl | 11.815 | 355.975 | 15.634 | 0.593 | 0.425 | 1081649 | 7.71 |
| truck | pd | 12.553 | 378.307 | 15.733 | 0.590 | 0.432 | 299881 | 5.76 |

## Paired deltas (pd - ctrl)

Lower is better for `train_ms`, `total_wall_s`, `lpips`; higher is better for `psnr`, `ssim`.
Per-scene rows are the mean of the three paired deltas for that scene; the `all` row is the
scene-level block-bootstrap mean with a 95% percentile interval over the whole matrix.
Strict dominance (per protocol section 5) requires a negative paired train_ms delta on every
seed for that scene and the quality guardrail (mean PSNR delta >= -0.05 dB, mean LPIPS delta <= 0.005).

| scene | delta train_ms (ms/step) | delta total wall (s) | delta PSNR (dB) | delta SSIM | delta LPIPS | strict dominance (train_ms) |
|---|---|---|---|---|---|---|
| bicycle | 2.025 | 60.882 | 1.371 | 0.008 | -0.036 | no |
| bonsai | 0.100 | 3.166 | 0.457 | 0.007 | -0.006 | no |
| garden | 0.807 | 24.342 | 0.567 | -0.001 | -0.021 | no |
| train | 0.025 | 0.922 | 0.568 | 0.002 | -0.011 | no |
| truck | 0.739 | 22.333 | 0.098 | -0.002 | 0.006 | no |
| all | +0.739 [+0.197, +1.511] | +22.329 [+6.055, +45.462] | +0.612 [+0.286, +1.027] | +0.003 [-0.001, +0.006] | -0.014 [-0.026, -0.002] | 0 of 5 |

## Time to target quality (pd vs paired ctrl final PSNR)

Pre-registered rule: pd wall_s of the first eval point whose PSNR reaches the paired ctrl arm's final (30k-step) PSNR.

| scene | pd runs | min step | max step | median wall_s |
|---|--:|--:|--:|--:|
| bicycle | 3 | 300 | 300 | 5.89 |
| bonsai | 3 | 300 | 300 | 3.22 |
| garden | 3 | 300 | 300 | 7.28 |
| train | 3 | 300 | 300 | 4.03 |
| truck | 3 | 300 | 300 | 4.10 |
