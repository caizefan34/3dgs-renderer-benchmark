# Confirmatory tables: Deep Blending held-out family (A100, 1080p)

- Protocol: `paper/confirmatory-protocol.md` (2026-08-06).
- Raw per-run values: `paper/tables/confirmatory-db-per-seed.json`; bootstrap artifacts: `paper/tables/confirmatory-db-<metric>-bootstrap.json`.
- Arm `ctrl` = full-resolution frozen baseline; arm `pd` = progressive-resolution + masked-Adam union-decay cell.
- Every cell below traces to the JSON artifacts above.

## Per-scene means (3 seeds)

| scene | arm | train_ms (ms/step) | total wall (s) | PSNR (dB) | SSIM | LPIPS | final_n | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|
| drjohnson | ctrl | 9.285 | 280.097 | 7.780 | 0.594 | 0.804 | 2840259 | 9.30 |
| drjohnson | pd | 8.812 | 266.062 | 9.250 | 0.616 | 0.796 | 1573525 | 7.69 |
| playroom | ctrl | 8.118 | 245.083 | 10.144 | 0.642 | 0.627 | 1240747 | 6.83 |
| playroom | pd | 8.069 | 243.773 | 10.171 | 0.642 | 0.628 | 453682 | 5.34 |

## Paired deltas (pd - ctrl)

Lower is better for `train_ms`, `total_wall_s`, `lpips`; higher is better for `psnr`, `ssim`.
Per-scene rows are the mean of the three paired deltas for that scene; the `all` row is the
scene-level block-bootstrap mean with a 95% percentile interval over the whole matrix.
Strict dominance (per protocol section 5) requires a negative paired train_ms delta on every
seed for that scene and the quality guardrail (mean PSNR delta >= -0.05 dB, mean LPIPS delta <= 0.005).

| scene | delta train_ms (ms/step) | delta total wall (s) | delta PSNR (dB) | delta SSIM | delta LPIPS | strict dominance (train_ms) |
|---|---|---|---|---|---|---|
| drjohnson | -0.473 | -14.035 | 1.470 | 0.022 | -0.008 | yes |
| playroom | -0.049 | -1.310 | 0.026 | 0.000 | 0.001 | no |
| all | -0.261 [-0.473, -0.049] | -7.672 [-14.035, -1.310] | +0.748 [+0.026, +1.470] | +0.011 [+0.000, +0.022] | -0.003 [-0.008, +0.001] | 1 of 2 |

## Time to target quality (pd vs paired ctrl final PSNR)

Pre-registered rule: pd wall_s of the first eval point whose PSNR reaches the paired ctrl arm's final (30k-step) PSNR.

| scene | pd runs | min step | max step | median wall_s |
|---|--:|--:|--:|--:|
| drjohnson | 3 | 300 | 300 | 2.39 |
| playroom | 3 | 300 | 300 | 2.62 |
