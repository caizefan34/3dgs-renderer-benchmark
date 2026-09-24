# Warp-cooperative emit correctness

Status: **PASS**.

The tested variant is `WARP_ALL`, one warp per projected Gaussian, with a
128-thread CTA for pass 2 (four Gaussians/CTA). The count pass retains its
baseline 256-thread CTA. Each lane writes `start_g + k`, where `k = lane +
32*n` is reconstructed as the baseline row-major `y → x` tile position.

## Pre-sort and downstream identity

For room, bicycle, and garden, camera indices 0 and 1 were tested (six real
camera/checkpoint cases). All six cases had byte-identical SHA-256 digests for:

- `tiles_per_gauss`;
- pre-sort `isect_ids` and `flatten_ids`;
- sorted `isect_ids` and `flatten_ids`;
- `isect_offsets`.

Thus `N_isects`, every preallocated prefix-sum range, tile order, key depth
bits, radix-sort input/output, and offset encoding are exactly unchanged.

## Render and gradients

RGB and alpha were bitwise identical in all six cases: max/mean absolute error
was 0 and PSNR was infinite. Backward comparisons use the same rendered loss
(`mean(rgb²) + mean(alpha)`). Largest observed absolute differences across the
six cases were: xyz `1.523e-3`, quaternion `6.927e-3`, scale `1.456e-3`,
opacity `1.118e-7`, SH `4.992e-7`. The corresponding mean errors are tiny
(worst case quaternion mean-relative error `6.39e-3`) and are consistent with
the existing non-deterministic floating atomic reduction in backward, not an
intersection-structure difference: all forward structures and rendered values
are byte-identical.

Raw evidence: `experiments/final_sprint/warp_emit/raw/correctness_candidate_128_summary.json`.

The separate repeated backward control passes; see `warp_emit_nondeterminism.md`.
