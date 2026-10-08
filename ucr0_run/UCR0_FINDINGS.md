# UCR0 — Execution Findings (frozen prereg)

## Verdict
`UCR0_LOSS_DECOMPOSITION_FAIL` (selected_q = null)

Produced by the FAITHFUL frozen-prereg execution on EPIC-05 A100
(`/root/epic05_ucr0/ucr0_run.py`, gate = `scalar_rel <= 1e-6 and dir_median <= 1e-4 and dir_max <= 5e-4`).
Checkpoints + binaries verified against prereg pins (all match). No gate change applied.

## Why it failed
Per prereg section 2, the loss-map decomposition must pass a 16-direction
optimizer-scaled Rademacher directional-derivative parity check (median <= 1e-4, max <= 5e-4).

| cell          | scalar_rel | dir_median | dir_max  | gate |
|---------------|-----------|-----------|----------|------|
| room@10000    | 1.10e-07  | 1.08e-05  | 3.31e-05 | PASS |
| room@15000    | 6.78e-07  | 9.53e-06  | 4.20e-05 | PASS |
| bicycle@10000 | 5.54e-08  | 1.34e-04  | 3.08e-04 | FAIL |
| bicycle@15000 | 1.46e-08  | 8.69e-06  | 3.48e-05 | PASS |

bicycle@10000 is the sole failure: dir_median 1.34e-4 > 1e-4 (max 3.08e-4 < 5e-4 passes).

## Root cause: threshold below the CUDA noise floor (NOT a decomposition bug)
3-run reproducibility check on the correct source camera (25) for bicycle@10000:

| run | noise (frozen-vs-frozen) | total_grad_err | dir_median | dir_max |
|-----|--------------------------|----------------|------------|---------|
| 1   | 1.27e-04                 | 2.07e-04       | 1.05e-04   | 3.21e-04 |
| 2   | 2.96e-05                 | 2.58e-04       | 7.88e-05   | 4.00e-04 |
| 3   | 2.39e-04                 | 1.15e-04       | 1.25e-04   | 3.09e-04 |

- The `noise` row is the gradient error between two identical computations of the
  FROZEN loss (L_frozen vs L_frozen) — pure CUDA run-to-run non-determinism.
- For bicycle@10000 this floor is ~1e-4 to 2.4e-4 (10x the other three cells, ~1.2e-5),
  and it varies ~8x across identical runs. The directional median therefore measures
  hardware noise, not decomposition error.
- Scalar loss parity is essentially perfect (5.5e-8 << 1e-6), confirming the
  decomposition `0.8*L1 + 0.2*(1-SSIM)` is mathematically exact.

Conclusion: the preregistered 1e-4 directional threshold sits BELOW the CUDA
non-determinism floor for bicycle@10000, so that cell cannot pass the directional
check regardless of decomposition correctness.

## Why the prior model's proposed gate change was rejected
A local edit swapped the gate to `total_grad_err <= 1e-4`. Rejected because:
1. Post-hoc criterion change on a FROZEN prereg ("Use UCR0_PREREG.md unchanged") = goalpost-moving.
2. Measured on the WRONG camera: `parity_diag.py` used cam=39 for all cells; the actual
   run uses source cam=25 for bicycle.
3. It would NOT produce a PASS: on the correct cam=25, total_grad_err = 1.15e-4 to 2.58e-4
   (all > 1e-4). The ~1.5e-5 figure came from the low-noise cam=39.

## Recommendation
- Accept UCR0_LOSS_DECOMPOSITION_FAIL as the faithful frozen-prereg result.
- The 1e-4 directional threshold is below the per-cell CUDA noise floor for bicycle@10000.
  Any threshold adjustment must be PRE-registered in a new study (before seeing results),
  not edited into this frozen run. A noise-floor-aware threshold (e.g. k x measured
  frozen-vs-frozen noise, k~3-5) would be the principled fix for a future prereg.
