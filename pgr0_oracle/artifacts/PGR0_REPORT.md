# PGR0 Projection Gradient Redundancy Oracle — Report

## CONCLUSION
VERDICT: **DROP_REDUNDANCY_CONFIRMED_NOT_EXPLOITABLE**

redundancy confirmed (R1-R4 cos=1.0) and reuse is gradient-safe, but the memory-aware analytical model shows saving the discarded intermediates costs more than the recompute it removes (memory-bound); not exploitable for end-to-end speedup without a CUDA microbench.

This is a fail-closed, CPU-only (no CUDA) redundancy screen. It makes **no
measured speedup claim**; the opportunity numbers below are analytical models.

## OBSERVED
- Provenance: gsplat 1.5.3 lineage, kernel `projection_ewa_3dgs_fused`, math primitives `Utils.cuh` (SHA256 in `MANIFEST_PGR0.json`).
- Source-audit result: PASS — analytic backward reimplemented from `Utils.cuh` and validated against `torch.autograd` (G1, G7).
- Scenes / cameras / dtypes / compensations: room/bicycle/garden × 2 cameras × {float64, float32} × {calc_compensations False, True}.
- Cells: 24 total, 24 redundant (rate 1.000).
- R1 projected-mean redundancy: cos = 1.000000, rel_l2 ≤ 0.00e+00.
- R2 covariance redundancy: cos = 1.000000, rel_l2 ≤ 0.00e+00.
- R3 Jacobian redundancy: cos = 1.000000, rel_l2 ≤ 0.00e+00.
- R4 reuse-vs-autograd gradient: cos ≥ 1.000000000, rel_l2 ≤ 7.54e-06 (worst across v_means/v_quats/v_scales/v_viewmats).
- Max rel_l2 (all measures): 2.383e-05; min cos: 1.000000000.
- Compute-only Amdahl upper bound: 1.749× (optimistic; ignores save memory).
- Memory-aware net-speedup proxy: 0.212× (memory_to_compute_ratio 9.67; save 34 floats = 136 bytes/Gaussian).

## EVIDENCE
- Records (JSONL): `artifacts/records/pgr0_records.jsonl` (float64), `artifacts/records/pgr0_records_fp32.jsonl` (float32).
- Stats: `artifacts/PGR0_STATS.json`. Verdict: `artifacts/PGR0_VERDICT.json`.
- Gradient similarity: `artifacts/PGR0_GRADIENT_SIMILARITY.json`.
- Reference implementation: `scripts/pgr0_projection_ref.py` (validated vs autograd).
- Oracle: `scripts/pgr0_oracle.py`. Analyzer: `scripts/pgr0_analyze.py`.
- Frozen source hashes: `artifacts/MANIFEST_PGR0.json` (`frozen_source_provenance`).

## HYPOTHESIS
PGR0 predicted that the projection backward's recomputation of R, world covar,
mean_c, covar_c, J, and covar2d is redundant with the forward (same inputs ⇒
same values), and that a cached-reuse backward would be gradient-equivalent to
autograd — so the recomputation is in principle removable.

## TEST
1. Reimplemented the fused projection forward + backward (`H8_MR == False`) in
   pure PyTorch on CPU, transcribing `Utils.cuh` exactly, including the
   non-obvious conventions: conic-adjoint off-diagonal `*.5`, `add_blur_vjp`
   receiving `covar2d_inv` (== conics), `quat_scale_to_covar_vjp` receiving
   `quat_to_rotmat(quat)` (not the camera R), and `quat_to_rotmat` returning the
   row-major rotation.
2. Validated the analytic backward against `torch.autograd` through the faithful
   forward (both `recompute` and `reuse` modes).
3. Measured R1/R2/R3 (forward vs backward-recomputed mean2d/covar2d/J) and R4
   (reuse & recompute gradient vs autograd) per active Gaussian.
4. Applied the fail-closed gates and an analytical opportunity model
   (compute-only Amdahl upper bound + memory-aware net-speedup proxy on an
   A100-SXM4-80GB bandwidth/compute proxy).

## GATE TABLE
| Gate | Result | Evidence |
|---|---|---|
| G0 provenance | PASS | CPU-only; frozen config; source hashes in MANIFEST |
| G1 semantics | PASS | `FAIL_CLOSED_GATES.md` G1; autograd validation |
| G2 numeric | PASS | max_rel_l2=2.38e-05, no NaN/Inf |
| G3 R1 mean redundancy | PASS | cos=1.0 in all cells |
| G4 R2 covariance redundancy | PASS | cos=1.0 in all cells |
| G5 R3 Jacobian redundancy | PASS | cos=1.0 in all cells |
| G6 R4 gradient equivalence | PASS | reuse vs autograd cos=1.0 |
| G7 reference validity | PASS | recompute vs autograd cos=1.0 |
| G8 opportunity compute-UB | PASS | 1.749x (optimistic) |
| G9 opportunity memory-aware | FAIL | 0.212x proxy (memory-bound) |

## INTERPRETATION
The redundancy is **real and gradient-safe**: the backward recomputes
intermediates bit-similar to the forward (R1–R3 cos = 1.0), and feeding
forward-saved intermediates into the same VJP chain reproduces the autograd
gradient (R4 cos = 1.0). However, the recompute is a small fraction of the
backward's work, and the forward currently *discards* these intermediates.
Eliminating the recompute would require the forward to save ~34 floats/Gaussian
(≈136 bytes) that it does not today. Against an A100 bandwidth/compute proxy,
the save+restore memory traffic costs ~9.7× more than the
recompute it removes, giving a net-speedup proxy of 0.21× (a slowdown).
The compute-only upper bound (1.75×) is favorable but ignores this
memory cost, so it is not decision-relevant on its own.

## PRIOR-ART BOUNDARY
PGR0 certifies intra-call forward/backward redundancy in the differentiable
projection backward and gradient-equivalence of a reuse path. It is distinct from
temporal caching, sorting/list reuse, and data-movement caching. See
`PGR0_PRIOR_ART_REFRESH.md`. No "first" claims are made.

## NEXT ACTION
No CUDA work is authorized by this verdict. A future CUDA microbench (P5-style,
exactness-first) would be required to confirm or refute the memory-aware model
before any implementation. If the microbench shows the fused fwd+bwd kernel can
hold intermediates in registers/local memory without spilling to global memory,
the candidate could be revisited; otherwise it should remain dropped.
