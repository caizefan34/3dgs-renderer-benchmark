# PGR0 Fail-Closed Gates and Verdict Policy

## Gate G0 — provenance / isolation
PASS only if:
- the gsplat kernel source (`ProjectionEWA3DGSFused.cu`) and math primitives
  (`Utils.cuh`) are hash-recorded;
- the run is CPU-only (no CUDA, no `.cu`/C++ build);
- the frozen config (packed=False, pinhole, eps2d=0.3, near=0.01, far=1e10) is recorded;
- no renderer/rasterizer source is changed.

Otherwise: INVALID RUN.

## Gate G1 — semantics source audit
PASS only if the installed/recorded gsplat behavior for the tested path is
source-verified against `Utils.cuh`:
- `posW2C`, `covarW2C`, `quat_to_rotmat`, `quat_scale_to_covar_preci`,
  `persp_proj`, `persp_proj_vjp`, `inverse_vjp`, `add_blur`, `add_blur_vjp`,
  `quat_scale_to_covar_vjp`, `quat_to_rotmat_vjp`;
- the backward passes `covar2d_inv` (== conics) to `add_blur_vjp`, and the conic
  adjoint off-diagonal carries the `*.5` factor;
- `quat_scale_to_covar_vjp` receives `quat_to_rotmat(quat)` (NOT the camera R);
- `quat_to_rotmat` returns the row-major rotation (glm column-major storage).

Otherwise: BLOCKED; the oracle must not collect evidence under an unproven model.

## Gate G2 — numeric sanity
Reject on NaN/Inf, non-finite intermediates, or degenerate (zero-norm) quaternions.

## Gate G3 — R1 projected-mean redundancy
The backward-recomputed `mean2d` must match the forward `mean2d`
(cosine ≥ `sim_threshold`, relative L2 ≤ `rel_l2_threshold`) on every active Gaussian.

## Gate G4 — R2 covariance redundancy
The backward-recomputed `covar2d` (pre-blur) must match the forward `covar2d`
(same thresholds).

## Gate G5 — R3 Jacobian redundancy
The backward-recomputed perspective Jacobian `J` must match the forward `J`
(same thresholds).

## Gate G6 — R4 gradient equivalence (reuse)
The reuse-path gradient (cached intermediates) must match the `torch.autograd`
ground-truth gradient (same thresholds) for `v_means`, `v_quats`, `v_scales`,
`v_viewmats`.

## Gate G7 — reference validity (recompute)
The shipped recompute-path gradient must also match autograd (same thresholds).
This validates the analytic backward itself before any reuse claim.

## Gate G8 — opportunity, compute-only upper bound
For certified redundancy rate `h`, the optimistic zero-cost-cache upper bound is

  S_UB = T_bwd / (T_bwd − h·T_recompute).

This is deliberately optimistic: it assumes caching the discarded intermediates
is free and ignores the memory traffic to save/restore them. Implementation is
forbidden if S_UB < 1.05 even for h=1.

## Gate G9 — opportunity, memory-aware (analytical)
Saving the discarded intermediates (R, covar, mean_c, covar_c, covar2d, J ≈ 34
floats/Gaussian) costs extra global-memory write (forward) + read (backward).
Against an A100-SXM4-80GB bandwidth/compute proxy, the memory-aware net-speedup
proxy must be ≥ 1.05 for the candidate to be worth a CUDA implementation.

This gate is an ANALYTICAL model, not a timing measurement (P0 runs no CUDA).

## Verdict policy (mechanical)
Apply gates in order:
- G0/G1/G2 fail → `BLOCKED`.
- G3–G7 any fail → `DROP_NO_GRADIENT_REDUNDANCY` (no provable redundancy).
- G3–G7 pass but G8 < 1.05 → `DROP_NO_GRADIENT_REDUNDANCY`.
- G3–G8 pass but G9 < 1.05 → `DROP_REDUNDANCY_CONFIRMED_NOT_EXPLOITABLE`
  (redundancy is real and gradient-safe, but not worth eliminating).
- all pass → `KEEP_FOR_FUTURE_CACHING_IMPLEMENTATION`.

A redundancy result is necessary but not sufficient for KEEP; the memory-aware
opportunity gate can still DROP. Never claim a speedup from P0.
