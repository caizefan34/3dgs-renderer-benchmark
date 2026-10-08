# PSC-P0 Frozen Preregistration v1

## Status
FROZEN BEFORE RAW RESULTS.

## Scientific question
Does 3DGS require image-rate 3D consistency, or can multiple high-frequency projected-state updates be amortized behind a lower-frequency multi-view 3D synchronization?

The mechanism under test is NOT merely "optimize in 2D" and NOT merely "use consensus/ADMM".
The candidate mechanism is:

3D theta -> project once -> short-lived per-view projected states z_v
-> several screen-space updates without projection VJP
-> one multi-view 3D synchronization.

## Frozen baseline / identity
- Baseline: P12/C0 = F9 + SCALAR_ADJOINT + H8_MR, PX=2
- Worktree: /root/epic05_frozen/higs_c0_final30k_worktree
- Trainer: /root/epic05_p12/publication_trainer_p12.py
- Scene: room
- Checkpoint: 25K
- Checkpoint sha256: b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d
- C0 extension sha256: d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- Core gsplat_cuda.so sha256: 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- Loss: 0.8 * L1 + 0.2 * (1 - SSIM), full-resolution fused SSIM
- Tile size: 16
- Background: black
- No densification change
- No optimizer change
- No resolution change
- No loss change

## PSC-P0 has two primary tests

### A. Compute opportunity upper bound
Measure one frozen C0 step as:
T_native = T_screen + T_global.

T_screen:
- raster forward from projected state
- loss
- raster backward to projected state

T_global:
- 3D -> projected-state projection
- projected-state -> 3D projection VJP
- 3D optimizer / other work that a screen-local inner loop can amortize

For K in {2,4,8}:
S_upper(K) = (T_screen + T_global) /
             (T_screen + T_global / K)

Also report:
f_global = T_global / T_native
S_max = 1 / (1 - f_global)

This is an oracle upper bound only. Consensus cost is assumed zero.

### B. Multi-view geometric integrability oracle
Primary geometry-only projected state:
z_v^G = (mu_2d, Sigma_2d/conic equivalent)

3D geometry parameter block:
theta_g = (mean_3d, covariance/scale-rotation local tangent)

For each selected Gaussian g and B-view stack:
Delta z ~= J Delta theta_g

Let d = - grad_z L in whitened projected-state coordinates.

Compute the least-squares projection of d onto Range(J):
d_hat = J pinv(J) d

Primary integrability ratio:
R_int = ||d_hat||^2 / ||d||^2

Energy-weighted cohort metric:
R_int_energy = sum_g ||d_hat_g||^2 / sum_g ||d_g||^2

B in {2,4,8}.

### Whitened screen tangent metric
Do not compare raw pixel-center and covariance coordinates directly.

For center:
delta_mu_tilde = Sigma_2d^{-1/2} delta_mu

For covariance:
delta_Sigma_tilde = Sigma_2d^{-1/2} delta_Sigma Sigma_2d^{-1/2}

Equivalent numerically stable coordinates are allowed if algebraically documented
and the metric is invariant to pixel/covariance units.

### Leave-one-view-out consistency
For each held-out view h:
1. solve Delta theta_g* using B-1 views;
2. predict held-out projected motion J_h Delta theta_g*;
3. compare with held-out descent -grad_z L.

C_hold = cosine(J_h Delta theta_g*, -grad_z L_h)

Primary statistic:
P_hold_pos = fraction of weighted cohort with C_hold > 0.
Also report median C_hold.

## Camera selection
- Deterministic.
- Camera 19 is the seed/reference camera.
- Select the remaining views by camera-center farthest-point sampling.
- Freeze the resulting 8 camera IDs before reading PSC metrics.
- Nested view sets B={2,4,8}; no result-conditioned camera replacement.

## Gaussian cohorts
Primary:
- top 4096 Gaussians by accumulated rendering contribution over the frozen 8 views.

Diagnostic only:
- 4096 uniformly sampled visible Gaussians.

Mechanical verdict uses PRIMARY cohort only.

## Mechanical verdict
STRONG_PROMOTE iff ALL:
1. S_upper(8) >= 1.50
2. R_int_energy(B=8) >= 0.65
3. P_hold_pos(B=8) >= 0.75
4. R_int_energy(B=8) >= 0.50 * R_int_energy(B=2)

DROP iff ANY:
1. S_upper(8) < 1.30
2. R_int_energy(B=4) < 0.40
3. P_hold_pos(B=8) <= 0.55

Otherwise:
INCONCLUSIVE

Precedence:
DROP is evaluated first. If no DROP condition fires, then test STRONG_PROMOTE.
Else INCONCLUSIVE.

## Forbidden in P0
- no ADMM implementation
- no persistent per-view training state
- no production CUDA consensus solver
- no full training
- no bicycle scene
- no multi-seed
- no densification changes
- no loss/resolution changes
- no optimizer changes
- no post-hoc gate edits
- no novelty claim from P0 alone

## Required outputs
- ENVIRONMENT.json
- FROZEN_IDENTITIES.json
- CAMERA_SET.json
- TIMING_RAW.json
- TIMING_SUMMARY.json
- PSC_RAW_PRIMARY.npz
- PSC_RAW_DIAGNOSTIC.npz
- PSC_METRICS.json
- PSC_P0_VERDICT.json
- PSC_P0_REPORT.md
- MANIFEST.sha256

## Evidence discipline
Report sections must be:
OBSERVED
EVIDENCE
HYPOTHESIS
TEST
CONCLUSION

No production implementation is authorized unless PSC_P0_VERDICT is STRONG_PROMOTE.
