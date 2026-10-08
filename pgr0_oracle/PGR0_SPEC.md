# PGR0 Frozen Research Specification

## 1. Research question
In the gsplat fused 3DGS projection backward, is the recomputation of continuous
intermediates (R, world covar, mean_c, covar_c, J, covar2d) from the raw inputs
redundant with the forward, and if so, is eliminating it (by caching the forward
intermediates) end-to-end speedup-positive?

This is a redundancy/extractability screen, NOT a temporal cache and NOT a
sorting/structure reuse result. The axis is: same Gaussian state, forward vs
backward within one projection call.

## 2. Projection state
For fixed view v and Gaussian state Θ = (means, quats, scales), the forward
computes, per Gaussian:

  R        = quat_to_rotmat(quats)                    # quaternion → rotation
  covar    = R S Sᵀ                                    # quat_scale_to_covar
  mean_c   = R_cam · means + t                         # posW2C
  covar_c  = R_cam · covar · R_camᵀ                    # covarW2C
  J, cov2d = persp_proj(mean_c, covar_c, K)            # perspective Jacobian + 2D cov
  cov2d_b  = cov2d + eps2d·I                            # add_blur
  conics   = cov2d_b⁻¹

The forward writes only (radii, means2d, depths, conics, compensations). It
DISCARDS R, covar, mean_c, covar_c, J, cov2d.

## 3. What the backward recomputes (the redundancy under test)
The shipped backward (`projection_ewa_3dgs_fused_bwd_kernel`, H8_MR == False)
recomputes R, covar, mean_c, covar_c, J from the raw inputs before running the
VJP chain. Because these are deterministic functions of the SAME inputs the
forward used, the recomputed values are mathematically identical to the forward
values. PGR0 measures this (R1/R2/R3) and confirms that a reuse path (which
reads the forward-saved intermediates instead) produces gradients identical to
autograd (R4).

## 4. What is NOT redundant (must still run)
The VJP chain — `inverse_vjp`, `persp_proj_vjp`, `posW2C_VJP`, `covarW2C_VJP`,
`quat_scale_to_covar_vjp`, `add_blur_vjp` — is the genuine gradient work and is
identical in both the recompute and reuse paths. Only the intermediate
recomputation differs.

## 5. Why the oracle precedes implementation
The oracle uses `torch.autograd` through the faithful forward as ground truth.
It never runs a CUDA kernel. It establishes, with no CUDA, whether (a) the
recomputation is redundant and (b) a reuse implementation is gradient-safe. It
then applies an analytical opportunity screen (compute-only and memory-aware)
to decide whether a future CUDA caching pack is worth building.

## 6. Non-claims
PGR0 must not claim:
- a measured speedup (P0 runs no CUDA);
- cross-GPU/cross-architecture generalization;
- any change to C0/AABS0/FGS-H0/H8 or rasterizer semantics;
- that the redundancy is exploitable — only that it is real and gradient-safe.
