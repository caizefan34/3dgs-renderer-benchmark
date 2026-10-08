# SplatMG Prior-Art / Boundary Guard — 2026-09-30

This is a limited current audit, not a novelty proof.

## Critical nearby work

### 3DGS-LM (ICCV 2025)
Uses Levenberg-Marquardt and PCG for 3DGS optimization.
Boundary:
- SplatMG must not claim the first second-order or preconditioned optimizer.
- Its proposed distinction is a multilevel/coarse PARAMETER SPACE correction over Gaussian aggregates, not a global LM system.

### Second-order / trust-region 3DGS work (2025–2026)
Multiple works already use Hessian/diagonal-Hessian/trust-region ideas.
Boundary:
- do not pitch SplatMG as “better curvature”.
- pitch only if evidence supports multilevel collective error modes.

### HiGS (2026)
Hierarchical macro-tile rendering architecture.
Boundary:
- rendering hierarchy, not optimizer multigrid.

### Hierarchical 3D Gaussian representations / HiSplat / differentiable hierarchy work
Use coarse-to-fine or LoD Gaussian representations.
Boundary:
- SplatMG keeps the final representation unchanged.
- hierarchy exists only as an optimization coordinate / correction space.

### Laplacian Frequency Hierarchies for Efficient 3DGS Training (2026-09)
Uses Laplacian image decomposition and separate frequency-staged Gaussian fields; archived low-frequency fields reduce active training cost.
Boundary:
- SplatMG MUST NOT use a coarse image pyramid or separate final frequency fields.
- SplatMG keeps one fine Gaussian model and the exact full-resolution objective.
- candidate novelty is Galerkin-style parameter-space coarse correction / V-cycle behavior.

### This project's failed P17 coarse-gradient route
P17 used area-downsampled GT/loss and had preregistered HARD_DROP.
Boundary:
- SG0 uses no low-resolution loss at all.
- all gradients, rollouts and oracle evaluations use the original full-resolution objective.

## Candidate reviewer-safe claim if later validated

A multilevel optimizer for 3DGS that introduces a transient Gaussian-aggregate correction space over the unchanged fine model and full-resolution training objective, with fine smoothing and coarse collective corrections designed to reduce expensive optimization iterations.

Do not claim “first multiscale 3DGS”, “first hierarchy”, “first coarse-to-fine training”, or “first second-order 3DGS”.
