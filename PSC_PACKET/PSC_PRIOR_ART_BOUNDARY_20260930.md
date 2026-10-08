# PSC-GS Prior-Art Boundary — 2026-09-30

This note is for novelty framing only. It MUST NOT alter the frozen PSC-P0 preregistration.

## Proposed PSC mechanism

PSC-GS proposes to make per-view projected Gaussian state an explicit high-frequency optimization variable:

- projected 2D mean;
- projected 2D conic/covariance;
- opacity/color-related projected state;

with a shared 3D Gaussian state synchronized at a lower frequency.

Conceptually:

`z_v ≈ P_v(theta)`

and a variable-splitting objective of the form:

`sum_v L_v(z_v) + rho * sum_v ||z_v - P_v(theta)||^2`

The intended systems hypothesis is that many high-frequency local optimization steps could avoid the full 3D projection Jacobian / projection-VJP path, while lower-frequency synchronization restores a single multiview-consistent 3D model.

## Closest prior-art boundaries found

### DOGS / DoGaussian (2024)

Paper:
`DOGS: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus`
arXiv: https://arxiv.org/abs/2405.13943

DOGS introduces ADMM into distributed 3DGS training. It maintains local block-level 3DGS models and one global 3DGS model, using consensus on shared 3D Gaussians.

Overlap with PSC:
- variable splitting / consensus optimization;
- local models plus a global shared model;
- training-speed motivation.

Critical distinction:
- DOGS splits a large scene spatially across local 3D Gaussian models.
- PSC proposes view-indexed *projected 2D Gaussian states* as the frequent local variables and amortizes synchronization back to a single 3D Gaussian model.
- Therefore PSC must not claim novelty merely for “ADMM”, “consensus”, or “local/global Gaussian optimization”.

### TriaGS (WACV 2026 / arXiv 2025)

Paper:
`TriaGS: Differentiable Triangulation-Guided Geometric Consistency for 3D Gaussian Splatting`
arXiv: https://arxiv.org/abs/2512.06269

TriaGS computes a multiview consensus 3D point by differentiable triangulation and adds a geometry-consistency loss.

Overlap:
- multiview consensus language;
- projected observations constrain a shared 3D representation.

Critical distinction:
- TriaGS adds a differentiable geometric consistency loss to ordinary end-to-end 3DGS optimization.
- It does not replace frequent 3D parameter updates with persistent per-view projected-state optimization variables.

### CoSplat (2026)

Paper:
`CoSplat: Resolving multi-view inconsistency in feed-forward 3D Gaussian Splatting`

CoSplat builds a confidence-aware global tri-plane consensus from multiple feed-forward input views.

Overlap:
- local-view versus global-consensus framing;
- multiview inconsistency as a central problem.

Critical distinction:
- CoSplat is a feed-forward reconstruction architecture.
- PSC is proposed as a per-scene iterative optimization / training acceleration mechanism.

### Per-view feed-forward Gaussian prediction work

Examples such as GPS-Gaussian and later per-view Gaussian prediction methods associate Gaussians with source views and fuse/filter them.

Overlap:
- per-view Gaussian representations exist.

Critical distinction:
- predicting/fusing per-view 3D Gaussians is not the same as treating the *screen-space projection of one shared 3D Gaussian model* as persistent auxiliary optimization variables.

## Reviewer-safe novelty boundary

Do NOT write:
- “first consensus optimization for 3DGS”;
- “first ADMM for Gaussian Splatting”;
- “first per-view Gaussian representation”.

A potentially defensible claim, only after a broader final audit, would be narrower:

> a training formulation that splits a shared 3DGS optimization problem into high-frequency view-indexed projected-state updates and lower-frequency synchronization to the shared 3D Gaussian parameters, with the goal of amortizing repeated 3D projection/VJP work.

Even this is a candidate claim, not yet established novelty.

## P0 implication

P0 should test the mechanism before implementation:
1. projected-state local descent has useful loss signal;
2. local projected updates are sufficiently jointly integrable / synchronizable into a shared 3D update;
3. leave-one-out cameras do not collapse;
4. the avoided 3D-side work has enough cost headroom.

If these fail, close PSC without production CUDA.
