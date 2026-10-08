# HomotopyGS — Prior-Art Guard (2026-09-30)

This is a limited current audit, not a novelty proof.

## Nearby work that constrains claims

### Mip-Splatting / anti-aliasing filters; Faster-GS (CVPR 2026)
Faster-GS implements/studies 3D smoothing and 2D Mip/low-pass filtering used to improve anti-aliasing and optimization behavior.

Implication:
- adding a Gaussian filter is NOT novel;
- changing `eps2d` is NOT itself a contribution;
- HomotopyGS is only interesting if a continuation SCHEDULE over renderer conditioning demonstrably reduces optimization effort on the final native objective.

### DashGaussian (CVPR 2025)
Progressively schedules rendering resolution and primitive count to fit higher image frequencies over training.

Implication:
- do not claim first coarse-to-fine or continuation-like 3DGS training;
- HomotopyGS must keep full image resolution and original loss, and vary renderer conditioning rather than image resolution.

### Project P17 coarse-loss multirate
This project's P17 used downsampled GT/loss and hard-dropped: cheap coarse gradients did not predict native full-resolution improvement.

Implication:
- H0 must never use lower-resolution supervision.
- Native endpoint evaluation is mandatory.

### Plug-and-Play PDE Optimization (CVPR 2026)
Adds viscosity/PDE-based optimization stabilization to 3DGS and uses MPM.

Implication:
- generic "smoother optimization" claims are too broad.
- candidate distinction is renderer-parameter continuation ending exactly at the native rendering model.

### 3DGS-LM / trust-region / second-order work
Already reduces iteration count via curvature-aware optimization.

Implication:
- HomotopyGS must not claim first fast-converging optimizer.
- Any paper would need direct comparison against modern optimizer baselines.

## Candidate narrow claim, only if later validated

A continuation strategy over differentiable Gaussian-renderer conditioning parameters that preserves full-resolution supervision and anneals exactly to the native 3DGS renderer, with measured reduction in optimization work on the endpoint objective.

Do not claim:
- first smoothing filter;
- first coarse-to-fine 3DGS;
- first continuation method in graphics generally;
- first stable 3DGS optimizer.
