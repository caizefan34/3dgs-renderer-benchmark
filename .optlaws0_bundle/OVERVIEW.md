# OPTLAWS0 — Latent Optimization Structure in 3D Gaussian Splatting

Frozen date: 2026-09-29

This campaign is a discovery sidecar to MG0. It does not propose a production
optimizer and does not run full training.

It tests three scientific laws:

L1. Gradient-Variance Concentration:
   Is most camera-sampling gradient variance concentrated in a tiny fraction of
   Gaussians / parameter blocks?

L2. View-Gradient Low-Rank Structure:
   For a fixed Gaussian, do gradients across cameras lie in a low-dimensional
   subspace after Adam-coordinate normalization?

L3. Occlusion–Conflict Structure:
   Can cross-view gradient disagreement be predicted from renderer visibility /
   transmittance / depth-role changes before computing a second backward?

The intended high-impact unifying hypothesis is:

> 3DGS is computationally dense but optimization-noise sparse:
> a small population of visibility-sensitive Gaussians generates most
> camera-sampling variance, and its view-conditioned gradients occupy a
> low-dimensional structured subspace.

This is NOT an established result. The campaign is designed to falsify it.
