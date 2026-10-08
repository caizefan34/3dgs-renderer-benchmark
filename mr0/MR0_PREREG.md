
# MacroRep-GS MR0 — Direct Macro-Primitive Capacity Upper Bound (FROZEN)

## 0. Question

Can many fine C0 Gaussians be replaced by far fewer directly-rendered, view-independent macro
Gaussians while preserving multi-view radiance/transmittance and collective first-order response
well enough to remove BOTH raster work and N-scaling optimizer work?

This is an optimistic capacity oracle. It is NOT a training method.

## 1. Cells

room/bicycle @10K/15K.

## 2. Strict representation boundary

A macro primitive is a standard directly-rendered anisotropic Gaussian with:
- one 3D mean
- one covariance
- one opacity
- standard C0 SH appearance

Forbidden:
- anchor MLP
- view-conditioned decoder
- generated child Gaussians
- latent grid
- neural field
- storing the original children at render time

This boundary intentionally distinguishes MR0 from Scaffold-GS-style anchor representations.

## 3. Fine groups

For each cell, use intersection-count-weighted deterministic sampling of 512 local fine-Gaussian groups.
Construct groups geometrically BEFORE any fitting:
- greedy nearest-neighbor groups
- K in {4,8,16}
- no image-loss-based regrouping
- same group list for all tested cameras

Report spatial radius, covariance diversity, SH diversity and visibility.

## 4. Oracle macro fitting

For each group K, fit ONE standard macro Gaussian.
Optimization is allowed to be expensive because this is a capacity upper bound.

Training views for fitting:
8 deterministic TRAIN cameras where the group has highest precomputed visibility,
chosen before macro optimization.

Objective:
- RGB contribution error of the group on transparent background
- alpha/transmittance error
- first-order collective response error for 12 frozen shared perturbations:
  translation xyz (3), log-scale xyz (3), rotation tangent xyz (3), opacity (1), SH-DC intensity/chroma (2)

The macro receives corresponding perturbations in its own parameterization.

No true-holdout views in fitting.

## 5. Full-scene replacement test

Using fitted macros, greedily replace disjoint groups in the full C0 scene.
Frozen target coverage fractions of FINE Gaussians:
- 25%
- 50%
- 75%

At each coverage evaluate exact native rendering on:
- 8 train evaluation cameras not used to select groups
- 8 corrected true-holdout cameras

Measure:
- PSNR/SSIM delta
- intersection-count reduction
- rendered primitive-count reduction
- optimizer-variable-count reduction
- collective Jacobian response capture
- HM0-profile-predicted full-iteration speed upper bound

## 6. Primary gate

A SINGLE K must pass BOTH scenes at BOTH 10K/15K.

### MR0_STRONG_PROMOTE
At 75% fine-Gaussian coverage:
- train PSNR median delta >= -0.08 dB
- true-holdout PSNR median delta >= -0.08 dB
- true-holdout worst delta >= -0.20 dB
- SSIM median delta >= -0.002
- collective-response cosine median >= 0.90
- response p10 >= 0.80
- intersection-count reduction >= 0.65
- optimized-variable reduction >= 0.65
- HM0-predicted iteration speed >= 3.0x
- no NaN/Inf

### MR0_PROMOTE
At 75% coverage:
- train PSNR median delta >= -0.15 dB
- true-holdout median >= -0.15 dB
- worst >= -0.35 dB
- SSIM median >= -0.004
- response cosine median >= 0.82
- p10 >= 0.68
- intersection reduction >= 0.55
- optimized-variable reduction >= 0.55
- HM0-predicted speed >= 3.0x
- no NaN/Inf

Else MR0_DROP.

## 7. Novelty guard

If passing requires:
- multiple generated children per macro,
- view-conditioned macro attributes,
- anchor neural features/MLPs,
then MR0 must return:
    MR0_NOT_DISTINCT_FROM_ANCHOR_FAMILY
rather than PROMOTE.

## 8. Interpretation

PROMOTE means only that a directly rendered macro representation has enough local capacity to justify
a NEW training representation study.

It does not prove that macro initialization/training/densification can reach this oracle.

## 9. Forbidden rescue

No:
- K outside {4,8,16}
- regrouping by image loss
- two-or-more macros per group
- learned decoder
- true-holdout fitting
- threshold relaxation
- local GPU
- full training

## 10. Outputs

MR0_ENVIRONMENT.json
MR0_SOURCE_IDENTITY.json
MR0_GROUPS.json
MR0_FITS.json
MR0_REPLACEMENT_RAW.json
MR0_SUMMARY.json
MR0_VERDICT.json
MR0_REPORT.md
MANIFEST.sha256
