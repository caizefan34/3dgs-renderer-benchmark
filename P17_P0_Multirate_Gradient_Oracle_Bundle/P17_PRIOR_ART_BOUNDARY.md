# P17 Prior-Art Boundary — frozen before experiment

**Date:** 2026-09-27

## Candidate mechanism under test

P17 tests **simultaneous primitive-wise multirate optimization**:

- a Gaussian's projected minor-axis bandwidth determines the image sampling rate used for its gradient;
- broad expensive primitives may use coarse supervision;
- narrow primitives retain high-resolution supervision;
- different primitive populations can therefore receive gradients derived from different image sampling rates within the same conceptual optimization step.

Novelty status:

`UNPROVEN / HIGH-RISK`.

## Known neighboring prior art

### Mip-Splatting — CVPR 2024

Mip-Splatting addresses aliasing across sampling rates using:
- a 3D smoothing filter constrained by input-view sampling frequency;
- a 2D mip filter approximating box filtering.

Overlap:
- projected Gaussian size and sampling frequency are central.

Difference currently under test:
- P17 is not primarily an anti-aliasing renderer;
- it asks whether parameter gradients for different Gaussian populations can be computed at different training sampling rates to reduce optimization work.

### Multi-Scale 3D Gaussian Splatting — CVPR 2024

This work constructs multi-scale Gaussian representations and selectively renders Gaussians appropriate for the requested image resolution. Its ablations also include multi-scale training.

Overlap:
- multiple image scales;
- Gaussian size / pixel coverage;
- resolution-aware primitive handling.

Novelty risk:
**HIGH**.

P17 cannot claim that multi-scale training or scale-aware Gaussian rendering is new.

Potential distinction:
- one scene state;
- one training step;
- primitive-wise routing of raw optimizer gradients to different sampling rates;
- compute reduction is the objective, not multi-resolution rendering quality.

### FreGS — CVPR 2024

FreGS uses progressive frequency regularization and coarse-to-fine frequency supervision to improve densification.

Overlap:
- low/high-frequency supervision.

Difference:
- global/progressive frequency curriculum rather than simultaneous primitive-wise gradient sampling rates.

### Frequency-Aware Gaussian Splatting Decomposition — 2025

This work groups Gaussians into Laplacian-pyramid frequency subbands and progressively trains frequency-decomposed layers.

Overlap:
- explicit Gaussian populations associated with image-frequency bands.

Novelty risk:
**VERY HIGH**.

P17 must not claim that assigning Gaussians to image-frequency bands is new.

Potential distinction:
- P17 does not create a decomposed representation;
- routing is view-dependent from current projected covariance;
- the same ordinary 3DGS representation is retained;
- the primary objective is reducing gradient-computation sampling rate while matching the full-resolution optimizer update.

### SIG / signal-structure-aware scheduling — 2026

SIG derives representation bandwidth / sampling frequency and dynamically schedules:
- image resolution,
- densification.

Overlap:
- sampling-frequency matching;
- training resolution as an optimization variable.

Novelty risk:
**VERY HIGH**.

Potential distinction:
- SIG uses a scene/global training-resolution scheduler;
- P17 tests simultaneous per-primitive sampling-rate assignment within one state/view.

### SAD-GS — SIGGRAPH 2026

SAD-GS uses projected Gaussian extent and local image frequency to guide structure-aware / anisotropic densification and substantially shortens optimization.

Overlap:
- projected extent + image frequency.

Difference:
- density-control policy vs gradient sampling-rate policy.

## Claims forbidden even if P17 succeeds

Do not claim:
- first multi-scale 3DGS training;
- first frequency-aware Gaussian optimization;
- first scale-aware Gaussian training;
- first bandwidth-aware 3DGS;
- first Gaussian/image-frequency matching.

Any eventual paper claim must be narrower and must survive a dedicated novelty audit after a realizable P17-P1 design exists.

## Candidate narrow claim

Only if later evidence supports it:

> Standard 3DGS evaluates all primitive gradients at one image sampling rate. We investigate a view-dependent, primitive-wise multirate optimizer that routes each primitive's gradient computation to the coarsest rate that preserves its full-resolution update, with the purpose of reducing training compute.

This wording is still provisional.
