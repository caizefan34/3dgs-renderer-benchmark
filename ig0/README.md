# InfoGeo-GS IG0 - CPU reference correctness layer

IG0 tests whether the intrinsic Fisher-Rao geometry of each anisotropic Gaussian
gives a better GEOMETRY UPDATE DIRECTION than native Adam when both moves are
matched to the same Gaussian-statistical trust radius. This directory implements
the **CPU-only, fp64 reference correctness layer** of `IG0_PREREG.md`.

## Constraint honored
Local GPU execution is OUT OF SCOPE for this run (goal: 不是本地GPU执行!).
No production CUDA is compiled or run. `IG0_ENVIRONMENT.json` records
`gpu_execution=false`, `cuda_used=false`.

## What is implemented (CPU, fp64, renderer-agnostic)
`ig0_fisher.py` - frozen math of prereg sections 3-8:
- anisotropic Gaussian Fisher-Rao metric (closed-form blocks + direct differential)
- SO(3) quaternion exponential map / log-map / relative log-map
- quaternion -> right-trivialized local tangent gradient Jacobian (section 4)
- per-Gaussian Fisher natural direction with MP identifiability convention (section 5)
- native Adam geometry displacement (section 6)
- FR-GLOBAL / FR-LOCAL magnitude matching (section 7)
- state construction (section 8)

`ig0_reference_checks.py` - runs the section 10 fp64 CPU correctness controls and
emits the supporting JSON artifacts:
- `IG0_FISHER_REFERENCE.json`   (controls 1 PSD, 2 metric parity, 6 NaN/invalid)
- `IG0_QUATERNION_CHECK.json`   (control 3 Jacobian vs torch autograd + FD)
- `IG0_RADIUS_PARITY.json`      (controls 4 FR-GLOBAL, 5 FR-LOCAL)
- `IG0_SOURCE_IDENTITY.json`    (section 1 identity, fail-closed)
- `IG0_ENVIRONMENT.json`
- `IG0_CAMERA_SEMANTICS.json`   (section 2 camera split + stratification)
- `IG0_REPORT.md`, `IG0_MANIFEST.json`, `IG0_REFERENCE_LOG.txt`

## Run
```
python ig0_reference_checks.py
```
Exit code 0 iff all six correctness controls pass.

## NOT produced here (GPU-gated, by design)
- `IG0_PROBES.json`, `IG0_TRUE_HOLDOUT.json` (section 9 probes need the frozen
  renderer + room/bicycle 10K/15K checkpoints on GPU)
- `IG0_SUMMARY.json`, `IG0_VERDICT.json` (mechanical verdict needs the probes)

The frozen extension/core commits named in section 1 are not present in this repo
and no checkpoints are available, so `IG0_SOURCE_IDENTITY.json` fails closed
(`identity_verified=false`). No summary or verdict is fabricated. The radius-parity
controls validate the magnitude-matching ALGEBRA with a deterministic analytic
gradient field (`gradient_source` is recorded), not the frozen-renderer Adam move.
