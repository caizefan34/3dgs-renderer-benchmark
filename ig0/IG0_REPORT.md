# InfoGeo-GS IG0 Report (CPU reference run)

Date: 2026-09-30 02:06 UTC

## Scope
- Executed: Sections 3-8 math + Section 10 fp64 CPU correctness controls.
- NOT executed: Section 9 GPU scientific probes (frozen renderer + checkpoints).
- Constraint honored: no local GPU / no production CUDA.

## Correctness controls (Section 10)
- 1_fisher_metric_psd (sec 10.1): PASS
- 2_fisher_metric_parity (sec 10.2): PASS
- 3_quaternion_jacobian_parity (sec 10.3): PASS
- 4_fr_global_radius_parity (sec 10.4): PASS
- 5_fr_local_radius_parity (sec 10.5): PASS
- 6_no_nan_inf_or_invalid (sec 10.6): PASS

Fisher metric worst relative error (closed-form vs differential): 2.420e-13
Quaternion Jacobian worst abs error (autograd): 1.110e-16
FR-GLOBAL radius parity relative error: 0.000e+00
FR-LOCAL radius parity median / p99: 0.000e+00 / 2.720e-16
Natural-direction descent fraction: 0.9967

## Overall correctness
- correctness_pass: true

## Blocked (GPU) inputs
- identity_verified: False (IDENTITY_UNVERIFIED_GPU_PATH_BLOCKED)
- checkpoints_present: True
- The mechanical verdict (IG0_STRONG_PROMOTE / IG0_PROMOTE / IG0_DROP)
  requires IG0_PROBES.json / IG0_TRUE_HOLDOUT.json, which need the frozen
  renderer on GPU. They are NOT produced here; no summary or verdict is fabricated.

## Camera semantics (Section 2)
- room: train=[20, 59, 98, 137, 175, 214, 253, 292] holdout=[16, 56, 96, 136, 176, 208, 248, 288] (of 311 cams)
- bicycle: train=[12, 36, 60, 84, 109, 133, 157, 181] holdout=[8, 32, 56, 80, 112, 136, 160, 184] (of 194 cams)

## Artifacts
- IG0_FISHER_REFERENCE.json
- IG0_QUATERNION_CHECK.json
- IG0_RADIUS_PARITY.json
- IG0_SOURCE_IDENTITY.json
- IG0_ENVIRONMENT.json
- IG0_CAMERA_SEMANTICS.json
