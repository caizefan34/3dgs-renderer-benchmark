# Deferred covariance/opacity pullback: existing interface rejected

2026-09-27. **Stop at the preregistered RGB parity gate.** No timing pairs,
garden probe, CUDA implementation, optimizer update or model continuation ran.

The tested identity is valid in real arithmetic: while parameters are fixed
within an accumulation block, sum covariance/opacity gradients over views,
then apply their common parameter Jacobian once. It is elementary chain-rule
reordering, not an original quality mechanism. This probe tested a concrete
implementation using the author's existing Python covariance computation and
`cov3D_precomp` renderer interface. It did not test an exact native-CUDA
covariance cache, nor does its failure disprove the identity.

The fixed author FastGS room@30K PLY contains 566,856 Gaussians, SHA256
`c2afc615fd244d561bd790e7d7721af1b7a7515222bd5385650e7b886c58516a`.
64 distinct original training cameras were selected with PCG64 seed 20260927;
only these image files were loaded. The first four were used for the required
native replay and deferred-parity sums, all at effective width 1600 and the
original `0.8 L1 + 0.2 (1 - fused SSIM)` loss. The planned 64-view timing
blocks were never entered.

| Check | Native replay | Deferred implementation | Frozen gate |
|---|---:|---:|---|
| Max RGB absolute difference | 0 | 0.00319773 | <= 0.0001 |
| Position gradient relative L2 | 0.000003480 | 0.00017892 | <= 0.001 OR max abs <= 1e-7 |
| DC gradient relative L2 | 0.000000149 | 0.00005036 | same |
| Higher SH gradient relative L2 | 0.000000156 | 0.00005141 | same |
| Scale gradient relative L2 | 0.00034213 | 0.00058926 | same |
| Rotation gradient relative L2 | 0.00039453 | 0.00088148 | same |
| Opacity gradient relative L2 | 0.000000145 | 0.00006585 | same |

All six raw parameter gradient groups pass, but the RGB discrepancy exceeds
the fixed threshold by about 32x. Native replay has zero RGB difference.
The largest image discrepancy's root cause has not been isolated; do not
state that it is definitively just rounding, a threshold crossing, or a
native renderer bug. Gradient parity alone cannot establish renderer parity.

The initial launch failed before loading a model because a helper module was
outside the script search path. The probe was made self-contained by adding
the SHA256 helper; the failed log is retained. This dependency repair preceded
all measurements and changed no gate. The corrected probe terminated normally
with `decision=stop_parity`.

Evidence: `fastgs_deferred_pullback_room_result.json`,
`fastgs_stage_profile_20260927/deferred_room.log`, and the import-failure log
in the same directory. The protocol SHA256 is
`842173411ec3b5d96fc4615572a467d1c417d9e4acd2fe542c5cce27e846456a`;
the measured probe SHA256 is
`e4294e3624b88c83b5ed74cb96574edde70c7406a23573bb334d3a1ade6e042d`.
The author `train.py` remains at SHA256
`1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963`.

Do not implement this Python-covariance version in the trainer or call it
a training optimization. Any different exact-native implementation would
still need enough measured removable complete-step cost, a separate quality
mechanism and source-level novelty review before promotion. The joint
full-resolution quality/time goal remains unmet.
