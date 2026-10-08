# Newborn optimizer mechanism controls

Specified before GPU outcomes, 2026-09-26. This is a mechanism audit to improve
future design, not a novelty claim or a complete BOGausS reproduction.

## Direct prior art

[BOGausS: Better Optimized Gaussian Splatting](https://doi.org/10.23919/EUSIPCO63237.2025.11226626),
EUSIPCO 2025, pp.765–769. Full accepted manuscript read from
[HAL](https://hal.science/hal-05444229/document), saved in topology_sources/.

- Section IV-C explicitly uses a per-Gaussian update count/lifespan for its
  modified sparse optimizer, incremented only when the Gaussian is updated.
- Section IV-D, Eq.(4), transfers lifespan and gradient statistics to children:
  `t_child=alpha_t*t_parent`, `m_child=alpha_g*m_parent`,
  `v_child=alpha_g^2*v_parent`; reported settings alpha_t=1, alpha_g=0.2.
- It also studies precision-aware updates and density-preserving splitting.
- Its Garden ablation reports 27.81 dB with the full method and 27.77 dB without
  state inheritance; authors discuss more optimization variation and pruning.
  These are author-reported numbers, not our replication or a speedup result.

Therefore lifespan-aware optimization, moment inheritance and scaling m/v by
alpha/alpha² are already published. Our exact dense-Adam fresh-age control is
different from BOGausS's modified sparse estimator. Calling it BOGausS would
misrepresent both the implementation and the comparison.

## Fixed protocol

- Same room/garden@2000 P11 states and native renderer as previous experiments.
- Use the UNION of all eight earlier groups: exactly 256 split parents, no new
  ranking or selection based on prior labels. Native split initialization unchanged.
- Three matched random split draws: 20260926, 20260927, 20260928.
- Per draw: native, fresh-age Adam, inherited moments (0.2*m, 0.04*v), constant
  newborn LR scale, then a native repeat. Two no-edit arms bracket the study.
  17 arms per scene, 256 updates, no further topology or SH-degree change.
- All policies use identical initial child tensors, verified by SHA256. All
  survivor moments are checked exactly at split. The helper only modifies
  newborn updates, leaving native survivor Adam updates unchanged.
- A is the original three candidate-generation views. B contains 16 camera
  indices evenly selected across the dataset list, excluding A. B is excluded
  from the diagnostic continuation but may have been seen by the checkpoint.
- Evaluate the original full-resolution L1+fusedSSIM and per-view PSNR at
  0,1,8,32,128,256. Record per-view metrics to expose view tradeoffs.
- Dense fresh-age means every optimizer update advances the age, including
  zero-gradient rows; it is not visibility-conditioned Sparse Adam.
- Constant LR scale matches fresh-age's FIRST-step amplitude, ignoring epsilon:
  at saved group step=1985, scale=1/2.9375040487. It stays constant afterward.
  This distinguishes an age effect from merely shrinking all newborn steps.
- Moment inheritance uses the published factor as a control on conventional
  Adam; it is a heuristic outside exact parent/child gradient correspondence.

Under an ideal constant nonzero gradient and negligible epsilon, the old-clock
zero-moment update relative to fresh Adam is
`q(a)=(1-beta1^a)/(1-beta1^(1985+a))*sqrt((1-beta2^(1985+a))/(1-beta2^a))`.
It rises from 2.9375 at age 1 to about 6.1069 at age 12, then falls to 1.9888
at age 256. This is an analytical constant-gradient example; it is not a measured
gradient history or evidence of actual instability in the 3DGS experiment.

## Correctness and decision gate

`test_newborn_updates.py`: compare corrected newborns against an independent
fresh Adam over 30 varying-gradient updates, with zero-gradient steps and a
changing LR; verify survivors remain bitwise equal to native Adam. Also compare
constant newborn scaling with a separate lower-LR old-clock Adam. Both passed.

Before advancing a policy, require useful improvement relative to native across
matched draws and BOTH A/B, with differences larger than native-repeat variation.
Fresh-age must also be compared with constant LR scaling. A tiny view-specific
win, one favorable draw, or a known inherited-moment benefit is not an innovative
training-speed result. Three split draws do not replace independent training
seeds or produce a reliable benchmark-level confidence interval.

Record complete wall time, per-step synchronization, overhead of newborn cloning
and correction, initial state restoration, and all evaluation. The diagnostic
post-step correction performs redundant work on newborns; if benefit existed,
a production implementation would update those rows once. No speed claim is
made from this prototype or from counting optimizer arithmetic alone.
