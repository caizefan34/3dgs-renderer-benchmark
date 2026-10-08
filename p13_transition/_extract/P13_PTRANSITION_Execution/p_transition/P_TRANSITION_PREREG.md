# P-TRANSITION preregistration
Date: 2026-09-27
Status: POST-SECTION7 / PRE-NEW-DATA

## Question
Does the relative local optimization efficiency of the PRISM screen chart versus the 9-DoF geometry chart exhibit a reproducible stage-dependent crossover during 3DGS training?

This is a mechanism probe only. It does NOT test or claim superiority over the frozen original-training Adam optimizer, because Section-7 `direct_geometry_adam` operates in the same 9-DoF geometry chart (`means + SPD covariance`) rather than the original `xyz/scaling/rotation` parameterization with preserved optimizer state.

## Frozen code / semantics
Base implementation: PRISM-GS v0.1.2 from Section-7.
Renderer/loss/snapshot/camera semantics remain identical to Section-7:
- gsplat pin: 512d366b67073d77ca099ede742683c165dfc23b
- fused SSIM binary sha256: dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb
- first five distinct IDs from the frozen TRAINING camera sequence
- first 3 optimize; last 2 audit
- max_active=512, common-visible selection only
- three full-resolution update steps
- screen penalty=0.001
- screen max delta=[1,1,0.1,0.1,0.1]
- float64 consensus solve, 3 iterations
- full-image fused loss
- no split/clone/prune and no full training

## Scenes / stages
Primary scenes: bicycle, garden.
Primary stages: 5000, 8000, 11000, 14000.

Only exact frozen snapshots are admissible.
If 8K or 11K exact snapshots do not exist, DO NOT interpolate, reconstruct, or replay without the original optimizer state.
Run 5K and 14K as a two-point precheck and label the result `TWO_POINT_PRECHECK_ONLY`.

## New seeds
271828, 314159, 161803

These seeds are intentionally disjoint from Section-7 seeds 20260927/28/29.

## LR grids
Screen nonlinear: 0.01, 0.03, 0.10
Direct geometry Adam: 0.001, 0.003, 0.010

The complete grid must be reported. No post-hoc LR replacement.

## Timing
Every candidate starts from the identical frozen local state.
Primary time:
`all_in_seconds = shared_setup_seconds + with_evaluation_seconds`.

Also report optimization-only time, but it is secondary.
Arm/LR execution order is deterministically shuffled per seed to reduce ordering bias.
Contaminated runs must be preserved and rerun, not silently overwritten.

## Metrics per candidate
train_gain = -train_delta
eta = train_gain / all_in_seconds
audit_per_train = audit_delta / max(train_gain, 1e-20)

The best candidate for each chart is selected ONLY by maximum eta within the preregistered grid.
All other candidates remain in the artifact.

Per seed:
R_eta = eta_best_screen / eta_best_geometry

## State-level aggregation
For each scene × stage:
- median R_eta over the 3 new seeds
- number of seeds with R_eta > 1.0
- median audit_per_train for the selected screen and geometry candidates
- peak memory ratio (screen / geometry)

## Two-point signal
A scene has a two-point crossover signal iff:
- 5K median R_eta <= 0.90
- 14K median R_eta >= 1.15
- at 14K at least 2/3 seeds have R_eta > 1.0
- at 14K selected screen audit_per_train is no worse than selected geometry in at least 2/3 seeds

Both bicycle and garden must satisfy this to label:
`TWO_POINT_STAGE_CROSSOVER_SIGNAL`.

This is NOT a mechanism confirmation.

## Four-point mechanism candidate
Requires exact 5K/8K/11K/14K snapshots.
For each scene:
- two-point conditions above hold
- Spearman correlation between stage and median log(R_eta) >= 0.8
- the sequence crosses R_eta=1 at least once from below to above
- no single intermediate stage reverses by more than 0.20 in R_eta relative to the previous stage

Both scenes must satisfy this to label:
`MECHANISM_SIGNAL_STAGE_DEPENDENT`.

Otherwise:
`HOLD_NO_REPRODUCIBLE_TRANSITION` or `TWO_POINT_PRECHECK_ONLY`.

## What a PASS permits
A PASS permits an equal-wall-time trajectory test with the same two coordinate charts.
It does NOT permit:
- a >1x or >3x training speedup claim
- comparison against original-training Adam
- full training
- a novelty claim
