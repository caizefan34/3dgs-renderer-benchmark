# Frozen native CUDA gate: accepted mask + two-scalar transpose

Date: 2026-09-27. Candidate source must live only in this directory and its matching isolated EPIC-05 checkout. The `public_baselines/fastgs` source remains unchanged.

## Frozen state and inputs

- FastGS source commit: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
- Room checkpoint: `room_big_capture4100`, iteration 4100, PLY SHA256 `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc`.
- Four training-only cameras from the prior bucket opportunity table: `DSCF4668`, `DSCF4770`, `DSCF4873`, `DSCF4976`; effective image size 1600x1066. No held-out/test images or views are loaded.
- Loss is author FastGS `0.8 * L1 + 0.2 * (1 - fused_ssim)` with native forward, background, and camera order held fixed.
- Compare render RGB max absolute error <= `1e-4` and scalar loss absolute error <= `1e-6` for every view.
- Compare all six parameter gradients (`_xyz`, `_features_dc`, `_features_rest`, `_scaling`, `_rotation`, `_opacity`) and the full `[N,4]` screen-space gradient, whose last two components are per-contribution x/y absgrad. Each group must have relative L2 <= `1e-3` or max absolute error <= `1e-7`.

The parity gate must pass all four views before timing starts. Any build failure, forward/loss mismatch, or gradient group outside tolerance stops this candidate without relaxing thresholds.

## Bounded timing gate (only after parity)

Any timing probe used here must be explicitly inspected for optimizer work. The existing fixed-state probe measures only forward + original loss + backward blocks; it does not include an optimizer update and must not be described as a complete training step or full-step wall time. Use identical state and the frozen camera sequence for native/candidate, with accepted-mask initialization and writes included. Measure process wall time in three alternating pairs on an idle A100, 16 warmup blocks, then five 128-block samples per process. Require all three pair medians to favor the candidate and median candidate/native wall ratio <= `0.95`; otherwise stop. No 30K run, quality claim, or production integration is authorized by this component gate.

## Resource accounting

Record candidate shared memory, registers/occupancy if available, mask bytes initialized/written, kernel launch/synchronization count, and atomic output count next to timing. The 32x64 `w/h` arrays use 16 KiB; including 64 shared uint masks gives 16.25 KiB per CTA. The added global accepted mask costs 4 bytes per bucket-pixel and is zeroed once per forward; each accepted contribution updates its pixel word. Include this cost in the measured forward and fixed-state F+loss+B blocks.
