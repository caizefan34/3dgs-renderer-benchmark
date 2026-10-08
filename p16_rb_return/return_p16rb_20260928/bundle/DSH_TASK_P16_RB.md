# DSH TASK — P16-RB 2026 Baseline Reset

Read `P16_RB_PREREG.md` and `BASELINE_PINS.json` first.

## Objective

Establish a same-hardware 2026 baseline and bottleneck census before any new research mechanism is proposed.

## Order

1. Create `/root/epic05_p16rb/`.
2. Freeze source identities and clone exact pinned commits.
3. Record all submodule commits and environments.
4. Verify P12 archived timing with the 500-step replay rule.
5. Build external methods.
6. Run Stage-0 semantic preflight.
7. Run clean full `room` pilot for:
   - FasterGS_CVPR26_main
   - FasterGS_CVPR26_fused
   - FastGS_CVPR26
   - SADGS_SIGGRAPH26
8. Evaluate all room outputs with the common metric pipeline.
9. Only methods passing the room pilot expand to bicycle/garden.
10. Collect the 5-point time-to-quality curves using side-channel checkpoints or separate curve replay.
11. Run separate bottleneck profiling replays at ~60% and final.
12. Aggregate.
13. Write report and STOP.

## Critical fairness rules

### Native method semantics first

Do not force FastGS/SADGS/Faster-GS onto the P12 loss or 30K schedule.

Use their own pinned author recipes.

### Common evaluation second

Every method must use the same:
- test split
- GT
- metric implementation.

Save rendered-image hashes or image-directory manifests.

### Resolution

Record exact H×W for each scene/method.

Do not call two runs directly comparable if the pixel counts differ by >2%.

### Timing

Primary training wall time excludes:
- final rendering/evaluation,
- checkpoint serialization that exists only for our measurements.

But record total command wall time separately.

Do not subtract native online method costs such as:
- densification,
- pruning,
- frequency analysis,
- scheduling,
- optimizer scoring.

Those are part of training.

### Contamination

One physical A100 only per compared run.
Record:
- GPU UUID
- foreign PIDs before/after
- peak VRAM
- start/end timestamp.

A contaminated run is retained verbatim and rerun unchanged.

## P12 reuse

Do not automatically rerun full P12.

Perform a clean 500-step timing replay at the frozen recipe.

If median iteration time is within ±5% of archived P12:
- reuse archived P12 full-training quality/wall result;
- label `ARCHIVE_REVALIDATED`.

Otherwise:
- run P12 triad again.

## Faster-GS setup

Pin:
- NeRFICG `e458ac6e41810ad6d9405687c1e698fadd28dac5`
- main `117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847`
- fused `ae665af2f91a4973bb92c40fdc4cba8d12fe3c96`

Use native NeRFICG benchmark/train path.

Do not silently replace the fused branch with a local backport.

## FastGS setup

Pin `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.

Use the exact `train_base.sh` commands for room/bicycle/garden recorded in the prereg.

## SADGS setup

Pin `029814fb5ed773e4a794cec5115951a64f610ab8`.

Use the exact upstream Mip-NeRF360:
- room = indoor config
- bicycle = outdoor config
- garden = special garden config.

## Instrumentation discipline

Primary native run:
- minimal wrapper only,
- no CUDA phase timers inserted.

Profiling replay:
- copied source tree,
- timing-only instrumentation,
- diff stored.

If instrumented replay changes loss/camera sequence unexpectedly:
- do not use the profile.

## Return package

Return the entire required-artifact set from the preregistration, plus:
- source diffs for build compatibility and profiling hooks,
- raw stdout/stderr logs,
- exact commands,
- environment freeze files.

Do not start a new optimization experiment after the verdict.
