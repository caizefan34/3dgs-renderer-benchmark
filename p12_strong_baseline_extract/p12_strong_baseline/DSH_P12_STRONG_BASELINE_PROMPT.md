# P12 — Strong-Baseline Revalidation
## Purpose

This phase is **not a new optimization search**.

P11 is closed with:

- `NO_NEW_OPTIMIZATION_JUSTIFIED`
- `STOP_NEW_OPTIMIZATION`

P12 answers a different reviewer-facing question:

> After replacing the frozen publication trainer's slow SepSSIM implementation
> with the P11-validated, same-semantics fused-SSIM implementation, do the two
> paper-critical performance conclusions still hold?

The only two conclusions to revalidate are:

1. the modern system endpoint: current gsplat main vs frozen C0;
2. the original F9 transform: A0 vs A1.

Do **not** rerun H8 as a rescue experiment.
Do **not** rerun Scalar.
Do **not** invent a new optimization.
Do **not** tune kernels, loss weights, launch configs, or schedules.

---

# 0. Strong loss baseline freeze

Use the exact fused-SSIM implementation/source/binary identity validated by P11.

Read identity from the authoritative P11 artifacts:
- `P11_SOURCE_IDENTITY.json`
- `P11_LOSS_EQUIVALENCE.json`
- `P11_LOSS_TIMING.json`

Do not reinstall or silently update fused-SSIM.

Loss semantics must remain:

```text
L = 0.8 * L1 + 0.2 * (1 - SSIM)
```

SSIM must use the same semantics validated in P11:
- 11x11 Gaussian window;
- sigma = 1.5 where applicable to the implementation contract;
- `padding="same"`;
- same image range;
- same reduction;
- same GT preprocessing.

The P11 L1 result is prior evidence that SepSSIM and this fused implementation
are `SEMANTICALLY_MATCHED_WITHIN_FP`.

Record hashes again; do not assume.

---

# 1. Hardware / scheduling rule

Primary system: EPIC-05, NVIDIA A100-SXM4-80GB.

Use the same clean-GPU policy as P8/P10/DSH-A/P11:
- no foreign compute process;
- record GPU UUID, clocks, temperature, power and memory;
- preserve contaminated attempts;
- rerun contaminated attempts unchanged;
- never average contaminated and clean runs.

## Important host-level rule

The P12-A and P12-B **full 30K campaigns must not execute concurrently on the
same EPIC host**.

Reason:
- F9 is expected to be an approximately 1% effect;
- host/CPU/I/O contention can be comparable to the effect size.

You may prepare/build/smoke-test both campaigns in parallel, but execute their
publication 30K paired runs in non-overlapping campaign windows.

---

# 2. Frozen implementations

## Current gsplat main

Commit:

`512d366b67073d77ca099ede742683c165dfc23b`

Use the same current-gsplat publication adapter already audited in P8/P10/P11.

No H8 patch for the P12 system headline unless the frozen current-gsplat baseline
already explicitly requires it. The intended P12-A BASE is **plain current
gsplat main**, matching P8's current-gsplat arm except for the common
same-semantics fused loss.

## C0

Use the exact frozen publication C0 implementation/source/binary lineage used in
the P8 matched endpoint.

No source modernization.
No new H8 tuning.
No trainable-HiGS changes.
No new launch search.

Only replace the common trainer loss implementation with the same P11-validated
fused-SSIM path.

## A0 / A1

Use the exact frozen P2/P7 lineage:
- A0 = frozen C0 substrate with F9 OFF, Scalar OFF, H8 OFF;
- A1 = A0 + F9 only.

No other code differences.

Again, only replace the common trainer loss implementation with the same
P11-validated fused-SSIM path.

---

# 3. Recipe lock

For all P12 publication runs:

- eps2d = 0.3
- seed = 42
- 30,000 iterations
- same dataset images/resolution
- same camera ordering logic
- same optimizer
- same LR schedule
- same densification/pruning schedule
- same SH schedule
- same absgrad semantics
- same rasterization mode
- same evaluation procedure
- same background policy
- same loss weights

Scenes:

- bicycle
- bonsai
- counter
- flowers
- garden
- kitchen
- room
- stump
- treehill
- train
- truck
- drjohnson
- playroom

No scene may be silently dropped.

---

# 4. Preflight — required before 30K

Run on:
- room
- bicycle
- garden

For every arm that will enter the campaign:

1. verify source SHA and `.so` SHA;
2. verify fused-SSIM source/binary hash;
3. confirm `padding="same"` is actually active;
4. confirm finite forward/loss/backward;
5. run >=200 training iterations;
6. confirm loss scale is comparable to the frozen recipe;
7. confirm expected switch behavior:
   - current gsplat is plain current main;
   - C0 is frozen C0;
   - A0 has F9 OFF;
   - A1 has F9 ON;
8. confirm no unexpected H8/Scalar path was enabled in A0/A1;
9. record peak VRAM;
10. record one short phase-timing sample.

Do not use this preflight to tune anything.

If a semantic or source-identity blocker appears, STOP and report it.

---

# 5. P12-A — Strong-baseline system endpoint

## Question

Under the P11-validated fused-loss baseline, is the frozen C0 system still
faster than current gsplat main under the strict matched 13-scene 30K protocol?

## Arms

- `GMAIN_FUSED`
- `C0_FUSED`

## Pairing

For every scene run a fresh pair.

Primary ratio:

`R_sys(scene) = wall_GMAIN_FUSED / wall_C0_FUSED`

Use a preregistered alternating arm order across scenes.

Each pair:
- same physical GPU;
- back-to-back;
- no other publication pair interleaved on that GPU.

Old P8 wall times are **not** denominators.
They are historical context only.

## Required measurements

Per run:
- total 30K wall time;
- iter/s;
- peak VRAM;
- final Gaussian count;
- PSNR;
- SSIM;
- LPIPS;
- loss curve checkpoints;
- Gaussian population trajectory;
- densification/pruning events;
- low-intrusion phase timing if the publication trainer already supports it.

## P12-A statistics

Compute:

- 13 scene ratios;
- geometric mean;
- median;
- 95% CI on log ratios;
- number of scenes with `R_sys > 1`;
- min/max ratio;
- equivalent C0 wall-time reduction:

`1 - 1 / GM_sys`

Quality:

`Δ = GMAIN_FUSED - C0_FUSED`

Report:
- mean and median ΔPSNR;
- mean and median ΔSSIM;
- mean and median ΔLPIPS;
- outliers.

Population:

- final-N ratio;
- `corr(log R_sys, log N_ratio)`.

Do not call the 30K wall ratio an intrinsic renderer-kernel ratio if trajectory
populations materially differ.

## Relation to old P8

Old strict matched P8 result:

`GMAIN / C0 GM = 1.0979x, 13/13`

Compare descriptively only.

Report:
- new fused-loss GM;
- old SepSSIM GM;
- absolute and relative change in log-ratio.

Do not average them.

Existing P8 fixed-state renderer replay remains valid renderer-execution context
because it does not depend on the training loss implementation.

## P12-A verdict

Use one:

### `SYSTEM_REVALIDATED_STRONG`
- GM > 1;
- >= 11/13 scenes positive;
- 95% CI entirely > 1;
- no systematic quality degradation;
- clean campaign.

### `SYSTEM_REVALIDATED_POSITIVE`
- GM > 1;
- majority of scenes positive;
- CI may overlap 1;
- no systematic quality degradation.

### `SYSTEM_NEUTRAL`
- approximately parity.

### `SYSTEM_REVERSED`
- current gsplat is consistently faster.

This is a **system endpoint** verdict, not an ownership claim.

Never attribute the whole system ratio to F9/H8.

---

# 6. P12-B — F9 strong-baseline revalidation

Execute only after the P12-A publication 30K campaign window is closed.

## Question

Does the original F9 materialization-free producer retain a reproducible
full-training benefit after the common SepSSIM bottleneck is removed?

## Arms

- `A0_FUSED`
- `A1_FUSED`

The only intended renderer difference is F9.

Primary ratio:

`R_F9(scene) = wall_A0_FUSED / wall_A1_FUSED`

## Pairing

Fresh 13-scene paired campaign.

Same rules:
- same physical GPU;
- back-to-back;
- preregistered alternating arm order;
- no reuse of old P2 denominators.

## Required measurements

Same as P12-A.

Additionally record, if already available without intrusive instrumentation:
- visible Gaussian count;
- visible-parameter staging bytes/time for A0;
- corresponding master-indexed producer time for A1.

No new profiling kernel is required.

## P12-B statistics

Compute:

- all 13 ratios;
- GM;
- median;
- 95% CI on log ratios;
- count `R_F9 > 1`;
- min/max;
- equivalent wall reduction.

Quality:

`Δ = A1_FUSED - A0_FUSED`

Report mean/median PSNR, SSIM, LPIPS.

Population/trajectory analysis is required because even numerically equivalent
changes may alter the optimization path via floating-point reassociation.

## Relation to old P2

Historical SXM P2:

`A0 -> A1 (F9) = 1.0111x, 13/13`

Report new and old results separately.

Do not pool them.

## P12-B verdict

### `F9_REVALIDATED_STRONG`
- GM > 1;
- >=10/13 positive;
- 95% CI entirely >1;
- no systematic quality degradation.

### `F9_REVALIDATED_POSITIVE_SMALL`
- GM > 1;
- >=8/13 positive;
- CI may overlap 1;
- no systematic quality degradation.

### `F9_NEUTRAL`
- approximately parity / mixed directions.

### `F9_NEGATIVE`
- consistently slower.

No tuning/rescue branch if neutral or negative.

---

# 7. H8 and Scalar policy

P12 must NOT rerun H8 full training to search for a positive result.

H8 is already closed:

- P10 fixed-state direct backward GM: 1.0193x;
- EPIC 13-scene 30K: 1.0011x, FULLTRAIN_NEUTRAL;
- mx 3-scene x 3-seed: 0.9971x, REPLICATION_NEUTRAL.

The new fused loss does not authorize a post-hoc H8 rescue campaign.

Scalar remains an implementation simplification.

---

# 8. Publication interpretation

The P12 objective is reviewer robustness, not contribution inflation.

Allowed interpretations:

- fused-SSIM is a **strong-baseline modernization**, not our contribution;
- P12-A updates the modern complete-system endpoint;
- P12-B determines whether F9's full-training effect survives a stronger common
  loss baseline;
- H8 remains an exact representation contribution with context-bound
  fixed-state performance and E2E-neutral full training.

Forbidden:

- claim fused-SSIM as ours;
- attribute the complete C0 ratio to F9/H8;
- pool old SepSSIM and new fused-loss GMs;
- rescue H8;
- change loss weights;
- use `padding="valid"` instead of the P11-validated `same`;
- silently drop unfavorable scenes;
- change thresholds after seeing results.

---

# 9. Required artifacts

Create:

## Common
- `P12_SOURCE_IDENTITY.json`
- `P12_FUSED_LOSS_LOCK.json`
- `P12_PREFLIGHT.json`
- `P12_CONTAMINATION.json`

## P12-A
- `P12A_SYSTEM_RUNS.json`
- `P12A_SYSTEM_QUALITY.json`
- `P12A_SYSTEM_TRAJECTORY.json`
- `P12A_SYSTEM_STATS.json`
- `P12A_SYSTEM_VERDICT.json`
- `P12A_SYSTEM_REPORT.md`

## P12-B
- `P12B_F9_RUNS.json`
- `P12B_F9_QUALITY.json`
- `P12B_F9_TRAJECTORY.json`
- `P12B_F9_STATS.json`
- `P12B_F9_VERDICT.json`
- `P12B_F9_REPORT.md`

Preserve:
- raw logs;
- commands;
- GPU snapshots;
- rejected contaminated attempts;
- checkpoint identities;
- source/binary hashes.

---

# 10. Final report format

Return exactly:

```text
P12 Strong-Baseline Revalidation

FUSED LOSS LOCK
P11 identity reused:
padding:
semantic lock:
source hash:
binary hash:

P12-A — CURRENT GSPLAT vs C0
valid scenes:
C0 faster:
GM GMAIN/C0:
95% CI:
median:
min/max:
equivalent C0 wall reduction:

mean ΔPSNR (gmain-C0):
mean ΔSSIM:
mean ΔLPIPS:
trajectory/population confound:

old P8 GM:
new fused-loss GM:
interpretation:

P12-A verdict:

P12-B — F9
valid scenes:
F9 faster:
GM A0/A1:
95% CI:
median:
min/max:
equivalent wall reduction:

mean ΔPSNR (A1-A0):
mean ΔSSIM:
mean ΔLPIPS:
trajectory/population confound:

old P2 F9 GM:
new fused-loss F9 GM:
interpretation:

P12-B verdict:

PAPER CONSEQUENCE
system endpoint claim:
F9 claim:
H8 claim:
what must change in abstract/introduction/results/limitations:

OVERALL:
P12_COMPLETE
```

No optimization search after P12.
