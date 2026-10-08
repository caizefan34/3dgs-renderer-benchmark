# DSH TASK — N1 ReBudgetGS Mechanism Measurement

Read:
1. `N0_NOVELTY_AUDIT.md`
2. `N1_PREREG.md`

This is a frozen mechanism-measurement campaign.

## Stage 0 — source/checkpoint audit

Verify:
- P12 trainer hash;
- C0 renderer/core hashes;
- 15K/20K/25K checkpoints for room+bicycle;
- optimizer state is restorable exactly.

If optimizer state is missing from a checkpoint, do not approximate it.
Return `N1_CHECKPOINT_BLOCKER` for that checkpoint.

## Stage 1 — deterministic probe construction

Create the frozen 64 anchors/checkpoint/scene and partner pools exactly as specified.

Write:
- `N1_PROBE_LIST.json`

Do this before computing any probe gradients.

## Stage 2 — counterfactual action measurements

For each probe:
- restore identical state;
- evaluate A0, A1, A2-random;
- evaluate all allowed A2 partners only for the diagnostic A2-best oracle;
- held-out set fixed before action evaluation.

Do not alter the live canonical checkpoints.

## Stage 3 — timing

Measure action timing separately with warmup/repeats.
No profiler during primary timing.

## Stage 4 — compact sketch audit

Use fixed random projection dimension 256.
Measure stale-sketch vs current-gradient conflict prediction.
Do not learn a projection.

## Stage 5 — budget frontier

Run the frozen offline assignment at rho .60/.75/.90.
Compare against matched-cost 0/1 and random-pair mixtures.

## Stage 6 — cross-scene cheap policy only if gated

Use the exact frozen logistic-regression specification.
No hyperparameter tuning.

## STOP conditions

N1 is not allowed to:
- implement the final online controller;
- run full 15K→30K ReBudgetGS;
- tune thresholds after inspecting bicycle;
- add a learned network;
- change the renderer;
- alter densification;
- touch garden.

Return all preregistered artifacts and STOP.
