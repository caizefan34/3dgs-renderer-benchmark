# N1 Independent Scientific Audit

## Status

The execution campaign is technically strong: checkpoint restoration, action determinism, timing hygiene, and artifact completeness all support using the measurements.

The preregistered rule-chain label `N1_SMALL_BUT_ACTIONABLE` is mechanically consistent with the old N1 rules.

However, this does **not** yet authorize N2.

## Critical correction

The report states that the best A2 partner is low-overlap for all 384 anchors.
Independent recomputation directly from the raw action and gradient-conflict rows gives:

- room: high 64 / medium 92 / low 36
- bicycle: high 55 / medium 87 / low 50

Therefore the mechanistic statement “A2 value comes from diverse low-overlap views, exactly where sketches are blind” is unsupported and must be removed or corrected in the canonical N1 report.

Random A2 improvements over a per-anchor A0/A1 oracle are also not concentrated in low-overlap partners:

- room: 9/192 anchors; high 4 / medium 4 / low 1
- bicycle: 46/192 anchors; high 14 / medium 16 / low 16

The stronger pattern is **phase dependence**:

- room: 8 of the 9 random-A2 improvements occur at 25K;
- bicycle: 39 of 46 occur at 20K.

## What N1 really establishes

1. One-step held-out update value is highly heterogeneous in the 15K–25K phase.
2. A per-anchor offline oracle can outperform a 0/1 oracle by 11.5% on room and 4.0% on bicycle at the preregistered budget.
3. The extra value of A2 is small and not cross-scene strong.
4. The production signal was never tested because the gate failed.
5. A one-step held-out oracle is not yet evidence that a sequential training policy will improve final quality/time.

## Main unresolved confound

A2 averages two gradients. Its apparent benefit may arise from either:

- genuine extra-view information / variance reduction; or
- effective update damping / step-size reduction.

The latter is especially plausible because many A1 updates worsen the fixed held-out loss, while A2 changes the update direction and magnitude.

This confound must be resolved before spending engineering effort on ReBudgetGS.
