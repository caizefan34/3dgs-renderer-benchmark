# P15-P0 Report - EdgeBudget Pareto Oracle (cost = exact Gaussian-tile intersections)
Verdict: **P0_HARD_DROP** - E_lp(0.50) >= E_bar_viab in only 0/6 cells: even at 50% of intersections the oracle cannot reach the minimum P14-viability energy level

Scope: no training, no CUDA changes, no speedup/novelty claim; oracle/diagnostic only. cells=6; the cost-aware frontier is a fractional-knapsack upper bound.

## 1. OBSERVED
P14 measured that 20-25% primitive updates (top gradient energy) retain almost all short-horizon optimization progress (R_train: q=0.10 median 0.887, q=0.20 median 0.980, q=0.25 median 0.994) but the same primitives consume most exact rasterizer intersections (P14 O2: median R_stream 0.91 at q=0.20 / 0.95 at q=0.25 vs the 0.30 gate), so P14 hard-dropped the freeze-and-compile path.

P15-P0 asks the cheapest falsification question before any new training implementation: does a COST-AWARE primitive subset exist when cost is the exact Gaussian-tile intersection count? I.e. can intersection work be cut to a small fraction while keeping the gradient-energy capture level that P14 measured to carry progress?

## 2. EVIDENCE
### 2.1 per-cell oracle table
| cell | N | isects | minJac | E_lp(.30) | E_bar_prog | E_lp(.50) | E_bar_viab | C_ref(q=.2) | C*(prog) | C*(viab) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle_c0 | 3,693,072 | 39,658,013 | 0.999997 | 0.9876 | 0.9992 | 0.9970 | 0.9992 | 0.6788 | 0.643 | 0.643 | - |
| bicycle_c1 | 3,693,072 | 39,702,537 | 0.999997 | 0.9849 | 0.9988 | 0.9964 | 0.9988 | 0.6565 | 0.623 | 0.623 | - |
| bicycle_c2 | 3,693,072 | 41,087,385 | 0.999989 | 0.9779 | 1.0000 | 0.9959 | 0.9968 | 0.8723 | 0.863 | 0.526 | - |
| room_c0 | 854,618 | 24,574,101 | 1.000000 | 0.9726 | 0.9997 | 0.9935 | 0.9953 | 0.8083 | 0.780 | 0.540 | - |
| room_c1 | 854,618 | 20,377,564 | 1.000000 | 0.9905 | 0.9999 | 0.9979 | 0.9986 | 0.7957 | 0.767 | 0.547 | - |
| room_c2 | 854,618 | 12,394,565 | 1.000000 | 0.9219 | 1.0000 | 0.9791 | 0.9990 | 0.9993 | 0.999 | 0.812 | - |

Median across the 6 cells: E_lp(0.30) = 0.9814, E_lp(0.50) = 0.9962; the P14 energy-top sets carry C_ref(q=.10) = 0.5791 and C_ref(q=.20) = 0.8020 of the intersections; E_bar_prog = 0.9998, E_bar_viab = 0.9987.

### 2.2 per-scene split
- room: E_lp(0.30) = [0.9726, 0.9905, 0.9219]; E_lp(0.50) = [0.9935, 0.9979, 0.9791]; C_ref(q=.20) = [0.8083, 0.7957, 0.9993]; C*(prog) = [0.7804323340251592, 0.7674054170557384, 0.9991971481048346]
- bicycle: E_lp(0.30) = [0.9876, 0.9849, 0.9779]; E_lp(0.50) = [0.997, 0.9964, 0.9959]; C_ref(q=.20) = [0.6788, 0.6565, 0.8723]; C*(prog) = [0.6426437703775023, 0.6233317029589318, 0.8633143238490354]

### 2.3 oracle frontier (median E_lp across cells)
| budget b | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|
| median E_lp | 0.9285 | 0.9695 | 0.9849 | 0.9924 | 0.9964 | 0.9995 | 1.0000 |

### 2.4 gate + counts
- gate: min Jaccard over all cells/q = 0.999989, max |dE| = 2.93e-08
- counts: n_strong=0/6, n_viab=0/6, n_relax=0/6 (informational 0.80-bar: n_viab80=0/6)

## 3. HYPOTHESIS
A cost-aware primitive subset exists: there is a selection rule over (energy S_i, exact intersection cost c_i) whose subset reaches the P14-measured progress-carrying energy level while using a small fraction of the exact Gaussian-tile intersections. Falsification: the oracle frontier cannot reach the minimum P14-viability energy level even when allowed 50% of the intersections.

## 4. TEST
Frozen protocol (preregistered before exports):
- six cells (room/bicycle x cell0/1/2), 16K initial state; progress proxy = P14 selector energy S_i (certified p14_selector reused in place); cost c_i = exact Gaussian-tile intersection count over the 8 cluster cameras via the exact C0 stream path, computed twice with bitwise certificate.
- P14 reproduction gate per q in {0.10,0.20,0.25,0.30,0.50}: exact active count, |dE| <= 5e-4, Jaccard >= 0.95, symmetric-difference energy share <= 1e-5; any failure => P0_EXPORT_INVALID.
- oracle: ratio-order fractional knapsack E_lp(b) and integer greedy E_int(b) on the budget grid 0.05..1.00; headline budgets 0.30 (P14 O2 gate) and 0.50 (P14 O2 hard-drop line); per-cell bars from P14's measured H=32 arms: E_bar_viab = energy of the smallest q with R_train >= 0.85; E_bar_prog = energy of the q=0.20 set.
- verdict: n_strong>=5 -> P0_STRONG_CANDIDATE; elif n_relax<=1 -> P0_HARD_DROP; else P0_HOLD_WEAK_PARETO. No q/alpha/budget was added or changed after results.
- epistemic bound (frozen): energy capture is a necessary-condition proxy; P15-P0 runs no optimization arms and cannot certify real progress retention.

## 5. CONCLUSION
**P0_HARD_DROP** - E_lp(0.50) >= E_bar_viab in only 0/6 cells: even at 50% of intersections the oracle cannot reach the minimum P14-viability energy level.

Next (mandated): STOP P15 primitive EdgeBudget path. Do not tune alpha/budgets.

What remains uncertain: (a) the energy proxy does not measure actual progress retention (P14 data show energy saturates while R_train still varies), so a positive P0 result would still require O1-style arms to certify; (b) the cost metric is the raw intersection count, not the compiled-stream segment structure that P14 O2 measured; (c) only one seed and two scenes are covered.

Single highest-value next experiment: none for the primitive EdgeBudget path; record the drop and move to a different mechanism class.

## Contamination ledger
Correctness-only work. Full nvidia-smi inventories (all GPUs + compute apps) are recorded in each export summary; the assembled ledger is artifacts/P15_P0_CONTAMINATION.json. Measurement GPU per export and foreign-app counts are listed there.

## Prior-art boundary
Cost-aware / visibility-aware primitive selection and sparse updates (Speedy-Splat, Taming3DGS SelectiveAdam, visibility-masked updates) are prior art. P15-P0 makes no novelty claim; it is a preregistered falsification oracle. Novelty status of the compiled-frozen-context mechanism remains UNPROVEN and unchanged by this phase.
