# R21 root-cause status

Status: NOT_ESTABLISHED — A–D diagnostic deliverables implemented and locally
tested; E–F require DSH MX evidence. Date: 2026-10-09.

## Observations and evidence strength

The supplied objective reports R20 A100 true AbsGrad parity (4.41M cosine 1.0,
max error approximately 1.3e-11), but radii/parent-prune differences and 5K
quality failures: Bicycle -.2584 dB / -85K Gaussians; Room -.1637 dB / -27K.
It reports first topology mismatch at 600: splits 331 vs 269 and 68 vs 49.
These are explicitly user-provided observations, not locally rerun results.

Actual source confirms:

1. Screen-size pruning is disabled at 600 and neither clone nor split reads
   max_radii2D. That direct explanation cannot account for the split difference.
2. Radius-derived visibility gates historical A/D every step. Extra visible
   rows with zero gradient increase D without increasing A. A CPU regression
   demonstrates this possible mechanism using the actual statistics function;
   the test is not evidence that the mechanism occurred in the MX trajectories.
3. Exported native C++ projection receives opacities; actual HiGS culling and
   render projection omit them. The wrapper documents opacity-aware tighter
   bounds. Exact frozen CUDA cutoff/rounding and actual Python dispatch are
   pending runtime source export; this supports investigating H1/H2.
4. Native checkpoint restore resets A/D and IDs and omits optimizer/RNG. R21
   diagnostics avoid this confound with a separate complete checkpoint format.
5. Reference clone resets M even for no clones, and topology Parameter
   replacement precedes Adam. These are shared behaviors, not established
   renderer-specific causes. Their actual states are captured for H5/H6.

## Hypothesis ledger

| Hypothesis | Current verdict | Evidence needed |
|---|---|---|
| H1 radii contract | SOURCE DIFFERENCE OBSERVED; causal role unconfirmed | Exact frozen projection source, identical-state radii/visibility and C/D contrasts |
| H2 culling/visibility | OPEN | Earliest differing stable IDs, same-camera masks, D intervention and history |
| H3 historical A/D | OPEN | First A or D change, per-step recurrence, E with explicit historical donor |
| H4 threshold sensitivity | OPEN | Grow/scale margins versus measured A/A and B/B repeat noise |
| H5 optimizer/topology maintenance | OPEN | Named moments, raw gradients, exact restored RNG, F downstream comparison |
| H6 lifetime/index/camera | OPEN | Actual projection subset/master mapping, stable lineage, K/view/GT digests, narrowed full states |

## Production decision and unresolved work

No production correction is justified yet. `R21_SOURCE_PATCH.diff` is
intentionally NOT GENERATED: the objective only requests it if a supported
mechanism requires a fix. Diagnostic code is directly delivered in this folder.
No std radii copying, disabled culling/densification, threshold adjustment or
fixed-count training is proposed as a production solution.

Outstanding authoritative evidence:

- MX R21 actual capture at 1,100,599,600 and exact first-difference intervals.
- Frozen missing projection/rendering source export and original R20 reports.
- Repeated same-state A–F from both trajectory checkpoints, original-history
  E mode, and A/A/B/B numerical repeatability.
- Mechanism-specific failing test, minimal isolated production patch (if
  supported), all nine regression/training/performance gates.

Final renderer parity, 1000-step smoke, 5K quality and performance are
NOT_EXECUTED. The overall R21 goal remains open pending DSH return evidence.

## Diagnostic review follow-up

Two tool defects were reproduced with failing CPU regressions and corrected:

- A/B replay previously checked raw parameters/Adam/RNG but did not reject a
  camera, historical-statistics or row-order mismatch. All common inputs now
  fail closed, including recipe, active SH, ID counters and numerical settings.
- The first-topology detector previously considered only final survivor IDs.
  It now separately reports lineage events and survivor-set divergence, so a
  birth then prune within one step is retained. Event hashing canonicalizes
  parent-child edges and the journal contract is `r21.step.v2`.

These fixes improve diagnostic validity; neither establishes an MX renderer
root cause. A search of local R21 checkpoint/run artifacts found no DSH return
evidence. The alternative WV0 source snapshot identifies a separate user-site
gsplat JIT build and has no matching frozen R20 binary identity; it cannot
replace the missing exact MX source export.
