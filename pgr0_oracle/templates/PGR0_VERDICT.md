# PGR0 Verdict

## CONCLUSION
VERDICT: <KEEP_FOR_FUTURE_CACHING_IMPLEMENTATION | DROP_REDUNDANCY_CONFIRMED_NOT_EXPLOITABLE | DROP_NO_GRADIENT_REDUNDANCY | BLOCKED>

## OBSERVED
- Provenance:
- Source-audit result:
- Scenes / cameras / dtypes / compensations:
- Active Gaussians (per cell):
- R1 projected-mean redundancy (cos / rel_l2):
- R2 covariance redundancy (cos / rel_l2):
- R3 Jacobian redundancy (cos / rel_l2):
- R4 reuse-vs-autograd gradient (cos / rel_l2, per param):
- Redundancy rate:
- Max rel_l2 / min cos:
- Compute-only Amdahl upper bound:
- Memory-aware net-speedup proxy:

## EVIDENCE
Link every statement to raw JSON/JSONL/report files. No hand-transcribed metrics
without source path.

## HYPOTHESIS
State what PGR0 predicted before the run.

## TEST
Describe the faithful reimplementation, the autograd ground truth, the
recompute vs reuse paths, the redundancy measures, and the analytical
opportunity model.

## GATE TABLE
| Gate | Result | Evidence |
|---|---|---|
| G0 provenance | | |
| G1 semantics | | |
| G2 numeric | | |
| G3 R1 mean redundancy | | |
| G4 R2 covariance redundancy | | |
| G5 R3 Jacobian redundancy | | |
| G6 R4 gradient equivalence | | |
| G7 reference validity | | |
| G8 opportunity compute-UB | | |
| G9 opportunity memory-aware | | |

## PRIOR-ART BOUNDARY
State only the narrow differentiator supported by the refreshed literature audit.

## NEXT ACTION
No CUDA work unless verdict is KEEP_FOR_FUTURE_CACHING_IMPLEMENTATION.
