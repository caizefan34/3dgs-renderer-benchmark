# CrossView-Transfer GS — CVT0 Start Here

Working name: **CVT-GS**.

## Question

Standard 3DGS samples one training camera and applies that view's stochastic update.
CVT0 asks whether different source cameras have systematically different ability to improve OTHER views, and whether this cross-view transfer signal contains information beyond simply choosing the highest-loss camera.

CVT0 is an oracle/opportunity study.
It does NOT implement a production camera scheduler.

## Critical prior-art boundary

TurboGS already performs non-random Geometry-Aware View Sampling and chooses a high-error anchor view.
Efficient Multi-View Training already attacks single-view gradient variance with multi-view batches.

Therefore CVT-GS is interesting only if:
- cross-view transfer differs materially across source views; AND
- a transfer oracle beats a highest-current-loss source; AND
- the source chosen using TRAIN-view transfer also improves TRUE held-out views.

Read:
1. `CVT0_PREREG.md`
2. `CVT0_SCHEMA.json`
3. execute
4. write `CVT0_SUMMARY.json`
5. run `python cvt0_analyze.py CVT0_SUMMARY.json`
6. analyzer owns verdict

No production CUDA.
No full training.
