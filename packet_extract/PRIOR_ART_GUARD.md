# CVT-GS Prior-Art Guard — 2026-09-30

Limited audit; NOT a novelty proof.

## TurboGS (ICML 2026)
TurboGS already uses Geometry-Aware View Sampling:
- chooses a high-error anchor;
- samples nearby / similarly oriented views;
- combines this with sparse pixel sampling and a hybrid optimizer.

Therefore CVT-GS cannot claim:
- first adaptive view sampling;
- first error-aware camera selection;
- first geometry-aware view selection.

CVT0 explicitly requires the transfer oracle to beat the highest-current-loss source.
If it does not, CVT-GS lacks a distinct mechanism.

## Efficient Multi-View Training for 3DGS (2025)
Already identifies single-view stochastic-gradient variance and uses multi-view batches with renderer/loss/density-control changes.

CVT-GS is not a multi-view batch method.
Its candidate distinction is choosing WHICH single training view should generate the next stochastic update based on expected cross-view transfer.

## NBV / active-view literature
POp-GS, LiTe-GS, AGILE-GS and related works select/acquire informative camera poses.

CVT-GS uses a fixed existing training set; it does not acquire a new view.

## Candidate narrow claim if later validated
A stochastic-view scheduler for per-scene 3DGS that estimates how an update from one existing training view transfers to other views, and schedules source views by predicted cross-view improvement rather than random sampling or current-view error.

A broader literature audit is mandatory before any novelty claim.
