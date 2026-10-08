# SRR0 Prior-Art Audit — Gate 3

Date: 2026-10-01 (UTC)
Remote identity: dsw-933622-64cc845c7d-lqswc / A100-SXM4-80GB / source SHA e458ac6e41810ad6d9405687c1e698fadd28dac5

## Scope
Fresh audit of (a) public 3DGS literature and (b) internal project evidence, per PRIOR_ART_BOUNDARY.md.
Question: does any work directly collide with SRR0 — a 3DGS training backend that explicitly selects a
**retained minimal forward state** for projection/SH backward using a **measured compute-memory Pareto policy**?

## Public literature
Searched knowledge current through 2026; representative adjacent works:
- **3DGS (Kerbl et al. 2023)** and **gsplat**: save 2D intermediates (conics, depths, colors, means2d) and
  recompute some projection backward. Fixed save/recompute policy; **no** measured Pareto selection of a
  *minimal* retained subset for projection/SH backward.
- **FasterGS / Faster-GS**: the strong backend under test; recomputes scale/rotation/covariance/SH-direction
  in `preprocess_backward`. Fixed tradeoff, not a Pareto-selected minimal retention.
- **Checkpointing / rematerialization (autodiff)**: generic technique (grad checkpointing), not a 3DGS-specific
  measured compute-memory Pareto policy for projection/SH backward.

**No direct public collision found.** No published 3DGS backend explicitly selects a retained minimal forward
state for projection/SH backward via a measured compute-memory Pareto policy.

## Internal project evidence
- **LAW-REPLAY** (ReplayGS, CLOSED): "mathematical reusable structure ≠ GPU-cheap materialized representation."
  SRR0's premise (materialize forward state to avoid backward recompute) is mechanism-adjacent to ReplayGS.
- **LAW-PSC / PSC-GS** (CLOSED): "global 3D work ≈ 4% of training; small projection/global-sync savings ≠ large
  training speedup." SRR0's recompute targets (projection/covariance/scale/SH) are global-3D quantities.
- **F1** (phase-c26): DROP, but on gsplat, different cohort (see SRR0_HISTORY_AUDIT.md). Not a source-cohort collision.

## Internal hard-boundary check (PRIOR_ART_BOUNDARY.md)
"if historical F1 forward→backward state reuse already evaluated the same source cohort and same sufficient
statistics with applicable economics, STOP." → F1 evaluated gsplat / different cohort / SH-color-only, so this
boundary is **not** triggered.

## Gate 3 result
**No direct public collision.** `DROP_PRIOR_ART_COLLISION` **not** triggered. Proceed to Gate 4.

## Note
LAW-REPLAY and LAW-PSC are *corroborating* priors (they predict SRR0's outcome) but are **not** a direct
source-cohort collision with F1, so they do not hard-stop the gate. The operative stop is Gate 4
(DROP_NO_COMPUTE_HEADROOM), confirmed by measurement.
