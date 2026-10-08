# DetGS0 Novelty Audit — Comparison Matrix

Idea under audit (DetGS0): **deterministic dynamic 3DGS training with exact
optimizer AND topology state consistency** — a single training framework where
both the per-Gaussian optimizer state (theta, m, v) and the topology state
(clone/split/prune lineage) are preserved bit-exactly across densification, and
training is deterministic (cross-run / cross-GPU reproducible).

Axes: (A) optimizer skipping, (B) topology awareness, (C) exact state preservation.
Evidence labels: OBSERVED (source/read), DERIVED, MEASURED, UNVERIFIED.

| # | System | A: optimizer skipping | B: topology awareness | C: exact state preservation | Evidence |
|---|--------|----------------------|----------------------|----------------------------|----------|
| 1 | INRIA 3DGS | No (dense Adam every iter) | No (standard densify) | No (atomic accum, cross-run not bitwise) | OBSERVED (graphdeco) |
| 2 | FastGS | No (SBS sampling for speed) | No | No (approximation) | OBSERVED (paper) |
| 3 | GS-Scale | No (host offload for scale) | No | No | OBSERVED (paper) |
| 4 | Sparse Adam | Yes (sparse step on visible rows) | No | No (modified Adam, no dense-equivalence claim) | OBSERVED ("A Step to Decouple Optimization in 3DGS") + DERIVED |
| 5 | VOS (internal) | Yes (visibility-deferred zero-grad transitions) | Partial (fences at densify/reset; SH-only; frozen-topology region) | Yes for optimizer state (bitwise replay), SH-only, 15K-30K | MEASURED (VOS_VERDICT.json) |
| 6 | ReproTopo (internal) | No | Yes (deterministic clone/split/prune state machine) | Yes for topology state (bitwise) | MEASURED (REPROTOPO0_REPORT.md) |
| 7 | DPC (internal) | No | No | Yes for gradients (bitwise deterministic backward), NOT optimizer state; ~1.6x slower | MEASURED (DPC0_REPORT.md) |
| 8 | LAVOS (internal) | Yes (lineage-aware VOS into dynamic topology) | Yes (lineage dependency) | Yes (in progress; synthetic only) | MEASURED-synthetic (LAVOS0_REPORT.md) |

## Novelty determination

Public prior art (arXiv sweep, DETGS0_ARXIV_SWEEP.md):
- ZERO public hits for deferred/lazy/virtual per-Gaussian optimizer state with
  exact replay. Sparse Adam is the closest but is a MODIFIED optimizer with no
  dense-equivalence claim.
- ZERO public hits for bit-exact optimizer-state preservation across densification.
- FastGS/GS-Scale are speed/scale; neither claims exact state consistency.
=> The exact-optimizer-state axis (C, optimizer side) has a confirmed PUBLIC gap.

Internal prior art:
- VOS owns exact optimizer-state virtualization (frozen-topology region).
- ReproTopo owns exact topology-state reproduction.
- DPC owns exact gradients (backward determinism), not optimizer state.
- LAVOS0 is the attempt to UNIFY exact optimizer + topology state into the dynamic
  (0-15K) phase — i.e. LAVOS0 IS the unified DetGS0 idea, internally.

## Verdict

NO_PUBLIC_COLLISION on both exact-optimizer-state and exact-topology-state axes
(confirmed by arXiv sweep). However, the UNIFIED claim (exact optimizer + topology
state, deterministic dynamic training) is INTERNALLY occupied by LAVOS0. Therefore:

- DetGS0 is NOVEL relative to PUBLIC prior art (real gap).
- DetGS0 is NOT novel relative to internal LAVOS0 (same idea, earlier).
- Recommendation: do NOT claim de-novo novelty. Position DetGS0 as the unified
  deterministic framework that SUBSUMES VOS (optimizer) + ReproTopo (topology) +
  DPC (gradient determinism) under one correctness contract, and differentiate
  from LAVOS0 by scope (LAVOS0 is a headroom PROMOTION gate; DetGS0 is the
  positioning of the unified exactness contract). Novelty claim must be scoped to
  "first unified exact optimizer+topology state contract for dynamic 3DGS", NOT
  "first exact optimizer state" (VOS) or "first deterministic 3DGS" (DPC).
