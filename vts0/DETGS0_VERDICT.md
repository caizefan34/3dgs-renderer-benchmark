# DetGS0 Novelty Audit — FINAL VERDICT
Date: 2026-10-01
Idea under audit: deterministic dynamic 3DGS training with exact optimizer AND
topology state consistency (single framework, both per-Gaussian optimizer state
(theta,m,v) and topology lineage preserved bit-exactly across densification).

VERDICT: NO_PUBLIC_COLLISION / NOVEL_VS_PUBLIC / NOT_NOVEL_VS_INTERNAL_LAVOS0
Recommendation: do NOT claim de-novo novelty. Scope the claim to "first UNIFIED
exact optimizer+topology state contract for dynamic 3DGS", and differentiate from
LAVOS0 (the internal predecessor that occupies the same unified idea).

## Prior-art coverage (COMPLETE)
Public (arXiv, 3 sweeps, DETGS0_ARXIV_SWEEP*.md):
- 0 hits: deferred/lazy/virtual per-Gaussian optimizer state with exact replay.
- 0 hits: bit-exact optimizer-state preservation across densification.
- 0 hits: exact + reproducible dynamic 3DGS training.
- Nearest public: Sparse Adam ("A Step to Decouple Optimization") = modified
  sparse optimizer, NO dense-equivalence claim. SkipGS = backward-compute
  skipping, not state virtualization. CAdam = modified moment estimator, no
  exactness. FastGS/GS-Scale = speed/scale, no exactness.
Internal:
- VOS = exact optimizer-state virtualization (frozen-topology region, SH-only).
- ReproTopo = exact topology-state reproduction (bitwise clone/split/prune).
- DPC = exact gradient determinism (bitwise backward), NOT optimizer state; ~1.6x slower.
- LAVOS0 = lineage-aware VOS into the dynamic (0-15K) phase = the UNIFIED idea.

## Three-axis matrix (DETGS0_MATRIX.md)
| System | optimizer skipping | topology awareness | exact state preservation |
|--------|-------------------|--------------------|-------------------------|
| 3DGS   | no                | no                 | no                      |
| FastGS | no (SBS sampling) | no                 | no                      |
| GS-Scale | no (host offload)| no                 | no                      |
| Sparse Adam | yes          | no                 | no (modified Adam)      |
| VOS    | yes               | partial            | yes (optimizer, SH, 15K-30K) |
| ReproTopo | no             | yes                | yes (topology)          |
| DPC    | no                | no                 | yes (gradients only)    |
| LAVOS  | yes               | yes                | yes (synthetic)         |

## Determination
- vs PUBLIC prior art: NO COLLISION on either exactness axis. The exact-optimizer-
  state axis has a confirmed public gap; the exact-topology-state axis is
  internally owned (ReproTopo) and absent publicly.
- vs INTERNAL prior art: the UNIFIED claim (exact optimizer + topology state,
  deterministic dynamic training) is OCCUPIED by LAVOS0. VOS, ReproTopo, DPC are
  each single-axis predecessors.
- Therefore DetGS0 is NOT a de-novo novelty. Its legitimate contribution is the
  UNIFICATION under one correctness contract that SUBSUMES VOS + ReproTopo + DPC.

## Novelty claim scoping (what DetGS0 MAY claim)
MAY claim: "first unified exact optimizer+topology state contract for dynamic 3DGS
training, subsuming VOS (optimizer), ReproTopo (topology), and DPC (gradient
determinism) under a single bitwise-correctness contract."
MUST NOT claim: "first exact optimizer state for 3DGS" (VOS), "first deterministic
3DGS" (DPC), or "first topology-exact densification" (ReproTopo). Must explicitly
differentiate from LAVOS0 (same unified idea; LAVOS0 is a headroom-promotion gate,
DetGS0 is the exactness-contract positioning).
