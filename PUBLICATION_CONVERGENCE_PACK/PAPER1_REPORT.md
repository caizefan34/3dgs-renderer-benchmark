# PAPER1 REPORT — Publication Evidence Matrix (Auto-Updated)

**Generated:** 2026-10-02  |  **Mode:** read-only synthesis. NO GPU work. NO new runs. NO cohort merging.
**Source of truth:** PAPER0_HARDENED_EVIDENCE_MATRIX.json + per-method *_VERDICT.json / *_SOURCE_LOCK.json.

## 0. What changed vs PAPER0 snapshot
- Incorporated **PBR0** (COMPLETE_RECONCILIATION), **PUB0** (COMPLETE CPU bottleneck ledger), current **TFC** status, current **DFC** status.
- Added pending placeholders for **TFC_FC0, DFC2, DFC_FC0, PUB0**.
- No method role was silently upgraded. Roles change ONLY per explicit gates.

## 1. Role summary
| Role | Methods |
|---|---|
| MAIN_CANDIDATE | TFC (pending TFC_FC0) |
| NEEDS_REVALIDATION | K50-B1 |
| SUPPORTING | DFC1-R1 saved-T |
| PENDING | DFC optimization family (DFC2 + DFC_FC0) |
| ANALYSIS | VOS, PUB0 |
| DROP | DFC1-R2 adaptive, IGF, CGC, KFC, AGS |

## 2. Pending placeholders
| Pack | Status | Role |
|---|---|---|
| TFC_FC0 | RUNNING (GPU; not part of this pack) | TFC production convergence/timing |
| DFC2 | PREPARED_QUEUED (no GPU) | DFC universal-fixed generalization |
| DFC_FC0 | PREPARED_QUEUED REQUIRES_VALIDATION (no GPU) | DFC full 30K convergence |
| PUB0 | COMPLETE (CPU-only) | Bottleneck/timing ledger |

## 3. Method ledger (cohorts NOT merged)
See PAPER1_EVIDENCE_MATRIX.json for full 17-field per-method records and PAPER1_MAIN_TABLE_DRAFT.json for the draft table. Key points:
- **TFC**: strongest renderer-level result (ellipse-exact footprint). Intersection pooled -38.2% (11 scenes). Legacy 1.3564x = footprint-stage fwd+bwd; PBR0 reconciled native full-step 1.1988x; PUB0 current-cohort full-step 1.0317x. Renderer-local only; total 30K wall-clock NOT_MEASURED (TFC_FC0 pending). 'pixel-identical' withdrawn -> 'numerically indistinguishable'.
- **K50-B1**: only method with real 30K full-step speedup (room 1.082x / bicycle 1.244x / garden 1.289x). NEEDS_REVALIDATION: canonical-pipeline revalidation, multi-seed, LPIPS, total-wall-time required.
- **DFC1-R1 saved-T**: SUPPORTING correctness fix (548x grad-error reduction at t=0); no standalone speedup.
- **DFC family**: PENDING. DFC2 DOES_NOT_GENERALIZE_UNIVERSALLY; DFC1-R2 FAIL_CLOSED on dense scenes; DFC_FC0 queued.
- **VOS/IGF/CGC/KFC/AGS**: DROP or ANALYSIS per authoritative verdicts (preserved negatives).

## 4. Cohort integrity
- Renderer-frozen-PLY cohorts (TFC/DFC/CGC/KFC/AGS) != full-training cohorts (K50-B1/VOS/IGF).
- K50-B1 stage4a (config-buggy 5K) != stage4b (clean 30K).
- EPIC-05 A100-SXM4-80GB != A100-PCIE-40GB (stage4a). K50-B1 is HISTORICAL multi-GPU (not per-run comparable).

## 5. Open gaps
See PAPER1_OPEN_GAPS.json (10 gaps). BLOCKER: G1 (renderer-local != training speedup), G2 (no measured total-training-time speedup for any method).

## 6. Publication readiness
NOT publication-ready. TFC_FC0, DFC2, DFC_FC0 remain unresolved; K50-B1 needs revalidation; no method has a measured total 30K production training wall-clock.
