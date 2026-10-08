# PUBLICATION CONVERGENCE REPORT — SUPPORT PACK

**Generated:** 2026-10-02  |  **Mode:** CPU / source-audit / artifact-only. NO new GPU experiments. NO K50 run. NO cohort merging.

## Summary
Executed the full PUBLICATION_CONVERGENCE_SUPPORT_PACK (Tasks 1-5 + final outputs) purely by synthesizing authoritative existing artifacts (PBR0, PUB0, TFC0/TFC2, DFC1-R1/DFC1-R2/DFC2/DFC_FC0, K50-B1 Stage4B, VOS0/VOS1, IGF/CGC/KFC/AGS). No GPU work was run; no K50 revalidation was performed; current GPU work (TFC_FC0/DFC2/DFC_FC0) was NOT started.

## COMPLETED
- **TASK1 FINAL_PROTOCOL_LOCK** — froze 5 terminology buckets + authoritative metric definitions (intersection reduction pooled vs mean-per-camera; PSNR/SSIM/LPIPS deltas; gradient cosine/relL2; Gaussian-count trajectory; integrated Gaussian population; total training speedup). Hard rules encoded (renderer-local != training speedup; no residual-derived optimizer cost; direct wall-clock headline). -> `FINAL_PROTOCOL_LOCK.json`, `FINAL_PROTOCOL_LOCK.md`.
- **TASK2 PAPER1_EVIDENCE_AUTOUPDATE** — evidence matrix (10 methods + 4 pending placeholders), 10-gap register, draft main table, report. TFC=MAIN_CANDIDATE(pending TFC_FC0); K50-B1=NEEDS_REVALIDATION; DFC1-R1=SUPPORTING; DFC family=PENDING; VOS/PUB0=ANALYSIS; IGF/CGC/KFC/AGS=DROP. No silent role upgrades.
- **TASK3 PAPER_FIGURE_DATA_PACK** — Figures A/B/C/E populated from authoritative data; Figure D all PENDING. Explicit null/PENDING for every missing point; no fabrication. -> `PAPER_FIGURE_DATA/`, `FIGURE_INDEX.json`, `FIGURE_CAPTION_FACTS.md`.
- **TASK4 REPRODUCIBILITY_PACK_PREP** — canonical/diagnostic/superseded source inventory with hashes, environment, commands, known caveats + explicit warnings. No algorithm changed.
- **TASK5 K50_REVALIDATION_PREREGISTRATION_ONLY** — froze the canonical future K50 protocol (activation gated on PUB0 + backward-bottleneck evidence; no threshold rescue after failure). NOT executed.

## RUNNING (not started by this pack)
- TFC_FC0 (current GPU priority), DFC2 (queued), DFC_FC0 (queued, requires validation).

## QUEUED
- PUB0 (COMPLETE CPU ledger), DFC2/DFC_FC0 isolated-node GPU runs, K50 canonical revalidation (downstream of PUB0).

## BLOCKED
- No measured total 30K production training wall-clock for any method (BLOCKER).
- K50 revalidation blocked until PUB0 review + explicit authorization.
- DFC_FC0 train_dfc.py unverified (200-iter smoke required).
- DFC2 no universally-generalizing fixed threshold.

## Publication readiness
**NOT publication-ready.** TFC_FC0 / DFC2 / DFC_FC0 remain unresolved; K50-B1 requires revalidation; no method has a measured total 30K production training wall-clock. Renderer-local speedups MUST NOT be presented as training speedups.

## Integrity
- All numbers trace to authoritative *_VERDICT.json / *_SOURCE_LOCK.json / *_REPORT.md / ledger JSONs.
- Cohorts kept explicit; historical negatives preserved; no invented or smoothed data.
