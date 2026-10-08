# PUBLICATION CONVERGENCE REPORT — HARDENING_V2

**Generated:** V2 (supersedes V1; V1 preserved unchanged at ../PUBLICATION_CONVERGENCE_PACK)  |  **Mode:** CPU/source-audit/artifact-only. NO GPU work.

## V2 corrections applied (no GPU)
1. **PBR0 1.1988x delisted from production.** Reclassified as Tier-C decomposed proxy / PBR0-M4 (PBR0_H1: M4 is a sum-of-stages microbenchmark, not a production iteration). Production TFC speedup = PENDING TFC_FC0. PUB0 1.0317x/1.0539x likewise Tier-C decomposed proxies.
2. **TFC gradient wording corrected.** Baseline self-noise (A-A) negligible/deterministic (aa_cos_median ~1.0). A-vs-B gradient difference is REAL (ab_vs_aa_envelope REAL_DIFF; train cam151 ab cos 0.99939 < aa 0.99995; means cos 0.99728). Legacy cosine>=0.99999 gate PARTIALLY fails (TFC2 min 0.999954). NOT noise-floor-equivalent. Training impact PENDING TFC_FC0.
3. **TFC 1.3564x scope corrected.** = FORWARD-RENDERING-PIPELINE timing (renderer fwd+bwd on frozen PLY), NOT a production training speedup.
4. **DFC2 reconciled to COMPLETED.** Authoritative raw out.jsonl (144 per-camera rows) recomputes the deliverable aggregates EXACTLY (quality-loss and splat-reduction max diff ~1e-15 / 0). DFC2_SOURCE_LOCK 'PREPARED_QUEUED' status field is STALE/superseded. Verdict: DOES_NOT_GENERALIZE_UNIVERSALLY (quality PASS all 6 scenes both thresholds; splat reduction 5.8-23.4%; no uniform 10% workload floor). DFC_FC0 remains queued.
5. **PUB0 removed from QUEUED** (COMPLETE CPU ledger).

## COMPLETED
- TASK1 protocol lock (production metrics = Tier-A PENDING; Tier-C taxonomy for PBR0-M4/PUB0).
- TASK2 evidence matrix / gaps / main-table-draft / report (all 6 rebuild targets).
- TASK3 Figure A/B/C/E data + Figure D PENDING; indexes + captions (Figure B/C reconciled).
- TASK4 REPRO source index / environment / commands / caveats.
- TASK5 K50_CANONICAL_PREREG (protocol frozen, NOT executed).

## RUNNING
- TFC_FC0 (current GPU priority; not started here).

## QUEUED
- DFC_FC0 200-iter smoke + 30K (awaiting isolated node). K50 canonical revalidation (downstream of PUB0; explicit authorization required).

## BLOCKED
- No Tier-A measured total-30K production training wall-clock (PENDING TFC_FC0/DFC_FC0).
- K50 revalidation waits for PUB0 review + authorization.
- DFC_FC0 train_dfc.py unverified.
- DFC2 COMPLETED negative: no universally-generalizing fixed threshold.

## Publication readiness
**NOT publication-ready.** Production TFC/DFC full-step and total 30K wall-clock are Tier-A PENDING. PBR0/PUB0 numbers are Tier-C decomposed proxies, not production training speedups. TFC 1.3564x is forward-rendering-pipeline timing only. DFC2 is a COMPLETED measured negative. K50-B1 requires revalidation.

## Integrity
- V1 pack preserved unchanged at ../PUBLICATION_CONVERGENCE_PACK. All numbers trace to authoritative artifacts; DFC2 completion verified by exact recomputation from raw out.jsonl.
