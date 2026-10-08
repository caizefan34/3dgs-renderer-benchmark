# PAPER0 HARDENED REPORT - Publication-Readiness Ledger (Hardened Snapshot)

**Generated:** 2026-10-02  
**Status:** HARDENED SNAPSHOT. The original `../PAPER0_EVIDENCE/` is a SNAPSHOT, NOT the final publication ledger. Historical verdicts are preserved (not overwritten).  
**Mode:** read-only synthesis. NO GPU work. NO new runs. NO cohort merging.

## 0. Hardening changes vs the SNAPSHOT

- **Snapshot marker**: original PAPER0 declared a SNAPSHOT; this directory is the working ledger.
- **TFC terminology**: 1.3564x marked LEGACY forward-pipeline (footprint-stage) timing - NOT full-E2E, NOT fwd+bwd of a training step. 'pixel-identical' withdrawn -> 'numerically indistinguishable' (PBR0: max abs RGB diff A-vs-B 3.2e-4; B==C bitwise; PSNR loss 0.0 dB). PBR0 reconciled native full-step = 1.1988x; PUB0 current-cohort TFC full-step = 1.0317x.
- **K50-B1**: MAIN -> MAIN_CANDIDATE, with 5 required blockers (canonical-pipeline revalidation, multi-seed, LPIPS, measured total training wall time, Stage4B-only primary evidence).
- **DFC family split**: DFC1-R1 saved-T = SUPPORTING_PASS; DFC1-R2 adaptive = DROP; DFC2 universal fixed = RUNNING; DFC-FC0 convergence = RUNNING; family role = PENDING.
- **Pending-result placeholders**: PBR0 (COMPLETE_RECONCILIATION), DFC2_GENERALIZATION (RUNNING), DFC_FC0_FULL_CONVERGENCE (RUNNING), PUB0 (COMPLETE).

## 1. Role summary (hardened)

| Role | Methods |
|---|---|
| MAIN_CANDIDATE | TFC, K50-B1 |
| SUPPORTING_PASS | DFC (DFC1-R1 saved-T backward fix) |
| DROP | DFC (DFC1-R2 adaptive), VOS (VOS1 hardened), IGF (IGF1), CGC (CGC1), KFC (KFC1), AGS (AGS1) |
| ANALYSIS (preserved negatives) | CGC0, IGF0, VOS0 SH-only, AGS0, KFC0 |
| PENDING | DFC family (until DFC2/DFC-FC0 resolve) |
| RUNNING | DFC2 universal fixed, DFC-FC0 full convergence |

## 2. Method ledger (hardened; cohorts NOT merged)

| Method | Role | Real CUDA | Scenes | Full-conv | Fwd | Fwd+Bwd | Full-train-step | Total-train-time | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| TFC | MAIN_CANDIDATE | yes | room,garden,bicycle,bonsai,counter,kitchen,stump,flowers,treehill,truck,train | NO | intersect kernel 1.38x faster; per-scene fwd render 1.15-1.42x | 1.17-1.51x per scene (footprint-stage fwd+bwd, TFC2_PROFILING) | RECONCILED_PENDING_PUB0 | NOT_MEASURED | TFC2_CERTIFIED (CONFIRMED; 1.3564x = legacy fwd-pipeline, reconciled by PBR0 to native full-step 1.1988x) |
| K50-B1 | MAIN_CANDIDATE | yes | room,garden,bicycle | YES_30K | NOT_APPLICABLE | MICROBENCH_ONLY | SUPPORTED_30K | IMPLIED_POSITIVE | Stage4A/4B real CUDA (30K full-step speedups; MAIN_CANDIDATE) |
| DFC | PENDING | yes | room,garden,bicycle | PARTIAL_RENDERER_STEP | SUPPORTED | SUPPORTED | ROOM_ONLY | NOT_MEASURED | family PENDING (R1 SUPPORTING_PASS / R2 DROP / DFC2 RUNNING / FC0 RUNNING) |
| VOS | DROP | no | room,garden,bicycle | YES_30K_MULTISEED | NOT_APPLICABLE | NOT_APPLICABLE | NONE | NONE | VOS1 QUALITY_FAIL (VOS0 KEEP superseded) |
| IGF | DROP | no | room,garden,bicycle | YES_30K | NOT_APPLICABLE | NOT_APPLICABLE | NEGATIVE | NEGATIVE | IGF1 DROP_NO_IMPORTANCE_FREEZE_HEADROOM |
| CGC | DROP | yes | room,garden,bicycle | NO | PARTIAL | PARTIAL_BELOW_GATE | NOT_MEASURED | NOT_MEASURED | CGC1 DROP_PRUNING_FIDELITY |
| KFC | DROP | yes | room | NO | FAIL | FAIL | NOT_RUN | NOT_MEASURED | KFC1 DROP_NO_KERNEL_GAIN |
| AGS | DROP | yes | room | NO | FAIL | NOT_REACHED | NOT_REACHED | NOT_MEASURED | AGS1 DROP_SORT_KERNEL_GAIN |

## 3. DFC family split

| Member | Status | Role | Verdict |
|---|---|---|---|
| DFC1-R1 saved-T | SUPPORTING_PASS | SUPPORTING | ADOPT_DFC1R1_SAVED_T_BACKWARD |
| DFC1-R2 adaptive | DROP | DROP | FAIL_CLOSED / DROP_NO_TERMINATION_HEADROOM |
| DFC2 universal fixed | RUNNING | PENDING | PREPARED_QUEUED (no GPU) |
| DFC-FC0 convergence | RUNNING | PENDING | PREPARED_QUEUED REQUIRES_VALIDATION (no GPU) |
| **DFC family** | **PENDING** | **PENDING** | resolves when DFC2/DFC-FC0 complete |

## 4. K50-B1 required blockers (MAIN_CANDIDATE, not final-certified)

- canonical current-pipeline revalidation (Stage4B used gsplat 1.5.3 + original 3DGS CUDA rasterizer; must be revalidated on the canonical current pipeline)
- multi-seed (single seed only; no multi-seed CI)
- LPIPS not measured (framework limitation; PSNR+SSIM only)
- measured total training wall time (only per-iter mean_ms; no separate total wall-time field)
- Stage4B-only primary evidence (stage4a room-only cohort is config-buggy 5K and must not be merged)

## 5. Pending-result placeholders

| Pack | Status | Role |
|---|---|---|
| PBR0 | COMPLETE_RECONCILIATION | TFC cross-pack reconciliation + gradient noise-floor audit |
| DFC2_GENERALIZATION | RUNNING (PREPARED_QUEUED, no GPU) | DFC universal-fixed threshold on held-out scenes |
| DFC_FC0_FULL_CONVERGENCE | RUNNING (REQUIRES_VALIDATION, no GPU) | DFC full 30K baseline vs DFC1-R2 |
| PUB0 | COMPLETE | Bottleneck ledger (timing normalization + Amdahl), CPU-only |

See `PAPER0_HARDENED_PENDING_RESULTS.json` for detail.

## 6. Cohort integrity (unchanged rules)

- Never merge results from different cohorts into one number.
- Never convert renderer-local speedup into training-step speedup.
- K50-B1 stage4a (room-only, A100-PCIE-40GB, config-buggy 5K) != stage4b (3-scene, A100-80GB, clean 30K).
- Historical failed/superseded gates preserved (KFC0/AGS0/CGC0/IGF0/DFC0/VOS0/TFC0).

## 7. Artifacts (this directory)

- PAPER0_HARDENED_EVIDENCE_MATRIX.json - corrected per-method ledger.
- PAPER0_HARDENED_DFC_FAMILY.json - DFC family split.
- PAPER0_HARDENED_PENDING_RESULTS.json - PBR0/DFC2/DFC-FC0/PUB0 placeholders.
- PAPER0_HARDENED_SNAPSHOT.json - snapshot marker (original PAPER0 = SNAPSHOT).
- PAPER0_HARDENED_REPORT.md - this report.
- MANIFEST.sha256 - SHA256 of all artifacts above.
