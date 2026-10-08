# FINAL PROTOCOL LOCK — Terminology & Metric Definitions

**Generated:** 2026-10-02  |  **Mode:** CPU / source-audit / artifact-only. NO GPU work. NO new runs. NO K50.
**Authority:** PUB0_TIMING_LEDGER.json, PBR0_VERDICT.json, TFC2_VERDICT.json, PAPER0_HARDENED, DFC2_VERDICT.json, DFC1R1_VERDICT.json.

This document freezes the ONLY terminology and metric definitions future reports and paper tables MUST use.

## 0. Hard rules (non-negotiable)
- Never call renderer-local timing "training speedup".
- Never call decomposed proxy timing "production full-step".
- Never infer pure optimizer cost from residual subtraction of independent loops.
- Production training headline MUST use directly measured wall-clock.
- Renderer workload reduction and training-time reduction MUST remain separate metrics.
- Keep all hardware/software cohorts explicit; never merge numbers across cohorts.

## 1. Frozen terminology
| # | Term | Definition | Canonical source | MUST NOT be called |
|---|---|---|---|---|
| 1 | intersection-only timing | Tile-intersection stage only (count+emit+offset, excl. sort/rasterize) | PUB0 bucket1; TFC2 prof_isect_speedup | training-step / full-step |
| 2 | forward rendering pipeline | preprocess + intersection + sort + rasterize (forward only) | PUB0 bucket2; TFC2 fwd_*_s | fwd+bwd / training-step |
| 3 | renderer forward+backward | fwd pipeline + backward kernels on a FROZEN PLY (no optimizer/densify) | PUB0 bucket3; TFC2 prof_e2e_speedup | production full training step |
| 4 | production iteration wall-clock | ONE full production iteration (preprocess+isect+sort+rasterize+bwd+densify+optimizer), same-run | PUB0 bucket4 (current cohort) | residual/subtraction estimate |
| 5 | total 30K production training wall-clock | End-to-end wall-clock for 30K iters, seed-fixed, same cohort (HEADLINE) | PENDING (TFC_FC0/DFC_FC0) | per-iter geomean / Amdahl ceiling |

## 2. Authoritative metric definitions
- **intersection reduction (pooled)** = 1 - sum(ellipse_isect)/sum(prodAABB_isect) over all cameras.
- **intersection reduction (mean-per-camera)** = mean over cameras of (1 - ellipse_i/prodAABB_i). Report SEPARATELY from pooled.
- **PSNR/SSIM/LPIPS deltas** = method - baseline on the SAME held-out set; '+' = better.
- **gradient cosine** = cos(g_method, g_ref); **relL2** = ||g_method-g_ref||/||g_ref||.
- **Gaussian-count trajectory** = final + clone/split/prune counts over 30K densification (full training only).
- **integrated Gaussian population** = time-integral of count over trajectory (workload metric).
- **total training speedup** = baseline_total_30K_wall / method_total_30K_wall (directly measured).

## 3. Authoritative values (as of 2026-10-02)
- TFC2 pooled intersection reduction (11 scenes): 0.38213; PBR0 native: 0.3927; TFC0 room 8-cam pooled: 0.42415.
- TFC legacy 1.3564x = footprint-stage fwd+bwd (TFC2_PROFILING); PBR0 reconciled native full-step = 1.1988x; PUB0 current-cohort TFC full-step = 1.0317x.
- Total 30K training wall-clock: NOT_MEASURED for every method (K50-B1 per-iter only).

## 4. Required cohort tags
EPIC05_A100_REMOTE · TFC_DFC_EPIC05 · VOS1_EPIC05 · K50B1_HISTORICAL · C51_STAGE4A_ROOM · C51_STAGE4B_3SCENE · FASTERGSFUSED_C0_30K.

## 5. PBR0_H1 harness audit (incorporated)
- PBR0 M4 'full training step' is a DECOMPOSED STAGE-TIMING MICROBENCHMARK (sum of 3 independently-best-timed stages), NOT a faithful production 3DGS iteration.
- Optimizer scope 4 groups / 11 elements (NO SH); densification/pruning EXCLUDED; opt.step timed in isolation.
- Valid for RELATIVE A-vs-B-vs-C footprint effect only (1.1988x native / 39.3% intersection-reduction). Absolute '0.5% optimizer' / '41 ms' are NOT production-representative (production optimizer ~30-41%).
- Rule: PBR0-style decomposed timing must NEVER be labeled a production full training step.

## 6. HARDENING_V2 corrections
- Production full-step / total-training speedup = Tier-A, PENDING TFC_FC0 / DFC_FC0.
- PBR0 1.1988x native = Tier-C decomposed proxy (PBR0-M4 sum-of-stages; PBR0_H1 shows it is NOT a production iteration). REMOVED from production_iteration_speedup.
- PUB0 current-cohort 1.0317x (TFC) / 1.0539x (DFC) = Tier-C decomposed timing proxy; NOT an authorized production training speedup.
- TFC 1.3564x pooled = FORWARD-RENDERING-PIPELINE timing (renderer fwd+bwd on frozen PLY), NOT a production training speedup.
- TFC gradient: baseline self-noise negligible/deterministic; A-vs-B gradient difference is REAL (exceeds self-noise envelope); legacy cosine>=0.99999 gate PARTIALLY fails; not noise-floor-equivalent; training impact PENDING TFC_FC0.

## 6. HARDENING_V2 corrections
- Production full-step / total-training speedup = Tier-A, PENDING TFC_FC0 / DFC_FC0.
- PBR0 1.1988x native = Tier-C decomposed proxy (PBR0-M4 sum-of-stages; PBR0_H1 shows it is NOT a production iteration). REMOVED from production_iteration_speedup.
- PUB0 current-cohort 1.0317x (TFC) / 1.0539x (DFC) = Tier-C decomposed timing proxy; NOT an authorized production training speedup.
- TFC 1.3564x pooled = FORWARD-RENDERING-PIPELINE timing (renderer fwd+bwd on frozen PLY), NOT a production training speedup.
- TFC gradient: baseline self-noise negligible/deterministic; A-vs-B gradient difference is REAL (exceeds self-noise envelope); legacy cosine>=0.99999 gate PARTIALLY fails; not noise-floor-equivalent; training impact PENDING TFC_FC0.
