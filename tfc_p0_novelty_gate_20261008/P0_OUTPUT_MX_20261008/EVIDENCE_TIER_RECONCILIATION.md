# EVIDENCE TIER RECONCILIATION — TFC-P0-MX

Per `DSH_EXECUTION_PROMPT.md` Stage 0B: this is a per-claim SOURCE / COMMIT / DATA COHORT / MEASUREMENT SCOPE / METRIC / STATUS / BIAS ledger. **No merged speedup headline.** Cohorts are kept separate and never composed.

## A. Cohort definitions (Stage 0A.3 — do not compare as one benchmark)

| cohort id | hardware | software | hosts TFC_FC0 30K? | hosts AccuTile30K 30K? |
|---|---|---|---|---|
| `EPIC05_A100_SXM4` | 8× A100-SXM4-80GB | torch 2.10, gsplat fork `937e2991…` (v1.5.3) | **YES** (vs AABB baseline) | no |
| `MX_A100_PCIE_40GB` | 8× A100-PCIE-40GB (bms-39468022-001) | TFC2/PBR0: torch 2.7.1+cu118; AccuTile30K: anysplats torch 2.4.1+cu124 | no | **YES** (B1A vs B1) |
| `gsplat_v1.5.3` | — | per-axis AABB tile enumeration | baseline | baseline (B1) |
| `gsplat_post_v1.5.3_accutile` | — | upstream main AccuTile, PR#927 (`28e794ca`/merge `3d4f9027`), local `1f331b6d…` | n/a | strong baseline (B1A/B2) |

The TFC_FC0 30K cohort (SXM4, torch 2.10, vs-AABB) and the AccuTile30K 30K cohort (PCIE, torch 2.4.1, vs-AABB) are **different hardware + torch + baseline**; their wall-clock ratios **cannot be composed** into a TFC-vs-AccuTile claim.

## B. Claim-by-claim tier ledger

### B1. 38.2% Gaussian–tile pair reduction (TFC2, 11 scenes / 88 camera records)
- **SOURCE:** `_tfc2_oracle/TFC2_VERDICT.json` (`measured_pooled = 0.3821339645678883`, tiles_eliminated 354 338 900).
- **COMMIT:** tfc1_rowwise.cu `a44e4ee6…`, tfc2_ablation.cu `3c9b2e0a…` (both frozen, re-hashed live).
- **COHORT:** MX, EPIC-05 gsplat v1.5.3.
- **MEASUREMENT SCOPE:** frozen-state tile-intersection **count** vs per-axis AABB, 11 scenes.
- **METRIC:** intersection-pair count reduction.
- **STATUS:** OBSERVED_IN_ARCHIVE.
- **TIER:** **TIER_C_FROZEN_PLY** (a count, not GPU time).
- **BIAS/MISSING:** is NOT GPU time; against the AABB baseline only.

### B2. 1.3564× (TFC2 legacy)
- **SOURCE:** `TFC2_VERDICT.json` `profiling_mean_e2e_speedup = 1.3564008423812486`.
- **COHORT:** MX, decomposed profiling (isect 1.465 / raster 1.381 / e2e 1.356).
- **MEASUREMENT SCOPE:** forward-rendering-pipeline timing.
- **METRIC:** forward-pipeline speedup.
- **STATUS:** HISTORICAL.
- **TIER:** **TIER_C_FORWARD_RENDERING_PIPELINE**.
- **BIAS:** NOT end-to-end renderer F+B, NOT production training wall-clock. `PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_OPEN_GAPS.json` demotes it to "forward-rendering-pipeline timing only."

### B3. 1.1988× (PBR0 native)
- **SOURCE:** `_pbr0_oracle/PBR0_TIGHTENING_ORACLE.json` `speedup_m4_full_training_step_s = 1.1988`; `PBR0_VERDICT.json` `realized_full_step_speedup_native = 1.1988`.
- **COHORT:** MX A100-PCIE, torch 2.7.1+cu118, gsplat v1.5.3.
- **MEASUREMENT SCOPE:** `m4_full_training_step_s` = **sum of 3 independently-best-timed stages** (fwd pipeline + backward + opt.step).
- **METRIC:** decomposed step-sum proxy.
- **STATUS:** NOT_PRODUCTION.
- **TIER:** **TIER_C_DECOMPOSED_PROXY**.
- **BIAS:** `PBR0_H1_HARNESS_AUDIT.json` proves M4 is a **DECOMPOSED STAGE-TIMING MICROBENCHMARK**, not a faithful production iteration (isolated opt.step between two syncs; 11 elems/Gaussian no SH; sum-of-stages denominator). HARDENING_V2 supersedes earlier production labelling.

### B4. 1.0317× (PUB0 current-cohort TFC full-step)
- **SOURCE:** `_pub0_ledger/build_pub0.py` / `PUB0_AMDAHL_LEDGER.json` (current-cohort TFC full-step bucket).
- **COHORT:** current cohort (torch 2.9/2.10, gsplat EPIC-05).
- **MEASUREMENT SCOPE:** decomposed Amdahl bucket.
- **METRIC:** decomposed full-step proxy.
- **STATUS:** NOT_PRODUCTION.
- **TIER:** **TIER_C_DECOMPOSED_PROXY**.
- **BIAS:** not an authorized production training speedup.

### B5. TFC_FC0 30K total wall geomean 1.0594×
- **SOURCE:** `_tfc_fc0_pack/deliverables/TFC_FC0_TIMING.json` (`speedup_30k_geomean = 1.059444702919323`; room 1.04668, garden 1.06790, bicycle 1.06387).
- **COMMIT:** gsplat fork head `937e2991…`, reference_v1 trainer `98753fbf…`, tfc1 `a44e4ee6…` (all re-hashed live, intact; parity gate PASS mode0==stock bit-identical).
- **COHORT:** EPIC-05 A100-SXM4-80GB, torch 2.10, gsplat v1.5.3.
- **MEASUREMENT SCOPE:** **real production 30K training wall-clock** (CUDA sync only before loop and after iter 30000; includes eval+checkpoint, identical A/B schedule).
- **METRIC:** TFC(mode1) / baseline(mode0) 30K wall ratio.
- **STATUS:** COMPLETE (resolves the "UNKNOWN_CHECK_LIVE_MX" Oct-2 ledger entry — the run finished and its artifacts are present and hash-verified).
- **TIER:** **TIER_A** (production wall-clock).
- **BIAS:** baseline is **gsplat-1.5.3 AABB (B1, weak)**, NOT AccuTile. Quality gate **FAIL** (garden dPSNR −0.0717 dB) ⇒ fail-closed. **Not composable with AccuTile30K.** Does NOT represent TFC-vs-strong-baseline headroom.

### B6. AccuTile30K 30K total wall geomean 1.1039×
- **SOURCE:** `reports/accutile30k/results.json` + `13scene-training-speed.md` (all-13 geomean training_speedup 1.1039×; per-scene bicycle 1.109, room 1.103, garden 1.113, …).
- **COMMIT:** B1 gsplat v1.5.3 pip `937e2991…`; B1A `/mnt/storage_pool/.../gsplat-true-accutile-v153` (strip-based AccuTile); upstream `28e794ca…`/PR#927 `3d4f9027`, v1.5.3 port SHA `33292a08…`.
- **COHORT:** MX A100-PCIE-40GB, anysplats torch 2.4.1+cu124, 13 scenes, seed 42, 30K.
- **MEASUREMENT SCOPE:** real production 30K wall-clock, B1A (AccuTile) vs B1 (AABB), single homogeneous cohort.
- **METRIC:** training_speedup = T_B1 / T_B1A.
- **STATUS:** COMPLETE (B1_REUSE audit: env-matched re-run; 13/13 complete).
- **TIER:** **TIER_A**.
- **BIAS:** This is **PRIOR ART** (Speedy-Splat SnugBox/AccuTile), classified `B1A_STRONG_PASS` = "FROZEN ENHANCED BASELINE", **NOT** a TFC contribution. The 3-scene SXM4 renderer figure (`accutile-a100-final.md`) fwd 1.248× / bwd 1.118× / tot 1.178× is renderer-only, different cohort. Quality: mean ΔPSNR +0.110 dB but garden −0.714 dB (one scene > −0.5 dB gate).

### B7. TFC-vs-AccuTile (B2) intersection headroom
- **SOURCE:** `_tfc_prior_art_audit/TFC_VS_ACCUTILE_MEMBERSHIP.json` (10 500 real cases, `tfc_only_total = 0`, `acc_only_total = 2257`, `max_acc_only = 1`); `TFC_CURRENT_GSPLAT_BASELINE_AUDIT.json` (advantage vs current-main ~0); `TFC_COMPLEXITY_OPERATION_AUDIT.json` (`O(rows) ≥ O(min(rows,cols))`).
- **MEASUREMENT SCOPE:** device-side CUDA set-equality + analytic asymptotic comparison (no timing).
- **STATUS:** DERIVED/RECOMPUTED (membership test is deterministic; timing explicitly NOT issued — GPU contended, 3 procs/GPU).
- **TIER:** structural comparison (not a timing claim).
- **BIAS:** no production TFC-vs-B2 wall-clock exists; recorded `NOT_MEASURED`, no speedup fabricated.

## C. Promotion rule applied
Only **B5** and **B6** are TIER-A production wall-clock. B5 is TFC-vs-weak-AABB and quality-FAIL; B6 is AccuTile-vs-AABB and is prior art. **Neither supports a TFC-vs-strong-baseline production speedup.** B1–B4 remain TIER-C and are forbidden as training-speedup headlines (`DECISION_PROTOCOL.md`).

## D. Legacy ledger reconciliation (`evidence/LEGACY_EVIDENCE_LEDGER.json`)
- `TFC2_Gaussian_tile_pairs_reduction 38.2%` → confirmed TIER_C_FROZEN_PLY, OBSERVED.
- `TFC2_legacy_1.3564x` → confirmed TIER_C_FORWARD_RENDERING_PIPELINE, HISTORICAL.
- `PBR0_1.1988x` / `PUB0_1.0317x` → confirmed TIER_C_DECOMPOSED_PROXY, NOT_PRODUCTION (HARDENING_V2).
- `TFC_FC0_30K_total_wall = null / UNKNOWN_CHECK_LIVE_MX` → **RESOLVED**: run is complete; TIER_A value = geomean 1.0594× vs AABB baseline (quality gate FAIL). The Oct-2 "pending" was stale.
- `TFC_gradient_equivalence` → REQUIRES_CAUSAL_VALIDATION; the A-vs-B difference is REAL and accumulates (garden −0.0717 dB); blanket noise-floor explanation disallowed.
