# Publication Readiness Scorecard

**Report ID:** publication-readiness
**Date:** 2026-09-24
**Machine-readable source:** `artifacts/publication-audit/readiness_matrix.json`
**Overall numeric score:** deliberately absent (per task instruction).

States: `COMPLETE` / `IN_PROGRESS` / `MISSING_BLOCKING` / `MISSING_HIGH` / `OPTIONAL`.

---

## 1. Scorecard

| Dimension | State | Evidence | Gap |
|---|---|---|---|
| **METHOD_CORRECTNESS** | COMPLETE | frozen exactness gates (`final30k-absgrad-compatibility.md` D1/D2 gradient equivalence; `h2-bwd-2r-exactness-closure.md`; `h8-0-moment-adjoint-gate.md`; `f9-1r-integration-closure.md`; `scalar-adjoint-freeze-smoke.md`); `binary_identity.json` pins both arm `.so` shas | none for the renderer exactness claim; the eps2d issue is comparability, not correctness |
| **FINAL_FULL_TRAINING** | COMPLETE | 13/13 pairs; `final_verdict.json = FINAL_PASS`; `failure_manifest.json` empty; `aggregate_performance.json publication_grade_for_aggregate = true` | single seed |
| **MATCHED_BASELINES** | COMPLETE | B1A AccuTile is the matched renderer baseline (same recipe, cohort, eval) | eps2d not equalized |
| **ABLATION** | **MISSING_BLOCKING** | A0 and A3 at 30K (13 scenes); C0-T2 V0/V1/V2/V3 renderer replay (3 scenes) | A1 (+F9) and A2 (+F9+SCALAR_ADJOINT) do not exist at 30K **anywhere in the repo** |
| **EXTERNAL_SYSTEMS** | MISSING_HIGH | `artifacts/training-all`: 210 jobs (original_3dgs 33, gsplat 48, higs_full 48, higs_proposed 48, speedy_splat 33) over 11 scenes x 3 seeds; `fastergs-c42-13scene-final.md` | no C0 V3 run against any external system under any protocol; Faster-GS/FastGS NOT_DIRECTLY_COMPARABLE |
| **EPS2D_SENSITIVITY** | MISSING_HIGH | only the absgrad gradient-equivalence gate D1/D2 | no eps2d timing sweep and no eps2d quality sweep at any scene count |
| **MULTI_SEED_ROBUSTNESS** | MISSING_HIGH | FINAL-30K is seed 42 only; the 3-seed cohort is the older protocol/method | no multi-seed C0 V3 vs B1A quality CIs under FINAL-30K |
| **CROSS_GPU** | MISSING_HIGH | `higs-paper-protocol.json` marks consumer/second-datacenter runners blocked; gencode pinned sm_80 | zero non-A100 measurement |
| **PROVENANCE** | COMPLETE | `provenance_audit.json` verdict **PUBLICATION_PROVENANCE_PASS**; 13/13 pairs; binary shas pinned; contamination downgrades applied | five documentation-level defects (D1–D5), none invalidating FINAL_PASS |
| **NOVELTY_AUDIT** | IN_PROGRESS | `prior_art_overlap.json` PARTIAL_OVERLAP + "STATED, NOT CLAIMED"; `prohibited_claims.json` | no independent literature verification; all novelty statements need `LITERATURE_VERIFICATION_REQUIRED` |
| **FIGURES** | IN_PROGRESS | 6 of 8 READY from frozen data (F1, F2, F4, F6, F7, F8) | F3 blocked on A1/A2; F5 lacks dense per-step N_GS; no plotting scripts created by this audit |
| **TABLES** | COMPLETE | T1–T6 all POPULATED from frozen aggregates (publication phase: T1 baselines ×4 arms, T2/T3 ladder 13 scenes, T4 eps2d 2×2, T5 multi-seed 3 seeds, T6 external) | regeneration: `aggregate_publication.py` → `make_figtables.py` |

---

## 2. Exact remaining blocking items

```text
BLOCKING-1  A1 (+F9) and A2 (+F9+SCALAR_ADJOINT) at 30K on a scene subset
            — OR a formal rescoping of every claim so that no per-module
              causal attribution is drawn from the 13-scene table.
            Owns: R-12, T2, F3, ABLATION dimension.

BLOCKING-2  Literature verification of the novelty boundary
            (exact projected-state elimination; sufficient-statistic moment-space
             geometry adjoint; scalar-adjoint collapse; Faster-GS / per-Gaussian
             backward distinction).
            Owns: R-02, R-03, R-06, NOVELTY_AUDIT dimension.
            No experiment can close this.

BLOCKING-3  Explicit reconciliation of the earlier HiGS confirmatory negative result
            (paper/higs-claims.json: acceleration attributable to ordinary early-stop;
             higs_full speedup ratio 0.965 vs gsplat; every HiGS candidate fails the
             quality non-inferiority margins) against the C0 V3 renderer-level claim.
            Owns: R-16. A disclosure/framing obligation, not an experiment.
```

## 3. Exact high-priority (non-blocking if the claims are scoped)

```text
HIGH-1  eps2d sensitivity arm (timing AND quality)                 — R-08, T4
HIGH-2  3-seed quality robustness on a scene subset (CIs)          — R-09, T6
HIGH-3  Minimal cross-GPU portability matrix (see §5)              — R-10
HIGH-4  State the framing explicitly (mechanism + decomposition,
        not the headline scalar) and justify AccuTile as baseline   — R-01, R-14
```

## 4. Optional items

```text
OPT-1  Dense per-step N_GS trajectories for F5 (currently summary counts only)
OPT-2  Cross-arm VRAM table (no frozen aggregate exists; would require per-run extraction + labelling)
OPT-3  Backfill per-run gpu_snapshot.json / config.json / checkpoints.json for the C0 arm (D2, R-17)
OPT-4  Non-publication banner files in artifacts/final-30k/C0_V3/ and P2_FINAL_V2/ (D1, R-18)
OPT-5  System-level table (T5) against original_3dgs / gsplat / Speedy-Splat, clearly labelled as the
       older method generation — or omit entirely
OPT-6  Excluding-drjohnson sensitivity row for the quality aggregate (R-13)
OPT-7  Inference-path statement (forward bit-identical, out of scope) (R-15)
```

---

## 5. Cross-GPU portability — planning only (nothing launched)

**No GPU resources are consumed by this audit.** Minimal matrix for later authorization:

| GPU | Scenes | Instruments | Columns |
|---|---|---|---|
| A100 (already the reference) | room, bicycle, garden | renderer-level forward / backward / F+B | mean ms, median ms, renderer peak memory |
| RTX 4090 (if available) | room, bicycle, garden | same | same |
| RTX 5070 Laptop | room, bicycle, garden | same | same |

- Metric: per-stage renderer time + renderer memory only. No 13-scene x 30K full-training run is proposed, and none is required unless separately authorized.
- Purpose: establish that the reclaimed-work direction (backward decrease, flat forward) is architecture-independent, and that sm_80-only gencode is not the source of the gain.
- Note: the older 3-seed cohort ran on **A100-SXM4-80GB**, a different A100 class; that is not portability evidence and must not be presented as such.

---

## 6a. Publication-phase ingestion (2026-09-24)

The scorecard above is preserved as its point-in-time state. The publication phase
(goal rounds 2–6) changes dimension states as follows — every change is backed by a
frozen artifact, not by rescoping:

| Dimension | Was | Now | Evidence |
|---|---|---|---|
| ABLATION | MISSING_BLOCKING | **RESOLVED (COMPLETE)** — Stage A 9/9 + Stage B 30/30 clean; 13-scene ladder final: A0→C0 geomean 1.0045, 10/13 faster, quality-neutral (12/13 within ±0.15 dB; drjohnson +0.770 labeled per-scene). Isolated single-module arms remain absent by frozen design (cumulative ladder only); the three branches sum to ~0.4% of the headline (`cumulative-ablation.md`). | `ablations/`, `a_arm_functional_gate.json` |
| EPS2D_SENSITIVITY (HIGH-1) | MISSING_HIGH | **COMPLETE** — P4 2×2 on room/bicycle/garden, 9/9 clean cells. Verdict: MATERIAL_CONFOUND on the quality axis (−0.100 dB mean vs FINAL-30K 0.070), wall not material (1.07% < half). Direction analysis bounds both biases; disclosure lands in every C0/B1A table. | `eps2d/p4_eps2d_verdict.json` |
| EXTERNAL_SYSTEMS | MISSING_HIGH | **COMPLETE (system level)** — faster-gs @3cb0b755 / fastgs @d4d33b6f / speedy-splat @b9dd42d (+ pinned submodules), each 13 scenes × {native, c42} = 26/26 runs rc=0. C0 coexists in Table 6 under the SYSTEM_LEVEL label — never cross-merged with the matched cohort. | `aggregates/p5_external.json`, `external-systems.md` |
| MATCHED_BASELINES | COMPLETE (eps2d gap) | **STRENGTHENED (COMPLETE)** — B1 (pristine v1.5.3) and B0 (original Graphdeco @54c035f) gates + FULL 13-scene expansions under the matched trainer. The eps2d gap is quantified (P4), not just disclosed. Accutile's garden effect (R-19) measured sign-consistent at 3 seeds. | `matched-baselines.md` |
| MULTI_SEED_ROBUSTNESS (HIGH-2) | MISSING_HIGH | **RESOLVED** — garden 3-seed pairs sign-consistent (quality AND wall); P6 4-scene × 3-seed subset: all per-scene speedups > 1, per-seed geomeans 1.0705/1.0735/1.0561, quality straddles zero on 3/4 scenes, drjohnson reproducible per-scene. | `multiseed/`, `p6_results.json`, `garden_seed3_table.json` |
| FIGURES | IN_PROGRESS | **PIPELINE COMPLETE** — figures A–G regenerate from frozen aggregates (`make_figtables.py`); the ladder figure (F3-equivalent) renders from the real A0→A3 chain; dense per-step N_GS (OPT-1) is figF, built from per-iteration series. All visually verified. | `figures/` |
| TABLES | IN_PROGRESS | **PIPELINE COMPLETE — ALL POPULATED** — tables 1–6 regenerate from `runs_master.json` (self-check: reproduces the FINAL-30K headline 1.0685×/+0.070 exactly). T2/T3 ladder final (13 scenes); T5 multi-seed final (3 seeds); T6 external (26 runs). | `tables/` |
| CROSS_GPU (HIGH-3) | MISSING_HIGH | UNCHANGED — out of publication-phase scope; claims remain A100-scoped. | — |
| NOVELTY_AUDIT / BLOCKING-2/3 | IN_PROGRESS / BLOCKING | UNCHANGED — literature verification and the R-16 reconciliation remain text obligations. | `reviewer-risk-audit.md` §6 |

Aggregate consequence: of the three BLOCKING items, **BLOCKING-1 is CLOSED by
experiment** (13-scene ladder complete; rescoping clause independently satisfied);
BLOCKING-2 and BLOCKING-3 remain non-experimental text obligations. Of the four HIGH
items, HIGH-1 is complete and HIGH-2 is RESOLVED (P6 + garden 3-seed); HIGH-4's
framing exists in `publication-summary.md` + `cumulative-ablation.md`; HIGH-3 is
unchanged and scoped.

---

## 6. Continuous integration policy (read-only)

As the master plan produces results (B0/B1, A1/A2, eps2d, multi-seed, external systems), validated artifacts are ingested into:

```text
artifacts/publication-audit/table_manifest.json        -> replace PENDING cells
artifacts/publication-audit/figure_manifest.json       -> F3 (A1/A2), F5 (N_GS)
artifacts/publication-audit/figure-data/*.csv          -> append rows, same pipeline
artifacts/publication-audit/readiness_matrix.json      -> reclassify dimensions
artifacts/publication-audit/reviewer_risks.json        -> close R-08 / R-09 / R-12
reports/publication/claim-evidence-map.md              -> upgrade SUPPORTED_WITH_CAVEAT -> SUPPORTED
```

Ingestion rules:

```text
1. Only validated, non-contaminated, PUBLICATION-grade runs may populate a cell.
2. A new arm must use artifacts/publication-audit/metric_definitions.json formulas; no ad-hoc numbers.
3. artifacts/final-30k/ (FINAL_PASS) is NEVER modified by ingestion.
4. Cross-cohort numbers are never merged into a single table.
5. Any cell not backed by a validated artifact stays the literal string PENDING.
```

---

## 7. Provenance verdict

```text
PUBLICATION_PROVENANCE_PASS
```

Basis: 26/26 SUCCESS, 0 contamination in the final table, 13/13 scenes in both arms, binary SHA pinned per arm, pre-registered verdict rule met, no hash mismatch, no missing scene.
Defects found (all documentation/coverage, none invalidating): D1 legacy dir coexistence (MEDIUM), D2 arm provenance asymmetry (MEDIUM), D3 no per-run commit/config SHA (LOW), D4 eps2d asymmetry (HIGH — comparability, not integrity), D5 embedded run-phase-global `contaminated_attempts` list (LOW).
`final_pass_impact: NONE`. **FINAL-30K remains untouched.**