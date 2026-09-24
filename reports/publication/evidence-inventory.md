# Publication Evidence Inventory

**Report ID:** publication-evidence-inventory
**Date:** 2026-09-24
**Mode:** read-only audit. No GPU timing. No production CUDA change. No new runs.
**Machine-readable source of truth:** `artifacts/publication-audit/evidence_inventory.json`

---

## 1. Scope and record schema

One authoritative inventory of every result record that could be cited in the paper.
Each record carries:

```text
claim | experiment | artifact path | report path | binary SHA / commit | scene(s)
resolution | seed | GPU | evidence class | validity | publication usability
```

Evidence classes (fixed vocabulary):

```text
FULL_TRAINING_PUBLICATION | MATCHED_TRAINING | CONTROLLED_REPLAY | MICROBENCH
MECHANISM_ORACLE | DIAGNOSTIC_ONLY | INVALID_SUPERSEDED
```

Publication-usability vocabulary:

```text
PUBLICATION_GRADE | DIAGNOSTIC | SUPERSEDED | INVALID
```

---

## 2. Records

| ID | Claim | Experiment | Evidence class | Validity | Usability |
|---|---|---|---|---|---|
| E01 | C0_V3_FINAL30K vs B1A_ACCUTILE matched 13-scene x 30K full training: **1.0685x geomean speedup**, quality neutral | FINAL-30K | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE |
| E02 | C0 V3 renderer-level nested F+B speedup vs V0 (order-balanced) | C0-T2 | CONTROLLED_REPLAY | VALID | PUBLICATION_GRADE |
| E03 | C0-T1 nested F+B engineering timing closure | C0-T1 | CONTROLLED_REPLAY | VALID_AS_ENGINEERING_CLOSURE | SUPERSEDED (as publication timing) |
| E04 | Original C0 timing / E_compose=1.0 / zero-variance summaries | C0 (original) | INVALID_SUPERSEDED | INVALID | SUPERSEDED |
| E05 | F9 removes projected-state materialization (exact forward module) | F9 standalone 5K promotion gate + timing | MATCHED_TRAINING | VALID | PUBLICATION_GRADE_FOR_MODULE_CLAIM_ONLY |
| E06 | H8-MR reformulates geometry differentiation (exact backward module) | H8-T0 direct backward timing closure | MICROBENCH | VALID | PUBLICATION_GRADE_FOR_MODULE_CLAIM_ONLY |
| E07 | SCALAR_ADJOINT removes redundant adjoint arithmetic (exact backward module) | H2-BWD-2 validation + C0 nested contribution | MICROBENCH | VALID | PUBLICATION_GRADE_FOR_MODULE_CLAIM_ONLY |
| E08 | Native HiGS hierarchy does not transfer cleanly to exact trainable semantics | P2-1A R3 semantic repair (hard stop) | MECHANISM_ORACLE | VALID (negative) | PUBLICATION_GRADE (negative claim) |
| E09 | Hierarchical reuse alone does not imply profitable backward acceleration | P2-0 / P2-1C / H7-B0 / P3-0 | DIAGNOSTIC_ONLY | VALID (negative) | PUBLICATION_GRADE (negative) / APPENDIX |
| E10 | HAR is not profitable on the final C0 stack | P3-H (DSH-H accumulation ceiling) | DIAGNOSTIC_ONLY | VALID (negative) | APPENDIX_ONLY |
| E11 | DSH-E P3 temporal speedup model as a HAR projected gain | DSH-E P3 | INVALID_SUPERSEDED | INVALID_SUPERSEDED_MODEL | INVALID (DO_NOT_USE) |
| E12 | E2 VJP->Adam fusion opportunity | E2-P0 | DIAGNOSTIC_ONLY | VALID (negative) | APPENDIX_ONLY |
| E13 | B1A AccuTile 13-scene record vs original gsplat | AccuTile 13-scene cohort | FULL_TRAINING_PUBLICATION | VALID_FOR_B1A_VS_ORIGINAL_RECORD_ONLY | PUBLICATION_GRADE (separate claim) |
| E14 | External training systems (Speedy-Splat / Faster-GS / FastGS) 13-scene comparison | C42 13-scene external baselines | FULL_TRAINING_PUBLICATION | VALID_BUT_PROTOCOL_DIFFERS | CONDITIONAL |
| E15 | Closed negative branches H4/H5/H6/H7/R6-A | H4-0R/4-P0, H5-0R/1, H6-0/0R, H7-B0, R6-A | DIAGNOSTIC_ONLY | VALID (negative) | APPENDIX_ONLY |

### Key pinned identities

| Arm / artifact | Identity |
|---|---|
| B1A_ACCUTILE `.so` | `0471fbd95cae4cfa658bc1e2e3f7abe801c6f58a3672e681a6d16af84279986b` |
| C0_V3_FINAL30K `.so` | `9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91` |
| shared core gsplat `.so` | `361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98` |
| scene-packing `.so` | `e032f943...` (shared by design) |
| legacy research/frozen C0 | `7ca1c6bf6c8e4307ecb8fcdbcaf2953bf95305d2c9814f84f3fb5130859301f6` |
| C0-T2 renderer commit | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| P3-H diagnostic binaries | V0 `a9265f1b...`, V1 `cd345921...`, V2 `84dd8dc9...`, V3 `57d89de9...`, V4 `4e3d02c3...` |

No per-run git commit hash is stored for FINAL-30K; identity rests on the per-arm binary SHA (stronger for the compiled artifact). See `provenance_audit.json` D3.

---

## 3. Separation

### 3.1 Publication-grade evidence
`E01, E02, E05, E06, E07, E08, E13, E14`

- E01 is the headline renderer-level full-training result.
- E02 is the renderer-level controlled replay (3 scenes).
- E05/E06/E07 are module-mechanism evidence, **module claim only** — none of them may be substituted into a composed-context number.
- E08 is a publication-grade **negative** claim.
- E13 is a separate baseline record (B1A vs original gsplat, older environment).
- E14 is publicable only with an explicit protocol caveat (see §4).

### 3.2 Diagnostic evidence
`E09, E10, E12, E15` — appendix-grade. Valid negative results, not main-paper speedup evidence.

### 3.3 Superseded evidence
`E03` (replaced as publication timing by E02: 15.12/15.87/21.16% -> 16.54/14.74/20.62%), `E04`.

### 3.4 Invalid evidence
`E04` (old-C0 `7ca1c6bf`, `compute_abs=false`, E_compose=1.0 tautology, zero-variance summaries — DIAGNOSTIC_PROTOCOL_MISMATCH),
`E11` (DSH-E P3 temporal speedup model = **INVALID_SUPERSEDED_MODEL**, DO_NOT_USE).

---

## 4. Protocol cohorts are NOT mutually mergeable

| Cohort | Scenes | Seeds | Device | Protocol |
|---|---|---|---|---|
| FINAL-30K (E01, E02, E13) | 13 | 1 (seed 42) | A100-PCIE-40GB | max_side 1920, reference_v1, full-res eval, torch 2.9.1+cu128 |
| training-all (system level) | 11 (**no flowers/treehill**) | 3 | A100-SXM4-80GB | official eval split, from-SfM |
| C42 external (E14) | 13 | 1 | A100-PCIE-40GB (mx) | NeRFICG/FastGS/Speedy-Splat native, DSSIM on 0.5x area-downsample, garden uses manually-created COLMAP data |

Any "C0 V3 vs Faster-GS/FastGS/Speedy-Splat" speedup would be cross-cohort and is **not supported by current evidence**.

---

## 5. Coverage gaps

**Missing metadata globally**
- per-run git commit hash (binary SHA is recorded per run instead)
- per-run GPU uuid / host for the `C0_V3_FINAL30K` arm (only cohort-level sm_80 A100 identity)

**Evidence absent**
- multi-seed full-training quality under the FINAL-30K protocol (seed 42 only)
- dedicated eps2d 0.1-vs-0.3 sensitivity sweep (only the absgrad gradient-equivalence gate D1/D2)
- cross-GPU portability runs (no 4090 / 5070 measurement of any kind)
- cumulative A0/A1/A2/A3 full-training ablation (only A0 and A3 at 30K; C0-T2 V0/V1/V2/V3 renderer replay on 3 scenes)
- no C0 V3 run against any external system under any protocol

---

## 6. Publication-phase records (2026-09-24, appended)

New evidence from the publication phase (P1/P2/P4/P5/P6). These inherit the schema and
vocabulary above. Statuses update §5's absent-evidence list item by item.

| ID | Claim / record | Experiment | Evidence class | Validity | Usability |
|---|---|---|---|---|---|
| E16 | B1 pristine-gsplat matched gate: wiring PASS; accutile garden effect −0.642/−0.806 dB (s42/s43), wall edge +2.2–4.6% | P1 B1 gate (3 scenes + garden seed pairs) | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE (supersedes E13's role for the accutile comparison; E13 remains a record of the older cohort) |
| E17 | B0 original-Graphdeco matched cohort: 2.42× slower on 13/13 scenes (wall geomean B1A/B0 0.4142); mean ΔPSNR −0.899 dB with bidirectional signed-grad failures (under-densifies stump/flowers/bicycle/treehill; over-densifies drjohnson +1.58 dB at N 1.54×) | P1 B0 gate (3 scenes) + 13-scene expansion (counter clean re-run after 1 contaminated attempt) | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE |
| E18 | Cumulative ladder A0→A1→A2→A3 at 30K, one binary, frames bit-identical, ~0.45% total wall (13 scenes: geomean 1.0045, 10/13 faster), quality-neutral (12/13 within ±0.15 dB) | P2 Stage A (9/9) + Stage B (30/30) — COMPLETE | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE |
| E19 | eps2d 2×2: MATERIAL_CONFOUND (quality −0.100 dB), wall 1.07%; headline bound 1.057×–1.0685× | P4 (9/9 clean cells) | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE |
| E20 | External systems 13-scene × {native, c42} × 3 systems, 26/26 each, commits + submodules pinned, garden collapse 12.4–13.9 dB | P5 (re-verified E14 cohort + provenance) | FULL_TRAINING_PUBLICATION | VALID_BUT_PROTOCOL_DIFFERS | CONDITIONAL (same class as E14; garden-data caveat below) |
| E21 | Garden accutile seed pairs COMPLETE (s42/s43/s44): effect +0.642/+0.806/+1.132 dB, wall edge +1.4–3.9%, sign-consistent both axes | P6 pre-arm | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE (garden scope) |
| E22 | P6 multi-seed subset COMPLETE (4 scenes × {B1A,C0} × s43/s44 + s42 refs): all 12 per-scene speedups > 1; per-seed geomeans 1.0705/1.0735/1.0561; quality straddles zero on 3/4 scenes; drjohnson reproducible (+1.102/+0.787/+1.597) | P6 (16 runs; 1 contamination re-run clean, forensics kept) | FULL_TRAINING_PUBLICATION | VALID | PUBLICATION_GRADE |

**§5 gap resolution:** eps2d sweep → closed by E19; cumulative ablation → closing with
E18; multi-seed → closing with E21/E22; "C0 vs external under any protocol" → E20
remains cross-cohort SYSTEM_LEVEL only (unchanged; §4's non-mergeability rule stands).

**E20 garden-data caveat (carried into C-15):** the C42 external cohort's garden runs
use **manually-created COLMAP data** (per §4) and 0.5×-area-downsampled DSSIM — not the
matched benchmark's garden. The external garden collapse (12.4–13.9 dB) is therefore
evidence about their pipeline on THEIR garden data, not a controlled numerics failure
on the standard garden scene. The report states this; the collapse is never used as a
matched-benchmark contrast.