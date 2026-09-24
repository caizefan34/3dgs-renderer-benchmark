# Claim-to-Evidence Map

**Report ID:** publication-claim-evidence-map
**Date:** 2026-09-24
**Canonical metric source:** `artifacts/publication-audit/metric_definitions.json` (FROZEN for all tables/figures)
**Machine-readable inventory:** `artifacts/publication-audit/evidence_inventory.json`

Labels: `SUPPORTED` / `SUPPORTED_WITH_CAVEAT` / `NOT_YET_SUPPORTED` / `DO_NOT_CLAIM`.
No claim is upgraded beyond what the frozen artifacts measure.

---

## 1. The ten required claims

### C-01 — C0 gives 1.0685x geomean full-training speedup
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E01 — 13/13 pairs, 30K, seed 42, `artifacts/final-30k/aggregate_performance.json`.
- Values: geomean **1.0685072019237971**, arithmetic mean 1.0686927737009724, n=13.
- Caveats that must travel in the same sentence:
  1. It is a **ratio**, not a time reduction.
  2. eps2d is **not equalized** between arms (B1A 0.1 vs C0 0.3) — disclosed, not controlled (D4 / R-08).
  3. Single seed.
  4. Matched-renderer scope only (vs B1A AccuTile); this is not a claim against external training systems.

### C-02 — C0 is faster on 13/13 scenes
**Label: SUPPORTED**
- Evidence: E01. Every per-scene speedup ratio > 1.0. No scene regresses.
- This is the strongest and least caveat-dependent statement in the paper. It survives the eps2d dispute in the sense that the direction is uniform, though the magnitude does not.

### C-03 — Quality is approximately neutral under the matched benchmark
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E01. mean dPSNR **+0.0696 dB**, mean dSSIM **+0.000282**, mean dLPIPS **−0.001668** (negative = better).
- Caveats:
  - Single seed (seed 42). This is a point estimate, not a CI (R-09).
  - `drjohnson` is a +1.1023 dB outlier that also carries the only final-N ratio > 1.05 (1.1018). It must be reported, not suppressed, and the aggregate must be shown to be robust without it.
  - Permitted wording: **"approximately neutral to marginally positive"**. Forbidden wording: "no quality loss", "quality-improving".

### C-04 — Backward is the primary stage where measured time decreases
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E01 `phase_timing.json`. mean per-iteration ms: backward **21.596 -> 19.236 (−10.9%)**; forward 4.507 -> 4.523 (**flat**); loss 7.000 -> 6.939; densify 1.105 -> 0.990; optimizer 5.014 -> 4.925.
- Caveat: this is a **where-the-time-goes observation, not a per-module causal attribution** (registry SS35). The flat forward is *required* by an exact-work-elimination claim, so it is evidence for, not against, the correctness story.

### C-05 — F9 removes projected-state materialization
**Label: SUPPORTED**
- Evidence: E05 — exactness/oracle gates (`reports/higs/f9-0-gatherless-projected-producer.md`, `f9-1-trainable-gatherless.md`, `f9-1r-integration-closure.md`) + standalone timing (forward gain room 18.9 / bicycle 14.8 / garden 23.2%; F+B 7.63 / 5.40 / 10.56%) on A100, patch `1faa05c1...`.
- Caveat: **module-claim only.** Standalone numbers must not be presented as composed-context gains, and the standalone base was a Trainable HiGS B2 + SCALAR_ADJOINT base, not the publication A0 (B1A AccuTile). No isolated 13-scene F9 30K arm exists.

### C-06 — SCALAR_ADJOINT removes redundant adjoint arithmetic
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E07 — exactness closure + freeze smoke (`h2-bwd-2-scalar-adjoint-validation.md`, `h2-bwd-2r-exactness-closure.md`, `scalar-adjoint-freeze-smoke.md`); C0 nested contribution 1.2 / 0.1 / 0.3%.
- Caveat: the removal is exact but **small inside C0**. Isolated standalone benefit does not transfer 1:1. Present as supporting engineering, never as the core contribution.

### C-07 — H8-MR reformulates geometry differentiation using sufficient statistics
**Label: SUPPORTED (mechanism) / DO_NOT_CLAIM (causal attribution in the composed stack)**
- Evidence: E06 — H8-T0 direct backward gain room 4.31 / bicycle 4.11 / garden 3.54% (geomean 3.97%), F+B 2.37 / 2.14 / 1.80% (geomean 2.09%); gates `h8-0-moment-adjoint-gate.md`, `h8-0r-opacity-absorbed-moment.md`, `h8-mr-production.md`; patch `8fb23acca9...`.
- Caveat: no causal H8 claim in the composed stack is authorized (registry SS35). The ~45–53% whole-backward gain and the 30.81% room figures are **INVALID** / `TIMING_SUBTRACTION_ARTIFACT` and must never appear.

### C-08 — Native HiGS hierarchy does not transfer cleanly to exact trainable semantics
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E08 — P2-1A R3 hard stop. Gates: A_quadratic **PASS**, B_support **FAIL**, C_ordering PASS, D_per_splat_weight **FAIL**, E_rgb_alpha **FAIL**. `timing_authorized: false`.
- Caveat: negative result on a 3-scene renderer fixture; a valid scientific negative, but its scope is the HiGS native hierarchy, not hierarchies in general.

### C-09 — Hierarchical reuse alone does not imply profitable backward acceleration
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E09 — P2-0 opportunity map, P2-1C preflight (entry-weighted 10.8 / 4.7 / 8.2%; 100% overlap with last-id pruning; frozen-128G skippable only 0.40 / 2.00 / 1.08%), H7-B0 roofline (H7B0_WEAK), P3-0 HAR oracle.
- Class: DIAGNOSTIC_ONLY. Appendix-grade; usable as the mechanistic justification for §4 of the paper.

### C-10 — HAR is not profitable on the final C0 stack
**Label: SUPPORTED_WITH_CAVEAT**
- Evidence: E10 — P3-H DSH-H diagnostic (V0-FULL / V1-ACCUM_SINK_ALL / V2-ACCUM_SINK_GEOM / V3-ACCUM_SINK_APPEARANCE / V4-WRITE_ONLY), verdict **HAR_CEILING_WEAK**; ALL_ACCUM_CEILING ≤ **0.54%** backward time (room −0.05 / bicycle +0.54 / garden −0.10).
- Caveat: DIAGNOSTIC / 3 scenes / renderer fixture. Appendix only. **The DSH-E P3 13–15% HAR projection is INVALID_SUPERSEDED_MODEL and must never be reused to inflate this claim.**

---

## 2. Claims that must NOT be made

| # | Forbidden claim | Why |
|---|---|---|
| N-01 | "C0 V3 improves quality" / "no quality loss" | mean deltas are near-zero, single seed; permitted: "approximately neutral to marginally positive" |
| N-02 | "1.0685x is a 6.85% time reduction" | `speedup − 1` is `speedup_excess`, **permitted only under that name**; it is neither 6.20% nor 6.41% |
| N-03 | Any HAR / H8 percentage in the 13–53% range | DSH-E P3 model INVALID_SUPERSEDED_MODEL; H8 45–53% INVALID; 30.81% TIMING_SUBTRACTION_ARTIFACT |
| N-04 | "higs_proposed is our candidate" | SUPERSEDED_METHOD_GENERATION; its own confirmatory analysis found no quality-matched acceleration |
| N-05 | "C0 V3 is Nx faster than Faster-GS / FastGS / Speedy-Splat" | cross-cohort; NOT_DIRECTLY_COMPARABLE; no such run exists |
| N-06 | "C0 V3 is faster than clean gsplat / original 3DGS" under one protocol | no run exists; the available system cohort is the older method generation on a different device |
| N-07 | "multi-seed robustness confirmed" | FINAL-30K is seed 42 only |
| N-08 | "portable / cross-GPU" | zero non-A100 measurement |
| N-09 | "inference speedup" | the claim is training-only; forward is flat by design |
| N-10 | "first to <do X>" / prior-art novelty as fact | `artifacts/research-registry/prohibited_claims.json`; needs LITERATURE_VERIFICATION_REQUIRED |
| N-11 | Per-module causal share of the 1.0685x from the 13-scene table | A1/A2 do not exist at 30K |
| N-12 | censor_time presented as an achieved TTQ | right-censoring bound only |
| N-13 | A cross-arm VRAM table | no frozen VRAM aggregate exists |

---

## 3. Numerical-terminology audit (canonical rules)

Pinned in `artifacts/publication-audit/metric_definitions.json`.

### 3.1 The 1.0685x verification — three percentages, only one is reported

| Variant | Formula | Value | Status |
|---|---|---|---|
| geomean of per-scene reductions | `exp(mean(ln(1 − 1/speedup_i)))` | **6.1989%** | REPORTED_VALUE (matches the report's "6.20%") |
| derived from geomean speedup | `1 − 1/geomean_speedup` | 6.4117% | DERIVED_ALTERNATE — not equal to the reported 6.20% |
| arithmetic mean of per-scene reductions | `mean(1 − 1/speedup_i)` | 6.3955% | secondary |
| speedup excess | `(speedup − 1) × 100` | **6.8507%** | PERMITTED ONLY AS `speedup_excess` |

**Rule:** any percentage labelled "time reduction" must name which of the three reduction variants it is. `6.8507%` must never be called a time reduction.
**Audit result:** as of 2026-09-24 **no report in the repository commits this error**. The rule is recorded to prevent it entering the paper and its figures.

### 3.2 Other audited metric classes

| Metric | Canonical rule | Frozen value |
|---|---|---|
| speedup ratio | `T_baseline / T_candidate`; primary aggregate = geomean | 1.0685x |
| arithmetic vs geometric mean | arithmetic for secondary/quality; geometric for primary speedup, N_GS ratio, TTQ | n/a |
| median | renderer microbench cells only (C0-T2, P3-H, H8-T0); never for the 30K aggregate | n/a |
| percentage-point difference | absolute difference of two percentages; never call it a "percent speedup" | n/a |
| dPSNR / dSSIM | `candidate − baseline`; positive = better | +0.0696 dB / +0.000282 |
| dLPIPS | `candidate − baseline`; **NEGATIVE = better**; sign must not be flipped | −0.001668 |
| TTQ censoring | `ttq=null` + `TTQ_NOT_REACHED` + `censor_time` (right-censoring bound); censored scenes excluded from the geomean | 1.1374x over **11/13** pairs (censored: bicycle, stump) |
| N_GS ratio | `N_candidate / N_baseline` after 30000 steps; a ratio, **not** a saving percentage | 0.9813x (~1.9% fewer Gaussians) |
| VRAM | not publication-populated; no frozen cross-arm aggregate | NOT_PUBLICATION_POPULATED |
| phase timing | per-iteration mean ms; describes **where** time goes, **not** per-module causality | backward 21.596 -> 19.236 ms |

### 3.3 Prohibited number classes (enforced)

```text
DSH-E P3 temporal model (95.1 s / 14.9%, 174.1 s / 13.1%, 155.2 s / 13.7%)  = INVALID_SUPERSEDED_MODEL
old-C0 (7ca1c6bf, compute_abs=false) wall time / speedup / quality / TTQ / N_GS = DIAGNOSTIC_PROTOCOL_MISMATCH
H8 ~45-53% whole-backward gain                                                = INVALID
H8 30.81% room                                                                = TIMING_SUBTRACTION_ARTIFACT
E_compose = 1.0                                                               = INVALID (tautological)
contaminated / shared-GPU wall time in any publication table                  = FORBIDDEN
old-C0 forward > F+B (physically impossible)                                  = INVALID
zero-variance timing summaries                                                = INVALID
```

---

## 4. Publication-phase claim updates (2026-09-24)

The rules above remain in force. The publication-phase experiments (P1/P2/P4/P5/P6)
change claim labels ONLY where new frozen artifacts measure what was missing.
New claims inherit the same labeling discipline.

### 4.1 New claims

- **C-11 — accutile improves garden final PSNR over pristine gsplat, sign-consistent
  across 3 seeds. Label: SUPPORTED (3 seeds, COMPLETE).** Evidence: garden b1a-vs-b1
  pairs at seeds 42/43/44: +0.642 / +0.806 / +1.132 dB (b1a − b1; mean +0.860,
  within-arm seed spreads 0.78/0.68 do not explain the between-arm delta); wall edge
  +1.4–3.9% on garden across seeds, +2.2–4.6% on all gate scenes. Both axes
  sign-consistent at all three seeds (`garden_seed3_table.json`). Scope: garden only
  at this evidence level; do not generalize to all scenes.
  **13-scene expansion (COMPLETE):** at cohort scale accutile is faster on 12/13
  scenes (b1a/b1 wall geomean 0.9662, ~3.4%) and mean ΔPSNR (b1 − b1a) is −0.097 dB
  (garden-dominated; the other 12 scenes near zero; N ratio 0.9999) — the gate's
  wall edge is uniform, the quality effect is scene-concentrated
  (`aggregates/p1_b1_vs_b1a.json`).
- **C-12 — B0 (original Graphdeco) trains 2.42× slower than B1A under the matched
  protocol (13/13 scenes). Label: SUPPORTED (COMPLETE, 13 scenes).** Wall geomean
  B1A/B0 = 0.4142; the 3-scene gate's 2.20× understated the cohort gap. Quality:
  mean ΔPSNR (B0 − B1A) −0.899 dB with bidirectional per-scene failures of the
  original signed-grad statistic — under-densification on the unbounded outdoor
  scenes (stump −3.14, flowers −2.52, bicycle −2.19, treehill −2.10) and
  over-densification on drjohnson (+1.58 dB at N 1.54×, the same trajectory-sensitive
  scene as R-13). Travels as the disclosed protocol difference (densification
  statistic), not as a B0 "quality regression" claim; the absgrad-motivation
  reproduction (Yu et al.) is stated at cohort scale.
- **C-13 — the eps2d asymmetry bounds the headline: 1.057× (matched-0.1) to 1.0685×
  (frozen-0.3), quality −0.100 dB mean against C0. Label: SUPPORTED (P4, 9/9 clean).**
  Per the pre-registered rule this is MATERIAL_CONFOUND on quality → the disclosure is
  mandatory wherever C-01 is stated; the direction analysis (biases oppose) supports
  robustness of the conclusion, not a relabeling of C-01 as unconfounded.
- **C-14 — the cumulative ladder A0→A1→A2→A3=C0 at 30K measures the three frozen
  optimizations' total wall effect at ~0.45% (13 scenes, COMPLETE: geomean 1.0045,
  10/13 faster), quality-neutral (12/13 within ±0.15 dB, mean +0.022 without
  drjohnson). Label: SUPPORTED (Stage A 9/9 + Stage B 30/30, all clean).** Frames
  bit-identical, grad rel-err ≤5e-05. This MEASURES the branches' share of the A0→C0
  delta; it does not attribute the C0-vs-B1A headline (cross-codebase). Per-increment
  splits (8/4/4 of 13 faster) are disclosed — no single increment carries a uniform
  direction; the ladder's drjohnson cell (+0.770) is the same trajectory-sensitive
  scene as the headline's (+1.102), labeled per-scene.
- **C-15 — faster-gs / fastgs / speedy-splat collapse on garden under their own
  protocols (12.4–13.9 dB). Label: SUPPORTED_WITH_CAVEAT (SYSTEM_LEVEL, 13 scenes
  each).** Usable only in the SYSTEM_LEVEL table with the not-matched-protocol label;
  per N-05 this does NOT unlock any "C0 is Nx faster than external system" claim.
  CAVEAT (mandatory): the external cohort's garden uses manually-created COLMAP data
  and 0.5×-area-downsampled DSSIM (evidence-inventory §4) — the collapse is evidence
  about their pipeline on their garden-data variant, NOT a controlled contrast on the
  standard garden scene, and must not be framed as a numerics-discrimination result.

### 4.2 Caveat upgrades on existing claims

- **C-01 caveat 2 (eps2d not equalized):** still true, now QUANTIFIED — cite C-13's
  bound (1.057×–1.0685×) alongside the disclosure instead of the bare "disclosed".
- **C-03 caveat (single seed):** narrowed — the 13-scene aggregate remains single-seed
  (seed 42), but the subset is now multi-seed: garden pairs at 3 seeds (C-11) and P6
  4 scenes × 3 seeds (E22: all per-scene speedups > 1, per-seed geomeans
  1.0705/1.0735/1.0561, quality straddling zero on 3/4, drjohnson reproducible).
  Wording: "quality-neutral within the matched benchmark; seed-robust on the
  4-scene multi-seed subset".
- **C-05/C-06/C-07 caveats (no 30K module arms):** the CUMULATIVE ladder now exists at
  30K (C-14); isolated single-module arms remain absent by frozen design, so the
  "module-claim only" wording stands, now with ladder evidence attached.
- **N-06 ("faster than clean gsplat / original 3DGS under one protocol"):**
  **RESOLVED** — B1 and B0 both ran the full 13-scene matched cohort (gates +
  expansions complete). Matched-protocol speed claims are claimable with their
  disclosed confounds: C0-vs-B1 (renderer internals + eps2d asymmetry, C-13 bound),
  C0-vs-B0 (+ the densification-statistic protocol difference, C-12: 2.42×, 13/13
  scenes). Quality columns inherit C-11/C-12 per-scene caveats. The N-06 gap named in
  the audit is closed by E16/E17.
- **N-11 (per-module causal share of 1.0685×):** refined, not lifted — the ladder
  measures module shares of the A0→C0 delta (~0.4% total); the headline remains
  cross-codebase and its decomposition stops at the §21 frozen boundary
  (accutile +2.2–4.6% within gsplat; HIGS base carries the rest).

### 4.3 R-13 drjohnson outlier — convergence-curve review executed (was "NO")

The review the audit listed as missing now exists (`r13_drjohnson_review.json`):
identical start (−0.001 dB at step 500, same 263 eval cameras — NOT an eval artifact),
divergence to a **+3.37 dB peak at 15K**, then a **+4.74 dB b1a recovery jump**
(15K→20K) closing most of the gap, settling at +1.102. C0 retains 10% more Gaussians
(1.1018 — the only scene >1.05). Characterization: a genuine mid-training trajectory
difference where c0's path is better and b1a partially recovers late — NOT noise-scale
(peak far beyond the ±0.6 garden band), NOT an artifact.

**SEED VERDICT (P6 drjohnson quartet, COMPLETE): the effect is REPRODUCIBLE.**
ΔPSNR (C0 − B1A) = **+1.102 / +0.787 / +1.597 dB at seeds 42/43/44** (all > +0.5;
mean +1.16) and speedup 1.0482/1.0590/1.0467 (all > 1, tight band). Per the
pre-registered rule this upgrades drjohnson from "outlier" to **"reproducible
scene-level effect"** — reported per-scene, never generalized (§20): the 13-scene mean
ΔPSNR claim stays "quality-neutral within the matched benchmark", with drjohnson
disclosed as the reproducible per-scene positive. This strengthens reporting
integrity: a real effect is named as real, not laundered into noise.