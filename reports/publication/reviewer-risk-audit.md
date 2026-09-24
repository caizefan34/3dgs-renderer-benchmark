# Reviewer-Risk Audit

**Report ID:** publication-reviewer-risk-audit
**Date:** 2026-09-24
**Stance:** adversarial. No favorable answer is invented; absent evidence is reported as absent.
**Machine-readable source:** `artifacts/publication-audit/reviewer_risks.json`

Severity: `BLOCKING` (cannot submit as a positive-result paper until resolved) / `HIGH` (very likely raised; needs new evidence or a softened claim) / `MEDIUM` (answerable with existing evidence + scoping) / `LOW` (presentation).

---

## 1. Blocking

### R-02 — Novelty relative to Faster-GS / per-Gaussian backward (Speedy-Splat lineage)
- **Severity:** BLOCKING
- **Current evidence:** `artifacts/higs-p3-0/prior_art_overlap.json` classifies per-Gaussian backward as PARTIAL_OVERLAP — it changes *recurrence ownership*, whereas C0 V3 proves the pixel recurrence is untouched. F9 is a projected-state materialization elimination, not a gradient-topology change.
- **Missing evidence:** no literature-verified statement that no prior work performs exact projected-state elimination while preserving the recurrence. Faster-GS has **never** been executed against C0 V3.
- **Best mitigation:** mark every literature-dependent novelty statement `LITERATURE_VERIFICATION_REQUIRED`; scope the contribution to "exact work elimination under unchanged semantics" and lean on the *proved semantic invariance*, rather than claiming to be first.
- **Covered by active experiments:** NO (this is a literature-verification task, not an experiment).

### R-12 — Lack of a full ablation (A1/A2 at 30K)
- **Severity:** BLOCKING
- **Current evidence:** only A0 (baseline) and A3 (full C0 V3) exist at 30K, plus the 3-scene renderer replay C0-T2 V0/V1/V2/V3 (room −16.54%, bicycle −14.74%, garden −20.62%).
- **Missing evidence:** A1 (+F9) and A2 (+F9+SCALAR_ADJOINT) at 30K; isolated F9-only / Scalar-only / H8-only 30K arms.
- **Best mitigation:** run the cumulative ladder on a scene subset; **or** restructure the claims so that no per-module causal attribution is drawn from the 13-scene table (the renderer-level C0-T2 replay stays as the mechanism figure).
- **Covered by active experiments:** YES (A1/A2 are on the master plan) — ingest on completion.

### R-16 — Internal inconsistency with the repository's own earlier HiGS confirmatory study
- **Severity:** BLOCKING
- **Current evidence:** `paper/higs-claims.json`:
  - `topology-exploration-11s0` (supported): *"every HiGS candidate fails the quality non-inferiority margins… The observed 27k acceleration is therefore attributable to ordinary early-stop, not to the HiGS mechanisms."*
  - `confirmatory-ni-decisions` (supported): higs_full is **not** faster than the gsplat control (wall-time delta +19.9 s, speedup ratio **0.965**).
  - `official-training-baselines` and `cross-hardware-training`: status **blocked**.
- **Missing evidence:** an explicit reconciliation showing the C0 V3 renderer claim is a *different claim class* from the earlier training-method claim, plus evidence that the earlier negative finding does not transfer.
- **Best mitigation:** never reuse any earlier "HiGS accelerates training" claim. Frame C0 V3 strictly as a renderer-level exact-work-elimination result against a matched renderer baseline; disclose the earlier negative training-method finding as a distinct, superseded method generation.
- **Covered by active experiments:** NO — this is a framing/disclosure obligation owned by the paper text.

---

## 2. High

### R-01 — Only ~1.07x end-to-end speedup: is this paper-worthy?
- Current: geomean 1.0685x; 13/13 scenes faster; backward 21.596 -> 19.236 ms. Missing: a stated frame that the contribution is the **exactness-preserving mechanism + per-stage decomposition**, not the headline scalar; no memory/time tradeoff argument written yet.
- Mitigation: do **not** headline the scalar; headline the stage decomposition and 13/13 consistency, presenting 1.0685x as the consequent end-to-end effect. Covered by active experiments: PARTIAL (they strengthen the decomposition, not the scalar).

### R-03 — Novelty relative to gsplat: are these just kernel engineering upstream could absorb?
- Current: exactness/oracle gates exist for F9, Scalar, H8; `prohibited_claims.json` forbids prior-art novelty as fact. Missing: no statement that these transforms are absent from gsplat main.
- Mitigation: present as engineering contributions with exactness proofs, explicitly separated from the scientific insight; concede that individual transforms are algebraic; claim the composition and its exactness, not primacy. Covered: NO.

### R-06 — Is H8-MR genuinely new or a re-parameterization of an existing moment accumulation?
- Current: `h8-mr-production.md`, `h8-0-moment-adjoint-gate.md`, `h8-0r-opacity-absorbed-moment.md`, `h8-t0-direct-timing-closure.md`; `prior_art_overlap.json` notes HAR's P_GEOM packet **is** the H8 moment set. Missing: literature verification that sufficient-statistic moment-space geometry adjoints are not previously described.
- Mitigation: claim it as a candidate scientific distinction and mark `LITERATURE_VERIFICATION_REQUIRED`; do not assert novelty as fact. Covered: NO.

### R-08 — eps2d 0.1 (B1A) vs 0.3 (C0) confound
- Current: `B1A_ACCUTILE/truck/provenance.json` eps2d 0.1 vs `C0_V3_FINAL30K/room/provenance.json` eps2d 0.3; `final30k-absgrad-compatibility.md` quantifies only the absgrad gradient-equivalence gate (D1/D2). Missing: no eps2d sensitivity sweep on quality **or** timing at any scene count.
- Mitigation: run an eps2d arm (Table 4) — demonstrate timing-insensitivity or match eps2d; if timing is eps2d-insensitive this collapses to LOW. Until then, disclose the confound in the same sentence as the 1.0685x claim. Covered: YES (permitted on the master plan).

### R-09 — Single-seed quality claims
- Current: all FINAL-30K runs are seed 42; deltas tiny (dPSNR +0.070 dB, dSSIM +0.0003, dLPIPS −0.0017). Missing: 3-seed CIs under the FINAL-30K protocol (the existing 3-seed cohort is the older protocol, 11 scenes).
- Mitigation: add a 3-seed subset arm; meanwhile scope the wording to "approximately neutral under the matched benchmark, single seed" and never "no quality loss". Covered: YES (master plan).

### R-10 — A100-only main evidence
- Current: all FINAL-30K evidence is A100-PCIE-40GB; consumer/second-datacenter runners `blocked`; gencode pinned sm_80. Missing: any non-A100 measurement. Note the older 3-seed cohort ran on A100-SXM4-80GB — a different A100 class, which is **not** portability evidence either.
- Mitigation: execute the minimal portability matrix (readiness report §6) on room/bicycle/garden for forward/backward/F+B. Until then scope all claims to A100. Covered: PLANNED_ONLY.

### R-11 — Lack of external baselines in the renderer comparison
- Current: the renderer table holds only C0 V3 and B1A AccuTile; system-level evidence exists separately for the **older** HiGS generation. Missing: C0 V3 vs original_3dgs / clean gsplat / Faster-GS / FastGS under one protocol.
- Mitigation: split the tables — renderer table = C0 V3 vs B1A (matched); system table = existing cohort clearly labelled, or omit. Never cross-merge cohorts. Covered: PARTIAL.

---

## 3. Medium

| ID | Objection | Current evidence | Missing | Mitigation | Covered |
|---|---|---|---|---|---|
| R-04 | F9 is merely kernel fusion / materialization avoidance | F9 production + integration-closure + 5K reports; patch exists | no standalone 30K F9-only 13-scene arm | state plainly that F9 is an exact work-elimination transform with proof; rest the scientific framing on H8-MR | UNKNOWN |
| R-05 | Scalar Adjoint is merely algebraic VJP simplification | validation + exactness closure + freeze smoke | no isolated Scalar-only 30K arm | concede algebraic nature; supply the measured removal as supporting engineering | PARTIAL |
| R-07 | The three transforms do not obviously belong to one theory | the three were frozen independently (registry C0_V3 / F9 / SCALAR_ADJOINT / H8) | a written unifying argument with per-sub-claim evidence | use the proposed framing but label it a framing **hypothesis**; the cumulative ablation is what would substantiate it — currently missing at 30K | UNKNOWN |
| R-13 | drjohnson +1.1 dB outlier looks like an eval/convergence artifact | dPSNR +1.1023, dSSIM +0.0114, dLPIPS −0.0273, final-N ratio 1.1018 (only >1.05) | no per-scene convergence-curve review of drjohnson | report the outlier explicitly and show the aggregate is robust with/without it; do not suppress | NO |
| R-14 | 13-scene cohort fairness / is AccuTile a fair strong baseline | 13 scenes across Mip-NeRF 360 / T&Ts / Deep Blending; AccuTile is the fastest clean gsplat variant | no justification doc for choosing AccuTile over clean gsplat; no accutile=false variant comparison | justify AccuTile as the strongest gsplat-family renderer and note a weaker baseline would inflate the speedup | UNKNOWN |

---

## 4. Low

| ID | Objection | Evidence | Mitigation |
|---|---|---|---|
| R-15 | Training-only relevance; inference unaddressed | claim is a training-time claim; forward path effectively bit-identical | scope to training; state inference is unchanged and out of scope; note forward 4.507 -> 4.523 ms is noise, not a regression |
| R-17 | Provenance asymmetry: candidate arm lacks per-run config/GPU/checkpoint files | `C0_V3_FINAL30K/<scene>/` has 7 files vs B1A's 10 (see `provenance_audit.json` D2) | document that the recipe is pinned at cohort level and the asymmetry is a harness-writing difference, not a measurement difference; no aggregate defect found |
| R-18 | Legacy trees (`P2_FINAL_V2/`, `C0_V3/`) coexist with the frozen tree | `provenance_audit.json` D1 | cite only `C0_V3_FINAL30K/` <-> `B1A_ACCUTILE/` plus top-level frozen aggregates; add a one-line note to the artifact-availability statement |

---

## 5. Top five, ranked

1. **R-02** novelty vs Faster-GS / per-Gaussian backward — BLOCKING, literature-dependent.
2. **R-12** no A1/A2 30K ablation — BLOCKING, experiment-dependent.
3. **R-16** internal precedent that contradicts an acceleration story — BLOCKING, framing-dependent.
4. **R-08** eps2d 0.1 vs 0.3 confound — HIGH, experiment-dependent (already permitted on the master plan).
5. **R-01** "1.07x is not enough" — HIGH, framing-dependent; the per-stage decomposition is the only credible rebuttal.

R-12 and R-16 are the two that most directly determine whether a positive-result paper is submittable.
R-02 is a literature-verification task rather than an experiment.

---

## 6. Publication-phase evidence ingestion (2026-09-24)

The audit above is preserved as a point-in-time stance. This section maps the
publication-phase experiments (P1/P2/P4/P5/P6, goal rounds 2–6) onto each risk.
Evidence pointers are frozen artifacts; nothing here softens a claim without new data.

| ID | Audit status | Ingested status | Evidence |
|---|---|---|---|
| R-12 | BLOCKING | **RESOLVED (Stage B COMPLETE)** — the cumulative ladder A0→A1→A2→A3 exists at 30K on all 13 scenes: Stage A 9/9 clean; Stage B 30/30 clean (geomean A0→C0 1.0045, 10/13 faster, mean ΔPSNR +0.079). Frames bit-identical across the chain, grad rel-err ≤ 5e-05 — stronger than the audit's ask. NOTE: the ladder's result (~0.4% total wall) *reinforces* the alternative mitigation: per-module causal attribution of the headline is now measured to be impossible; the decomposition in `cumulative-ablation.md` is the claim structure. | `ablations/`, `a_arm_functional_gate.json` |
| R-02 | BLOCKING | UNCHANGED (literature task). Partial evidence gain: faster-gs/fastgs/speedy-splat now executed on all 13 scenes (system level, not head-to-head matched); their shared garden collapse (12.4–13.9 dB) is context, not a novelty proof. | `aggregates/p5_external.json` |
| R-16 | BLOCKING | UNCHANGED (framing obligation). `publication-summary.md` follows the prescribed framing: renderer-level exact-work-elimination claim; the earlier training-method generation disclosed as superseded. | `publication-summary.md` |
| R-08 | HIGH | **RESOLVED-WITH-DISCLOSURE** — P4 complete (9/9 clean cells): MATERIAL_CONFOUND on the quality axis (C0@0.3 −0.100 dB vs @0.1, exceeding the FINAL-30K mean 0.070); wall NOT material (1.07% < half the speedup). Direction analysis: biases oppose (speed inflated ~1.1% → matched ≈ 1.057×; quality deflated ~0.10 dB → matched ≈ +0.170). The audit's "collapses to LOW" did not occur — it remains a disclosed, robustness-bounded confound in every C0/B1A table. | `eps2d/p4_eps2d_verdict.json`, `eps2d-sensitivity.md` |
| R-09 | HIGH | **RESOLVED (P6 + garden complete)** — 3-seed garden pairs: accutile effect sign-consistent on quality AND wall (+0.642/+0.806/+1.132 dB; +1.4–3.9%). P6 4-scene subset: all 12 per-scene speedups > 1; per-seed geomeans 1.0705/1.0735/1.0561 (headline 1.0685 straddled); quality straddles zero on 3/4 scenes; drjohnson reproducible per-scene (+1.16 mean, labeled). Without-drjohnson geomeans 1.059–1.078. | `multiseed/`, `p6_results.json`, `garden_seed3_table.json` |
| R-01 | HIGH | SHARPENED — the decomposition is now measured, not hypothetical: three branches ~0.4%, accutile +2.2–4.6% (bounded by the B1 gate), infrastructure the rest. The paper cannot and does not attribute the headline to the branches. | `cumulative-ablation.md` |
| R-10 | HIGH | UNCHANGED — A100-only; portability matrix out of publication-phase scope; claims scoped to A100. | — |
| R-11 | HIGH | **RESOLVED** — B0 (original Graphdeco @54c035f) and B1 (pristine v1.5.3) ran the FULL matched 13-scene cohort (gates + expansions; E16/E17). External systems on all 13 scenes at SYSTEM_LEVEL, never cross-merged with the matched cohort. | `matched-baselines.md`, `external-systems.md` |
| R-04/05/07 | MEDIUM | R-07's "cumulative ablation missing at 30K" RESOLVED (see R-12). Isolated single-module arms remain absent by frozen design (cumulative ladder only). | `tables/table2..3` |
| R-13 | MEDIUM | **RESOLVED — REPRODUCIBLE.** P6 measured drjohnson × seeds 43/44 directly: +1.102/+0.787/+1.597 dB (all > +0.5), speedup 1.047–1.059. The outlier is a reproducible scene-level effect, reported per-scene, never generalized. | `p6_results.json`, claim map §4.3 |
| R-14 | MEDIUM | **RESOLVED** — the audit's missing "accutile=false variant comparison" now exists: B1 vs B1A on the full 13-scene cohort + garden × 3 seeds. Accutile adds ~3.4% wall (12/13) and the garden quality effect is sign-consistent (+0.642/+0.806/+1.132) — so B1A is the STRONGER baseline and the headline speedup is conservative against it. | `aggregates/p1_b1_vs_b1a.json`, `garden_seed3_table.json` |

New risks surfaced by the publication phase (not in the original audit):

- **R-19 (new, HIGH): accutile garden quality effect.** B1A embeds a scene-dependent
  quality advantage over pristine gsplat — now measured sign-consistent at 3 seeds on
  garden (+0.642/+0.806/+1.132 dB). Controlled in the headline (both arms
  exact-accumulate) but must be disclosed wherever B1A is described as
  "gsplat + speed". Evidence: `garden_seed3_table.json` (3 seeds).
- **R-20 (new, MEDIUM): B0 bicycle −2.19 dB.** The original densification statistic
  under-densifies texture-heavy scenes (N 77% of B1A). Reported as a finding with the
  single-disclosed-protocol-difference caveat; not smoothed. Evidence:
  `b0_gate_verdict.json`.