# Paper Evidence Outline

**Report ID:** publication-paper-evidence-outline
**Date:** 2026-09-24
**Nature:** evidence outline, not prose. Each subsection lists required / available / missing evidence and the claim labels it may carry.
**Claim labels:** `SUPPORTED` / `SUPPORTED_WITH_CAVEAT` / `NOT_YET_SUPPORTED` / `DO_NOT_CLAIM` (see `claim-evidence-map.md`).

---

## 1. Introduction

| | |
|---|---|
| Required | a framing that does not rest on the 1.0685x scalar; a one-sentence statement of the exactness guarantee; the matched-renderer scope |
| Available | 13/13 scenes faster; backward 21.596 -> 19.236 ms; forward flat 4.507 -> 4.523 ms; exactness gates frozen |
| Missing | the written framing itself (R-01); the AccuTile-as-baseline justification (R-14) |
| Permitted claims | C-02 `SUPPORTED`; C-01 `SUPPORTED_WITH_CAVEAT` (with eps2d disclosure in the same sentence) |
| Forbidden | any primacy/novelty statement without `LITERATURE_VERIFICATION_REQUIRED`; "quality-improving"; the 6.85% "time reduction" wording |

## 2. Background / Bottleneck Analysis

| | |
|---|---|
| Required | the per-stage cost structure of the matched baseline; where the reclaimable work actually is |
| Available | `phase_timing.json` per-scene + aggregate (backward 21.596, loss 7.000, optimizer 5.014, forward 4.507, densify 1.105 mean ms); the negative-result map (P2-1C: entry-weighted 10.8/4.7/8.2% but 100% overlap with last-id pruning; frozen-128G skippable 0.40/2.00/1.08%) |
| Missing | nothing blocking; the analysis is complete on the 3-scene renderer fixture |
| Permitted claims | C-04 `SUPPORTED_WITH_CAVEAT`; C-09 `SUPPORTED_WITH_CAVEAT` (labelled diagnostic) |
| Forbidden | any per-module causal attribution from the 13-scene table |

## 3. Method

### 3.1 Exact projected-state work elimination (F9)
- Required: exactness proof + mechanism description + module-level timing.
- Available: `f9-0-gatherless-projected-producer.md`, `f9-1-trainable-gatherless.md`, `f9-1r-integration-closure.md`, `f9-final-5k.md`; patch `higs-f9-gatherless-final.patch` (`1faa05c1...`); forward +18.9/+14.8/+23.2%, F+B +7.63/+5.40/+10.56%.
- Missing: isolated 13-scene 30K F9-only arm (T3, non-blocking if no per-module causal share is claimed); standalone base is B2+SCALAR, not B1A.
- Permitted: C-05 `SUPPORTED` (module claim only).
- Forbidden: presenting standalone gains as composed gains; presenting F9 as a new algorithm rather than an exact elimination (R-04).

### 3.2 Scalar adjoint simplification
- Required: exactness closure + measured redundant-work removal.
- Available: `h2-bwd-2-scalar-adjoint-validation.md`, `h2-bwd-2r-exactness-closure.md`, `scalar-adjoint-freeze-smoke.md`; C0 nested contribution 1.2/0.1/0.3%.
- Missing: isolated Scalar-only 30K arm (T3).
- Permitted: C-06 `SUPPORTED_WITH_CAVEAT`.
- Forbidden: claiming this as the core contribution (R-05).

### 3.3 Moment-space geometry adjoint (H8-MR)
- Required: moment-adjoint gate + production description + direct backward timing; a *candidate* novelty statement.
- Available: `h8-0-moment-adjoint-gate.md`, `h8-0r-opacity-absorbed-moment.md`, `h8-mr-production.md`, `h8-t0-direct-timing-closure.md`; patch `higs-h8-mr.patch` (`8fb23acc...`); direct backward +4.31/+4.11/+3.54% (geomean 3.97%), F+B geomean 2.09%.
- Missing: literature verification (R-06); isolated 30K H8-only arm; causal attribution in the composed stack is not authorized.
- Permitted: C-07 `SUPPORTED` as a mechanism; `DO_NOT_CLAIM` for any causal share.
- Forbidden: H8 45–53% whole-backward gain (`INVALID`); H8 30.81% room (`TIMING_SUBTRACTION_ARTIFACT`).

## 4. Why obvious hierarchical alternatives fail

This is the paper's differentiator and is fully supported by internal negative results.

### 4.1 Negative-result consolidation

| Branch | Hypothesis | Oracle / gate | Measured result | Decision | Why it failed | Insight that survives | Paper placement |
|---|---|---|---|---|---|---|---|
| **H4** | Per-B16 prefiltering / family-level culling pays | H4-0R oracle repair, H4-P0 B16 prefilter | oracle not realizable at the required fidelity | **DROP_H4_FAMILY** | the oracle precondition does not hold on the trainable path | prefiltering cannot be assumed free | appendix |
| **H5** | Front-end load gating / batch-factored backward | H5-0R load gating, H5-1 batch-factored, last-id pruning | load gating WEAK; batch-factored DROP; last-id pruning LASTID_PRUNING_MARGINAL | WEAK / DROP | reclaimed work is largely already removed by ordinary last-id pruning | the "cheap" wins are already taken | appendix |
| **H6** | One-touch coverage of the backward | H6-0, H6-0R direct-bucket | one-touch WEAK; H6-0R FORWARD_REOPEN_STRONG | WEAK (backward) | the win reappears on the forward path, not the backward path | direction of the gain matters | appendix |
| **H7** | Macro-owner roofline shows headroom for hierarchy | H7-B0 roofline | H7B0_WEAK | DROP | no exploitable macro-owner headroom | roofline bounds the hierarchical idea | appendix |
| **R6-A** | CUDA prototype of the hierarchical reduction | R6-A minimal CUDA prototype | −55.9% to −91.5% backward **regression** | DROP | prototype overhead dominates the reclaimed work | implementation cost is real, not hypothetical | appendix |
| **P2-1A** | Native HiGS hierarchy transfers to exact trainable semantics | R3 semantic repair, gates A–E | A PASS, **B_support FAIL**, C PASS, **D_per_splat_weight FAIL**, **E_rgb_alpha FAIL**; `timing_authorized: false` | DROP native hierarchy | the native hierarchy's support/weight/colour semantics do not match exact trainable semantics | the transfer gap is semantic, not performance | **main paper (§4)** as the central negative |
| **P2-1C** | Hierarchical entry skipping is a large free win | preflight | entry-weighted 10.8/4.7/8.2% but **100% overlap** with last-id pruning; frozen-128G skippable 0.40/2.00/1.08% | DROP | the opportunity is already consumed | quantified *why* hierarchy does not pay | main paper (§4) as the quantitative core |
| **P3-HAR** | Hierarchical adjoint reduction reclaims ≥13–15% backward | DSH-H accumulation-ceiling diagnostic (V0–V4) | verdict **HAR_CEILING_WEAK**; ALL_ACCUM_CEILING ≤ **0.54%** backward (room −0.05, bicycle +0.54, garden −0.10) | DROP | global atomic accumulation is not a material cost on the final stack | an explicit ceiling closes the branch | appendix |
| **E2** | VJP->Adam fusion pays | E2-P0 decomposition | O_net 0.79/2.25/0.77% of a full iteration; perfect-fusion upper bound 1.32/3.31/1.22% | DROP | the upper bound is too small to matter | fusion headroom is bounded | appendix |

**Hard prohibition:** the DSH-E P3 temporal model (room 95.1 s / 14.9%, bicycle 174.1 s / 13.1%, garden 155.2 s / 13.7%) is `INVALID_SUPERSEDED_MODEL` and must **never** be reused — not in text, not in a figure, not as motivation.

| | |
|---|---|
| Required | the semantic-transfer failure (P2-1A) and the preflight quantification (P2-1C) |
| Available | complete for both, on 3-scene renderer fixtures |
| Missing | nothing blocking; scope must be stated as 3-scene renderer fixtures |
| Permitted | C-08 `SUPPORTED_WITH_CAVEAT`; C-09 `SUPPORTED_WITH_CAVEAT`; C-10 `SUPPORTED_WITH_CAVEAT` (appendix) |
| Forbidden | any HAR percentage in the 13–53% band; presenting any of these as a full-training measurement |

## 5. Experiments

### 5.1 Setup
- Required: protocol, cohort, hardware, clean-GPU gating, pinned binaries.
- Available: `artifacts/final-30k/protocol.json`, `README.md`, `binary_identity.json`, `provenance_audit.json` (PUBLICATION_PROVENANCE_PASS); 30000 iters, seed 42, max_side 1920, reference_v1, A100-PCIE-40GB, torch 2.9.1+cu128, `timing_grade=PUBLICATION`.
- Missing: per-run commit/config SHA (D3); per-run GPU snapshot for the candidate arm (D2). Documentation only.
- Must disclose: eps2d 0.1 vs 0.3; single seed; the run-phase-global `contaminated_attempts` list.

### 5.2 Matched renderer comparison (Table 1)
- Available: **POPULATED** — geomean 1.0685x, arithmetic 1.0687x, geomean reduction 6.1989%, mean dPSNR +0.0696 dB, dSSIM +0.000282, dLPIPS −0.001668, geomean final-N ratio 0.9813x, TTQ geomean 1.1374x over 11/13 pairs.
- Missing: eps2d equalization/sensitivity (R-08); multi-seed CIs (R-09).
- Permitted: C-01, C-02, C-03, C-04.
- Forbidden: "no quality loss"; any percentage called "time reduction" without naming the variant.

### 5.3 Ablation (Table 2 / Table 3 / Figure 3)
- Available: A0 and A3 at 30K (13 scenes); C0-T2 renderer replay V0/V1/V2/V3 on 3 scenes.
- Missing: **A1 and A2 at 30K — BLOCKING**; isolated F9-only / Scalar-only / H8-only 30K (non-blocking if no per-module share is claimed).
- Permitted today: A0 vs A3; the renderer-level replay as a *mechanism* figure; standalone module gates as exactness evidence.
- Forbidden: any per-module causal share of 1.0685x.

### 5.4 External system comparison (Table 5)
- Available: `artifacts/training-all` (210 jobs, 11 scenes x 3 seeds, older HiGS generation) — `gsplat vs higs_proposed 1.04x, mean dPSNR −0.380 dB`; `higs_full vs higs_proposed 1.02x`; `gsplat vs speedy_splat 0.41x`; `higs_full` vs gsplat 0.965x per its own confirmatory analysis.
- Missing: **no C0 V3 run against any external system under any protocol**; Faster-GS/FastGS are `NOT_DIRECTLY_COMPARABLE`; Turbo-GS `NOT_EXECUTED` (training code unreleased); flowers and treehill absent from the cohort.
- Permitted: system table populated only with an explicit "older method generation, different device/protocol" label — or omit.
- Forbidden: any cross-cohort C0-vs-external speedup; presenting higs_proposed as the candidate.

### 5.5 Sensitivity / robustness (Tables 4 and 6)
- Available: only the absgrad gradient-equivalence gate D1/D2; the older-protocol 3-seed quality cohort (context only).
- Missing: eps2d timing + quality sweep; FINAL-30K multi-seed CIs — both MISSING_HIGH.
- Permitted today: "approximately neutral under the matched benchmark, single seed".

## 6. Limitations

Required content (all backed by this audit): single seed; eps2d not equalized; A100-only; no external baseline under the matched protocol; no A1/A2 30K ablation; novelty subject to `LITERATURE_VERIFICATION_REQUIRED`; the earlier HiGS training-method negative result disclosed as a distinct superseded generation.

## 7. Conclusion

Permitted: the exactness-preserving renderer retrieval of the projected-state and adjoint work, measured as a uniform 13/13-scene direction with a flat forward and a backward-attributed decrease, plus the negative finding that hierarchical reuse does not pay on this stack.
Forbidden: any primacy claim; any statement stronger than "approximately neutral to marginally positive" quality.

---

## 8. Publication-phase outline update (2026-09-24)

The outline above is preserved. The publication-phase experiments (P1/P2/P4/P5/P6)
fill the §5 evidence slots as follows; §1–§4, §6–§7 framing rules are unchanged except
where noted. Claim/evidence IDs refer to `claim-evidence-map.md` §4 and
`evidence-inventory.md` §6.

### 8.1 §5.1 Setup — extended cohort
- The matched cohort now includes B0 (original Graphdeco @54c035f, original densification
  statistic disclosed) and B1 (pristine gsplat v1.5.3, `860936f2…`) arms under the same
  trainer/protocol, plus the env-switched a-chain on the frozen HIGS binary.
- Clean-GPU scheduling is now a documented file-driven queue with contamination
  forensics (`artifacts/publication/README.md`).
- Still must disclose: eps2d asymmetry (now with E19's bound), subset multi-seed scope
  (E21/E22), A100-only.

### 8.2 §5.2 Matched renderer comparison (Table 1)
- Table 1 now carries B0/B1 columns beside B1A/C0 (regenerated from `runs_master.json`).
- R-08 (eps2d) is CLOSED as a measurement question: E19 bounds the headline at
  1.057×–1.0685× with the quality bias −0.100 dB against C0; the same-sentence
  disclosure remains mandatory and now cites the bound.
- R-09 (multi-seed) CLOSED: garden seed pairs (E21) + P6 subset (E22: all 12
  per-scene speedups > 1, per-seed geomeans 1.0705/1.0735/1.0561, quality
  straddling zero on 3/4 scenes, drjohnson reproducible per-scene).
- Permitted/forbidden lists unchanged; C-01 caveat 2 is quantified rather than bare.

### 8.3 §5.3 Ablation — the BLOCKING gap is closed by E18
- A1 and A2 at 30K now exist (Stage A 9/9 clean; Stage B 13-scene completing), one
  frozen binary, env-switched, frames bit-identical, grad rel-err ≤ 5e-05.
- The ladder measures the three branches' total wall effect at ~0.4% (quality-neutral)
  — this is §5.3's honest populated content: the branches do NOT carry the headline;
  the cross-codebase decomposition (accutile +2.2–4.6% within gsplat; HIGS base the
  rest) is stated and stops at the §21 frozen boundary.
- "Forbidden: per-module causal share of 1.0685×" STANDS (N-11 refined, not lifted).

### 8.4 §5.4 External system comparison
- E20 supersedes the older-generation system table: faster-gs @3cb0b755 / fastgs
  @d4d33b6f / speedy-splat @b9dd42d, each 13 scenes × {native, c42}, 26/26 rc=0,
  commits + submodules pinned. Still SYSTEM_LEVEL / cross-cohort — the N-05
  prohibition on C0-vs-external speedups STANDS.
- The garden collapse carries the data-provenance caveat (their garden = manual COLMAP
  + 0.5× DSSIM; C-15 SUPPORTED_WITH_CAVEAT).

### 8.5 §5.5 Sensitivity / robustness
- eps2d timing + quality sweep: CLOSED by E19 (Table 4 populated, 9/9 clean cells).
- Multi-seed: CLOSED by E21 (garden, 3 seeds) + E22 (P6 4-scene subset, 3 seeds:
  all per-scene speedups > 1, per-seed geomeans 1.0705/1.0735/1.0561, quality
  straddling zero on 3/4, drjohnson reproducible per-scene). §5.5 wording:
  "quality-neutral within the matched benchmark; seed-robust on the multi-seed
  subset (garden pairs + 4-scene P6)".

### 8.6 §6 Limitations — updated lines
- "no A1/A2 30K ablation" — REMOVE (E18 closes it).
- "no external baseline under the matched protocol" — REPLACE with "B0/B1 matched
  baselines exist (gates + 13-scene expansions, E16/E17); external systems remain
  system-level only".
- "single seed" — NARROW to "single seed on the 13-scene aggregate; 3 seeds on the
  garden pairs and the P6 4-scene subset (E21/E22)".
- eps2d line — UPDATE to cite the E19 bound instead of "not equalized".
- Unchanged: A100-only; novelty LITERATURE_VERIFICATION_REQUIRED; the earlier HiGS
  training-method negative disclosed as a superseded generation.