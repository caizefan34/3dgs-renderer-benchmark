# FRONTIER-REASSESS0-MX — Remaining Backward Frontier Reassessment

Lane: REASSESSMENT / ORACLE / FEASIBILITY. Decision experiment only — no production optimization implemented unless promoted.
Date: 2026-10-07. Hardware: MX A100-PCIE-40GB. Baseline: frozen H-FasterGS canonical (RASTER-REBASE0-MX).

---

## 1. Executive Summary

Question: after BWD-ARITH0 (arithmetic frontier), BWD-ARITH1 (exact G-cache, DROP_REUSE_TRAFFIC_OVERHEAD), and BWD-DEPREG0 (NO_ACTIONABLE_DEPREG_FRONTIER), is there any remaining high-value EXACT backward optimization that removes expf/SFU or interaction cost without full-interaction HBM traffic — or is the exact frontier too small, favoring sparse/approximate methods?

Answer: **NO_REMAINING_HIGH_VALUE_BWD_FRONTIER.** All three exact lanes fail the promotion gate:
- **F1 (existing-state reconstruction)**: G is not algebraically recoverable from stored state without a new per-interaction cache (= ARITH1, traffic-dominated, S_backward 0.791). → EXACT_RECONSTRUCTION_TOO_EXPENSIVE.
- **F2 (exact SFU reformulation)**: expf already compiles to MUFU.EX2 + minimal correction polynomial; the only faster forms (__expf, exp2f) are APPROXIMATE (same SFU count, drop the polynomial). No exact faster variant exists. → EXACT_SFU_ALREADY_OPTIMAL.
- **F3 (exact interaction elimination)**: production already applies every provable exact skip (validity, boundary, last_contributor early-termination, alpha-threshold). No additional structurally-zero interaction exists. → NO_EXACT_SKIP.
- **F4 (approximate sparse reference, K50-B1)**: K50_MX_FAIL_QUALITY, geomean speedup 0.7945 (SLOWER) + severe quality regression (room dPSNR -15.3@30k). → APPROX_FRONTIER_NOT_DECISIVE.

Exact renderer ceiling: <1.05× (no exact mechanism reaches 5%). The only >5% ceiling (O20 expf-free 1.1116× renderer) is NOT exactly realizable. A negative exact result is a successful outcome (spec §27).

---

## 2. Prior Evidence

Read-only ingestion (FRONTIER_REASSESS0_PRIOR_EVIDENCE.json), not reinterpreted:
- **BWD_ARITH0**: BWD_ARITHMETIC_FRONTIER_FOUND; primary COMPUTE_COUNT_BOUND, secondary TRANSCENDENTAL_SFU. O20 (expf→linear) = −15.65% bwd geomean, S_renderer geomean 1.1116. blend SASS: 712 inst, 4 MUFU.EX2, 67 regs, 0 spill.
- **BWD-ARITH1-RECOMPUTE-GAUSSIAN**: DROP_REUSE_TRAFFIC_OVERHEAD. Exact FP32 G-cache correct but slower: S_backward 0.791, S_renderer 0.832. Cache LOAD alone at peak HBM = 1.45–1.59× the O20 expf savings.
- **BWD_DEPREG0_MX**: NO_ACTIONABLE_DEPREG_FRONTIER. 67 regs/0 spill; dependency oracle 0.9978; occupancy/gradient-family flat.
- **K50-B1** (sparse reference): K50_MX_FAIL_QUALITY.

---

## 3. Closed Frontiers

Per spec §0/§21, these are CLOSED and NOT reopened: full-interaction FP32 Gaussian cache, register-lifetime tuning, occupancy tuning, spill reduction, generic recurrence/dependency refactor, blind kernel splitting.

---

## 4. Remaining Hypotheses

Four frontier classes evaluated (spec §3): F1 EXISTING_STATE_RECONSTRUCTION, F2 EXACT_SFU_REFORMULATION, F3 EXACT_INTERACTION_ELIMINATION, F4 APPROX_SPARSE_FRONTIER.

---

## 5. Existing-State Dataflow Audit

blend_backward_cu (kernels_backward.cuh:253-482) + blend_cu (kernels_forward.cuh:395-460). Per-interaction quantities backward loads/computes: mean2d, conic, opacity, color, bucket_color_transmittance, tile_final_transmittances, gaussian G (computed), alpha (computed), transmittance (computed). Full map: FRONTIER_REASSESS0_EXISTING_STATE_MAP.json.

Critical finding: the forward stores bucket_color_transmittance = float4(color_pixel, transmittance) **only when j%32==0** (32-Gaussian boundaries), plus tile_final_transmittances (final only). **Per-interaction alpha/G is never stored.**

---

## 6. Reconstruction Exactness

Candidate identities audited (FRONTIER_REASSESS0_RECONSTRUCTION_EXACTNESS.json):
- G = alpha/opacity: alpha not stored per-interaction → requires new cache (=ARITH1). INVALID.
- G from transmittance ratio (alpha=1−T_after/T_before): requires per-interaction T, not stored. INVALID.
- G from bucket_color_transmittance snapshot: snapshot is at 32-boundaries of running T/color, cannot isolate a single interaction's G. INVALID.
- Production expf recompute: EXACT_IN_REAL_ARITHMETIC, but it is the cost under investigation.

Pathological cases (opacity==0, small opacity, saturation, thresholding, early-termination, underflow) all collapse to the alpha-threshold skip already in production. Surviving exact candidates: none. Verdict: NO_EXACT_RECONSTRUCTION.

---

## 7. Reconstruction Oracle

No viable exact zero-extra-traffic candidate exists. The only exact way to avoid backward expf by reusing a stored value is the ARITH1 FP32 G-cache, already measured S_backward 0.791 (regression). Re-running would only re-confirm ARITH1. → EXACT_RECONSTRUCTION_TOO_EXPENSIVE. (FRONTIER_REASSESS0_EXISTING_STATE_ORACLE.json)

---

## 8. SFU Compiler/SASS Audit

Method: cuobjdump --dump-ptx + --dump-sass (sm_80, CUDA 12.8). PTX shows ex2.approx.ftz.f32 for all three expf formulations. Full blend_backward_cu SASS opcode counts (FRONTIER_REASSESS0_SFU_AUDIT.json, raw/f2_sass_*.txt):

| variant | FADD | FFMA | FMUL | MUFU.EX2 | MUFU.RCP |
|---|---|---|---|---|---|
| expf (r0) | 38 | 121 | 87 | 9 | 9 |
| __expf | 34 | 105 | 95 | 9 | 9 |
| exp2f | 34 | 105 | 95 | 9 | 9 |

All three: 67 registers, 0 spill. Key finding: **MUFU.EX2 count is invariant (9)** across expf/__expf/exp2f — the expf→SFU mapping is fixed. The only difference is ~12 fewer float ops (the correction polynomial) in the fast variants, which makes them approximate. exp2f(x·log2e) produces the same bare-EX2 SASS as __expf → NOT bit-exact to expf.

---

## 9. SFU Variant Results

- expf (production): EXACT_PATH, 0 ULP, MUFU.EX2 9, 246 float ops.
- __expf: APPROX_PATH (~2 ULP fast intrinsic), MUFU.EX2 9, 234 float ops.
- exp2f(x·log2e): APPROX_PATH (same bare-EX2 SASS as __expf, not bit-exact to expf), MUFU.EX2 9, 234 float ops.

F2_exact = NO_GO: no exact faster SFU variant exists; expf already optimal on SM80. Approximate potential (~12 float ops saved, same SFU count) recorded as F4 context only. (FRONTIER_REASSESS0_SFU_VARIANTS.json)

Timing note: paired timing under heavy MX contention (all 8 GPUs foreign jobs, multiple at 97–99% util) was not completable within caps; per spec §10 timing is required only if an exact admissible alternative exists, and none does. Static SASS/PTX is the decisive F2 audit and is contention-independent.

---

## 10. Exact Interaction-Elimination Audit

Existing exact skips already in production (FRONTIER_REASSESS0_EXACT_SKIP_MAP.json):
- !valid_primitive (line 291), !valid_pixel boundary (398), idx out-of-range (412), tile_primitive_idx >= last_contributor early-termination (412), alpha < min_alpha_threshold (427).

Candidate additional skips audited: opacity==0 (subsumed by alpha-threshold, since G>0), zero incoming adjoint (data-dependent, not structural — would change semantics), conic==0 (G=exp(0)=1, NOT zero contribution — cannot skip), gaussian underflow (subsumed by alpha-threshold). exact_skip_fraction_new = 0.

---

## 11. Exact Skip Oracle

No new exact skip condition exists; production already applies every provable exact skip. An oracle removing "additional" exact skips would remove nothing. The only way to skip MORE interactions is an approximate importance threshold (forbidden in F3). BWD-DEPREG0's recurrence oracle (0.9978) confirms processed interactions are not further exactly-eliminable. → NO_EXACT_SKIP. (FRONTIER_REASSESS0_EXACT_SKIP_ORACLE.json)

---

## 12. Approximate Sparse Reference

K50-B1 (mx_overnight_chain/_k50_mx_revalidation), not modified (FRONTIER_REASSESS0_APPROX_REFERENCE.json):
- Verdict: K50_MX_FAIL_QUALITY, gate_pass false.
- geomean speedup 30K: 0.7945 (SLOWER than baseline). Per-scene: room 0.566, garden 0.959, bicycle 0.925.
- Quality: room dPSNR −6.6@1k to −15.3@30k; bicycle −2.5 to −9.4; garden mixed (+1.6@30k but late). Severe regression on 2/3 scenes.
- Conclusion: K50-B1 is NOT decisive — slower AND fails quality. Timing contended (non-exclusive); quality gates contention-independent (authoritative).

---

## 13. Amdahl Ceiling

If expf were completely free (FRONTIER_REASSESS0_AMDAHL.json):
- O20 expf-free ceiling: S_backward geomean 1.1856, S_renderer geomean 1.1116 (room 1.1153, bicycle 1.1462, garden 1.0744).
- BUT this is an APPROXIMATE oracle (expf→linear), NOT exactly realizable. The only exact way to reuse a stored G is the ARITH1 cache (traffic-dominated, S_backward 0.791). F2 exact SFU is already optimal. F3 exact skip = none.
- Exactly-realizable exact renderer ceiling: <1.05 (no exact mechanism reaches 5%).
- Verdict: the O20 1.1116× ceiling is unrealizable exactly; terminate exact micro-optimization.

---

## 14. Exact-vs-Approximate Comparison

Decision matrix (FRONTIER_REASSESS0_DECISION_MATRIX.json):
- F1 EXACT: ARITH1 cache regression (S_backward 0.791). EXACT_RECONSTRUCTION_TOO_EXPENSIVE.
- F2 EXACT: no exact faster variant. EXACT_SFU_ALREADY_OPTIMAL.
- F3 EXACT: no exact skip. NO_EXACT_SKIP.
- F4 APPROX (K50-B1): slower + quality fail. APPROX_FRONTIER_NOT_DECISIVE.

All three exact lanes fail; the approximate sparse lane is not decisive. No exact nor approximate lane offers validated upside.

---

## 15. Cross-Scene Consistency

F1/F2/F3 are scene-independent (source/SASS static analysis; exactness is algebraic). F4 K50-B1 is inconsistent (garden slightly positive at long horizons, room/bicycle strongly negative). The exact negative result is consistent across the analysis.

---

## 16. Limitations

- MX host heavily contended (all 8 GPUs foreign jobs ~22–28GB, multiple 97–99% util). F2/F3 static evidence (SASS/source) is contention-independent and decisive. F4 K50 quality gates are contention-independent; its timing is contended but the quality FAIL is authoritative. No authoritative timing was produced this lane due to contention.
- ncu 2021.3.1 incompatible (error 6); nvprof 11.5 too old. Static SASS/PTX used for F2 (decisive for instruction-mix).
- F1 oracle references ARITH1 rather than re-running (no viable exact candidate to measure).

---

## 17. Terminal Verdict

**NO_REMAINING_HIGH_VALUE_BWD_FRONTIER** (spec §19/§20/§22/§27).

Per-frontier: F1 EXACT_RECONSTRUCTION_TOO_EXPENSIVE; F2 EXACT_SFU_ALREADY_OPTIMAL; F3 NO_EXACT_SKIP; F4 APPROX_FRONTIER_NOT_DECISIVE. Exact renderer ceiling <1.05×; the O20 1.1116× expf-free ceiling is not exactly realizable. A negative exact result is successful (§27). Do NOT implement the next lane automatically.

---

## 18. Next Lane

NONE for exact backward micro-optimization. The exact backward frontier is exhausted. If research continues, the productive axis is approximate/sparse interaction reduction — but the only prior sparse evidence (K50-B1) is itself a FAIL on this cohort, so a new sparse method would need a fundamentally different mechanism, not a retry of K50.

---

## 19. Evidence Classification

OBSERVED:
- Forward stores bucket_color_transmittance only at j%32==0 + tile_final_transmittances; per-interaction alpha/G never stored (source).
- SASS: MUFU.EX2 invariant (9) across expf/__expf/exp2f; expf 38 FADD/121 FFMA/87 FMUL, fast variants 34/105/95; all 67 regs/0 spill (cuobjdump).
- PTX: ex2.approx.ftz.f32 for all three formulations.
- K50-B1: K50_MX_FAIL_QUALITY, geomean speedup 0.7945, room dPSNR −15.3@30k.
- O20: S_renderer geomean 1.1116 (room 1.1153/bicycle 1.1462/garden 1.0744).
- ARITH1: S_backward 0.791, S_renderer 0.832.

DERIVED:
- expf→MUFU.EX2 mapping is fixed; faster variants drop the correction polynomial (approximate).
- No zero-extra-traffic exact G-reconstruction exists.
- Exactly-realizable exact renderer ceiling <1.05.

INFERRED:
- The O20 1.1116× expf-free ceiling is not exactly realizable (caching is traffic-dominated; exact SFU optimal; no exact skip).

HYPOTHESESed:
- (none material; the negative result is directly observed.)

ORACLE_ONLY:
- O20 (expf→linear) is an approximate oracle bounding expf-removal upside, not an exactly-realizable target.
- K50-B1 is approximate/sparse reference only.
