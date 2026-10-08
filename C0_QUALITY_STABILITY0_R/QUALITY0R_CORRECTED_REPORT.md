# C0-QUALITY-STABILITY0-R — Corrected Publication Audit

**Type:** non-destructive report correction of `C0_QUALITY_STABILITY0_MX/`.
**Training rerun:** none. **Raw results modified:** none. **Protocol altered:** none.
Every corrected number traces to an authoritative source JSON under `/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX`.

## 1. Verdict (frozen, preserved)

- **OFFICIAL_CAMPAIGN_VERDICT = `QUALITY_SEED_SENSITIVE`** (frozen under protocol; preserved, not replaced).
- **Corrected publication characterization = `QUALITY_SEED_SENSITIVE_WITH_NEGATIVE_DIRECTIONAL_BIAS`** (added, not a replacement).

## 2. Frozen quality-gate recount (authoritative)

Thresholds: dPSNR_floor = -0.05 dB, dSSIM_floor = -0.0005, dLPIPS_ceil = +0.001, catastrophic = -0.5 dB.
Direction convention: delta = C0 - A0 (positive dPSNR/dSSIM and negative dLPIPS = C0 better).

| scene | seed | dPSNR | dSSIM | dLPIPS | psnr | ssim | lpips | all-gates | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | 42 | -0.0854 | -0.00180 | 0.00215 | F | F | F | fail | no |
| bicycle | 43 | -0.0352 | -0.00079 | 0.00065 | P | F | P | fail | no |
| bicycle | 44 | -0.0367 | -0.00079 | 0.00064 | P | F | P | fail | no |
| garden | 42 | -0.4963 | -0.02866 | 0.03155 | F | F | F | fail | no |
| garden | 43 | -0.5843 | -0.02163 | 0.01832 | F | F | F | fail | YES |
| garden | 44 | 0.4686 | 0.01198 | -0.00580 | P | P | P | PASS | no |
| room | 42 | -0.0350 | -0.00058 | 0.00048 | P | F | P | fail | no |
| room | 43 | -0.0241 | -0.00021 | 0.00091 | P | P | P | PASS | no |
| room | 44 | -0.0097 | -0.00039 | 0.00015 | P | P | P | PASS | no |

Aggregates: **pass-all = 3/9 · fail-any = 6/9 · fail-all-three = 3/9 · catastrophic = 1/9.**

> **REPORT_ACCOUNTING_ERROR:** the original `QUALITY_REPORT.md` claimed 6/9 pass, 3/9 fail. The authoritative recount is **3/9 pass-all, 6/9 fail-any**. Corrected in `QUALITY0R_GATE_RECOUNT.json`.

## 3. Directionality (DIRECTIONAL_DIAGNOSTIC_ONLY)

- negative = 8, positive = 1, zero = 0
- median = -0.0352 dB, mean = -0.0931 dB, min = -0.5843, max = 0.4686
- exact two-sided sign test (n=9, k=1): **p = 0.039062**
- Label: **DIRECTIONAL_DIAGNOSTIC_ONLY**. The 9 observations come from 3 scenes x 3 seeds and are not fully independent; this is NOT confirmatory significance.

## 4. Statistical interpretation

- mean dPSNR = -0.0931 dB, bootstrap 95% CI = [-0.2330, 0.0199] (crosses zero).
- **Allowed:** "The aggregate mean degradation is statistically inconclusive because the bootstrap CI crosses zero."
- **Forbidden:** "There is no systematic negative effect." — failure to establish a mean degradation is NOT evidence of quality equivalence / non-inferiority.

## 5. Sensitivity

- mean all 9 = -0.0931 dB
- mean without garden seed44 (largest positive response) = -0.1633 dB
- leave-one-scene-out means: room -0.0229, bicycle -0.0524, garden -0.2040
- leave-one-pair-out mean range = [-0.1633, -0.0317]
- Garden seed44 is the **largest positive response** (not classified as an outlier; no pre-defined outlier rule exists). The aggregate mean is sensitive to scene/pair inclusion.

## 6. Garden interpretation (corrected)

- garden dPSNR range = [-0.5843, 0.4686]; non-garden range = [-0.0854, -0.0097].
- garden spread = 1.053 dB vs non-garden 0.076 dB.
- **Corrected:** garden dominates the variance and **both tails** of the quality response (largest negative garden s42 -0.4963 AND largest positive garden s44 +0.4686). It must NOT be reduced to a purely negative scene.

## 7. Fixed-state vs trajectory (mandatory audit)

- **SAME_STATE:** iter1 loss reldiff = 0.0 (shared initialization).
- **TRAJECTORY_STATE (removed from fixed-state block):** iter15000 = 0.08591253104142235, iter30000 = 0.040077519324971944 — these come from independently trained A0/C0 states and are **trajectory divergence**, not fixed-state equivalence.
- **Fixed-state equivalence = SUPPORTED** on SAME-STATE forward/gradient tests within tolerance only.

## 8. Exactness terminology (corrected)

- ALGEBRAICALLY_EXACT = **SUPPORTED** (by derivation of F9/Scalar/H8; binary hash 9baf8655 only establishes same substrate, not algebraic exactness).
- BIT_EXACT = **NOT_SUPPORTED** (FP reassociation changes rounding).
- FLOATING_POINT_EQUIVALENT_WITH_TOLERANCE = **SUPPORTED** (SAME-STATE forward/gradient within tolerance only).
- FULL_TRAINING_TRAJECTORY_EQUIVALENT = **NOT_SUPPORTED** (n_gaussians/loss diverge at iter 600-700).
- FULL_TRAINING_QUALITY_EQUIVALENT = **NOT_ESTABLISHED** (frozen non-inferiority gates not consistently passed).

## 9. Non-inferiority

- Pairs passing all three frozen gates = **3/9**.
- Pre-registered non-inferiority outcome = **QUALITY_NEUTRALITY_NOT_ESTABLISHED**. This must NOT be softened because the mean CI crosses zero.
- Mean-effect inference and pre-registered non-inferiority gate outcome are reported separately (different questions).

## 10. Speed (unchanged measurements, corrected framing)

- overall A0->C0 GM = 1.01089 (~1.09% full-training wall improvement), positive pairs = 0/9.
- Reviewer-safe framing: **small but reproducible same-parent speed benefit (~1.09%)**. NOT "major speedup" / "substantial acceleration".

## 11. Mechanism wording (discipline)

- OBSERVED: first differing tracked variable = n_gaussians at iter 600-700.
- SUPPORTED: fixed-state FP-equivalence within tolerance.
- INFERRED: FP reassociation causes densification threshold flip.
- UNRESOLVED: which exact threshold/event first changed and why.
- Preferred: "The evidence is consistent with FP-level reassociation being amplified through discrete densification decisions, but the exact causal threshold crossing was not directly captured."

## 12. Module attribution (preserved, scoped)

- seed42 SINGLE_SEED_DIAGNOSTIC ONLY: A0->A1 F9 dPSNR = +0.0109, A1->A2 Scalar = -0.1885, A2->A3 H8 = -0.0280.
- Do NOT claim "Scalar causes the multi-seed degradation" unless multi-seed A1/A2 data exist (they do not).

## 13. Corrected reviewer-safe conclusion

C0 remains **algebraically exact at a fixed state** (derivation-supported) and provides a **small but reproducible same-parent training-speed benefit (~1.09%)**. However, **floating-point reassociation prevents bit-identical training trajectories**: the first observed divergence is Gaussian population at iter 600-700, consistent with discrete topology decisions amplifying small numerical differences. Across the 3-scene x 3-seed cohort, **8/9 matched C0 runs have lower PSNR** than A0, while the **aggregate mean dPSNR is statistically inconclusive** (bootstrap CI crosses zero) because of large scene/seed variability, particularly on garden. **Frozen quality non-inferiority is NOT established** (only 3/9 pairs pass all three gates). C0 is therefore a **supporting exact-adjoint/dataflow contribution, not the main performance headline**.

## 14. Publication status (axes kept separate)

- **EVIDENCE_QUALITY = ACCEPTABLE** (this audit is internally consistent and reviewer-safe).
- **C0_AS_PRIMARY_PERFORMANCE_CONTRIBUTION = WEAK** (~1.09% fair speed + quality neutrality not established).
- These axes are distinct and must not be conflated.

## 15. Terminal status

**COMPLETE_REPORT_CORRECTION.**
