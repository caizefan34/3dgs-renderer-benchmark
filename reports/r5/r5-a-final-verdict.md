# R5-A: Final Verdict

## R5-A = FAIL

### Pre-Registered Gate Result

**R5-A: Tanks & Temples Regularization Reproducibility Gate — FAIL**

The R4 positive ΔPSNR on T&T (train +0.48, truck +2.13) is **not reproducible** across paired multi-seed experiments. The regularization hypothesis is unsupported by current evidence.

## Pre-Registered Gate Conditions (not modified post-hoc)

| Outcome | Conditions |
|---------|------------|
| PASS_STRONG | Both scenes mean ΔPSNR > 0 AND ≥2/3 seeds positive per scene AND ≥5/6 positive overall |
| PASS_WEAK | Combined mean ΔPSNR > 0 but only 1 scene reproducible OR 4/6 positive |
| FAIL | Combined mean ΔPSNR ≤ 0 OR R4 gains not reproducible |

## Results vs. Gate (seed-0 from final_results.json, R4 original)

| Condition | Required | Actual | Met? |
|-----------|----------|--------|------|
| Both scenes mean > 0 | train AND truck | train = −0.44 dB | ❌ |
| ≥2/3 positive per scene | both ≥ 2/3 | train = 1/3 | ❌ |
| ≥5/6 positive overall | ≥5/6 | 3/6 | ❌ |
| Combined mean > 0 | > 0 | +0.27 dB | ✅ |
| ≥4/6 positive (WEAK) | ≥4/6 | 3/6 | ❌ |

**FAIL condition met**: "R4 gains not reproducible" — train seed 0 gain (+0.48) reversed in seeds 1 (−0.64) and 2 (−1.16).

## Complete Paired Results

### ΔPSNR by Scene and Seed

| Scene | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Positive |
|-------|--------|--------|--------|------|-----|----------|
| train | +0.4752 | −0.6433 | −1.1648 | **−0.44** | 0.86 | 1/3 |
| truck | +2.1261 | −0.0003 | +0.8128 | **+0.98** | 1.05 | 2/3 |
| **Combined** | | | | **+0.27** | 1.03 | **3/6** |

### Statistical Tests

| Test | Train | Truck | Combined |
|------|-------|-------|----------|
| Paired t-test p (one-sided) | 0.78 | 0.13 | 0.28 |
| Sign test p | 0.875 | 0.500 | 0.656 |
| 95% CI | [−2.59, +1.71] | [−1.62, +3.57] | [−0.77, +1.30] |
| Rejects H₀? | No | No | No |

### SSIM (all 6 pairs) — Exploratory

| Metric | Value |
|--------|-------|
| Mean ΔSSIM | −0.0137 |
| Negative pairs | 6/6 |
| One-sided exact binomial p (H₁: ΔSSIM < 0) | 0.015625 (1/64) |
| Two-sided exact binomial p | 0.031250 (1/32) |

**No directional SSIM hypothesis was pre-registered.** The SSIM analysis is therefore **exploratory**.
The two-sided exact p = 0.03125 is nominally significant at α=0.05, but this was not a pre-registered
endpoint and should be treated as hypothesis-generating, not confirmatory.

## Root Cause Analysis

### Why did R4 show positive ΔPSNR on T&T?

The R4 result was a **seed-0 coincidence**. Multi-seed testing reveals:

- Train: seed 0 was the ONLY positive seed (1/3). The +0.48 dB gain reverses to −0.64 and −1.16 with different seeds.
- Truck: 2/3 positive, but seed 1 is −0.0003 (effectively zero) and the magnitude varies wildly (0 to +2.13).

The R4 report observed 2 positive scenes out of 13 total. No pre-registered null hypothesis or
probability calculation was specified for this pattern. Post-hoc probability claims about the
likelihood of observing 2/13 positive results by chance are not appropriate as a basis for
inferring that the T&T pattern was unlikely under a pre-specified null, because no such null was
pre-registered and the 13 scenes are not exchangeable (they span 3 datasets with different
characteristics). The multi-seed replication test (R5-A) is the appropriate falsification
mechanism, and it returns FAIL.

### Densification perturbation

Candidate C consistently produces 8–15% more Gaussians than baseline (R_GS = 1.08–1.15), with the divergence occurring during the active densification phase (iter 500–15000). This is a real systematic effect of gradient suppression on densification, but it does NOT translate to reproducible quality improvement. The extra Gaussians appear to be "phantom" densification — triggered by modified gradient signals rather than genuine quality-driven densification.

### Provenance discrepancy

The R4 candidate_c `training_metrics.json` for train/truck was overwritten by an unidentified
re-run at 2026-09-19 01:04–01:07. The original metrics survive in `final_results.json`
(mtime 2026-09-18 23:58:04, sha256 prefix `ea645ac3a0f61f1b`), which is the authoritative seed-0
source used in all corrected R5-A reports. The gate result (FAIL) is identical under both the
original and re-run seed-0 numbers. See `r5-a-provenance-reconciliation.md` for full forensics.

## Decision: Close T&T Regularization Branch

Per the pre-registered protocol:

> **FAIL → close T&T regularization branch, no R5-B.**

### R5-B will NOT be executed.

The certificate-guided backward skip mechanism does not produce a reproducible regularization/generalization effect on Tanks & Temples scenes. The R4 positive results were seed noise, not a systematic signal.

## Final State

| Item | Status |
|------|--------|
| Candidate C accelerator | **DROP** (R4: 2.7× slower, avg −1.52 dB across 13 scenes) |
| Candidate C T&T regularization | **DROP** (R5-A: not reproducible, FAIL gate) |
| Candidate C | **CLOSED** |
| R5-B | **CANCELLED** |

## Implications for Candidate C

| Aspect | Status |
|--------|--------|
| As accelerator | DROPPED (R4: 2.7× slower, avg −1.52 dB across 13 scenes) |
| As regularizer on T&T | DROPPED (R5-A: not reproducible, FAIL gate) |
| SSIM impact | Consistently negative (6/6 pairs, two-sided exact p=0.03125, exploratory) |
| Densification impact | Systematic over-densification (+8–15% Gaussians) without quality benefit |
| Overall verdict | **ABANDON Candidate C entirely** |

## Files Produced

| File | Description |
|------|-------------|
| `reports/r5/r5-a0-r4-seed-config-audit.md` | R4 seed/config provenance audit (ORIGINAL_R4_PAIR_REUSABLE=YES, with metrics overwrite caveat) |
| `reports/r5/r5-a-provenance-reconciliation.md` | R4 seed-0 metrics discrepancy forensics and resolution |
| `reports/r5/r5-a-run-manifest.md` | Run manifest (cohort, schedule, config) |
| `reports/r5/r5-a-multiseed-results.md` | Complete 12-run results table (corrected seed-0) |
| `reports/r5/r5-a-statistical-analysis.md` | Paired t-test, sign test, effect size, CIs, SSIM exploratory analysis |
| `reports/r5/r5-a-densification-diagnostics.md` | N_GS(t), R_GS(t), phase analysis (with seed-0 caveat) |
| `reports/r5/r5-a-final-verdict.md` | This file |
| `reports/r5/r5-a-results.json` | Machine-readable results |
| `reports/r5/figures/` | Convergence visualizations (3 PNG files) |

## Experimental Integrity

- ✅ Pre-registered gate used without modification
- ✅ R4 seed=0 pair provenance verified (provenance.json, camera_sequence.npy from original run)
- ✅ Seed-0 metrics sourced from `final_results.json` (authoritative R4 original artifact)
- ✅ Provenance discrepancy documented and resolved (see r5-a-provenance-reconciliation.md)
- ✅ 3 seeds × 2 scenes × 2 methods = 12 complete paired runs
- ✅ All runs completed to 30K iterations
- ✅ Identical configs (verified by config_sha256)
- ✅ Same hardware cohort (A100-PCIE-40GB, gsplat 1.5.3, PyTorch 2.4.1+cu124)
- ✅ Frozen Candidate C (no modifications from R4)
- ✅ No p-hacking (gate evaluated exactly as pre-registered)
- ✅ SSIM analysis labeled exploratory (no directional hypothesis pre-registered)
- ✅ Post-hoc probability claim removed (no pre-specified null for 2/13 pattern)
