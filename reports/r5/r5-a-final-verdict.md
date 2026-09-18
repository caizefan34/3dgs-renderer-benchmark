# R5-A: Final Verdict

## R5-A = FAIL

### Pre-Registered Gate Result

**R5-A: Tanks & Temples Regularization Reproducibility Gate — FAIL**

The R4 positive ΔPSNR on T&T (train +0.48/+0.52, truck +2.13/+1.94) is **not reproducible** across paired multi-seed experiments. The regularization hypothesis is unsupported by current evidence.

## Pre-Registered Gate Conditions (not modified post-hoc)

| Outcome | Conditions |
|---------|------------|
| PASS_STRONG | Both scenes mean ΔPSNR > 0 AND ≥2/3 seeds positive per scene AND ≥5/6 positive overall |
| PASS_WEAK | Combined mean ΔPSNR > 0 but only 1 scene reproducible OR 4/6 positive |
| FAIL | Combined mean ΔPSNR ≤ 0 OR R4 gains not reproducible |

## Results vs. Gate

| Condition | Required | Actual | Met? |
|-----------|----------|--------|------|
| Both scenes mean > 0 | train AND truck | train = −0.43 dB | ❌ |
| ≥2/3 positive per scene | both ≥ 2/3 | train = 1/3 | ❌ |
| ≥5/6 positive overall | ≥5/6 | 3/6 | ❌ |
| Combined mean > 0 | > 0 | +0.24 dB | ✅ |
| ≥4/6 positive (WEAK) | ≥4/6 | 3/6 | ❌ |

**FAIL condition met**: "R4 gains not reproducible" — train seed 0 gain (+0.52) reversed in seeds 1 (−0.64) and 2 (−1.16).

## Complete Paired Results

### ΔPSNR by Scene and Seed

| Scene | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Positive |
|-------|--------|--------|--------|------|-----|----------|
| train | +0.52 | −0.64 | −1.16 | **−0.43** | 0.86 | 1/3 |
| truck | +1.94 | −0.00 | +0.81 | **+0.92** | 0.97 | 2/3 |
| **Combined** | | | | **+0.24** | 1.00 | **3/6** |

### Statistical Tests

| Test | Train | Truck | Combined |
|------|-------|-------|----------|
| Paired t-test p (one-sided) | 0.78 | 0.12 | 0.29 |
| Sign test p | 0.875 | 0.500 | 0.656 |
| 95% CI | [−2.57, +1.71] | [−1.50, +3.34] | [−0.74, +1.22] |
| Rejects H₀? | No | No | No |

### SSIM (all 6 pairs)

| Metric | Value |
|--------|-------|
| Mean ΔSSIM | −0.0139 |
| Negative pairs | 6/6 |
| Sign test p | 0.016 (**significant**) |

Candidate C consistently degrades SSIM (p = 0.016). This is the only statistically significant finding.

## Root Cause Analysis

### Why did R4 show positive ΔPSNR on T&T?

The R4 result was a **seed-0 coincidence**. With n=1 per scene, the 2 positive results out of 13 total scenes (11 negative) had a P = C(13,2)/2^13 ≈ 1.1% probability of occurring by chance — unlikely but not impossible. The 2 positive results happened to both be T&T scenes, creating a false pattern.

Multi-seed testing reveals:
- Train: seed 0 was the ONLY positive seed (1/3). The +0.52 dB gain reverses to −0.64 and −1.16 with different seeds.
- Truck: 2/3 positive, but seed 1 is exactly 0.00 (no effect) and the magnitude varies wildly (0 to +1.94).

### Densification perturbation

Candidate C consistently produces 8-15% more Gaussians than baseline (R_GS = 1.08-1.15), with the divergence occurring during the active densification phase (iter 500-15000). This is a real systematic effect of gradient suppression on densification, but it does NOT translate to reproducible quality improvement. The extra Gaussians appear to be "phantom" densification — triggered by modified gradient signals rather than genuine quality-driven densification.

## Decision: Close T&T Regularization Branch

Per the pre-registered protocol:

> **FAIL → close T&T regularization branch, no R5-B.**

### R5-B will NOT be executed.

The certificate-guided backward skip mechanism does not produce a reproducible regularization/generalization effect on Tanks & Temples scenes. The R4 positive results were seed noise, not a systematic signal.

## Implications for Candidate C

| Aspect | Status |
|--------|--------|
| As accelerator | DROPPED (R4: 2.7× slower, avg −1.52 dB across 13 scenes) |
| As regularizer on T&T | DROPPED (R5-A: not reproducible, FAIL gate) |
| SSIM impact | Consistently negative (6/6 pairs, p=0.016) |
| Densification impact | Systematic over-densification (+8-15% Gaussians) without quality benefit |
| Overall verdict | **ABANDON Candidate C entirely** |

## Files Produced

| File | Description |
|------|-------------|
| `reports/r5/r5-a0-r4-seed-config-audit.md` | R4 seed/config provenance audit (ORIGINAL_R4_PAIR_REUSABLE=YES) |
| `reports/r5/r5-a-run-manifest.md` | Run manifest (cohort, schedule, config) |
| `reports/r5/r5-a-multiseed-results.md` | Complete 12-run results table |
| `reports/r5/r5-a-statistical-analysis.md` | Paired t-test, sign test, effect size, CIs |
| `reports/r5/r5-a-densification-diagnostics.md` | N_GS(t), R_GS(t), phase analysis |
| `reports/r5/r5-a-final-verdict.md` | This file |
| `reports/r5/r5-a-results.json` | Machine-readable results (on remote) |

## Experimental Integrity

- ✅ Pre-registered gate used without modification
- ✅ R4 seed=0 pair verified reusable (provenance audit)
- ✅ 3 seeds × 2 scenes × 2 methods = 12 complete paired runs
- ✅ All runs completed to 30K iterations
- ✅ Identical configs (verified by config_sha256)
- ✅ Same hardware cohort (A100-PCIE-40GB, gsplat 1.5.3, PyTorch 2.4.1+cu124)
- ✅ Frozen Candidate C (no modifications from R4)
- ✅ No p-hacking (gate evaluated exactly as pre-registered)
