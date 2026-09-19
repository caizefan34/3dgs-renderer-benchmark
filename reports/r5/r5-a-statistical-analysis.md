# R5-A: Statistical Analysis

## 1. Pre-Registered Gate Evaluation

### Gate Definition (pre-registered, not modified after seeing results)

| Outcome | Conditions |
|---------|------------|
| **PASS_STRONG** | Both scenes mean ΔPSNR > 0 AND ≥2/3 seeds positive per scene AND ≥5/6 positive overall |
| **PASS_WEAK** | Combined mean ΔPSNR > 0 but only 1 scene reproducible OR 4/6 positive |
| **FAIL** | Combined mean ΔPSNR ≤ 0 OR R4 gains not reproducible |

### Gate Evaluation (using final_results.json for seed-0)

```
train: mean = −0.44 dB, positive = 1/3
truck: mean = +0.98 dB, positive = 2/3
combined: mean = +0.27 dB, positive = 3/6
```

| Condition | Required | Actual | Met? |
|-----------|----------|--------|------|
| Both scenes mean > 0 | train AND truck > 0 | train = −0.44 | ❌ |
| ≥2/3 positive per scene | both scenes ≥ 2/3 | train = 1/3 | ❌ |
| ≥5/6 positive overall | ≥5/6 | 3/6 | ❌ |
| Combined mean > 0 | > 0 | +0.27 | ✅ |
| ≥4/6 positive (WEAK) | ≥4/6 | 3/6 | ❌ |

### **R5-A = FAIL**

The combined mean is technically positive (+0.27 dB), but:
1. Train is NOT reproducible (1/3 positive, mean −0.44)
2. Only 3/6 positive overall (exactly chance level)
3. FAIL condition explicitly states "R4 gains not reproducible" → train R4 gain (seed 0: +0.48) is NOT reproduced in seeds 1 and 2

## 2. Per-Scene Paired t-Test

### Train (n=3)
- Paired ΔPSNR: [+0.4752, −0.6433, −1.1648]
- Mean: −0.4443 dB
- Std: 0.8611 dB
- SEM: 0.4972 dB
- t-statistic: −0.8936
- df: 2
- p-value (one-sided, H₁: ΔPSNR > 0): 0.78
- 95% CI: [−2.59, +1.71]
- **Conclusion**: Cannot reject H₀. No evidence of regularization effect on train.

### Truck (n=3)
- Paired ΔPSNR: [+2.1261, −0.0003, +0.8128]
- Mean: +0.9795 dB
- Std: 1.0535 dB
- SEM: 0.6084 dB
- t-statistic: 1.6102
- df: 2
- p-value (one-sided, H₁: ΔPSNR > 0): 0.13
- 95% CI: [−1.62, +3.57]
- **Conclusion**: Cannot reject H₀ at α=0.05. Trend is positive but not statistically significant with n=3.

### Combined (n=6)
- Paired ΔPSNR: [+0.4752, −0.6433, −1.1648, +2.1261, −0.0003, +0.8128]
- Mean: +0.2676 dB
- Std: 1.0328 dB
- SEM: 0.4215 dB
- t-statistic: 0.6349
- df: 5
- p-value (one-sided, H₁: ΔPSNR > 0): 0.28
- 95% CI: [−0.77, +1.30]
- **Conclusion**: Cannot reject H₀. The +0.27 dB combined mean is not statistically distinguishable from zero.

## 3. Sign Test

### Train
- Positive: 1/3 → P(≥1 positive | n=3, p=0.5) = 0.875 → Not significant

### Truck
- Positive: 2/3 → P(≥2 positive | n=3, p=0.5) = 0.500 → Not significant

### Combined
- Positive: 3/6 → P(≥3 positive | n=6, p=0.5) = 0.656 → Not significant (exactly chance level)

## 4. Effect Size

### Cohen's d (paired)
- Train: d = −0.44/0.86 = −0.52 (medium negative)
- Truck: d = 0.98/1.05 = 0.93 (large positive, but n=3)
- Combined: d = 0.27/1.03 = 0.26 (small)

## 5. Reproducibility Analysis

### R4 Seed 0 Reproducibility

| Scene | R4 ΔPSNR (seed 0) | Seed 1 | Seed 2 | Reproducible? |
|-------|--------------------|--------|--------|----------------|
| train | +0.4752 | −0.6433 | −1.1648 | ❌ NO — sign reversal |
| truck | +2.1261 | −0.0003 | +0.8128 | ⚠️ PARTIAL — 2/3 positive but magnitude varies 0–2.13 |

### Inter-seed Variability

| Scene | Range of ΔPSNR | CV (std/|mean|) | Sign consistency |
|-------|----------------|-----------------|------------------|
| train | [−1.16, +0.48] = 1.64 | 1.94 | 1/3 positive |
| truck | [−0.00, +2.13] = 2.13 | 1.08 | 2/3 positive |

The coefficient of variation > 1.0 for both scenes indicates the effect is dominated by noise rather than a systematic signal.

## 6. SSIM Analysis — Exploratory

All 6 paired comparisons show ΔSSIM < 0:

| Scene | Seed | ΔSSIM |
|-------|------|-------|
| train | 0 | −0.0146 |
| train | 1 | −0.0192 |
| train | 2 | −0.0224 |
| truck | 0 | −0.0014 |
| truck | 1 | −0.0159 |
| truck | 2 | −0.0088 |

- Mean ΔSSIM: −0.0137
- 6/6 negative

### Exact binomial (sign) test

**No directional SSIM hypothesis was pre-registered.** The SSIM analysis is therefore **exploratory**.

| Test | p-value | Formula | Interpretation |
|------|---------|---------|----------------|
| One-sided exact (H₁: ΔSSIM < 0) | 0.015625 | (1/2)^6 = 1/64 | If a directional hypothesis had been pre-registered, this would be significant at α=0.05 |
| Two-sided exact | 0.031250 | 2 × (1/2)^6 = 1/32 | Significant at α=0.05 even without directional pre-registration |

**Label**: Exploratory finding. The two-sided exact p = 0.03125 is nominally significant, but this was not a pre-registered endpoint. The result should be treated as hypothesis-generating, not confirmatory.

## 7. Summary

| Metric | Train | Truck | Combined |
|--------|-------|-------|----------|
| Mean ΔPSNR | −0.44 | +0.98 | +0.27 |
| Positive seeds | 1/3 | 2/3 | 3/6 |
| Paired t-test p (one-sided) | 0.78 | 0.13 | 0.28 |
| Sign test p | 0.875 | 0.500 | 0.656 |
| 95% CI includes 0? | Yes | Yes | Yes |
| R4 reproducible? | No | Partial | No |
| Mean ΔSSIM | −0.019 | −0.009 | −0.014 |
| SSIM sign test (two-sided, exploratory) | — | — | p=0.03125 |

**The R4 positive ΔPSNR on T&T is not reproducible. The regularization hypothesis is unsupported.**

## Note on Seed-0 Source

All seed-0 numbers use `final_results.json` (R4 original run, mtime 2026-09-18 23:58:04).
The candidate_c `training_metrics.json` in the R4 directory was overwritten by an unidentified
re-run at 2026-09-19 01:04–01:07. See `r5-a-provenance-reconciliation.md` for full forensics.

The gate result is identical (FAIL) under both the original and re-run seed-0 numbers.
