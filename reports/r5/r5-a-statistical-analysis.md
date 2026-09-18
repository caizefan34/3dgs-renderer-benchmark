# R5-A: Statistical Analysis

## 1. Pre-Registered Gate Evaluation

### Gate Definition (pre-registered, not modified after seeing results)

| Outcome | Conditions |
|---------|------------|
| **PASS_STRONG** | Both scenes mean ΔPSNR > 0 AND ≥2/3 seeds positive per scene AND ≥5/6 positive overall |
| **PASS_WEAK** | Combined mean ΔPSNR > 0 but only 1 scene reproducible OR 4/6 positive |
| **FAIL** | Combined mean ΔPSNR ≤ 0 OR R4 gains not reproducible |

### Gate Evaluation

```
train: mean = −0.43 dB, positive = 1/3
truck: mean = +0.92 dB, positive = 2/3
combined: mean = +0.24 dB, positive = 3/6
```

| Condition | Required | Actual | Met? |
|-----------|----------|--------|------|
| Both scenes mean > 0 | train AND truck > 0 | train = −0.43 | ❌ |
| ≥2/3 positive per scene | both scenes ≥ 2/3 | train = 1/3 | ❌ |
| ≥5/6 positive overall | ≥5/6 | 3/6 | ❌ |
| Combined mean > 0 | > 0 | +0.24 | ✅ |
| ≥4/6 positive (WEAK) | ≥4/6 | 3/6 | ❌ |

### **R5-A = FAIL**

The combined mean is technically positive (+0.24 dB), but:
1. Train is NOT reproducible (1/3 positive, mean −0.43)
2. Only 3/6 positive overall (exactly chance level)
3. FAIL condition explicitly states "R4 gains not reproducible" → train R4 gain (seed 0: +0.52) is NOT reproduced in seeds 1 and 2

## 2. Per-Scene Paired t-Test

### Train (n=3)
- Paired ΔPSNR: [+0.52, −0.64, −1.16]
- Mean: −0.43 dB
- Std: 0.86 dB
- SEM: 0.50 dB
- t-statistic: −0.87
- df: 2
- p-value (one-sided, H₁: ΔPSNR > 0): 0.78 (i.e., P(data | H₀) is more likely)
- 95% CI: [−2.57, +1.71]
- **Conclusion**: Cannot reject H₀. No evidence of regularization effect on train.

### Truck (n=3)
- Paired ΔPSNR: [+1.94, −0.00, +0.81]
- Mean: +0.92 dB
- Std: 0.97 dB
- SEM: 0.56 dB
- t-statistic: 1.64
- df: 2
- p-value (one-sided, H₁: ΔPSNR > 0): 0.12
- 95% CI: [−1.50, +3.34]
- **Conclusion**: Cannot reject H₀ at α=0.05. Trend is positive but not statistically significant with n=3.

### Combined (n=6)
- Paired ΔPSNR: [+0.52, −0.64, −1.16, +1.94, −0.00, +0.81]
- Mean: +0.24 dB
- Std: 1.00 dB
- SEM: 0.41 dB
- t-statistic: 0.59
- df: 5
- p-value (one-sided, H₁: ΔPSNR > 0): 0.29
- 95% CI: [−0.74, +1.22]
- **Conclusion**: Cannot reject H₀. The +0.24 dB combined mean is not statistically distinguishable from zero.

## 3. Sign Test

### Train
- Positive: 1/3 → P(≥1 positive | n=3, p=0.5) = 0.875 → Not significant

### Truck
- Positive: 2/3 → P(≥2 positive | n=3, p=0.5) = 0.500 → Not significant

### Combined
- Positive: 3/6 → P(≥3 positive | n=6, p=0.5) = 0.656 → Not significant (exactly chance level)

## 4. Effect Size

### Cohen's d (paired)
- Train: d = −0.43/0.86 = −0.50 (medium negative)
- Truck: d = 0.92/0.97 = 0.95 (large positive, but n=3)
- Combined: d = 0.24/1.00 = 0.24 (small)

## 5. Reproducibility Analysis

### R4 Seed 0 Reproducibility

| Scene | R4 ΔPSNR (seed 0) | Seed 1 | Seed 2 | Reproducible? |
|-------|--------------------|--------|--------|----------------|
| train | +0.52 | −0.64 | −1.16 | ❌ NO — sign reversal |
| truck | +1.94 | −0.00 | +0.81 | ⚠️ PARTIAL — 2/3 positive but magnitude varies 0-1.94 |

### Inter-seed Variability

| Scene | Range of ΔPSNR | CV (std/|mean|) | Sign consistency |
|-------|----------------|-----------------|------------------|
| train | [−1.16, +0.52] = 1.68 | 2.00 | 1/3 positive |
| truck | [−0.00, +1.94] = 1.94 | 1.05 | 2/3 positive |

The coefficient of variation > 1.0 for both scenes indicates the effect is dominated by noise rather than a systematic signal.

## 6. SSIM Analysis

All 6 paired comparisons show ΔSSIM < 0:

| Scene | Seed | ΔSSIM |
|-------|------|-------|
| train | 0 | −0.0130 |
| train | 1 | −0.0192 |
| train | 2 | −0.0224 |
| truck | 0 | −0.0042 |
| truck | 1 | −0.0159 |
| truck | 2 | −0.0088 |

- Mean ΔSSIM: −0.0139
- 6/6 negative — binomial test P(6/6 | p=0.5) = 0.016
- **Statistically significant SSIM degradation** (p < 0.05)

This is the only statistically significant finding: Candidate C consistently degrades SSIM, even when PSNR improves.

## 7. Summary

| Metric | Train | Truck | Combined |
|--------|-------|-------|----------|
| Mean ΔPSNR | −0.43 | +0.92 | +0.24 |
| Positive seeds | 1/3 | 2/3 | 3/6 |
| Paired t-test p (one-sided) | 0.78 | 0.12 | 0.29 |
| Sign test p | 0.875 | 0.500 | 0.656 |
| 95% CI includes 0? | Yes | Yes | Yes |
| R4 reproducible? | No | Partial | No |
| Mean ΔSSIM | −0.018 | −0.010 | −0.014 |
| SSIM sign test | 3/3 neg* | 3/3 neg* | 6/6 neg** |

*Not significant per-seene (n=3), **Significant at p=0.016

**The R4 positive ΔPSNR on T&T is not reproducible. The regularization hypothesis is unsupported.**
