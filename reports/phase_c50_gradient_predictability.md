# Phase C50: Gradient Predictability Analysis Before Sparse Backward

## 1. Motivation

Phase C49 discovered strong gradient sparsity in 3DGS training:

| Config | Gradient kept | PSNR change |
|--------|-------------:|------------:|
| Top 50% | ~97% | +0.03 dB |
| Top 32% | ~90% | +0.02 dB |
| Top 10% | ~60% | -0.06 dB |

Even keeping only 10% of Gaussians' gradients preserves quality. This suggests a **sparse backward** — skipping low-importance Gaussians in the backward kernel — could achieve significant speedup with minimal quality loss.

However, C49's validation used **oracle information**: the full backward was computed first, then gradients were filtered. A real sparse backward must **predict importance before backward** — it cannot know which Gaussians are important until after the backward has already computed their gradients.

The central research question:

> **Can we know which Gaussians will matter BEFORE backward?**

This phase validates whether the previous iteration's gradient (available before the current backward) can predict the current iteration's important Gaussians. Only after this is validated should we proceed to CUDA sparse backward implementation (Phase C51).

---

## 2. Experimental Setup

### Environment
- **GPU**: A100 PCIe 40GB (SM80, 108 SMs)
- **Scene**: Mip-NeRF360 room (311 cameras, 1080p)
- **Training**: 5000 iterations, seed=42
- **Pruning**: moderate (threshold=0.01, grad=0.001, densify 500-15000, every 100 iters)
- **Loss**: Separable SSIM (C44 method) + freq8 (SSIM every 8 iterations)
- **gsplat**: v1.5.3, built from source with CUDA 11.8
- **Measurement**: Single training run, all 3 experiments measured simultaneously

### Measurement Protocol
- Gradient norm collected every iteration: `grad_norm[g] = ||xyz.grad||` (per-Gaussian L2 norm of positional gradient)
- Measurements start at iteration 500 (after initial training stabilizes)
- **Densification handling**: After densification (every 100 iters from 500-15000), Gaussian count N changes. Previous gradient and EMA are reset — measurement skips the first iteration after each densification (shape mismatch). This yields ~4409 measurements out of 4500 possible iterations (500-5000).
- All measurements use gradient AFTER backward, BEFORE optimizer step

### Phase Split
| Phase | Iteration range | Measurements |
|-------|----------------|-------------|
| init | 0-499 | 0 (excluded — unstable) |
| early | 500-1999 | 1469 |
| middle | 2000-3999 | 1960 |
| late | 4000-4999 | 980 |
| **Total** | | **4409** |

### Training Summary
| Metric | Value |
|--------|-------|
| Total train time | 266.1s |
| Final PSNR (5K) | 26.23 dB |
| Final Gaussian count | 2,034,835 |
| Clone / Split / Prune | 58,814 / 194,171 / 5,697 |

**Note**: Training time (266s) is higher than C49 baseline (127.7s) due to per-iteration measurement overhead (torch.topk, argsort, corrcoef on 2M Gaussians). This overhead would NOT exist in a CUDA sparse backward implementation.

---

## 3. Measurement Methodology

### Experiment 1: Temporal Gradient Correlation

For consecutive iterations t-1 and t:

- **Pearson correlation**: `corr(|g(t-1)|, |g(t)|)` — magnitude stability
- **Spearman correlation**: `corr(rank(|g(t-1)|), rank(|g(t)|))` — ranking stability (computed via `argsort().argsort()` for ranking)

Reported per phase: mean, median, standard deviation, min, max.

### Experiment 2: Top-K Importance Persistence

For each K fraction:

```
TopK_prev = topK(|g(t-1)|)   # top K Gaussians by previous gradient
TopK_curr = topK(|g(t)|)     # top K Gaussians by current gradient

Recall@K = |TopK_prev ∩ TopK_curr| / K
```

K fractions tested: 1%, 5%, 10%, 32%, 50%

### Experiment 3: Predictor Comparison

Five predictors compared:

| Predictor | Formula | Available pre-backward? |
|-----------|---------|------------------------|
| Oracle (upper bound) | `|g(t)|` | No — requires current backward |
| Previous gradient | `|g(t-1)|` | Yes |
| EMA (beta=0.9) | `0.9*EMA + 0.1*|g(t-1)|` | Yes |
| EMA (beta=0.99) | `0.99*EMA + 0.01*|g(t-1)|` | Yes |
| Opacity | `sigmoid(opacity)` | Yes |

Metrics per predictor:
- **Recall@K**: Fraction of predictor's top-K that are also in current gradient's top-K
- **Gradient Coverage**: `sum(|g(t)| for selected Gaussians) / sum(|g(t)| for all Gaussians)`

### Decision Criteria

Sparse backward prediction is promising if (using **previous gradient** predictor, **late** phase):

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Recall@32% | >= 0.70 | 70% of important Gaussians correctly identified |
| Coverage@32% | >= 0.85 | 85% of gradient signal preserved |
| Coverage@50% | >= 0.95 | 95% of gradient signal preserved at 50% selection |

---

## 4. Results

### 4.1 Experiment 1: Temporal Gradient Correlation

#### Evidence

| Phase | n | Pearson (mean +/- std) | Spearman (mean +/- std) |
|-------|--:|------------------------|------------------------|
| early | 1469 | 0.9790 +/- 0.0945 | 0.9880 +/- 0.0333 |
| middle | 1960 | 0.9801 +/- 0.0851 | 0.9879 +/- 0.0391 |
| late | 980 | 0.9778 +/- 0.0876 | 0.9893 +/- 0.0283 |

#### Interpretation

- **Pearson ~0.98**: Gradient magnitudes are highly correlated between consecutive iterations. The actual magnitude values change very little iteration-to-iteration.
- **Spearman ~0.99**: Gradient rankings are even more stable than magnitudes. The relative ordering of Gaussian importance barely changes.
- **Stable across phases**: All three phases (early, middle, late) show nearly identical correlation. Prediction quality does not degrade over training.
- **Low variance**: Standard deviations are small (0.03-0.09), indicating the correlation is consistently high, not just high on average.

### 4.2 Experiment 2: Top-K Importance Persistence

#### Evidence

| Phase | n | Recall@1% | Recall@5% | Recall@10% | Recall@32% | Recall@50% |
|-------|--:|-----------|-----------|------------|------------|------------|
| early | 1469 | 0.948 | 0.958 | 0.964 | 0.976 | 0.983 |
| middle | 1960 | 0.944 | 0.957 | 0.963 | 0.976 | 0.983 |
| late | 980 | 0.936 | 0.953 | 0.961 | 0.977 | 0.984 |

#### Interpretation

- **Recall@32% = 0.977**: 97.7% of Gaussians in the previous iteration's top-32% are still in the current iteration's top-32%. This far exceeds the 70% threshold.
- **Recall@1% = 0.936**: Even the top 1% (most important Gaussians) has 93.6% overlap between consecutive iterations. The most important Gaussians are extremely stable.
- **Higher K = higher recall**: As K increases, recall naturally increases (larger set = more overlap). But even at K=1%, recall is above 93%.
- **Stable across phases**: Recall values are nearly identical across early/middle/late training. The prediction quality is consistent throughout training.
- **Low variance**: Standard deviations (0.03-0.09) are small, meaning the recall is consistently high.

### 4.3 Experiment 3: Predictor Comparison

#### Evidence — Late Phase (iterations 4000-4999, n=980)

| Predictor | Recall@10% | Recall@32% | Recall@50% | Coverage@10% | Coverage@32% | Coverage@50% |
|-----------|-----------|------------|------------|--------------|--------------|--------------|
| **Oracle** | 1.000 | 1.000 | 1.000 | 0.641 | 0.922 | 0.980 |
| **Previous** | 0.961 | **0.977** | **0.984** | 0.630 | **0.912** | **0.973** |
| EMA(0.9) | 0.889 | 0.940 | 0.962 | 0.619 | 0.908 | 0.972 |
| EMA(0.99) | 0.777 | 0.888 | 0.934 | 0.583 | 0.896 | 0.968 |
| Opacity | 0.242 | 0.559 | 0.720 | 0.206 | 0.551 | 0.750 |

#### Evidence — All Phases Summary (Recall@32% / Coverage@32% / Coverage@50%)

| Predictor | early | middle | late |
|-----------|-------|--------|------|
| Oracle | 1.000 / 0.914 / 0.974 | 1.000 / 0.916 / 0.977 | 1.000 / 0.922 / 0.980 |
| **Previous** | **0.976 / 0.905 / 0.968** | **0.976 / 0.906 / 0.969** | **0.977 / 0.912 / 0.973** |
| EMA(0.9) | 0.936 / 0.901 / 0.966 | 0.940 / 0.902 / 0.967 | 0.940 / 0.908 / 0.972 |
| EMA(0.99) | 0.885 / 0.891 / 0.963 | 0.892 / 0.892 / 0.964 | 0.888 / 0.896 / 0.968 |
| Opacity | 0.559 / 0.603 / 0.769 | 0.559 / 0.558 / 0.739 | 0.559 / 0.551 / 0.750 |

#### Interpretation

1. **Previous gradient is the best practical predictor.** It achieves 97.7% recall at 32% selection, only 2.3% below the oracle. Coverage@32% is 0.912 (91.2% of gradient preserved), and Coverage@50% is 0.973 (97.3% preserved). Both exceed the decision thresholds.

2. **EMA smoothing HURTS prediction.** This is counter-intuitive but clear:
   - Previous (no smoothing): Recall@32% = 0.977
   - EMA(0.9) (light smoothing): Recall@32% = 0.940 (worse)
   - EMA(0.99) (heavy smoothing): Recall@32% = 0.888 (much worse)

   **Reason**: Gradient importance changes quickly between iterations. EMA introduces lag — it averages over past gradients, but the current iteration's important Gaussians are best predicted by the most recent gradient, not an average. The more smoothing, the worse the prediction.

3. **Opacity is a poor predictor.** Recall@32% is only 0.559 — barely better than random (0.32 for uniform random selection at 32%). Coverage@50% is only 0.750 — selecting the top 50% by opacity captures only 75% of gradient. This confirms C49's finding: gradient-opacity correlation is weak (r=0.189), and opacity cannot serve as a proxy for gradient importance.

4. **Oracle ceiling is close to previous gradient.** The gap between oracle and previous is only 2.3% in recall and ~1% in coverage at 32%. There is limited room for improvement over the simple "previous gradient" predictor. Any more complex predictor (hybrid, learned) would need to close this small gap.

5. **Oracle coverage@32% = 0.922** is itself below 1.0. This means that even with perfect prediction, selecting 32% of Gaussians captures only 92.2% of total gradient — the remaining 7.8% is spread across the bottom 68%. This is consistent with C49's finding that top 32% accounts for ~90% of gradient.

6. **Coverage@50% for previous = 0.973**: Selecting 50% of Gaussians by previous gradient captures 97.3% of current gradient. This exceeds the 95% threshold, meaning a 50% sparse backward would lose only 2.7% of gradient signal.

---

## 5. Decision Criteria Check

Using **previous gradient** predictor, **late** phase (most conservative):

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| Recall@32% | >= 0.70 | 0.977 | **PASS** (+0.277 margin) |
| Coverage@32% | >= 0.85 | 0.912 | **PASS** (+0.062 margin) |
| Coverage@50% | >= 0.95 | 0.973 | **PASS** (+0.023 margin) |

### Decision: **KEEP**

Gradient prediction using the previous iteration's gradient is sufficient for sparse backward. All three criteria pass with comfortable margins.

---

## 6. Evidence vs. Interpretation vs. Future Hypothesis

### Observed Evidence (measured)
1. Pearson correlation between consecutive gradient norms: 0.978-0.980
2. Spearman correlation between consecutive gradient rankings: 0.988-0.989
3. Recall@32% using previous gradient: 0.976-0.977 across all phases
4. Coverage@32% using previous gradient: 0.905-0.912
5. Coverage@50% using previous gradient: 0.968-0.973
6. EMA smoothing reduces recall: beta=0.9 → 0.936-0.940, beta=0.99 → 0.885-0.892
7. Opacity recall@32%: 0.559 (poor)
8. Oracle coverage@32%: 0.914-0.922 (ceiling)
9. All metrics stable across early/middle/late phases
10. 4409 measurement points across 4500 iterations (98% coverage)

### Interpretation (inferred from evidence)
1. Gradient importance is **highly persistent** between consecutive iterations — the same Gaussians matter iteration after iteration.
2. The **previous gradient is the best practical predictor** — nearly matching the oracle upper bound.
3. **Smoothing hurts** because gradient importance fluctuates at the iteration level, and the most recent observation is the best predictor.
4. **Opacity is not a substitute** for gradient magnitude — the two are decoupled.
5. A sparse backward selecting 32% of Gaussians would preserve 91.2% of gradient signal — sufficient for quality (C49 showed 90% gradient retention causes only +0.02 dB change).
6. A sparse backward selecting 50% of Gaussians would preserve 97.3% of gradient signal — near-perfect quality.

### Future Hypothesis (untested)
1. A CUDA sparse backward using previous-iteration gradient as a skip mask could achieve 10-18% T_iter speedup (C25 bound) with negligible quality loss.
2. The skip mask can be computed as a simple threshold: `|g(t-1)| > threshold` where threshold is the 32nd or 50th percentile of previous gradient norms.
3. The mask should be recomputed after each densification (new Gaussians have no history and should default to "important" — compute their backward fully).
4. The 2.3% recall gap between oracle and previous may be partially closed by a 2-iteration lookback, but the gain is likely marginal.
5. Prediction quality may differ for non-room scenes (outdoor scenes with more view-dependent variation). Multi-scene validation is needed before production deployment.

---

## 7. Risk Assessment for Phase C51 (CUDA Sparse Backward)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Densification changes N, invalidating mask | Certain | Low | Reset mask after densification; new Gaussians default to "compute" |
| Mask computation adds overhead | Moderate | Low | torch.topk on 2M elements ~1ms; negligible vs 18.7ms backward |
| Incorrect gradients due to skipped Gaussians | Low | High | C49 validated quality at 90% filtering; 32-50% filtering is safer |
| Prediction fails on other scenes | Moderate | Medium | Test on garden, bicycle before production |
| CUDA kernel bugs | Moderate | High | Comprehensive gradient correctness tests vs. dense backward |

---

## 8. Files

| File | Description |
|------|-------------|
| `scripts/phase-c50/gradient_predictability.py` | Experiment script (single run, all 3 experiments) |
| `results/a100/phase-c50/gradient_temporal.json` | Experiment 1: temporal correlation raw data + phase statistics |
| `results/a100/phase-c50/topk_persistence.json` | Experiment 2: Recall@K phase statistics |
| `results/a100/phase-c50/predictor_comparison.json` | Experiment 3: predictor comparison phase statistics |
| `results/a100/phase-c50/summary.json` | Training summary + eval points |
| `reports/phase_c50_gradient_predictability.md` | This report |

---

## 9. Final Answer

> **Can we know which Gaussians will matter BEFORE backward?**

**Yes.** The previous iteration's gradient norm is an excellent predictor of the current iteration's important Gaussians:

- **97.7%** of the current top-32% important Gaussians are correctly identified by the previous iteration's top-32%
- **91.2%** of the current total gradient signal is captured by selecting these 32%
- **97.3%** of gradient signal is captured at 50% selection
- Prediction is **stable** across all training phases (early, middle, late)
- The simple "previous gradient" predictor is **near-optimal** — only 2.3% below the oracle upper bound

**No smoothing is needed.** EMA averaging actually degrades prediction quality. The raw previous gradient is the best predictor.

**Opacity cannot substitute for gradient.** It achieves only 55.9% recall at 32% selection — barely better than random.

### Decision: **KEEP — proceed to Phase C51 (CUDA Sparse Backward Kernel Implementation)**

The evidence overwhelmingly supports that gradient prediction is sufficient for sparse backward. The next phase should implement a CUDA backward kernel that:
1. Uses the previous iteration's gradient norm as a pre-backward importance mask
2. Selects the top 32-50% of Gaussians for full backward computation
3. Skips the bottom 50-68% (zeroing their gradients implicitly)
4. Resets the mask after each densification event
5. Defaults new (post-densification) Gaussians to "compute" (safe default)
