# Phase C53-Discovery — Gaussian Computational Utility: Predictive Signal Audit

## 0. Research Objective

C49–C52 established that gradient importance is concentrated (C49), previous gradient predicts current gradient (C50), Gaussian-level backward masking changes densification dynamics (C51), but explicitly ranking densification candidates by predicted gradient does NOT consistently outperform uniform allocation (C52).

C53 asks a more fundamental question:

> **What observable signal available at iteration t can best predict the actual computational/optimization utility of Gaussian i during future iterations?**

This is a DISCOVERY phase — no algorithm design, no speedup benchmarks, no feature combinations.

---

## 1. Experimental Design

### Scenes
- **Primary**: room (1.6M initial GS), garden (1.8M initial GS)
- **Secondary**: bicycle (6.1M initial GS)
- **Controls**: room seed=123, garden seed=123

### Training
Canonical C45/C51/C52 configuration: 1920×1080, seed=42/123, L1+D-SSIM, Adam eps=1e-15, canonical LR/densification/pruning, 30K iterations, 8 GPUs.

### Checkpoints
8 fixed checkpoints: 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000.
At each checkpoint: sample 100K Gaussians (fixed seed), record 7 signals, track future utility over Δ=10/50/100 iterations.

### Signals (no combinations)
| Signal | Description | Source |
|--------|------------|--------|
| S1 prev_grad_norm | Gradient norm from previous iteration | model.xyz.grad |
| S2 ema_grad_norm | EMA(decay=0.9) of per-iteration gradient norm | Tracking tensor |
| S3 visibility_count | Visible iterations in last ~50-iteration window | meta["radii"] > 0 |
| S4 screen_radius_mean | Mean screen-space radius (pixels) over recent window | meta["radii"].max() |
| S5 opacity | sigmoid(opacity) | model.opacity |
| S6 scale_norm | ‖exp(scales)‖ | model.scales |
| S7 age | current_iter - creation_iter | Tracking tensor |

### Future Utility Targets
| Target | Definition | Measurement |
|--------|-----------|-------------|
| U1 future_gradient | Σ‖g_i(τ)‖ for τ ∈ [t+1, t+Δ] | Per-iteration gradient norm |
| U2 future_update | Σ‖Δθ_i(τ)‖ for τ ∈ [t+1, t+Δ] | Per-iteration parameter delta (xyz, opacity, scale, rot, SH) |
| U3 future_render | Visibility count, mean screen radius, projected area, tiles | meta["radii"], meta["tiles_per_gauss"] |

### Gaussian Tracking
Persistent IDs maintained across topology changes (densification + pruning). ID→index mapping updated after each event. 100K sampled Gaussians tracked through future windows.

---

## 2. Core Results — Signal Ranking

### Table 1: Mean Pearson Correlation (across all scenes, checkpoints, deltas)

| Signal | Future Gradient | Future Update (xyz) | Future Visibility | Overall | Rating |
|--------|----------------|--------------------|--------------------|---------|--------|
| **ema_grad_norm** | **0.559** | 0.134 | 0.175 | 0.289 | **STRONG** |
| **visibility_count** | 0.175 | **0.294** | **0.894** | **0.454** | **STRONG** |
| **opacity** | 0.209 | 0.145 | 0.247 | 0.200 | **STRONG** |
| prev_grad_norm | 0.202 | 0.054 | 0.060 | 0.105 | STRONG |
| screen_radius_mean | 0.023 | 0.043 | 0.015 | 0.027 | WEAK |
| scale_norm | -0.039 | -0.049 | -0.176 | -0.088 | DROP |
| age | -0.303 | -0.110 | -0.392 | -0.268 | DROP |

### Table 2: Recall@10% (fraction of top-10% utility Gaussians in top-10% signal Gaussians)

| Signal | Future Gradient | Future Update | Future Visibility | Random Baseline |
|--------|----------------|---------------|-------------------|-----------------|
| ema_grad_norm | 0.632 | 0.177 | 0.215 | 0.10 |
| visibility_count | 0.211 | 0.181 | 0.689 | 0.10 |
| opacity | 0.288 | 0.140 | 0.231 | 0.10 |
| prev_grad_norm | 0.281 | 0.149 | 0.178 | 0.10 |

### Table 3: Cross-Scene Consistency

| Signal | Room | Garden | Bicycle | Std | All Positive? |
|--------|------|--------|---------|-----|---------------|
| ema_grad_norm | 0.277 | 0.299 | 0.304 | 0.011 | ✅ |
| visibility_count | 0.460 | 0.433 | 0.509 | 0.032 | ✅ |
| opacity | 0.219 | 0.162 | 0.225 | 0.029 | ✅ |
| prev_grad_norm | 0.100 | 0.114 | 0.092 | 0.009 | ✅ |
| screen_radius_mean | 0.027 | 0.022 | 0.019 | 0.003 | ✅ |
| scale_norm | -0.050 | -0.102 | -0.141 | 0.037 | ❌ (all negative) |
| age | -0.205 | -0.297 | -0.324 | 0.051 | ❌ (all negative) |

### Table 4: Age-Stratified Best Signal

| Age Group | Best for Gradient | Pearson | Best for Update | Pearson | Best for Visibility | Pearson |
|-----------|------------------|---------|-----------------|---------|---------------------|---------|
| < 100 (new) | opacity | 0.08 | opacity | -0.01 | opacity | 0.06 |
| 100–500 | ema_grad_norm | 0.59 | visibility_count | 0.24 | visibility_count | 0.82 |
| 500–2000 | ema_grad_norm | 0.64 | visibility_count | 0.31 | visibility_count | 0.81 |
| > 2000 | ema_grad_norm | 0.59 | visibility_count | 0.32 | visibility_count | 0.86 |

*(Values shown are for room; garden and bicycle show the same pattern.)*

---

## 3. Temporal Horizon Analysis

### How correlation changes with prediction horizon Δ

#### Future Gradient
| Signal | Δ=10 | Δ=50 | Δ=100 | Trend |
|--------|------|------|-------|-------|
| ema_grad_norm | 0.44 | 0.60 | 0.64 | ↑ Increases |
| visibility_count | 0.14 | 0.20 | 0.21 | ↑ Increases |
| opacity | 0.16 | 0.23 | 0.24 | ↑ Increases |

#### Future Update (xyz)
| Signal | Δ=10 | Δ=50 | Δ=100 | Trend |
|--------|------|------|-------|-------|
| visibility_count | 0.39 | 0.30 | 0.22 | ↓ Decreases |
| ema_grad_norm | 0.18 | 0.13 | 0.10 | ↓ Decreases |
| opacity | 0.15 | 0.16 | 0.13 | ↓ Decreases |

#### Future Visibility
| Signal | Δ=10 | Δ=50 | Δ=100 | Trend |
|--------|------|------|-------|-------|
| visibility_count | 0.86 | 0.93 | 0.95 | ↑↑ Strongly increases |
| opacity | 0.24 | 0.26 | 0.26 | → Stable |
| ema_grad_norm | 0.17 | 0.19 | 0.19 | → Stable |

*(Values are means across room, garden, bicycle)*

**Key finding**: visibility_count's prediction of future visibility INCREASES with horizon (0.86→0.95 at Δ=10→100). This means visibility is a stable, long-range predictor of rendering contribution. For future update, all signals decay with horizon, but visibility remains the best at all horizons.

---

## 4. Densification-Aware Subanalysis

### Signal Lift for Predicting Densification Outcomes (Room)

| Signal | Clone Lift | Split Lift | Prune Lift |
|--------|-----------|-----------|-----------|
| ema_grad_norm | **36.3×** | **22.6×** | 0.18× |
| prev_grad_norm | 7.8× | **37.0×** | 2.7× |
| opacity | 1.2× | 4.7× | **103.6×** |
| visibility_count | 2.1× | 0.15× | 5.2× |
| scale_norm | 0.0× | 0.15× | 3.2× |

*(Lift = recall / random_baseline. 1.0× = no better than random.)*

### Interpretation

1. **Gradient predicts clone/split (lift 22-37×)**: This is BY DESIGN — the densification rule uses gradient threshold. Gradient tells you WHO will be densified.

2. **Opacity predicts pruning (lift 104×)**: Also BY DESIGN — pruning uses opacity threshold. Low opacity → pruned.

3. **Visibility does NOT predict clone/split (lift 0.15-2.1×)**: Visibility is not used by the densification mechanism.

4. **This explains C52's failure**: C52 used gradient to rank densification candidates, but gradient just predicts the densification DECISION (which is gradient-based), not the densification UTILITY (which is better predicted by visibility). Gradient-based allocation is circular: it selects Gaussians that the gradient rule would have selected anyway.

---

## 5. Optimization-Aware vs Rendering-Aware Utility

### The Critical Finding: Two Distinct Utility Dimensions

| Utility Dimension | Best Signal | Pearson | What It Represents |
|-------------------|------------|---------|-------------------|
| **Future Gradient** | ema_grad_norm | 0.56 | Optimization importance (where loss gradient flows) |
| **Future Update** | visibility_count | 0.29 | Actual parameter changes (where optimization happens) |
| **Future Visibility** | visibility_count | 0.89 | Rendering contribution (which Gaussians are computed) |

**Gradient predicts gradient, but visibility predicts actual computation.** The future gradient (where the loss gradient flows) and future rendering/update (where computation is actually spent) are DIFFERENT utility dimensions, predicted by DIFFERENT signals.

This maps to the spec's **Outcome D**: optimization importance and representation-growth importance are two different variables. The C52 failure occurred because gradient importance was used to predict densification allocation utility, but gradient only predicts the gradient-based densification decision, not the rendering/computational utility of the densified Gaussians.

---

## 6. Answers to Research Questions

### 1. What observable signal best predicts future Gaussian utility?

**visibility_count**. It has the highest overall mean Pearson (0.454), is cross-scene consistent (std=0.032), and its predictive power INCREASES with temporal horizon. It best predicts future rendering contribution (Pearson 0.89) and future parameter updates (Pearson 0.29).

### 2. Is gradient actually the best predictor?

**No, not overall.** EMA gradient is the best predictor of FUTURE GRADIENT (Pearson 0.56), but it is weak for future rendering (0.17) and moderate for future updates (0.13). For overall computational utility (rendering + updates), visibility_count is superior.

### 3. Does visibility predict future utility better than gradient?

**Yes, for rendering and update utility.** visibility_count Pearson: 0.89 (rendering), 0.29 (update). ema_grad_norm Pearson: 0.17 (rendering), 0.13 (update). The advantage is consistent across all 3 scenes.

### 4. Does screen-space footprint carry stronger information?

**No.** screen_radius_mean is WEAK (Pearson 0.02-0.04). The raw screen-space radius does not carry meaningful predictive information. This is surprising — larger screen footprint does not correlate with future utility.

### 5. Which signal best predicts future optimization updates?

**visibility_count** (Pearson 0.29 for future xyz update). This is intuitive: Gaussians that are frequently visible receive more gradient signal and thus more parameter updates. The EMA gradient (0.13) is weaker because it measures WHERE gradient flows, not HOW MUCH actual parameter change occurs.

### 6. Which signal best predicts future densification?

**Gradient signals (ema_grad_norm, prev_grad_norm)** with lift 22-37× for clone/split prediction. But this is tautological — the densification rule IS gradient-based. Gradient predicts the densification DECISION, not the densification UTILITY.

### 7. Are these two utilities fundamentally different?

**Yes.** Future gradient (optimization importance) is best predicted by ema_grad_norm (Pearson 0.56). Future rendering/update (computational utility) is best predicted by visibility_count (Pearson 0.89/0.29). These are different signals predicting different dimensions. This is the most important finding of C53.

### 8. Does the best signal remain stable across room/garden/bicycle?

**Yes.** visibility_count: Pearson 0.43-0.51 (std=0.032). ema_grad_norm: Pearson 0.28-0.30 (std=0.011). Both are remarkably consistent across 3 diverse scenes (indoor room, outdoor garden, outdoor bicycle).

### 9. How long into the future does the signal remain predictive?

- **visibility_count → future visibility**: Pearson 0.86 (Δ=10) → 0.95 (Δ=100). INCREASES with horizon. Extremely long-range predictor.
- **ema_grad_norm → future gradient**: Pearson 0.44 (Δ=10) → 0.64 (Δ=100). INCREASES with horizon. Long-range predictor.
- **visibility_count → future update**: Pearson 0.39 (Δ=10) → 0.22 (Δ=100). DECREASES but remains best signal.
- **All signals → future update decay with horizon**, but visibility_count remains the best at all horizons.

### 10. Does Gaussian age change the predictor ranking?

**Yes, dramatically.**
- **New Gaussians (age < 100)**: ALL signals are weak (Pearson < 0.12). Future utility of newly created Gaussians is essentially unpredictable from observable signals. opacity is marginally best.
- **Mature Gaussians (age > 100)**: ema_grad_norm best predicts future gradient (0.50-0.64), visibility_count best predicts future update/visibility (0.19-0.33 / 0.82-0.98). The ranking is STABLE across all mature age groups.

This suggests any selective computation strategy should NOT skip new Gaussians — their utility is unpredictable, so they should all be computed.

### 11. Is there a strong enough signal to justify a new algorithm?

**Yes.** visibility_count is a strong, cross-scene, long-range predictor of future rendering contribution (Pearson 0.89, increasing with horizon) and future parameter updates (Pearson 0.29). It is cheap to obtain (already computed by the rasterizer). This justifies investigating visibility-aware selective computation.

### 12. If no signal is sufficiently predictive, should the direction be stopped?

**Not applicable — a strong signal was found.** visibility_count exceeds all gate criteria. However, the finding is nuanced: visibility predicts rendering/update utility, not gradient utility. An algorithm should use visibility for selective COMPUTATION (which Gaussians to render/update), not for selective DENSIFICATION (which was C52's failed approach).

---

## 7. Decision

### **KEEP → Method Design (Visibility-Aware Selective Computation)**

#### Gate Evaluation

| Gate | Criterion | Result |
|------|-----------|--------|
| A | Signal clearly outperforms gradient | ✅ visibility_count: 0.89 vs 0.17 (rendering), 0.29 vs 0.13 (update) |
| B | Advantage in ≥2/3 scenes | ✅ Consistent across all 3 scenes (std=0.032) |
| C | Clear temporal horizon | ✅ Pearson 0.86→0.95 (Δ=10→100), INCREASES with horizon |
| D | Low cost to obtain | ✅ Already computed by rasterizer (meta["radii"]) |

#### What Was Discovered

1. **Two distinct utility dimensions**: Future gradient (optimization importance) and future rendering/update (computational utility) are predicted by different signals. Gradient → gradient, visibility → rendering/update.

2. **visibility_count is the strongest overall signal**: Pearson 0.89 for future visibility, 0.29 for future update, cross-scene consistent, temporal horizon INCREASES with Δ.

3. **C52's failure is explained**: C52 used gradient to predict densification allocation utility, but gradient only predicts the densification decision (tautological), not the rendering/computational utility of densified Gaussians.

4. **New Gaussians are unpredictable**: All signals fail for age < 100. Any selective computation must include all new Gaussians.

5. **age and scale are negative predictors**: Older and larger Gaussians have less future utility. This is a free signal (already available) that could complement visibility.

#### What Stage 1 Should Address

- **Visibility-aware selective computation**: Skip rendering/backward for Gaussians with low visibility_count. Since visibility predicts future rendering contribution (0.89) and future updates (0.29), low-visibility Gaussians contribute less to both rendering and optimization.
- **Age-gated strategy**: Always compute new Gaussians (age < 100), use visibility for selective computation of mature Gaussians.
- **NOT visibility-aware densification**: Visibility does NOT predict densification (lift 0.15-2.1×). Densification should remain gradient-based.

---

## 8. Outcome Classification

This result matches **Outcome B** from the spec:

> visibility >> gradient, cross-scene stable → consider visibility-aware selective computation

AND partially **Outcome D**:

> gradient strong for future update but weak for future densification → optimization importance and representation-growth importance are two different variables

The full picture:
- Gradient is strong for future GRADIENT (0.56), not future UPDATE (0.13)
- Visibility is strong for future VISIBILITY (0.89) and future UPDATE (0.29)
- These are genuinely different utility dimensions

---

## 9. Limitations

1. **Visibility window is ~50 iterations** (not 100): The offset-50 reset gives a 50-iteration visibility window at checkpoints, not 100. This is shorter than ideal but sufficient for signal measurement.

2. **Screen-space radius is per-camera**: The screen_radius_mean is averaged over a recent window of training cameras, not over all cameras. A Gaussian visible in one camera may be invisible in another. The signal is camera-dependent.

3. **No LPIPS**: LPIPS not available. Quality is not measured in this discovery phase.

4. **Densification outcomes are sparse**: Most sampled Gaussians have outcome=0 (unchanged). Clone/split/prune events are rare in a 100-iteration window. The densification lift values are averaged over few events and may be noisy.

5. **No feature combinations tested**: By design, only original signals are tested. Combinations (e.g., visibility × age) might be stronger but are out of scope.

6. **Control seeds confirm trends**: room_seed123 and garden_seed123 show the same signal ranking as seed=42, confirming reproducibility.

7. **Bicycle has 6M+ initial Gaussians**: Memory-intensive (22GB GPU). Results are consistent with room/garden despite the scale difference.

8. **Future update decay**: All signals' prediction of future update DECAYS with horizon. The best prediction is at Δ=10 (short horizon). This limits the practical use of selective computation to short-range prediction.

---

## 10. Deliverables

| Deliverable | Status |
|-------------|--------|
| `reports/phase-c53-discovery.md` | This file |
| `results/a100/phase-c53-discovery/room_signals.json` | ✅ |
| `results/a100/phase-c53-discovery/garden_signals.json` | ✅ |
| `results/a100/phase-c53-discovery/bicycle_signals.json` | ✅ |
| `results/a100/phase-c53-discovery/future_gradient.json` | ✅ |
| `results/a100/phase-c53-discovery/future_update.json` | ✅ |
| `results/a100/phase-c53-discovery/future_render_proxy.json` | ✅ |
| `results/a100/phase-c53-discovery/densification_analysis.json` | ✅ |
| `results/a100/phase-c53-discovery/temporal_horizon.json` | ✅ |
| `results/a100/phase-c53-discovery/age_stratification.json` | ✅ |
| `results/a100/phase-c53-discovery/cross_scene_comparison.json` | ✅ |
| `results/a100/phase-c53-discovery/final_signal_ranking.json` | ✅ |
| `results/a100/phase-c53-discovery/room_raw.npz` | ✅ (91 MB) |
| `results/a100/phase-c53-discovery/garden_raw.npz` | ✅ (90 MB) |
| `results/a100/phase-c53-discovery/bicycle_raw.npz` | ✅ (89 MB) |
| `results/a100/phase-c53-discovery/room_seed123_raw.npz` | ✅ (91 MB) |
| `results/a100/phase-c53-discovery/garden_seed123_raw.npz` | ✅ (91 MB) |
| `scripts/phase-c53-discovery/collect_signals.py` | ✅ |
| `scripts/phase-c53-discovery/collect_future_utility.py` | ✅ |
| `scripts/phase-c53-discovery/analyze_predictability.py` | ✅ |
| `scripts/phase-c53-discovery/analyze_densification.py` | ✅ |
| `scripts/phase-c53-discovery/analyze_age.py` | ✅ |
| `scripts/phase-c53-discovery/analyze_temporal.py` | ✅ |
| `scripts/phase-c53-discovery/create_final_comparison.py` | ✅ |

---

## 11. Research Discipline

### Observed (directly measured)
- 7 signals at 8 checkpoints for 100K Gaussians across 3 scenes + 2 controls
- 3 future utility targets at Δ=10/50/100
- Densification outcomes (clone/split/prune/unchanged)
- Gaussian age and genealogy tracking

### Derived (computed from observations)
- Pearson/Spearman correlations between signals and utilities
- Recall@K and Coverage@K
- Temporal horizon trends
- Age-stratified predictor rankings
- Densification prediction lift

### Interpretation
- visibility_count is the strongest predictor of future computational utility
- Gradient and visibility predict different utility dimensions
- C52's failure is explained by the tautological nature of gradient-based densification prediction
- New Gaussians are unpredictable — any selective strategy must compute all new Gaussians

### Hypothesis (for future investigation)
- Visibility-aware selective computation can reduce rendering/backward cost without proportional quality loss
- Age-gated strategy (compute all new, selectively compute mature) could be effective
- The negative correlation of age/scale with utility suggests a "Gaussian maturation" effect
- Combination of visibility + age might be stronger than either alone (not tested in this phase)
