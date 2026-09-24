# Phase C53-Validation — Leakage-Free Computational Utility Validation

## 0. Purpose

C53-Discovery found visibility_count → future_visibility Pearson ≈ 0.89–0.95 and concluded visibility is the strongest predictor of future computational utility. However, this result contains a critical methodological issue: **the predictor (visibility_count) is closely related to the target (future_visibility)**. The 0.89–0.95 correlation may be self-prediction / target leakage rather than evidence that visibility predicts actual computational work.

This phase resolves that problem by measuring:

```
current observable signal → future ACTUAL computational work
```

NOT:

```
current observable signal → future version of itself
```

---

## 1. The Critical Finding

### The C53-Discovery result was mostly target leakage.

| Test | Pearson | Interpretation |
|------|---------|---------------|
| visibility_count → future_vis (C53-Discovery headline) | **0.929** | Self-prediction |
| visibility_count → future_tile_work (actual rasterization) | **0.141** | Drops 85% |
| visibility_count → future_pixel_work (actual pixel rendering) | **0.019** | Near zero |
| visibility_count → RESIDUAL tile_work (after removing future_vis) | **-0.000** | **ZERO** |

After regressing future tile work on future visibility and computing the residual, current visibility has **zero** correlation with the remaining work. The 0.141 correlation between current visibility and future tile work is **entirely mediated** by future visibility. Visibility contains no information about future computational work beyond what future visibility already provides.

### A different signal was the true predictor all along.

| Signal | → tile_work (actual) | → vis (leakage target) | Rating |
|--------|---------------------|----------------------|--------|
| **screen_radius_mean** | **0.693** | 0.014 | **STRONG** |
| scale_norm | 0.424 | -0.192 | MODERATE |
| visibility_count | 0.141 | 0.929 | WEAK (leakage) |
| ema_grad_norm | 0.061 | 0.186 | WEAK |
| prev_grad_norm | 0.024 | 0.062 | DROP |
| opacity | 0.013 | 0.255 | DROP |
| age | 0.011 | -0.414 | DROP |

screen_radius_mean was rated **WEAK** in C53-Discovery because it was compared against the wrong target (future_vis = 0.014). When the target is actual computational work (tile intersections), screen_radius_mean is the **strongest predictor by far** (0.693).

---

## 2. Instrumentation Validation

### Does tiles_per_gauss track actual CUDA work?

**Yes.** Measured 500 iterations on room with CUDA event timing:

| Proxy | Correlation with Forward Time | Correlation with Backward Time |
|-------|------------------------------|-------------------------------|
| **total_tiles (tiles_per_gauss sum)** | **0.982** | **0.936** |
| n_visible | 0.873 | 0.863 |
| total_area (projected pixel area) | 0.081 | 0.058 |
| n_gaussians | 0.000 | 0.000 |

tiles_per_gauss has near-perfect correlation with actual forward rendering time (0.982) and backward time (0.936). This validates it as a legitimate computational work proxy.

**Note**: total_area (projected pixel area) has very low correlation with time (0.081). Rendering cost is dominated by tile intersections, not pixel coverage. This means pixel_work (area) is a poor work proxy, while tile_work (tiles) is excellent.

---

## 3. Experimental Design

### Data Source
Reused C53-Discovery raw `.npz` data (5 scenes × 8 checkpoints × 100K Gaussians × 3 deltas). The existing data already contains `tiles_mean` (actual rasterization work) and `area_mean` (pixel work) as future utility targets — these ARE the leakage-free targets needed.

### Leakage-Free Targets
| Target | Key in .npz | What It Measures | Validated? |
|--------|------------|-----------------|-----------|
| T1 future_tile_work | tiles_mean | Tile intersection count (rasterization work) | ✅ Pearson 0.982 with CUDA time |
| T2 future_backward_work | tiles_mean × grad_mean (proxy) | Backward work estimate | ⚠️ Proxy (element-wise product of means) |
| T3 future_pixel_work | area_mean | Projected pixel area | ❌ Low correlation with CUDA time (0.081) |
| T4 future_update | update_xyz | Parameter update magnitude | Secondary target |

### Signals (same 7 as C53-Discovery)
S1 prev_grad_norm, S2 ema_grad_norm, S3 visibility_count, S4 screen_radius_mean, S5 opacity, S6 scale_norm, S7 age.

### Scenes
room, garden, bicycle + controls (room seed=123, garden seed=123).

---

## 4. Required Final Tables

### Table 1 — Leakage-Free Prediction (Pearson, Δ=50, mean across room/garden/bicycle)

| Signal | Future Tile Work | Future Pixel Work | Future Update | Future Grad | Future Vis (LEAK) |
|--------|-----------------|-------------------|--------------|-------------|-------------------|
| **screen_radius_mean** | **0.693** | **0.549** | 0.039 | 0.019 | 0.014 |
| scale_norm | 0.424 | 0.241 | -0.060 | -0.048 | -0.192 |
| visibility_count | 0.141 | 0.019 | 0.303 | 0.201 | **0.929** |
| ema_grad_norm | 0.061 | 0.014 | 0.129 | **0.602** | 0.186 |
| prev_grad_norm | 0.024 | 0.005 | 0.050 | 0.213 | 0.062 |
| opacity | 0.013 | -0.001 | 0.151 | 0.227 | 0.255 |
| age | 0.011 | 0.003 | -0.117 | -0.332 | -0.414 |

### Table 2 — Work Type (Pearson, Δ=50, mean across room/garden/bicycle)

| Signal | Forward (tiles) | Intersection (tiles) | Backward (update) | Backward (grad) |
|--------|----------------|---------------------|-------------------|-----------------|
| **screen_radius_mean** | **0.549** | **0.693** | 0.039 | 0.019 |
| scale_norm | 0.241 | 0.424 | -0.060 | -0.048 |
| visibility_count | 0.019 | 0.141 | **0.303** | 0.201 |
| ema_grad_norm | 0.014 | 0.061 | 0.129 | **0.602** |
| prev_grad_norm | 0.005 | 0.024 | 0.050 | 0.213 |
| opacity | -0.001 | 0.013 | 0.151 | 0.227 |
| age | 0.003 | 0.011 | -0.117 | -0.332 |

### Table 3 — Horizon (Pearson, target=tile_work, mean across room/garden/bicycle)

| Signal | Δ=10 | Δ=50 | Δ=100 | Trend |
|--------|------|------|-------|-------|
| **screen_radius_mean** | **0.631** | **0.693** | **0.698** | ↑ |
| scale_norm | 0.400 | 0.424 | 0.431 | ↑ |
| visibility_count | 0.139 | 0.141 | 0.143 | → |
| ema_grad_norm | 0.059 | 0.061 | 0.062 | → |
| prev_grad_norm | 0.021 | 0.024 | 0.024 | → |
| opacity | 0.011 | 0.013 | 0.013 | → |
| age | 0.011 | 0.011 | 0.012 | → |

### Table 4 — Age (best signal per age group, target=tile_work, Δ=50)

| Age Group | Best Signal | Pearson | Interpretation |
|-----------|------------|---------|---------------|
| < 100 (new) | scale_norm | 0.726 | New GS: scale predicts work (screen_radius not yet accumulated) |
| 100–500 | screen_radius_mean | 0.799 | Mature: screen radius is strongest |
| 500–2000 | screen_radius_mean | 0.777 | Stable across mature groups |
| > 2000 | screen_radius_mean | 0.693 | Slightly lower but still strong |

### Table 5 — Cross-Scene (Pearson, target=tile_work, Δ=50)

| Signal | Room | Garden | Bicycle | Mean | Std |
|--------|------|--------|---------|------|-----|
| **screen_radius_mean** | **0.619** | **0.771** | **0.687** | **0.693** | **0.062** |
| scale_norm | 0.580 | 0.425 | 0.266 | 0.424 | 0.128 |
| visibility_count | 0.112 | 0.138 | 0.172 | 0.141 | 0.024 |
| ema_grad_norm | 0.034 | 0.082 | 0.066 | 0.061 | 0.020 |
| prev_grad_norm | 0.012 | 0.038 | 0.021 | 0.024 | 0.011 |
| opacity | 0.022 | -0.008 | 0.025 | 0.013 | 0.015 |
| age | 0.038 | 0.012 | -0.015 | 0.011 | 0.022 |

### Table 6 — Visibility Leakage Control (mean across room/garden/bicycle, Δ=50)

| Test | Mean | Std | n | Verdict |
|------|------|-----|---|---------|
| current_vis → future_tile_work | 0.141 | 0.108 | 24 | Weak positive |
| future_vis → future_tile_work | 0.152 | 0.110 | 24 | Contemporaneous upper bound |
| **current_vis → RESIDUAL tile_work** | **-0.000** | **0.008** | **24** | **ZERO — full leakage** |
| current_vis → future_update | 0.303 | 0.285 | 24 | Moderate positive |
| current_vis → RESIDUAL update | -0.011 | 0.032 | 24 | Near zero |
| current_vis → future_vis (C53 ORIGINAL) | 0.929 | 0.057 | 24 | The leakage result |

---

## 5. Recall@K (Leakage-Free, target=tile_work, Δ=50)

| Signal | Recall@10 | Recall@20 | Recall@50 | Coverage@20 |
|--------|-----------|-----------|-----------|-------------|
| **screen_radius_mean** | **0.602** | **0.569** | 0.618 | **0.611** |
| scale_norm | 0.350 | 0.374 | 0.508 | 0.503 |
| visibility_count | 0.298 | 0.477 | **0.763** | 0.426 |
| ema_grad_norm | 0.157 | 0.293 | 0.609 | 0.337 |
| opacity | 0.144 | 0.286 | 0.588 | 0.237 |
| prev_grad_norm | 0.144 | 0.268 | 0.547 | 0.314 |
| age | 0.133 | 0.221 | 0.454 | 0.240 |
| *Random baseline* | *0.10* | *0.20* | *0.50* | *0.20* |

screen_radius_mean has the highest Recall@10 (0.602 — 6× random) and Coverage@20 (0.611 — 3× random). visibility_count has higher Recall@50 (0.763) because it identifies many low-work Gaussians correctly, but its Coverage@20 (0.426) is much lower than screen_radius (0.611).

---

## 6. Backward Work Analysis

### Proxy: backward_work = tiles_mean × grad_mean (element-wise product)

| Signal | Pearson | Recall@20 |
|--------|---------|-----------|
| **screen_radius_mean** | **0.338** | 0.297 |
| ema_grad_norm | 0.180 | 0.683 |
| scale_norm | 0.153 | 0.186 |
| visibility_count | 0.045 | 0.423 |
| prev_grad_norm | 0.067 | 0.364 |
| opacity | 0.028 | 0.391 |
| age | -0.024 | 0.123 |

**Note**: This is a PROXY (product of means, not mean of products). True per-iteration backward work would require new instrumentation. However, the pattern is clear: screen_radius_mean also dominates for backward work, followed by ema_grad_norm. visibility_count is weak (0.045).

---

## 7. Answers to Final Questions

### 1. Does visibility predict actual future CUDA/rasterization work?

**Weakly.** visibility_count → future_tile_work Pearson = 0.141. This is positive and cross-scene consistent (std=0.024), but it is WEAK — far below the 0.929 that prompted C53-Discovery's KEEP decision. The instrumentation validation confirms tile_work is a valid work proxy (Pearson 0.982 with CUDA forward time).

### 2. Does visibility predict future backward work?

**Very weakly.** visibility_count → backward_work_proxy = 0.045. Visibility does not predict backward work. EMA gradient (0.180) and screen_radius (0.338) are both better.

### 3. Does visibility outperform EMA gradient for actual work?

**No.** For tile_work: visibility 0.141 vs ema_grad 0.061 — visibility is slightly better. For backward_work: visibility 0.045 vs ema_grad 0.180 — ema_grad is much better. For pixel_work: visibility 0.019 vs ema_grad 0.014 — both near zero. Neither visibility nor EMA gradient is the best predictor of actual work; screen_radius_mean is (0.693).

### 4. Does visibility outperform random ranking?

**Yes, marginally.** visibility_count Recall@10 = 0.298 (vs 0.10 random), Coverage@20 = 0.426 (vs 0.20 random). It is better than random, but far below screen_radius_mean (Recall@10 = 0.602, Coverage@20 = 0.611).

### 5. Does the result survive removing future-visibility leakage?

**No.** After regressing future_tile_work on future_visibility, the residual correlation of current visibility with the remaining work is **-0.000** (zero). The 0.141 correlation is entirely mediated by future visibility. Visibility contains no independent information about future computational work.

### 6. Does current visibility contain information about future-work residuals?

**No.** The residual correlation is -0.000 ± 0.008 across 24 measurements (3 scenes × 8 checkpoints). This is indistinguishable from zero. Current visibility has no predictive information about future tile work beyond what future visibility provides.

### 7. Is the predictor useful at Δ=10, 50, 100?

**For visibility: marginally at all horizons, but not strong at any.** visibility → tile_work: 0.139 (Δ=10), 0.141 (Δ=50), 0.143 (Δ=100) — stable but weak. For screen_radius_mean: 0.631 (Δ=10), 0.693 (Δ=50), 0.698 (Δ=100) — strong and slightly increasing.

### 8. Does Gaussian age change the result?

**Yes, but differently than C53-Discovery found.**
- **New Gaussians (age < 100)**: screen_radius_mean = 0 (no accumulated screen-radius signal yet — new Gaussians haven't been rendered enough to build the signal). scale_norm is the best predictor (0.726), because it directly determines the Gaussian's spatial extent and thus its tile intersections.
- **Mature Gaussians (age ≥ 100)**: screen_radius_mean is consistently best (0.693–0.799). The ranking is stable across all mature age groups.

The "Gaussian maturation" effect from C53-Discovery is confirmed: new Gaussians are unpredictable by screen_radius (which needs rendering history), but their scale (a static property) is an excellent predictor.

### 9. Is the result consistent across room/garden/bicycle?

**Yes.** screen_radius_mean: 0.619 / 0.771 / 0.687 (std=0.062). visibility_count: 0.112 / 0.138 / 0.172 (std=0.024). Both are consistent. Control seeds (room_seed123, garden_seed123) show the same patterns.

### 10. Is there one signal that reliably predicts future computational utility?

**Yes: screen_radius_mean.** It has:
- Pearson 0.693 for tile_work (strongest of all signals)
- Cross-scene std = 0.062 (consistent)
- No leakage (→ future_vis = 0.014, near zero)
- Stable across horizons (0.631 → 0.698)
- Best for all mature age groups (0.693–0.799)
- Recall@10 = 0.602 (6× random)
- Coverage@20 = 0.611 (3× random)

### 11. Is that signal cheap enough to obtain before computation?

**Yes.** screen_radius_mean is already computed by the gsplat rasterizer as `meta["radii"]`. It is available at every forward pass at no additional cost. It requires a ~50-iteration accumulation window (same as visibility_count), but the per-iteration cost is negligible (reading `meta["radii"].max()`).

For new Gaussians (age < 100) where screen_radius hasn't accumulated, `scale_norm` (from `model.scales`) is a direct property available at zero cost and has Pearson 0.726 for tile_work.

### 12. Only if the answer is convincingly YES: should Method Design begin?

**Yes, but with a modified direction.** The C53-Discovery direction (visibility-aware selective computation) is NOT supported. The correct direction is **footprint-aware selective computation**: use screen_radius_mean (for mature Gaussians) and scale_norm (for new Gaussians) to predict which Gaussians will dominate future rasterization work, and selectively skip computation for low-footprint Gaussians.

---

## 8. Decision

### **MODIFY — Footprint-Aware Selective Computation**

#### Gate Evaluation

| Gate | Criterion | Result |
|------|-----------|--------|
| A | visibility predicts actual future work | ✅ 0.141 (weak positive) |
| B | advantage survives removing future-vis leakage | ❌ residual = -0.000 (zero) |
| C | result in ≥2/3 scenes | ✅ std=0.024 |
| D | mature Gaussians show useful prediction | ❌ for visibility; ✅ for screen_radius |
| E | instrumentation overhead controlled | ✅ Pearson 0.982 validates proxy |

**Visibility fails Gate B (leakage removal).** The 0.929 headline result was self-prediction.

**However, screen_radius_mean passes all gates for actual work:**

| Gate | Criterion | screen_radius_mean |
|------|-----------|-------------------|
| A | predicts actual future work | ✅ 0.693 (strong) |
| B | no leakage (→ future_vis = 0.014) | ✅ near zero |
| C | ≥2/3 scenes consistent | ✅ std=0.062 |
| D | mature Gaussians predicted | ✅ 0.693–0.799 |
| E | instrumentation validated | ✅ tile_work proxy 0.982 with CUDA time |

#### What Was Discovered

1. **C53-Discovery's 0.929 was target leakage.** visibility_count predicts future_visibility (itself), not actual computational work. After residual analysis, the correlation with actual work is zero.

2. **screen_radius_mean is the true predictor of rasterization work.** Pearson 0.693 for tile intersections, 0.549 for pixel coverage. It was misrated as WEAK in C53-Discovery because it was compared against the wrong target.

3. **Instrumentation is validated.** tiles_per_gauss has 0.982 correlation with actual CUDA forward time. The work proxy is legitimate.

4. **Scale_norm is the predictor for new Gaussians.** For age < 100, screen_radius hasn't accumulated, but scale_norm (a static property) has Pearson 0.726 for tile_work.

5. **The utility dimensions are different from what C53-Discovery concluded:**
   - Rasterization work: best predicted by screen_radius_mean (0.693)
   - Parameter update: best predicted by visibility_count (0.303)
   - Gradient flow: best predicted by ema_grad_norm (0.602)
   - These are THREE different utilities with THREE different best predictors.

6. **C52's failure is re-explained.** C52 used gradient to predict densification allocation. But gradient predicts densification decisions (tautological), not computational work. The actual computational work is predicted by screen_radius_mean, which was never tested.

#### What Method Design Should Address

- **Footprint-aware selective computation**: Skip rendering/backward for Gaussians with low screen_radius_mean. Since screen_radius predicts future tile intersections (0.693), low-footprint Gaussians contribute less to both rendering and computation.
- **Age-gated strategy**: For new Gaussians (age < 100), use scale_norm instead of screen_radius_mean (which hasn't accumulated yet). scale_norm has Pearson 0.726 for new Gaussians.
- **NOT visibility-aware**: Visibility does not survive leakage removal and should not be the primary signal for selective computation.
- **NOT gradient-aware for computation**: Gradient predicts gradient flow (0.602), not rasterization work (0.061). Gradient is for optimization, not for computational cost prediction.

---

## 9. Outcome Classification

This result matches the spec's **MODIFY** criterion:

> If another signal is better for actual backward work, then define the exact utility domain before algorithm design.

The exact utility domain is:
- **Signal**: screen_radius_mean (mature) + scale_norm (new)
- **Target**: future tile intersection count (rasterization work)
- **Cost**: zero (already computed by rasterizer)
- **Scope**: skip computation for low-footprint Gaussians

---

## 10. Limitations

1. **Backward work is a proxy.** The backward_work_proxy = tiles_mean × grad_mean is a product of means, not a sum of per-iteration products. True per-iteration backward work would require new instrumentation. The proxy pattern (screen_radius > ema_grad > visibility) is consistent but the exact values are approximate.

2. **Pixel work is a poor CUDA time proxy.** Projected area has only 0.081 correlation with CUDA time. Rendering cost is dominated by tile intersections, not pixel coverage. The pixel_work target is retained for comparison but should not be used as a computational cost proxy.

3. **Screen radius requires accumulation.** The screen_radius_mean signal is accumulated over a ~50-iteration window. For new Gaussians (age < 100), this signal is not yet available. scale_norm serves as the fallback for new Gaussians.

4. **No LPIPS.** Quality is not measured. Any future selective computation algorithm must measure rendering quality impact.

5. **Camera dependence.** screen_radius_mean is camera-dependent (a Gaussian's screen radius varies by viewpoint). The signal is an average over recent training cameras, which approximates the expected rendering cost across viewpoints but may not be optimal for specific test views.

6. **Control seeds confirm patterns.** room_seed123 and garden_seed123 show the same signal ranking, confirming reproducibility.

7. **No feature combinations tested.** By design, only original signals are tested. Combinations (e.g., screen_radius × scale_norm) might be stronger but are out of scope.

8. **The residual analysis is exploratory.** The linear regression residual is a simple OLS fit, not a causal model. The zero residual correlation strongly suggests no independent information, but cannot prove causality.

---

## 11. Research Discipline

### Observed (directly measured)
- 7 signals at 8 checkpoints for 100K Gaussians across 5 runs (3 scenes + 2 controls)
- 3 future utility targets (tile_work, pixel_work, update) + 2 secondary (grad, vis)
- tiles_per_gauss correlation with CUDA time: 0.982 (forward), 0.936 (backward)
- 500 iterations of CUDA event-timed rendering for instrumentation validation

### Derived (computed from observations)
- Pearson/Spearman/Recall@K/Coverage@K between signals and leakage-free targets
- Residual analysis: current_vis → residual_tile_work = -0.000
- Time-shifted comparison: predictive (0.141) vs contemporaneous (0.152)
- Age-stratified predictor rankings
- Temporal horizon trends

### Interpretation
- The C53-Discovery result (visibility → future_vis = 0.929) was target leakage
- screen_radius_mean is the true predictor of rasterization computational work
- The direction should shift from visibility-aware to footprint-aware selective computation
- New Gaussians need scale_norm (screen_radius not yet accumulated)

### Hypothesis (for future investigation)
- Footprint-aware selective computation can reduce rendering/backward cost by skipping low-screen-radius Gaussians
- Age-gated strategy (scale_norm for new, screen_radius for mature) could be effective
- The combination of screen_radius_mean + scale_norm might cover all age groups
- The three utility dimensions (rasterization, update, gradient) may require different selective strategies

---

## 12. Deliverables

| Deliverable | Status |
|-------------|--------|
| `reports/phase-c53-validation.md` | This file |
| `results/a100/phase-c53-validation/instrumentation_validation.json` | ✅ |
| `results/a100/phase-c53-validation/future_work_analysis.json` | ✅ |
| `results/a100/phase-c53-validation/leakage_analysis.json` | ✅ |
| `results/a100/phase-c53-validation/backward_work_analysis.json` | ✅ |
| `results/a100/phase-c53-validation/intersection_work_analysis.json` | ✅ |
| `results/a100/phase-c53-validation/age_analysis.json` | ✅ |
| `results/a100/phase-c53-validation/horizon_analysis.json` | ✅ |
| `results/a100/phase-c53-validation/final_comparison.json` | ✅ |
| `scripts/phase-c53-validation/analyze_future_work.py` | ✅ |
| `scripts/phase-c53-validation/analyze_leakage.py` | ✅ |
| `scripts/phase-c53-validation/analyze_backward_work.py` | ✅ |
| `scripts/phase-c53-validation/analyze_intersection.py` | ✅ |
| `scripts/phase-c53-validation/analyze_age.py` | ✅ |
| `scripts/phase-c53-validation/analyze_horizon.py` | ✅ |
| `scripts/phase-c53-validation/create_final_comparison.py` | ✅ |
| `scripts/phase-c53-validation/instrument_work.py` | ✅ |
| `scripts/phase-c53-validation/collect_actual_work.py` | ✅ (for future T2 collection) |
| `scripts/phase-c53-validation/quick_leakage_check.py` | ✅ |

---

## 13. Summary: What Changed from C53-Discovery

| Aspect | C53-Discovery | C53-Validation |
|--------|---------------|----------------|
| **Headline result** | visibility → future_vis = 0.929 | visibility → actual work = 0.141 (leakage) |
| **Best predictor of work** | visibility_count (WRONG) | screen_radius_mean (0.693) |
| **Leakage check** | Not performed | Residual = -0.000 (confirmed leakage) |
| **Instrumentation** | Not validated | tiles_per_gauss ↔ CUDA time = 0.982 |
| **Decision** | KEEP (visibility-aware) | MODIFY (footprint-aware) |
| **New Gaussian predictor** | opacity (0.08, wrong target) | scale_norm (0.726, correct target) |
| **Direction** | Visibility-aware selective computation | Footprint-aware selective computation |

The most important rule from the spec is satisfied:

> The old C53 result (visibility → future visibility = 0.89) is NOT used as evidence for computational utility. The target for this stage is current observable → future actual work, not current observable → future version of itself.

The old 0.89 number was not used. The leakage-free analysis independently confirmed that screen_radius_mean, not visibility_count, is the predictor of actual computational work.
