# Phase R0.3 — Forward-Time Gate Discovery

**Semantic label**: `REFERENCE_V1_ABSGRAD`  
**Date**: 2026-09-14  
**Scene**: mipnerf360/room  
**Renderer**: gsplat 1.5.3 (absgrad mode, grow_grad2d=0.0008, patched with importance_mask)  
**Decision**: **C51_SYSTEMS_OPTIMIZATION**

---

## Executive Summary

Phase R0.3 tests whether any naturally-available forward-time signal (tiles_per_gauss, projected_radius, opacity) can predict CURRENT optimization-gradient importance beyond trivial visibility filtering.

**Key findings**:

1. **Visible fraction is ~27.6%** (median 30.3%) — only about 1 in 4 Gaussians is visible from any camera. All previous "all-Gaussian" coverage measurements were dominated by this visibility confound.

2. **Among visible Gaussians, forward signals are weak predictors of G_opt**:
   - tiles_per_gauss: Spearman = 0.155 (nearly zero)
   - projected_radius: Spearman = 0.142 (nearly zero)  
   - opacity: Spearman = 0.425 (moderate but insufficient)

3. **Opacity is the best signal** but still falls short: Top-50%(opacity|visible) captures 77.9% of G_opt mass (vs Oracle 95.7%, Random 50.0%). It retains 77.9% gradient mass while removing 45.8% of backward work — a measurable but imperfect Pareto tradeoff.

4. **Masked backward scaling is sublinear**: f(K=50%) = 0.824 — masking 50% of visible Gaussians saves only 17.6% of backward time, not 50%. The T/buffer update must still process all Gaussians. Estimated E2E saving at K=50%: ~7.0%.

5. **The visibility confound is enormous**: tiles_per_gauss and projected_radius show 100% all-Gaussian coverage at K=50% (because they perfectly exclude invisible Gaussians), but only 57% and 59% visible-conditional coverage. Opacity has minimal confound (80.6% → 77.9%, gain=0.027).

**Decision: C51_SYSTEMS_OPTIMIZATION** — opacity provides a measurable speed opportunity (retaining 77.9% gradient mass while removing 45.8% of work, ~7% E2E saving) but is a conventional importance-based approximate skip. It does not meet the ≥90% gradient-mass retention threshold for C51_MECHANISM_CANDIDATE, and opacity-based importance skipping is well-established in the literature.

---

## 1. Visibility Confound Removed

### Visible Fraction Per Iteration

| Window | Mean | Median |
|--------|:----:|:------:|
| 2000 | 0.327 | 0.367 |
| 5000 | 0.255 | 0.292 |
| 10000 | 0.264 | 0.296 |
| 14000 | 0.257 | 0.286 |
| **Overall** | **0.276** | **0.303** |

Only ~27.6% of Gaussians are visible from any given camera. All primary ranking experiments operate ONLY on visible Gaussians (i ∈ V_t).

---

## 2-3. Target and Candidate Signals

**Target**: G_opt_total (primary), per-parameter G_opt (secondary: xyz, opacity, scale, rotation, SH)

**Candidate forward signals** (all FREE, naturally available):

| Signal | Source | Availability | Cost |
|--------|--------|:------------:|:----:|
| tiles_per_gauss | meta["tiles_per_gauss"] from forward rasterization | After forward, before backward | FREE |
| projected_radius | max(meta["radii"], dim=-1) from forward projection | After forward, before backward | FREE |
| opacity | model.get_opacity (sigmoid of learned parameter) | Before forward (model state) | FREE |

No invented features, no learned predictors, no EMA, no combinations.

---

## 4. Visible-Conditional Correlations

### Forward Signal vs G_opt_total (among visible Gaussians only)

| Signal | Pearson (median) | Spearman (median) |
|--------|:----------------:|:-----------------:|
| tiles_per_gauss | 0.060 | 0.155 |
| projected_radius | 0.065 | 0.142 |
| **opacity** | **0.203** | **0.425** |

### Opacity vs Per-Parameter G_opt

| Parameter | Pearson | Spearman |
|-----------|:-------:|:--------:|
| total_norm | 0.203 | 0.425 |
| xyz_norm | 0.202 | 0.429 |
| opacity_abs | -0.014 | 0.116 |
| scale_norm | 0.076 | 0.379 |
| rot_norm | 0.151 | 0.416 |
| shs_norm | 0.000 | -0.131 |

**Key finding**: Opacity correlates moderately with xyz, scale, and rotation gradients (Spearman 0.38-0.43) but NOT with opacity gradient itself (Spearman 0.12) or SH gradient (Spearman -0.13). This makes sense: high-opacity Gaussians contribute more to the rendered image, generating larger position/scale/rotation gradients. But their opacity gradient depends on whether they're under/over-estimated, not their absolute opacity.

tiles_per_gauss and projected_radius have near-zero correlation with G_opt among visible Gaussians. Their apparent predictive power in all-Gaussian measurements was entirely the visibility confound.

---

## 5. Visible-Conditional Gradient-Mass Coverage

### Required Table

| Signal | K | Coverage | Oracle | Random |
|--------|--:|:--------:|:------:|:------:|
| tiles_per_gauss | 10% | 0.129 | 0.620 | 0.100 |
| tiles_per_gauss | 20% | 0.247 | 0.782 | 0.200 |
| tiles_per_gauss | 32% | 0.386 | 0.884 | 0.320 |
| tiles_per_gauss | 50% | 0.572 | 0.958 | 0.500 |
| projected_radius | 10% | 0.140 | 0.620 | 0.100 |
| projected_radius | 20% | 0.269 | 0.782 | 0.200 |
| projected_radius | 32% | 0.413 | 0.884 | 0.320 |
| projected_radius | 50% | 0.594 | 0.958 | 0.500 |
| **opacity** | **10%** | **0.248** | **0.620** | **0.100** |
| **opacity** | **20%** | **0.426** | **0.782** | **0.200** |
| **opacity** | **32%** | **0.591** | **0.884** | **0.320** |
| **opacity** | **50%** | **0.779** | **0.958** | **0.500** |

**Key finding**: tiles_per_gauss and projected_radius barely beat random among visible Gaussians (57.2% and 59.4% vs 50.0% random at K=50%). Opacity is the only signal with meaningful predictive power (77.9% at K=50%), but it still falls well short of the Oracle (95.8%) and the 90% threshold for C51_MECHANISM_CANDIDATE.

---

## 6. Marginal Value Beyond Visibility

### Visibility Confound Decomposition

| Signal | K | All-Gaussian Coverage | Visible-Conditional Coverage | Visibility Gain |
|--------|--:|:--------------------:|:---------------------------:|:---------------:|
| tiles_per_gauss | 50% | 1.000 | 0.572 | **0.428** |
| tiles_per_gauss | 32% | 1.000 | 0.386 | **0.614** |
| projected_radius | 50% | 1.000 | 0.594 | **0.406** |
| projected_radius | 32% | 1.000 | 0.413 | **0.587** |
| opacity | 50% | 0.807 | 0.779 | **0.027** |
| opacity | 32% | 0.634 | 0.591 | **0.043** |

**Critical finding**: tiles_per_gauss and projected_radius show 100% all-Gaussian coverage at K=50% — this looks impressive until you realize it's ENTIRELY the visibility confound. These signals are zero for invisible Gaussians, so Top-50% by these signals perfectly excludes all invisible Gaussians (which have zero G_opt). Once you condition on visibility, their coverage drops to barely above random.

Opacity has minimal visibility confound (gain = 0.027 at K=50%). This is because opacity is non-zero for all Gaussians (it's a model parameter), so it doesn't naturally filter by visibility. Its predictive power is genuine, not confounded.

**Conclusion**: tiles_per_gauss and projected_radius should NOT be described as predictive — their benefit disappears within V. Only opacity has genuine within-visible predictive power.

---

## 7-8. Work-Gradient Pareto

### Required Pareto Table

| Signal | Visible K | Gradient Mass Retained | Work Retained | Work Removed |
|--------|----------:|:---------------------:|:-------------:|:------------:|
| tiles_per_gauss | 10% | 0.129 | 0.500 | 0.500 |
| tiles_per_gauss | 20% | 0.247 | 0.651 | 0.349 |
| tiles_per_gauss | 32% | 0.386 | 0.764 | 0.236 |
| tiles_per_gauss | 50% | 0.572 | 0.875 | 0.125 |
| projected_radius | 10% | 0.140 | 0.455 | 0.545 |
| projected_radius | 20% | 0.269 | 0.608 | 0.392 |
| projected_radius | 32% | 0.413 | 0.728 | 0.273 |
| projected_radius | 50% | 0.594 | 0.849 | 0.151 |
| **opacity** | **10%** | **0.248** | **0.081** | **0.919** |
| **opacity** | **20%** | **0.426** | **0.184** | **0.816** |
| **opacity** | **32%** | **0.591** | **0.322** | **0.678** |
| **opacity** | **50%** | **0.779** | **0.542** | **0.458** |

**The interesting regime** (high gradient mass + low work retained):

- **Opacity K=50%**: Retains 77.9% gradient mass while removing 45.8% of work. This is the best Pareto point — meaningful work removal with substantial gradient retention.
- **Opacity K=32%**: Retains 59.1% gradient mass while removing 67.8% of work. More aggressive but loses too much gradient.
- **tiles_per_gauss K=50%**: Retains 57.2% gradient mass but removes only 12.5% of work. Poor Pareto — barely any work savings because high-tile Gaussians are also high-work Gaussians.

tiles_per_gauss has a fundamental Pareto problem: the Gaussians with the most tiles ARE the ones doing the most work. Selecting high-tile Gaussians retains almost all the work (87.5% at K=50%) while only capturing 57.2% of gradient mass. This is worse than random.

Opacity has a favorable Pareto because high-opacity Gaussians do more rendering work per pixel but not necessarily more tile-Gaussian incidences. Low-opacity Gaussians contribute less to the image but may still touch many tiles.

---

## 9. Forward Availability Cost

| Signal | Cost | Description |
|--------|:----:|-------------|
| tiles_per_gauss | FREE | Produced by forward rasterization as meta output |
| projected_radius | FREE | Derived from meta["radii"], a forward projection output |
| opacity | FREE | Model parameter, available at any time without computation |

All three signals are FREE — no additional forward reduction or computation is needed. This satisfies criterion 1 for C51_MECHANISM_CANDIDATE.

---

## 10. Correct Raster-Backward Ceiling

### Masked Backward Scaling (deterministic random masks, N=780,884)

| Retention K | Backward Time (ms) | f(K) | Saving |
|:-----------:|:-----------------:|:----:|:------:|
| 100% (all visible) | 22.48 | 1.000 | 0.0% |
| 75% | 20.38 | 0.907 | 9.3% |
| 50% | 18.52 | 0.824 | 17.6% |
| 32% | 17.30 | 0.769 | 23.1% |
| No mask (reference) | 22.24 | — | — |

### Scaling Function f(K)

The masked backward scaling is **sublinear**: f(K=50%) = 0.824, not 0.50. Masking 50% of visible Gaussians saves only 17.6% of backward time because:

1. The T/buffer update must still process ALL visible Gaussians (for rendering correctness)
2. Only the gradient computation (v_rgb, v_conic, v_xy, v_opacity atomicAdd) is skipped for masked Gaussians
3. The kernel's per-pixel loop still iterates over all intersecting Gaussians

### Realistic E2E Ceiling

From R0.2 backward profile: backward = 39.9% of total iteration time.

- At K=50% (opacity): E2E saving = (1 - 0.824) × 39.9% = **7.0%**
- At K=32% (opacity): E2E saving = (1 - 0.769) × 39.9% = **9.2%**
- At K=75% (opacity): E2E saving = (1 - 0.907) × 39.9% = **3.7%**

The 7.0% E2E saving at K=50% exceeds the 5% threshold, but only modestly. The sublinear scaling significantly limits the achievable speedup.

---

## 11. Novelty Warning

### Comparison with Existing Literature

| Prior Work | Concept | Overlap with Opacity-Based Skip |
|-----------|---------|:------------------------------:|
| **PUP 3D-GS** (sensitivity pruning) | Uses opacity-based importance for PRUNING (removing Gaussians permanently) | **High** — same signal (opacity), different operation (prune vs skip backward) |
| **MaskGaussian** (masked rasterization) | Masks Gaussians in forward rendering for speed | **Moderate** — masking concept, but for forward not backward |
| **GETA-3DGS** (render-aware saliency) | Uses rendering contribution for Gaussian importance | **Moderate** — contribution-based importance, related to opacity |
| **Faster-GS** (gradient approximation) | Approximates per-Gaussian backward | **Low** — different mechanism (approximation vs skipping) |
| **Historical C51** (previous-gradient masking) | Uses xyz gradient norm for backward masking | **Low** — different signal (historical gradient vs current opacity) |

### Structural Distinctness Assessment

Opacity-based backward skipping is **NOT structurally distinct** from existing work:
- PUP 3D-GS already uses opacity as an importance signal for permanent pruning
- The concept of "skip low-importance Gaussians in backward" is a natural extension of any importance-based pruning to the backward pass
- The key insight (high-opacity Gaussians dominate gradient mass) is expected from the rendering equation: contribution ∝ opacity × T × color

**Verdict**: Opacity-based selective backward does not appear structurally distinct enough to justify external novelty review. It is a conventional importance-based approximate skip applied to the backward pass.

---

## 12. Decision

### C51_SYSTEMS_OPTIMIZATION

**Rationale**:

Opacity is the best forward-time signal tested, but it does NOT qualify as C51_MECHANISM_CANDIDATE:

| Criterion | Required | Opacity Result | Pass? |
|-----------|----------|:--------------:|:-----:|
| 1. Available before backward at negligible cost | Yes | FREE (model parameter) | ✅ |
| 2. Visible-conditioned prediction strong | Spearman > 0.5 | 0.425 | ❌ |
| 3. Retains ≥90% G_opt mass at K=50% | ≥ 0.90 | 0.779 | ❌ |
| 4. Eliminates substantial backward work | ≥ 20% removed | 45.8% removed | ✅ |
| 5. E2E saving > 5% | > 5% | ~7.0% | ✅ |
| 6. Structurally distinct from existing work | Yes | No (overlaps PUP 3D-GS) | ❌ |

Opacity fails criteria 2, 3, and 6. It provides a measurable speed opportunity (~7% E2E) but is a conventional importance-based approximate skip with crowded novelty.

tiles_per_gauss and projected_radius fail all criteria — their apparent predictive power was entirely the visibility confound.

**Conclusion**: No free forward signal within the visible set meets the C51_MECHANISM_CANDIDATE bar. The opacity-based mechanism is a viable systems optimization (retains 77.9% gradient mass, removes 45.8% work, ~7% E2E saving) but not a novel mechanism. The existing C51 CUDA sparse backward infrastructure can be repurposed to use opacity as the mask signal instead of previous-iteration xyz gradient.

**This closes the C51 discovery pipeline.** The three-phase investigation (R0.1 → R0.2 → R0.3) has conclusively established:

1. **R0.1**: Gradient concentration is real (Gini 0.75) but temporal predictability is poor (Pearson 0.03)
2. **R0.2**: Previous-iteration masking is unsupported (52.8% coverage ≈ random); G_dens predicts G_opt well (92.7%) but is available too late (15.8% ceiling)
3. **R0.3**: Forward-time signals are weak within visible set; opacity is best (77.9% coverage, Spearman 0.43) but conventional and imperfect; masked backward scaling is sublinear (f(50%)=0.82)

---

## Deliverables

| File | Description |
|------|-------------|
| `results/reference_v1/r0.3/visible_conditional_correlations.json` | Part 4: Per-iteration Pearson/Spearman among visible |
| `results/reference_v1/r0.3/visible_gradient_mass_coverage.json` | Part 5: TopK(signal\|visible) → G_opt coverage |
| `results/reference_v1/r0.3/visibility_confound_decomposition.json` | Part 6: All-Gaussian vs visible-only coverage |
| `results/reference_v1/r0.3/work_gradient_pareto.json` | Part 7+8: Gradient mass vs work retained/removed |
| `results/reference_v1/r0.3/signal_availability_cost.json` | Part 9: Forward signal availability classification |
| `results/reference_v1/r0.3/masked_backward_scaling.json` | Part 10: f(K) measured with deterministic random masks |
| `results/reference_v1/r0.3/final_decision.json` | Part 12: C51_SYSTEMS_OPTIMIZATION decision |
| `baseline/r0.3/continuation_runner.py` | R0.3 runner with visible-conditional metrics |
| `baseline/r0.3/r0.3_analysis.py` | Analysis producing all 7 JSON outputs |
| `scripts/phase-r0.3/benchmark_masked_backward.py` | Masked backward scaling benchmark |
| `scripts/phase-r0.3/run_all_continuations.sh` | Parallel 4-GPU continuation launcher |
| `baseline/reference_v1/gaussian_model.py` | Modified: explicit gaussian_id propagation (from R0.2) |

---

## Appendix A: Why Opacity Works (Within Visible)

The rendering equation for 3DGS is: C_pixel = Σ_i α_i × T_i × c_i, where α_i = opacity × Gaussian_2D_contribution. High-opacity Gaussians contribute more to the rendered image, generating larger L1/SSIM gradients that propagate back through them. This creates a natural correlation: high opacity → high contribution → high gradient.

However, the correlation is imperfect (Spearman 0.43) because:
1. **View angle matters**: A high-opacity Gaussian viewed edge-on contributes little
2. **Occlusion matters**: A high-opacity Gaussian behind another gets T≈0, contributing nothing
3. **Gradient direction matters**: Opacity gradient depends on over/under-estimation, not absolute opacity
4. **SH gradients uncorrelated**: SH gradient (the largest parameter dimension) has zero correlation with opacity (Spearman -0.13)

The 77.9% coverage at K=50% means that 22.1% of gradient mass is in low-opacity Gaussians — these are typically edge Gaussians, newly-cloned Gaussians with low initial opacity, or Gaussians in occluded regions.

---

## Appendix B: The Sublinear Scaling Problem

The masked backward kernel (from C51 Stage 4A) skips gradient computation for masked Gaussians but still performs the T/buffer update for ALL visible Gaussians. This is because:

1. **T (transmittance) is cumulative**: T_i = Π_{j<i} (1 - α_j). Skipping a Gaussian's gradient computation doesn't allow skipping its T update — subsequent Gaussians' gradients depend on T.

2. **The per-pixel loop is over sorted Gaussians**: Each pixel iterates through all intersecting Gaussians in depth order. The loop body has two parts: (a) T/buffer update (always needed) and (b) gradient computation (skippable). Only part (b) is saved.

3. **AtomicAdd is the bottleneck**: The gradient computation ends with atomicAdd to global memory. Skipping this saves the atomic operation but not the memory reads (means2d, conics, colors are still loaded for the T update).

This means f(K) = (fixed_cost + K × variable_cost) / (fixed_cost + variable_cost), where fixed_cost (T update) ≈ 60-70% of the kernel time. Hence f(50%) ≈ 0.82, not 0.50.

This sublinear scaling fundamentally limits the achievable speedup from ANY importance-based backward masking, regardless of prediction quality. Even with a perfect Oracle mask at K=50%, the backward saving would be only 17.6%, giving ~7% E2E.
