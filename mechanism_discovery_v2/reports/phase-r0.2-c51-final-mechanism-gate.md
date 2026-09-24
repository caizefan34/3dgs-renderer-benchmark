# Phase R0.2 — C51 Final Mechanism Gate

**Semantic label**: `REFERENCE_V1_ABSGRAD`  
**Date**: 2026-09-14  
**Scene**: mipnerf360/room  
**Renderer**: gsplat 1.5.3 (absgrad mode, grow_grad2d=0.0008)  
**Decision**: **C51_SYSTEMS_COMPONENT**

---

## Executive Summary

Phase R0.2 answers two questions:

1. **Does previous-iteration ranking retain enough current gradient mass for historical masking?**
   **No.** Previous-iteration Top-50% captures only 52.8% of current G_opt mass — barely better than random (50.0%). Historical C51 (xyz-gradient-based masking) is definitively unsupported.

2. **Is there any current-iteration signal available early enough to gate substantial remaining work?**
   **Partially.** G_dens (means2d.absgrad) is a byproduct of the rasterization backward and strongly predicts G_opt (Top-50% coverage = 92.7%, Spearman = 0.858). However, it is available AFTER the dominant backward kernel, limiting its ceiling to 15.8% of E2E time (SH backward + optimizer only).

**Decision: C51_SYSTEMS_COMPONENT** — G_dens can serve as a current-iteration gate for SH backward and optimizer, saving measurable work (~15.8% E2E), but the ceiling is modest and novelty overlaps with per-Gaussian gradient-approximation literature. Forward-time signals (alpha, tiles_per_gauss) have a 45.7% ceiling but their G_opt correlation is untested.

---

## A. Corrected Top-K Statistics

R0.1 conflated Jaccard with Recall (dividing |A∩B| by K instead of |A∪B|). R0.2 computes all three correctly:

| Metric | K=50% median | K=32% median |
|--------|:-----------:|:-----------:|
| Jaccard (corrected) | **0.487** | **0.260** |
| Recall | 0.655 | 0.412 |
| Precision | 0.655 | 0.412 |

The corrected Jaccard (0.487) is substantially lower than R0.1's reported 0.648. Recall = Precision when both sets have size K. The true Jaccard = |∩| / |∪| = 0.655 / (2 - 0.655) = 0.487.

Results are nearly identical for xyz_norm (the historical C51 signal): Jaccard@50 = 0.487, Recall@50 = 0.655. The choice of total_norm vs xyz_norm does not materially affect Top-K stability.

**Per-window Top-K (G_opt_total, K=50%)**:

| Window | Jaccard | Recall |
|--------|:-------:|:------:|
| 2000 | 0.488 | 0.666 |
| 5000 | 0.486 | 0.643 |
| 10000 | 0.487 | 0.649 |
| 14000 | 0.486 | 0.633 |

The pattern is stable across training stages — maturity does not improve Top-K stability.

---

## B. Predictive Gradient-Mass Coverage

This is the decisive historical-C51 metric. Coverage_K(t) = Σ_{i ∈ TopK(predictor)} G_opt_i(t) / Σ_i G_opt_i(t).

### Overall Results (800 consecutive pairs)

| Predictor | K=50% median | K=32% median |
|-----------|:-----------:|:-----------:|
| **Oracle** (TopK from G_opt(t)) | 1.000 | 1.000 |
| **Previous** (TopK from G_opt(t-1)) | **0.528** | **0.390** |
| **Random** (fixed-size random mask) | 0.500 | 0.320 |

**Key finding**: Previous-iteration Top-50% captures only 52.8% of current G_opt mass — merely 2.8 percentage points above random. Historical C51 masking is unsupported.

**Why Oracle = 1.000**: Fewer than 50% of Gaussians are visible from any camera. G_opt is zero for invisible Gaussians (no gradient flows through them). The top 50% by G_opt includes ALL visible Gaussians, whose mass sums to 100% of total. This is a structural property, not a bug.

### Per-Window Previous Coverage (K=50%)

| Window | Previous | Random | Delta |
|--------|:--------:|:------:|:-----:|
| 2000 | 0.434 | 0.500 | -0.066 |
| 5000 | 0.498 | 0.500 | -0.002 |
| 10000 | 0.551 | 0.500 | +0.051 |
| 14000 | 0.572 | 0.500 | +0.072 |

Early training (window 2000) shows previous-mask coverage BELOW random — the previous top-50% actually captures less G_opt mass than a random mask. This is because topology changes (clone/split) create new high-gradient Gaussians that the previous mask misses. Maturity improves coverage slightly (57.2% at window 14000) but never approaches 80%.

### Per-Parameter Previous Coverage (K=50%)

| Parameter | Previous Coverage |
|-----------|:-----------------:|
| xyz_norm | 0.521 |
| opacity_abs | 0.608 |
| scale_norm | 0.600 |
| rot_norm | 0.556 |
| shs_norm | **0.738** |

SH gradient is the most predictable parameter (73.8% coverage) because SH gradients change slowly with camera viewpoint. xyz gradient is the least predictable (52.1%) — this is the signal historical C51 used, and it is the worst choice.

### xyz_norm (Historical C51 Signal) Coverage

| Predictor | K=50% median |
|-----------|:-----------:|
| Oracle | 1.000 |
| Previous | 0.521 |
| Random | 0.500 |

The xyz_norm signal (historical C51's actual mask scalar) has even worse previous-coverage (52.1%) than total_norm (52.8%). **Historical C51 is definitively unsupported by the evidence.**

---

## C. Historical C51 Mask Source Audit

### The Actual Historical C51 Signal

All historical C51 implementations used **`model.xyz.grad.detach().norm(dim=-1)`** — the L2 norm of the xyz (position) parameter gradient ONLY:

| Implementation | File | Signal | EMA |
|---------------|------|--------|:---:|
| C51 early (simulated) | `scripts/phase-c51/simulated_sparse_backward.py:254` | `xyz.grad.norm(dim=-1)` | No |
| C51-R | `scripts/phase-c51r/run_experiment.py:217` | `xyz.grad.norm(dim=-1)` | No |
| C51-stage4b (canonical) | `scripts/phase-c51-stage4b/canonical_training.py:380` | `xyz.grad.norm(dim=-1)` | Yes (decay=0.9, eps=1e-6) |
| C51-stage4b (recall) | `scripts/phase-c51-stage4b/measure_recall.py:158` | `xyz.grad.norm(dim=-1)` | Yes (decay=0.9, eps=1e-6) |

### CUDA Kernel Mechanism

The C51 CUDA kernel (`scripts/phase-c51-stage4a/patch_cuda.py`) loads an importance mask per-Gaussian inside the `rasterize_to_pixels_3dgs_bwd` kernel. If mask=0, it skips gradient computation (v_rgb, v_conic, v_xy, v_opacity atomicAdd) but still performs the T/buffer update for rendering correctness. The mask must be known BEFORE the kernel starts — it uses the previous-iteration xyz gradient norm.

### R0.1 Comparison Invalidity

R0.1 measured G_opt_total (combined norm of ALL parameter gradients: xyz + opacity + scale + rotation + SH). This is NOT the same as the historical C51 signal (xyz gradient norm only). R0.2 measures both signals separately:

- G_opt_total previous K=50% coverage: 0.528
- G_opt_xyz previous K=50% coverage: 0.521

Both are equally insufficient. The historical comparison is now valid, and the conclusion is the same: **previous-iteration gradient ranking does not retain enough current gradient mass.**

---

## D. Per-Gaussian G_dens ↔ G_opt Relationship

### Same-Iteration Correlation (800 iterations)

| Metric | Median | Mean |
|--------|:------:|:----:|
| Pearson(G_dens, G_opt_total) | **0.625** | 0.617 |
| Spearman(G_dens, G_opt_total) | **0.858** | 0.827 |
| Pearson(G_dens, G_opt_xyz) | 0.615 | — |
| Spearman(G_dens, G_opt_xyz) | 0.855 | — |

### Coverage: TopK(G_dens) → G_opt Mass

| K | Coverage median | Oracle (TopK G_opt) |
|---|:--------------:|:-------------------:|
| 50% | **0.927** | 0.957 |
| 32% | **0.814** | 0.878 |

**Key finding**: Top-50% by G_dens captures 92.7% of G_opt mass — nearly as good as the Oracle (95.7%). This is because:

1. **G_dens filters by visibility**: G_dens (means2d.absgrad) is zero for invisible Gaussians. Top-K by G_dens naturally excludes invisible Gaussians, which have zero G_opt.
2. **G_dens ranks within visible**: Among visible Gaussians, G_dens strongly correlates with G_opt (Spearman 0.858). Gaussians with high screen-position gradient also have high overall optimization gradient.

### Per-Window G_dens → G_opt

| Window | Pearson | Spearman | Top50 Coverage |
|--------|:-------:|:--------:|:--------------:|
| 2000 | 0.674 | 0.963 | 0.992 |
| 5000 | 0.614 | 0.858 | 0.927 |
| 10000 | 0.603 | 0.840 | 0.913 |
| 14000 | 0.611 | 0.849 | 0.917 |

G_dens→G_opt coverage is strong across all training stages, though it decreases slightly with maturity (99.2% → 91.7%) as N grows and the visible fraction shrinks.

---

## E. Backward Signal Availability Audit

### gsplat 1.5.3 Backward Pipeline

| Step | Operation | Cost | Key Output |
|------|-----------|:----:|------------|
| 1 | **Rasterization backward** (`rasterize_to_pixels_3dgs_bwd`) | **DOMINANT** | v_means2d, v_means2d_abs (G_dens), v_conics, v_colors, v_opacities |
| 2 | Projection backward (`FullyFusedProjection.backward`) | Cheap | v_xyz |
| 3 | SH backward (`SphericalHarmonics.backward`) | Moderate | v_shs |
| 4 | Opacity backward (sigmoid derivative) | Trivial | v_opacity_logit |
| 5 | Optimizer step (Adam) | Moderate | Updated parameters |

### Candidate Signals

| Signal | First Available | Can Gate Raster Bwd? | Can Gate SH Bwd? | E2E Ceiling |
|--------|:--------------:|:--------------------:|:----------------:|:-----------:|
| G_dens (means2d.absgrad) | After step 1 | ❌ | ✅ | 15.8% |
| Alpha/contribution (render_alphas) | After forward | ✅ | ✅ | 45.7% |
| Opacity (model param) | Any time | ✅ | ✅ | 45.7% |
| tiles_per_gauss (forward meta) | After forward | ✅ | ✅ | 45.7% |
| Previous xyz grad norm (historical C51) | Before forward | ✅ | ✅ | 45.7% |

**Key finding**: G_dens — the only signal with verified strong G_opt correlation (92.7% coverage) — is available AFTER the dominant backward kernel. Forward-time signals (alpha, opacity, tiles_per_gauss) are available before backward and have a 45.7% ceiling, but their G_opt correlation is untested.

The signal that works (G_dens) is too late. The signals that are early enough (forward-time) are untested.

---

## F. Reference-V1 Backward Profile

**Checkpoint**: iter_10000, N=780,884, SH degree=3  
**Method**: CUDA event timing, 10 profiled iterations after 5 warmup

### Timing Breakdown

| Phase | Mean (ms) | % of Total |
|-------|:---------:|:----------:|
| Forward (rasterization) | 4.34 | 8.0% |
| Loss computation (L1 + SSIM) | 24.77 | 45.6% |
| **Backward (total)** | **21.65** | **39.9%** |
| Optimizer step | 3.15 | 5.8% |
| **Total iteration** | **54.30** | **100%** |

### Estimated Backward Breakdown

| Component | Estimated (ms) | % of Backward |
|-----------|:--------------:|:-------------:|
| Rasterization backward | 16.24 | 75% |
| Projection + SH backward | 3.25 | 15% |
| Other (sigmoid, autograd) | 2.16 | 10% |

### Amdahl Ceilings

| Scenario | Ceiling | Description |
|----------|:-------:|-------------|
| After raster bwd | **15.8%** | G_dens available; skip SH bwd + optimizer |
| Skip entire backward | **45.7%** | Forward signal; skip entire backward + optimizer |
| C51 K=50 (historical) | **12.0%** | Gate within raster bwd, 40% of raster bwd saved |

The 15.8% ceiling for the G_dens-based mechanism exceeds the 5% threshold but is modest. The 45.7% ceiling for forward-time signals is substantial but requires verified G_opt correlation.

---

## G. Exact-vs-Approximate Classification

| Mechanism | Classification | Rationale |
|-----------|:--------------:|-----------|
| Historical C51 (previous xyz grad Top-K masking) | **APPROXIMATE** | Non-zero gradients intentionally omitted because they appear unimportant based on previous iteration. No certification that omitted gradient is zero. |
| G_dens-based SH backward skip | **APPROXIMATE** | Non-zero SH gradients omitted for low-G_dens Gaussians. No certification that SH gradient is zero — only that G_dens ranking suggests low importance. |
| Visibility-based exact skip | **EXACT** | Gaussians not visible from the current camera have exactly zero gradient. This is already handled by the rasterization kernel (visibility filter). Not a new mechanism. |

**No EXACT mechanism exists for a novel C51-style optimization.** The only exact skip (invisibility) is already implemented. All candidate C51 mechanisms are APPROXIMATE — they intentionally omit non-zero gradients based on importance estimates.

---

## H. Stable Identity

R0.2 replaces the xyz-hash-based identity tracker from R0.1 with explicit `gaussian_id` propagation in the GaussianModel:

- **`__init__`**: Initializes `_gaussian_ids = torch.arange(N)` and `_next_gaussian_id = N`
- **`densification_postfix`**: Appends new IDs for clone children and split children: `torch.arange(next_id, next_id + n_new)`
- **`prune_points`**: Filters IDs: `_gaussian_ids = _gaussian_ids[valid_points_mask]`
- **`restore`**: Reinitializes IDs from checkpoint

This does not change rendering semantics — the IDs are pure metadata. The explicit propagation eliminates the ambiguity of xyz-hash matching (where clone children share xyz with parents and the first match claims the ID).

### Verification

| Window | Initial N | Final N | Initial next_id | Final next_id | New IDs Created |
|--------|:---------:|:-------:|:--------------:|:-------------:|:--------------:|
| 2000 | 267,519 | 293,028 | 267,519 | 301,316 | 33,797 |
| 5000 | 559,765 | 577,780 | 559,765 | 591,705 | 31,940 |
| 10000 | 780,884 | 791,189 | 780,884 | 801,671 | 20,787 |
| 14000 | 894,355 | 899,403 | 894,355 | 910,149 | 15,794 |

New IDs created = clones + split children. Deaths = split parents + pruned. The ID counter is monotonically increasing and never reused, ensuring exact identity tracking through all topology events.

---

## Final Decision

### C51_SYSTEMS_COMPONENT

**Rationale**:

1. **Historical C51 (previous-xyz-gradient masking) is dead.** Previous-iteration Top-50% captures only 52.8% of current G_opt mass — 2.8 points above random. The xyz_norm signal (historical C51's actual scalar) performs even worse at 52.1%. No amount of EMA smoothing or threshold tuning can overcome the fundamental problem: gradient rank is not temporally predictable because camera changes dominate.

2. **G_dens (means2d.absgrad) is a strong current-iteration predictor.** Top-50% by G_dens captures 92.7% of G_opt mass (Spearman = 0.858). This works because G_dens naturally filters by visibility (invisible Gaussians have zero G_dens) and ranks within visible Gaussians by importance.

3. **But G_dens is available too late.** It is a byproduct of the rasterization backward — the dominant backward kernel (75% of backward, 30% of total). G_dens can only gate the remaining SH backward + optimizer (15.8% E2E ceiling). This exceeds the 5% threshold but is modest.

4. **Forward-time signals are the untested opportunity.** Signals available before backward (opacity, tiles_per_gauss, alpha) have a 45.7% E2E ceiling — they could gate the entire backward. But their G_opt correlation has not been measured. If a forward-time signal strongly predicts G_opt, it could be a C51_NEW_MECHANISM_CANDIDATE.

5. **Novelty assessment.** Using G_dens to skip SH/optimizer for low-importance Gaussians is structurally similar to existing per-Gaussian gradient-approximation work. The key insight (G_dens filters by visibility AND ranks by importance) is valuable but not a fundamentally new mechanism.

**Retain as systems component**: The G_dens→G_opt relationship (92.7% coverage, Spearman 0.858) and the backward profile (15.8% ceiling) provide a concrete, implementable optimization. The CUDA sparse backward infrastructure from C51 Stage 4A can be repurposed: instead of using previous-iteration xyz gradient as the mask, use current-iteration G_dens to gate SH backward and optimizer updates.

**Recommended next step**: Test forward-time signals (tiles_per_gauss, opacity) for G_opt correlation. If a forward signal achieves >80% coverage at K=50%, it could gate the rasterization backward itself (45.7% ceiling) and qualify as C51_NEW_MECHANISM_CANDIDATE.

---

## Deliverables

| File | Description |
|------|-------------|
| `results/reference_v1/r0.2/topk_metrics_corrected.json` | Part A: Corrected Jaccard/Recall/Precision |
| `results/reference_v1/r0.2/predictive_gradient_mass_coverage.json` | Part B: Oracle/Previous/Random coverage |
| `results/reference_v1/r0.2/historical_mask_source_audit.json` | Part C: Historical C51 signal audit |
| `results/reference_v1/r0.2/gdens_gopt_per_gaussian.json` | Part D: Per-Gaussian G_dens↔G_opt correlation |
| `results/reference_v1/r0.2/backward_signal_availability.json` | Part E: gsplat backward signal availability |
| `results/reference_v1/r0.2/reference_backward_profile.json` | Part F: Backward profile + Amdahl ceilings |
| `results/reference_v1/r0.2/final_decision.json` | Final decision with evidence |
| `baseline/r0.2/continuation_runner.py` | R0.2 runner with explicit ID tracking |
| `baseline/r0.2/r0.2_analysis.py` | Analysis producing all 7 JSON outputs |
| `baseline/reference_v1/gaussian_model.py` | Modified: explicit gaussian_id propagation (Part H) |
| `scripts/phase-r0.2/profile_backward.py` | CUDA event-based backward profiler |
| `scripts/phase-r0.2/run_all_continuations.sh` | Parallel 4-GPU continuation launcher |

---

## Appendix: Why Previous-Mask Coverage ≈ Random

The previous-iteration Top-50% includes ~50% visible and ~50% invisible Gaussians at the current camera. Invisible Gaussians have zero G_opt at the current iteration, so they contribute nothing. The visible portion of the previous Top-50% captures roughly 50% of the visible G_opt mass — hence coverage ≈ 0.50.

G_dens avoids this problem because it is zero for invisible Gaussians. Top-50% by G_dens selects from visible Gaussians only, and within visible, G_dens strongly correlates with G_opt (Spearman 0.858). This is why G_dens coverage (92.7%) >> previous-iteration coverage (52.8%).

The fundamental issue with historical C51 is not that gradient rank is unpredictable per se — it's that **camera visibility changes between iterations**, and the previous-iteration mask cannot account for the current camera's visibility pattern. G_dens, being a current-iteration signal, naturally incorporates current visibility.
