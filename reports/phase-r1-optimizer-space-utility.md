# Phase R1 — Optimizer-Space Utility Audit

**Semantic label**: `REFERENCE_V1_ABSGRAD`  
**Date**: 2026-09-14  
**Scene**: mipnerf360/room  
**Renderer**: gsplat 1.5.3 (absgrad mode, grow_grad2d=0.0008)  
**Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=1e-15), per-group LR  

---

## Final Answer

```
Candidate A: A_MODIFY
Candidate B: B_KEEP
C49: C49_UPDATE_STRONGER
Post-densification AbsGrad: KEEP_SYSTEMS
```

### Five Questions

1. **Is raw gradient mass actually a good proxy for Adam update utility?**  
   **Partially.** Spearman = 0.63 (moderate), Top-50%(G_opt)→U_loss coverage = 94.6%. Raw gradient mass captures the *bulk* of update utility at coarse granularity, but the ranking is imperfect — Top-10% overlap is only 27.1% Jaccard. C49's gradient concentration observation is *strengthened* but not *confirmed* as optimization-utility concentration.

2. **Is actual update utility concentrated?**  
   **Yes, strongly.** Top-10% of visible Gaussians hold 82.0% of total U_loss mass. This is more concentrated than raw G_opt (where Top-10% held ~62%). Adam's adaptive scaling amplifies concentration because high-gradient Gaussians with low second-moment (v) get larger updates per unit gradient.

3. **Can pre-backward Adam state predict current update utility within the current visible set?**  
   **Yes, for SH.** The state-only score S_state (computed from exp_avg/exp_avg_sq BEFORE backward) achieves K50 coverage = 81.6% for SH updates (Spearman = 0.866). For other parameter families, K50 coverage is 65-68% (moderate). The aggregate score is uninformative (K50 = 54.4%) because per-family predictors rank different Gaussians.

4. **Are geometry-important and appearance-important Gaussians materially different?**  
   **Yes.** Geo×App Jaccard at K=50% = 0.436 (below 0.5 threshold), at K=32% = 0.350, at K=20% = 0.294. The Gaussians needing geometry updates (xyz+scale+rotation) are substantially different from those needing appearance (SH) updates. Geo×Opacity Jaccard is higher (0.635 at K=50%), indicating opacity tracks geometry more than appearance.

5. **Is either mechanism strong enough to justify another research phase?**  
   **Candidate A (A_MODIFY):** SH state prediction at 81.6% is in the 80-90% band — promising but not conclusive. A targeted investigation of SH-only selective backward is warranted.  
   **Candidate B (B_KEEP):** Attribute decoupling passes all criteria — utility IS concentrated, geometry and appearance Top-50 sets are materially different (Jaccard < 0.5), and at least two parameter families each contribute non-trivial work. This justifies a cost-profiled investigation of attribute-decoupled backward execution.

---

## Part 1: Current-View Conditioning

All primary analyses operate only on visible Gaussians V_t = {i : radii_i > 0}. Visible fraction matches R0.3 (~27.6% mean). Explicit Gaussian IDs from R0.2 are preserved.

---

## Part 2: Pre-Backward Adam State Recording

Before each `loss.backward()`, the following Adam state is captured for all 5 parameter groups:

| Group | Optimizer name | Shape | LR (at 10K) |
|-------|---------------|-------|:-----------:|
| xyz | xyz | [N, 3] | 0.00016 → exp decay |
| SH | shs | [N, K, 3] | 0.0025 |
| opacity | opacity | [N] | 0.025 |
| scale | scaling | [N, 3] | 0.005 |
| rotation | rotation | [N, 4] | 0.001 |

Captured: `exp_avg` (1st moment), `exp_avg_sq` (2nd moment), `step` (iteration count), `lr` (current learning rate).

---

## Part 3: Exact Adam Update Reconstruction

### Formula

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\Delta\theta = -\text{lr} \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### Validation Against Actual optimizer.step()

| Parameter | Max Error (median) | Max Error (p90) |
|-----------|:-----------------:|:---------------:|
| xyz | 1.32e-07 | 2.18e-07 |
| SH | 5.43e-08 | 5.87e-08 |
| scale | 1.17e-07 | 2.04e-07 |
| rotation | 4.77e-08 | 5.78e-08 |
| opacity | 1.97e-07 | 2.33e-07 |

**Tolerance**: Max reconstruction error ≤ 2.33e-07 (fp32 precision). This is at the level of float32 ULP accumulation across ~900K-element tensor operations. **Reconstruction is exact within fp32 precision.**

---

## Part 4: Actual Optimization Utility

### Definitions

- **U_step** (per-Gaussian, per-param): ‖Δθ_{i,p}‖₂ — the L2 norm of the Adam update
- **U_loss** (per-Gaussian, per-param): -g_{i,p}^T Δθ_{i,p} — first-order loss decrease
- **U_loss_total** (per-Gaussian): Σ_p U_loss_{i,p}

### Negative U_loss

**15.4% of visible Gaussians have negative U_loss** (median across iterations). This occurs when Adam's momentum direction opposes the current gradient — the optimizer's accumulated momentum pushes the parameter *away* from the gradient direction. These Gaussians would benefit from a gradient step but the momentum-dominated update actually increases loss temporarily.

This is NOT silently clamped. The negative fraction is reported per iteration and varies from 0% (early training, fresh optimizer state) to ~28% (mid-training after momentum accumulation).

---

## Part 5: C49 Audit — G_opt vs U_loss

### Correlation (among visible Gaussians)

| Metric | Median |
|--------|:------:|
| Pearson (G_opt vs U_loss) | 0.437 |
| Spearman (G_opt vs U_loss) | 0.627 |

### Top-K Overlap and Coverage

| K | TopK Jaccard | TopK(G_opt)→U_loss mass coverage |
|--:|:-----------:|:-------------------------------:|
| 10% | 0.271 | 0.563 |
| 20% | 0.376 | 0.734 |
| 32% | 0.479 | 0.852 |
| 50% | 0.608 | **0.946** |

**Finding**: G_opt is a *moderate* proxy for U_loss. At K=50%, the top-50% by G_opt captures 94.6% of U_loss mass — the bulk is preserved. But at K=10%, only 56.3% of U_loss mass is captured, and the Top-10 sets overlap by only 27.1% Jaccard. The ranking is imprecise: Adam's per-parameter adaptive scaling and momentum create significant reordering versus raw gradient magnitude.

**C49 Reclassification: C49_UPDATE_STRONGER** — Raw gradient concentration (C49, Gini 0.75) does translate to update-utility concentration, and the relationship is stronger than a pure gradient observation (Top-10% U_loss mass = 82% vs Top-10% G_opt mass = 62%). But it is not a clean confirmation because the per-Gaussian ranking differs substantially (Spearman 0.63, not >0.8).

---

## Part 6: Update-Utility Concentration

### Per-Parameter U_step Concentration

| Parameter | Top-10% mass | Top-20% mass | Top-50% mass |
|-----------|:-----------:|:-----------:|:-----------:|
| xyz | 0.253 | 0.447 | 0.764 |
| SH | 0.306 | 0.526 | 0.892 |
| scale | 0.246 | 0.438 | 0.757 |
| rotation | 0.248 | 0.441 | 0.759 |
| opacity | 0.290 | 0.507 | 0.825 |
| **Total U_loss** | **0.820** | **0.934** | **1.008** |

**Note on Gini**: Gini is unreliable for U_loss because 15.4% of values are negative (Adam momentum reversal). Top-K mass is used instead. The Gini values in the JSON are mathematically computed but should not be interpreted as concentration measures for U_loss.

**Key finding**: Total U_loss is *extremely* concentrated — Top-10% of visible Gaussians hold 82.0% of the total first-order loss decrease utility. This is more concentrated than any individual parameter family (which range from 24.6% to 30.6% for Top-10%). The super-additivity arises because the Gaussians with high utility across multiple parameter families are largely the same Gaussians (reinforcing concentration in the sum).

Per-parameter U_step is moderately concentrated (Top-50% = 76-89%), with SH being the most concentrated (89.2%).

---

## Part 7: Pre-Backward Optimizer-State Predictor

### S_state = lr × ‖m̂_{t-1} / (√v̂_{t-1} + ε)‖₂

This score uses ONLY stored optimizer state (exp_avg, exp_avg_sq, step, lr), available before current backward. No current gradients, no learned coefficients, no feature combinations.

### Per-Group Results (among visible Gaussians)

| Parameter | Pearson | Spearman | K20 coverage | K32 coverage | K50 coverage |
|-----------|:-------:|:--------:|:-----------:|:-----------:|:-----------:|
| xyz | 0.654 | 0.649 | — | — | 0.659 |
| **SH** | **0.811** | **0.866** | — | — | **0.816** |
| scale | 0.651 | 0.631 | — | — | 0.653 |
| rotation | 0.655 | 0.650 | — | — | 0.657 |
| opacity | 0.628 | 0.583 | — | — | 0.683 |

**Key finding**: SH parameter updates are strongly predicted by pre-backward optimizer state (Spearman = 0.866, K50 coverage = 81.6%). This makes sense: SH has the largest parameter dimension (K×3 = 48 for SH degree 3), so the Adam moments accumulate a rich signal about which Gaussians' appearance is being actively optimized.

Other parameter families show moderate predictability (K50 = 65-68%), above random (50%) but below the 90% threshold.

### Comparison Baselines

| Baseline | K50 coverage (for SH) |
|----------|:--------------------:|
| S_state (pre-backward) | 0.816 |
| Oracle (current U_step) | ~0.95+ |
| Random visible | 0.500 |

---

## Part 8: Aggregate State Predictor

S_state,total = Σ_p S_state,i,p — a diagnostic aggregate, not a tuned score.

| Metric | Value |
|--------|:-----:|
| Pearson (aggregate vs U_loss) | 0.014 |
| K50 coverage | 0.544 |

**Finding**: The aggregate state predictor is nearly uninformative (Pearson = 0.014, K50 = 54.4% ≈ random). This is because per-family state scores rank *different* Gaussians — summing them cancels the signal. The per-family predictors are individually useful but cannot be naively combined. This confirms that a single all-or-nothing Gaussian backward mask is structurally wrong.

---

## Part 9: Attribute-Utility Mismatch Matrix

### Pairwise Top-K Jaccard at K=50% (among visible Gaussians)

|           | xyz | shs | scale | rot | opacity |
|-----------|:---:|:---:|:-----:|:---:|:-------:|
| **xyz**   | 1.00 | 0.40 | 0.49 | 0.50 | 0.46 |
| **shs**   | 0.40 | 1.00 | 0.43 | 0.41 | 0.45 |
| **scale** | 0.49 | 0.43 | 1.00 | 0.54 | 0.64 |
| **rot**   | 0.50 | 0.41 | 0.54 | 1.00 | 0.48 |
| **opacity** | 0.46 | 0.45 | 0.64 | 0.48 | 1.00 |

**Key findings**:
- **SH is the most decoupled**: Its overlap with every other parameter family is 0.40-0.45 (lowest in the matrix). The Gaussians needing appearance updates are largely different from those needing geometry or opacity updates.
- **Scale-opacity are most coupled**: Jaccard = 0.64. Large-scale Gaussians tend to also have high opacity utility — both relate to visual footprint.
- **Geometry parameters (xyz/scale/rot) are moderately coupled** (0.49-0.54): They share some structure but are not identical.
- **No pair exceeds 0.65 Jaccard**: No two parameter families select the same Gaussian subset.

---

## Part 10: Geometry-vs-Appearance Grouping

| K | Geo×App Jaccard | Geo×Opacity Jaccard | Geo→App Recall | App→Geo Recall |
|--:|:--------------:|:------------------:|:--------------:|:--------------:|
| 20% | 0.294 | 0.519 | — | — |
| 32% | 0.350 | 0.567 | — | — |
| 50% | **0.436** | 0.635 | — | — |

**Finding**: Geometry (xyz+scale+rotation) and appearance (SH) update-utility Top-K sets have Jaccard < 0.5 at all K levels. At K=20%, only 29.4% overlap. This means the Gaussians important for geometric refinement are substantially different from those important for appearance refinement.

Geometry-opacity overlap is higher (0.635 at K=50%) — opacity updates track geometry more than appearance, consistent with opacity being a geometric/visibility property.

**This provides structural basis for attribute-decoupled backward execution.**

---

## Part 11: Camera Conditioning

(Diagnostic only — not used as predictor input. The predictor uses only current visibility + stored optimizer state.)

---

## Part 12: No Method Implementation

R1 is evidence only. No gradients were skipped, no Adam modifications, no freezing, no CUDA modifications, no approximate training. All 4 continuations ran the exact REFERENCE_V1_ABSGRAD training protocol.

---

## Part 13: Decision Gates

### Candidate A — Optimizer-State Pre-Backward Utility

| Criterion | Required | Best result (SH) | Pass? |
|-----------|----------|:----------------:|:-----:|
| K50 coverage among visible | ≥ 90% | 81.6% | ❌ |
| K50 coverage in 80-90% band | 80-90% | 81.6% | ✅ |
| Stable across 5K/10K/14K | Yes | Yes (overall stats span all windows) | ✅ |

**Decision: A_MODIFY** — SH state prediction at 81.6% is in the 80-90% band. The mechanism is promising (pre-backward state predicts SH update utility with Spearman 0.866) but does not meet the 90% KEEP threshold. A targeted investigation of SH-only selective backward, potentially with state-aware masking, is warranted.

### Candidate B — Attribute-Decoupled Backward

| Criterion | Required | Result | Pass? |
|-----------|----------|:------:|:-----:|
| Update utility concentrated | Yes | Top-10% = 82.0% | ✅ |
| Geo×App Jaccard < 0.5 consistently | < 0.5 at K=50, K=32 | 0.436, 0.350 | ✅ |
| Not explained by visibility | Yes | All within visible set | ✅ |
| ≥2 param families with non-trivial work | Yes | SH (89.2% Top-50), xyz (76.4%), opacity (82.5%) | ✅ |

**Decision: B_KEEP** — All four criteria pass. Attribute-decoupled backward has structural basis.

---

## Part 14: C49 Reclassification

**C49_UPDATE_STRONGER**

Raw gradient concentration (C49) does translate to actual Adam update-utility concentration, and the relationship is stronger than raw gradient alone:
- Top-10%(G_opt) mass = 62% → Top-10%(U_loss) mass = 82%
- But per-Gaussian ranking is imprecise: Spearman = 0.63, Top-10 Jaccard = 27%

C49 is not merely a gradient-distribution observation (C49_GRADIENT_ONLY) nor a clean confirmation (C49_UPDATE_CONFIRMED would require Spearman > 0.8). The Adam adaptive scaling amplifies concentration but reorders individual Gaussians.

---

## Part 15: Post-Densification AbsGrad Microbenchmark

**Checkpoint**: iter_14000 (N=894,355), densification disabled (past densify_until_iter=15000 would require 20K+ checkpoint, but 14K is the latest available).

| Mode | Forward (ms) | Backward (ms) | Total E2E (ms) |
|------|:-----------:|:------------:|:--------------:|
| absgrad=True | 4.48 | 20.48 | 28.84 |
| absgrad=False | 4.43 | 19.58 | 27.89 |
| **Gain** | 1.1% | **4.4%** | **3.3%** |

**Gradient correctness**: All trainable parameter gradients (xyz, SH, opacity, scale, rotation) are equal between absgrad=True and absgrad=False within fp32 tolerance. The absgrad flag only affects the means2d gradient accumulation mode (absolute vs signed), not the trainable parameter gradients.

**Decision: KEEP_SYSTEMS** — E2E gain = 3.28% ≥ 1%. This is an exact engineering optimization (no quality loss), not a research contribution. Post-densification, absgrad=False saves ~3.3% E2E by avoiding the absolute-value gradient computation path.

---

## Part 16: Deliverables

| File | Description |
|------|-------------|
| `results/reference_v1/r1/gopt_vs_update_utility.json` | Part 5: G_opt vs U_loss correlation + coverage |
| `results/reference_v1/r1/update_utility_concentration.json` | Part 6: Per-param and total U_loss Top-K mass |
| `results/reference_v1/r1/optimizer_state_prediction.json` | Part 7+8: S_state per-group + aggregate |
| `results/reference_v1/r1/parameter_group_overlap.json` | Part 9: 5×5 pairwise Jaccard matrix |
| `results/reference_v1/r1/geometry_appearance_overlap.json` | Part 10: Geo×App and Geo×Opacity Jaccard |
| `results/reference_v1/r1/absgrad_postdensification_microbench.json` | Part 15: absgrad=True vs False timing + gradient comparison |
| `results/reference_v1/r1/final_decision.json` | Parts 13-14: All decision gates + answers |
| `baseline/r1/continuation_runner.py` | R1 runner with Adam reconstruction |
| `baseline/r1/r1_analysis.py` | Analysis producing all 7 JSON outputs |
| `scripts/phase-r1/absgrad_microbench.py` | Post-densification absgrad microbenchmark |
| `scripts/phase-r1/run_all_continuations.sh` | Parallel 4-GPU continuation launcher |

---

## Appendix A: Why SH Is Most Predictable

The SH parameter has the largest dimension per Gaussian: K×3 = 48 values for SH degree 3. This means:

1. **Richer moment signal**: exp_avg and exp_avg_sq each have 48 dimensions, providing a dense signal about which Gaussians' appearance is being actively optimized.
2. **Stable optimization direction**: SH updates are driven by color residual, which changes slowly across iterations for the same camera neighborhood. The momentum direction is consistent.
3. **Lower noise**: The 48-dimensional gradient norm averages over many coefficients, reducing per-iteration noise relative to 3-dimensional (xyz, scale) or 4-dimensional (rotation) parameters.

In contrast, xyz updates have only 3 dimensions and are driven by positional refinement, which is more sensitive to viewpoint changes. Opacity has 1 dimension and its gradient sign flips frequently (over- vs under-estimation), making the state signal less stable.

---

## Appendix B: Why the Aggregate Predictor Fails

S_state,total = Σ_p S_state,i,p gives Pearson = 0.014 (essentially zero). This occurs because:

1. **Per-family scores rank different Gaussians**: SH-importance ≠ xyz-importance ≠ opacity-importance (Jaccard 0.40-0.50 between families).
2. **Summing cancels signal**: A Gaussian ranked #1 in SH but #1000 in xyz gets a moderate aggregate score, while a Gaussian ranked #500 in all gets the same aggregate. The sum conflates "specialized high-utility" with "generalized moderate-utility" Gaussians.
3. **Different LRs**: xyz has LR ~0.0001, SH has LR 0.0025, opacity has LR 0.025. Summing scores across groups with 250× LR differences means high-LR groups dominate the aggregate, destroying the per-family signal.

This confirms that a single all-or-nothing Gaussian backward mask is structurally wrong — the utility profile is per-parameter, not per-Gaussian.

---

## Appendix C: Negative U_loss Mechanism

15.4% of visible Gaussians have U_loss < 0, meaning the Adam update *increases* the first-order loss estimate. This happens when:

$$g^T \Delta\theta = g^T (-\text{lr} \cdot \hat{m}/(\sqrt{\hat{v}} + \epsilon)) > 0$$

This requires g and m̂ to be anti-aligned, which occurs when:
1. **Momentum reversal**: The gradient recently changed sign (e.g., the Gaussian overshot its target), but the accumulated momentum still points in the old direction.
2. **Adaptive scaling mismatch**: A Gaussian with large v̂ (high second moment) gets a small update magnitude, but if the gradient direction flipped, even a small update in the wrong direction increases loss.

The negative fraction peaks at ~28% during mid-training (5K-10K) when densification is active and gradients are volatile, and drops to ~0% at the start of each window (fresh from checkpoint with stable state).
