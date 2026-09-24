# Phase R0.1 — Evidence Protocol Correction + C51 Go/No-Go

**Semantic label**: `REFERENCE_V1_ABSGRAD`  
**Git commit**: `32ab80e` (tag: `baseline/reference-v1-absgrad`)  
**Date**: 2026-09-14  
**Scene**: mipnerf360/room  
**Renderer**: gsplat 1.5.3 (absgrad mode, grow_grad2d=0.0008)

---

## Executive Summary

Three evidence-protocol issues from the R0 report have been corrected:

1. **C50 was measured across distant checkpoints** (500→1000→2000...) rather than true consecutive iterations. R0.1 measures lag-1 for every t→t+1 pair across 200-iteration windows.

2. **C49 measured densification gradient (G_dens), not optimization gradient (G_opt)**. R0.1 separates G_dens (means2d.absgrad, used by density control) from G_opt (per-Gaussian parameter gradients from backward, relevant to C51 selective backward).

3. **C53 heavy-tail was measured, but true workload persistence was not recalibrated**. R0.1 measures true W(t)→W(t+1) lag-1 persistence with exact identity tracking.

**C51 Decision: C51_MODIFY**

G_opt concentration is strong (Top-50% mass = 95.4%, Gini = 0.751), but lag-1 predictability is insufficient (Top-50% Jaccard median = 0.648 < 0.80, Pearson median = 0.033). Previous-gradient masking is unsupported. C51 should investigate instantaneous/current-iteration mechanisms only.

---

## 1. Baseline Freeze

| Field | Value |
|-------|-------|
| Git commit | `32ab80e` |
| Tag | `baseline/reference-v1-absgrad` |
| git_dirty | false |
| Semantic label | REFERENCE_V1_ABSGRAD |
| Densification | absgrad=True, grow_grad2d=0.0008 |
| Pinned source | graphdeco-inria/gaussian-splatting@54c035f |

The baseline was renamed from REFERENCE_V1 to REFERENCE_V1_ABSGRAD because densification uses absgrad=True with grow_grad2d=0.0008 (gsplat AbsGS-style adaptation), not literal original-3DGS signed-gradient densification (which uses 0.0002).

No paper experiment may use `--allow_dirty`.

---

## 2. Experimental Protocol

### 2.1 Checkpoint Collection

A 14K-iteration training run (NOT 30K) was executed using the frozen REFERENCE_V1_ABSGRAD code to collect checkpoints at iterations 2000, 5000, 10000, and 14000 — all before densification ends at 15000.

| Checkpoint | N Gaussians | SH Degree |
|-----------|-------------|-----------|
| 2000 | 267,519 | 2 |
| 5000 | 559,765 | 3 |
| 10000 | 780,884 | 3 |
| 14000 | 894,355 | 3 |

### 2.2 Continuation Windows

From each checkpoint, a 200-iteration continuation was run on a separate GPU (4 GPUs in parallel), using the frozen canonical camera sequence from the 30K run. Full backward was used. No C51 masking. Topology behavior was canonical (densification at every 100 iterations, opacity reset at every 3000).

| Window | Iterations | Densification Events | N Start → End |
|--------|-----------|---------------------|---------------|
| 2000 | 2001–2200 | 2 (at 2100, 2200) | 267,519 → 292,984 |
| 5000 | 5001–5200 | 2 (at 5100, 5200) | 559,765 → 577,582 |
| 10000 | 10001–10200 | 2 (at 10100, 10200) | 780,884 → 788,974 |
| 14000 | 14001–14200 | 2 (at 14100, 14200) | 894,355 → 897,513 |

Total: 800 iteration pairs measured (4 windows × 200 iterations - 4 first iterations with no previous).

### 2.3 Two Gradient Signals

**G_dens** — Densification gradient:
- Source: `means2d.absgrad` from gsplat rasterization
- Scaling: pixel-space (`grad *= width/2, height/2`)
- Used by: density control (clone/split decisions)
- This is the signal measured in the R0 report's C49

**G_opt** — Optimization/backward gradient:
- Source: per-Gaussian parameter `.grad` after `loss.backward()`, before `optimizer.step()`
- Components: `||grad_xyz||`, `|grad_opacity|`, `||grad_scale||`, `||grad_rot||`, `||grad_shs||`
- Combined: `G_opt_total = sqrt(sum of squared component norms)`
- Used by: C51 selective backward (determines which Gaussians can be skipped)

These two signals are NEVER merged in the analysis.

### 2.4 Exact Identity Tracking

Stable Gaussian IDs are maintained through clone/split/prune using xyz hash matching:
- Survivor keeps ID (exact xyz match between pre- and post-densification)
- Clone gets new child ID (same xyz as parent but parent claims the match first)
- Split parent dies (removed, no match in post-densification)
- Split children get new IDs (different xyz from any pre-densification Gaussian)
- Pruned ID dies (removed, no match in post-densification)

Only the same surviving Gaussian (matched by stable ID) is compared between t and t+1. Array indices are never compared after topology edits.

---

## 3. C49 — Gradient Concentration (Corrected)

### 3.1 G_opt Concentration

| Window | Gini | Top-1% | Top-10% | Top-32% | Top-50% |
|--------|-----|--------|---------|---------|---------|
| 2000 | 0.836 | — | 0.721 | — | 0.992 |
| 5000 | 0.770 | — | 0.650 | — | 0.959 |
| 10000 | 0.738 | — | 0.603 | — | 0.950 |
| 14000 | 0.745 | — | 0.611 | — | 0.953 |
| **Mature mean** | **0.751** | **0.252** | **0.621** | **0.878** | **0.954** |

### 3.2 G_dens Concentration

| Metric | Mature Mean | Mature Median |
|--------|------------|--------------|
| Gini | 0.660 | 0.647 |
| Top-1% | 0.139 | 0.101 |
| Top-10% | 0.481 | 0.458 |
| Top-32% | 0.817 | 0.811 |
| Top-50% | 0.930 | 0.929 |

### 3.3 G_opt vs G_dens Correlation

| Metric | Pearson |
|--------|---------|
| Gini | 0.904 |
| Top-10% mass | 0.882 |

G_opt and G_dens are highly correlated in their concentration patterns (Pearson 0.90), but G_opt is more concentrated (Gini 0.75 vs 0.66, Top-10% 62% vs 48%). This makes sense: G_opt includes all parameter gradients (xyz, opacity, scale, rotation, SH), while G_dens is only the 2D screen-position gradient.

### Required C49 Table

| Window | Signal | Gini | Top1 Mass | Top10 | Top32 | Top50 |
|--------|--------|-----:|----------:|------:|------:|------:|
| Mature | G_opt | 0.751 | 0.252 | 0.621 | 0.878 | 0.954 |
| Mature | G_dens | 0.660 | 0.139 | 0.481 | 0.817 | 0.930 |

**Central question answer**: Yes, optimization-gradient mass IS concentrated under REFERENCE_V1_ABSGRAD. Top-10% of Gaussians hold 62.1% of G_opt mass, and Top-50% hold 95.4%. Gini = 0.751 confirms strong concentration.

---

## 4. C50 — True Lag-1 Temporal Predictability (Corrected)

### 4.1 G_opt Lag-1 (Overall, 800 pairs)

| Metric | Mean | Median | P10 | P90 |
|--------|------|--------|-----|-----|
| Pearson | 0.075 | 0.033 | -0.012 | 0.242 |
| Spearman | 0.289 | 0.271 | -0.033 | 0.649 |
| Top-1% Jaccard | 0.084 | 0.050 | 0.000 | 0.216 |
| Top-5% Jaccard | 0.160 | 0.131 | 0.001 | 0.356 |
| Top-10% Jaccard | 0.216 | 0.182 | 0.018 | 0.468 |
| Top-20% Jaccard | 0.316 | 0.282 | 0.078 | 0.611 |
| Top-32% Jaccard | 0.444 | 0.415 | 0.217 | 0.731 |
| Top-50% Jaccard | 0.662 | 0.648 | 0.520 | 0.822 |
| Recall@10 | 0.216 | 0.182 | 0.018 | 0.468 |
| Recall@20 | 0.316 | 0.282 | 0.078 | 0.611 |

### 4.2 Per-Window G_opt Lag-1

| Window | Pearson (median) | Top-10% Jaccard (median) | Top-50% Jaccard (median) |
|--------|-----------------|--------------------------|--------------------------|
| 2000 | 0.035 | 0.142 | 0.666 |
| 5000 | 0.037 | 0.217 | 0.642 |
| 10000 | 0.038 | 0.191 | 0.648 |
| 14000 | 0.020 | 0.186 | 0.633 |

Pearson is consistently near-zero across all windows. Top-50% Jaccard is moderate (~0.65) but Top-10% is low (~0.19). The pattern is stable across training stages — maturity does not improve predictability.

### 4.3 Topology-Event Stratification

| Category | Pearson (mean) | Top-10% Jaccard (mean) | N pairs |
|-----------|---------------|------------------------|---------|
| NO_TOPOLOGY_EVENT | 0.076 | 0.216 | 792 |
| AFTER_CLONE_SPLIT_EVENT | 0.040 | 0.114 | 4 |
| AFTER_PRUNE_RESET_EVENT | — | — | 0 |

**Finding**: Topology changes DESTROY gradient predictability. After clone/split events, Pearson drops from 0.076 to 0.040 and Top-10% Jaccard drops from 0.216 to 0.114. However, with only 4 topology-event pairs, this finding has limited statistical power. No opacity-reset events occurred in these windows (resets happen at multiples of 3000, which don't fall within any 200-iter window).

### 4.4 Camera-Conditioned C50

Camera center distance terciles: similar (< 2.518), medium (2.518–7.829), dissimilar (≥ 7.829).

| Category | Pearson (mean) | N pairs |
|-----------|---------------|---------|
| Similar (< 2.518) | 0.154 | 265 |
| Medium (2.518–7.829) | 0.040 | 266 |
| Dissimilar (≥ 7.829) | 0.032 | 265 |

**Finding**: Camera similarity DOES affect gradient predictability, but weakly. Similar cameras show Pearson 0.154 (5x higher than dissimilar at 0.032), but even similar-camera predictability is too low for reliable prediction. This is **view-driven**: the modest predictability comes from view coherence, not Gaussian-intrinsic gradient structure.

---

## 5. C53 — True Lag-1 Workload Persistence (Corrected)

### 5.1 Workload Lag-1 (Overall, 800 pairs)

| Metric | Mean | Median |
|--------|------|--------|
| Pearson | 0.261 | 0.224 |
| Spearman | 0.299 | 0.279 |
| Top-1% Jaccard | 0.222 | 0.182 |
| Top-10% Jaccard | 0.272 | 0.244 |
| Top-50% Jaccard | 0.646 | 0.635 |
| Recall@10 | 0.272 | 0.244 |
| Recall@20 | 0.342 | 0.317 |

### 5.2 Camera-Conditioned Workload

| Category | Pearson (mean) | N pairs |
|-----------|---------------|---------|
| Similar (< 2.518) | 0.391 | 265 |
| Medium (2.518–7.829) | 0.204 | 266 |
| Dissimilar (≥ 7.829) | 0.188 | 265 |

**Finding**: Workload persistence is **view-dependent** (view-driven). Similar cameras show Pearson 0.391 (moderate), while dissimilar cameras drop to 0.188 (weak). The 2x ratio between similar and dissimilar confirms that workload persistence is primarily view coherence, not Gaussian-intrinsic structure.

**C53 classification**: This is **View-driven persistence**, not View-robust persistence. If it were view-robust, dissimilar cameras would still show high persistence. They do not.

---

## 6. G_opt vs G_dens: Are They the Same Signal?

| Metric | G_opt | G_dens | Correlation |
|--------|-------|--------|-------------|
| Gini (mature mean) | 0.751 | 0.660 | 0.904 |
| Top-10% mass | 0.621 | 0.481 | 0.882 |

G_opt and G_dens are highly correlated (Pearson 0.90) but NOT identical. G_opt is more concentrated because it aggregates gradients across all parameters (xyz, opacity, scale, rotation, SH), while G_dens is only the 2D screen-position gradient. The correlation means that Gaussians with high screen-position gradient also tend to have high overall optimization gradient, but the relationship is not perfect.

---

## 7. C49/C50/C53 Phase Naming Corrections

The R0 report incorrectly attributed the value 0.861 to C50 (gradient persistence). This value was from C53-Validation2 (workload persistence), NOT C50 (gradient temporal predictability).

**Corrected historical labels**:
- C49 = gradient concentration
- C50 = gradient temporal predictability
- C53 = computational/workload utility

The R0 report's C50 section stated "Historical comparison: The historical C50 Validation phase found persistence=0.861." This is wrong — 0.861 was C53 workload persistence, not C50 gradient persistence. The R0 report should be corrected.

---

## 8. The 14 Final Questions

### Q1: Is G_opt concentrated under Reference V1?

**Yes.** G_opt Gini = 0.751, Top-10% = 62.1%, Top-50% = 95.4%. The optimization gradient is strongly concentrated — a small fraction of Gaussians account for most of the gradient mass.

### Q2: Is G_dens concentrated?

**Yes, moderately.** G_dens Gini = 0.660, Top-10% = 48.1%, Top-50% = 93.0%. Less concentrated than G_opt but still heavy-tailed.

### Q3: Are G_opt and G_dens actually correlated?

**Yes, highly.** Gini Pearson = 0.904, Top-10% Pearson = 0.882. They measure related but not identical phenomena. G_opt is strictly more concentrated because it includes all parameter gradients.

### Q4: Is G_opt(t) predictive of G_opt(t+1)?

**No, not at the rank level.** Pearson median = 0.033 (near zero). The absolute gradient magnitude has almost no linear correlation between consecutive iterations. Spearman median = 0.271 — slightly better, indicating some rank structure but weak.

### Q5: What is true consecutive-iteration Pearson/Spearman?

- **Pearson**: mean=0.075, median=0.033, p10=-0.012, p90=0.242
- **Spearman**: mean=0.289, median=0.271, p10=-0.033, p90=0.649

The Pearson is near-zero. The Spearman is moderate but inconsistent (p10=-0.033 to p90=0.649). Some iteration pairs show decent rank correlation, but it is not reliable.

### Q6: What is true top-K Jaccard/Recall?

- Top-1% Jaccard: median=0.050 (essentially no overlap)
- Top-10% Jaccard: median=0.182 (18% overlap)
- Top-50% Jaccard: median=0.648 (65% overlap)
- Recall@10: median=0.182
- Recall@20: median=0.282

The top-1% Gaussians by gradient are almost completely replaced between iterations. Even the top-50% only has 65% overlap. For C51's purpose (skipping low-gradient Gaussians), the bottom 50% is somewhat stable, but the top-K (where the decision to keep/skip matters most) is not.

### Q7: Does topology change destroy gradient persistence?

**Yes.** After clone/split events:
- Pearson drops from 0.076 → 0.040 (47% decrease)
- Top-10% Jaccard drops from 0.216 → 0.114 (47% decrease)

However, only 4 topology-event pairs were measured (limited statistical power). No prune-reset events occurred in these windows.

### Q8: Does camera change destroy gradient persistence?

**Partially.** Similar cameras: Pearson=0.154. Dissimilar cameras: Pearson=0.032. Camera change reduces predictability by ~5x, but even similar-camera predictability is too low for reliable prediction.

### Q9: Is tile workload truly persistent from iteration t to t+1?

**Moderately, but view-dependent.** Overall Pearson=0.224, Top-10% Jaccard=0.244. Workload is more persistent than gradient (C50 Pearson=0.033) but still weak. The persistence is primarily view-driven (similar camera Pearson=0.391 vs dissimilar=0.188).

### Q10: Is workload persistence camera-dependent?

**Yes.** Similar-camera Pearson (0.391) is 2.1x higher than dissimilar-camera Pearson (0.188). This is **view-driven persistence**: the workload correlation comes from viewing the same scene from similar viewpoints, not from Gaussian-intrinsic workload structure. If it were view-robust, dissimilar cameras would also show high persistence.

### Q11: Does historical C49 survive for the actual optimization gradient?

**Yes.** G_opt concentration (Gini=0.751, Top-10%=62.1%) confirms that gradient mass is concentrated. Historical C49 (gradient concentration, MEDIUM sensitivity) survives for G_opt. The concentration is even stronger for G_opt than for G_dens.

### Q12: Does historical C50 survive under the correct protocol?

**No.** Historical C50 was measured across distant checkpoints and showed moderate persistence. Under the correct consecutive-iteration protocol:
- Pearson median = 0.033 (was effectively unmeasured before)
- Top-10% Jaccard = 0.182
- Top-50% Jaccard = 0.648

The near-zero Pearson means gradient rank is NOT temporally predictable. The R0 report already found low C50 persistence (Pearson 0.02–0.18) with distant checkpoints; the true consecutive-iteration protocol confirms this with even more data. **C50 does not survive as a useful predictor.**

### Q13: Does C53 persistence survive?

**Partially, as view-driven only.** True lag-1 workload Pearson = 0.224 (overall), but this is driven by camera similarity (0.391 for similar vs 0.188 for dissimilar). The historical C53 value of 0.861 was measured across distant checkpoints and likely inflated by the old implementation's deviations. Under the correct protocol, workload persistence is real but modest and view-dependent. **C53 survives as a view-driven phenomenon, not as Gaussian-intrinsic structure.**

### Q14: Should C51 receive an expensive canonical rerun?

**No.** The evidence gate is:
- Top-50% G_opt mass ≥ 80%: **0.954 ✓** (PASS)
- Top-50% lag-1 recall ≥ 80%: **0.648 ✗** (FAIL)

G_opt concentration is sufficient, but lag-1 predictability is insufficient. C51 as previously formulated (predict-from-history masking) is unsupported by the evidence. An expensive canonical rerun would not change this conclusion.

---

## 9. Final Decision

### C51_MODIFY

**Rationale**: G_opt concentration exists (Top-50% = 95.4%, Gini = 0.751, Top-10% = 62.1%), confirming that a small fraction of Gaussians dominate the optimization gradient. This is promising for selective backward.

However, lag-1 predictability is weak:
- Pearson median = 0.033 (gradient magnitude is essentially uncorrelated between consecutive iterations)
- Top-10% Jaccard = 0.182 (only 18% of the most important Gaussians are shared between t and t+1)
- Top-50% Jaccard = 0.648 (below the 80% gate)
- Topology changes further degrade predictability (Pearson 0.076 → 0.040)
- Camera similarity provides modest improvement (Pearson 0.032 → 0.154) but not enough

**Implication**: Previous-gradient masking (using G_opt(t) to predict G_opt(t+1)) is unsupported. The Gaussian that has the highest gradient at one iteration is unlikely to have the highest gradient at the next.

**Recommended next step**: Investigate only whether an instantaneous/current-iteration exact execution mechanism exists — i.e., can we identify low-gradient Gaussians at the CURRENT iteration without relying on historical prediction? The strong concentration (95.4% in top-50%) suggests that if an instantaneous identification mechanism exists, it could skip ~50% of Gaussians while retaining 95% of gradient information.

**Retain as systems assets**: CUDA sparse backward implementation, gradient correctness verification, kernel microbenchmarks. These have value independent of the C51 masking strategy.

---

## 10. Deliverables

| File | Description |
|------|-------------|
| `results/reference_v1/r0.1/provenance.json` | Experiment provenance (per window) |
| `results/reference_v1/r0.1/c49_gopt_concentration.json` | G_opt concentration per iteration per window |
| `results/reference_v1/r0.1/c49_gdens_concentration.json` | G_dens concentration per iteration per window |
| `results/reference_v1/r0.1/c50_true_lag1_gopt.json` | True lag-1 G_opt metrics (distributions over 800 pairs) |
| `results/reference_v1/r0.1/c50_true_lag1_gdens.json` | True lag-1 G_dens metrics |
| `results/reference_v1/r0.1/c50_topology_stratified.json` | C50 stratified by topology event type |
| `results/reference_v1/r0.1/c50_camera_conditioned.json` | C50 stratified by camera similarity |
| `results/reference_v1/r0.1/c53_true_lag1_workload.json` | True lag-1 workload persistence |
| `results/reference_v1/r0.1/c53_camera_conditioned.json` | C53 stratified by camera similarity |
| `results/reference_v1/r0.1/final_go_no_go.json` | C51 Go/No-Go decision with evidence |
| `results/reference_v1/r0.1/gopt_gdens_correlation.json` | G_opt vs G_dens correlation analysis |
| `baseline/r0.1/continuation_runner.py` | Continuation runner with exact identity tracking |
| `baseline/r0.1/r0.1_analysis.py` | Analysis script producing all JSON outputs |
| `scripts/phase-r0.1/run_all_continuations.sh` | Parallel 4-GPU continuation launcher |

---

## Appendix A: Per-Parameter G_opt Concentration

The G_opt total_norm aggregates gradients across all parameters. Individual parameter concentrations:

| Parameter | What it measures | Relevance to C51 |
|-----------|-----------------|------------------|
| xyz_norm | Position gradient | Directly determines geometry update magnitude |
| opacity_abs | Opacity gradient | Determines visibility update |
| scale_norm | Scale gradient | Determines size update |
| rot_norm | Rotation gradient | Determines orientation update |
| shs_norm | SH/color gradient | Determines appearance update (largest dimension) |

All per-parameter concentrations are saved in `c49_gopt_concentration.json` under the `_{paramname}` keys.

---

## Appendix B: Why Pearson is Near-Zero but Spearman is Moderate

The near-zero Pearson (0.033) with moderate Spearman (0.271) indicates that while the RANK ORDER of gradients has some stability (Spearman), the MAGNITUDES are uncorrelated (Pearson). This means:
- A Gaussian that is "high gradient" at t tends to be "above-average gradient" at t+1 (rank preserved)
- But the exact magnitude changes drastically (linear correlation lost)
- This is consistent with gradient oscillation: the same Gaussians remain important, but their gradient magnitudes fluctuate widely between iterations

For C51 selective backward, this means:
- Rank-based selection (top-K by gradient) has ~18% overlap at Top-10% — too low for reliable prediction
- Threshold-based selection would be even less reliable due to magnitude fluctuations
- The concentration (95.4% in Top-50%) means a coarse bottom-50% skip might work if identified instantaneously
