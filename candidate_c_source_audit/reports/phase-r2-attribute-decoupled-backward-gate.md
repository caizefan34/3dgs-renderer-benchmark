# Phase R2 — Attribute-Decoupled Backward Feasibility Gate

**Semantic label**: `REFERENCE_V1_ABSGRAD`  
**Date**: 2026-09-14  
**Scene**: mipnerf360/room  
**Renderer**: gsplat 1.5.3  
**Checkpoint**: iter_10000 (N=780,884)

---

## Final Answer

```
Candidate B: B_DROP
Candidate A: DROP
C49: C49_DOWNGRADE
Best exclusive derivative family: Appearance/SH
Measured maximum realistic E2E opportunity: 1.13%
```

### Five Questions

1. **Is attribute utility separation real after signed-utility correction?**  
   Yes. SH utility is extremely concentrated (C10_pos = 89.8%, min K for 95% = 20%), geometry less so (C10_pos = 72.4%, min K for 95% = 50%). The separation survives correction — SH and geometry active sets are genuinely different.

2. **Which derivative paths are actually separable?**  
   - `spherical_harmonics_bwd`: **SH_PATH_COUPLED** — computes both v_coeffs (dL/dSH) and v_dirs (dL/dxyz via view direction). Cannot skip dSH without breaking dL/dxyz.
   - `projection_ewa_3dgs_fused_bwd`: GEOMETRY_ONLY, separable in principle.
   - `rasterize_to_pixels_3dgs_bwd`: SHARED — all four gradient paths (v_colors, v_opacity, v_means2d, v_conics) are computed in one kernel loop.

3. **How much GPU time belongs exclusively to each attribute family?**  
   - SH backward: 0.28ms = **1.0% E2E** (exclusive)
   - Projection backward: 0.17ms = **0.6% E2E** (exclusive)
   - Raster backward: 11.26ms = **41.4% E2E** (shared, cannot be attribute-decomposed)
   - Other overhead: 10.63ms = 39.1% E2E (autograd graph + loss + SSIM)

4. **Does oracle attribute decoupling produce a useful work/utility Pareto frontier?**  
   No. The oracle masks show that utility IS concentrated per family (SH needs only 20% for 95% positive utility), but the exclusive kernels that could be skipped are so small (<1ms each) that skipping them saves <1.2% E2E. The real cost is in the shared raster kernel, which cannot be decomposed by attribute.

5. **Is Candidate B strong enough for a full training implementation?**  
   No. The maximum realistic E2E opportunity is 1.13%, far below the 5% B_SYSTEMS_KEEP threshold and the 10% B_STRONG_KEEP threshold. The exclusive derivative-family work is <5% E2E.

---

## Part 1: Corrected U_loss Concentration

### Signed-Utility Correction

Because 14.0% of visible Gaussians have negative U_loss (Adam momentum reversal), raw Gini is invalid. We use positive-only (C_K+) and absolute (C_K^abs) concentration:

| Family | Neg Fraction | Neg/Pos Ratio | C10_pos | C50_pos | C10_abs | C50_abs |
|--------|:----------:|:------------:|:-------:|:-------:|:-------:|:-------:|
| Total | 14.0% | 0.013 | 0.758 | 0.988 | 0.738 | 0.981 |
| Geometry | 15.3% | — | 0.724 | 0.988 | 0.709 | 0.980 |
| SH | 10.1% | — | **0.898** | **1.000** | 0.882 | 1.000 |
| Opacity | 26.5% | — | 0.818 | 0.998 | 0.775 | 0.987 |

**Key finding**: After signed-utility correction, concentration is confirmed but slightly weaker than R1's uncorrected Top-10% mass of 82.0%. The corrected C10_pos = 75.8% for total utility. SH remains the most concentrated (C10_pos = 89.8%).

**C49 Reclassification: C49_DOWNGRADE** — The corrected positive-utility concentration (C10_pos = 75.8%) is lower than R1's uncorrected Top-10% mass (82.0%). The negative utility fraction (14%) and neg/pos ratio (1.3%) introduce enough confounding that C49 cannot be cleanly classified as update-utility concentration. The gradient concentration observation (C49, Gini 0.75) is a gradient-distribution phenomenon that partially — but not cleanly — translates to update utility.

---

## Part 2: Exact Backward Dependency Graph

### Autograd Graph Structure

```
loss.backward()
│
├─ SSIM backward + L1 backward
│   └─ v_render_colors [H, W, 3], v_render_alphas [H, W, 1]
│
├─ _RasterizeToPixels.backward()
│   │  Input: v_render_colors, v_render_alphas
│   │  CUDA: rasterize_to_pixels_3dgs_bwd
│   │  Output: v_means2d, v_conics, v_colors, v_opacities
│   │
│   ├─ v_colors [N, 3]  → feeds SH backward (APPEARANCE)
│   ├─ v_opacities [N]   → feeds opacity grad (OPACITY_ONLY)
│   ├─ v_means2d [N, 2]  → feeds projection backward (GEOMETRY)
│   └─ v_conics [N, 3]   → feeds projection backward (GEOMETRY)
│
├─ _SphericalHarmonics.backward()
│   │  Input: v_colors
│   │  CUDA: spherical_harmonics_bwd
│   │  Output: v_coeffs [N, K, 3], v_dirs [N, 3]
│   │
│   ├─ v_coeffs → dL/dSH (APPEARANCE)
│   └─ v_dirs   → dL/dxyz (GEOMETRY, via dirs = means - campos)
│
├─ _FullyFusedProjection.backward()
│   │  Input: v_means2d, v_conics
│   │  CUDA: projection_ewa_3dgs_fused_bwd
│   │  Output: v_means, v_covars, v_quats, v_scales
│   │
│   ├─ v_means  → dL/dxyz (GEOMETRY)
│   ├─ v_scales → dL/dscale (GEOMETRY)
│   └─ v_quats  → dL/drot (GEOMETRY)
│
└─ Activation backward
    └─ dL/dopacity (via sigmoid), dL/dscaling (via exp), dL/drot (via normalize)
```

### Critical Dependency: SH → xyz

**SH backward computes BOTH dL/dSH and dL/dxyz.**

Source: `gsplat/cuda/_wrapper.py:1831-1845`:
```python
def backward(ctx, v_colors: Tensor):
    dirs, coeffs, masks = ctx.saved_tensors
    compute_v_dirs = ctx.needs_input_grad[1]  # True if xyz requires grad
    v_coeffs, v_dirs = spherical_harmonics_bwd(
        num_bases, sh_degree, dirs, coeffs, masks,
        v_colors.contiguous(), compute_v_dirs)
    return None, v_dirs, v_coeffs, None
```

The forward pass computes `dirs = means - campos` (view direction). The backward produces `v_dirs` which flows back to `means` (xyz) through this dependency. This means **skipping SH backward for selected Gaussians would make dL/dxyz incorrect** for those Gaussians.

### Classification: SH_PATH_COUPLED

The SH coefficient path is **NOT separable** from the geometry contribution. dL/dSH and dL/dxyz(via v_dirs) are computed in the same kernel call. Skipping one breaks the other.

---

## Part 3: Derivative Work Classification

| Component | Classification | Exclusive? | Cost (ms) |
|-----------|:-------------:|:----------:|:---------:|
| rasterize_to_pixels_3dgs_bwd | **SHARED** | No | 11.26 |
| spherical_harmonics_bwd | **COUPLED** (SH+xyz) | No | 0.28 |
| projection_ewa_3dgs_fused_bwd | **GEOMETRY_ONLY** | Yes | 0.17 |
| Loss + SSIM + autograd overhead | **SHARED** | No | 10.63 |
| Opacity gradient (v_opacities) | **OPACITY_ONLY** (within shared kernel) | No | — |

---

## Part 4: Source-Level SH Decomposition

**Verdict: SH_PATH_COUPLED**

The `spherical_harmonics_bwd` CUDA kernel computes:
1. `v_coeffs[n, k, c]` — gradient w.r.t. SH coefficients (→ dL/dSH)
2. `v_dirs[n, c]` — gradient w.r.t. view direction (→ dL/dxyz via dirs = means - campos)

Both outputs share the same input `v_colors` and the same SH evaluation. The arithmetic for `v_dirs` involves the derivative of SH basis functions w.r.t. the direction vector, which is interleaved with the coefficient gradient computation.

**Can dSH be skipped while preserving geometry contribution?**  
No. The `v_dirs` output (which feeds dL/dxyz) is computed from the same SH basis evaluation as `v_coeffs`. Skipping the coefficient gradient computation would require a separate kernel that computes only `v_dirs` — this is theoretically possible but would require:
1. A new CUDA kernel that computes only the view-direction derivative
2. This kernel would still need to evaluate the SH basis functions
3. The savings would be minimal since SH backward is only 0.28ms total

---

## Part 5: Raster Backward Decomposition

**CDIM = 3** (RGB channels at rasterization), NOT 48 SH coefficients.

The raster backward kernel computes per-pixel, per-Gaussian:
- `v_colors[n, 3]` — 3 atomicAdds (one per RGB channel) → feeds SH backward
- `v_opacities[n]` — 1 atomicAdd → feeds opacity gradient
- `v_means2d[n, 2]` — 2 atomicAdds → feeds projection backward
- `v_conics[n, 3]` — 3 atomicAdds → feeds projection backward

Total: 9 atomicAdds per Gaussian per pixel intersection, plus the T/buffer update.

The kernel cannot be attribute-decomposed without rewriting it to have separate code paths for each output. The T (transmittance) computation is shared — it must process all Gaussians in depth order regardless of which gradient paths are active.

---

## Parts 6-8: Backward Component Timing

### Measurement Setup
- Checkpoint: iter_10000, N=780,884
- 5 camera viewpoints, 20 warmup, 100 measured iterations
- CUDA event timing with synchronized barriers
- Instrumented autograd backward functions (raster_bwd, proj_bwd, sh_bwd)

### Results

| Component | Median (ms) | % of Backward | % of E2E |
|-----------|:----------:|:------------:|:--------:|
| rasterize_to_pixels_3dgs_bwd | 11.26 | 50.4% | **41.4%** |
| spherical_harmonics_bwd | 0.28 | 1.3% | **1.0%** |
| projection_ewa_3dgs_fused_bwd | 0.17 | 0.8% | **0.6%** |
| Other (loss+autograd) | 10.63 | 47.6% | 39.1% |
| **Total backward** | **22.34** | 100% | 82.1% |
| **E2E** | **27.18** | — | 100% |

---

## Part 9: Derivative-Family Amdahl Table

| Derivative Family | Exclusive (ms) | Shared (ms) | Max Removable E2E % |
|-------------------|:--------------:|:-----------:|:-------------------:|
| Appearance / SH coeff | 0.28 | 0 | 1.0% |
| Geometry / Projection | 0.17 | 0 | 0.6% |
| Opacity | 0 | 11.26 (shared raster) | 0% |
| Raster (shared) | 0 | 11.26 | 0% |
| Other overhead | 0 | 10.63 | 0% |

**No exclusive derivative family has ≥5% E2E cost.** The shared raster kernel (41.4% E2E) dominates and cannot be attribute-decomposed without a full CUDA kernel rewrite.

---

## Parts 10-11: Oracle Attribute Masks

### Family-Specific Oracle Utility Retention

| Family | K=20% pos | K=32% pos | K=50% pos | K=75% pos |
|--------|:---------:|:---------:|:---------:|:---------:|
| Geometry | 0.867 | 0.944 | 0.988 | 0.998 |
| **SH** | **0.972** | **0.995** | **1.000** | 1.000 |
| Opacity | 0.928 | 0.977 | 0.998 | 1.000 |

### Min K for 95% Positive Utility

| Family | Min K (median) |
|--------|:--------------:|
| SH | **20%** |
| Opacity | **32%** |
| Geometry | **50%** |

SH utility is extremely concentrated — only 20% of visible Gaussians are needed to capture 95% of positive SH update utility. Geometry is the least concentrated (50% needed).

---

## Part 12: Attribute-Decoupled Oracle Combination

| Family | Keep % | Positive Utility Retained |
|--------|:------:|:------------------------:|
| Geometry | 50% | 95% |
| SH | 20% | 95% |
| Opacity | 32% | 95% |

If these masks could be applied to skip exclusive kernel work:
- Skip SH bwd for 80% of Gaussians: save 0.28ms × 0.8 = 0.22ms = 0.8% E2E
- Skip proj bwd for 50% of Gaussians: save 0.17ms × 0.5 = 0.09ms = 0.3% E2E
- **Combined: ~1.1% E2E**

---

## Part 13: SH State Predictor vs Oracle

| Metric | Value |
|--------|:-----:|
| Precision (K=50%) | 0.675 |
| Recall (K=50%) | 0.675 |
| Positive utility coverage | 0.703 |

The SH state predictor from R1 achieves 70.3% positive utility coverage at K=50%, which is below the 80% threshold for SUPPORTING_COMPONENT. **Candidate A: DROP.**

---

## Part 14: Novelty Boundary

### AdamW-GS
Decouples optimizer/update/regularization semantics. Does NOT condition execution of different backward derivative families on different Gaussian active sets. It modifies the optimizer, not the backward computation graph.

### Faster-GS
Uses per-Gaussian backward and gradient approximation. Does NOT use attribute-specific active sets or suppress selected derivative families independently. It approximates the full backward, not specific attribute paths.

Neither prior work performs attribute-decoupled backward execution. However, the reason is likely that the computational savings are negligible — as this phase demonstrates.

---

## Part 15: Decision Gates

### Candidate B: B_DROP

| Criterion | Required | Result | Pass? |
|-----------|----------|--------|:-----:|
| Separation survives correction | Yes | C10_pos: SH=89.8%, Geo=72.4% | ✅ |
| ≥1 family with ≥5% exclusive E2E | Yes | Max = 1.0% (SH) | ❌ |
| Oracle sparsity removes meaningful work | Yes | Combined = 1.1% E2E | ❌ |
| Combined oracle ≥10% E2E (STRONG) | ≥10% | 1.1% | ❌ |
| Combined oracle 5-10% E2E (SYSTEMS) | 5-10% | 1.1% | ❌ |

**B_DROP**: Exclusive derivative-family work is <5% E2E. The shared raster kernel (41.4% E2E) cannot be attribute-decomposed without a full CUDA rewrite. The maximum realistic E2E opportunity is 1.13%.

### Candidate A: DROP

SH state predictor positive utility coverage = 70.3% < 80% threshold.

---

## Part 16: C49 Decision

**C49_DOWNGRADE**

After signed-utility correction:
- C10_pos = 75.8% (was 82.0% uncorrected in R1)
- Negative fraction = 14.0%, neg/pos ratio = 1.3%
- The signed confounding reduces the clean concentration signal

C49's gradient concentration (Gini 0.75) is a gradient-distribution observation that partially translates to update utility, but the signed nature of U_loss (14% negative) and the reduction from 82% to 76% in corrected concentration means it cannot sustain the C49_UPDATE_STRONGER classification from R1.

---

## Part 17: Deliverables

| File | Description |
|------|-------------|
| `results/reference_v1/r2/corrected_update_utility_concentration.json` | Part 1: Positive/absolute concentration |
| `results/reference_v1/r2/backward_dependency_graph.json` | Parts 2-5: Dependency graph + SH decomposition |
| `results/reference_v1/r2/derivative_family_cost.json` | Parts 6-9: Component timing + Amdahl table |
| `results/reference_v1/r2/gradient_correctness.json` | Part 7: Gradient validation (from R1) |
| `results/reference_v1/r2/oracle_attribute_masks.json` | Parts 10-11: Family-specific oracle masks |
| `results/reference_v1/r2/attribute_pareto.json` | Part 12: Oracle combination |
| `results/reference_v1/r2/sh_state_vs_oracle.json` | Part 13: SH state predictor vs oracle |
| `results/reference_v1/r2/final_decision.json` | Parts 15-16: Decision gates + C49 |

---

## Appendix: Why Attribute Decoupling Fails Computationally

The structural basis for attribute decoupling is real (R1 showed Geo×App Jaccard < 0.5, and R2 confirms separation survives signed-utility correction). But the computational structure of gsplat's backward makes it infeasible:

1. **The raster kernel is monolithic**: `rasterize_to_pixels_3dgs_bwd` computes all four gradient paths (v_colors, v_opacity, v_means2d, v_conics) in a single per-pixel, per-Gaussian loop. The T (transmittance) accumulation is shared and must process all Gaussians regardless of which attributes are active.

2. **Exclusive kernels are tiny**: SH backward (0.28ms) and projection backward (0.17ms) are the only exclusively-attribute kernels. Together they account for 1.6% of E2E. Even with perfect oracle masks (skip 80% of SH, 50% of geometry), the saving is ~1.1% E2E.

3. **SH is coupled to xyz**: The SH backward produces both dL/dSH and dL/dxyz (via view direction). Skipping SH backward for selected Gaussians would break their geometry gradients. This coupling means even the "exclusive" SH backward is not truly attribute-exclusive.

4. **The real opportunity is in the shared kernel**: To save meaningful time, one would need to decompose the raster kernel itself — running separate passes for v_colors-only, v_opacity-only, v_geometry-only Gaussians. This requires a full CUDA kernel rewrite and would likely increase overhead from multiple passes.

The attribute-decoupled backward concept is structurally valid but computationally moot in the current gsplat architecture.
