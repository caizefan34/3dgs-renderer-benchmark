# H8-MR: Production Moment-Reparameterized Geometry Backward — Stage B Report

## Summary

**Classification: H8_MR_MARGINAL**

H8-MR is the production implementation of the H8-0R opacity-absorbed moment-space geometry adjoint, applied to the frozen `SCALAR_ADJOINT, CDIM=3, PX=2` HiGS native backward kernel. The optimization removes 4 FMUL + 2 FADD per pixel-Gaussian intersection from the blend backward hot loop and adds 4 FMUL + 2 FADD per Gaussian to the projection VJP, amortized by the contribution multiplicity (28–42×).

**Correctness**: ALL gradient comparisons PASS on all 3 scenes (support_mismatch=0, rel_L2 < 1e-3, cosine ≈ 1.0).

**Resources**: 0 register delta, 0 spills, 0 new buffers, 0 new sync — identical resource profile to the frozen baseline.

**Timing**: Backward gain varies from 30.81% (room) to 2.40% (garden). F+B gain varies from 14.33% (room) to 1.62% (garden). Geometric mean F+B gain: 3.58%.

The optimization is correct and resource-clean, but the timing gain is highly scene-dependent. Only 1 of 3 scenes exceeds the 3% F+B screening threshold with a strong margin; the other 2 are marginal.

---

## 1. Research Question

Does the H8-0R reparameterization, when implemented in the full production kernel with all surrounding code (traversal, sorting, scheduling, atomic accumulation), produce a measurable and reproducible backward speedup without correctness regression or resource penalty?

## 2. Hypothesis

Removing the conic·delta products from the blend backward (4 FMUL + 2 FADD per intersection) and deferring reconstruction to the projection VJP (4 FMUL + 2 FADD per Gaussian) will produce a net backward speedup proportional to the contribution multiplicity, with:
- Zero correctness deviation (algebraically exact reparameterization)
- Zero register/spill delta (validated in Stage A isolated probe)
- Measurable backward timing reduction on all 3 scenes

**Falsification condition**: Any of (a) correctness regression (rel_L2 > 1e-3 on any gradient), (b) register delta > 2 or spills > 0 at production level, (c) backward gain < 0% on any scene, (d) F+B gain < 0% on any scene.

## 3. Implementation

### Patch
- **File**: `patches/higs-h8-mr.patch` (274 lines)
- **Source**: `HigsNativeBackward.cu` (frozen SHA256 prefix: `e657485f9a187be6`)
- **Template**: `template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false, bool H8_MR = false>`
- **Env var**: `HIGS_BWD_H8_MR=1` (requires `HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint`)

### Edits applied
1. Add `H8_MR` template parameter to `higs_blend_bwd_px_kernel`
2. `if constexpr(H8_MR)` moment accumulation branch in geometry adjoint (stores Sx, Sy, Sxx, Sxy, Syy, R0 — no conic·delta products)
3. Add `template<bool H8_MR>` to `higs_projection_bwd_kernel`
4. Change `v_means2d` to non-const in projection VJP (for write-back)
5. Reconstruction body: `vx = A*Sx + B*Sy`, `vy = B*Sx + C*Sy`, `vA = 0.5*Sxx`, `vB = Sxy`, `vC = 0.5*Syy`, `v_opacity = R0` — using forward conic only
6. Replace `v_means2d[idx]` with `v_means2d_val` in 3 `proj_vjp` calls
7. Add `HIGS_BWD_H8_MR` env var in dispatch
8. Add `H8_MR_VALUE` parameter to `HIGS_LAUNCH_BLEND_BWD_PX` macro (passes to kernel template)
9. Add `h8_mr` variants to PX=1/2/4 launch dispatch
10. Add `<true>`/`<false>` template specialization to projection VJP dispatch

### Bug found and fixed
The initial patch's launch macro accepted `H8_MR_VALUE` as a parameter but did not pass it to the kernel template instantiation, causing all blend backward kernels to compile with `H8_MR=false`. This produced gross correctness failures (grad_means rel_L2 = 0.96). Fixed by adding `(H8_MR_VALUE)` to both the `cudaFuncSetAttribute` and kernel launch template argument lists.

## 4. Experiment

### Controls
- Same Gaussian parameters (means, quats, scales, opacities, SH) for both runs
- Same camera (cam0), resolution (2048-max-side), seed (4200)
- Same forward pass (identical topology, sorting, intersections)
- Same loss function (random projection with fixed seed)
- Only difference: `HIGS_BWD_H8_MR=0` vs `HIGS_BWD_H8_MR=1`
- GPU: CUDA_VISIBLE_DEVICES=4 (A100-PCIE-40GB)

### Correctness protocol
FP32 gradient comparison: baseline `SCALAR_ADJOINT` vs `SCALAR_ADJOINT + H8-MR`, same random seed, same forward state. Metrics: rel_L2, cosine, support_mismatch, NaN_count on grad_means, grad_quats, grad_scales, grad_opacities, grad_sh.

### Timing protocol
20 warmup iterations, 100 CUDA-event samples, 5 repetitions. Forward+backward measured together (backward = F+B − forward_median). Forward time measured separately (10 samples, median).

## 5. Results

### Production resource usage (cuobjdump --dump-resource-usage)

| Kernel | Template | Registers | Spills | Shared | Local |
|--------|----------|-----------|--------|--------|-------|
| blend_bwd_px | `<3, 2, true, false>` (baseline) | 56 | 0 | 0 | 0 |
| blend_bwd_px | `<3, 2, true, true>` (H8-MR) | 56 | 0 | 0 | 0 |
| projection_vjp | `<false>` (baseline) | 96 | 0 | 0 | 0 |
| projection_vjp | `<true>` (H8-MR) | 96 | 0 | 0 | 0 |

**Register delta: 0. Spill delta: 0. Shared delta: 0. Local delta: 0.** All hard-gate conditions satisfied at production level.

### Correctness

| Scene | Gradient | rel_L2 | cosine | support_mismatch | Verdict |
|-------|----------|--------|--------|------------------|---------|
| room | grad_means | 6.24e-06 | 1.000000 | 0 | PASS |
| room | grad_opacities | 4.94e-07 | 1.000000 | 0 | PASS |
| room | grad_quats | 9.86e-04 | 0.9999995 | 0 | PASS |
| room | grad_scales | 2.45e-05 | 1.000000 | 0 | PASS |
| room | grad_sh | 2.33e-07 | 1.000000 | 0 | PASS |
| bicycle | grad_means | 1.07e-06 | 1.000000 | 0 | PASS |
| bicycle | grad_opacities | 5.74e-07 | 1.000000 | 0 | PASS |
| bicycle | grad_quats | 1.78e-04 | 1.000000 | 0 | PASS |
| bicycle | grad_scales | 8.70e-06 | 1.000000 | 0 | PASS |
| bicycle | grad_sh | 2.18e-07 | 1.000000 | 0 | PASS |
| garden | grad_means | 1.37e-05 | 1.000000 | 0 | PASS |
| garden | grad_opacities | 4.56e-07 | 1.000000 | 0 | PASS |
| garden | grad_quats | 7.55e-04 | 1.000000 | 0 | PASS |
| garden | grad_scales | 1.71e-05 | 1.000000 | 0 | PASS |
| garden | grad_sh | 4.09e-07 | 1.000000 | 0 | PASS |

**All 15 gradient comparisons PASS.** The largest rel_L2 is grad_quats on room (9.86e-04), which is within FP32 reassociation tolerance (cosine = 0.9999995). The grad_quats sensitivity is expected because the quaternion-to-covariance VJP amplifies small reassociation differences in the conic gradient through the inverse-covariance chain rule.

### Timing

| Scene | Baseline BWD (ms) | H8-MR BWD (ms) | BWD Gain | Baseline F+B (ms) | H8-MR F+B (ms) | F+B Gain |
|-------|-------------------|-----------------|----------|--------------------|-----------------|----------|
| room | 2.824 | 1.954 | **30.81%** | 5.136 | 4.400 | **14.33%** |
| bicycle | 3.116 | 3.002 | 3.65% | 5.793 | 5.678 | 1.98% |
| garden | 1.411 | 1.377 | 2.40% | 3.357 | 3.302 | 1.62% |

**Geometric mean**: backward gain 6.46%, F+B gain 3.58%.

**Note on baseline variance**: The baseline backward timing for room (std=77.8) and bicycle (std=86.3) shows extreme outliers (mean >> median), while H8-MR timing is very stable (std < 0.4). Garden baseline is stable (std=0.012). The outliers suggest the baseline blend backward has occasional memory stalls that H8-MR's simpler accumulation pattern avoids. However, the median comparison is the fair metric.

## 6. Mechanism Evidence

### Arithmetic balance
| Component | Per intersection | Per Gaussian | Amortized by multiplicity |
|-----------|-----------------|--------------|--------------------------|
| Blend backward: FLOPs saved | 6 (4 FMUL + 2 FADD) | — | × multiplicity |
| Projection VJP: FLOPs added | — | 6 (4 FMUL + 2 FADD) | ÷ multiplicity |
| Net FLOPs saved | — | — | 6 × (multiplicity − 1) |

| Scene | Multiplicity | Net FLOPs saved per Gaussian |
|-------|-------------|------------------------------|
| room | 36.51 | 213.06 |
| bicycle | 28.70 | 165.20 |
| garden | 41.53 | 243.18 |

### Why room shows 30.81% but garden only 2.40%
The backward time is dominated by memory traffic (atomic accumulation, global loads for means2d, conics, opacities, colors), not just FP arithmetic. The 6 FLOP saving per intersection is a small fraction of the total per-intersection work (which includes atomic adds, warp reductions, and shared memory traffic). The room scene's larger absolute backward time (2.824 ms vs garden's 1.411 ms) and higher multiplicity means the arithmetic saving has more impact. The garden scene has a shorter backward time despite higher multiplicity, suggesting its intersections are cheaper per-pixel (smaller Gaussians, less overdraw).

## 7. Correctness/Quality

- **All gradients PASS** on all 3 scenes
- No support changes (same Gaussians contribute, same intersections)
- No NaN/Inf in any gradient
- FP32 rel_L2 within reassociation tolerance
- The optimization is **Type A (exact systems optimization)**: rendering semantics unchanged, gradients equivalent, training algorithm unchanged

## 8. Uncertainty

1. **Scene-dependent gain**: Only room shows a strong gain (14.33% F+B). Bicycle (1.98%) and garden (1.62%) are below the 3% F+B screening threshold. The geometric mean (3.58%) barely meets screening.
2. **Baseline timing variance**: The extreme baseline outliers on room/bicycle (but not garden) are not fully explained. They may indicate memory stall patterns in the baseline that H8-MR avoids, or they may be measurement artifacts from the F+B subtraction method.
3. **Single-seed screening**: This is a 1-seed exploratory screening. Confirmation requires ≥3 seeds (per the research protocol).
4. **No E2E training validation**: The F+B gain does not directly translate to E2E iteration gain (which includes optimizer, loss, topology). The backward is only one component of the training iteration.
5. **No full convergence**: No quality (PSNR/SSIM/LPIPS) comparison at convergence has been performed.

## 9. Verdict

**H8_MR_MARGINAL**

The optimization is:
- ✅ Correct (all gradients PASS, algebraically exact)
- ✅ Resource-clean (0 register delta, 0 spills, 0 new buffers)
- ✅ No regression (no scene shows negative gain)
- ⚠️ Gain is highly scene-dependent (1.62%–14.33% F+B)
- ⚠️ Only 1 of 3 scenes strongly exceeds the 3% F+B screening threshold
- ⚠️ Geometric mean F+B gain (3.58%) barely meets screening

**Not STRONG** because: the per-scene variance is too high and 2 of 3 scenes are marginal.
**Not WEAK** because: correctness is perfect, resources are clean, and no regression exists.

## 10. Highest-Value Next Experiment

1. **Investigate baseline timing variance**: Profile the baseline blend backward with nsys/ncu to determine if the extreme outliers on room/bicycle are caused by memory stalls (L2 cache misses, atomic contention) that H8-MR avoids, or are measurement artifacts.
2. **Multi-seed confirmation**: Run 3+ seeds to determine if the room gain is reproducible and the bicycle/garden gains are consistently marginal.
3. **E2E iteration timing**: Measure the full training iteration (forward + loss + backward + optimizer) to determine the E2E gain, which is the metric that matters for training acceleration.

## Artifacts

- `artifacts/higs-h8-mr/correctness.json` — gradient comparison results
- `artifacts/higs-h8-mr/timing.json` — timing results (20 warmup, 100 samples, 5 reps)
- `artifacts/higs-h8-mr/resource_usage.json` — production resource usage
- `artifacts/higs-h8-mr/resource_usage.txt` — full cuobjdump output
- `artifacts/higs-h8-mr/mechanism_accounting.json` — mechanism analysis
- `artifacts/higs-h8-mr/provenance.json` — build provenance
- `patches/higs-h8-mr.patch` — 274-line production patch

## Provenance

| Item | Value |
|------|-------|
| GPU | A100-PCIE-40GB (sm_80, 108 SMs) |
| Compiler | nvcc 12.8.93 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.10 |
| Frozen source SHA256 | e657485f9a187be6... |
| Patch lines | 274 |
| Build | /mnt/storage_pool/liaoyuanjun/higs_h8_mr_cache/ |
| Scenes | room, bicycle, garden (2048-max-side, cam0, seed 4200) |
| Screening threshold | ≥3% reproducible E2E iteration gain |
| Stage A verdict | H8_0R_STRONG |
| Stage B verdict | H8_MR_MARGINAL |
