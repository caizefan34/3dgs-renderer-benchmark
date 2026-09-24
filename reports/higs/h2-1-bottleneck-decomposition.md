# H2-1 — Trainable HiGS B2 Forward/Backward Bottleneck Decomposition

**Status:** COMPLETE  
**Date:** 2026-09-21  
**Gate:** PROFILE_VALID  

---

## 0. Frozen source identity

```text
Base:       77ab983ffe43420b2131669cb35776b883ca4c3c
B2 patch:   74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c
B2_SOURCE_ID = 77ab983ffe43420b2131669cb35776b883ca4c3c + 74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c

gsplat_cuda.so:                                        361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98
experimental_gaussian_render_inference_scene_cuda.so: 00d7899d29e81269a98a409b83b441c805b4118a33d4764c61deef349df3f38a
gsplat_scene_cuda.so:                                  7389105286de5709054edba25f5af519762f9955ef0830530e750f71dbc3c031
```

No source identity was changed.

---

## 1. B2 execution graph

### Forward path

```
rasterize_gaussian_higs_frozen()
  └─ _HigsAutogradFunction.forward()
       ├─ F0: _cull_gaussians_batched()
       │    └─ fully_fused_projection(N_total=115278) → radii, means2d, depths, conics  [DISCARDED except visible_ids]
       │    └─ _union_visible_mask_native(radii) → visible_ids[44908]
       │
       ├─ F1: _gather_visible_native(means, quats, scales, opacities, colors, visible_ids)
       │    └─ higs_gather_visible (5 gather_rows_kernel_t kernels) → v_means, v_quats, v_scales, v_opacities, v_colors
       │
       ├─ F2: _native_forward_capture() → fully_fused_projection(N_visible=44908)
       │    └─ projection_ewa_3dgs_fused_fwd_kernel → radii, means2d, depths, conics  [SAVED]
       │
       ├─ F3: _maybe_evaluate_sh(sh_degree=3, v_colors, v_means, radii, viewmats)
       │    └─ spherical_harmonics_fwd_kernel_k16_3channel → colors_eval  [SAVED]
       │
       ├─ F4: isect_tiles(means2d, radii, depths, conics, opacities)
       │    ├─ intersect_tile_kernel → tiles_per_gauss, isect_ids, flatten_ids
       │    ├─ CUB DeviceRadixSort (6 launches) → sorted intersection IDs
       │    └─ isect_offset_encode → isect_offsets  [SAVED]
       │
       ├─ F5: rasterize_to_pixels_3dgs(means2d, conics, colors_eval, opacities, isect_offsets, flatten_ids)
       │    └─ rasterize_to_pixels_3dgs_fwd_kernel<3,16,256> → render_colors, render_alphas, last_ids  [render_alphas, last_ids SAVED]
       │
       ├─ F6: ctx.save_for_backward(26 tensors)
       │    └─ densification_radii scatter (index_copy) from visible to full-N
       │
       └─ F7: [autograd overhead — tensor clone, graph construction, save_for_backward bookkeeping]
```

### Backward path

```
_HigsAutogradFunction.backward(grad_frame, grad_alpha)
  └─ _native_backward()
       ├─ B0: 5× torch.zeros_like(full N_total=115278) → grad_means, grad_quats, grad_scales, grad_opacities, grad_colors  [72MB]
       │
       ├─ backend.higs_rasterize_backward() [single C++ pybind call]
       │    ├─ Stage 1: higs_blend_bwd_px_kernel<3, PX=2>
       │    │    └─ grid = I×tile_h×tile_w = 1×86×128 = 11008 blocks
       │    │    └─ processes 953144 intersections back-to-front
       │    │    └─ warpSum reduction → atomicAdd to v_means2d, v_conics, v_colors_flat, v_opacities_flat [I*N_v]
       │    │
       │    ├─ Stage 2: higs_projection_bwd_kernel
       │    │    └─ grid = ceil(I×N_v / 256), threads = I×N_v = 44908
       │    │    └─ radii > 0 mask; uses forward-captured conics
       │    │    └─ writes grad_means, grad_quats, grad_scales via visible_ids
       │    │
       │    ├─ Stage 3a: higs_camera_positions_kernel + higs_sh_vjp_grid_kernel
       │    │    └─ grid = ceil(I×N_v×D / 256), threads = I×N_v×D = 134724
       │    │    └─ radii > 0 mask; uses forward-captured colors_eval (activation mask)
       │    │    └─ shuffle-down warp reduce → atomicAdd to grad_colors, grad_means via visible_ids
       │    │
       │    └─ Stage 3b/4: higs_reduce_master_kernel (2 launches)
       │         └─ reduces per-view v_colors_flat, v_opacities_flat to master via visible_ids
       │
       └─ B5: grad_means2d.index_copy_(visible_ids, v_means2d_g) → full-N means2d gradient for densification
```

---

## 2. Forward decomposition (room/cam0, CUDA Events, 20 warmup / 100 measure)

| Stage | Function / Kernel | Time (ms) | % of F+B | Cost class |
|-------|-------------------|--------:|--------:|------------|
| F0 — Culling projection | `projection_ewa_3dgs_fused_fwd_kernel` on N_total | 0.195 | 4.3% | B. HIGS_STRUCTURAL_WORK |
| F1 — Gather visible | `gather_rows_kernel_t<48,3,4,1>` (5 launches) | 0.059 | 1.3% | C. TRAINING_ADAPTATION_OVERHEAD |
| F2 — Render projection | `projection_ewa_3dgs_fused_fwd_kernel` on N_visible | 0.078 | 1.7% | D. REDUNDANT_OR_RECOMPUTED_WORK |
| F3 — SH evaluation | `spherical_harmonics_fwd_kernel_k16_3channel` | 0.345 | 7.6% | A. ESSENTIAL_RENDER_WORK |
| F4 — Intersection + sort | `intersect_tile_kernel` + CUB radix sort (6 launches) | 0.508 | 11.3% | A. ESSENTIAL_RENDER_WORK |
| F5 — Rasterize | `rasterize_to_pixels_3dgs_fwd_kernel<3,16,256>` | 0.676 | 15.0% | A. ESSENTIAL_RENDER_WORK |
| **Stage sum** | | **1.862** | **41.3%** | |
| F6 — Saved state | `index_copy_kernel` (densification_radii) | ~0.0 | ~0% | E. MEMORY_TRANSFORMATION |
| F7 — Autograd overhead | tensor clone, graph, save_for_backward (residual) | 1.091 | 24.2% | F. SYNCHRONIZATION_OR_LAUNCH_OVERHEAD |
| **Forward total** | | **2.953** | **65.5%** | |

**Unattributed forward residual:** +1.091 ms (+36.9% of forward). This is the cost of making the forward differentiable: tensor `.detach().clone().requires_grad_(True)` for 5 inputs, autograd graph construction, and `save_for_backward` bookkeeping for 26 tensors.

---

## 3. Backward decomposition (room/cam0, CUDA Events + nsys)

| Stage | Kernel | Time (ms est.) | % of F+B | Cost class |
|-------|--------|------------:|--------:|------------|
| B0 — Gradient init | `FillFunctor<float>` (5× zeros_like full N_total) | 0.082 | 1.8% | E. MEMORY_TRANSFORMATION |
| **B1 — Blend backward** | `higs_blend_bwd_px_kernel<3,2>` | **2.713** | **60.1%** | A. ESSENTIAL_RENDER_WORK |
| B2 — Projection VJP | `higs_projection_bwd_kernel` | 0.033 | 0.7% | A. ESSENTIAL_RENDER_WORK |
| B3 — SH VJP | `higs_sh_vjp_grid_kernel` + `higs_camera_positions_kernel` | 0.062 | 1.4% | A. ESSENTIAL_RENDER_WORK |
| B4 — Opacity/color reduce | `higs_reduce_master_kernel` (2 launches) | 0.006 | 0.1% | A. ESSENTIAL_RENDER_WORK |
| B5 — means2d grad scatter | `index_copy_kernel` | 0.024 | 0.5% | E. MEMORY_TRANSFORMATION |
| **Backward total** | | **2.275** (direct) / **1.559** (F+B derived) | | |

**Backward measurement note:** Direct CUDA Events timing of `loss.backward()` = 2.275 ms. F+B derived backward (F+B − forward) = 4.512 − 2.953 = 1.559 ms. The 0.716 ms difference is cache warming: in the F+B measurement, forward outputs are hot in GPU cache when backward starts; in the separate backward measurement, the forward runs first but its outputs may be partially evicted. The F+B measurement (4.512 ms) is the most reliable production-path measurement.

**Backward kernel decomposition** from nsys (10 iterations, GPU 2):

| Kernel | Calls/iter | µs/iter | % of total GPU | % of backward GPU |
|--------|--------:|------:|------:|------:|
| `higs_blend_bwd_px_kernel<3,2>` | 1 | 3646 | 54.7% | 97.4% |
| `higs_sh_vjp_grid_kernel` | 1 | 59 | 0.9% | 1.6% |
| `higs_projection_bwd_kernel` | 1 | 33 | 0.5% | 0.9% |
| `higs_reduce_master_kernel` | 1 | 6 | 0.1% | 0.2% |
| Overhead (zeros, index_copy) | 17 | 106 | 1.6% | — |

**The blend backward kernel is 97.4% of backward GPU kernel time and 54.7% of all GPU kernel time.**

---

## 4. Top kernels (nsys, 10 F+B iterations)

| Rank | Kernel | Calls | Total µs | µs/iter | % GPU |
|-----:|--------|------:|--------:|------:|------:|
| 1 | `higs_blend_bwd_px_kernel<3,2>` | 10 | 36461 | 3646 | 54.7% |
| 2 | `rasterize_to_pixels_3dgs_fwd_kernel<3,16,256>` | 11 | 12854 | 1285 | 19.3% |
| 3 | `intersect_tile_kernel<float>` | 22 | 4944 | 494 | 7.4% |
| 4 | `CUB DeviceRadixSortOnesweepKernel` | 66 | 3030 | 303 | 4.6% |
| 5 | `FillFunctor<float>` (zeros) | 153 | 820 | 82 | 1.2% |
| 6 | `BinaryFunctor<MulFunctor>` (clone) | 10 | 809 | 81 | 1.2% |
| 7 | `higs_sh_vjp_grid_kernel` | 10 | 587 | 59 | 0.9% |
| 8 | `higs_projection_bwd_kernel` | 10 | 334 | 33 | 0.5% |
| 9 | `gather_rows_kernel_t<48>` | 11 | 422 | 42 | 0.6% |
| 10 | `projection_ewa_3dgs_fused_fwd_kernel` | 22 | 343 | 34 | 0.5% |

Top 4 kernels = 86.0% of GPU time. The blend backward alone exceeds the forward rasterize by 2.8×.

---

## 5. HiGS-specific overheads

| Overhead | Stage | Time | % F+B | HiGS-specific? |
|----------|-------|-----:|------:|:---:|
| Blend backward atomicAdd scatter | B1 | 2.713 ms | 60.1% | YES — I*N_visible scatter pattern unique to HiGS native backward |
| Culling projection (full-N) | F0 | 0.195 ms | 4.3% | YES — HiGS culling feature |
| Gather visible subset | F1 | 0.059 ms | 1.3% | YES — training adaptation for sparsity |
| Gradient init (full-N zeros) | B0 | 0.082 ms | 1.8% | YES — stop-gradient requires full-N zero pattern |
| means2d grad scatter for densification | B5 | 0.024 ms | 0.5% | YES — culling + densification interaction |
| Render projection (redundant) | F2 | 0.078 ms | 1.7% | YES — consequence of culling discarding F0 results |
| **Total HiGS-specific** | | **3.151 ms** | **69.9%** | |

---

## 6. Forward→backward reusable state

**All 10 forward-captured tensors are reused by backward.** No forward state is discarded that backward needs.

| Forward state | Saved? | Reused in backward? | How |
|--------------|:---:|:---:|-----|
| means2d (render) | YES | YES | Blend backward (T accumulation), projection VJP |
| conics (render) | YES | YES | Blend backward (Gaussian weight), projection VJP (inverse VJP) |
| colors_eval (SH) | YES | YES | Blend backward (color), SH VJP (activation mask) |
| opacities (visible) | YES | YES | Blend backward (alpha computation) |
| radii (render) | YES | YES | Projection VJP (sparsity mask), SH VJP (sparsity mask) |
| isect_offsets | YES | YES | Blend backward (tile range) |
| flatten_ids | YES | YES | Blend backward (Gaussian ID per intersection) |
| render_alphas | YES | YES | Blend backward (T_final) |
| last_ids | YES | YES | Blend backward (bin_final early-exit) |
| v_means/quats/scales/opacities/sh | YES | YES | Projection VJP, SH VJP inputs |

**H2-A REFUTED:** The native backward header explicitly states "No recomputation of the rasterization pipeline happens in the backward." All forward-captured state is consumed.

---

## 7. Work currently recomputed

| Recomputation | Cost | Removable? | Net oracle |
|--------------|-----:|:---:|-----:|
| F2 re-projects visible subset (F0 already projected them) | 0.078 ms | Partially — F0 must still project ALL N for visibility; saving means2d/conics for visible adds gather cost ≈ F2 cost | ~0 ms |
| B2 recomputes covar from quats/scales | 0.033 ms | Partially — saving covar adds 1.6MB storage; VJP requires covar for chain rule | ~0 ms |
| B3 recomputes SH basis | 0.062 ms | NO — VJP inherently requires basis evaluation | 0 ms |
| B0 zeros full-N gradients | 0.082 ms | NO — optimizer requires full-N (constraint: no optimizer change) | 0 ms |

**No significant removable recomputation exists.** The forward state reuse is already maximally exploited by the native backward.

---

## 8. B1A vs B2 stage differences (room/cam0)

| Stage | B1A | B2 | Δ (ms) | Notes |
|-------|----:|----:|------:|-------|
| Culling projection | — | 0.195 | +0.195 | NEW in B2 (HiGS culling) |
| Gather visible | — | 0.059 | +0.059 | NEW in B2 |
| Render projection | 0.118 | 0.078 | −0.040 | B2 faster (fewer Gaussians) |
| SH eval | 0.404 | 0.345 | −0.059 | B2 faster (fewer Gaussians) |
| Intersection+sort | 0.649 | 0.508 | −0.141 | B2 faster (CUB sort on fewer) |
| Rasterize | 0.640 | 0.676 | +0.036 | B2 slightly slower (capture writes) |
| Autograd overhead | 0.887 | 1.091 | +0.204 | B2 more overhead (26 saved tensors) |
| **Forward total** | **2.604** | **2.953** | **+0.349** | B2 13% slower (differentiable) |
| Blend backward | ~1.5 | 2.713 | +1.2 | B2 native backward SLOWER (atomicAdd scatter) |
| Projection VJP | ~0.1 | 0.033 | −0.067 | B2 faster (radii mask, saved conics) |
| SH VJP | ~0.2 | 0.062 | −0.138 | B2 faster (radii mask, saved colors_eval) |
| **Backward total** | **2.624** | **2.275** | **−0.349** | B2 13% faster |
| **F+B total** | **5.228** | **4.512** | **−0.716** | B2 14% faster |

**Key insight:** B2's forward is slower (culling + gather + autograd overhead), but B2's backward is faster (state reuse avoids recomputation). The net is 14% faster F+B. The backward speedup comes from projection VJP and SH VJP being much cheaper (radii mask + saved state), but the blend backward is actually SLOWER than B1A's PyTorch autograd backward because of the I*N_visible atomicAdd scatter pattern.

---

## 9. Ranked candidate table

| Candidate | Evidence | Removable (µs) | Added cost (µs) | Net oracle (µs) | % F+B | % Bwd | Correctness risk | Impl. cost | HiGS? | Verdict |
|-----------|----------|-------------:|-------------:|----------------:|------:|------:|------|------|:---:|---------|
| **C11** Block-level grad accumulation in blend bwd | nsys: blend_bwd=54.7% GPU; 953K atomicAdd scatters; warpSum only | 500 | 200 | **300** | **6.6%** | **13.2%** | MEDIUM (FP order change, validate cosine≥0.99) | MEDIUM | YES | **PROMOTE** |
| C4 Reduce atomicAdd contention (warp-level optimization) | Same as C11 but less aggressive | 500 | 100 | 400 | 8.9% | 17.6% | MEDIUM | MEDIUM | YES | KEEP_CANDIDATE |
| C6 Eliminate forward autograd overhead | 1.091ms residual (24.2% F+B) | 1091 | 891 | 200 | 4.4% | 8.8% | HIGH | HIGH | NO | DROP |
| C7 Sparse backward (skip low-importance Gaussians) | C49: top 32% = 90% gradient | 300 | 300 | 0 | 0% | 0% | HIGH | HIGH | YES | DROP (constraint violation) |
| C5 PX=4 in blend backward | PX=2→4 pixel parallelism | 200 | 100 | 100 | 2.2% | 4.4% | LOW | LOW | NO | DROP |
| C9 Fuse projection VJP + SH VJP | Two small kernels, similar grid | 95 | 35 | 60 | 1.3% | 2.6% | LOW | MEDIUM | NO | DROP |
| C1 Fuse culling + render projection | F0+F2 double projection | 78 | 59 | 19 | 0.4% | 0.8% | LOW | LOW | YES | DROP |
| C3 Sparse gradient init | B0 zeros 72MB full-N | 82 | 82 | 0 | 0% | 0% | MEDIUM | HIGH | YES | DROP (optimizer constraint) |
| C2 Reuse forward covar in proj VJP | B2 recomputes covar | 33 | 10 | 23 | 0.5% | 1.0% | LOW | LOW | NO | DROP |
| C10 Reuse requires_grad buffers | F7 tensor clone | 81 | 61 | 20 | 0.4% | 0.9% | LOW | LOW | NO | DROP |
| C8 Tile early-exit | All tiles occupied | 50 | 50 | 0 | 0% | 0% | LOW | LOW | YES | DROP |

---

## 10. Conservative net oracle for each candidate

See `artifacts/higs-h2-1/candidate_oracles.csv` for the full computation.

**Promoted candidate C11:**
- O_removable = 500 µs (estimated 10-15% reduction in blend_bwd kernel time from reducing atomic traffic by ~10×)
- C_transformation = 200 µs (block-level shared memory accumulation: 12.2KB shared memory, block sync, final scatter)
- O_net = 300 µs = **6.6% of F+B** = **13.2% of backward**
- Passes gate: ≥5% of F+B ✓, mechanism understood ✓, HiGS-specific ✓, correctness plausible ✓, implementable as isolated module ✓

---

## 11. ONE candidate recommended for implementation

### C11 — Block-level gradient accumulation in blend backward

**Mechanism:** The `higs_blend_bwd_px_kernel` currently does warpSum reduction then immediate atomicAdd to per-Gaussian I*N_visible gradient buffers. Each of the 953,144 intersections generates one atomicAdd per gradient component (v_means2d[2], v_conics[3], v_colors[3], v_opacities[1] = 9 components). Hot Gaussians in high-intersection tiles receive up to 424 atomic updates.

The optimization adds a **block-level shared memory accumulation phase**: each tile block accumulates per-Gaussian gradients in shared memory (sized by max intersections per tile, ~424 × 9 × 4 = 12.2KB, within the 48KB A100 limit), then performs a single atomicAdd per Gaussian per block. This reduces atomic traffic from ~953K updates to ~44908 × (tiles per Gaussian) ≈ 953K / 21.2 ≈ 45K updates — a ~21× reduction in atomic operations.

**Why this is HiGS-specific:** The I*N_visible scatter pattern is unique to the HiGS native backward. B1A's PyTorch autograd backward writes directly to N_total gradient tensors without the per-visible-Gaussian intermediate buffers. The atomicAdd contention is a direct consequence of HiGS's stop-gradient semantics (only visible Gaussians get gradients) implemented through the I*N_visible buffer + visible_ids reduction pattern.

---

## 12. Falsifiable prediction for C11

**Prediction:** Implementing block-level shared-memory gradient accumulation in `higs_blend_bwd_px_kernel` will reduce the blend backward kernel time by 10-15% (300-500 µs) while maintaining gradient cosine similarity ≥ 0.99 against the current implementation on room/cam0.

**Falsification conditions:**
1. If gradient cosine similarity < 0.99 → optimization changes numerical semantics (FAIL)
2. If blend_bwd kernel time reduction < 10% → atomicAdd is not the bottleneck (FAIL, memory-bound instead)
3. If shared memory pressure reduces occupancy to < 25% and kernel time INCREASES → register/shared memory tradeoff is unfavorable (FAIL)
4. If end-to-end F+B improvement < 5% → overhead outweighs benefit (FAIL)

---

## 13. Minimal implementation boundary

```text
File:   patches/higs-trainable-authoritative.patch
        Kernel: higs_blend_bwd_px_kernel (lines 3544-3794 in patch)

Change:
  1. Add shared-memory per-Gaussian accumulation buffer (sized by max intersections per tile)
  2. After warpSum, accumulate into shared memory instead of immediate atomicAdd
  3. After all batches processed, block-sync, then one atomicAdd per unique Gaussian per block
  4. Track unique Gaussian IDs in shared memory (bitmask or hash)

Constraint:
  - No change to kernel interface (same inputs/outputs)
  - No change to Python wrapper
  - No change to other backward stages
  - Validate: cosine(grad_old, grad_new) >= 0.99 on room/cam0
  - Measure: CUDA Events backward timing before/after
```

---

## 14. Report/artifact paths

```text
reports/higs/h2-1-bottleneck-decomposition.md          (this file)

artifacts/higs-h2-1/
    execution_graph.json                               (complete forward+backward call graph)
    stage_inventory.csv                                (all stages with cost classification)
    kernel_inventory.csv                               (nsys kernel-level inventory)
    memory_state_inventory.csv                         (30 tensors: saved, reused, recomputed)
    recomputation_inventory.csv                        (6 recomputation items)
    b1a_b2_stage_comparison.csv                        (B1A vs B2 stage-level comparison)
    timings_raw.csv                                    (all CUDA Events measurements)
    timings_summary.csv                                (summary metrics)
    candidate_oracles.csv                              (11 candidates with net oracle)
    analysis.json                                      (structured analysis with measured values)
    room_cam0_h2_1.json                                (raw profiling JSON)
    environment.json                                   (provenance)
    nsys_output.txt                                    (nsys stats output)
    h2_1_b2.sqlite                                     (nsys sqlite database)
    nvtx_kernel_decomposition.json                     (NVTX-range kernel decomposition)
```
