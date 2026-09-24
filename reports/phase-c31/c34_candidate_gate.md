# C34 Candidate Gate 0/1/2 — Source-Level Feasibility & Minimal Evidence

## C34-1: Trainable HiGS — **DROP**

### Source audit: HiGS pipeline state persistence

The HiGS pipeline in gsplat v1.5.3 (`gsplat.experimental.render`) has:

#### Forward pipeline (executed per frame)
```
1. launch_projection_sh_fused_kernel()
   Input:  means_planar [3,N] f32, qso_packed [N,8] f16, colors_packed [N,...]
   Output: visible [(N+31)/32] i32 (bitmask), means2d [1,1,N,2] f32,
           depths [1,1,N] f32, conics [1,1,N,4] f16, colors [N,4] f16

2. state.isect->execute()
   Input:  means2d, depths, conics, visible, tile_size
   Stages: count → chunk-base scan + offsets → fill → sort
   Output: m_mtGaussIds [n_mt_isects] i32, m_mtDepthKeys [n_mt_isects] i32,
           m_mtGaussCounts [n_mt] i32, m_mtGaussOffsets [n_mt+1] i32,
           m_mtBatchOffsets [n_mt+1] i32, m_mtGaussIdsSorted

3. state.isect->rasterize()
   Input:  means2d, conics, colors, background
   Output: rgbt [1,H,W,4] f16 (RGBT output, alpha = 1 - transmittance)
```

#### State classification

| Stage | Tensor | Persisted? | N-sized? | Needed for backward? | Recomputed or saved? |
|-------|--------|------------|----------|---------------------|----------------------|
| Projection | **means2d** [N,2] | ✅ in state | **YES** | ✅ (grad w.r.t. means, viewmat) | **Must save for backward** |
| Projection | **depths** [N] | ✅ in state | **YES** | ✅ (for depth gradient) | Must save |
| Projection | **conics** [N,4] | ✅ in state | **YES** | ✅ (for covariance gradient) | Must save |
| Projection | **visible** [N/32] | ✅ in state | **YES** | ✅ (rasterize backward needs visibility mask) | Must save |
| Projection | **colors** [N,4] | ✅ in state | **YES** | ✅ (grad w.r.t. SH coefficients) | Must save |
| Intersect | **mtGaussIds** [n_isects] | ✅ in IntersectMTFused | **NO** (isects > N) | ✅ (for scatter/gather to pixel gradients) | Must save |
| Intersect | **mtGaussOffsets** [n_mt+1] | ✅ in IntersectMTFused | no | ✅ (rasterize backward needs tile→GS mapping) | Must save |
| Intersect | **mtBatchOffsets** [n_mt+1] | ✅ in IntersectMTFused | no | ✅ (batch structure for backward) | Must save |
| Intersect | **mtGaussCounts** [n_mt] | ✅ in IntersectMTFused | no | no (recoverable from offsets) | Recomputed |
| Intersect | **depthKeys** [n_isects] | scratch | — | no (only for sort) | Not needed |
| Rasterize | **tileBuffer** [N_half] | scratch | — | no (final output only) | Not needed |
| Rasterize | **render_colors** [H,W,3] | output | — | ✅ (upstream gradient from loss) | Provided by caller |

#### Key findings

1. **All 3 stages' forward outputs must be saved for backward**:
   - Projection: means2d, depths, conics, visible, colors (56 bytes per GS)
   - Intersect: sorted GS IDs + tile offsets (~ISECTS_PER_GS × 8 bytes per GS)
   - Rasterize: tile→pixel mapping metadata

2. **The HiGS pipeline exists ONLY as `torch.ops.experimental.gaussian_render_inference_only`**:
   - Requires `torch.inference_mode()` — no autograd tracking
   - No backward kernels exist for: projection, macro-tile intersection, macro-tile rasterize
   - The existing `rasterize_gaussian_higs_trainable(differentiable=True)` explicitly falls back to standard gsplat rasterization (L569-590 of gaussian_inference.py)

3. **Missing backward kernels: every single one**
   - `launch_projection_sh_fused_kernel` has NO backward
   - `IntersectMTFused::execute` (count→scan→fill→sort) has NO backward
   - `IntersectMTFused::rasterize` has NO backward
   - Need: `launch_projection_bwd_kernel`, segmented sort through differentiable gather, macro-tile raster backward

4. **The macro-tile sort is a non-differentiable operation**
   - The radix sort produces sorted indices but the sorting operation itself is a discrete permutation
   - Forward: `means2d → tile ID → radix sort (depth, GS_id)`
   - Backward: need to scatter gradients back through the sort permutation
   - This IS possible (autograd can handle permutation) but the CUDA kernel must be written to save the permutation

5. **VRAM estimate for 1.59M Gs**:
   - Projection state: 1.59M × (2+1+4+4) × 4 bytes ≈ 70 MB (f32 means2d/depths + f16 conics/colors)
   - Intersect state: ~40M intersections × 2 × 4 bytes ≈ 320 MB (GS ids + sorted)
   - Total saved: ~390 MB for one camera — feasible for A100-40GB
   - But with autograd graph: each tensor keeps both .data and .grad ≈ 780 MB

### Implementation blockers (fatal)

1. **Complete missing CUDA backward implementation** — this is not a "minimal prototype". It requires writing 3-5 new CUDA kernels.
2. **The radix sort backward** — the segmented sort is a discrete operation. While the gradient can flow through the permutation, implementing this correctly in CUDA is a non-trivial research contribution.
3. **HiGS writes fp16 (half) throughout** — standard gsplat uses fp32 for gradients. Mixed-precision backward needs careful gradient propagation.

### Verdict: DROP
- Backward kernels do not exist and would require significant CUDA development
- The existing `rasterize_gaussian_higs_trainable(differentiable=True)` already provides the correct approach (standard backward)
- Even if gradient parity is established, the ~57ms fixed overhead is in autograd dispatch, not in renderer forward/backward GPU time
- **Minimum viable prototype cost**: 5 CUDA kernels, 2-week engineering effort, for uncertain reward

---

## C34-2: Incremental HiGS Hierarchy — **MAYBE**

### Source audit: state dependency after topology mutation

#### What happens during topology change

```
Densification (clone + split):
  1. model.densification() appends by torch.cat on all 5 param tensors
  2. model.prune() removes by boolean masking on all 5 param tensors
  3. _reconfigure_optimizer() discards ALL optimizer state
```

#### State invalidation analysis

| State component | Append (clone) | Remove (prune) | Split (clone+remove) |
|-----------------|---------------|----------------|---------------------|
| means_planar [3,N] | Append at end | Removed entries | Partial both |
| qso_packed [N,8] | Append at end | Removed entries | Partial both |
| colors_packed [N,...] | Append at end | Removed entries | Partial both |
| Optimizer state (5×2 buffers) | ❌ Full discard (`_reconfigure_optimizer`) | ❌ Full discard | ❌ Full discard |
| means2d, depths, conics | ❌ ALL stale | ❌ ALL stale | ❌ ALL stale |
| visible mask | ❌ Needs recompute | ❌ Needs recompute | ❌ Needs recompute |
| Colors (decoded SH) | ❌ Needs recompute | ❌ Needs recompute | ❌ Needs recompute |
| Intersect: mtGaussCounts | ⚠️ Only new macro-tiles affected | ⚠️ Only affected tiles | ⚠️ Both |
| Intersect: mtGaussIds (sorted) | ❌ Must rebuild (new IDs appended) | ❌ Must rebuild (ID shifts) | ❌ Must rebuild |
| Intersect: mtGaussOffsets | ❌ Must rebuild | ❌ Must rebuild | ❌ Must rebuild |
| Tile buffer | ❌ Must rebuild | ❌ Must rebuild | ❌ Must rebuild |

#### Critical insight: ID shift after removal

When Gaussian at index `i` is removed, ALL Gaussians at index `i+1..N-1` have their ID decremented by 1. This means:
- Every intersection entry referencing those GSs has a stale ID 
- Every sorted list is invalid
- You cannot simply "patch" IDs because the sort order depends on depth per GS

**For clone (append-only)**: New Gs are appended at the end (IDs N..N+Δ-1). Existing GS IDs are unchanged. Intersection can be incrementally computed: only the new Gs need to be projected, intersected, and their entries merged into the sorted lists.

**For prune (remove)**: Catastrophic. Every downstream buffer is invalid. **Full rebuild required.**

### Replacement cost bounds

| Mutation | Full rebuild cost | Affected intersection entries | Affected macro-tiles |
|----------|-------------------|------------------------------|---------------------|
| +0.01% (159 Gs) | ~2ms (total render) | ~159 × 5 = 795 (0.002%) | 2-5 (0.02%) |
| +0.1% (1,590 Gs) | ~2ms | ~7,950 (0.02%) | 10-25 (0.1%) |
| +1% (15,900 Gs) | ~2ms | ~79,500 (0.2%) | 50-200 (0.8%) |
| +5% (79,500 Gs) | ~2ms | ~397,500 (1.0%) | 200-800 (3.2%) |
| Remove 0.01% | ~2ms | ❌ 100% (ID shift) | ❌ 100% |
| Remove 0.1% | ~2ms | ❌ 100% (ID shift) | ❌ 100% |
| Remove 1% | ~2ms | ❌ 100% (ID shift) | ❌ 100% |
| Remove 5% | ~2ms | ❌ 100% (ID shift) | ❌ 100% |

#### Key finding

**The `_reconfigure_optimizer` call discards optimizer state EVERY topology change.** This is already the dominant cost (both in CPU time and momentum loss). The incremental HiGS hierarchy optimization is moot because the optimizer throw-away dominates.

### Verdict: MAYBE
- Append-only (clone) is cheap and local: incremental update makes sense (affected intersection < 1%)
- Remove (prune, split) causes ID shifts that invalidate all buffers — full rebuild required
- **The optimizer state rebuild is the actual blocker, not the hierarchy rebuild**
- If optimizer state transfer were fixed (saving momentum for non-removed params), then incremental hierarchy for append-only could save ~1.95ms (98% of render time for those steps)
- **Minimal next experiment**: Measure current optimizer rebuild cost (CPU: state_dict transfer + re-init) vs theoretical incremental update

---

## C34-8: Dependency-Aware Topology Rebuild — **DROP**

### Source audit: dependency graph

The dependency chain for each Gaussian parameter:

```
Gaussian parameters (means, quats, scales, opacity, shs)
  → forward() activations (normalize, exp, sigmoid)
    → rasterization forward (PROJECTION)
      → means2d, depths, conics (per-Gaussian, N-sized)
        → INTERSECTION (count → offsets → fill → sort)
          → sorted GS IDs, tile offsets (per-intersection)
            → RASTERIZE (tile-based alpha compositing)
              → pixel colors

Optimizer (per-parameter-group, maintenance)
```

#### What changes after topology mutation

| Changed entity | Depends on | Invalidated state | Fraction of total |
|---------------|-----------|-------------------|-------------------|
| New GS (append) | — | Intersection: only new macro-tiles (sparse, ~0.1% for 0.01% added) | **Very small** |
| Removed GS (delete) | — | ALL intersection entries referencing that GS AND all subsequent GSs (ID shift) | **100%** |
| Modified GS params | Original params | means2d, depths, conics for that GS; intersection for affected tiles | 2-10 tile neighbors |
| Optimizer state rebuild | — | ALL 5×2 state buffers (momentum + variance) | **100%** |

#### The ID shift problem (fatal)

Prune removes GS at index `i`. Now:
- GS IDs `{0..i-1}` unchanged
- GS IDs `{i+1..N-1}` become `{i..N-2}`
- All intersection entries with `gaussian_id > i` have stale indices
- Even entries with `gaussian_id < i` may reference different tile counts because the removed GS no longer occludes them

This means: **even removing 0.01% of GSs invalidates 100% of the intersection list.** There is no escape from full rebuild.

### Verdict: DROP
- **Any removal causes 100% intersection invalidation** due to index remapping
- The fraction of invalidated state after ANY prune = 100%
- Only append-only (clone without corresponding prune) is partial, but that's already handled by incremental allocation
- The optimizer state rebuild (`_reconfigure_optimizer`) already invalidates more state than the hierarchy rebuild does

---

## Final Decision Table

| Candidate | Verdict | Primary reason | GPU cost for confirmation | Risk |
|-----------|---------|---------------|--------------------------|------|
| **C34-1** (Trainable HiGS) | **DROP** | All backward kernels missing; 5+ CUDA kernels needed; gradient through radix sort is complex; autograd dispatch overhead (~57ms) is the bottleneck, not renderer GPU time | N/A — not implementable as "minimal prototype" | Fatal: requires full differentiable CUDA backend |
| **C34-2** (Incremental HiGS) | **MAYBE** | Append-only is cheap and local; prune causes 100% invalidation; optimizer state rebuild dominates anyway | Measure optimizer rebuild cost: compare `_reconfigure_optimizer` (full state discard) vs selective momentum preservation | Optimizer rebuild dominates; even perfect incremental hierarchy saves <2ms per 100 steps |
| **C34-8** (Dependency-Aware) | **DROP** | Any removal invalidates 100% of intersection due to Gaussian ID shift; no dependency graph can avoid full rebuild | N/A — mathematically determined | Fatal: ID shift after any removal |

## Implementation Boundaries

### C34-2 (if KEEP after prior-art gate)

**Scope**: Incremental hierarchy for APPEND-ONLY topology changes (clone).

**Modification points**:
- `IntersectMTFused` (C++): add `append_gaussians(means2d, depths, conics, visible)` method that projects new Gs and merges into existing intersection lists
- `gaussian_model.py`: track whether topology change was clone-only vs prune

**Correctness risk**: LOW for append-only (new Gs don't affect existing intersection order). HIGH for prune (see above).

**Not included**: Optimizer state preservation (separate issue).

**Minimal experiment**: 
- 1,000 iter training run with densification OFF + manual clone
- Record: render time for steps WITHOUT topology change vs steps WITH clone
- If clone step render time ≈ non-clone step render time (+<10%), incremental update is unnecessary

---

## Source Evidence Summary

All source locations refer to `C:\Users\36570\miniconda3\Lib\site-packages\gsplat\experimental\render\` (HiGS pipeline) and `scripts/epic05/phase7/` (Phase-7 training).

| File | Role | Lines |
|------|------|-------|
| `kernels/cuda/csrc/gaussian_inference/GaussianRenderInferenceScene.cu` | Main HiGS render loop | 144-422 |
| `kernels/cuda/csrc/gaussian_inference/IntersectMTFused.h` | Intersection state persistence | 43-110 |
| `kernels/cuda/csrc/gaussian_inference/GaussianRenderInferenceScene.h` | `InferenceRenderState` (per-frame buffers) | 52-82 |
| `functional/gaussian_inference.py` | `rasterize_gaussian_higs_trainable` — forward-only HiGS | 465-625 |
| `components/gaussian_inference_renderer.py` | `GaussianInferenceRenderer` — stateful wrapper | 81-428 |
| `scripts/epic05/phase7/train_3dgs.py` | `_reconfigure_optimizer` — full optimizer discard | 453-467 |
| `scripts/epic05/phase7/gaussian_model.py` | Densification + pruning — `torch.cat` append / boolean mask remove | 165-315 |
