# Phase 13C — New CUDA/Kernel Optimization Reconnaissance Report

**Date:** 2026-08-28  
**Author:** DSH coding agent  
**Status:** COMPLETE

---

## 1. Overview

Phase 13B identified three potential CUDA optimization candidates:
1. Segmented sort
2. Visibility culling
3. Adaptive tile selection

This report provides the detailed reconnaissance requested: source location, current implementation status, ablation possibility, forward/backward/GT quality/training risk, and potential E2E benefit.

---

## 2. Segmented Sort

### Source Location

- **File:** `gsplat/cuda/csrc/sort_tiles.cu`
- **CUDA primitive:** `cub::DeviceSegmentedRadixSortPairs`
- **API exposure:** `segmented` parameter in `gsplat.rasterization(segmented=True/False)`

### Current Implementation Status

**EXPERIMENTAL FLAG — already exists but untested in this benchmark.**

- The `segmented` parameter is passed through the Python API to the `sort_tiles` CUDA kernel
- When `segmented=True`, the code calls `cub::DeviceSegmentedRadixSortPairs` instead of the default `cub::DeviceRadixSortPairs`
- The difference: segmented sort maintains per-tile key-value segments separately, while the default sorts across the entire buffer
- **Must verify that `segmented=True` actually changes the CUB workload** — if gsplat's default sort already behaves as segmented (or the flag is a NO-OP in the current version), this optimization is already applied

### Independent Ablation Possible?

**YES.** Simple Python-level parameter flip:
```python
rendered, alpha, info = rasterization(
    ..., segmented=False,  # baseline (default)
)
rendered, alpha, info = rasterization(
    ..., segmented=True,  # experimental
)
```

### Risk Assessment

| Risk | Level | Explanation |
|:-----|:-----:|:------------|
| Forward correctness | **Medium** | Segmented sort reorders keys differently. Must verify pixel-identical output (max_abs_diff=0.0). Small FP differences possible from different reduction order. |
| Backward correctness | **Medium** | Backward kernel reads sorted key-value pairs. Different sort order could change gradient reduction order → FP differences in gradients. |
| GT quality | **Low** | If forward is pixel-identical, GT quality is identical. |
| Training | **Low** | If forward and backward are correct within FP precision, training dynamics are unchanged. |

### Potential E2E Benefit

| Claim | Evidence |
|:------|:---------|
| 20-40% sort speedup | Claimed in literature for segmented sort |
| Sort as % of forward time | ~5-15% (estimated from kernel timing in Phase 8) |
| **E2E forward benefit** | **1-6%** (20-40% of 5-15%) |
| **Training benefit** | Even smaller (sort is ~5% of total training step) |

### Verification Step

Before labeling this as a new optimization, run:
```python
import torch
# Profile with segmented=False and segmented=True
# Capture cub::DeviceSegmentedRadixSort kernel duration
```

**If the flag is already default or produces no timing difference, this is a NO-OP and should be removed from the candidate list.**

---

## 3. Visibility Culling

### Source Location

- **File:** `gsplat/cuda/csrc/projection_ewa_3dgs_packed_fwd.cu`
- **Kernel:** `_fully_fused_projection_packed_kernel`
- **Mechanism:** Frustum culling — Gaussians outside camera frustum are excluded from packed output

### Current Implementation Status

**ALREADY IMPLEMENTED IN BASELINE.**

The packed projection kernel already performs:
1. **Frustum culling** (Gaussian center must be in front of camera)
2. **Compaction** (CUB DeviceScan + scatter to remove invisible Gaussians)

This IS the visibility culling that reduces per-frame workload from "all Gaussians" to "visible Gaussians only."

### Independent Ablation Possible?

**NO — without modifying CUDA source.** However, the M2 comparison (packed vs dense) serves as the ablation:
- Phase 10A (2026-09-23) completed full 30K training for both modes
- **M2 packed/dense is a NO-OP for training** — identical performance, quality, and trajectory
- Inference benefit: 2.02× forward speedup on room (packed projects/culls only visible 24.5% of Gs)

### Risk Assessment

| Risk | Level | Explanation |
|:-----|:-----:|:------------|
| Forward correctness | **Low** — already in baseline |
| Backward correctness | **Low** — Phase 9A verified gradient correctness |
| GT quality | **None** — bit-exact output (Phase 9A) |
| Training | **None** — Phase 10A confirmed NO-OP |

### Potential E2E Benefit

**0% for additional culling beyond packed.** The baseline already has frustum culling.

Remaining opportunities for **finer-grained culling** (very hard CUDA engineering):
- Per-tile culling: A Gaussian still gets processed in all tiles it touches (mean occupancy ~8 tiles for tile16)
- To eliminate per-tile overhead, need per-tile visibility test → requires modifying tile intersection kernel
- Could theoretically reduce intersection count by 30-50% for outdoor scenes

### Important

> **Visibility culling should be REMOVED from the new optimization candidate list.** It is already implemented as the packed projection pipeline. Proposing it as new would be misrepresenting the baseline.

---

## 4. Adaptive Tile Selection

### Source Location

- **File:** `gsplat/csrc/gsplat.cpp` (Python bindings) → `rasterization()` API
- **Parameter:** `tile_size` (default=16)
- **Proposed mechanism:** Vary `tile_size` per forward/backward call based on workload characteristics

### Current Implementation Status

**NOT IMPLEMENTED.** Currently `tile_size` is fixed per `rasterization()` call:
- No per-image or per-batch variation support
- No automatic selection based on scene characteristics
- Must be set at the training script level

### Independent Ablation Possible?

**YES.** The 8-size sweep already serves as the ablation:
- All sizes are pixel-identical (max_abs_diff=0.0)
- Gradient correctness verified (Phase 8)
- The only variable is performance

### Risk Assessment

| Risk | Level | Explanation |
|:-----|:-----:|:------------|
| Forward correctness | **Low** — same compiled kernel binary, only launch config changes. Pixel-identical for all 8 sizes. |
| Backward correctness | **Low** — REG=40, SHARED=1024 identical for all tile sizes. Gradient correctness verified. |
| GT quality | **None** — pixel-identical output. |
| Training | **Low** — tile_size is a runtime parameter that does not change training loop logic. 500-step sanity passes for all tested sizes. BUT: training winner ≠ snapshot winner, so adaptive selection must use training-aware criteria, not snapshot timing. |

### Potential E2E Benefit

| Metric | Expected Benefit | Note |
|:-------|:----------------:|:-----|
| Forward (snapshot) | 10-15% (scene-dependent) | tile20 beats tile16 by 11-19% |
| Training | **UNCERTAIN** | tile32 beats tile16 by 58% on room, but tile20/tile24 not trained yet |

### Implementation Complexity

| Aspect | Difficulty |
|:-------|:----------|
| Per-scene static selection | **Trivial** (config file change) |
| Per-frame dynamic selection | **Medium** (need workload estimator) |
| Training-aware selection | **Hard** (need to model backward + optimizer cost) |

---

## 5. Updated Candidate Ranking

| Rank | Candidate | Status | E2E Benefit | Priority |
|:----:|:----------|:------:|:-----------:|:--------:|
| **1** | Adaptive tile selection (per-scene static) | Ready for Phase 14A | 10-15% forward, UNCERTAIN training | HIGH |
| **2** | Segmented sort verification | Must verify first | 1-6% forward, <1% training | MEDIUM |
| — | Visibility culling | **REMOVED** (already in baseline) | 0% | N/A |
| **3** | Per-tile culling (new proposal) | CUDA engineering required | 30-50% intersection reduction (outdoor) | LOW (high effort) |

---

## 6. Proposed Next Steps

### Immediate (Phase 14A or 14B decision needed)

1. **Verify Segmented Sort impact:**
   ```python
   profile with segmented=[True, False] on room, bicycle, garden
   measure sort kernel wall time via CUDA events
   if no difference: remove from candidate list
   ```

2. **Complete room 30K training for tile20** to resolve training winner question:
   - If tile20 < tile32 in training time → strong adaptive argument
   - If tile20 ≈ tile32 → tile32 is safe default for room
   - If tile20 > tile32 → snapshot proxy is invalid for training

### Deferred

3. **Per-tile culling**: Only pursue if segmented sort is verified and adaptive tile selection is implemented. Requires modifying CUDA tile intersection kernel — non-trivial.

4. **Adaptive tile selection implementation**: Wait for full training results before deciding between Phase 14A (implementation) and Phase 14B (training characterization).
