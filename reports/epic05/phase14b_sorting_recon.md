# Phase 14B — Sorting Optimization Reconnaissance

**Date:** 2026-09-23
**Author:** DSH coding agent
**Source:** gsplat v1.5.3 (installed at `C:\Users\36570\miniconda3\Lib\site-packages\gsplat`)

---

## 1. Executive Summary

The `segmented` sort flag (`sort=True, segmented=True` in `isect_tiles()`) **is already implemented** using CUB's `cub::DeviceSegmentedRadixSort::SortPairs`. The default path uses global `cub::DeviceRadixSort::SortPairs` over the full 64-bit key. Segmented sort is a **fully independent, single-boolean toggle** at every stack layer.

The key question is whether the segmented sort reduces end-to-end forward+backward time in the renderer.

---

## 2. Ten Questions — Answers

### Q1: Segmented sort — is it already implemented?

**YES — in two places.**

**Main pipeline:** CUB `DeviceSegmentedRadixSort::SortPairs` in `IntersectTile.cu:343–394`

The function `segmented_radix_sort_double_buffer()` wraps CUB and sorts per-image segments independently, sorting only `32 + tile_n_bits` bits (depth + tile_id) instead of the full 64 bits.

**Experimental/HiGS path:** Custom block-level segmented sort in `SegmentedSort.cu` (~1120 lines). This is a high-performance four-tier cascaded segmented sort. **It is NOT connected to the main `isect_tiles` pipeline** — it is used internally by the HiGS `GaussianInferenceRenderer`, which explicitly rejects `segmented` as an unsupported kwarg.

### Q2: Default path — does it really use segmented sort?

**NO.** Default `sort=True, segmented=False` uses global CUB radix sort:
```cpp
// Intersect.cpp:122-144
if (segmented)
    segmented_radix_sort_double_buffer(...);  // per-image segments
else
    radix_sort_double_buffer(...);            // global over all n_isects
```

### Q3: Segmented vs global — what are the differences?

| Aspect | Global (default) | Segmented |
|--------|-----------------|-----------|
| **CUB primitive** | `DeviceRadixSort::SortPairs` | `DeviceSegmentedRadixSort::SortPairs` |
| **Sort key bits** | `0..32+tile_n_bits+image_n_bits` (full ~42-48 bits) | `0..32+tile_n_bits` (depth+tile only) |
| **Segments** | 1 (all items) | I (one per image) |
| **Preprocessing** | None | O(I) cumsum of per-image intersections |

### Q4: Input/output — still same n_isects?

**YES.** The two-pass intersect kernel computes intersections identically regardless of sort selection. Sort is a post-processing step on same `isect_ids[n_isects]` and `flatten_ids[n_isects]` arrays.

### Q5: Does it reduce global sorting work?

**YES — two mechanisms:**

1. **Narrower key width:** Segmented sort only sorts `32 + tile_n_bits` bits per pass vs full `32 + tile_n_bits + image_n_bits` for global. CUB radix sort work scales with key width.

2. **Smaller sub-sorts:** Each segment ~ `n_isects / I` items. For single-camera training (I=1), segments degenerate to single segment → **no sub-sort benefit**. For multi-camera (I>1), each segment is smaller → radix sort cost per segment is O(S_i * bits).

### Q6: Does it add extra preprocessing?

**YES — trivial O(I) cost:**
```cpp
offsets = at::cumsum(at::sum(tiles_per_gauss, -1).view({-1}), 0);
offsets = at::cat({at::tensor({0}), offsets});
```
This computes per-image boundaries in the intersection array for CUB's offset format. Negligible for I ~ 1-100.

### Q7: Does it support backward?

**YES.** `isect_tiles()` is `@torch.no_grad()`. The tensors are saved in `meta` and consumed by `rasterize_to_pixels()` backward pass via `_RasterizeToPixels.backward()`. Both sort paths produce identical `isect_offsets` and same depth ordering per (image, tile) pair.

### Q8: Does it change output ordering?

**YES — across images only.** Within the same image and same tile, both produce identical depth-sorted order. Across images, the segmented sort concatenates per-image segments, while the global sort interleaves by full 64-bit key. The `isect_offset_kernel` only cares about contiguous (image, tile) groups, so ordering differences across images don't affect the offset structure.

### Q9: Is pixel equivalence maintained?

**YES.** The `isect_offset_kernel` groups by `(image_id, tile_id)` — both sort methods guarantee all items for a given (image, tile) are contiguous and depth-sorted. Therefore `isect_offsets` and per-pixel rasterization are identical. `torch.allclose(expected, actual, atol=0, rtol=0)` should pass.

### Q10: Can it be independently ablated?

**YES — single boolean flag `segmented=True`** propagated through:
- `rasterization(..., segmented=True)` — high-level API in `rendering.py:58,641`
- `isect_tiles(sort=True, segmented=True)` — low-level API in `_wrapper.py:451`
- C++ dispatch in `Intersect.cpp:122-133`
- CUDA sort selection in `IntersectTile.cu:343-394`

No other code path depends on the flag. Can be ablated with `timeit` wrapping a single `isect_tiles()` call.

---

## 3. Benchmark Design

### Candidates

| Configuration | `segmented` | Expected effect |
|:--------------|:-----------:|:----------------|
| baseline | `False` | Global radix sort (default) |
| optimized | `True` | Per-image segmented radix sort |

### Test Scenes

| Scene | Gs | Intersections (tile16) | Intersections (tile20) |
|:------|:--:|:---------------------:|:---------------------:|
| room (training iter 30000) | 1,146,273 | 3,219,239 | 2,373,219 |
| bicycle | 6,130,970 | 6,083,523 | 4,979,642 |
| garden | 5,800,000 | 6,127,280 | 5,129,564 |

### Metrics

| Metric | Collection method | Priority |
|:-------|:-----------------:|:--------:|
| Sort time (ms) | CUDA event around `isect_tiles()` | P0 |
| Total intersect+sort time | CUDA event around full isect pipeline | P0 |
| Forward total | Full forward pass timing | P0 |
| Backward total | Full backward pass timing | P0 |
| n_isects | From output sizes | P0 |
| Temporary memory | `torch.cuda.max_memory_allocated()` | P0 |
| Pixel correctness | `torch.allclose()` vs baseline | P0 |
| Gradient sanity | Gradient norm comparison | P0 |

### Benchmark Protocol

1. Use frozen snapshots (room iter30000, bicycle SfM init, garden SfM init)
2. Warmup: 5 forward passes (discard)
3. Benchmark: 50 forward+backward passes
4. Record: sort time, fwd total, bwd total, n_isects, peak memory
5. Verify: pixel equivalence with `torch.allclose(rendered, baseline)`
6. Verify: gradient equivalence with gradient norm comparison

---

## 4. Expected Outcomes

| Scenario | Probability | Action |
|:---------|:-----------:|:-------|
| A. Sort 2-5× faster, total fwd+bwd ≥5% faster | ~30% | Proceed to 500-step training sanity |
| B. Sort faster but total time neutral | ~40% | Document as potential composability candidate |
| C. No benefit or slower | ~20% | Abandon, move to visibility/culling recon |
| D. Correctness issue | ~10% | Investigate and file bug, move on |

**Note:** For single-camera training (I=1), segments degenerate to a single segment. The only benefit is the narrower key width (`32+tn` vs `32+tn+in` bits). For multi-camera training, per-image sub-sorting is an additional benefit.

---

## 5. Visibility/Culling Reconnaissance (Phase 14C — Preview)

| Existing culling | Status |
|:-----------------|:-------|
| Frustum culling | In projection kernel (radius > 0 check) |
| Radius culling | Via `radius_clip` parameter (M4 — FALSIFIED for training benefit) |
| Visibility filtering | Inherent in packed mode (compaction during projection) |
| Packed projection | Already default — invisible Gs excluded before rasterization |

**Key finding:** The current renderer already performs frustum and radius culling during projection. The packed projection (`packed=True`) is the default and filters Gaussians with zero radius (outside frustum). M4 (radius_clip) was FALSIFIED in Phase 11 as providing <1% workload reduction at quality-preserving thresholds.

No new visibility/culling optimization exists that isn't already default or already FALSIFIED. Full reconnaissance in separate document.

---

## 6. Files Created

- `reports/epic05/phase14b_sorting_recon.md` (this file)
- `reports/epic05/phase14b_sort_benchmark.md` (benchmark results — pending execution)
- `results/epic05/phase14b_sorting.json` (benchmark data — pending execution)
- `reports/epic05/phase14_visibility_recon.md` (visibility culling assessment)
