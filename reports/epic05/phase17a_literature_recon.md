# Phase 17A — Literature / Source Reconnaissance

**Date:** 2026-10-17
**Scope:** Existing sorting/tile visibility approaches in the 3DGS ecosystem that relate to intersection materialization, sorting, and tile-level ordering.

---

## 1. gsplat (nerfstudio-project/gsplat) — Main Pipeline

**Source location:** `gsplat/cuda/csrc/IntersectTile.cu`

### Current approach
- Two-pass intersection: count → cumsum → materialize → global radix sort
- CUB `DeviceRadixSort::SortPairs` over full 64-bit key
- Key encoding: `image_id | tile_id | depth` — depth is full float32 bitcast
- Offset kernel infers tile boundaries from sorted key

### Existing sort variant: segmented sort
- `segmented=True` → CUB `DeviceSegmentedRadixSort::SortPairs`
- Per-image segments (I segments)
- Narrower key: `tile_id | depth` (no image_id in sort range)
- **Phase 14B finding**: 1.9–4.5× slower than global sort
- **gsplat docstring warning**: "additional global memory access is needed, which results in slower overall performance"

### Assessment
**STATUS: CURRENT. NOT A NOVEL CANDIDATE.**
Segmented sort is already implemented, benchmarked, and found non-beneficial.

---

## 2. gsplat (experimental) — HiGS / GaussianInferenceRenderer

**Source location:** `gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/SegmentedSort.cu`

### Approach
Custom four-tier cascaded block-level segmented sort (~1120 lines):
- Block sort with 8-bit radix (instead of CUB's 4-bit)
- Four cascade tiers: CTA sizes 64, 128, 256, 512
- Each tier has a minimum segment size threshold
- Merge kernel combines sorted segments (ping-pong buffers)
- Designed for many segments (macro-tiles of 8×8 Gaussians)

### Key parameters
- `BLOCK_SORT_ITEMS_PER_THREAD = 16`
- `BLOCK_SORT_RADIX_BITS = 8` (vs CUB's default 4)
- `MERGE_CTA_SIZE = 256`, `MERGE_ITEMS_PER_THREAD = 15`

### Integration
- **NOT connected to main `isect_tiles` pipeline**
- Used internally by `GaussianInferenceRenderer.render()`
- `GaussianInferenceRenderer` is **inference-only** (no backward pass)

### Assessment
**STATUS: EXISTING. NOT APPLICABLE TO DIFFERENTIABLE TRAINING.**
The HiGS sort path is not differentiable. Its architecture reveals that a custom tile-level sort can be efficient at inference, but the cost of maintaining backward gradient paths is non-trivial.

---

## 3. diff-gaussian-rasterization (graphdeco-inria)

### Approach
Original 3DGS rasterization (Kerbl et al. 2023):
- Two-pass intersection: count → prefix sum → materialize
- Sorting: single global radix sort (CUB)
- Same fundamental architecture as gsplat

### Differences from gsplat
- No `segmented` option
- No `packed` mode  
- Single-camera only (no image_id in key)
- Sort key: `tile_id | depth` (no image_id)

### Assessment
**STATUS: SAME ARCHITECTURE. NOT A NOVEL DIRECTION.**
The original paper's approach is the basis for gsplat. No structural innovation in sorting.

---

## 4. Taming 3DGS (taming-3dgs)

### Known details
- Aims to reduce Gaussian count through pruning and merging
- Specifically targets the intersection count problem
- After aggressive Gaussian reduction, the sorting workload drops proportionally

### Assessment
**STATUS: DIFFERENT PROBLEM DOMAIN.**
Taming 3DGS reduces intersection count at the Gaussian representation level (fewer Gaussians → fewer intersections). This is orthogonal to sorting algorithm optimization. The current study is about sorting/dataflow optimization at fixed Gaussian count.

---

## 5. Speedy-Splat / Fast-Gauss / FlashGS

### Speedy-Splat
- Same pipeline architecture (project → intersect → sort → rasterize)
- Optimizes projection and rasterization kernels via CUDA tuning
- **No reported innovation in sorting**

### FlashGS
- Shared-memory optimizations for rasterization
- Tile-level work scheduling improvements
- **No reported sorting pipeline changes**

### Assessment
**STATUS: NOT APPLICABLE TO SORTING.**
These projects optimize the projection/rasterization kernels, not the intersect/sort pipeline.

---

## 6. HiGS (High-Performance Gaussian Splatting)

### Source location
`gsplat/experimental/` — bundled with gsplat

### Key architectural difference
- **Macro-tiles** (8×8 Gaussians per macro-tile): Coarse-level Gaussian-to-tile assignment
- Fine tiles for rasterization
- Custom four-tier cascade segmented sort (see Section 2)
- Macro-tile intersection reduces the number of segments for sorting

### Why it's faster at inference
1. Fewer intersection comparisons per Gaussian
2. Custom sort tuned for many small segments
3. Pre-computed Gaussian metadata (means, covars, SH) stored in efficient format

### Why it's NOT applicable to differentiable training
1. **No backward pass** — the kernel intentionally skips gradient computation
2. Differentiable rasterization requires all intersection metadata for gradient flow
3. The cascade sort's non-deterministic merge patterns would complicate autograd

### Assessment
**STATUS: EXISTING. INFERENCE-ONLY. KEY ARCHITECTURAL REFERENCE.**

---

## 7. 3DGUT (3D Gaussian for Unconstrained Techniques)

### Known details
- Adds camera distortion and rolling shutter support to gsplat
- Built on `with_ut=True` (Unscented Transform) path
- Uses the same `isect_tiles` → sort → offset → rasterize pipeline
- **No sorting innovation**

### Assessment
**STATUS: SAME PIPELINE. NOT A SOURCE OF NOVEL SORTING IDEAS.**

---

## 8. TC-GS (Tensor Core 3D Gaussian Splatting)

### Known details
- Maps alpha evaluation to Tensor Core matrix operations
- Does NOT change the intersect/sort pipeline
- Sorting is pre-Tensor Core: same global radix sort

### Assessment
**STATUS: SAME SORTING. APPLIES AFTER SORT. NOT APPLICABLE.**

---

## 9. Other CUDA Radix Sort Optimizations (General)

### Key observations
- CUB `DeviceRadixSort` is already highly optimized for NVIDIA GPUs
- CUB's 4-bit radix is tuned for occupancy and shared memory utilization
- 8-bit radix (as in HiGS) can reduce passes but increases shared memory per block
- CUB uses a different internal strategy: it computes histograms in tiles, then re-scans

### Alternative algorithms
- **Thrust sort**: Slower than CUB for large arrays on modern GPUs
- **Bitonic sort**: O(n log² n) vs radix O(n×bits). Not competitive at scale
- **Merge sort**: Better for segmented data but higher constant factors
- **Sample sort**: Better cache behavior but complex to implement

### Assessment
**STATUS: GENERAL KNOWLEDGE. CUB IS ALREADY NEAR-OPTIMAL FOR GLOBAL RADIX SORT ON GPU.**

---

## 10. What's NOT in Any Existing gsplat/Renderer

After exhaustive source checking, the following approaches are **NOT implemented** in any differentiable 3DGS renderer:

### Candidate Spaces (Unoccupied)

| Approach | In any 3DGS renderer? | In differentiable one? |
|----------|----------------------|----------------------|
| Tile-local bucketization (write directly to per-tile arrays) | NO | NO |
| Hierarchical binning (coarse tile group → fine tile) | NO | NO |
| Persistent thread-block tile assignment (GPU workgroup per tile) | Only in rasterization | Not for intersection |
| Bounded per-tile queues with overflow | NO | NO |
| On-the-fly tile binning (no materialization before sort) | NO | NO |
| Two-level indexing (tile group → tile → depth sort) | NO | NO |
| Depth-only partial sort within tile buckets | NO | NO |
| Forward metadata reuse in backward (skip re-loading attributes) | NO | NO |
| Coalesced backward traversal (reuse tile-group loading) | NO | NO |
| Hybrid sort: tile bucket for membership + per-bucket depth sort | NO | NO |
| Redundant intersection elimination (before materialization) | NO | NO |
| Occlusion-aware intersection filtering | NO | NO |
| Attribute compression in sort values (shrink flatten_ids) | NO | NO |
| Key-value fusion (pack key and value into one word) | NO | NO |

---

## 11. Summary: Existing vs Novel

| Technique | Existing in gsplat? | Differentiable? | Viable for training? |
|-----------|-------------------|----------------|---------------------|
| Global radix sort (CUB) | YES (default) | YES | YES (current) |
| Per-image segmented sort (CUB) | YES (segmented=True) | YES | NO (slower) |
| Macro-tile sort (HiGS cascade) | YES (experimental) | NO | NO |
| Depth key compression (C1) | YES (tested) | YES | NO (<1% benefit) |
| Tile-local bucketization | NO | — | POTENTIAL |
| Hierarchical binning | NO | — | POTENTIAL |
| Bounded tile queues | NO | — | POTENTIAL |
| Forward metadata reuse | NO | — | POTENTIAL |
| Coalesced backward | NO | — | POTENTIAL |
| Redundant intersection elimination | NO | — | POTENTIAL |

**Conclusion**: There are genuinely unoccupied candidate spaces in the differentiable 3DGS renderer ecosystem. No existing differentiable renderer has explored tile-local bucketization, hierarchical binning, or forward metadata reuse for backward gradient computation.
