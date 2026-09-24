# Phase 13B — Candidate Reconnaissance Report

**Date:** 2026-08-28  
**Author:** DSH coding agent  
**Status:** COMPLETE (investigation only, no implementation)

---

## 1. Context

The expanded tile-size sweep (Phase 13B) has fully characterized the performance-quality landscape for tile_size ∈ {4, 8, 12, 16, 20, 24, 28, 32} across three diverse scenes. Key findings:
- All tile sizes are pixel-identical (quality-preserving)
- tile20 is the strongest universal forward candidate
- tile12 is best for garden forward, tile24 for bicycle forward
- tile20 dominates fwd+bwd for outdoor scenes

This report identifies promising directions for new CUDA/kernel/data-structure optimizations beyond tile-size tuning.

---

## 2. Candidate 1: Adaptive Tile-Size Selection

| Field | Value |
|:------|:------|
| **Name** | `adaptive_tile_size` |
| **Motivation** | Scene-dependent optimum confirmed: room prefers tile20, garden prefers tile12. A single fixed tile size leaves 10-15% performance on the table. |
| **Affected Stage** | Rasterization (tile generation + intersection) |
| **Expected Benefit** | 10-15% forward speedup over tile16; 5-10% over tile20 for mixed workloads |
| **Correctness Risk** | Low — tile size is purely computational, output is bit-identical |
| **Training Risk** | Low — same reasoning, no gradient path change |
| **Measurement Plan** | Implement workload feature observer (e.g., first-camera tpg_std), then select tile_size per scene. Validate against all 3 scenes. |
| **Priority** | HIGH — directly addresses the scene-dependence finding |

## 3. Candidate 2: Tile-Workload-Aware Intersection Sorting

| Field | Value |
|:------|:------|
| **Name** | `workload_aware_sorting` |
| **Motivation** | Current radix sort sorts all intersections globally (CUB). For large outdoor scenes, the sort dominates runtime. Tile-local or segmented sorting could be faster. |
| **Affected Stage** | Sorting (CUB DeviceRadixSort) |
| **Expected Benefit** | 20-40% sort speedup for high-intersection workloads; may reduce temp memory |
| **Correctness Risk** | Medium — sorting order must remain deterministic for gradient correctness |
| **Training Risk** | Medium — any change to gradient accumulation order affects bit-exactness |
| **Measurement Plan** | Implement segmented sort per tile (already partially supported with `segmented=True` flag). Compare sort-only timing. Verify pixel equivalence. |
| **Priority** | MEDIUM — segmented sort already exists as experimental flag |

## 4. Candidate 3: Adaptive Block/Grid Sizing

| Field | Value |
|:------|:------|
| **Name** | `adaptive_block_sizing` |
| **Motivation** | Current kernel uses `tile_size × tile_size` threads always. For sparse workloads (low Gaussians/tile), smaller blocks waste threads. For dense workloads, 1024-thread blocks may be suboptimal due to occupancy limits. |
| **Affected Stage** | Rasterization (kernel launch config) |
| **Expected Benefit** | 5-15% in sparse or dense extreme workloads |
| **Correctness Risk** | Low — block size doesn't affect computation |
| **Training Risk** | Low |
| **Measurement Plan** | Profile occupancy vs block size for different workload regimes. Decouple block size from tile size. |
| **Priority** | LOW — benefit is incremental, complexity is high |

## 5. Candidate 4: Visibility/Culling Optimization

| Field | Value |
|:------|:------|
| **Name** | `improved_visibility_culling` |
| **Motivation** | Current culling is radius-clip and frustum-based. For outdoor scenes, many Gaussians outside the view frustum still generate projection work. |
| **Affected Stage** | Projection (fully_fused_projection) |
| **Expected Benefit** | 10-30% for sparse outdoor scenes with many invisible Gaussians |
| **Correctness Risk** | Low — culling only affects which Gaussians enter the pipeline, not their computation |
| **Training Risk** | Low — non-differentiable culling doesn't affect gradients |
| **Measurement Plan** | Profile visible vs total Gaussians ratio. Implement frustum culling with bounding box hierarchy. |
| **Priority** | MEDIUM — visible ratio for outdoor scenes is only ~30% (1.8M/6.1M) |

## 6. Candidate 5: Memory Layout / Data Movement

| Field | Value |
|:------|:------|
| **Name** | `optimized_memory_layout` |
| **Motivation** | The packed path scatters/gathers Gaussian data. For scenes with high intersection counts, memory bandwidth may be a bottleneck. |
| **Affected Stage** | Rasterization (shared memory → global memory) |
| **Expected Benefit** | 5-20% depending on memory bottleneck |
| **Correctness Risk** | Low — same computation, different memory pattern |
| **Training Risk** | Low |
| **Measurement Plan** | Profile L1/L2 cache hit rates, memory throughput. Implement coalesced memory access patterns. |
| **Priority** | LOW — requires detailed profiling first |

## 7. Candidate 6: Backward Pass Optimization

| Field | Value |
|:------|:------|
| **Name** | `backward_reduction_optimization` |
| **Motivation** | Phase 8B showed backward pass is the dominant cost for tile16. For tile20+, backward becomes comparable to or less than forward (outdoor scenes). But for room tile16, backward is 2× forward. |
| **Affected Stage** | Backward rasterization (rasterize_to_pixels_bwd_kernel) |
| **Expected Benefit** | 20-40% backward speedup for high-density scenes |
| **Correctness Risk** | High — gradient computation must be numerically correct |
| **Training Risk** | High — any gradient change affects training dynamics |
| **Measurement Plan** | Profile backward pass scatter/atomic operations. Investigate gradient reduction fusion. |
| **Priority** | MEDIUM — high reward but high risk |

---

## 8. Optimization Priority Matrix

| Candidate | Expected Benefit | Correctness Risk | Training Risk | Implementation Cost | Priority |
|:----------|:----------------:|:----------------:|:-------------:|:-------------------:|:--------:|
| Adaptive tile_size | 10-15% | Low | Low | Medium | **HIGH** |
| Segmented sort | 20-40% (sort only) | Medium | Medium | Low | MEDIUM |
| Visibility culling | 10-30% | Low | Low | High | MEDIUM |
| Backward optimization | 20-40% | High | High | High | MEDIUM |
| Adaptive block size | 5-15% | Low | Low | Medium | LOW |
| Memory layout | 5-20% | Low | Low | High | LOW |

---

## 9. Recommendation

> **Start with adaptive tile-size selection (Candidate 1).** It has the highest benefit-to-risk ratio, directly addresses the scene-dependence finding, and requires no CUDA kernel changes — only a selection oracle + configuration API.

> **Next: investigate segmented sort (Candidate 2)** as it already exists in the codebase as `segmented=True`.

> **Defer: backward optimization and memory layout** until the adaptive tile-size baseline is established.

---

*Report generated 2026-08-28*
