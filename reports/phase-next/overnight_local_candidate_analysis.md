# Overnight Local Candidate Analysis

**Date:** 2026-10-19  
**Method:** Pure local source/dataflow/profiling analysis. No Web/arXiv/GitHub search, no CUDA implementation, no new benchmarks.  
**DSH Search Capability:** NOT AVAILABLE — all claims rely on local evidence.

---

## 0. Fixed State

| Candidate | Status | Rationale |
|:----------|:-------|:----------|
| **C1** | `FROZEN / ARCHIVED-PENDING` | Source audit complete. P5 CUDA verification BLOCKED by toolchain. Hypothesis open. |
| **C17-1** | `DROP` | Too high risk/cost. Memory overhead +176–204 MB. Per-tile overflow handling complex. Phase 17B proven correct but not implementable in current phase. |
| **C17-3** | `DROP` | Source audit (c17_3_source_audit.md) proves no baseline gap. All attributes already saved via ctx (756 KB, fits L2). Shared memory batch loading provides intra-tile reuse. Dominant cost is flatten_ids (704 MB) which C17-3 does not reduce. |
| **M1–M5** | Already characterized | No repeat needed. |

---

## 1. C17-2: Investigation

**WARNING:** Two distinct proposals with the same label exist in the history:

| Version | Target | Mechanism | Status |
|:--------|:-------|:----------|:-------|
| **C17-2 v1** (design.md, source_audit.md) | Rasterization forward kernel | Active pixel mask + early batch termination | Design completed, implementation not started |
| **C17-2 v2** (v2_design_review.md) | Sort pipeline | Two-phase sort: narrow-key CUB + per-tile depth sort | **GATE REJECTED** (G5: Phase B cost may exceed Phase A savings) |

`reports/phase-next/optimization_candidates.md` refers to C17-2 as two-phase sort (v2). The `can` was written before the detailed design review (2026-10-19) which rejected the candidate.

### 1.1 C17-2 v2: Why the Design Review Failed

**G5 Failure Analysis** — The v2_design_review.md established:

#### Phase A Savings

| Config | Baseline passes | Phase A passes | Saving |
|:-------|:---------------:|:--------------:|:------:|
| 1080p tile16 (46→14 bits) | 12 | 4 | 8 passes (67%) |
| Sort traffic reduction | 1.73 GB | 576 MB | **~1.15 GB saved** |
| Estimated sort time saved | ~5.5ms | ~1.8ms | **~3.7ms saved** |

Source: `v2_design_review.md` §2.2. Pass counts are calculated as `ceil(end_bit/4)` per CUB convention. These are estimates — CUB's actual RADIX_BITS is architecture-dependent (4-bit or 8-bit). Source only confirms `end_bit` reduction from 46→14 (for Phase A), not the actual pass count.

#### Phase B Cost (Bitonic Sort)

| P2 | Occupancy (blocks/SM) | Waves (8160 tiles) | Est. time |
|:--:|:---------------------:|:------------------:|:---------:|
| 64 | 64 | 3 | Negligible |
| 256 | 16 | 5 | Baseline |
| 1024 | 4 | 19 | **4× baseline** |
| 4096 | 1 | 76 | **16× baseline** |

Source: `v2_design_review.md` §3.7, §3.8. For tiles with P2≥1024 (n≥513, covering ~1200 tiles in bicycle t16), occupancy collapses to ≤4 blocks/SM, requiring 19+ waves. Bitonic sort O(n log² n) complexity per tile combines with wave overhead.

#### Net Sort Time (Estimated)

| Scenario | Phase A | Phase B | Total | vs Baseline (5.5ms) |
|:---------|:-------:|:-------:|:-----:|:-------------------:|
| Optimistic | 1.8ms | 1.5ms | 3.3ms | **1.67×** |
| Moderate | 1.8ms | 3.0ms | 4.8ms | **1.15×** |
| Pessimistic | 1.8ms | 5.0ms | 6.8ms | **0.81×** |

**Key uncertainty**: The **3× range** in Phase B estimates could not be resolved without actual CUDA measurement. Worse, the CUB segmented sort fallback (Approach 1 in §3.4) was shown to produce **identical total traffic** (864 MB vs baseline 864 MB) — demonstrating that the "elegant" reuse of CUB primitives yields zero net benefit.

**Verbatim from v2_design_review.md §8:**
> > **Phase B's per-tile bitonic sort for 6M+ items across 8160 variable-size tiles has a cost that can be as large as the CUB work it replaces, particularly when occupancy collapse for large tiles is accounted for.**

**Gate Verdict: NOT PASSED — REJECTED.**

### 1.2 C17-2 v1: Rasterization Early Batch Termination

**Status: DESIGN STAGE ONLY. Never implemented.**

| Dimension | Evidence |
|:----------|:---------|
| **Target** | `rasterize_to_pixels_3dgs_fwd_kernel` compositing loop in `RasterizeToPixels3DGSFwd.cu` |
| **Mechanism** | Shared-memory `active_set` (256-bit mask) + `active_count`. When pixel T drops below 1e-4, mark pixel done. When all 256 pixels done, skip remaining batches. |
| **Expected workload reduction** | 20–40% compositing loop reduction (bicycle t16: 35% est.) |
| **Why it might work** | Background pixels never saturate → prevent batch-level early exit. Academic estimate: mean saturation depth << max saturation depth for mixed foreground/background tiles. |
| **Forward impact** | Only changes forward kernel. Sort/offset unchanged. |
| **Backward impact** | `last_ids` unchanged (same T threshold). Backward none. |
| **Correctness guarantee** | Mathematical: T monotonically decreasing, 1e-4 threshold same as baseline. |

**Critical evidence gap**: The design estimates a **forward speedup of 1.09–1.18×** and **E2E training speedup of 1.03–1.08×** (source: `design.md` §4.2). However, Phase 8E per-kernel CUDA timing showed:

| Stage | % of Forward (tile16) | % of Forward (tile32) |
|:------|:--------------------:|:--------------------:|
| Rasterization | **< 1%** | **1.7%** |
| Intersect+Sort | 96.8% | 93.6% |

Source: `phase8e_per_kernel_forward_timing.md` §1.2, §1.3.

**This is FATAL to C17-2 v1's premise.** If rasterization is <1% of forward time, even a 100% improvement in the rasterization kernel would yield <1% forward speedup. The design document's estimates were based on source-level batch analysis (15.5× batch launch volume) that Phase 8E proved was irrelevant — the rasterization kernel is not the bottleneck.

### 1.3 Corrected C17-2 Status

**Both versions of C17-2 are currently non-viable:**

| Version | Status | Reason |
|:--------|:-------|:-------|
| **v1: Active pixel tracking** | **DROP** | Phase 8E falsified the premise: rasterization < 1% of forward time |
| **v2: Two-phase sort** | **DEFER** | G5 failure: Phase B cost uncertainty could negate Phase A savings. Requires prior-art check on hierarchical sorting + dedicated CUDA microbenchmark |

---

## 2. H6 Intersection Structure — Sub-Component Breakdown

Based on Phase 8D (`phase8d_forward_workload_analysis.md`) and Phase 8E (`phase8e_per_kernel_forward_timing.md`).

| Sub-component | Status | Evidence |
|:--------------|:-------|:---------|
| **H6-A: Intersection count (n_isects)** | **SUPPORTED** | Fixed 4.00× geometric ratio between tile16/tile32 across all 6 checkpoints. 145M→176M (t16) vs 36M→44M (t32). Source: `phase8d_forward_workload_analysis.md` §3.3 (M3). |
| **H6-B: Duplication** | **PARTIAL** | Each Gaussian-tile pair is unique. No redundant intersections. However, for large Gaussians covering 2160 tiles (mean), the same Gaussian index appears in flatten_ids 2160 times — this is NOT duplication in the intersection structure, but redundant loading in the batch/sort pipeline. Per-phase, the same flatten_id appears ~8160 times (100% tile occupancy). Source: `phase8e_per_kernel_forward_timing.md` §6 H6. |
| **H6-C: Representation** | **SUPPORTED** | 64-bit key encoding: 1-bit image_id + 13-bit tile_id + 32-bit depth (baseline). Each intersection costs 8+4 = 12 bytes for isect_ids + flatten_ids. Source: `v2_design_review.md` §1.2. |
| **H6-D: Unnecessary generation** | **NOT SUPPORTED** | No evidence of unnecessary intersections. All tile-Gaussian pairs represent real spatial overlap. Two-pass intersect only generates what's strictly needed. Source: `source_audit.md` §3.1. |
| **H6-E: Memory traffic** | **SUPPORTED** | Intersection arrays dominate memory: ~6-7 GB total for tile16 (isect_ids 1.4 GB + flatten_ids 0.7 GB + sorted copies + CUB temp ~3 GB). Intersect+sort consumes 93-97% of forward time due to this traffic. Source: `phase15_candidate_recon.md` §Q3. |
| **H6-F: Workload imbalance** | **SUPPORTED** | Per-tile Gaussian count varies from 0 to 14,174 (bicycle t32). Per-tile mean: 490 (room t16) to 751 (garden t16). Top-decile tiles have >2,515 Gaussians. Empty tiles: 0% (100% tile occupancy). Source: `design.md` §1 and `phase8d_forward_workload_analysis.md` §3.6. |

---

## 3. Full Candidate Matrix

### 3.1 All Candidates

| # | Candidate | Bottleneck Evidence (0–5) | Source Clarity (0–5) | Correctness Risk | Implementation Cost | Isolation | M1 Composability | Local Recommendation |
|:-:|:----------|:------------------------:|:--------------------:|:----------------:|:-------------------:|:---------|:-----------------|:---------------------|
| **1** | **C17-2 v2: Two-Phase Sort** | **5** — Sort is 46–88% of forward (Phase 8E). 12→4 CUB passes saved. Evidence: `phase8e_per_kernel_forward_timing.md` §2. | **5** — Full source trace in `v2_design_review.md`. Key layout, CUB API, offset kernel all documented. | LOW — Sort is `@torch.no_grad()`. Phase A+Phase B produces identical ordering (mathematically proven). | MEDIUM — New sort dispatch logic. Custom bitonic kernel or segmented sort wrapper. | ✅ Forward pixel-level allclose test. | Complementary — tile_size-agnostic. | **DEFER** — GATE FAILED. Phase B cost uncertainty prevents reliable speedup. See §1.1. |
| **2** | **C17-2 v1: Active Pixel Tracking** | **1** — Rasterization < 1% of forward (Phase 8E). The hypothesized workload (compositing loop) is NOT the bottleneck. Evidence: `phase8e_per_kernel_forward_timing.md` §1.2. | **5** — Complete design in `design.md`. Source audit identifies exact kernel lines. | VERY LOW — Mathematical guarantee (same T threshold). | LOW — Template parameter in single kernel file. | ✅ Deterministic tile workbench test. | Independent — rasterization optimization, orthogonal to sort. | **DROP** — Premise falsified by Phase 8E. Rasterization < 1% of forward time. |
| **3** | **C1: Depth Key Compression** | **4** — Sort is bottleneck. CUB end_bit reduction from 46→30 verified in source. Evidence: `c1_source_audit_final.md` §3.1. | **5** — Complete source audit: 5 lines changed, `end_bit` formula, offset shift verified. | LOW-MEDIUM — ~5–15% intra-tile ordering collisions. PSNR impact estimated < 0.01 dB (NOT measured on CUDA). | LOW — ~10 lines patch already written. | ✅ Patch ready at `patches/IntersectTile.c1.cu`. | Complementary — independent of tile_size. | **DEFER** — BLOCKED by CUDA toolchain. P5 verification failed (build incompatibility).  |
| **4** | **Segmented Sort (I>1)** | **2** — No evidence for multi-camera benefit. Single-camera benchmark: 1.9–4.5× SLOWER (Phase 14B). Evidence: `phase14b_sort_benchmark.md` §2. | **4** — Full source trace: 2 CUB paths, 4 code layers. Evidence: `phase14b_sorting_source_trace.md`. | MEDIUM — Different inter-image ordering. Pixel-identical per-image proven for I=1. | NEGLIGIBLE — Single boolean `segmented=True`. | ✅ Python-level flag comparison. | Unknown — not tested with M1. | **NEEDS PRIOR-ART CHECK** (for multi-camera only; single-camera: DROP) |
| **5** | **CUB Radix Sort Policy Tuning** | **2** — CUB dominates sort time. But default CUB policy is already architecture-optimized. Evidence: `optimization_candidates.md` §2. | **2** — CUB does not expose RADIX_BITS as runtime parameter. Would need custom compile or CUB fork. | LOW — CUB policy change doesn't affect output. | HIGH — Requires custom CUB build or compile-time policy override. | ❌ Cannot isolate without custom CUB build. | Complementary. | **DEFER** — Implementation cost too high for uncertain gain. Let CUB's built-in architecture selection work. |
| **6** | **Host Sync Elimination** | **3** — `.item<int64_t>()` sync is real: ~5–10µs per iteration. Pipeline serialization prevents CPU-GPU overlap. Evidence: `Intersect.cpp` line 68 (from `optimization_candidates.md` §2). | **4** — Source location identified: `Intersect.cpp:68`. Mechanism clear. | LOW — GPU math unchanged. Only allocation strategy changes. | LOW — Pre-allocated max-size buffers + CUDA event tracking. | ✅ Kernel launch latency comparison. | Independent. | **DEFER** — Gain too small (0.5–2% forward). Worth doing only as part of a larger pipeline refactor. |
| **7** | **Intersection Buffer Reuse (C3)** | **1** — PyTorch caching allocator already handles same-size allocations. n_isects varies ±20%, adding minor overhead. Evidence: `phase15_candidate_recon.md` §C3. | **3** — Buffer allocation in `Intersect.cpp`. | NONE — Buffer management only. | LOW — Pre-allocated max-size buffers with `.narrow()`. | ✅ Peak memory comparison. | Independent. | **DROP** — Gain < 1%. PyTorch caching allocator already mitigates. |
| **8** | **Warp-Atomic Gradient Accumulation (C4)** | **2** — AtomicAdd contention is real for large Gaussians but current per-warp reduction already provides 32× reduction. No CUDA profiler measurement available. Evidence: `phase15_candidate_recon.md` §C4. | **3** — Backward kernel lines 256–274 identified. Shared-memory accumulator approach documented. | LOW-MEDIUM — FP non-associativity of atomicAdd. Gradients acceptably different. | MEDIUM — Shared-memory gradient accumulator + flush kernel. | ✅ Gradient norm comparison. | Independent — backward optimization. | **DEFER** — Needs Nsight Compute profiling to validate contention hypothesis. |
| **9** | **Float16 Color/Opacity Storage (C5)** | **1** — Backward re-reads colors/opacities (756 KB total). Fits L2. Evidence: `phase15_candidate_recon.md` §C5. | **3** — Storage format change in backward kernel. | LOW — float16→float32 conversion in kernel. | MEDIUM — Type change propagates through forward+backward storage. | ✅ Gradient comparison. | Independent. | **DROP** — Arrays are 756 KB, fit in L2 cache. No bandwidth bottleneck. |

### 3.2 Already-Decided Candidates

| Candidate | Recommendation | Key Evidence |
|:----------|:---------------|:-------------|
| C17-1 (Tile-Local Queues) | **DROP** | Memory overhead +176–204 MB. CUDA complexity ~500+ lines across 6 files. Phase 17B failure analysis confirms. |
| M1 tile32 hybrid | **DROP** | Already characterized (Phase 12). Scene-dependent: room +58%, bicycle/garden −44%. |
| M2 packed/dense | **DROP** | Already characterized as training-neutral (Phase 10A: 1.03×). |
| M3 SH degree | **DROP** | <4% forward impact (Phase 12). Quality knob, not performance. |
| M4 radius_clip | **DROP** | Non-beneficial (Phase 11). |
| M5 eps2d | **DROP** | Non-beneficial (Phase 11). |

---

## 4. H6 Intersection Structure — Additional Analysis

### 4.1 Tile Count Distribution Data

From `design.md` §1 (Phase 17B capacity analysis):

| Scene | tile_size | per-tile mean | per-tile max | n_isects | Empty tiles |
|:------|:---------:|:-------------:|:------------:|:--------:|:-----------:|
| room | 16 | 490 | 2,996 | 2.27M | 0% |
| bicycle | 16 | 746 | 6,882 | 6.08M | 0% |
| garden | 16 | 751 | 4,361 | 6.13M | 0% |
| bicycle | 32 | 1,737 | 14,174 | — | 0% |

From `phase8d_forward_workload_analysis.md` §3.3 (room scene):

| Checkpoint | t16 n_isects | t32 n_isects | Ratio |
|:-----------|:------------:|:------------:|:-----:|
| iter5000 | 145,250,227 | 36,314,474 | 4.00× |
| iter30000 | 175,809,454 | 43,954,474 | 4.00× |

**100% tile occupancy**: Every Gaussian covers the entire screen in real trained scenes. tpg_median = total_tiles for both tile16 and tile32. This means there is NO spatial sparsity to exploit.

### 4.2 Intersection Growth Pattern

From Phase 8C/8D cross-check: n_isects grows 1.21× from iter5000→iter30000 (room), matching Gaussian count growth. The 4.00× tile16/tile32 ratio is perfectly geometric (tile_count_ratio = 8160/2040 = 4.00). No superlinear intersection growth.

The tile16 forward runtime growth (1.00×→9.64×) is NOT from intersection count growth (1.00×→1.21×). Per-intersection cost grows from 0.724 µs to 5.766 µs — a 7.96× increase that Phase 8E attributes to the sort kernel's linear scaling with sort input.

---

## 5. Source Code Mapping

| Candidate | File | Kernel/Function | Lines |
|:----------|:-----|:---------------|:------|
| C1 depth compression | `IntersectTile.cu` | `intersect_tile_kernel` (Pass 2) | 97, 100–103, 113 |
| C1 depth compression | `IntersectTile.cu` | `radix_sort_double_buffer` | 328 (end_bit) |
| C1 depth compression | `IntersectTile.cu` | `segmented_radix_sort_double_buffer` | 383 (end_bit) |
| C1 depth compression | `IntersectTile.cu` | `intersect_offset_kernel` | 233, 252 (shift) |
| C17-2 v2 (two-phase sort) | `IntersectTile.cu` | `radix_sort_double_buffer` | 296–339 |
| C17-2 v2 (two-phase sort) | `IntersectTile.cu` | `segmented_radix_sort_double_buffer` | 343–394 |
| C17-2 v2 (two-phase sort) | `IntersectTile.cu` | `intersect_offset_kernel` | 227 (baseline) |
| C17-2 v2 (two-phase sort) | `Intersect.cpp` | `intersect_tile()` | 119–148 |
| C17-2 v1 (active pixel) | `RasterizeToPixels3DGSFwd.cu` | `rasterize_to_pixels_3dgs_fwd_kernel` | Entire kernel |
| Segmented sort | `IntersectTile.cu` | `segmented_radix_sort_double_buffer` | 343–394 |
| Host sync elimination | `Intersect.cpp` | `intersect_tile()` | 68 |
| CUB policy tuning | `IntersectTile.cu` | `radix_sort_double_buffer` | 296–339 (CUB call) |
| Gradient reduction (C4) | `RasterizeToPixels3DGSBwd.cu` | `rasterize_to_pixels_3dgs_bwd_kernel` | 251–275 |
| Buffer reuse (C3) | `Intersect.cpp` | `intersect_tile()` | 96–97, 120–121 |
| Float16 storage (C5) | `RasterizeToPixels3DGSBwd.cu` | Parameters declaration | 22–25 |

All paths relative to `gsplat/cuda/csrc/` (installed gsplat v1.5.3).

---

## 6. Key Methodological Issue: Phase 8C vs Phase 8E Discrepancy

Phase 8C (monolithic `rasterization()` call) reported tile16 forward times of 331–1014ms with 9.64–25.51× tile16/tile32 ratios.

Phase 8E (decomposed CUDA wrapper functions) reported tile16 forward times of 122–200ms with 3.72–6.00× ratios.

**Root cause**: Phase 8C's monolithic call includes autograd graph construction and intermediate tensor allocation overhead that scales with problem size. Phase 8E's decomposed approach provides accurate kernel-level timing.

**Implication**: All Phase 8C-based workload analysis (Phase 8D hypothesis matrix H7 "Rasterization Batch Overhead") was based on inflated numbers. Phase 8E is the authoritative data source.

**For this analysis**: All bottleneck evidence relies on Phase 8E data. Phase 8C/8D numbers are cited only for historical context.

---

## 7. Composability Analysis with M1 (Tile Size)

| Candidate | M1 Composability | Explanation |
|:----------|:-----------------|:------------|
| C17-2 v2 Two-Phase Sort | **Complementary** | Tile-size agnostic. Phase A pass count depends only on bit width, not tile count. M1 tile32 produces 4× fewer intersections → 4× less work for both Phase A and Phase B. Ratio preserved. |
| C1 Depth Compression | **Complementary** | Independent of tile_size. End_bit reduction applies equally to both tile16 and tile32. |
| Segmented Sort | **Unknown** | Not tested with multiple tile sizes. Phase 14B only tested tile16, tile20, tile32 independently. |
| Host Sync Elimination | **Independent** | Not tile_size dependent. Affects the .item() call which runs identically for both sizes. |
| C4 Gradient Reduction | **Independent** | Backward optimization. Not affected by tile_size. |
| C3/C5 Buffer | **Independent** | Buffer management, not tile_size-specific. |

**No candidate is redundant with M1.** All optimizable candidates are either complementary or independent.

---

## 8. Local Recommendation Summary

| Decision | Candidates | Rationale |
|:---------|:-----------|:----------|
| **DROP** | C17-2 v1 (active pixel), C17-3 (bwd cache), C17-1 (tile queues), M2, M3, M4, M5, C3 (buffer reuse), C5 (float16) | Premise falsified, baseline gap proven nonexistent, or already characterized as non-beneficial. |
| **DEFER** | C17-2 v2 (two-phase sort), C1 (depth compression), CUB policy tuning, Host sync elimination, C4 (gradient reduction) | Promising but blocked by: G5 failure, CUDA toolchain, implementation cost, or insufficient evidence. Revisit when conditions change. |
| **NEEDS PRIOR-ART CHECK** | Segmented Sort (multi-camera I>1) | Single-camera tested and rejected. Multi-camera case untested. |

**No candidate receives "STRONG CANDIDATE FOR EXTERNAL PRIOR-ART CHECK"** — the top candidates either have fatal local evidence against them (C17-2 v1, C17-3) or are blocked by local constraints (C17-2 v2 GATE FAILURE, C1 BUILD BLOCKED).

---

## 9. Data Gaps

| Gap | Impact | What's Needed |
|:----|:-------|:--------------|
| Multi-camera (I>1) benchmark for segmented sort | Unknown | Single `segmented=True` benchmark with I=2, 4, 8. Currently zero evidence. |
| CUB RADIX_BITS policy for RTX 5070 sm_80 | 2× uncertainty in pass count | Nsight Compute or CUB source trace to determine actual pass count. |
| Per-tile depth saturation distribution | C17-2 v1 estimation error | Forward kernel debug counters (active pixels per tile over training). |
| Phase B CUDA microbenchmark | C17-2 v2 GATE FAILURE | Isolated bitonic sort timing on real tile distributions. |
| C1 sort wall time | C1 validation blocked | Working CUDA build on compatible toolchain. |

---

## 10. Key Open Questions for External Search

1. **Hierarchical/segmented sorting for differentiable 3DGS**: Is there published work on "first group by tile, then sort by depth" that could provide existing CUDA patterns or occupancy models?

2. **Sort key width reduction in 3DGS**: Beyond RoofGS (July 2025), are there newer papers on depth bit-width reduction?

3. **Multi-camera training optimizations**: Are there published benchmarks or training-time optimizations specifically for multi-view 3DGS training that use segmented sorting or tile-group-aware scheduling?

---

*End of Overnight Local Candidate Analysis*
