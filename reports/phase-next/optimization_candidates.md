# Next-Phase Optimization Candidates

**Status:** Updated with overnight local candidate analysis (2026-10-19) — all C17-2 statuses revised based on v2_design_review GATE FAILURE and Phase 8E evidence

**Date:** 2026-10-19 (updated from 2026-09-06)

**Method:** All candidates assessed against existing evidence from Phases 8D, 8E, 14B, 15, C17A, v2_design_review, and c17_3_source_audit. Overnight analysis complete. No new profiling, literature search, or implementation was performed.

---

## 1. Existing Evidence Landscape

### Confirmed Forward Bottleneck (Phase 8E — RTX 5070, real room scene)

| Stage | % of Forward | Scales with tile_size? | Candidate addressed? |
|:------|:------------:|:----------------------|:---------------------|
| **intersect + sort** | **93–97%** | Yes (4× for tile16) | Sort optimization (C17-2) |
| └── CUB radix sort | **60–80%** | Yes (3.2–4.0×) | Sort key width (C1, BLOCKED) |
| offset | 3–4% | Yes (4×) | Implicit offset (C17-1, high risk) |
| projection | <1% | No | N/A |
| SH evaluation | <1% | No | N/A |
| rasterize | <1% | No | N/A |

### Confirmed Backward Pattern (Phase 15 — source audit)

- Backward re-reads all forward tensors from global memory (means2d, conics, colors, opacities)
- Warp-reduced atomicAdd for gradient accumulation (already optimized 32× by per-warp reduction)
- Projection VJP recomputes geometry (memory-optimal design)

### Blocked Measurements (Nsight Compute on WDDM)

- Warp execution efficiency, L2 cache hit rate, register spilling, warp stall reasons
- These would inform but are not required for the candidates below

---

## 2. Candidate Screening Table

All candidates are assessed without web/literature search. "Prior-art status" is based on source evidence from the existing codebase and previously reviewed papers (RoofGS, HiGS, FlashGS, Speedy-Splat, TC-GS).

| # | Candidate | Bottleneck | Existing Evidence | Expected Gain | Correctness Risk | Implementation Cost | Isolated Test? | Prior-Art Status | Recommendation |
|:-:|:----------|:-----------|:-----------------|:--------------|:-----------------|:-------------------|:---------------|:-----------------|:--------------|
| **1** | **Two-phase sort (C17-2 v2)** — narrow-key CUB for tile contiguity + per-tile local depth sort | CUB radix sort dominates forward (60–80%). Single global sort at ~70% of forward time. | Phase 8E: sort_est = 93.5ms (tile16), 29.1ms (tile32). Sort ratio ~4×. **v2_design_review**: GATE FAILED (G5). Phase B bitonic sort for large tiles (P2≥1024) shows occupancy collapse to ≤4 blocks/SM → 19+ waves → cost may equal or exceed Phase A savings. Segmented-CUB Phase B produces ZERO net traffic reduction. | Uncertain. Optimistic: 1.67× sort; Moderate: 1.15×; **Pessimistic: 0.81× (slowdown)**. Phase B cost cannot be bounded below baseline without CUDA microbenchmark. | **LOW** — mathematically proven ordering equivalence. Sort is `@torch.no_grad()`. | **MEDIUM** — modified sort dispatch. New Phase B kernel (bitonic or segmented). Offset kernel unchanged. | ✅ Forward pixel-level allclose test. | **NOT_EXISTING** — but **GATE FAILED locally** regardless of prior art. Phase B occupancy problem is the blocker. | **DEFER** — GATE FAILED (G5). Phase B cost uncertainty. Prior-art check needed for hierarchical sorting approaches. |
| **2** | **Backward metadata cache (C17-3)** — pre-load Gaussian attributes into compact cache on forward, reuse in backward | Backward re-reads means2d/conics/colors/opacities for every tile. Gaussians spanning T tiles → T redundant reads. | **SOURCE AUDIT (c17_3_source_audit.md):** No baseline gap exists. Four attribute arrays total 756 KB (fits in L2). ALL arrays already saved via `ctx.save_for_backward` (NOT recomputed). Shared memory batch loading provides intra-tile reuse. Dominant bandwidth cost is flatten_ids (704 MB), which C17-3 does not reduce. A software cache adds global reads (Phase 1) without reducing existing traffic. | No measurable gain. The "redundant" reads are served from L2 cache (756 KB < 3 MB L2 on RTX 5070). | **NONE** (no arithmetic change), but also NONE (no bottleneck removed). | **LOW-MEDIUM** (new kernel or kernel phase) for ZERO gain. | N/A — no target metric improves. | **NOT_EXISTING** but also NOT_NEEDED — hardware L2 cache + software shared memory batch cache already solve this. | **DROP** (source audit proves no baseline gap) |
| **3** | **Segmented sort (multi-camera, I>1)** — use gsplat's built-in `segmented=True` for multi-image batched renders | CUB radix sort on very large input. For multi-camera, single global sort includes all images, while segmented sort sorts per-image segments independently. | Phase 14B: segmented sort is 1.9–4.5× *slower* than global sort for I=1 (single camera). Source confirms segmented reduces end_bit from `32+tile_n_bits+image_n_bits` to `32+tile_n_bits` (narrower), but CUB segmented sort overhead outweighs benefit for single segment. **Not tested for I>1.** | For I>1 (e.g., 4–8 cameras): each segment is I× smaller → shorter sort per segment. Estimated 10–30% sort improvement for multi-camera. Zero for I=1. | **MEDIUM** — segmented sort produces different inter-image item ordering. Must verify pixel equivalence per-image. Phase 14B confirms segmented pass-through is pixel-identical for I=1. | **NEGLIGIBLE** — single boolean parameter `segmented=True`. No code changes. Already exists in gsplat 1.5.3. | ✅ Python-level `segmented=True/False` comparison. | **EXISTS_IN_BASELINE** — not a new mechanism. Evaluation gap: never profiled for multi-camera (I>1). | **PRIOR-ART CHECK REQUIRED** |
| **4** | **Host sync elimination** — replace `cumsum[-1].item()` CPU synchronization with CUDA-event async pipeline | CPU waits for GPU cumsum completion before launching Pass 2 and sort kernels. Pipeline underutilization. | Intersect.cpp line 68: `n_isects = cum_tiles_per_gauss[-1].item<int64_t>();` This is a blocking host-device sync. CPU cannot launch downstream kernels until cumsum finishes. | Reduces CPU-GPU round-trip latency by 5–20 µs per forward call. For 100K training iterations: 0.5–2 seconds saved. Negligible per-iteration, but measurable in microbenchmarks. | **LOW** — only change is how the kernel launch wait is structured. Sorting/timing unchanged. | **LOW** — modify Intersect.cpp to use pre-allocated max-size buffers + CUDA event for completion query. | ✅ Kernel launch latency comparison. No pixel change. | **STANDARD_TECHNIQUE** — CUDA stream synchronization. Not novel but applying it to gsplat's JIT allocation pattern is a valid optimization. | **DEFER** (gain too small to justify pipeline disruption) |
| **5** | **C1 depth key compression** — 16-bit depth in sort key instead of 32-bit | CUB radix sort uses 46-bit key (baseline). Reducing to 30-bit reduces sort passes proportional to bits-per-pass. | Source audit: C1 reduces end_bit by 16 bits (46→30 global, 45→29 segmented). Python smoke test: PSNR 34.6–39.6 dB (tile-level composite, not pixel-level — not formal CUDA evidence). **P5 CUDA verification BLOCKED** by toolchain incompatibility. | Unvalidated. Expectation: ~33% fewer CUB radix passes (assuming 4-bit RADIX_BITS, 12→8 passes). Total forward gain: unknown, depends on sort fraction of runtime. | **LOW** — 16-bit depth preserves IEEE 754 ordering for depth but reduces precision. Tile-level depth ordering within 16 bits is sufficient for 16×16 pixel regions. Forward-only, sort is `@torch.no_grad()`. | **NEGLIGIBLE** (patch already written). P5 build blocked by target environment. | ✅ Patch ready. Requires compatible CUDA env to execute. | **DIFFERENTIATED_FROM** RoofGS (2608.15785): no scene bounds, no division, no clamp, fixed 16-bit upper-half truncation. | **DEFER** (blocked by environment, re-open when compatible gsplat/CUDA/compiler available) |
| **6** | **Tile-local bounded queues (C17-1)** — replace global intersect materialization + CUB sort with per-tile pre-allocated buffers + local sort | Entire intersect+sort pipeline (~93–97% of forward). Also eliminates offset kernel (~3%). | Phase 8E: intersect+sort = 193.7ms tile16 (96.8% of forward). Offset = 5.58ms (2.8%). Rasterization = 0.44ms (<1%). Total transform cost: ~200ms = dominant. | Eliminates global sort entirely. Replaces with per-tile shared-memory sort. Estimated: 50–70% forward reduction (speculative without implementation data). | **HIGH** — overflow handling (tiles exceeding max_per_tile). Per-tile depth sort must match CUB ordering exactly. Changes the entire intersection data structure. Backward kernel must consume new format. | **HIGH** — new buffer manager, new intersect kernel, new per-tile sort kernel, overflow fallback, offset elimination, rasterizer adapter. | ✅ Forward: per-tile depth ordering comparison. Rendered image allclose. But many intermediate checks needed. | **NOT_EXISTING** in differentiable 3DGS. Related to HiGS macro-tile approach (inference-only). | **DROP** (too high risk/cost for current phase; revisit only if simpler candidates fail) |
| **7** | **M1 tile32 hybrid (scene-dependent dispatch)** — select tile_size per scene based on characteristics | No bottleneck addressed — it is an existing characterization result. | Phase 12: M1 tile32 validated: room +58%, bicycle/garden −44%. Scene-dependent effect. Cross-scene FALSIFIED. | Depends on scene. Conditional: 1.58× on room, 0.69× on bicycle. Not universal. | **NONE** — pixel-identical, gradient-identical, quality-identical (Phase 12). | **TRIVIAL** — config parameter change. | ✅ Already validated across 3 scenes. | **COMPLETED** — fully characterized in Phase 12. No remaining research gap unless new scene types are added. | **DROP** (already characterized; no remaining research questions) |
| **8** | **Packed/dense mode optimization** — reduce compaction overhead of packed mode for training | Packed mode's compaction overhead cancels its inference advantage during training. | Phase 10A: M2 packed/dense neutral (1.03× during training). Phase 9A: gradient correctness confirmed. | Not beneficial (1.03× training neutral). | — | — | — | — | **DROP** (already characterized as neutral) |
| **9** | **SH degree adaptive selection** — reduce SH degree during training iterations to save compute | SH evaluation in forward (<1% of forward time). SH backward/jacobian in backward (<2% of step). | Phase 12: M3 SH degree impact <4% of forward timing. SH degree is a quality knob, not a performance knob. | <4% forward. Negligible training impact. | **LOW** — different degrees produce different quality outputs (expected). | **LOW** — parameter change. | ✅ Already validated across 3 scenes. | **COMPLETED** — fully characterized. Not a performance optimization. | **DROP** (not a performance knob) |
| **10** | **Radix sort internal policy tuning** — CUB RADIX_BITS parameter (4 vs 8 bits per pass) | CUB radix sort is dominant cost (60–80% of forward). Default CUB policy may not be optimal for large inputs on Ampere. | Phase 8E: sort = 93.5ms (tile16), 29.1ms (tile32). CUB selects RADIX_BITS based on architecture. On sm_80 (A100), CUB uses 4-bit or 8-bit passes. Source only: cannot determine without running. | Speculative. If 8-bit passes reduce passes by 2×, sort time could drop proportionally. Estimated 20–40% sort reduction. | **LOW** — CUB policy tuning does not change algorithm output. Same sort, different internal passes. | **HIGH** — requires custom CUB compilation or policy override. CUB does not expose RADIX_BITS as a runtime parameter. Would need local CUB fork or compile-time policy. | ❌ Cannot isolate without custom CUB build. | **STANDARD** — CUB policy tuning is known. Specific optimal policy for gsplat workload on sm_80 is unknown. | **PRIOR-ART CHECK REQUIRED** |

---

## 3. Critical Update: C17-2 Status Change

### Overnight Analysis (2026-10-19) Conclusion

**C17-2 v2 (Two-Phase Sort)**: **GATE FAILED** — G5 (Phase B cost does not exceed Phase A savings) could not be satisfied.

The detailed design review (`reports/phase-c17-c2/v2_design_review.md`) showed:
- Phase A saves ~8 CUB passes (67% reduction in sort passes)
- Phase B (per-tile bitonic sort) for large tiles (P2≥1024, n≥513) suffers occupancy collapse → ≤4 blocks/SM → 19–38 waves
- The elegant CUB segmented-sort Phase B produces **identical total traffic** as baseline (12 passes regardless)
- Net sort time change: **0.81×–1.67×**, with pessimistic case being a **slowdown**

**C17-2 v1 (Active Pixel Tracking)**: **DROP** — Phase 8E falsified the premise. Rasterization takes <1% of forward time.

### Impact on Recommendation
- **Removed from PROTOTYPE** — both versions are non-viable with current evidence
- C17-2 v2 moved to **DEFER** — requires a solution to the Phase B occupancy problem before reconsideration

See `reports/phase-next/overnight_local_candidate_analysis.md` for full analysis.

---

## 4. Candidate Recommendation Summary

### PROTOTYPE candidates (proceed to implementation)

**NONE** — C17-2 v2 (previously the only PROTOTYPE) was GATE FAILED by design review. No candidate currently qualifies.

### PRIOR-ART CHECK candidates (need external literature before proceeding)

| Candidate | What to Check |
|:----------|:--------------|
| Segmented sort (multi-camera, I>1) | Is multi-camera segmented sort documented anywhere? What speedup on multi-view training? |
| CUB radix sort policy tuning | What is the optimal RADIX_BITS for sm_80 with 175M elements? Does CUB choose the right policy? |
| C1 depth compression | Any new papers on depth bit-width reduction for 3DGS sorting since RoofGS (July 2025)? |
| C17-2 v2 two-phase sort | Any published work on per-tile depth sort after tile-grouping for 3DGS? (Prior-art check is needed before the candidate can even be re-evaluated.) |

### DEFER candidates

| Candidate | Defer Until |
|:----------|:------------|
| C1 depth key compression | Compatible CUDA/gsplat/compiler environment is available |
| Host sync elimination | A specific microbenchmark shows the .item() sync is a measurable bottleneck in training throughput |

### DROP candidates

| Candidate | Reason |
|:----------|:-------|
| C17-1 Tile-Local Bounded Queues | Too high risk/cost for current phase. Revisit only if C17-2 prototype shows strong per-tile sort benefit. |
| C17-3 Backward Metadata Cache | **Source audit (c17_3_source_audit.md) confirms no baseline gap.** Four attribute arrays total 756 KB (fits L2). ALL already saved via ctx. Shared memory batch loading provides intra-tile reuse. Software cache adds complexity for zero gain. |
| M1 tile32 hybrid | Already fully characterized. No new research questions. |
| M2 packed/dense | Already fully characterized as neutral. |
| M3 SH degree | Already characterized as non-performance. |
| M4 radius_clip | Already characterized as non-beneficial. |
| M5 eps2d | Already characterized as non-beneficial. |

---

## 5. Next Steps

**NO PROTOTYPE CANDIDATE CURRENTLY VIABLE.**

The overnight analysis (2026-10-19) found that:

1. **C17-2 v2 (Two-Phase Sort)** — the only candidate that reached PROTOTYPE status — was **GATE FAILED** by its own design review (G5: Phase B cost uncertainty).
2. **C17-2 v1 (Active Pixel Tracking)** was premised on rasterization being a significant bottleneck. Phase 8E falsified this (rasterization < 1% of forward time).
3. **C1 (Depth Compression)** remains BLOCKED by CUDA toolchain.
4. **C17-3 (Backward Cache)** was DROPPED by source audit (no baseline gap).
5. **Segmented Sort** is 1.9–4.5× slower for single-camera and untested for multi-camera.

### Recommendation

**Do not implement any renderer optimization at this stage.** Instead:

1. **External prior-art search** (ChatGPT) for C17-2, C1, and segmented sort — the three candidates in the `external_research_queue.md`
2. **Multi-camera segmented sort benchmark** if multi-view training is in scope (I>1)
3. **Return to renderer optimization** only when a candidate passes both (a) prior-art check AND (b) local evidence gate

The existing renderer operates at near-peak efficiency for the current algorithmic approach. No simple code change will produce a breakthrough improvement.

---

## 6. Reference: C17-2 v2 Patch Architecture (For Future Reference)

If C17-2 v2 is revisited, the following architecture was identified:

1. **Phase A**: narrow-key CUB sort using `begin_bit=32, end_bit=32+tile_n_bits+image_n_bits` on the existing key format (depth in lower 32 bits). This sorts only the tile-grouping bits.
2. **Phase B**: per-tile depth sort (bitonic in shared memory or CUB segmented sort). The offset kernel runs AFTER Phase B.
3. **Offset kernel**: unchanged — reads the final depth-sorted `isect_ids`.

**Blocking issue**: Phase B cost for large tiles (P2≥1024) is the unresolved problem. Three approaches were evaluated:
- **Bitonic sort (Approach 2)**: Occupancy collapse, 3×–5× variance in cost estimate
- **Segmented CUB sort (Approach 1)**: Zero net traffic reduction (12 passes total same as baseline)
- **Warp-level hybrid (Approach 3)**: Not analyzed in detail

No implementation should proceed until the Phase B cost is bounded below the Phase A savings through a CUDA microbenchmark.
