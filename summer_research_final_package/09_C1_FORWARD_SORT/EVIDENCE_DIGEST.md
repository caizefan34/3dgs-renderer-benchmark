# S4 — Forward-Pass / Sorting Optimization Candidates Digest

**Scope:** C1 (depth-key compression), segmented sort, C17-1 tile-local queues, C17-2/3, C18-1, C43, C26/G2, visibility/culling, opacity-skip. All claims cite file paths. Project period 2026-07..2026-09; renderer = gsplat v1.5.3.

---

## 1. C1 — Depth-Key Compression (32→16-bit depth, radix-pass reduction)

**Formulation.** The baseline 64-bit intersection sort key packs `image_id | tile_id | depth` where depth is the full 32-bit float32 bitcast to uint32 (`*(int32_t*)&depth`) occupying bits [31:0]; `tile_id<<32`; `image_id<<(32+tile_n_bits)`. Sort range `end_bit = 32 + tile_n_bits + image_n_bits` (=46 for 1080p/tile16/I=1) (`reports/epic05/phase16_c1_key_encoding.md`; `reports/phase-c17-c2/c1_source_audit_final.md` §1–3).

C1 truncates depth to the **upper 16 bits** (`depth_id_enc >> 16`), reorders fields to `depth_upper | (tile_id<<16) | iid_enc`, shifts image_id to `<<(16+tile_n_bits)`, and reduces `end_bit` to `16 + tile_n_bits + image_n_bits` (=30) (`patches/IntersectTile.c1.cu` lines 97,103,113,328; `reports/phase-c17-c2/c1_source_audit_final.md` §1.2,§3.1). The offset kernel shift changes `>>32`→`>>16` (line 233). **Five-line CUDA patch**, no new kernels, no extra memory (`reports/phase-c17-c2/c1_sort_mechanism_verification.md` Finding 8; `reports/phase-c17-c2/c1_source_audit_final.md` §6).

**Expected benefit.** End_bit drops 46→30 (global) / 45→29 (segmented) — a 16-bit reduction (`reports/phase-c17-c2/c1_source_audit_final.md` §7). Early reports state this yields **12→8 CUB radix passes** assuming `RADIX_BITS=4` (`reports/epic05/phase16_c1_correctness.md`; `reports/epic05/phase16_c1_sort_benchmark.md`). Sort memory traffic falls 144N→96N bytes (16.8GB→11.2GB at 58M isects) (`reports/epic05/phase16_c1_sort_benchmark.md`). Estimated E2E speedup **+9.4–9.8%** on A100 via sort-isolation (sort = 28–32% of forward) (`reports/a100_validation/candidate_parallel_screening.md` C1; `reports/a100_validation/track_c_c1_c43_validation.md` §1.3).

**⚠️ Correction on pass count.** `c1_source_audit_final.md` §7 explicitly walks this back: the actual CUB pass count is a **runtime policy** (`RADIX_BITS` commonly 4 or 8, architecture-dependent), not source-determined. The source only proves the 16-bit `end_bit` reduction; "12→8 passes" is an assumption, not a verified fact. The post-hoc correction is appended to `c1_sort_mechanism_verification.md` (line 570) and supersedes the earlier unverified claim.

**Measured benefit.**
- **RTX 5070 Laptop (measured, C1 built):** <1% speedup. 100K Gs −7.4% (noise), 200K Gs +0.8%, 500K Gs (58M isects) **+0.8%** (32.05ms→31.78ms). Root cause: rasterization dominates; sort is not the bottleneck on this GPU; CUB 4-bit passes are already efficient (`reports/epic05/phase16_c1_sort_benchmark.md`; `reports/epic05/phase16_c1_correctness.md` §5). Verdict: **CORRECT but NOT BENEFICIAL**; "do NOT proceed to composability."
- **A100 (estimation only — build failed):** +9.8% estimated from sort-isolation × 33% pass saving. The C1 patch could not be compiled on the A100 (system `nvcc 11.5` incompatible with gsplat's C++17/PyTorch 2.7.1 JIT) (`reports/a100_validation/track_c_c1_c43_validation.md` §1.2; `reports/phase-c17-c2/c1_p5_sort_performance.md`). P5 validation = **NEGATIVE (validation attempt) / INCONCLUSIVE overall** — no runtime claim can be made; C1 frozen pending a compatible toolchain (`reports/phase-c17-c2/c1_p5_sort_performance.md` §6).

**Correctness.** PROVEN at source level: 16-bit depth preserves IEEE-754 ordering for positive floats (right-shift is monotonic) → **zero inversions**; offset kernel unaffected (reads only tile_id+image_id); backward reads `flatten_ids` not `isect_ids` → unaffected; tie rate 0.01–0.02% within collision groups (per-tile tie rate 0.06% mean simulated, 13–20% real scenes) (`reports/epic05/phase16_c1_correctness.md` §3; `reports/phase-c17-c2/c1_source_audit_final.md` §5; `reports/a100_validation/track_c_c1_c43_validation.md` §1.4). Synthetic forward test: no NaN/Inf (`reports/epic05/phase16_c1_correctness.md` §4).

**Decision.** Split across reports: A100 screening **KEEP** (estimation); RTX 5070 measurement **NOT BENEFICIAL**; A100 P5 **VALIDATION-BLOCKED**. The only *measured* CUDA result is the RTX 5070 <1%. The A100 +9.8% is an estimate, not a measurement.

**Commit / patch.** `patches/IntersectTile.c1.cu` (5-line patch); no dedicated commit hash surfaced in `git log` for the patch file itself. C1 is deployed in the installed gsplat source (`reports/phase-next/sorting_pipeline_deep_analysis.md` §2.2; `reports/phase-next/multicamera_segmented_sort_source_audit.md`).

---

## 2. Segmented Sort (per-image CUB DeviceSegmentedRadixSort)

**What it is.** A single-boolean toggle `segmented=True` in `isect_tiles()` routes to `cub::DeviceSegmentedRadixSort::SortPairs`, sorting I per-image segments independently with a narrower key (`end_bit = 32 + tile_n_bits`, excluding `image_n_bits`) vs the default global `cub::DeviceRadixSort::SortPairs` over all n_isects (`reports/epic05/phase14b_sorting_recon.md` Q1–Q3; `reports/epic05/phase14b_sorting_source_trace.md`). Already implemented in stock gsplat; ablatable with one flag.

**Experiment setup.** Phase 14B: RTX 5070 Laptop, room scene 1,593,376 Gs (state-30000), 1080p, I=1, `torch.inference_mode`, 10 passes, CUDA-synced median (`reports/epic05/phase14b_sort_benchmark.md`).

**Benchmark results (exact).**
| tile | baseline (ms) | segmented (ms) | ratio | correct? |
|---|---|---|---|---|
| 16 | 7.63 | 34.51 | **4.52× slower** | bit-exact |
| 20 | 12.43 | 24.41 | **1.96× slower** | bit-exact |
| 32 | 9.25 | 17.71 | **1.91× slower** | bit-exact |

**C17-0 (tile-segmented depth-only sort, A100):** a 4-stage pipeline (histogram + exclusive scan + counting-sort scatter + segmented radix sort). Sort time 1.14→2.87ms (**2.5× slower**), E2E −44.7%, cross-scene 2.5–4.6× slower (room −152%, bicycle −179%, garden −362%). isect_ids 100% match; flatten_ids differ only by tie-breaking; depth monotonicity preserved (`reports/a100_validation/c17_0_tile_segmented_sort.md`).

**Why it failed.** (1) For I=1 the segment degenerates to one segment → zero sub-sort benefit; the narrower key saves only `image_n_bits`=1 bit → **zero pass-count reduction** (`reports/phase-next/multicamera_segmented_sort_source_audit.md` §4.3,§10.3). (2) CUB segmented sort adds per-segment histogram/scan/dispatch overhead. (3) C17-0's counting-sort scatter (atomicAdd × 8.7M, ~1000 contentions/tile) costs ~1.3ms, dwarfing the ~0.1ms sort saving (`reports/a100_validation/c17_0_tile_segmented_sort.md` §3.3). (4) gsplat's own docs warn segmented sort needs extra offset memory → "slower overall performance in most use cases" (`reports/epic05/phase14b_sort_benchmark.md` §4). (5) With C1 deployed, segmented removes only 1–4 image bits → zero pass reduction for I≤4, ≤1 pass for I≥8 (`reports/phase-next/multicamera_segmented_sort_source_audit.md` §4.3). (6) Potential packed-mode bug: `offsets` has shape [2] but `n_segments=I>1` → OOB read (`reports/phase-next/multicamera_segmented_sort_source_audit.md` §3.2,§6.2).

**Decision.** **DROP** / NOT BENEFICIAL. Phase 14B closed; multi-camera source audit recommends DROP (no plausible mechanism for speedup; local evidence sufficient, no prior-art check needed) (`reports/phase-next/multicamera_segmented_sort_source_audit.md` §12).

---

## 3. C17-1 — Tile-Local Queues / Fused Per-Tile Sort

**Design.** Replace the global CUB radix sort with per-tile shared-memory sort. Recommended "Option B": reduced-bit global sort by tile_id only (14-bit → 2 passes) + `cub::BlockRadixSort` per tile in shared memory (32-bit depth, 4 passes) + fused offset generation (`reports/c17_1_source_analysis.md` §7). Final-sprint CUDA design: Gaussian-parallel count kernel → exclusive scan → Gaussian-parallel atomic-claim slot (tile-contiguous flatten_ids) → `DeviceSegmentedRadixSort` per existing tile segment with key `(float32-depth-bit-pattern, input-index)` preserving stable baseline tie order (`reports/final_sprint/c17_1_cuda_design.md`).

**Experiment / result.** Correctness proven in **Python** (Phase 17B): 34,800,973 intersections, 25,932 tiles — 0 missing/extra/duplicate, 100% tile-order match, bit-exact pixel output by ordering identity, max per-tile count 14,174 (bicycle t32) (`reports/a100_validation/candidate_parallel_screening.md` C17). A100 sort-isolation estimates **+16.1–25.8%** E2E (sort = 32.2% of forward, savings 50–80% of sort) (`reports/a100_validation/candidate_parallel_screening.md`; `reports/a100_validation/feasibility_analysis_3_candidates.md` A.7). Tile distribution: 99.77% of tile16 tiles fit 48KB shared mem; bicycle 0.23% overflow (max 8,981) needs fallback (`reports/a100_validation/feasibility_analysis_3_candidates.md` A.4).

**CUDA status.** **PROMISING_BUT_INCOMPLETE.** The A100 resident gsplat 1.5.3 had a broken torch install; isolated recovery build failed on incompatible CUDA 12.4/CCCL headers before module link. Pixel/NaN/gradient parity **NOT RUN** (`reports/final_sprint/c17_1_correctness.md`; `reports/final_sprint/c17_1_final.md`). 3-scene and 13-scene benchmarks **NOT RUN** (`reports/final_sprint/c17_1_3scene_benchmark.md`; `reports/final_sprint/c17_1_13scene_benchmark.md`). Implementation committed as `ec6bbf4` (dirty/untracked content left untouched) (`reports/final_sprint/c17_1_final.md`).

**Decision.** Not an optimization yet — must not be called one without the correctness gate. Correctness gate status: **NOT PASSED**.

**C17-2 (cross-tile differential membership).** Feasibility audit only — no implementation. Adjacent-tile overlap is substantial (horizontal Jaccard P50 0.43–0.62; B reuses 38–60% from A), shared subsequence is 100% order-preserving (tautological — same depth → same order), insertions clustered (median block 1–2). But backward kernel's `flatten_ids[idx]` random access requires full materialization or a structural rewrite; trained-checkpoint data inaccessible. **Decision: CONTINUE WITH REDESIGN** (from within-tile compression to cross-tile differential) (`reports/phase-a100/c17_2_cross_tile_differential_feasibility.md`; `reports/phase-a100/c17_2_data_integrity_cross_tile_audit.md` §9).

**C17-3 (backward metadata cache).** Source audit only — **DROP**. Premise falsified: all four attribute arrays (means2d+conics+colors+opacities = 756KB) already fit L2 (3–40MB) and are saved via `ctx.save_for_backward`; shared-memory batch load already gives intra-tile reuse. The dominant backward cost is `flatten_ids` (704MB at 176M isects), which C17-3 does not reduce. A software cache would *add* traffic (`reports/phase-next/c17_3_source_audit.md`).

---

## 4. Other Forward-Pass Candidates

**C18-1 (sorted-state stability / incremental sort).** Measured sorted-state reuse across consecutive training iterations (room, 500 steps). Pairwise order preservation P50=**99.86%** (>95% gate), new-entry ratio P50=**1.51%** (<10% gate) → **GO**, proceed to C18-2 CUDA prototype. Strategies: B (reuse+merge, sort only 1.5% new entries) or C (local repair, only changed tiles) (`reports/phase-c18/c18-1_sorted_state_gate.md`). Caveat: membership reuse ≠ sorted-state reuse; pairwise flips (0.14%) occur for near-equal depths.

**C43 (adaptive tile size).** Screening showed +90.5% render speedup for tile32 vs tile16 on a 10K checkpoint — **a checkpoint artifact** (1M Gs with large radii covering few tiles). Multi-scene SfM-init validation: tile16 faster on all 3 scenes (room −13.3%, bicycle −32.2%, garden −36.6%). **Decision: DROP** — tile16 is optimal on A100 for SfM-init training (`reports/a100_validation/track_c_c1_c43_validation.md` §2; `reports/a100_validation/candidate_parallel_screening.md` C43).

**C26 / G2 (sort→raster co-design).** room tile16 cam5: sort time **0.712ms = 7.9% of T_iter**; mean 17.4 tiles/Gaussian; depth span/tile 5.28. **Verdict: DROP** — sort time negligible; ordering is not the bottleneck; any depth-preserving reordering is traversal-equivalent (`results/a100/c17-c33/phase-c26/g2_sort_raster.json`).

**Visibility / culling.** Already default in gsplat: frustum culling + radius>0 check in projection kernel; packed projection filters zero-radius (invisible) Gaussians. M4 (radius_clip) was **FALSIFIED** in Phase 11 (<1% workload reduction at quality-preserving thresholds). No new visibility/culling optimization exists that isn't already default or already falsified (`reports/epic05/phase14b_sorting_recon.md` §5; `reports/epic05/phase14_visibility_recon.md`). Forward-time signal analysis (R0.3): visible fraction ~27.6%; among visible, tiles_per_gauss Spearman 0.155, projected_radius 0.142, opacity 0.425 — opacity is best but conventional (overlaps PUP 3D-GS); **C51_SYSTEMS_OPTIMIZATION**, ~7% E2E ceiling, not a novel mechanism (`reports/phase-r0.3-forward-gate-discovery.md`).

**LPT tile scheduling (Idea 12).** Bitonic-sort fine tiles by Gaussian count. Aggregate **−0.63% FPS**; P99 improved on Truck (−11.8%) but regressed on 3/5 scenes. **DROP** (`reports/final-conclusions.md` line 51).

---

## 5. Memory Implications

- **C1:** Keys remain int64 — **no memory footprint change**; only the sort *range* changes (`reports/epic05/phase16_c1_correctness.md` §6). Sort memory traffic 144N→96N bytes (33% less) (`reports/epic05/phase16_c1_sort_benchmark.md`).
- **Sort pipeline (room, 1080p, tile16, 176M isects, C1 deployed):** CUB SortPairs = 33.7GB traffic (92.4% of pipeline), 8 passes × 4.22GB/pass, 361GB/s effective (81% of RTX 5070 peak 448GB/s) (`reports/phase-next/sorting_pipeline_deep_analysis.md` §4.3,§7.2). CUB temp storage unknown (policy-dependent).
- **flatten_ids:** 176M×4 = **704MB** — the dominant backward bandwidth cost, unaffected by C1/C17-3 (`reports/phase-next/c17_3_source_audit.md` §2,§6).
- **Attribute arrays:** means2d+conics+colors+opacities = **756KB** (nnz=21K) — fits L2 on all GPUs (`reports/phase-next/c17_3_source_audit.md` §5.2).
- **Precision:** C1 tie buckets span <0.78% relative depth (at depth=1.0, bucket 1.000→1.007812); per-tile tie rate 0.06% mean (simulated 100 Gs/tile), 13–20% mean real scenes — <1% depth reordering within tiles, negligible visual impact (`reports/epic05/phase16_c1_correctness.md` §3).

---

## 6. Numerical Correctness / Equivalence Tests

| Candidate | Test | Result | Source |
|---|---|---|---|
| C1 | IEEE-754 monotonicity proof | Zero inversions (proven) | `phase16_c1_correctness.md` §3 |
| C1 | Synthetic forward (50K Gs, t16/20/32) | No NaN/Inf; rendered mean 0.3820 identical | `phase16_c1_correctness.md` §4 |
| C1 | Backward read-path audit | Backward reads flatten_ids not isect_ids → unaffected | `c1_source_audit_final.md` §5.3 |
| Segmented sort | `torch.allclose` pixel output | Bit-exact (max diff 0.0) | `phase14b_sort_benchmark.md` §3 |
| Segmented sort | Gradient sanity | Finite gradients | `phase14b_sort_benchmark.md` §3 |
| C17-0 | isect_ids / tile_offsets / depth-monotonicity | 100% match across 6 cameras | `c17_0_tile_segmented_sort.md` §2.2 |
| C17-0 | flatten_ids | Differ (tie-breaking only, non-blocking) | `c17_0_tile_segmented_sort.md` §2.3 |
| C17-1 | Python order match (34.8M isects, 25,932 tiles) | 100% exact; bit-exact pixel by ordering identity | `candidate_parallel_screening.md` C17 |
| C17-1 | CUDA pixel/NaN/gradient parity | **NOT RUN** (build failed) | `final_sprint/c17_1_correctness.md` |
| C43 | dPSNR / dSSIM across tile sizes | 0.00 / 0.0000 (identical) | `track_c_c1_c43_validation.md` §2.3 |
| C18-1 | Pairwise order preservation | P50=0.9986; flip ratio 0.0014 | `c18-1_sorted_state_gate.md` §3 |
| C17-2 | Shared-subsequence order consistency | 100% across 4000+ pairs, all directions, both scenes | `c17_2_cross_tile_differential_feasibility.md` §4 |

No `torch.autograd.gradcheck` was found named in these reports; correctness rests on source audits, ordering-identity proofs, and `allclose`/PSNR/SSIM gates.

---

## 7. Open Contradictions

1. **C1 pass count.** `phase16_c1_*` reports assert 12→8 passes (RADIX_BITS=4). `c1_source_audit_final.md` §7 and `sorting_pipeline_deep_analysis.md` §4.2 mark this **unverified** — actual passes depend on CUB runtime `RADIX_BITS` policy (4 or 8), not source. Only the 16-bit `end_bit` reduction is source-verified. The post-hoc correction at `c1_sort_mechanism_verification.md` line 570 supersedes the earlier claim.

2. **C1 decision.** A100 screening = **KEEP** (+9.8% est); RTX 5070 measurement = **NOT BENEFICIAL** (<1%); A100 P5 = **VALIDATION-BLOCKED** (build failed). The A100 +9.8% is an *estimate* (sort-isolation × 33%), not a measurement; the only built-and-measured result is the RTX 5070 <1%.

3. **C43 speedup.** Screening +90.5% (10K checkpoint, large radii) vs validation −13% to −37% (SfM init). Resolved as a checkpoint artifact, but the screening number remains in `candidate_parallel_screening.md` ranked #2.

4. **C17-1.** Screening estimates +16–26% (sort-isolation); final-sprint status PROMISING_BUT_INCOMPLETE (CUDA build failed, no benchmark). The estimate and the "incomplete" status coexist without a measured CUDA result.

5. **Sort as bottleneck.** `sorting_pipeline_deep_analysis.md` says CUB SortPairs = 46.7–70.7% of forward (RTX 5070, room, large N) — supporting sort optimization. But `phase16_c1_correctness.md` §5 says "rasterization dominates" and `g2_sort_raster.json` says sort = 7.9% of T_iter (room cam5, A100). The discrepancy is scene/checkpoint/camera-dependent (N_isects 1.1M vs 176M) and GPU-dependent — not a true contradiction but a workload-sensitivity caveat.

6. **Segmented-sort key benefit under C1.** Pre-C1, segmented saved `image_n_bits` (1 bit → no pass reduction); post-C1, same. `multicamera_segmented_sort_source_audit.md` concludes the benefit is negligible for I≤4, yet `phase14b_sorting_recon.md` originally framed the narrower key as a benefit. The later audit supersedes.
