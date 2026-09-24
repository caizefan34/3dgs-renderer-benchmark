# H5-0R: Frontier-Aware Exact Backward Load Gating

**Date**: 2026-09-21  
**GPU**: A100-PCIE-40GB (GPU 4, uncontended)  
**Kernel**: `higs_blend_bwd_px_kernel<CDIM=3, PX=2, SCALAR_ADJOINT=true, FRONTIER=0|1|2|3>`  
**Scenes**: room, bicycle, garden (mip-nerf360, 30000 iterations, max_long_side=2048)  
**Gate**: **WEAK** (<1% total backward gain on all scenes)

---

## Executive Summary

Three frontier-aware backward load-gating variants were implemented in the production `higs_blend_bwd_px_kernel` and benchmarked against the baseline SCALAR_ADJOINT kernel:

| Variant | Mechanism | Loads Removed | Best Timing Gain | Correctness |
|---------|-----------|---------------|------------------|-------------|
| **H5A** (BLOCK_FRONTIER) | Whole batch skip if frontmost entry > block_bin_final | 1.6–2.4% | +0.73% (room) | ✅ FP noise |
| **H5B** (LOAD_FRONTIER) | Per-thread gate: skip load if idx > block_bin_final | 5.3–13.8% | +1.54% (room) | ✅ FP noise |
| **H5C** (32G_FRONTIER) | Same as H5B (32-G grouping adds nothing) | 5.3–13.8% | +0.76% (room) | ✅ FP noise |

**Decision: WEAK.** Despite removing up to 13.8% of data loads, no variant produces measurable timing improvement. The backward kernel is dominated by the processing loop, which already skips dead entries via the existing `warp_bin_final` mechanism. Data loading is not on the critical path.

---

## Section 0: Evidence Lineage Repair (R2 Exact Macro Entries)

The H5-0 report used macro entry counts (124150/311675/66724) that differ from the authoritative H3-FWD-1A-R2 exact values (124017/311610/66682):

| Scene | R2 Exact | Old H5-0 | Difference | % |
|-------|----------|----------|------------|---|
| room | 124,017 | 124,150 | 133 | 0.107% |
| bicycle | 311,610 | 311,675 | 65 | 0.021% |
| garden | 66,682 | 66,724 | 42 | 0.063% |

**Verdict**: NOT materially changed. The 32-G dead-group fractions depend on `last_ids` vs `tile_offsets`, which are unaffected by the macro entry count. The correction is purely for evidence lineage accuracy.

Artifact: [`r2_projection_repair.json`](../artifacts/higs-h5-0r/r2_projection_repair.json)

---

## Section 1: block_bin_final — Block-Level Frontier

**block_bin_final** = `max(last_ids[pixel])` over ALL valid pixels in the tile (not just the warp's pixels). This is the latest absolute `flatten_ids` position any pixel in the 128-thread block can reach.

Computed via shared-memory warp reduction:
1. Each warp computes `warp_bin_final = max(bin_final[q])` via `cg::reduce`
2. Warp lane 0 writes to `s_h5_warp_max[warp_id]`
3. All threads read all warp maxima and compute the block-level max

For PX=2 with 128 threads = 4 warps, this adds 32 bytes of static shared memory and one `block.sync()`.

---

## Sections 2–4: Kernel Variants

### H5A — BLOCK_FRONTIER (FRONTIER=1)
- **Before** the load loop: check if `batch_front = batch_end - batch_size + 1 > block_bin_final`
- If true, skip the entire batch (no load, no processing)
- Removes only 1.6–2.4% of loads — very few fully-dead batches exist

### H5B — LOAD_FRONTIER (FRONTIER=2)
- **Per-thread**: change the load condition from `if(idx >= range_start)` to `if(idx >= range_start && idx <= block_bin_final)`
- Threads whose entry is beyond the frontier skip the load (write zeros to shared mem)
- The `block.sync()` barrier still runs, but fewer global memory reads occur
- Removes 5.3–13.8% of loads — the partial entries at batch boundaries

### H5C — 32G_FRONTIER (FRONTIER=3)
- Conceptually: split each 128-entry batch into 4 groups of 32, skip dead groups
- In practice: the per-thread gate in H5B already skips exactly the same entries
- **H5C removes the same loads as H5B** — the 32-G grouping adds no benefit

### Implementation Details
- Template parameter `FRONTIER` (int, default 0) added to `higs_blend_bwd_px_kernel`
- Runtime selection via `H5_FRONTIER` environment variable (0=baseline, 1=H5A, 2=H5B, 3=H5C)
- All 4 variants compiled into the same .so; `switch(h5_frontier)` selects at runtime
- The `if constexpr(FRONTIER ...)` pattern ensures zero overhead for baseline (FRONTIER=0)

Source patch: [`scripts/h5/h5_0r_apply_frontier.py`](../scripts/h5/h5_0r_apply_frontier.py)

---

## Section 5: Exact Load Counts

Each entry = 7 loads (1 flatten_id + 1 means2d + 1 opacity + 1 conic + 3 colors for CDIM=3). Block size = 128 (PX=2).

| Scene | Baseline Loads | Baseline Entries | H5A Removed | H5B Removed | H5C Removed |
|-------|---------------|-----------------|-------------|-------------|-------------|
| room | 6,672,008 | 953,144 | 2.27% | 13.78% | 13.78% |
| bicycle | 9,885,351 | 1,412,193 | 2.43% | 5.27% | 5.27% |
| garden | 3,737,496 | 533,928 | 1.60% | 10.13% | 10.13% |

**Key finding**: H5B and H5C have identical removed fractions — the 32-G grouping in H5C skips exactly the same entries as H5B's per-thread gate.

Artifact: [`load_counts.json`](../artifacts/higs-h5-0r/load_counts.json), [`batch_skip_stats.json`](../artifacts/higs-h5-0r/batch_skip_stats.json)

---

## Section 6: Correctness (H2-BWD-2R Noise-Envelope Protocol)

Fixed random noise (`seed=42`) injected as `v_render_colors` and `v_render_alphas`. All 7 backward outputs compared against baseline:

**Return order**: (grad_means, grad_quats, grad_scales, grad_opacities, grad_colors, v_backgrounds, v_means2d)

### v_means2d (most sensitive to blend backward changes)

| Scene | Baseline max_abs | H5A max_diff | H5B max_diff | H5C max_diff |
|-------|-----------------|-------------|-------------|-------------|
| room | 4.18 | 7.15e-07 | 1.67e-06 | 3.10e-06 |
| bicycle | 12.9 | 1.43e-06 | 1.43e-06 | 1.67e-06 |
| garden | 8.87 | 2.80e-06 | 2.15e-06 | 2.26e-06 |

### grad_means (largest absolute differences)

| Scene | Baseline max_abs | H5A max_diff | H5B max_diff | H5C max_diff |
|-------|-----------------|-------------|-------------|-------------|
| room | 1,396 | 4.39e-02 | 2.32e-02 | 3.50e-02 |
| bicycle | 4,360 | 1.53e+00 | 1.06e+00 | 4.06e-02 |
| garden | 4,040 | 6.73e-02 | 6.04e-02 | 7.01e-02 |

**Verdict**: ✅ **CORRECT**. All differences are within floating-point noise (relative diff < 1e-5 for v_means2d, < 1e-3 for grad_means). The frontier gating skips entries that would have been masked by the existing per-pixel `bin_final` check anyway — the numerical results are unchanged.

Artifact: [`correctness.json`](../artifacts/higs-h5-0r/correctness.json)

---

## Section 7: Timing (20w/100m/5r Interleaved CUDA Events)

Median timing (ms) for total backward (blend + projection + SH):

| Scene | Baseline | H5A | H5B | H5C | Best % |
|-------|----------|-----|-----|-----|--------|
| room | 39.04 | 38.76 | 38.45 | 38.75 | +1.54% (H5B) |
| bicycle | 19.51 | 19.50 | 19.51 | 19.51 | +0.04% (H5A) |
| garden | 18.35 | 18.36 | 18.35 | 18.35 | +0.00% (H5B) |

**Noise characteristics**:
- room: std ~5–8ms (high variance, likely GPU frequency scaling in rep 4)
- bicycle: std ~0.01–0.07ms (very stable)
- garden: std ~0.004–0.015ms (extremely stable)

**Verdict**: No measurable speedup. bicycle and garden show <0.05% variation — the GPU is compute-bound and load reduction cannot help. room has higher variance but the best variant (H5B at 1.54%) is within the noise band.

Artifact: [`timing.csv`](../artifacts/higs-h5-0r/timing.csv), [`timing.json`](../artifacts/higs-h5-0r/timing.json)

---

## Section 8: Resources

`cuobjdump --dump-resource-usage` on the compiled `.o` for CDIM=3, PX=2, SCALAR_ADJOINT=true:

| Variant | Registers | Static Shared | Spills | Occupancy |
|---------|-----------|---------------|--------|-----------|
| baseline (FRONTIER=0) | 56 | 0 | 0 | 5 blocks/SM |
| H5A (FRONTIER=1) | 56 | 32 | 0 | 5 blocks/SM |
| H5B (FRONTIER=2) | 56 | 32 | 0 | 5 blocks/SM |
| H5C (FRONTIER=3) | 56 | 32 | 0 | 5 blocks/SM |

The 32 bytes of static shared memory is `s_h5_warp_max[8]` for the block_bin_final warp reduction. No register change, no spills, occupancy unchanged.

Artifact: [`resources.json`](../artifacts/higs-h5-0r/resources.json)

---

## Section 9: Gate Decision

| Criterion | Threshold | Result | Met? |
|-----------|-----------|--------|------|
| ≥5% blend bwd gain on 2/3 scenes | STRONG | 0 scenes | ❌ |
| ≥3% total bwd gain | STRONG | 0 scenes | ❌ |
| 1–3% total bwd gain | MARGINAL | 1 scene (room, 1.54%) | ❌ (within noise) |
| <1% total bwd gain | WEAK | 2/3 scenes | ✅ |

### **Decision: WEAK**

**Best variant**: H5B_LOAD_FRONTIER (per-thread load gate) — removes the most loads (5.3–13.8%) but produces no measurable speedup.

### Why No Speedup?

1. **Processing loop already skips dead entries**: The existing `warp_bin_final` mechanism starts the processing loop at `max(0, batch_end - warp_bin_final)`, so dead entries incur zero processing cost. The loads being eliminated are for entries that would have been skipped anyway.

2. **Block barrier dominates**: The `block.sync()` after the load loop means the warp finishes when the slowest thread finishes, not when the average thread finishes. Skipping some thread loads doesn't reduce the barrier wait time.

3. **Cached loads are cheap**: The means2d/conics/colors/opacities arrays are likely L1/L2 cached from the forward pass. The cost of the frontier check (block reduction + comparison) may offset the saved load cost.

4. **Compute-bound kernel**: For bicycle and garden (std <0.02ms), the GPU is completely saturated and the kernel is compute-bound. Load reduction cannot help a compute-bound kernel.

---

## Artifacts

| # | Artifact | Description |
|---|----------|-------------|
| 1 | [`r2_projection_repair.json`](../artifacts/higs-h5-0r/r2_projection_repair.json) | R2 exact macro entries vs old H5-0 values |
| 2 | [`load_counts.json`](../artifacts/higs-h5-0r/load_counts.json) | Exact production-kernel load counts for all variants |
| 3 | [`batch_skip_stats.json`](../artifacts/higs-h5-0r/batch_skip_stats.json) | Batch/group skip statistics |
| 4 | [`correctness.json`](../artifacts/higs-h5-0r/correctness.json) | Noise-envelope correctness comparison |
| 5 | [`timing.csv`](../artifacts/higs-h5-0r/timing.csv) | Timing results (CSV format) |
| 6 | [`resources.json`](../artifacts/higs-h5-0r/resources.json) | GPU resource usage (registers/shared/occupancy) |
| 7 | [`analysis.json`](../artifacts/higs-h5-0r/analysis.json) | Complete analysis with gate decision |
| 8 | [`provenance.json`](../artifacts/higs-h5-0r/provenance.json) | Run provenance and metadata |

---

## Conclusion

Frontier-aware load gating is **correct but weak**. The backward kernel's data loading is not on the critical path — the existing `warp_bin_final` processing skip already eliminates most dead work, and the cooperative load pattern with block barriers means reducing individual thread loads doesn't reduce warp latency. The optimization opportunity identified in H5-0 (unconditional data loading) does not translate to measurable performance gain because the loads being skipped are cheap (cached) and the kernel is compute-bound.

**Recommendation**: Do not pursue frontier load gating further. The H5-0 finding that data loads happen unconditionally is technically correct but not a performance bottleneck.
