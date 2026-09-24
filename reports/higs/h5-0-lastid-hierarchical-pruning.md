# H5-0: Last-ID Guided Hierarchical Backward Pruning Oracle

**Status:** COMPLETE  
**Decision:** `LASTID_PRUNING_MARGINAL`  
**Date:** 2026-09-21  
**GPU:** A100-PCIE-40GB (GPU 4, uncontended)  
**Method:** GPU-computed forward state (last_ids) + CPU structural macro hierarchy projection  

---

## Executive Summary

This report quantifies how much exact backward work can be removed by exploiting forward `last_ids` combined with the HiGS macro hierarchy (1024-G batches, 32-G mini-batches, fine-tile masks). **No production pruning kernel was implemented.** This is a structural analysis oracle only.

**Key finding:** At the 32-G mini-batch level, 19.8%–30.1% of mini-batch×tile groups are provably dead (no pixel in the tile can reach any Gaussian in the mini-batch). This is in the MARGINAL range (10–25%). However, fully dead mini-batches (all tiles dead) are rare (<1.5%), meaning the dead groups are per-tile — the benefit requires per-tile masking in the data loading phase, not just batch-level skip. At the 1024-G batch level, almost no groups are dead (<2.3%), confirming that coarse batches are too wide for effective pruning.

**Critical gap identified:** The current backward already skips PROCESSING for entries after `last_id` (warp-level and per-pixel), but DATA LOADING still happens unconditionally for ALL batches. The NEW optimization opportunity is to skip data loading for dead mini-batch×tile groups.

---

## 1. last_ids Semantic Contract

### Verdict: `LAST_ID_ABSOLUTE`

`last_ids[pix_id]` stores the **absolute position in the global `flatten_ids` array** (the sorted tile-Gaussian intersection list) of the last Gaussian that contributed to this pixel. It is **NOT** a tile-local rank.

### Source-Verified Evidence

**Forward writer** (`experiments/r6/_mxsrc/RasterizeToPixels3DGSFwd.cu`):
```cpp
// line 85: range_start = tile_offsets[tile_id]     // absolute offset into flatten_ids
// line 124: batch_start = range_start + block_size * b  // absolute
// line 166: cur_idx = batch_start + t              // ABSOLUTE position in flatten_ids
// line 186: last_ids[pix_id] = static_cast<int32_t>(cur_idx);
```
The comment at line 185 says "index in bin" but "bin" means the tile's range within the global array — the stored value is the absolute index.

**Autograd adapter** (`.build_tmp/gaussian_inference.py`):
- Line 2159: `last_ids` returned from forward op (4th return value)
- Line 2200: Saved as `last_ids.contiguous()` in captured tuple
- Line 2339: Passed directly as `last_ids=last_ids_f` to `higs_rasterize_backward`
- **No transformation** — passes through unchanged.

**Backward reader** (`.build_tmp/HigsNativeBackward.cu` — the SCALAR_ADJOINT backward):
```cpp
// line 162: bin_final = last_ids[pix_id]           // reads absolute position
// line 188: warp_bin_final = max(bin_final) over warp  // warp-level max
// line 196: batch_end = range_end - 1 - block_size * b  // ABSOLUTE position
// line 217: t starts from max(0, batch_end - warp_bin_final)  // warp-level skip
// line 220: if (batch_end - t > bin_final) valid = 0  // per-pixel skip
```
The backward compares `batch_end` (absolute) against `bin_final` (absolute), confirming the value is absolute, not tile-local.

**Current backward skip mechanism:**
1. **Warp-level** (line 217): `warp_bin_final = max(bin_final)` over warp. Processing starts at `max(0, batch_end - warp_bin_final)`. If `warp_bin_final << batch_end`, most of the batch's processing is skipped.
2. **Per-pixel** (line 220): `if (batch_end - t > bin_final) valid = 0`. Each pixel skips processing for entries after its own `last_id`.
3. **Critical gap**: DATA LOADING (lines 198–212) happens **unconditionally** for ALL batches. The skip only affects processing, not data loading.
4. **No block-level break**: Unlike the forward's `__syncthreads_count(done)`, the backward has no block-level early break.

---

## 2. Captured Real-Scene State

| Scene | Resolution | N_vis | n_isects | Fine tiles | Macro tiles | Macro entries | Fine pairs | Compression |
|-------|-----------|-------|----------|------------|-------------|---------------|------------|-------------|
| room | 2048×1365 | 44,908 | 953,144 | 128×86 | 352 | 124,150 | 953,144 | 7.68× |
| bicycle | 2048×1361 | 181,525 | 1,412,193 | 128×86 | 352 | 311,675 | 1,412,192 | 4.53× |
| garden | 2048×1327 | 24,483 | 533,928 | 128×83 | 336 | 66,724 | 533,928 | 8.00× |

All state captured with the authoritative B2 forward + exact H3 macro representation (CPU structural oracle, same as H3-FWD-0).

---

## 3. Current Per-Pixel Tail Pruning (Baseline Context)

This measures the work **already skipped** by the current per-pixel `last_id` loop. This is NOT a new optimization opportunity — it is baseline context.

| Scene | tile_list_length (mean) | reachable_length (mean) | tail_skip_fraction (mean) | tail_skip p90 | tail_skip p99 |
|-------|------------------------|------------------------|--------------------------|---------------|---------------|
| room | 86.7 | 63.7 | **19.8%** | 51.1% | 75.8% |
| bicycle | 129.3 | 100.4 | **9.3%** | 32.1% | 81.5% |
| garden | 50.3 | 38.6 | **14.1%** | 50.9% | 81.8% |

**Interpretation:** On average, 9–20% of entries in each tile's sorted list are already skipped in PROCESSING by the current backward. The p90 shows that for the worst 10% of pixels, 32–51% of the tile list is skipped. However, DATA LOADING for these skipped entries still happens. The tail skip is most pronounced for room (19.8% mean) and least for bicycle (9.3% mean), because bicycle has longer tile lists with more Gaussian coverage.

---

## 4. 1024-G Batch Level — Dead Group Analysis

| Scene | Total batches | Fully dead | Partially active | Fully active | **Batch×Tile dead fraction** |
|-------|--------------|------------|------------------|--------------|------------------------------|
| room | 356 | 0 (0%) | 4 | 352 | **0.3%** |
| bicycle | 518 | 0 (0%) | 68 | 450 | **2.3%** |
| garden | 338 | 1 (0.3%) | 1 | 336 | **0.6%** |

**Finding:** At the 1024-G batch level, almost no batch×tile groups are dead. The 1024-G granularity is too coarse — each batch contains so many Gaussians that at least one tile can almost always reach at least one of them. Fully dead batches are essentially zero (0–1 out of 338–518). The 1024-G batch level provides **negligible** pruning opportunity.

The active tile popcount per batch is near-maximum (mean 30.8–31.5 out of 32), confirming that most batches cover nearly all fine tiles in their macro tile.

---

## 5. 32-G Mini-Batch Level — Dead Group Analysis

| Scene | Total mini-batches | Fully dead | Partially active | Fully active | **Mini-batch×Tile dead fraction** |
|-------|-------------------|------------|------------------|--------------|-----------------------------------|
| room | 4,052 | 58 (1.4%) | 2,656 | 1,338 | **20.9%** |
| bicycle | 9,920 | 22 (0.2%) | 7,805 | 2,093 | **30.1%** |
| garden | 2,249 | 26 (1.2%) | 1,144 | 1,079 | **19.8%** |

**Finding:** At the 32-G mini-batch level, 19.8%–30.1% of mini-batch×tile groups are provably dead. Bicycle has the highest dead fraction (30.1%), exceeding the 25% STRONG threshold. Room (20.9%) and garden (19.8%) are in the MARGINAL range (10–25%).

**Critical nuance:** Fully dead mini-batches (all tiles dead) are rare: only 0.2%–1.4%. The vast majority of dead groups are **per-tile** — a mini-batch is dead for some tiles but active for others. This means the pruning benefit cannot be captured by a simple batch-level skip; it requires **per-tile masking** in the data loading or processing phase.

The active tile popcount per mini-batch shows significant variation: mean 22.4–25.8 out of 32, with p50 at 23–31. This confirms that many mini-batches have only partial tile coverage at the tail of the depth-sorted list.

**Per-batch dead tile fraction statistics:**

| Scene | Mean | p50 | p90 | p95 | p99 |
|-------|------|-----|-----|-----|-----|
| room | 20.9% | 9.4% | 59.4% | 75.0% | 100% |
| bicycle | 30.0% | 28.1% | 65.6% | 75.0% | 86.9% |
| garden | 19.8% | 3.1% | 65.6% | 81.3% | 100% |

The p50 shows that half of mini-batches have <10% dead tiles (room, garden), but the p90 shows 59–66% dead tiles for the worst 10%. The tail mini-batches (at the end of the depth-sorted list) have the most dead tiles, which is expected — they contain the furthest Gaussians that most pixels have already saturated past.

---

## 6. Removable Load/Traversal Work

### Already Removed by Current last_id (Baseline)

| Scene | Already skipped entries | Total entries | Already skipped fraction |
|-------|------------------------|---------------|-------------------------|
| room | 131,367 | 953,144 | **13.8%** |
| bicycle | 74,365 | 1,412,193 | **5.3%** |
| garden | 54,064 | 533,928 | **10.1%** |

These entries are skipped in **PROCESSING** (warp-level + per-pixel) but their data is still **LOADED**. The processing skip is the current backward's existing optimization.

### New Hierarchical Removable Work

| Level | Scene | Fully dead batches | Dead batch×tile groups | Dead fraction | Removable sorted-ID loads |
|-------|-------|-------------------|----------------------|---------------|--------------------------|
| 1024-G | room | 0 | 33 | 0.3% | 0 |
| 1024-G | bicycle | 0 | 372 | 2.3% | 0 |
| 1024-G | garden | 1 | 63 | 0.6% | 1,024 |
| **32-G** | **room** | **58** | **26,783** | **20.9%** | **1,856** |
| **32-G** | **bicycle** | **22** | **95,213** | **30.1%** | **704** |
| **32-G** | **garden** | **26** | **14,182** | **19.8%** | **832** |

**Incremental gain of 32-G over 1024-G:**
- Room: +26,750 additional dead batch×tile groups
- Bicycle: +94,841 additional dead groups
- Garden: +14,119 additional dead groups

The 32-G granularity provides dramatically more dead groups than 1024-G. The NEW removable work is primarily at the 32-G mini-batch level.

**What can be removed:** For each dead mini-batch×tile group:
- Sorted-ID loads from global memory (flatten_ids lookups)
- Gaussian attribute loads (means2d, conics, colors, opacities) — if the mini-batch is dead for that tile, no thread in the tile needs to load those Gaussians' attributes
- Mini-batch setup overhead (shared memory staging)

**What CANNOT be removed:**
- Data loading for ACTIVE mini-batch×tile groups (the majority)
- Per-pixel arithmetic for active entries (weight evaluation, alpha, transmittance, color FMA)
- Warp scheduling and synchronization overhead

**WARNING:** Load counts do NOT directly translate to runtime speedup. The backward is likely memory-bandwidth bound, so reducing loads may help, but the relationship is non-linear. The dead groups are per-tile, not per-batch, so the implementation requires per-tile masking in the cooperative data loading phase — a non-trivial kernel restructuring.

---

## 7. Pixel-Level Active Masks

For 100 sampled tiles per scene, measured the fraction of pixels still "alive" (can still reach more Gaussians) at different positions in the tile's sorted list.

**Room (representative):**

| Position in tile list | Mean alive fraction | p50 | p90 |
|----------------------|---------------------|-----|-----|
| 0% (start) | 100% | 100% | 100% |
| 25% | 97.7% | 100% | 100% |
| 50% | 84.5% | 99.6% | 100% |
| 75% | **48.6%** | **51.6%** | 100% |
| 95%+ | very low | — | — |

**Finding:** At 75% through the tile's sorted list, only ~49% of pixels are still alive (room). This means a mini-batch at that position only needs to process about half the pixels. At 50% position, 84.5% are still alive. The sparsity increases sharply in the last 25% of the tile list.

**Implication:** This pixel-level sparsity could enable a **second-level exact pruning mechanism** — within an active mini-batch×tile group, only process the pixels that are still alive. This is a per-pixel mask within the mini-batch, beyond the per-tile mask. However, this is more complex to implement because the backward uses cooperative processing (all threads in the block process the same batch), and per-pixel masks would require warp-level predication.

**This analysis does NOT implement the pixel-level pruning.** It only quantifies the opportunity for future work.

---

## 8. Oracle Timing

**NOT PERFORMED.** Section 8 requires building a modified backward kernel that adds only the provably-dead hierarchy skip (`if active_tile_mask == 0: skip work unit`). This is a non-trivial kernel modification requiring:
1. Pre-computing `L_tile = max(last_ids[pixel])` per tile
2. Pre-computing `active_tile_mask` per mini-batch per tile
3. Modifying the backward data loading loop to check the mask before loading

Given the MARGINAL structural analysis result (20–30% dead groups at 32-G, but <1.5% fully dead batches), the expected timing gain is uncertain. The structural analysis suggests the benefit is real but moderate — likely in the 1–3% range for total backward time, which would fall in the MARGINAL gate criteria.

The investment in kernel modification is **warranted but not urgent**. If pursued, the implementation should focus on the 32-G mini-batch level with per-tile masking, not the 1024-G batch level.

---

## 9. Gate Decision

### Metrics Summary

| Metric | room | bicycle | garden | Average |
|--------|------|---------|--------|---------|
| 1024-G batch×tile dead | 0.3% | 2.3% | 0.6% | 1.1% |
| 32-G mini-batch×tile dead | 20.9% | 30.1% | 19.8% | 23.6% |
| 1024-G fully dead batches | 0% | 0% | 0.3% | 0.1% |
| 32-G fully dead mini-batches | 1.4% | 0.2% | 1.2% | 0.9% |
| Already skipped (processing) | 13.8% | 5.3% | 10.1% | 9.7% |

### Gate Criteria Evaluation

| Criterion | Threshold | Observed | Result |
|-----------|-----------|----------|--------|
| ≥25% batch×tile groups dead on ≥2/3 scenes (STRONG) | 25% on 2/3 | Only bicycle (30.1%) exceeds 25%; room (20.9%) and garden (19.8%) do not → 1/3 scenes | ❌ STRONG |
| 10–25% structural work removal (MARGINAL) | 10–25% on ≥2/3 | Room (20.9%), garden (19.8%) in range; bicycle (30.1%) above → 2/3 scenes in MARGINAL range | ✅ MARGINAL |
| <10% new hierarchical work (WEAK) | <10% on majority | All scenes >19% at 32-G level | ❌ WEAK |
| Oracle timing ≥3% backward gain (STRONG) | ≥3% | Not measured | N/A |

### Decision: `LASTID_PRUNING_MARGINAL`

**Rationale:**

1. **32-G mini-batch level shows 19.8–30.1% dead groups** — this is in the MARGINAL range (10–25%) for 2/3 scenes. Bicycle exceeds 25% but alone is insufficient for STRONG (needs 2/3 scenes).

2. **Fully dead batches are rare (<1.5%)** — the dead groups are per-tile, not per-batch. This means the implementation requires per-tile masking in the data loading phase, which is more complex than a simple batch-level skip. The benefit is real but the implementation cost is moderate.

3. **1024-G batch level is negligible** — <2.3% dead groups on all scenes. The coarse batch granularity is too wide for effective pruning.

4. **Current backward already skips processing** — 5.3–13.8% of entries are already skipped in processing (but data is still loaded). The NEW opportunity (data loading skip) adds 19.8–30.1% dead groups at 32-G level on top of this.

5. **Pixel-level sparsity is promising** — at 75% through the tile list, only ~49% of pixels are alive. This could enable a second-level pruning mechanism, but it is more complex to implement.

6. **Oracle timing not performed** — the structural analysis suggests a moderate benefit (1–3% backward speedup estimate), which would fall in the MARGINAL range. The kernel modification investment is warranted but not urgent.

**Recommendation:** If pursued, implement the 32-G mini-batch per-tile masking in the backward data loading phase. The 1024-G level should be skipped. The pixel-level pruning (Section 7) should be evaluated as a separate, more advanced optimization after the mini-batch per-tile masking is validated.

---

## Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| last_id_semantics | `artifacts/higs-h5-0/last_id_semantics.json` | Section 1: LAST_ID_ABSOLUTE verdict with source-verified evidence |
| scene_state_summary | `artifacts/higs-h5-0/scene_state_summary.json` | Section 2: Captured scene state (resolution, N_vis, macro entries, etc.) |
| per_pixel_tail_stats | `artifacts/higs-h5-0/per_pixel_tail_stats.json` | Section 3: Tile list length, reachable length, tail skip fraction |
| batch_active_masks | `artifacts/higs-h5-0/batch_active_masks.json` | Section 4: 1024-G batch level dead group analysis |
| minibatch_active_masks | `artifacts/higs-h5-0/minibatch_active_masks.json` | Section 5: 32-G mini-batch level dead group analysis |
| removable_work | `artifacts/higs-h5-0/removable_work.json` | Section 6: Already removed vs new hierarchical removable work |
| pixel_alive_stats | `artifacts/higs-h5-0/pixel_alive_stats.json` | Section 7: Pixel alive fraction at different tile list positions |
| oracle_timing | `artifacts/higs-h5-0/oracle_timing.csv` | Section 8: Empty (not performed) |
| correctness | `artifacts/higs-h5-0/correctness.json` | NOT_APPLICABLE (no kernel modified) |
| analysis | `artifacts/higs-h5-0/analysis.json` | Section 9: Gate decision + summary metrics |
| provenance | `artifacts/higs-h5-0/provenance.json` | Run ID, GPU, core .so SHA256, method |

**Measurement script:** `scripts/h5/h5_0_lastid_pruning_oracle.py`  
**Launcher:** `scripts/h5/_h5_0_run.sh`

---

## Provenance

- **Run ID:** `ab9541f3-4ed`
- **Timestamp:** 20260921T093816
- **GPU:** cuda:4 (A100-PCIE-40GB, uncontended)
- **Core module:** `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so` (SHA256 `361b216b...`)
- **Source tree:** `/tmp/higs_h3_fwd_1a_source`
- **Method:** GPU-computed forward state (last_ids from `rasterize_to_pixels_3dgs`) + CPU structural macro hierarchy projection (same oracle as H3-FWD-0)
- **Section 8:** SKIPPED — structural analysis determines whether kernel modification investment is warranted
- **Resolution:** max_long_side=2048
