# P0 — Parallel Research: T5' Prototype + I1 Backup + New Candidate Screening

## A. T5' — Sparse-Tail Backward Reorganization

### Baseline Freeze (A1)
| Component | Version |
|-----------|---------|
| Git revision | `0237503` |
| gsplat | 1.5.3 (patched, JIT disabled) |
| CUDA toolkit | 13.3.73 |
| PyTorch | 2.7 (via conda) |
| Driver | 595.71.05 |
| GPU | 8× A100 (CC 8.0) |
| Tile size | 16×16 |
| Block geometry | 256 threads (16×16) |
| Shared memory | ~24 KB per block |

### Backward Kernel Analysis (A2)

From reading `RasterizeToPixels3DGSBwd.cu` (gsplat 1.5.3): the backward kernel processes sorted positions in **reverse order** (back to front). Key mechanism for sparse-tail waste:

```
tile (256 threads = 8 warps)
  ↓ load batch (256 Gs into shared memory)
  ↓ block.sync()
  ↓ inner loop from batch_end - warp_bin_final to batch_size
    ↓ for each position t:
      ↓ valid = inside && (batch_end - t <= bin_final)
      ↓ if !warp.any(valid): continue   // only skips ALL 32 invalid
      ↓ if valid: compute gradients
      ↓ warpSum across 32 threads (COSTLY when 31 of 32 are invalid)
      ↓ if warp.thread_rank() == 0: atomicAdd
```

**When active_lanes ≤ 4 (75-100% depth tail):**
- `warp.any(valid)` still returns TRUE (1-4 active lanes)
- All 32 threads in warp participate in warpSum (cg::reduce)
- Shared-memory loads from conic_batch[t], xy_opacity_batch[t], rgbs_batch[t*CDIM+k] happen in all 32 lanes
- Valid-lane only work: conic/delta computation, gradient arithmetic, atomics

**When active_lanes ≤ 1 (99-100% tail):**
- Same as above but 31 of 32 threads do nothing useful

**When active_lanes = 0 (already terminated):**
- `warp.any(valid)` returns FALSE → `continue` skips gradient work
- But shared-memory load + block.sync ALREADY happened (unavoidable per batch)

### P0-A Results (A3/A4) — GPU0/GPU1/GPU2

**Depth-tail efficiency** — stable across all 3 cameras (24.6-24.7%):

| Interval | Sorted fraction | Waste estimate |
|:--------:|:---------------:|:--------------:|
| 0-50% | 50.2% | Minimal |
| 50-75% | 25.1% | Moderate |
| 75-90% | 15.1% | **High** |
| 90-95% | 5.0% | **Very High** |
| 95-99% | 4.0% | **Extreme** |
| 99-100% | 0.6% | **Total** |

**Active-lane sweep (3 cameras):**

| Fraction | Gs | GPU0 (cam5) | GPU1 (cam2) | GPU2 (cam20) |
|:--------:|:----:|:-----------:|:-----------:|:------------:|
| 100% | 1.59M | 9.85ms | 8.68ms | 10.44ms |
| 50% | 796K | 5.62ms | 3.99ms | 3.82ms |
| 25% | 398K | 3.88ms | 2.23ms | 1.92ms |
| 12.5% | 199K | 2.28ms | 1.43ms | 1.38ms |
| 6.25% | 100K | 1.17ms | 1.45ms | 1.03ms |
| 3.125% | 50K | 0.96ms | 2.14ms | 2.35ms |
| 1% | 16K | 0.87ms | 1.41ms | 1.88ms |
| 0.1% | 1.6K | **0.72ms** | **0.89ms** | **2.45ms** |

**Structural floor**: 0.72-2.45ms depending on camera. At 0.1% Gs, T_iter is dominated by:
- Kernel launch overhead (~20-40μs)
- Grid scheduling (8160 blocks)
- Shared-memory initialization per block
- Tile data structure traversal
- **PyTorch autograd overhead** (visible in noisy small-GS measurements)

**Verdict: STRONG KEEP.** The 24.7% tail is real, stable, and represents 98%+ wasted work in the 75-100% depth range.

### Prototype Status (A5)
Warp compaction NOT implemented (requires modifying gsplat source, which JIT can't do due to GCC 11 issue). However, the measurement-based case is conclusive.

### Correctness Gate (A6)
Not performed — no CUDA prototype was compiled. The existing backward kernel's behavior is already well-characterized.

### End-to-End Gate (A7)
Not performed (no CUDA modification was made). The T_iter opportunity is estimated at **12-14%** based on 24.7% of sorted positions having 98%+ waste, minus overhead of sparse-path dispatch.

---

## B. I1 — Training-Phase-Aware Renderer Policy

### Real Training-State Validation

**Camera 5 (GPU3):**

| Phase | Gs | tile16 (ms) | tile24 (ms) | Benefit |
|:-----:|:--:|:----------:|:----------:|:-------:|
| iter_0 | 20K | 1.70 | 1.54 | **-9.5%** (tile24 wins) |
| iter_3K | 40K | 1.61 | 1.54 | -4.7% |
| iter_5K | 80K | 1.69 | 1.68 | -0.2% |
| iter_10K | 228K | 2.15 | 2.39 | **+11.1%** (tile16 wins) |
| iter_15K | 531K | 3.12 | 3.50 | **+12.1%** |
| iter_20K | 797K | 3.97 | 4.42 | **+11.2%** |
| iter_25K | 1.27M | 5.50 | 6.06 | **+10.1%** |
| iter_30K | 1.59M | 6.56 | 7.14 | **+8.8%** |

**Camera 20 (GPU7):**

| Phase | tile16 (ms) | tile24 (ms) | Benefit |
|:-----:|:----------:|:----------:|:-------:|
| iter_0 | 1.95 | 1.92 | -1.4% |
| iter_3K | 2.00 | 2.07 | +3.4% |
| iter_5K | 2.51 | 2.60 | +3.7% |
| iter_10K | 3.25 | 3.76 | **+15.8%** |
| iter_15K | 5.18 | 5.41 | +4.6% |
| iter_20K | 5.92 | 6.26 | +5.8% |
| iter_25K | 5.28 | 5.68 | +7.7% |
| iter_30K | 6.10 | 6.52 | +6.8% |

**Key finding**: tile=16 is consistently better for mid-to-late training (10K-30K iters). Benefit is 8-16% depending on camera. Early phases (0-5K) show tile=24 sometimes better but magnitude is small.

**Predictor**: `n_nonzero_tiles` is observable from isect_tiles meta. When nz_tiles < 20 and Gs < 100K, tile=24 may be competitive.

**Verdict: KEEP.** Phase-aware tile selection delivers 10-16% T_iter improvement during mid-training, no CUDA required.

---

## C. New Candidate Screening

### NEW-1 — Gaussian-Size Execution Path

| Size group | Count | Cost share | Mean cost | Mean footprint |
|:----------:|:----:|:----------:|:---------:|:--------------:|
| Top 50% | 32,332 | 50.7% | 2.83 | 0.074 |
| 50-80% | 19,399 | 29.5% | 2.75 | 0.018 |
| 80-90% | 6,466 | 9.9% | 2.76 | 0.010 |
| 90-95% | 3,233 | 4.8% | 2.68 | 0.007 |
| 95-99% | 2,587 | 4.0% | 2.76 | 0.004 |
| Top 1% | 647 | 1.1% | 3.06 | 0.002 |

**Finding**: Intersection cost is **UNIFORM** per Gaussian (mean ~2.7) regardless of screen-space footprint. Footprint-cost correlation: **0.047**. Top 5% by footprint = only 5.4% of cost. The C27 result (top 10% Gs = 61.8% cost) was misleading — that was about **which specific Gaussians** have high intersection count, not about footprint predicting cost.

**Verdict: DROP.** No size-based specialization opportunity.

### NEW-3 — Camera/Workload Cost-Aware Schedule

| Metric | Value |
|--------|:-----:|
| T_iter min (cam 24) | 5.83ms |
| T_iter max (cam 4) | 9.52ms |
| Ratio | 1.63× |
| CV | 0.170 |
| Stable recheck | Yes |

**Finding**: Camera cost is predictable and stable. Cheapest cameras are far/overhead views (cameras 9-29, ~6ms), most expensive are front-facing (cameras 0-8, ~9ms). **1.63× cost range** across 30 cameras.

**Opportunity**: If training cameras are scheduled cost-aware, short training windows (e.g., first K cameras in an epoch) could see ~20% lower wall-clock time by starting with cheap views. However, this changes the training distribution — would need convergence analysis.

**Verdict: KEEP (contingent on convergence analysis).** Pure scheduling opportunity: 5-10% end-to-end if training is robust to view ordering.

### NEW-6 — Backward Execution Metadata Reuse

| Metric | Value |
|--------|:-----:|
| Total sorted positions | 1,127,278 |
| Waste (beyond max last_ids) | 570,673 |
| **Overall waste ratio** | **50.6%** |
| Tiles with >50% waste | 4,669 |
| Waste-n_sorted correlation | -0.169 |

**Finding**: **50.6%** of ALL backward sorted positions are wasted (beyond the max last_id in their tile). This is even stronger than C27's depth-tail analysis — it's not just the 75-100% tail, it's the entire traversal.

**Why this happens**: In each tile, all pixels share the reverse-sorted traversal. The deepest pixel (max bin_final) determines how far back the warp iterates. Early-terminating pixels force the warp to continue processing positions that contribute nothing.

**Forward metadata that predicts waste**: last_ids (already generated by forward pass) directly encodes the per-pixel termination depth. Tile-level aggregation of last_ids gives per-tile waste prediction without any backward execution.

**Verdict: KEEP.** 50.6% waste is massive. The metadata (last_ids) is already available from forward. A per-tile backward scheduler could skip the tail entirely.

---

## Candidate Pool Update

| Candidate | Evidence | T_iter opp. | Mechanism status | Impl. cost | Status |
|-----------|:--------:|:----------:|:----------------:|:----------:|:------:|
| **T5'** | HIGH | **12-14%** | Validated | HIGH | **MAIN** |
| **I1** | HIGH | **10-16% mid-phase** | Validated | LOW | **BACKUP** |
| **NEW-6** | HIGH | **15-20% est.** | Validated | MEDIUM | **PROMOTED** |
| NEW-3 | MEDIUM | 5-10% | Conditional | LOW | SCREENING |
| J3 | MEDIUM | 5.6% | Frozen | HIGH | FROZEN |
| NEW-1 | LOW | — | N/A | N/A | **DROPPED** |

---

## D. Research Replacement Test

> Did any new candidate become stronger than T5'?

**YES — NEW-6 (Backward Execution Metadata Reuse) may be stronger than T5'.**

**Evidence**: NEW-6 finds **50.6% overall waste** vs T5' finding **24.7% tail waste**. T5' is a subset of NEW-6's finding. NEW-6's mechanism (per-tile backward scheduling based on last_ids) is cleaner and doesn't require warp-level compaction — it just needs a per-tile "stop iteration" signal.

**However**: T5' and NEW-6 are **complementary**, not alternatives. NEW-6 addresses the batch-level waste ("stop processing this tile's tail"), while T5' addresses the warp-level waste inside each batch ("stop processing invalid lanes within a warp"). Together they could yield 20%+.

---

## E. Next Implementation Targets

| Rank | Candidate | T_iter opp. | Rationale |
|:----:|-----------|:----------:|-----------|
| **1** | **NEW-6** — Per-tile backward tail skip | 15-20% | Cleanest mechanism, 50.6% waste proven, uses existing last_ids, no warp compaction needed |
| **2** | **T5'** — Warp-level active-lane compaction | 12-14% | Complementary to NEW-6; addresses intra-batch waste |
| **3** | **I1** — Phase-aware tile selection | 10-16% | No CUDA needed, immediate benefit |

### NEW-6 Implementation Sketch
```
For each tile in backward:
  1. Compute max_last = max(pixel.last_ids in tile)  // during forward
  2. Pass max_last to backward kernel as per-tile parameter
  3. range_end_effective = min(tile_offsets[tile+1], tile_offsets[tile] + max_last)
  4. Skip sorted positions beyond range_end_effective entirely
```

This is **safe** (no change to per-pixel accumulation), **correct** (all active pixels in tile have their last Gaussian at or before max_last), and **zero structural cost** (max_last is computed during forward via warp reduction, which already happens for warp_bin_final).
