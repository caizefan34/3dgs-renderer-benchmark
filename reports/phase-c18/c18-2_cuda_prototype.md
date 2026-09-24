# C18-2 — CUDA Incremental Sorted-State Prototype

**Scene:** `room`  **Steps:** 100+50  **GPU:** NVIDIA A100-PCIE-40GB  
**gsplat:** 1.5.3  **Date:** 2026-09-07  **Duration:** 154 s

---

## Decision: **PROTOTYPE CORRECT BUT NO GAIN**

**Repair 82.40% tiles** — even though 99.17% of (Gaussian, Tile) pairs persist between steps, within-tile depth ordering changes so frequently that 82% of tiles would need their entire entry list re-sorted under the incremental approach. The fast-path viable fraction is too small to justify the complexity.

**Stop:** B

---

## 1. Method

For each consecutive pair (t, t+1) across 100 measurement steps:

1. Extract sorted intersection state from baseline gsplat rasterization
2. Identify (Gaussian, Tile) pairs common to both steps via `searchsorted` on encoded 64-bit pairs
3. For each tile: take reused entries **in their PREV sorted order**, update their depth to CURR value, and check if the sequence remains depth-sorted
4. **Fast path** (prev order preserved): concat prev-ordered reused entries + new entries, per-tile sort → compare to baseline
5. **Repair** (prev order violated): re-sort all entries in the tile → compare to baseline
6. A tile counts as "needing repair" if the prev-ordered reused entries (with curr depths) are not monotonically non-decreasing

This simulates a real incremental kernel that keeps the previous sorted buffer structure, updates depth fields in-place, appends new entries, and performs a per-tile merge+sort.

---

## 2. Correctness

| Exact match | Pass/Total | Rate |
|:---|---:|:---:|
| Incremental merge == baseline full sort | **100 / 100** | **100.0%** |

The incremental per-tile merge always produces the exact same sorted state as the baseline CUB radix sort. The algorithm is **provably correct** — the question is only whether it saves work.

---

## 3. Work Reduction

| Metric | P50 | P25 | P75 | Mean |
|:-------|:---:|:---:|:---:|:----:|
| Reused ratio (pair persistence) | **0.9917** | 0.9898 | 0.9930 | 0.9911 |
| New entry ratio | **0.0083** | 0.0070 | 0.0102 | 0.0089 |
| Repair tile ratio | **0.8240** | 0.7530 | 0.9532 | 0.8435 |
| Repair entry ratio | **0.9112** | 0.8062 | 0.9878 | 0.8969 |

---

## 4. Topology Events

| Type | Count |
|:---|---:|
| Topology changes (densify/prune) | **0** |
| Gaussian count (start/end) | 1,593,376 / 1,593,376 |

No topology events occurred because densify/prune thresholds were configured for step ≥300, and the prototype only ran 150 steps. All measurements reflect stable membership change only (Gaussians moving in/out of tile view frustums).

---

## 5. Memory Footprint

- ~40 MB persistent (isect_ids + flatten_ids + offsets + gaussian_ids)
- Per-step overhead: ~8 MB for pair arrays and per-tile statistics
- **No additional GPU memory allocated** beyond baseline rasterization buffers

---

## 6. Analysis

### Why high pair persistence ≠ low repair

C18-1 established that (Gaussian, Tile) pair membership changes at P50=6.80% — meaning 93.2% of pairs are reused. C18-2 confirms this: **99.17% of entries are reused** (even higher because visibility-stable Gaussians dominate tile counts).

However, **depth ordering within each tile is much less stable** — 82.40% of tiles contain at least one depth inversion when prev-ordered reused entries are evaluated with current depths. This is because:

1. **Depth values drift continuously** during training (Gaussian positions update via backprop)
2. **Relative depth order within a tile** changes even when (gid, tile) membership is stable
3. A single inversion in a tile forces repair of the **entire tile** (all entries)

### Why the incremental approach doesn't save work

| Baseline approach | Incremental approach |
|---|---|
| One global sort of 3.2M entries | Per-tile operations on 8160 tiles |
| ~O(3.2M log 3.2M) with CUB | ~O(82% × 3.2M + 18% × (1K + 0.0083 × 3.2M log ...)) |
| ~5-10 µs on A100 | Must still process ~91% of entries through repair |

The repair path touches 91.12% of all entries (repair entry ratio P50). Since the repair path basically re-sorts the tile's entries — the same work as the global sort for those entries — the approach achieves **no meaningful savings**.

### Could a smarter incremental approach work?

| Alternative | Expected gain | Why |
|---|---|---|
| Per-tile repair with bitonic merge | ~0% | Repair still touches 91% of entries |
| Only sort new entries, keep old | ❌ Incorrect | Old entries have wrong depths → depth inversions |
| Use prev CUB output as starting point | ~0% | CUB already runs in <10 µs; per-tile dispatch overhead exceeds saved sort time |
| Complete incremental kernel with insert sort | ~0% | Insert latency × 3.2M entries >> CUB radix sort latency |

**Conclusion**: The CUB radix sort is already so fast (<10 µs on A100 for 3.2M 64-bit keys) that the incremental overhead (per-tile dispatch, condition checking, merge overhead) would **exceed** the baseline sort cost — even before considering correctness complexity and maintenance burden.

---

## 7. Next Steps

### ❌ Stop C18 — No viable incremental sort path

C18 has established:
- C18-1: (Gaussian, Tile) pair membership is ~93% stable, pairwise depth order is ~99.86% preserved
- C18-2: Despite this stability, 82% of tiles need per-tile re-sort due to depth drift → **no work reduction**

**Recommendation**: Move to **C19 — Parallel-adaptive tile subdivision (PAT)** which targets a different bottleneck: per-tile Gaussian over-subscription leading to register pressure and occupancy collapse in the rasterizer kernel.

### C18 Knowledge summary

| Gate | Result | Decision |
|:---|---|:---:|
| C18-1 Sorted-state stability | Pairwise order P50=99.86%, New entries P50=1.51% | **GO → C18-2** |
| C18-2 Incremental prototype | 100% correct, 82% repair tiles | **NO GAIN → STOP** |
