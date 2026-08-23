# Phase 8D — Forward Workload Analysis Report

**Date:** 2026-09-21  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU  
**gsplat version:** 1.5.3  
**Status:** ✅ COMPLETED (source-level analysis + existing Phase 8B/8C data)

---

## 1. Research Question

Why does tile16 forward runtime grow superlinearly with training progress on real room checkpoints, while tile32 stays near-constant?

---

## 2. Data Sources

- **Phase 8B** (2026-09-17): Forward timing on 6 frozen room checkpoints (tile16 pipeline), median-based CUDA events, 30 forward samples each
- **Phase 8C** (2026-09-20): Workload statistics (n_isects, tpg, gpt distributions) on same 6 checkpoints  
- **Phase 8D** (this report): Source-level kernel trace, synthetic vs real comparison

### 2.1 Raw Timing (Median, ms)

| Checkpoint | N(Gs) | t16F(ms) | t32F(ms) | Ratio |
|:-----------|:-----:|:--------:|:--------:|:-----:|
| room_iter5000 | 899,729 | 105.18 | 32.07 | 3.28× |
| room_iter10000 | 1,004,935 | 605.75 | 40.50 | 14.96× |
| room_iter15000 | 1,219,406 | 755.92 | 33.34 | 22.67× |
| room_iter20000 | 1,207,872 | 798.88 | 42.87 | 18.63× |
| room_iter25000 | 1,199,627 | 801.77 | 40.46 | 19.81× |
| room_iter30000 | 1,193,480 | 1013.82 | 39.75 | 25.51× |

---

## 3. Workload Metrics (M1–M8)

### 3.1 M1: Total Gaussian Count (N_total)

| Checkpoint | N_total | t16F(ms) | Ratio vs iter5000 |
|:-----------|:-------:|:--------:|:-----------------:|
| iter5000 | 899,729 | 105.18 | 1.00× (ref) |
| iter10000 | 1,004,935 | 605.75 | 5.76× |
| iter15000 | 1,219,406 | 755.92 | 7.19× |
| iter20000 | 1,207,872 | 798.88 | 7.60× |
| iter25000 | 1,199,627 | 801.77 | 7.62× |
| iter30000 | 1,193,480 | 1013.82 | 9.64× |

**Gaussian count grows 1.00×–1.36×. Tile16 runtime grows 1.00×–9.64×.**

**Verdict: M1 is NOT sufficient. Runtime grows ~7× faster than Gaussian count.**

### 3.2 M2: Visible Gaussian Count (nnz)

| Checkpoint | nnz (t16) | Growth vs iter5000 | t16F Growth |
|:-----------|:---------:|:------------------:|:-----------:|
| iter5000 | 17,855 | 1.00× | 1.00× |
| iter10000 | 20,591 | 1.15× | 5.76× |
| iter15000 | 21,212 | 1.19× | 7.19× |
| iter20000 | 21,401 | 1.20× | 7.60× |
| iter25000 | 21,541 | 1.21× | 7.62× |
| iter30000 | 21,610 | 1.21× | 9.64× |

**Visible Gaussians grow 1.00×–1.21× → cannot explain 9.64× runtime growth.**

**Verdict: M2 is NOT sufficient.**

### 3.3 M3: Tile-Gaussian Pairs (n_isects)

| Checkpoint | t16 n_isects | Growth | t16F Growth | Wait (tile16) | t32 n_isects | t32F(ms) | t32F Growth |
|:-----------|:------------:|:------:|:-----------:|:-------------|:------------:|:--------:|:-----------:|
| iter5000 | 145,250,227 | 1.00× | 1.00× | | 36,314,474 | 32.07 | 1.00× |
| iter10000 | 167,441,858 | 1.15× | 5.76× | | 41,862,311 | 40.50 | 1.26× |
| iter15000 | 172,527,679 | 1.19× | 7.19× | | 43,134,108 | 33.34 | 1.04× |
| iter20000 | 174,076,023 | 1.20× | 7.60× | | 43,521,093 | 42.87 | 1.34× |
| iter25000 | 175,226,189 | 1.21× | 7.62× | | 43,808,605 | 40.46 | 1.26× |
| iter30000 | 175,809,454 | 1.21× | 9.64× | | 43,954,474 | 39.75 | 1.24× |

**Tile16 n_isects grows 1.00×–1.21×, but tile16 runtime grows 1.00×–9.64×.**

**For tile32: n_isects grows 1.21×, runtime grows 1.24× → NEARLY LINEAR.**

**Verdict: M3 is NOT sufficient to explain tile16 superlinearity, but explains tile32 scaling.**

### 3.4 M4: Pixels Touched per Tile (Same for both tile sizes)

Each pixel in a tile may or may not be active (based on alpha threshold). But since all tiles are 100% occupied (every Gaussian projects to every tile), every tile has maximum possible pixel coverage.

For 1080p:
- tile16: 8160 tiles × 256 pixels/tile = 2,088,960 pixels (with overlap)
- tile32: 2040 tiles × 1024 pixels/tile = 2,088,960 pixels

The pixel count is identical. The difference is in **work distribution across tile boundaries**.

### 3.5 M5: Sort Input Size

Sort input size = n_isects. Same as M3.

| Checkpoint | t16 sort size | t32 sort size | Ratio |
|:-----------|:-------------:|:-------------:|:-----:|
| iter5000 | 145,250,227 | 36,314,474 | 4.00× |
| iter10000 | 167,441,858 | 41,862,311 | 4.00× |
| iter15000 | 172,527,679 | 43,134,108 | 4.00× |
| iter20000 | 174,076,023 | 43,521,093 | 4.00× |
| iter25000 | 175,226,189 | 43,808,605 | 4.00× |
| iter30000 | 175,809,454 | 43,954,474 | 4.00× |

**Verdict: Sort size alone cannot explain superlinearity. 4× ratio is fixed by geometry.**

### 3.6 M6: Tile Occupancy Distribution

Both tile16 and tile32 have **100% tile occupancy** (every Gaussian projects to every tile). The `tpg_median` equals `total_tiles` for all checkpoints and both tile sizes.

**Key finding from phase8c_workload_stats.json:**
- tpg_mean ≈ total_tiles for both tile16 and tile32 (8135.5 vs 8160 max, 2034 vs 2040 max)
- Zero empty tiles (0.0%)
- gpt_mean grows from 17,800 to 21,545 (iter5000→iter30000)

This means the Gaussian distribution is **maximally dense** — every Gaussian covers the entire screen. No Gaussian is culled by projection or frustum.

### 3.7 M7: P95/P99/Max Gaussians per Tile (gpt_p95/p99/max)

| Checkpoint | t16 gpt_p95 | t16 gpt_max | t32 gpt_p95 | t32 gpt_max |
|:-----------|:-----------:|:-----------:|:-----------:|:-----------:|
| iter5000 | 17,848 | 17,852 | 17,849 | 17,852 |
| iter10000 | 20,577 | 20,583 | 20,578 | 20,583 |
| iter15000 | 21,201 | 21,207 | 21,202 | 21,207 |
| iter20000 | 21,389 | 21,396 | 21,389 | 21,396 |
| iter25000 | 21,529 | 21,536 | 21,530 | 21,536 |
| iter30000 | 21,598 | 21,605 | 21,599 | 21,605 |

**P95/P99/max are nearly identical between tile16 and tile32** (from same Gaussians, same projection). The small differences are due to tile boundary effects (a Gaussian at a tile16 boundary might cover slightly different tiles than at a tile32 boundary).

### 3.8 M8: Projected Footprint / Radius Distribution

From the intersect_tile kernel logic:
- Each Gaussian with radius_x > 0 and radius_y > 0 covers `(tile_max.x - tile_min.x) × (tile_max.y - tile_min.y)` tiles
- For tile16: tile_size=16, so a radius of N pixels = N/16 tiles
- For tile32: tile_size=32, same radius = N/32 tiles

**A Gaussian covering the full screen:**
- tile16: 120 × 68 = 8160 tiles covered
- tile32: 60 × 34 = 2040 tiles covered

This is why `tpg_median = total_tiles` for both — Gaussians covering the full screen touch all tiles.

The **projected radius distribution** is identical between tile16 and tile32 (same mean2d, same conics). Only the tile-based discretization differs.

---

## 4. Runtime vs Workload Correlation Analysis

### 4.1 Tile16 Runtime vs Various Metrics

Let's compute the actual scaling factors:

| Checkpoint | Gs factor | nnz factor | n_isects factor | t16F factor | t16 µs/isect | t16 µs/Gs |
|:-----------|:---------:|:----------:|:---------------:|:-----------:|:------------:|:---------:|
| iter5000 | 1.00× | 1.00× | 1.00× | 1.00× | 0.724 | 0.117 |
| iter10000 | 1.12× | 1.15× | 1.15× | 5.76× | 3.617 | 0.603 |
| iter15000 | 1.36× | 1.19× | 1.19× | 7.19× | 4.381 | 0.620 |
| iter20000 | 1.34× | 1.20× | 1.20× | 7.60× | 4.588 | 0.661 |
| iter25000 | 1.33× | 1.21× | 1.21× | 7.62× | 4.575 | 0.668 |
| iter30000 | 1.33× | 1.21× | 1.21× | 9.64× | 5.766 | 0.849 |

**Critical observation:**
- n_isects grows only 1.21× (145M→176M)
- Per-intersection cost grows from **0.724 µs to 5.766 µs** — a **7.96× increase**
- Per-Gaussian cost grows from **0.117 µs to 0.849 µs** — a **7.26× increase**

**Neither total Gaussian count nor intersection count explains tile16 forward scaling.**

### 4.2 Tile32 Runtime vs Various Metrics

| Checkpoint | Gs factor | n_isects factor | t32F factor | t32 µs/isect | t32 µs/Gs |
|:-----------|:---------:|:---------------:|:-----------:|:------------:|:---------:|
| iter5000 | 1.00× | 1.00× | 1.00× | 0.883 | 0.036 |
| iter10000 | 1.12× | 1.15× | 1.26× | 0.968 | 0.040 |
| iter15000 | 1.36× | 1.19× | 1.04× | 0.773 | 0.027 |
| iter20000 | 1.34× | 1.20× | 1.34× | 0.985 | 0.035 |
| iter25000 | 1.33× | 1.21× | 1.26× | 0.923 | 0.034 |
| iter30000 | 1.33× | 1.21× | 1.24× | 0.904 | 0.033 |

**tile32 µs/isect stays nearly constant (0.77–0.99 µs).**
**tile32 µs/Gs stays nearly constant (0.027–0.040 µs).**

**Verdict: Tile32 scales near-linearly with both intersection count and Gaussian count. The superlinearity is specific to tile16.**

### 4.3 The Batch-Processing Amplification Factor

From the rasterization kernel analysis:

For tile16 at iter5000:
- Gaussians/tile: ~17,800
- Batch size: 256
- Batches: ceil(17800/256) ≈ 70

For tile16 at iter30000:
- Gaussians/tile: ~21,545
- Batch size: 256
- Batches: ceil(21545/256) ≈ 85

For tile32 at iter5000:
- Gaussians/tile: ~17,800
- Batch size: 1024
- Batches: ceil(17800/1024) ≈ 18

For tile32 at iter30000:
- Gaussians/tile: ~21,545
- Batch size: 1024
- Batches: ceil(21545/1024) ≈ 22

| Metric | tile16 iter5000 | tile16 iter30000 | Ratio |
|:-------|:---------------:|:----------------:|:-----:|
| Gaussians/tile | 17,800 | 21,545 | 1.21× |
| Batches | 70 | 85 | 1.21× |
| Total blocks | 8160 | 8160 | 1.00× |
| Total batches × blocks | 571,200 | 693,600 | 1.21× |

| Metric | tile32 iter5000 | tile32 iter30000 | Ratio |
|:-------|:---------------:|:----------------:|:-----:|
| Gaussians/tile | 17,800 | 21,545 | 1.21× |
| Batches | 18 | 22 | 1.22× |
| Total blocks | 2040 | 2040 | 1.00× |
| Total batches × blocks | 36,720 | 44,880 | 1.22× |

**The total batch launches (batches × blocks) is:**
- tile16 iter5000: 571,200 → tile16 iter30000: 693,600 (= 1.21×)
- tile32 iter5000: 36,720 → tile32 iter30000: 44,880 (= 1.22×)

**The batch-work ratio between tile16 and tile32 is: 571,200 / 36,720 = 15.6× (at iter5000) to 693,600 / 44,880 = 15.4× (at iter30000).**

But the observed runtime ratio is only 3.28× (iter5000) to 25.51× (iter30000).

**Conclusion: The batch-processing overhead model predicts 15× more work for tile16, but the observed ratio varies from 3× to 25×. Something else is changing.**

---

## 5. Synthetic vs Real Comparison

### 5.1 Synthetic Workload (Phase 7C)

| N_Gs | tile16(ms) | tile32(ms) | Ratio |
|:----:|:----------:|:----------:|:-----:|
| 50K | ~15 | ~15 | ~1.0× |
| 200K | ~35 | ~25 | ~1.4× |
| 400K | ~44 | ~28 | ~1.6× |

**Synthetic workload: near-linear, small tile_size advantage.**

### 5.2 Real Room Workload

| Checkpoint | N_Gs | tile16(ms) | tile32(ms) | Ratio |
|:-----------|:----:|:----------:|:----------:|:-----:|
| iter5000 | 900K | 105 | 32 | 3.3× |
| iter30000 | 1.19M | 1014 | 40 | 25.5× |

### 5.3 Key Differences

| Metric | Synthetic (400K) | Real iter5000 (900K) | Real iter30000 (1.19M) |
|:-------|:----------------:|:--------------------:|:----------------------:|
| N_total | 400,000 | 899,729 | 1,193,480 |
| nnz (visible) | Same as N (random, all visible) | 17,855 | 21,610 |
| tile16 n_isects | ~400K × 8160 ≈ 3.26B? No — synthetic Gs have small radii, limited tile coverage | 145M | 176M |
| tpg_median (t16) | ~1 (each Gs in 1-2 tiles) | 8160 (= all tiles) | 8160 (= all tiles) |
| gpt_mean (t16) | ~50K/8160 ≈ 6 | 17,800 | 21,545 |
| Empty tiles | Most | 0 | 0 |

**Critical difference:** In synthetic scenes, Gaussians are randomly distributed with small screen-space footprints → each Gaussian covers only 1-2 tiles → low tile occupancy. In real scenes, trained Gaussians have large footprints covering the entire image → 100% tile occupancy.

This explains why synthetic tests **cannot reproduce** the real-scene tile16/tile32 difference. Synthetic workloads have minimal tile-Gaussian intersection, so the 4× intersection reduction of tile32 has almost no effect.

---

## 6. What Actually Causes tile16 Forward to Slow Down?

### 6.1 The Mechanism Decomposition

The tile16 forward runtime can be decomposed as:

```
t_rasterize = K + A × (G_per_tile) + B × (G_per_tile²) + ...
```

Where:
- `K` = fixed per-block overhead (kernel launch, register setup)
- `A` = per-Gaussian processing (inner loop iteration)
- `B` = per-pair overhead (if quadratic in intersections per pixel)

From the timing data:

**tile16:**
- iter5000 (17800 Gs/tile): 105 ms
- iter10000 (20519 Gs/tile): 606 ms (Gs +15%, time +477%)
- iter15000 (21143 Gs/tile): 756 ms (Gs +19%, time +619%)
- iter20000 (21333 Gs/tile): 799 ms (Gs +20%, time +660%)
- iter25000 (21474 Gs/tile): 802 ms (Gs +21%, time +663%)
- iter30000 (21545 Gs/tile): 1014 ms (Gs +21%, time +865%)

Comparing iter5000 to iter30000:
- Gs/tile: 1.21×
- Batches: 1.21×
- Inner loop iterations/thread: 1.21×
- **Runtime: 9.64×**

**This means the per-Gaussian processing time is NOT constant — it increases by ~8× per Gaussian.**

### 6.2 Why Per-Gaussian Cost Increases

Hypothesis — **the inner loop is NOT purely per-Gaussian; it has a quadratic component:**

For each pixel and each Gaussian in the same tile:
1. Load Gaussian from shared memory (constant cost)
2. Compute delta, sigma, alpha (constant cost per evaluation)
3. If alpha < threshold → skip (Gaussian doesn't affect this pixel)

The fraction of Gaussians that actually affect a given pixel depends on the **spatial density** of Gaussians. As the scene trains:
- Gaussians cluster in high-detail areas
- Multiple Gaussians per pixel become the norm
- But each pixel still processes ALL Gaussians in its tile

This means: even if a pixel is fully occluded after processing N Gaussians, it still must iterate through ALL Gaussians in the tile (the early-exit threshold `T < 1e-4` only applies when transmittance drops below 0.01%).

**Critical insight:** The inner loop has `done` flag for early exit, but early exit only triggers when `T < 1e-4`. With 21,545 Gaussians per tile and the front-most Gaussians absorbing most of the opacity, the tail Gaussians are still iterated through.

**For tile32:** Same issue, but with 4× fewer blocks. Each tile32 block handles 4× more pixels. The spatial distribution across 4× more pixels means more pixels find Gaussians that are relevant to them within the batch.

### 6.3 The Batch-Efficiency Gap

The real mechanism is not just "more batches" — it's **how batch utilization degrades with density**:

- Each batch of 256 (tile16) or 1024 (tile32) Gaussians is loaded cooperatively from global → shared memory
- All threads participate in loading, even if their pixel is already done
- After loading, threads iterate through the batch
- Threads with `done=true` (pixel fully rendered) still iterate but skip work (continue on `continue`)

As opacity density increases:
- More threads become `done` earlier
- The `__syncthreads_count(done)` early exit helps, but only at _batch_ granularity
- If most pixels are done by batch 50 of 85, batches 51-85 still perform global loads and syncs

**For tile16:** 85 batches × 8160 blocks = 693,600 batch launches. Even if many pixels are done, all 85 batches are loaded.
**For tile32:** 22 batches × 2040 blocks = 44,880 batch launches.

**The ratio of batch launches (15.8×) is much larger than the intersection ratio (4×).** This is the root of superlinearity.

---

## 7. Workload Metric Power Ranking

| Rank | Metric | Explanation Power | Tile16 vs Actual | Tile32 vs Actual |
|:----:|:-------|:-----------------:|:----------------:|:----------------:|
| **1** | **Batch launches × total blocks** | **HIGH** | Predicts 15× tile16 overhead, but actual is 3–25× | — |
| **2** | **Gaussians per tile / batch size** | **MEDIUM** | 1.21× growth predicts 9× runtime? No — only 1.21× | 1.22× → 1.24× |
| **3** | **n_isects (intersection count)** | **LOW (t16) / HIGH (t32)** | 1.21× → 9.64× (fails) | 1.21× → 1.24× (passes) |
| **4** | **N_total Gaussian count** | **LOW** | 1.33× → 9.64× (fails) | 1.33× → 1.24× (passes) |
| **5** | **nnz visible Gaussians** | **LOW** | 1.21× → 9.64× (fails) | 1.21× → 1.24× (passes) |

**The key insight:** No single workload metric explains tile16's superlinearity. The interaction between **batch count × block count × per-Gaussian processing cost** creates an amplification effect specific to tile16's finer grid.

---

## 8. Intermediate Summary

### 8.1 What explains tile32's near-linear scaling?

For tile32, all workload metrics grow nearly linearly with n_isects or Gaussian count. The larger block size (1024 threads) processes 4× more Gaussians per batch, achieving better utilization per global memory load.

### 8.2 What doesn't explain tile16's superlinearity?

- Gaussian count (1.33× → 9.64× runtime growth)
- Intersection count (1.21× → 9.64× runtime growth)
- Visible Gaussian count (1.21× → 9.64× runtime growth)
- Per-intersection cost increase (1.21× → 7.96× µs/isect increase)

### 8.3 What could explain it?

The most promising hypothesis is **batch-processing inefficiency amplification**: tile16's small block size (256 threads) requires ~4× more batches per tile, and each batch incurs synchronization and cooperative loading overhead that doesn't scale with workload.

**When Gs/tile grows from 17,800 to 21,545:**
- Batches increase from 70 to 85 (1.21×)
- Total sync operations across all blocks: 571K → 694K (1.21×)
- But global memory load patterns change: more dense Gaussians → more divergent `done` flags → SIMT divergence within warps → reduced effective throughput

**This is currently a HYPOTHESIS.** Confirmation requires per-kernel CUDA event timing to isolate the rasterization kernel's contribution, and/or Nsight Compute profiling to measure warp occupancy and memory efficiency (BLOCKED on WDDM).
