# C17-2 Data Integrity + Cross-Tile Overlap Deep Audit

## 1. N_isect Reconciliation

### Root Cause

The discrepancy between the Corrected A100 Baseline and the C17-2 Membership Audit is caused by **different Gaussian sources**, not a bug.

| Factor | Corrected A100 Baseline | Membership Audit (c17_2) | Membership Audit (native redo) |
|--------|------------------------|--------------------------|-------------------------------|
| **Gaussian source** | Trained checkpoint (30k steps) | Raw SfM PLY | Raw SfM PLY |
| **Resolution** | Native (room 3114×2075) | **Resized** (1080×1080) | Native (room 3114×2075) |
| **Tile grid (room)** | 195×130 = 25,350 | 68×68 = 4,624 | 195×130 = 25,350 |
| **Tile grid (bicycle/garden)** | 310×206 = 63,860 / 325×211 = 68,575 | 120×68 = 8,160 | 310×206 = 63,860 / 325×211 = 68,575 |
| **Scale activation** | `torch.exp` applied | `torch.exp` applied | `torch.exp` applied |
| **Camera** | First camera, native K | First camera, **resized K** | First camera, native K |
| **Packed** | Yes | Yes | Yes |

### Why trained checkpoints produce vastly different n_isects

During 30k training steps:
- **Gaussian scales grow** to fill the scene (each Gaussian's 2D footprint expands)
- **Gaussians are pruned**: low-opacity Gaussians removed, survivors refined
- Result: **fewer visible Gaussians** (room: 16k vs 389k), each covering **many more tiles** (mean ≈ 1,585 vs 11.7)
- Total intersections: smaller count × much larger per-Gaussian footprint = far more intersections

### Quantitative Comparison

| Scene | Pipeline | Visible G | n_isects | Mean tiles/G | Tile grid |
|-------|----------|-----------|----------|-------------|-----------|
| room | Baseline (trained ckpt, native) | 16,076 | **25,475,247** | 1,584.8 | 195×130 |
| room | Audit (PLY, native) | 389,340 | **4,561,939** | 11.7 | 195×130 |
| room | Audit (PLY, resized 1080p) | 389,647 | **1,690,772** | 4.3 | 68×68 |
| bicycle | Baseline (trained ckpt, native) | 13,654 | **18,550,853** | 1,358.7 | 310×206 |
| bicycle | Audit (PLY, native) | 1,805,171 | **11,585,886** | 6.4 | 310×206 |
| bicycle | Audit (PLY, resized 1920×1080) | 1,805,534 | **4,861,253** | 2.7 | 120×68 |
| garden | Baseline (trained ckpt, native) | 100,582 | **11,257,252** | 111.9 | 325×211 |
| garden | Audit (PLY, native) | 2,247,791 | **11,671,990** | 5.2 | 325×211 |
| garden | Audit (PLY, resized 1920×1080) | 2,248,958 | **5,110,309** | 2.3 | 120×68 |

**Reconciliation status for each scene:**

| Scene | Reconciled? | Cause of remaining gap (PLY native vs trained ckpt) |
|-------|------------|------------------------------------------------------|
| room | **YES** (explained: different source) | Trained Gaussian scale explosion: 16k visible × 1,585 mean tiles vs 389k × 11.7 |
| bicycle | **YES** (explained: different source) | Same mechanism: 13.7k × 1,359 vs 1.8M × 6.4 |
| garden | **YES** (explained: different source) | Same mechanism: 100k × 112 vs 2.2M × 5.2. Garden is closest (ratio 1.04) because SfM PLY is denser for vegetation. |

### Corrected workload identity

The Corrected A100 Baseline uses **trained checkpoints** at **native resolution**. The membership audit initially used **raw PLY at resized resolution**. The native PLY collection (this audit) now provides a partial bridge, but **the true baseline workload is inaccessible locally** (checkpoints require A100). All cross-tile overlap analysis below uses the native PLY data as the best available proxy.

---

## 2. Arithmetic Consistency Verification

Previous apparent inconsistency resolved:

| Scene | Master G | Visible G | n_isects | Mean_tpg | n_visible × mean | n_isects − visible×mean |
|-------|---------|-----------|---------|---------|-----------------|------------------------|
| room | 1,593,376 | 389,340 | 4,561,939 | 11.7171 | 4,561,939 | **0.00** ✓ |
| bicycle | 6,131,954 | 1,805,171 | 11,585,886 | 6.4182 | 11,585,886 | **0.00** ✓ |
| garden | 5,834,784 | 2,247,791 | 11,671,990 | 5.1926 | 11,671,990 | **0.00** ✓ |

**No arithmetic bug.** The earlier apparent discrepancy (1,593,376 × 4.3 ≠ 1,690,772) was caused by multiplying the **master** Gaussian count by the mean over **visible** Gaussians only.

### n_isects definition (confirmed for both pipelines)

```
n_isects = sum_g tiles_per_gaussian(g)
        = isect_ids.numel()
        = flatten_ids.numel()
```

All three are identical in both the CUB-sorted and unsorted outputs (confirmed by `IntersectTile.cu` source: lines 102-115 where each tile intersection appends one `isect_id` and one `flatten_id`).

---

## 3. Tile Grid Reconciliation

| Scene | Resolution | Tile grid (16) | Total tiles | Notes |
|-------|-----------|----------------|-------------|-------|
| room (resized) | 1080 × 1080 | 68 × 68 | **4,624** | Used in original c17_2 audit |
| room (native) | 3114 × 2075 | 195 × 130 | **25,350** | Matches corrected baseline |
| bicycle (resized) | 1920 × 1080 | 120 × 68 | **8,160** | Used in original c17_2 audit |
| bicycle (native) | 4946 × 3286 | 310 × 206 | **63,860** | Matches corrected baseline |
| garden (resized) | 1920 × 1080 | 120 × 68 | **8,160** | Used in original c17_2 audit |
| garden (native) | 5187 × 3361 | 325 × 211 | **68,575** | Matches corrected baseline |

The "68,575" tile count is **garden at native resolution**, not a different scene. The "8,160" is garden at 1920×1080. These are **different workloads** — different resolution, different tile grid, and for the membership audit, different Gaussian source.

---

## 4. Camera / Image Dimension

Membership audit (both resized and native) uses:
- **I = 1** (single camera, packed mode)
- `tile_offsets` shape: `[1, th, tw]`
- `flatten_ids`: single 1D array for the single image
- In packed mode, Gaussian IDs are local to `[0, nnz)` where `nnz` is the number of Gaussians surviving projection (not the global Gaussian count). These are mapped to global ids via `gaussian_ids[flatten_ids]`.

The corrected baseline also uses `I = 1` (first camera, single view).

---

## 5. Cross-Tile Overlap (Native PLY)

### Horizontal Jaccard

| Scene | P25 | P50 | P75 | P90 | P95 | P99 | Mean | Pairs |
|-------|-----|-----|-----|-----|-----|-----|------|-------|
| room | 0.4390 | **0.6225** | 0.7391 | 0.8333 | 0.8921 | 0.9524 | 0.6073 | 25,220 |
| bicycle | 0.3114 | **0.5094** | 0.6667 | 0.7778 | 0.8318 | 0.9111 | 0.5354 | 63,654 |
| garden | 0.2762 | **0.4339** | 0.5294 | 0.6316 | 0.7032 | 0.7895 | 0.4381 | 68,364 |

### Vertical Jaccard

| Scene | P25 | P50 | P75 | P90 | P95 | P99 | Mean | Pairs |
|-------|-----|-----|-----|-----|-----|-----|------|-------|
| room | 0.4307 | **0.6096** | 0.7159 | 0.8013 | 0.8622 | 0.9452 | 0.6073 | 25,155 |
| bicycle | 0.2472 | **0.4198** | 0.5641 | 0.6881 | 0.7704 | 0.8485 | 0.4434 | 63,550 |
| garden | 0.2187 | **0.3590** | 0.4467 | 0.5417 | 0.6138 | 0.8092 | 0.3760 | 68,250 |

### Diagonal Jaccard (down-right, representative)

| Scene | P25 | P50 | P75 | P90 | P95 | P99 | Mean | Pairs |
|-------|-----|-----|-----|-----|-----|-----|------|-------|
| room | 0.2976 | **0.4583** | 0.5812 | 0.6842 | 0.7647 | 0.9091 | 0.4728 | 25,026 |
| bicycle | 0.1418 | **0.2920** | 0.4583 | 0.6000 | 0.6923 | 0.8000 | 0.3414 | 63,345 |
| garden | 0.1250 | **0.2340** | 0.3265 | 0.4118 | 0.4878 | 0.6902 | 0.2605 | 68,040 |

### Key observations

1. **Cross-tile overlap is substantial**: Horizontal Jaccard P50 ranges from 0.43 to 0.62 across scenes. This is a geometrically meaningful signal — it comes from Gaussians whose 2D footprints span adjacent tiles.

2. **Overlap varies by scene**: room > bicycle > garden. This correlates with point cloud density and scene complexity.

3. **Horizontal > Vertical**: Horizontal neighbors have higher overlap, consistent with typical camera aspect ratio (wider horizontal field → more horizontal tile coverage).

4. **Adjacent tile pairs are the most relevant**: ~25K-68K neighbor pairs analyzed per scene.

---

## 6. Relative Order Consistency

**Critical finding: 100% order consistency across all neighbor directions and all scenes.**

| Scene | Neighbor | Order agreement P50 | P99 | Pairs checked |
|-------|----------|-------------------|-----|---------------|
| room | H | **1.000000** | 1.000000 | 25,220 |
| room | V | **1.000000** | 1.000000 | 25,155 |
| room | DD | **1.000000** | 1.000000 | 25,026 |
| bicycle | H | **1.000000** | 1.000000 | 63,654 |
| bicycle | V | **1.000000** | 1.000000 | 63,550 |
| bicycle | DD | **1.000000** | 1.000000 | 63,345 |
| garden | H | **1.000000** | 1.000000 | 68,364 |
| garden | V | **1.000000** | 1.000000 | 68,250 |
| garden | DD | **1.000000** | 1.000000 | 68,040 |

### Why this is guaranteed

Both tiles use the same depth values (from the same projection). The CUB sort key is `depth_upper | (tile_id << 16)`. Within a tile, ordering is purely by `depth_upper`. A shared Gaussian has the same depth in both tiles, so its relative position among other shared Gaussians is preserved. **This is a tautological property of depth-sorted membership** — not a coincidence.

### Implications for reuse

Perfect order consistency means that if tile B's membership is represented as a delta from tile A's membership:
- The **shared subset** maintains the same relative order
- Only **insertions and deletions** need position information
- No reordering metadata is needed

---

## 7. Reuse Potential

For the "shared set as base, add/remove delta" representation:

| Scene | Direction | B's membership that can be reused from A (P50) | B's membership that must be added (P50) |
|-------|-----------|-----------------------------------------------|----------------------------------------|
| room | H → | 57.9% | 42.1% |
| room | V → | 59.8% | 40.2% |
| bicycle | H → | 50.4% | 49.6% |
| bicycle | V → | 44.1% | 55.9% |
| garden | H → | 42.8% | 57.2% |
| garden | V → | 38.4% | 61.6% |

**Interpretation**: For the median adjacent tile pair, ~43-60% of the Gaussian IDs are shared. The set-difference delta (additions + removals) would be ~40-57% of the full membership list. This is a substantial reduction, but:
- The deltas themselves would still need to be stored
- The backward kernel would need set-difference logic (non-trivial GPU implementation)
- Cross-tile dependencies complicate tile-independent processing

---

## 8. Representation Properties (Native PLY, Correct Workload)

| Property | Result | Research Value |
|----------|--------|----------------|
| **Delta locality** | NEGLIGIBLE: ∣Δ∣≤4 in 0.04-0.06%, median |Δ| = 26–745 | **None** |
| **Run-length** | NEGLIGIBLE: max run = 2–3, <6K runs across millions | **None** |
| **ID range** | FULL: per-tile range ≈ full ID space (P50 range 1.6M–6.1M) | **None** |
| **Cross-tile overlap** | **SUBSTANTIAL**: Jaccard P50 0.36–0.62 (horizontal/vertical) | **Moderate** |
| **Order consistency** | **PERFECT**: 100% for all shared pairs | **Enabling** |
| **Duplicate** | ZERO: no within-tile duplicates | **None** |
| **Bit-width** | MARGINAL: 9-11 unused bits per int32 (23-34%) | **Low** |

### What makes cross-tile overlap different

Unlike delta encoding and run-length (which fail because depth-sorted order destroys Gaussian-ID locality), cross-tile overlap is a **geometric property**: the same Gaussian projects onto multiple adjacent tiles. This property is independent of the sort order and is preserved regardless of how membership is represented. The perfect order consistency further enables a set-difference representation where only insertions/removals need to be tracked.

---

## 9. C17-2 Status

### Decision: **CONTINUE WITH REDESIGN**

### Reason

The original C17-2 hypothesis was that **within-tile** membership has exploitable structure (delta locality, run-length). This is **falsified** — the within-tile structure is effectively random in Gaussian-ID space.

However, **cross-tile overlap** is a qualitatively different property:
- **Substantial** (Jaccard P50 0.36–0.62; B's reuse from A: 38–60%)
- **Spatially consistent** (horizontal > vertical > diagonal, stable across scenes)
- **Order-preserving** (100% shared-pair order consistency)
- **Geometrically grounded** (Gaussian 2D footprint spans tiles)

This suggests a **redesign** of C17-2 from "within-tile compression" to **"cross-tile differential membership"**: represent each tile's membership as a reference to its neighbor's membership plus a small set of insertions/removals.

### Research question for redesigned C17-2

> Can the ordered membership representation be reduced by encoding tile-to-tile membership differences instead of storing each tile's full sorted list independently?

### Key challenges for the redesign

1. **GPU implementation complexity**: Set-difference logic in the backward kernel
2. **Cross-tile dependencies**: Tile processing is no longer independent
3. **Delta size distribution**: Need to model the insertion/removal set sizes
4. **Reference tile selection**: Which neighbor(s) to reference for maximum reuse

---

## 10. Appendix: Delta Locality (Native PLY, Deferred to Prior Findings)

| Scene | Pairs | ∣Δ∣ mean | ∣Δ∣ P50 | ∣Δ∣ ≤ 4 | ∣Δ∣ ≤ 16 | LEB128 mean bytes |
|-------|-------|---------|--------|---------|----------|-------------------|
| room | 4,536,589 | 13,475 | 26 | 0.07% | 0.12% | ~4.0 |
| bicycle | 11,522,026 | 27,778 | 150 | 0.09% | 0.13% | ~4.0 |
| garden | 11,603,415 | 21,061 | 745 | 0.06% | 0.09% | ~4.0 |

Confirming: within-tile delta encoding is not viable.

---

*Data collected from native-resolution PLY at `gsplat.rasterization()` with `tile_size=16`, `packed=True`, activated scales. Cross-tile overlap computed on 25K–68K adjacent tile pairs per scene. Order consistency verified over all shared Gaussian pairs ≥2.*
