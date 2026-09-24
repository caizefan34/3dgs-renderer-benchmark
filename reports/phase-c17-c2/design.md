# C17-2 Design: Rasterization Early Batch Termination (Active Pixel Tracking)

**Date:** 2026-10-19  
**Phase:** C17-2 Design (before implementation)  
**Audit Reference:** `source_audit.md`  
**Baseline:** gsplat v1.5.3 (true baseline — 46-bit key, 12 CUB passes. **C1 not deployed.**)

---

## 1. Hypothesis

> **The forward rasterization kernel processes a significant fraction (30–70%) of Gaussians in each tile that contribute zero visible energy to individual pixels, because those pixels' transmittance has already dropped below the `1e-4` termination threshold. Tracking the per-pixel active mask and dynamically terminating batches for saturated pixels will reduce per-pixel Gaussian processing load, and hence reduce total kernel execution time, without changing any numerical output.**

### SUPPORTING EVIDENCE (from audit)

From Phase 17B capacity analysis:

| Scene | tile_size | per-tile mean | per-tile max | n_isects | Estimated saturation fraction |
|-------|:---------:|:-------------:|:------------:|:--------:|:----------------------------:|
| room | 16 | 490 | 2,996 | 2.27M | Sky/background tiles → early sat; dense tiles → late sat |
| bicycle | 16 | 746 | 6,882 | 6.08M | Significant tail: top-decile tiles with >2,515 Gaussians |
| garden | 16 | 751 | 4,361 | 6.13M | Similar to bicycle |

**Key insight**: The mean per-tile Gaussian count is 490–751, but tiles in dense regions (foliage, complex geometry) have thousands more. However, each pixel within a tile saturates at a different Gaussian. The pixel with the *last* saturation determines when the tile ends — but all other pixels continue to process Gaussians they'll never see.

### Expected workload reduction

```
Tile Pixels:            [P0][P1][P2][P3]...[Pn]
Saturation Gaussian#:   [42][300][15][800]...[1]

Current work:  max(saturation) × n_pixels ≈ 800 × n_pixels
C17-2 work:    Σ(saturation)               ≈ 42+300+15+800+...+1
                     = Σ(saturation)
                     = n_pixels × mean(saturation)

Reduction ratio:  mean(saturation) / max(saturation)
```

For typical tiles, mean saturation depth is significantly lower than max saturation depth (often 40–60% lower), giving an estimated **20–40% workload reduction** in the compositing loop.

---

## 2. Bottleneck

### Current rasterization kernel inner loop

```cuda
// Forward kernel — per-tile, per-batch
for (uint32_t b = 0; b < num_batches; ++b) {
    if (__syncthreads_count(done) >= block_size) break; // Only exits when ALL pixels done
    
    // Each thread loads 1 Gaussian
    uint32_t idx = batch_start + tr;
    if (idx < range_end) {
        g = flatten_ids[idx];
        id_batch[tr] = g;
        xy_opacity_batch[tr] = {means2d[g], opacities[g]};
        conic_batch[tr] = conics[g];
    }
    block.sync();
    
    // Each thread processes ALL Gaussians in batch
    for (uint32_t t = 0; t < batch_size && !done; ++t) {
        // compute alpha, compositing...
        if (next_T <= 1e-4f) done = true; // pixel-done — but still loops through batch!
    }
}
```

**Critical inefficiency**: The `done` flag only prevents processing **further batches** for that pixel, but **within each batch**, the pixel still iterates through all Gaussians up to the one that makes it saturate. Furthermore, the batch-level `__syncthreads_count(done) >= block_size` check only fires when **all 256 pixels in the tile** are done — which rarely happens for tiles with any foreground/background mix.

### GPU occupancy impact

- Tile block size = 256 threads (one warp per 8 pixels, 8 warps per block)
- When only 1–2 warps still have active pixels, the other 6–7 warps are **wasted** on Gaussians they'll never composite
- This wastes both compute (warp schedulers issue instructions for inactive threads) and memory bandwidth (unnecessary Gaussian attribute loads)
- On A100 with 108 SMs, 8,160 tile blocks = ~76 blocks per SM → severe oversubscription makes each warp's inefficiency multiply

---

## 3. Proposed Mechanism

### 3.1 Active Pixel Mask Tracking

Add a **shared-memory active pixel set** to the forward kernel:

```
┌──────────────────────────────────────────────┐
│  Shared Memory Layout (current → C17-2)      │
│                                              │
│  [id_batch] [xy_opacity_batch] [conic_batch] │
│  └─────── current (unchanged) ───────┘       │
│                                              │
│  NEW: [active_set] — uint32_t[8] = 256 bits  │
│       1 bit per pixel in tile (tile_size²)   │
│       bit=1: pixel still compositing         │
│       bit=0: pixel saturated (done)          │
│                                              │
│  NEW: [active_count] — uint32_t              │
│       Popcount of active_set                 │
└──────────────────────────────────────────────┘
```

### 3.2 Algorithm Change

```
Phase 1 (current, unchanged):
  - Initialize active_set = all 1s (all pixels active)
  - active_count = block_size

Phase 2 (modified loop):
  For each batch b:
    Each thread loads 1 Gaussian (same as current)
    block.sync()
    
    For each Gaussian t in batch:
      if (active_set[thread] == 1) AND thread pixel is inside image:
        Compute alpha, compositing (same as current)
        if (next_T <= 1e-4f):
          active_set[thread] = 0  // mark pixel as done
          atomicSub(active_count, 1)
    
    block.sync()
    
    // NEW: Early batch termination check
    if (active_count == 0):
        break  // all pixels done — no more batches needed
    
    // NEW: Optional sparse-mode switch
    if (active_count < SPARSE_THRESHOLD):
        Switch to warp-gather mode: only active pixels load Gaussians
        // (Phase 2 refinement — not in initial implementation)
```

### 3.3 Sparse Mode (Phase 2 Refinement)

When `active_count < SPARSE_THRESHOLD` (e.g., 32 active pixels = 1 warp worth):
- Instead of having all 256 threads load Gaussians (wasting 224 threads)
- Only the active threads participate in loading
- Use a warp-reduce to determine which pixels still need data
- The remaining inactive threads can be repurposed or the tile exits earlier

**Note**: This is optional and would be evaluated separately. The core C17-2 is just the active mask + early termination.

---

## 4. Expected Workload Reduction

### 4.1 Analytical Model

For a tile with `N` Gaussians and `tile_size²` pixels:

**Current work** (worse case): `N × tile_size²` × (alpha computation)
**C17-2 work**: `Σ_{pixel} saturation_depth(pixel)` × (alpha computation)

**Worst case** (all pixels saturate at the same Gaussian): same as baseline (no reduction)
**Best case** (large disparity in saturation depths): up to 50–60% reduction
**Typical case** (mixed foreground/background): 20–40% reduction

### 4.2 Per-Scene Estimate

| Scene | tile | Mean per-tile | Est. reduction | Fwd time saved | Est. fwd speedup |
|-------|:----:|:-------------:|:--------------:|:--------------:|:----------------:|
| room | t16 | 490 | ~25% | ~0.7ms (of 7.5ms) | ~1.09× |
| bicycle | t16 | 746 | ~35% | ~4.6ms (of 25.9ms) | ~1.18× |
| bicycle | t32 | 1,737 | ~40% | ~6.2ms (of 37.2ms) | ~1.17× |

**Key insight**: The scenes with the most Gaussians per tile (bicycle, garden) benefit the most because they have the longest tail of Gaussians that many pixels never see.

### 4.3 Workload Metric Reduction

| Metric | Baseline | C17-2 Expected |
|--------|----------|----------------|
| Total Gaussian-pixel evaluations | n_isects × tile_size² | Σ_pixel saturation_idx(pixel) |
| For room t16 (estimate) | 2.27M × 256 = 581M | ~435M (25% fewer) |
| For bicycle t16 (estimate) | 6.08M × 256 = 1.56B | ~1.01B (35% fewer) |
| For garden t16 (estimate) | 6.13M × 256 = 1.57B | ~1.02B (35% fewer) |

---

## 5. Memory Impact

### 5.1 Device Memory
- **None** — no new device allocations. The active set is in **shared memory** only.
- Total shared memory increase: `8 × uint32_t + 1 × uint32_t = 36 bytes`
- Current shared memory: `tile_size² × (4 + 12 + 12) = 256 × 28 = 7,168 bytes` (tile_size=16)
- Increase: `36 / 7,168 = 0.5%` — negligible

### 5.2 Shared Memory Impact
- No change to `max_dynamic_shared_memory_size` — well within the 48 KB per-block limit
- No impact on occupancy (shared memory usage still dominated by id_batch + xy_opacity_batch + conic_batch)

---

## 6. Synchronization Impact

- **Added**: One `block.sync()` after `active_count` update per Gaussian in the batch (to ensure mask consistency before next iteration)
- **Added**: One `block.sync()` before early-exit check (to ensure all threads agree on active_count before decision)
- **Removed**: None

**Net change**: 1 extra `block.sync()` per batch (compared to current 2 per batch: after load, before compositing). The termination check `block.sync()` can be merged with the existing post-compositing sync.

**Analysis**: The `__syncthreads_count(done)` call is **replaced** with a faster shared-memory atomic counter query. `__syncthreads_count` requires a full warp-barrier and reduction across all warps in the block. A simple shared `active_count` read is a single `ld.shared` instruction.

---

## 7. Correctness Risk

### Risk Assessment: VERY LOW

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Wrong pixel output for saturated pixels | Extremely low | High | The active_set only prevents processing of pixels that already have `T < 1e-4`. Once `T` drops below threshold, no additional Gaussian can contribute (by definition: `next_T = T × (1-α)` can only stay ≤ T, and `new_color = color × α × T` becomes vanishingly small). |
| Off-by-one in early termination | Low | Medium | The termination check is after processing each batch. All pixels in the batch are fully processed before checking. No partial-batch processing. |
| Race condition in active_set update | Low | Low | `atomicSub` ensures correct counter update. Threads only modify their own bits. |

### Mathematical Guarantee

Since each pixel's `T` is monotonically decreasing and the threshold `1e-4` is the same as the baseline termination condition, **any pixel that exits early would have produced identical output to baseline**. The forward pass is mathematically identical — only fewer iterations per pixel.

---

## 8. Differentiability Risk

### Risk Assessment: NONE

- `last_ids` (the index of the last Gaussian that contributed per pixel) is **unchanged** — early termination does not change which Gaussian was last to contribute, because the last-contributing Gaussian is determined by `next_T > 1e-4f` (inclusive) threshold crossing, which is the same condition that triggers active_set clearing.
- The backward kernel reads `last_ids` and `render_alphas` — both unchanged.
- No change to the autograd graph structure.

**Proof**: If a pixel saturates at Gaussian `k` (i.e., `T_after_k < 1e-4`), then:
- Baseline sets `done = True` and `cur_idx = batch_start + t` (where t is the index within the batch)
- C17-2 sets `active_set[pixel] = 0` at the same point
- Both stop compositing further Gaussians
- Both record the same `last_ids[pixel]`
- The backward kernel reads `last_ids` → same behavior

---

## 9. Failure / Fallback Path

### 9.1 Worst-Case Scenario
If `active_count` tracking introduces overhead but the scene has no early-saturating pixels (all pixels in all tiles saturate at roughly the same Gaussian), C17-2 adds overhead without benefit.

**Fallback**: A compile-time or runtime flag `USE_ACTIVE_SET` to disable the optimization:
```cuda
// At compile time:
template <bool USE_ACTIVE_SET>
__global__ void rasterize_to_pixels_3dgs_fwd_kernel(...);

// With runtime fallback:
if (use_active_set) {
    kernel<true><<<grid, block, shmem>>>(...);
} else {
    kernel<false><<<grid, block, shmem>>>(...);
}
```

### 9.2 Overhead Quantification
| Operation | Overhead |
|-----------|:--------:|
| Per-Gaussian active_set check | 1 bit test + 1 branch |
| Per-pixel-saturation atomicSub | 1 global atomic per pixel |
| Per-batch active_count read | 1 shared memory load |
| Per-batch early-exit branch | 1 compare + 1 branch |

**Maximum overhead estimate**: < 1% of kernel time (one extra ALU op per Gaussian per pixel).

---

## 10. Expected Performance Signature

### 10.1 When C17-2 Wins

| Condition | Improvement | Explanation |
|-----------|:-----------:|-------------|
| High per-tile Gaussian count | Larger | More "tail" Gaussians that many pixels never see |
| Mixed foreground/background tiles | Largest | Sky pixels saturate early on first opaque Gaussians |
| Large tile size (t32) | Moderate | More pixels per tile, but foreground might be denser |
| High resolution | Same % | Scaling with resolution maintains ratio |
| Scenes with depth variation | Larger | More disparity in saturation depths across pixels |

### 10.2 When C17-2 Loses (or breaks even)

| Condition | Impact | Explanation |
|-----------|:------:|-------------|
| All pixels saturate at same Gaussian (e.g., single flat surface) | ~0% | No reduction possible |
| Very small per-tile counts (< 100) | ~0% | Loop overhead dominates |
| Sparse scenes with few Gaussians per tile | ~0% | No tail to truncate |

### 10.3 Expected Metrics

**SUPPORTED (from data analysis)**:
- The forward rasterization kernel processes `n_isects × 256` Gaussian-pixel evaluations per kernel launch
- For bicycle t16: 6.08M × 256 = 1.56B evaluations
- C17-2 can skip evaluations for pixels already below T < 1e-4
- Even a 10% reduction → 156M fewer evaluations → measurable time saving

**HYPOTHESIS (to be measured)**:
- Room t16: ~1.09× forward speedup, ~1.03× E2E speedup
- Bicycle t16: ~1.18× forward speedup, ~1.08× E2E speedup  
- Overall training speedup (500-step): ~1.03–1.08× depending on scene

---

## Implementation Constraints

### Minimal / Isolated / Reversible

1. **Minimal**: Modify only the forward rasterization kernel (`RasterizeToPixels3DGSFwd.cu`)
2. **Isolated**: Template parameter `USE_ACTIVE_SET` enables/disables at compile time
3. **Reversible**: Baseline path = `USE_ACTIVE_SET=false` — bit-exact to current output
4. **Graduated**: Only implement active_set tracking + early batch termination in Phase 1. Sparse-mode warp-gather is Phase 2 (if justified).

### Debug Counters (to be added)

All compiled out when USE_ACTIVE_SET=false. When enabled:

| Counter | Location | Purpose |
|---------|----------|---------|
| `total_batches_processed` | Per-launch | Total batches across all tiles |
| `batches_saved_by_early_exit` | Per-launch | Batches skipped due to active_count==0 |
| `gaussians_loaded` | Per-launch | Total Gaussian loads |
| `gaussians_skipped` | Per-launch | Gaussian loads avoided by early termination |
| `active_pixels_per_tile` | Per-tile (trace mode) | Distribution of active pixels at exit |

All counters stored in pinned host-visible memory for retrieval per forward call.

---

## Composability with C17-1

C17-2 and C17-1 are **orthogonal optimizations**:

| Aspect | C17-1 | C17-2 | Combined |
|--------|-------|-------|----------|
| Target | Intersection + sort | Rasterization | Both |
| Data path | Changes `isect_ids`/`flatten_ids` → per-tile buffers | No change | Compatible (C17-2 reads from whatever C17-1 provides) |
| Memory | Large increase (per-tile buffers) | Negligible | Large increase from C17-1 |
| Benefit | Reduce sort time | Reduce rasterization time | Additive |
| Correctness | Proven independent | Independent verification needed | Same as individual |

**No interference expected.** C17-2 only cares about per-tile Gaussian count and ordering — both are preserved by C17-1.

---

*End of Design — proceed to Pre-Implementation Report*
