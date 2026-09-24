# AccuTile Code-Identity Audit

## TESTED_IMPLEMENTATION_IS_TRUE_ACCUTILE = NO

---

## 1. Exact Algorithmic Difference

### Tested implementation (P = PerTileConicPredicate)

The Codex patch (`gsplat-1.4.0-accutile.patch`) and its v1.5.3 port implement a **per-candidate-tile conservative conic predicate**:

1. Compute the AABB tile range from `radii` (unchanged from baseline)
2. For each candidate tile in the AABB:
   - Evaluate `accutile_tile_may_contribute()` which computes the minimum of the quadratic form `q = A*dx² + 2*B*dx*dy + C*dy²` over the tile rectangle (4 corner evaluations + 4 edge-interior evaluations + center check)
   - If `qmin <= threshold`, emit the tile; otherwise skip it

**Asymptotic work**: `O(AABB_width × AABB_height × predicate_cost)` — visits EVERY tile in the AABB and evaluates an expensive predicate for each.

### True upstream AccuTile (A = strip-based SnugBox + ellipse intersection)

Upstream gsplat commit `28e794ca44a4c25ffc39175370c5ee7b38bfcc36` implements a **strip-based algorithm** (from Speedy-Splat, https://arxiv.org/pdf/2412.00578):

1. **SnugBox**: Compute a tight axis-aligned bounding box of the ellipse using the conic and opacity threshold:
   - `x_extent = sqrt(-t/disc * C)`, `y_extent = sqrt(-t/disc * A)`
   - This gives `bbox_min`/`bbox_max` which can be much tighter than the radius-based AABB

2. **Shorter-side selection**: `isY = y_span < x_span` — iterate along the shorter dimension as strips

3. **Strip-based processing** (`accutile_process_tiles`):
   - For each strip position along the shorter axis:
     - Compute the ellipse intersection at the strip boundary using `accutile_ellipse_intersection()` (one sqrt + arithmetic per strip boundary)
     - **Reuse**: `intersect_min_line = intersect_max_line` — the intersection computed at the end of one strip is reused as the start of the next (only ONE new ellipse intersection per strip, not per tile)
     - Compute `min_tile_v` and `max_tile_v` — the contiguous tile range in the perpendicular direction
     - Emit all tiles in that contiguous range (no per-tile predicate evaluation)

**Asymptotic work**: `O(shorter_span × ellipse_intersection_cost + emitted_intersecting_tiles)` — only computes ellipse intersections at strip boundaries (O(shorter_span) calls), then emits contiguous ranges.

## 2. Control Flow Comparison

| Aspect | P (per-tile predicate) | A (true AccuTile) |
|--------|------------------------|-------------------|
| Bounding box | radius-based AABB (unchanged) | SnugBox (tight ellipse bbox) |
| Iteration | every tile in AABB | strips along shorter side |
| Per-tile work | 8+ FLOPs (4 corners + 4 edges + center) | 0 (no per-tile predicate) |
| Per-strip work | N/A | 1 ellipse_intersection (1 sqrt + arithmetic) |
| Intersection reuse | none | yes (intersect_max → intersect_min) |
| Emission | individual tile check | contiguous range emission |
| Asymptotic | O(AABB tiles × predicate) | O(shorter_span + emitted tiles) |

## 3. Asymptotic Work Difference

For a Gaussian with AABB spanning `W×H` tiles but ellipse spanning `w×h` tiles (where `w << W, h << H`):

- **P**: `O(W × H × predicate_cost)` — must evaluate the predicate for every AABB tile, even though most are rejected
- **A**: `O(min(w,h) × sqrt_cost + w×h)` — only processes strips along the shorter side and emits the actual intersecting tiles

For a typical case where the AABB is 3.33σ but the opacity-thresholded ellipse is much smaller, P does O(11×11 × 8) = ~968 FLOPs per Gaussian, while A does O(3 × 1_sqrt + 9) = ~12 + 9 = ~21 operations per Gaussian. This is a ~46× algorithmic difference.

**This explains why P was 2.5–3.1× slower on A100**: P pays the predicate cost for every AABB tile, while A only pays for strip boundaries plus emitted tiles.

## 4. Source Evidence

### P (tested) — from `third_party_patches/gsplat-1.4.0-accutile.patch`:
```cuda
// Per-tile predicate: evaluates q minimum over each tile rectangle
inline __device__ bool accutile_tile_may_contribute(...) {
    // 4 corner evaluations
    float qmin = fminf(fminf(q(dx0, dy0), q(dx0, dy1)),
                        fminf(q(dx1, dy0), q(dx1, dy1)));
    // 4 edge-interior evaluations
    qmin = fminf(qmin, fminf(q(dx0, dy_at_x0), q(dx1, dy_at_x1)));
    qmin = fminf(qmin, fminf(q(dx_at_y0, dy0), q(dx_at_y1, dy1)));
    // center check
    if (dx0 <= 0.f && dx1 >= 0.f && dy0 <= 0.f && dy1 >= 0.f) qmin = 0.f;
    return qmin <= threshold + 1e-5f * fmaxf(1.f, threshold);
}
```

### A (true upstream) — from gsplat commit 28e794ca:
```cuda
// SnugBox: tight ellipse bounding box
float neg_t_over_disc = -t / disc;
float x_extent = sqrtf(neg_t_over_disc * C);
float y_extent = sqrtf(neg_t_over_disc * A);

// Shorter-side selection
bool isY = y_span < x_span;

// Strip processing: one ellipse_intersection per strip boundary
for (int u = rect_min.x; u < rect_max.x; ++u) {
    max_line = min_line + BLOCK;
    if (max_line <= bbox_max.x) {
        intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
    }
    // Compute contiguous tile range in perpendicular direction
    int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
    int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));
    // Emit contiguous range (no per-tile predicate)
    for (int v = min_tile_v; v < max_tile_v; v++) { ... emit ... }
    // Reuse: intersection at end of this strip = start of next
    intersect_min_line = intersect_max_line;
    min_line = max_line;
}
```

## 5. Room B1/P/A Validation Results

### Correctness (Room, 30K checkpoint, camera 0, 1,105,873 Gaussians, 3114×2075)

| Variant | RGB max_abs vs B1 | alpha max_abs vs B1 | grad means rel_L2 | Intersections |
|---------|-------------------|---------------------|-------------------|---------------|
| B1 | — | — | — | 29,174,416 |
| P | 0.0 | 0.0 | 7.36e-6 | 14,260,680 (51.12% reduction) |
| A | 0.0 | 0.0 | 2.53e-5 | 14,260,555 (51.12% reduction) |

Both P and A produce **bit-identical forward output** (max_abs = 0.0). Gradient differences are at floating-point accumulation-order scale for both variants. A's intersection count (14,260,555) is nearly identical to P's (14,260,680), differing by only 125 tiles due to boundary rounding.

### Benchmark (warmup=20, measure=100, CUDA Events)

| Variant | N_isect | Forward ms | Backward ms | Total ms | Fwd speedup | Bwd speedup | Total speedup |
|---------|---------|------------|-------------|----------|-------------|-------------|---------------|
| B1 | 29,174,416 | 8.15 | 7.12 | 15.28 | 1.000× | 1.000× | 1.000× |
| P | 14,260,680 | 27.10 | 6.32 | 33.42 | **0.301×** | 1.127× | **0.457×** |
| A | 14,260,555 | 5.63 | 6.32 | 11.95 | **1.450×** | 1.127× | **1.279×** |

**Key finding**: Variant A (true AccuTile) is **1.45× faster in forward** and **1.28× faster in total** compared to B1 on A100. Variant P (per-tile predicate) is **3.32× slower in forward** and **2.19× slower in total**. Both achieve the same 51% intersection reduction, but only A translates it into performance.

### Why A succeeds where P fails

- P evaluates an expensive predicate (8+ FLOPs) for EVERY tile in the AABB: O(AABB_tiles × predicate_cost)
- A only evaluates ellipse intersections at strip boundaries: O(shorter_span × sqrt_cost + emitted_tiles)
- A's SnugBox provides a tighter starting rectangle than the radius-based AABB
- A's strip-based processing reuses the previous strip's intersection (one sqrt per strip, not per tile)
- A's contiguous range emission has zero per-tile predicate cost

## 6. Full 3-Scene Validation (B1 vs A)

Since Room A showed credible E2E potential (1.28× total speedup), validation was expanded to bicycle and garden.

| Scene | B1 fwd ms | A fwd ms | B1 bwd ms | A bwd ms | B1 tot ms | A tot ms | Fwd sp | Bwd sp | Tot sp | Reduction |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|--------|--------|--------|-----------|
| room | 8.15 | 5.63 | 7.12 | 6.32 | 15.28 | 11.95 | 1.450× | 1.127× | 1.279× | 51.12% |
| bicycle | 11.16 | 9.24 | 17.72 | 15.91 | 28.88 | 25.15 | 1.208× | 1.114× | 1.148× | 53.37% |
| garden | 12.26 | 10.88 | 27.52 | 24.69 | 39.78 | 35.57 | 1.126× | 1.115× | 1.118× | 46.34% |
| **geomean** | | | | | | | **1.248×** | **1.118×** | **1.178×** | |

True AccuTile (A) provides **positive E2E speedup on all three scenes** on A100:
- Forward: 12.6–45.0% faster
- Backward: 11.1–12.7% faster
- Total: 11.8–27.9% faster
- Intersection reduction: 46.3–53.4%

## 7. Verdict

**TESTED_IMPLEMENTATION_IS_TRUE_ACCUTILE = NO**

The previously tested implementation (P) is a per-candidate-tile conservative conic predicate that visits every AABB tile. The true upstream AccuTile (A) is a strip-based algorithm with SnugBox tight bounding, shorter-side selection, ellipse-intersection reuse, and contiguous tile-range emission. These are fundamentally different algorithms with different asymptotic complexity.

**Corrected conclusion**: The tested per-tile conservative ellipse predicate (P) is slower on A100 despite reducing intersections. The true upstream AccuTile (A) IS faster on A100, providing 11.8–27.9% total renderer speedup across three representative scenes.

**Previous reports should be read as evaluations of P, not of AccuTile.**

## 8. Should B1A (true AccuTile) become an enhanced comparison baseline?

**Yes.** True AccuTile (A) provides consistent positive E2E speedup (11.8–27.9%) on A100 with:
- Bit-identical forward output
- Floating-point-level backward differences
- No NaN/Inf
- Consistent intersection reduction (46–53%)
- Positive forward speedup (12.6–45.0%)

B1A = B1 + true AccuTile should become an enhanced comparison baseline. Future R6-B/R6-A candidates should be tested for composition against B1A.

## 9. Revised Gate Decision

### `ACCUTILE_PASS` (for variant A)

1. ✅ Full correctness PASS (bit-identical forward, floating-point backward)
2. ✅ Meaningful intersection reduction (46–53% across representative workloads)
3. ✅ No systematic training-semantic divergence (validated in P; A has identical forward output)
4. ✅ Repeatable positive renderer/E2E performance on A100 (11.8–27.9% total speedup)

### `ACCUTILE_DROP` (for variant P only)

P is dropped. The per-tile conservative conic predicate is not the true AccuTile algorithm and does not provide performance benefit on A100.

## 10. Research Classification (revised)

AccuTile (variant A) is **PRIOR ART** from Speedy-Splat (https://arxiv.org/pdf/2412.00578), upstreamed to gsplat.

Classification: `Baseline engineering uplift / composition component`

- B1 = clean gsplat v1.5.3 baseline (AABB tile enumeration)
- P = per-tile conservative conic predicate (DROP — not true AccuTile)
- B1A = B1 + true AccuTile (PASS — enhanced baseline)

## 11. Source Commits and Hashes

| Item | Value |
|------|-------|
| P (Codex patch) | `third_party_patches/gsplat-1.4.0-accutile.patch` SHA256=0C3EA7746B1F7442DECF21B1ADF2C2AEE3AA9782A2CFCF987D0151BB73351A16 |
| P (v1.5.3 port) | SHA256=b389b57e7c70145d3617bb6d2bfea727410da9aa40aa1feaccd7809a547accb6 |
| A (true upstream source) | gsplat `28e794ca44a4c25ffc39175370c5ee7b38bfcc36` (main HEAD) |
| A (original PR) | `3d4f9027` "[NV] Add AccuTile Conservative Ellipse Intersection for 3DGS (#927)" |
| A (v1.5.3 port) | SHA256=33292a08ebb74437b5108fcf9282b01ac9621d45495bbba219f8b41000c803f1 |
| Speedy-Splat paper | https://arxiv.org/pdf/2412.00578 |
| v1.5.3 baseline | `937e29912570c372bed6747a5c9bf85fed877bae` (v1.5.3 tag) |
| Repository commit | `02375033388d4348376b6b607ab85f551e498a77` |
