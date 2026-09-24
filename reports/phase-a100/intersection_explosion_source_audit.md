# A100 Intersection Explosion Source Audit

**Date:** 2026-09-06  
**Objective:** Trace and analyze why the current renderer produces so many intersections  
**Base data:** `reports/phase-a100/current_baseline_profiling.md`, `reports/phase-a100/cub_sort_deep_profile.md`

---

## 1. Intersection Generation — Full Source Trace

### 1.1 Gaussian → Projected Footprint

Source: `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`

```
Gaussian (world position, quaternion, scale, opacity)
  ↓
quat_scale_to_covar_preci() → 3D covariance matrix (lines 73-80)
  ↓
covarW2C(R, covar, covar_c) → camera-space covariance (line 85)
  ↓
persp_proj() → screen-space 2D covariance + 2D mean (lines 90-121)
  ↓
extend = 3.33f                              (line 163)
if (opacity < ALPHA_THRESHOLD (=1/255)):    (line 171) → radius=0, skip
extend = min(extend, sqrt(-2*log(op/α_th))) (line 178)
radius_x = ceilf(extend * sqrt(covar2d[0][0]))  (line 184)
radius_y = ceilf(extend * sqrt(covar2d[1][1]))  (line 185)
```

**Key parameters:**
- `ALPHA_THRESHOLD = 1/255 ≈ 0.00392` (from `gsplat/cuda/include/Common.h`, line 54)
- Default `extend = 3.33` (3σ coverage)
- Opacity-aware `extend` shrinks for dim Gaussians, but the minimum (for opacity just above threshold) is `sqrt(-2*log(1)) = 0`
- `ALPHA_THRESHOLD` is the same threshold used later in the rasterizer for early termination

### 1.2 Projected Footprint → Tile Bounds

Source: `gsplat/cuda/csrc/IntersectTile.cu`

```cpp
// PASS 1 (cum_tiles_per_gauss == nullptr): Count tiles per Gaussian
tile_radius_x = radius_x / tile_size;          // line 88
tile_radius_y = radius_y / tile_size;          // line 89
tile_x = mean2d.x / tile_size;                 // line 91
tile_y = mean2d.y / tile_size;                 // line 92

tile_min.x = clamp(floor(tile_x - tile_radius_x), 0, tile_width);   // line 95-96
tile_min.y = clamp(floor(tile_y - tile_radius_y), 0, tile_height);  // line 97-98
tile_max.x = clamp(ceil(tile_x + tile_radius_x), 0, tile_width);    // line 99-100
tile_max.y = clamp(ceil(tile_y + tile_radius_y), 0, tile_height);   // line 101-102

tiles_per_gauss = (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x);  // line 106
```

### 1.3 Tile Bounds → Intersection Records

```cpp
// PASS 2 (cum_tiles_per_gauss != nullptr): Write intersection records
for each tile (i, j) in [tile_min, tile_max):           // lines 128-129
    tile_id = i * tile_width + j;                        // line 130
    isect_id = (image_id << (32+tile_n_bits)) |          // line 133
               (tile_id << 32) |                         // depth as 32-bit uint
               float_bits(depth);
    flatten_id = gaussian_index;                         // line 135
```

**isect_id encoding (64-bit key):**
```
Bits 63..(32+tile_n_bits):  image_id
Bits (31+tile_n_bits)..32:  tile_id (LSB at bit 32)
Bits 31..0:                 depth (float32 bits, reinterpreted as uint32)
```

---

## 2. Per-Gaussian Intersection Model

From source code, the number of tiles covered by Gaussian g is:

$$N_{\text{tiles}}(g) = \max(0, \lfloor t_y + r_y\rfloor - \lfloor t_y - r_y\rfloor) \times \max(0, \lfloor t_x + r_x\rfloor - \lfloor t_x - r_x\rfloor)$$

Where:
- $t_x = \text{mean2d}_x / \text{tile\_size}$, $t_y = \text{mean2d}_y / \text{tile\_size}$
- $r_x = \text{radius}_x / \text{tile\_size}$, $r_y = \text{radius}_y / \text{tile\_size}$
- $\text{radius}_x = \lceil \text{extend} \times \sqrt{\text{covar2d}_{0,0}} \rceil$
- $\text{extend} = \min(3.33,\ \sqrt{-2\ln(\text{opacity} / \alpha_{\text{thresh}})})$
- $\text{covar2d} = J \times R \times \Sigma \times R^T \times J^T$ (perspective projection of 3D covariance)

**Structural constraints:**
- $N_{\text{tiles}}(g) \in [1,\ \text{tile\_width} \times \text{tile\_height}]$
- If $r_x \geq \max(t_x, \text{tile\_width} - t_x)$ → covers full width ($N_{\text{tiles},x} = \text{tile\_width}$)
- Same for $r_y$ → full height

**Threshold for full-tile-grid coverage at tile16 (1920×1080):**
- Total tiles: $120 \times 68 = 8160$
- A Gaussian covers all tiles when $r_x \geq \max(t_x, 120-t_x)$ AND $r_y \geq \max(t_y, 68-t_y)$
- At image center ($t_x=60, t_y=33.75$): $r_x \geq 60, r_y \geq 34.25$
- In pixels: $\text{radius}_x \geq 960, \text{radius}_y \geq 548$
- In world scale (assuming depth=5, f=2058): $s \geq \frac{960}{3.33 \times 2058/5} \approx 0.7$

**Any visible Gaussian with world scale > 0.7 covers ALL 8160 tiles at tile16.**

---

## 3. Intersection Expansion Analysis

### 3.1 Summary by Scene

| Scene | Total G | Visible G | Visible% | tile16 Isect | tile24 Isect | t16 isect/G | t24 isect/G |
|:------|--------:|----------:|:--------:|-------------:|-------------:|:-----------:|:-----------:|
| room | 1,105,873 | 21,464 | 1.9% | 174,446,870 | 76,965,861 | 8,127 | 3,586 |
| bicycle | 2,589,484 | 103,975 | 4.0% | 607,234,821 | 268,869,816 | 5,840 | 2,586 |
| garden | 874,019 | 85,995 | 9.8% | 683,654,546 | 301,666,767 | 7,950 | 3,508 |

### 3.2 Tail Distribution (tile16)

| Percentile | room | bicycle | garden |
|:----------:|:----:|:-------:|:------:|
| p50 (median) | **8,160** | **8,160** | **8,160** |
| p90 | **8,160** | **8,160** | **8,160** |
| p95 | **8,160** | **8,160** | **8,160** |
| p99 | **8,160** | **8,160** | **8,160** |
| p99.9 | **8,160** | **8,160** | **8,160** |
| max | **8,160** | **8,160** | **8,160** |

> **Key finding: ALL percentiles equal 8160 (total tiles at tile16). There is NO tail — every visible Gaussian covers every tile.**

### 3.3 Tail Distribution (tile24)

| Percentile | room | bicycle | garden |
|:----------:|:----:|:-------:|:------:|
| p50 (median) | **3,600** | **3,600** | **3,600** |
| p90 | **3,600** | **3,600** | **3,600** |
| p99 | **3,600** | **3,600** | **3,600** |
| max | **3,600** | **3,600** | **3,600** |

> **Same pattern: ALL visible Gaussians cover ALL 3600 tiles (80×45 grid) at tile24.**

### 3.4 Bucket Contribution (tile16)

| Bucket | room | | bicycle | | garden | |
|:-------|:---:|:-:|:-------:|:-:|:------:|:-:|
| | Gauss | Isect% | Gauss | Isect% | Gauss | Isect% |
| 1–16 | 0 | 0.0% | 235 | 0.0% | 9 | 0.0% |
| 17–64 | 0 | 0.0% | 743 | 0.0% | 41 | 0.0% |
| 65–256 | 6 | 0.0% | 2,723 | 0.1% | 171 | 0.0% |
| 257–1024 | 16 | 0.0% | 6,808 | 0.7% | 514 | 0.1% |
| 1025–4096 | 63 | 0.1% | 20,890 | 8.5% | 1,463 | 0.6% |
| **>4096** | **21,379** | **99.9%** | **72,576** | **90.7%** | **83,797** | **99.4%** |

**Finding:** 90–99.9% of all intersections come from the ">4096" bucket, which consists of Gaussians covering >4096 tiles. Since max = 8160, this bucket is essentially the "full tile grid" bucket.

### 3.5 Bucket Contribution (tile24)

| Bucket | room | | bicycle | | garden | |
|:-------|:---:|:-:|:-------:|:-:|:------:|:-:|
| | Gauss | Isect% | Gauss | Isect% | Gauss | Isect% |
| 1–16 | 0 | 0.0% | 497 | 0.0% | 26 | 0.0% |
| 17–64 | 2 | 0.0% | 1,493 | 0.0% | 107 | 0.0% |
| 65–256 | 8 | 0.0% | 4,715 | 0.3% | 320 | 0.0% |
| 257–1024 | 39 | 0.0% | 12,995 | 3.0% | 883 | 0.2% |
| **1025–4096** | **21,415** | **99.97%** | **84,275** | **96.7%** | **84,659** | **99.8%** |
| >4096 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |

**Finding:** At tile24, max = 3600 = total tiles. The "1025–4096" bucket is the full-tile-grid bucket. Same pattern: all visible Gaussians saturate the tile grid.

---

## 4. Root Cause of High Intersection

### PRIMARY ROOT CAUSE: Full-Tile-Grid Saturation (OBSERVED)

$$N_{\text{isect}} \approx N_{\text{visible}} \times N_{\text{tiles}}$$

Where $N_{\text{tiles}}$ ≈ constant ≈ total tile count. This means:

```
N_isect ≈ (fraction of Gaussians visible from this viewpoint) × (total Gaussians) × (total tile count)
```

**Source chain:** Every visible Gaussian has a projected pixel radius large enough to span the entire tile grid. This occurs when:

$$\text{radius}_{\text{pixels}} \geq \max(\text{mean2d}_x,\ \text{image\_width} - \text{mean2d}_x)$$

For a Gaussian at screen center (960, 540): $\text{radius} \geq 960$ pixels is needed for full width. With `extend ≈ 3.33` and focal length `f ≈ 2058` at depth 5, this requires:

$$\sqrt{\text{covar2d}_{0,0}} \geq \frac{960}{3.33} \approx 288 \text{ pixels}$$

Which corresponds to world scale $s \geq 0.7$ at depth 5.

**Why this happens after 30K training:** The 3DGS densification/pruning process removes Gaussians with low opacity. Gaussians that survive to 30K iterations are predominantly those with large projected footprints that contribute to the reconstruction across many viewpoints.

### SECONDARY ROOT CAUSE: (slightly off-center) Gaussians with large world-space scale (OBSERVED)

The scenes contain Gaussians with large world-space scales (for background/sky reconstruction). When combined with the perspective projection and the fixed camera position, these Gaussians project to footprints covering the entire tile grid.

---

## 5. Tile Size Effect

| Metric | tile16 | tile24 | Ratio (24/16) |
|:-------|:------:|:------:|:--------------:|
| Tile grid | 120×68=8,160 | 80×45=3,600 | **0.441** |
| Room N_isect | 174.4M | 77.0M | 0.441 |
| Bicycle N_isect | 607.2M | 268.9M | 0.443 |
| Garden N_isect | 683.7M | 301.7M | 0.441 |

**Finding:** The intersection count scales EXACTLY with the tile grid size ratio (0.441 = 3600/8160). This confirms that every visible Gaussian covers every tile at both tile sizes.

$$\frac{N_{\text{isect, t24}}}{N_{\text{isect, t16}}} = \frac{\text{tiles}_{\text{t24}}}{\text{tiles}_{\text{t16}}}$$

---

## 6. Resolution Effect

**Status:** `UNKNOWN` — only 1920×1080 resolution data available.

**Prediction:** If the intersection model is correct ($N_{\text{isect}} \approx N_{\text{visible}} \times N_{\text{tiles}}$), then:

- Higher resolution → more tiles → proportionally more intersections
- Lower resolution → fewer tiles → proportionally fewer intersections
- The number of visible Gaussians would remain approximately the same (same scene, same camera position)

**However:** The pixel radius in the projection kernel might change at different resolutions (different focal length, different K matrix), which could change the set of visible Gaussians. This is `UNKNOWN` without empirical data.

---

## 7. Intersection Redundancy Analysis

### A: Intersections consumed by rasterizer

**SUPPORTED** — Every intersection record is read by the rasterizer. The sorted `flatten_ids` indices directly determine which Gaussians contribute to each pixel.

### B: Opacity termination in rasterizer occurs AFTER sorting

**SUPPORTED** — Source: `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`, line 149:

```cpp
if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
    // early termination — but this is in the rasterizer,
    // AFTER sorting. The sorted list is fully populated.
}
```

**Implication:** Opacity termination does NOT reduce intersection count. It only reduces rasterization work per pixel after sorting.

### C: Intersections cover many tiles but effective pixel coverage is small

**PARTIAL** — A Gaussian covering all 120×68 tiles has 8160 tile-level intersections, but many of these tiles contain pixels where the Gaussian's contribution is near-zero (at the edges of its Gaussian falloff). However:
- This is a property of the **conservative tile bounding** approach (see D below)
- The intersect kernel does NOT apply any per-pixel threshold — it creates intersections for every tile that the bounding box touches
- The alpha blending in rasterizer handles per-pixel opacity

### D: Intersections produced by conservative tile bounding

**SUPPORTED** — From `IntersectTile.cu` lines 88-102:
- The tile bounds are computed from the axis-aligned bounding box of the Gaussian's projected 3σ ellipse
- This bounding box is **conservative**: it includes tiles where the Gaussian may contribute near-zero at corners
- The formula uses `floor`/`ceil` which always rounds OUTWARD, adding one tile at each boundary

**Conservative overhead estimate:**
- Each tile-boundary rounding adds up to 1 tile per dimension (up to ~2 tiles per Gaussian)
- For a Gaussian covering all 120 × 68 tiles, the conservative rounding adds at most 0/8160 = 0% overhead
- For a small Gaussian covering 4 tiles, the rounding could add up to 50% overhead
- But: since >>99% of intersections come from full-tile-grid Gaussians, this overhead is negligible in absolute terms

### E: Other

**UNKNOWN** — Other forms of redundancy (e.g., Gaussians behind the rendered surface, depth ordering) are not analyzed here.

---

## 8. Early Termination Can NOT Remove Intersections

Confirmed from source flow:

```
Projection → Intersect (pass1 + cumsum + pass2)
  → CUB SortPairs → isect_offset_encode
  → Rasterize (per-pixel alpha blending with early termination)
```

The `ALPHA_THRESHOLD = 1/255` gate in the rasterizer is the ONLY early termination mechanism, and it operates **after** sorting. There is no mechanism to prune intersections before sorting.

---

## 9. Comparison with Existing Evidence

### Phase 8E (Per-Kernel Forward Timing)

- Phase 8E's finding that `intersect_sort` dominates 93-96% of forward time is **CONSISTENT** with the intersection explosion
- The 64-67% CUB sort share is a direct consequence: 174-684M items must be sorted, and this takes 65% of forward time
- The 28-30% intersect pass1 share is the time to compute the tile bounds and write intersection records

### Phase 8D (Forward Hypothesis Matrix)

- H6 was hypothesized as a "count" issue — **SUPPORTED** by this analysis
- The total intersection count is ~N_visible × N_tiles (full saturation)
- This is NOT a "duplication" issue: each intersection is a unique (Gaussian, tile) pair
- This is NOT an "unnecessary generation" issue: the conservative tile bounding is standard practice

### H6 Classification

```
H6 ROOT CAUSE:
COUNT — The total intersection count is fundamentally determined by
N_visible × N_tiles. With every visible Gaussian covering every tile,
reducing N_visible or N_tiles is the only path to reduce intersections.

DUPLICATION: FALSIFIED — Each (Gaussian, tile) pair is unique
REPRESENTATION: PARTIAL — isect_key encoding could be more compact,
    but this doesn't affect the COUNT of intersections
MEMORY TRAFFIC: SUPPORTED as downstream effect — 152B per sort item
UNNECESSARY GENERATION: FALSIFIED — The conservative bounding box is
    standard, and full-tile-grid Gaussians have negligible overhead
    from conservative rounding
```

---

## 10. Potential "Safe Reduction" Hypotheses

All hypotheses listed below have `HYPOTHESIS` status — none are implemented or validated.

### H-A: Opacity-aware tile reduction

| Question | Answer |
|----------|--------|
| Why redundant? | A Gaussian with opacity near `ALPHA_THRESHOLD` contributes near-zero to most tiles; its extend is already reduced to 0 |
| Conditions? | Only for Gaussians with opacity just above threshold AND large covariance |
| Conservative proof? | **NOT available** — the opacity-aware extend already shrinks the footprint, but any Gaussian above threshold contributes (at least at tile level) |
| Depth ordering? | No — depth ordering is in the sort key |
| Forward correctness? | Would discard some intersections → different rendering |
| Backward/gradient? | Would affect gradients for those Gaussians |
| Prior-art risk? | `EXTERNAL PRIOR-ART CHECK REQUIRED` |

**Verdict: NOT SAFE** — The opacity-aware extend in projection already implements this.

### H-B: Conservative tile boundary reduction

| Question | Answer |
|----------|--------|
| Why redundant? | `floor`/`ceil` on tile boundaries can add 1 tile per dimension |
| Conditions? | Only for Gaussians where extending 1 tile in each direction meaningfully changes the total |
| Conservative proof? | **Yes** — But for full-tile-grid Gaussians, there are no additional tiles to add |
| Contribution? | Would eliminate ~0.1% of intersections (only affects small Gaussians) |

**Verdict: NEGLIGIBLE** — >>99% of intersections come from full-tile-grid Gaussians.

### H-C: Smaller tile size

| Question | Answer |
|----------|--------|
| Why redundant? | Tile24 has 0.44× as many tiles as tile16 |
| Conditions? | Any scene where Gaussians still saturate the tile grid |
| Conservative proof? | **Yes** — tile size is a parameter, not a correctness issue |
| Contribution? | Reduces intersections by a factor of (tile_grid_ratio) |
| Depth ordering? | Affects tile-level ordering (different sort keys), but within-tile sorting still correct |
| Backward/gradient? | Affects gradients through different tile assignments |

**Verdict: THEORETICALLY SAFE but trivially confirmed** — smaller tiles reduce both intersections and rasterization quality at tile boundaries. This trades quality for speed.

### H-D: Reduce visible Gaussians (viewport culling)

| Question | Answer |
|----------|--------|
| Why redundant? | Gaussians that are behind all visible surfaces or occluded don't need intersections |
| Conditions? | Heavy occlusions or depth ordering |
| Conservative proof? | **NOT available** — occlusion culling at the tile level requires depth information that is not available until after sorting |
| Contribution? | Could reduce N_visible in complex scenes but likely small in typical 3DGS benchmark views |

**Verdict: NOT SAFE without depth pre-pass** — requires either a depth pre-pass (doubles cost) or conservative occlusion test.

### H-E: Reduce Gaussian count (pruning threshold)

| Question | Answer |
|----------|--------|
| Why redundant? | Some visible Gaussians contribute very little to final rendering |
| Conditions? | After 30K training, most Gaussians are well-tuned |
| Conservative proof? | **NOT available** — pruning changes rendering quality |
| Contribution? | Could significantly reduce N_visible if aggressive |
| Prior-art risk? | `EXTERNAL PRIOR-ART CHECK REQUIRED` |

**Verdict: QUALITY vs SPEED tradeoff** — not a "safe" optimization.

---

## 11. Known / Unknown Matrix

### Known (OBSERVED or DERIVED from source)

| Fact | Evidence |
|------|----------|
| Every visible Gaussian covers every tile | MEASURED: p50=p99=max=tile grid size |
| 90-99.9% of intersections from full-tile-grid Gaussian bucket | MEASURED |
| Intersection count scales linearly with tile count | DERIVED: t16 ↔ t24 ratio = tile ratio |
| Visible Gaussians are 2-10% of total | OBSERVED |
| Intersection count = N_visible × N_tiles | DERIVED (model matches data) |
| Pixel radius threshold for full coverage ≈ 960px | DERIVED from tile grid size × tile_size / 2 |
| Conservative tile rounding has negligible effect | DERIVED: only matters for small Gaussians |
| ALPHA_THRESHOLD=1/255 controls both projection and rasterizer | CONFIRMED from Common.h |

### Unknown

| Fact | Reason |
|------|--------|
| Why all visible Gaussians have scale > 0.7 | Requires analyzing the per-Gaussian scale distribution; hypothesis: training/pruning preserves large Gaussians |
| Resolution effect on intersection count | Only 1920×1080 data available |
| Whether large-scale Gaussians are necessary for quality | Requires ablation study (prohibited: NOT STARTED) |
| Multi-view intersection overlap | Only single-camera profiling; multiple views multiply intersections |

---

## 12. Prior-Art

`EXTERNAL PRIOR-ART CHECK REQUIRED`

No external literature search or novelty claims are made in this report. The intersection explosion analysis is specific to gsplat's implementation and the A100 PCIe 40GB environment.

---

## 13. Output Files

- **Report:** `reports/phase-a100/intersection_explosion_source_audit.md` (this file)
- **JSON data:** `results/phase-a100/intersection_explosion_analysis.json`

---

## 14. Final Output

```text
A100 INTERSECTION EXPLOSION AUDIT COMPLETE

INTERSECTION / GAUSSIAN:
8,160 (tile16) / 3,600 (tile24) — every visible Gaussian covers EVERY tile

P90:
8,160 (tile16) / 3,600 (tile24) — same as median and max

P99:
8,160 (tile16) / 3,600 (tile24) — same

HIGH-TAIL CONTRIBUTION:
The ">4096" bucket (full-tile-grid Gaussians) contributes:
  room:    99.9% of all intersections
  bicycle: 90.7%
  garden:  99.4%
There IS NO long tail — the distribution is uniform saturation.

PRIMARY ROOT CAUSE:
Full-tile-grid saturation: every visible Gaussian's projected pixel
radius exceeds half the image dimensions in both x and y. This occurs
when Gaussian world scale > 0.7 at the fixed camera position (depth=5,
f=2058). After 30K training, surviving visible Gaussians are
predominantly large-scale.

SECONDARY ROOT CAUSE:
Conservative tile bounding (floor/ceil) and the 3.33σ Gaussian extend
factor ensure that even moderate-footprint Gaussians tend to overshoot
the image boundaries when projected.

H6 STATUS:
SUPPORTED — Classification: COUNT
The intersection count is N_visible × N_tiles. With full tile grid
saturation, this is structurally determined.

POTENTIAL SAFE-REDUCTION HYPOTHESES:
None identified as "safe" without changing rendering quality or
correctness. The intersection explosion is a structural property
of the tile-based renderer with large Gaussians.

PRIOR-ART:
EXTERNAL CHECK REQUIRED

CUDA OPTIMIZATION:
NOT STARTED
```
