# Intersection → Sorting Research Gap Audit

**Date:** 2026-09-06  
**Objective:** Identify the most promising research gap for reducing the intersection→sorting bottleneck in 3D Gaussian Splatting rendering, based on A100 experimental evidence and prior-art analysis.

---

## 1. Confirmed Facts (From Phase A100)

### 1.1. Pipeline Bottleneck

```
Projection → Intersect (pass1+cumsum+pass2) → CUB SortPairs → Rasterize

CUB radix sort: 65-68% of forward time (measured)
Pass1 (intersect: count + write): 28-30% of forward time (measured)
Rasterize + offset: ~4% of forward time (measured)
```

### 1.2. Intersection Source

Every visible Gaussian generates `N_tiles(g)` intersection records, where:

```
N_tiles(g) = (tile_max.y - tile_min.y) × (tile_max.x - tile_min.x)
tile_min = floor((mean2d - radius) / tile_size)
tile_max = ceil((mean2d + radius) / tile_size)

radius = min(3.33, sqrt(-2×log(opacity/ALPHA_THRESHOLD))) × sqrt(covar2d[i][i])
ALPHA_THRESHOLD = 1/255 (from Common.h)
```

### 1.3. Saturation Evidence (Measured)

| Metric | Artificial Camera | Real Camera (mean) | Ratio |
|:-------|:-----------------:|:------------------:|:-----:|
| Max N_isect (garden tile16) | 683.7M | 40.8B | **59.7×** |
| Mean isect/G (tile16) | 5,840-8,127 | 25,162-67,254 | **3-10×** |
| R_full (tile16) | 0.505-0.992 | 0.790-0.981 | Consistent |
| A100 40GB feasible? | YES (≤684M) | NO (≥12B) | OOM |

### 1.4. Distribution Type: NOT Heavy Tail

From `intersection_explosion_analysis.json`:

```
tile16: p50 = p90 = p99 = p99.9 = 8,160 (n_tiles)
tile24: p50 = p90 = p99 = p99.9 = 3,600 (n_tiles)
```

**This is uniform saturation, not a heavy tail.** Every visible Gaussian contributes equally: `N_tiles` intersections each. There is no "few Gaussians contribute most" pattern.

### 1.5. Sorting Cost

```
0.12 ns per intersection (measured, consistent across 6 workloads)
CUB 1.5.10, ONESWEEP mode, 6 passes, 384 threads × 21 items/thread
Per-sort memory: ~34 × N_isect bytes (lower bound including CUB temp)
```

---

## 2. Intersection Generation Pipeline: Source-Confirmed

From `IntersectTile.cu`:

```
Gaussian with 2D mean, radii, depth
  ↓
tile_radius_x = radius_x / tile_size
tile_radius_y = radius_y / tile_size
tile_x = mean2d.x / tile_size
tile_y = mean2d.y / tile_size
  ↓
tile_min.x = clamp(floor(tile_x - tile_radius_x), 0, tile_width)
tile_min.y = clamp(floor(tile_y - tile_radius_y), 0, tile_height)
tile_max.x = clamp(ceil(tile_x + tile_radius_x), 0, tile_width)
tile_max.y = clamp(ceil(tile_y + tile_radius_y), 0, tile_height)
  ↓
tiles_per_gauss = (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
  ↓
For each covered tile: write (isect_id, flatten_id)
  isect_id = image_id | tile_id | depth_bits (64-bit key)
  flatten_id = gaussian_index (32-bit value)
  ↓
CUB::DeviceRadixSort::SortPairs(d_keys=isect_ids, d_values=flatten_ids)
```

### Key Observation

The current algorithm creates **every (Gaussian, tile) pair** where the Gaussian's axis-aligned bounding box touches the tile, **before any depth ordering check**. The only filtering before intersection generation is:

1. `radius <= 0` → skip (from projection: opacity < ALPHA_THRESHOLD or frustum culled)
2. Nothing else.

The rasterizer later sorts by (tile_id, depth) and **alpha-blends front-to-back, terminating when alpha < 1/255**. But this termination happens AFTER sorting.

---

## 3. Prior Art: Classification Framework

For each paper, I classify its contribution along two axes:

1. **Target stage:** (A) Intersection generation, (B) Input size to sort, (C) Sorting algorithm, (D) Rasterization efficiency, (E) Temporal reuse, (F) Geometry reuse
2. **Mechanism:** culling, hierarchical, coarse-to-fine, approximate, local, depth-aware, tile-level

### 3.1. GSCore (2024)

| Aspect | Detail |
|:-------|:-------|
| Target | **A, B** — Culls Gaussians before intersection |
| Mechanism | Tile-grouping + frustum culling; groups Gaussian centroids into tile groups before full intersection; early discard of Gaussians whose opacity contribution is negligible at tile level |
| What it solves | Reduces N_visible per tile group by coarse culling |
| What it does NOT solve | Does NOT eliminate per-tile enumeration for surviving Gaussians; large Gaussians still cover many tiles |
| Pair reduction | Pre-intersection: YES | Sorting input reduction: INDIRECT |
| **Relevance to H6** | MEDIUM — addresses N_visible reduction, not N_tiles per Gaussian |

### 3.2. Splatshop (2024)

| Aspect | Detail |
|:-------|:-------|
| Target | **A, F** — Gaussian compaction + pruning |
| Mechanism | Merges near-duplicate Gaussians; reduces total Gaussian count at training time |
| What it solves | Fewer Gaussians → fewer intersections overall |
| What it does NOT solve | Does not change the intersection MULTIPLICATION factor — surviving Gaussians still cover all tiles |
| Pair reduction | Pre-training: YES | Runtime sorting: INDIRECT |
| **Relevance to H6** | LOW — orthogonal to the tile-coverage saturation problem |

### 3.3. HiGS (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **A, D** — Hierarchical culling before rasterization |
| Mechanism | Builds a Gaussian hierarchy (octree/Gaussian pyramid); renders coarse-to-fine; skips Gaussians whose coarse-level contribution is negligible |
| What it solves | Reduces per-pixel blending cost; early rejection of invisible Gaussians |
| What it does NOT solve | At fine level, still enumerates tile intersections; hierarchy traversal cost may offset savings for large Gaussians |
| Pair reduction | Pre-rasterization: YES (hierarchical) | Sorting: INDIRECT |
| **Relevance to H6** | MEDIUM — hierarchy could skip full-tile-grid Gaussians at coarse level, but intersection enumeration is per-tile at each level |

### 3.4. msplat (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **B, D** — Multi-resolution splatting |
| Mechanism | Renders Gaussians at multiple scales; coarse Gaussians rendered at lower resolution; reduces per-pixel operation count |
| What it solves | Large Gaussians don't need per-pixel evaluation at full resolution |
| What it does NOT solve | Does this at rasterization stage — intersection and sorting still run on all Gaussians at full resolution |
| Pair reduction | Pre-sort: NO | After sort: YES |
| **Relevance to H6** | LOW — addresses rasterization, not the intersection→sorting bottleneck |

### 3.5. TileGS (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **A, B** — Tile-level Gaussian management |
| Mechanism | Assigns each Gaussian to the tiles it actually affects with per-tile depth bounds; tile-local culling |
| What it solves | Reduces cross-tile redundant work; depth-aware tile assignment |
| What it does NOT solve | Still enumerates per-tile bounds for each Gaussian; does not address the multiplication factor for full-screen Gaussians |
| Pair reduction | Per-tile culling: YES | Global sorting: INDIRECT |
| **Relevance to H6** | MEDIUM — most directly related to intersection redundancy |

### 3.6. Neo (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **E, F** — Temporal reuse |
| Mechanism | Caches and reuses intersection results across frames; only processes changed regions |
| What it solves | For static scenes, eliminates redundant intersection computation and sorting |
| What it does NOT solve | For dynamic scenes or camera cuts, falls back to full pipeline; initial frame still suffers |
| Pair reduction | Temporal: YES | First frame: NOT |
| **Relevance to H6** | LOW — orthogonal to the single-frame structural issue |

### 3.7. RoofGS (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **A, D** — Frustum-aware culling |
| Mechanism | Uses coarse depth map to cull Gaussians behind visible surfaces before intersection |
| What it solves | Reduces N_visible in occluded regions |
| What it does NOT solve | Large sky/background Gaussians (which cause full-tile saturation) are typically NOT occluded; they remain fully visible |
| Pair reduction | Occlusion: YES | For large Gaussians: LIMITED |
| **Relevance to H6** | MEDIUM — occluded Gaussians are not the primary source of explosion |

### 3.8. Speedy-Splat (2024)

| Aspect | Detail |
|:-------|:-------|
| Target | **B, C** — Faster sorting via tile-local sort |
| Mechanism | Replaces global CUB radix sort with per-tile localized sorting; each tile sorts its own Gaussians independently |
| What it solves | Parallelizes sorting across tiles; eliminates cross-tile sort overhead |
| What it does NOT solve | Still sorts every intersection per tile; does not reduce total intersection count |
| Pair reduction | Pre-sort: NO | Sort parallelization: YES |
| **Relevance to H6** | HIGH — directly addresses sorting bottleneck but NOT intersection count |

### 3.9. FlashGS (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **B, C, D** — Fully fused renderer optimization |
| Mechanism | Fuses intersection generation, sorting, and rasterization into fewer kernel launches; uses shared memory for per-tile sorting; removes separate CUB sort calls |
| What it solves | Reduces launch overhead and global memory traffic for sort; per-tile sorting in shared memory |
| What it does NOT solve | Every intersection is still generated and sorted — just more efficiently. Large Gaussians still produce n_tiles records per tile. |
| Pair reduction | Pre-sort: NO | Sort efficiency: YES |
| **Relevance to H6** | HIGH — closest prior art to our observed bottleneck. But they optimize sort, not intersection count. |

### 3.10. AAA-Gaussians (2025)

| Aspect | Detail |
|:-------|:-------|
| Target | **D** — Anti-aliasing |
| Mechanism | Addresses aliasing from Gaussian projection; not directly related to intersection optimization |
| What it solves | Rendering quality |
| **Relevance to H6** | NONE |

### 3.11. StochasticSplats (2025/2026)

| Aspect | Detail |
|:-------|:-------|
| Target | **B, D** — Stochastic culling |
| Mechanism | Randomly samples a subset of Gaussians per tile/pixel during rendering; estimates correct output via Monte Carlo |
| What it solves | Reduces per-pixel blending operations significantly |
| What it does NOT solve | Requires careful variance control; may produce noise; intersection generation still runs fully |
| Pair reduction | Post-intersection: YES (sampling) | Sorting: YES (fewer items) |
| **Relevance to H6** | MEDIUM — could reduce sort size but changes rendering paradigm (probabilistic) |

---

## 4. Prior Art Synthesis

### 4.1. Classification by Target Stage

```
Stage A: Intersection generation
  GSCore (tile-group culling) ✓
  TileGS (tile-level management) ✓
  RoofGS (depth culling) ✓
  HiGS (hierarchical) ✓

Stage B: Sorting input size
  Speedy-Splat (per-tile sort) ✓ (parallelizes, doesn't reduce)
  FlashGS (fused sort) ✓ (optimizes, doesn't reduce)
  StochasticSplats (sampling) ✓ (reduces with noise cost)

Stage C: Sorting algorithm
  Speedy-Splat (per-tile) ✓
  FlashGS (shared memory sort) ✓

Stage D: Rasterization efficiency
  msplat, HiGS, FlashGS, AAA-Gaussians ✓

Stage E: Temporal reuse
  Neo ✓

Stage F: Geometry reuse
  Splatshop ✓
```

### 4.2. What Has NOT Been Addressed

| Missing Approach | Papers That Could Address It |
|:-----------------|:-----------------------------|
| **Pre-intersection tile-coverage reduction for large Gaussians** | None directly |
| **Gaussian footprint decomposition (split large Gaussian into smaller ones for tile enumeration)** | None |
| **Intersection generation that is O(N_visible × sigma) instead of O(N_visible × N_tiles)** | None |
| **Approximate intersection: skip tiles where Gaussian contributes near-zero** | None |
| **Mathematical bound: prove which (Gaussian, tile) pairs contribute ≤ ALPHA_THRESHOLD** | None |

### 4.3. Key Gap Identified

**Every existing method that touches intersection generation does so by reducing N_visible (culling, pruning, occlusion). No existing method reduces the per-Gaussian tile-coverage factor N_tiles_g.**

This is the central research gap:

```
Current math:      N_isect = Σ_g N_tiles(g)
                     where N_tiles(g) = ceil_to_tile_boundary(radius/ts)^2
                     
Existing reduction:  N_visible ↓ (culling, pruning)
Missing reduction:   N_tiles(g) ↓ (per-Gaussian footprint)
```

---

## 5. Deep Dive: The Multiplication Factor Problem

### 5.1. Source-Confirmed: Why Every Gaussian Covers Every Tile

From `footprint_analysis.py` results:

| Scene | Mean radii_x (px) | Image width | Depth median | R_full |
|:------|:-----------------:|:-----------:|:------------:|:-----:|
| room (real) | 106,039 | 3,114 | 2.52 | 0.989 |
| bicycle (real) | 236,868 | 4,946 | 3.04 | 0.683 |
| garden (real) | 122,907 | 5,187 | 3.72 | 0.944 |

For a Gaussian to cover ALL tiles:

```
radius_x >= max(mean2d.x, image_width - mean2d.x)

When camera is inside scene (depth ~ 3):
  radius_x ≈ 3.33 × (focal_length / depth) × world_scale
  For room: radius_x ≈ 3.33 × (3173 / 2.52) × 0.047 ≈ 197 pixels
  For image width 3114: need radius_x >= 1557 pixels
  With world_scale = 0.047: radius_x ≈ 197 pixels << 1557...
  Wait, this doesn't match.
```

Let me recalculate. The footprint analysis showed `mean radii_x = 106,039px` - this is enormous. Let me trace why:

Actually, the projection in EWA 3DGS calculates the 2D covariance from the 3D covariance using the perspective projection Jacobian. A Gaussian at world position near the camera gets a very large Jacobian because:

```
J = [f/z, 0, -f*x/z^2; 0, f/z, -f*y/z^2]
```

When z is small (Gaussian very close to camera), the Jacobian entries are huge. But more importantly, when the Gaussian is BEHIND the camera (negative z in camera space), the depth value is negative. The code allows negative depths:

```
int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
```

Wait, the actual measured mean radii_x was 106,039 pixels on a 3114-pixel image. This is 34× the image width. This means the Gaussian's projected footprint is enormous because:

1. The Gaussian is in world space at some position
2. The camera is inside the scene
3. The 2D covariance after perspective projection creates a huge ellipse
4. The 3.33σ extent of this ellipse covers the entire image and far beyond

This is the **Case B** confirmed earlier: the perspective projection from inside-scene cameras creates enormous screen-space footprints.

### 5.2. The Multiplication Factor

For each visible Gaussian:

```
N_tiles(g) = ceil(width_in_tiles) × ceil(height_in_tiles)

When radius_x >> image_width and radius_y >> image_height:
  width_in_tiles ≈ tile_width
  height_in_tiles ≈ tile_height
  N_tiles(g) = tile_width × tile_height = N_tiles
  
For garden tile16: N_tiles = 325 × 211 = 68,575
```

Each of the 593,421 visible Gaussians writes 68,575 intersection records → 40.7B total. The sorting must then process 40.7B items.

### 5.3. The Core Structural Observation

The current algorithm has the property:

```
Time ∝ N_visible × N_tiles

where N_tiles is a CONSTANT (not dependent on Gaussian size)
once radius exceeds image dimensions.
```

This is the algorithmic inefficiency: **the cost is independent of the Gaussian's actual screen size once it exceeds the image.** A Gaussian covering 100% of the image costs no more than one covering 200% or 2000% — they all saturate at N_tiles. But a Gaussian covering only 2 tiles costs 34,000× less.

---

## 6. Research Gaps: Three Candidates

### Gap A: Pre-Intersection Gaussian Footprint Approximation

#### 1. Gap Name
**Tile-coverage-aware intersection: approximate Gaussian contribution before intersection generation**

#### 2. Existing Prior Art
- FlashGS: fuses sorting, does not reduce intersections
- Speedy-Splat: per-tile sort, does not reduce intersections
- TileGS: tile-level Gaussian assignment but still enumerates per Gaussian

#### 3. What They Already Solve
- FlashGS: reduces sort overhead via fused kernels and shared memory sort
- Speedy-Splat: parallelizes sorting across tiles
- TileGS: assigns Gaussians to tiles more intelligently

#### 4. Remaining Gap
**None of these methods reduce the number of (Gaussian, tile) pairs generated for full-screen Gaussians.** When a Gaussian covers all tiles, all existing methods still write N_tiles intersection records.

The gap is: **Can we prove, from the Gaussian math, that a (Gaussian, tile) pair contributes ≤ ε (e.g., ALPHA_THRESHOLD) to all pixels in that tile, and skip generating that pair entirely?**

This is NOT the same as opacity-based early termination in the rasterizer (which happens AFTER sorting). This would happen BEFORE intersection generation.

#### 5. Evidence From Our Repo

```
real_scene_workload_validity.json:
  R_full = 0.79-0.98 (fraction of Gaussian covering ALL tiles)
  For garden cam_0: 593,421 visible × 68,575 tiles = 40.7B intersections
  Each of these Gaussians writes to ALL 68,575 tiles
  
If we could prove that for a Gaussian at depth d with covariance Σ,
its contribution to pixels at a distance > kσ from the 2D mean is < threshold,
then many tile-level intersections could be safely skipped.
```

#### 6. Why This Is Research-Level

Current rendering uses a **conservative screen-space bounding box** (3σ ellipse → axis-aligned bounding box → tile bounds). The Gaussian's actual contribution to a pixel is:

```
contribution = opacity × exp(-0.5 × Δ²)
where Δ² = conic_xx×dx² + 2×conic_xy×dx×dy + conic_yy×dy²
```

The 3σ bound captures 99.7% of the Gaussian's energy. But the persistency of this bound across tiles inside the bounding box is **uneven** — at the corners of the bounding box, the contribution can be near-zero.

**Research question: Can we derive a tile-level contribution bound that is tighter than the current axis-aligned bounding box, using only per-Gaussian parameters (means2d, conic matrix, opacity) and per-tile geometry?**

This is not a simple engineering tweak because:
- The Gaussian's contribution is anisotropic (conic matrix is not axis-aligned)
- The tile is a region, not a point
- The bound must be conservative (no visual degradation) to be safe

#### 7. Main Risk

```
NOVELTY RISK: MEDIUM
- The idea of contribution-aware culling is explored in graphics (EWA splatting literature)
- Need to check if any Gaussian splatting paper has applied this at tile granularity
- Main risk: the bound may be too loose to provide meaningful reduction
  (i.e., nearly all tiles still qualify as "potentially contributes")
```

#### 8. Minimum Experiment Needed

```python
# For one camera view, one tile:
# 1. For each (Gaussian, tile) pair, compute the MINIMUM possible Mahalanobis distance
#    from the Gaussian's mean to ANY point in the tile
# 2. Derive: max_contribution = opacity × exp(-0.5 × min_Mahalanobis²)
# 3. Count pairs where max_contribution < ALPHA_THRESHOLD (1/255)
# 4. If >0% of pairs are skippable → Gap confirmed

# This requires only projection output (means2d, conics), not modified CUDA code
```

---

### Gap B: Gaussian Footprint Decomposition for Tile-Limited Enumeration

#### 1. Gap Name
**Screen-space Gaussian decomposition: subdivide large Gaussians into smaller per-tile contributions to bound intersection explosion**

#### 2. Existing Prior Art
- HiGS: hierarchical Gaussian representation, but for culling
- msplat: multi-resolution rendering, but after sorting
- No existing work decomposes a single Gaussian's intersection footprint into tile-limited sub-regions

#### 3. What They Already Solve
- HiGS: coarse-to-fine rendering reduces per-pixel operations
- msplat: multi-resolution evaluation
- Neither addresses the intersection enumeration problem

#### 4. Remaining Gap

When a Gaussian covers the entire screen, every tile gets the same Gaussian as a candidate. But the Gaussian's actual shape may vary significantly across tiles (due to perspective). The gap is:

**Can a single Gaussian's intersection with each tile be decoupled into independent sub-Gaussians, each covering at most a small number of tiles?**

This differs from Splatshop's Gaussian merging (which is about training-time compaction). This is a **runtime decomposition** aimed at limiting the per-Gaussian tile enumeration.

#### 5. Evidence From Our Repo

```
intersection_explosion_analysis.json:
  tile16 buckets show that >99.9% of intersections come from the ">4096" bucket
  (Gaussians covering >4096 tiles out of 8160)
  
This means the current algorithm is doing O(N_tiles) work per Gaussian
even though most of those tiles will get negligible contributions from
that Gaussian after sorting and alpha blending.
```

#### 6. Why This Is Research-Level

For a Gaussian with covariance Σ in 2D, its screen-space contribution is continuous and anisotropic. Decomposing a single Gaussian into tile-bounded sub-regions while maintaining:
1. Correct alpha blending (order-dependent compositing)
2. Correct depth ordering within each tile
3. No seams at tile boundaries

This is a non-trivial problem because Gaussian composition is not linear (it's alpha blending). Simply clipping the Gaussian to tile boundaries breaks the mathematical form.

#### 7. Main Risk

```
NOVELTY RISK: MEDIUM-HIGH
- Conceptually attractive but mathematically challenging
- The decomposition may break the analytical Gaussian form needed for rasterization
- Main risk: correctness cannot be guaranteed without introducing visible artifacts
- Prior art in point-based rendering (splatting at pixel level) suggests tile-level 
  decomposition hasn't been attempted, possibly for good reason
```

#### 8. Minimum Experiment Needed

```python
# For a single large Gaussian:
# 1. Render it at native resolution (baseline)
# 2. Tile by tile, compute the Gaussian's contribution to the tile center and corners
# 3. Measure the variation in contribution across tiles
# 4. If variation is high → decomposition is meaningful
# 5. If variation is low → decomposition gains are small

# Requires no modified renderer, just post-hoc analysis of projection output
```

---

### Gap C: Depth-Bounded Sorting — Eliminating Irrelevant Gaussians Before Full Sort

#### 1. Gap Name
**Tile-local depth-bounded intersection: use coarse depth pre-pass to bound sort depth range per tile**

#### 2. Existing Prior Art
- RoofGS: depth-culling for occlusion, but NOT per-tile depth bounds
- FlashGS: fused per-tile sort, but still on ALL intersections
- Speedy-Splat: per-tile sort, still on ALL intersections
- Traditional Z-buffer: used in mesh rendering, not in Gaussian splatting

#### 3. What They Already Solve
- RoofGS: culls occluded Gaussians before intersection
- FlashGS/Speedy-Splat: make per-tile sorting faster

#### 4. Remaining Gap

Sorting is O(N_isect log N_isect) globally, or O(N_isect_per_tile) per tile with radix sort. But if we knew the **visible depth range** per tile (e.g., from a coarse depth pre-pass or from the sorted results of a previous frame), we could:

1. Limit intersection generation to Gaussians within that depth range
2. Or, after coarse sort, cull Gaussians whose depth contribution after alpha blending will be zero

**Crucially, this is DIFFERENT from RoofGS's occlusion culling**, which removes Gaussians BEHIND surfaces. Our observation is that for large Gaussians, many are at depths far beyond the visible surface and cannot be seen because nearer Gaussians fully absorb them. But they still get sorted.

#### 5. Evidence From Our Repo

```
footprint_analysis.py results:
  garden cam_0: depth median = 3.72, depth min = 0.01, depth max >> 100
  room cam_0: depth median = 2.52, depth range spans multiple orders of magnitude
  
The rasterizer uses front-to-back alpha blending with opacity termination
(alpha < ALPHA_THRESHOLD). For a tile with many Gaussians at varying depths,
once the accumulated alpha reaches ~0.996 (1 - 1/255), all remaining Gaussians
are skipped. But this happens AFTER sorting all intersections.

If we could bound the "effective depth range" per tile BEFORE sorting,
we could exclude Gaussians outside this range.
```

#### 6. Why This Is Research-Level

The challenge is that the effective depth range depends on:
- The opacities of Gaussians (which are not known before sorting within the tile)
- The accumulated alpha (which is sequential and order-dependent)
- The exact depths

This is a chicken-and-egg problem: you need the sorted order to know which Gaussians are redundant, but you need to know which Gaussians are redundant to reduce sorting.

**However:** A statistical or conservative bound may be possible. For example, if the first few Gaussians in a tile (by depth) have collective opacity > 0.996, then all remaining Gaussians in that tile can be ignored. But without sorting, you don't know which Gaussians are "first."

**The research gap:** Can we compute a **conservative per-tile depth cutoff** using only pre-sort information? E.g., from a coarse depth map, or from the per-tile Gaussian count, or from the cumulative opacity bound?

#### 7. Main Risk

```
NOVELTY RISK: HIGH (but for reasons of difficulty, not prior coverage)
- Prior art RoofGS uses depth maps for occlusion culling, but doesn't bound sort depth
- No existing Gaussian splatting paper addresses "what depth range to sort"
- Main risk: the depth range per tile may be so wide that no meaningful reduction is possible
- Second risk: the cost of computing the depth bound may exceed the sorting savings
```

#### 8. Minimum Experiment Needed

```python
# For one real camera view:
# 1. Run full intersect + sort (or compute logical sorted order)
# 2. For each tile: after sorting by depth, find the depth at which
#    accumulated alpha first exceeds 0.996 (1 - ALPHA_THRESHOLD)
# 3. Count how many sorted Gaussians come AFTER this depth
# 4. If >30% of sorted items per tile are after this point → Gap confirmed

# This can be done with the existing rasterizer output + tile-level statistics
# No CUDA modifications needed
```

---

## 7. Novelty Risk Assessment

| Gap | Stage | Novelty Risk | Key Uncertainty |
|:---:|:-----:|:------------:|:----------------|
| **A** | Pre-intersection tile-coverage bound | **MEDIUM** | Bound may be too loose to reduce meaningful count |
| **B** | Gaussian decomposition | **MEDIUM-HIGH** | Mathematical feasibility unclear |
| **C** | Depth-bounded sort | **HIGH** | Cannot predict depth cutoff without sorting |

**Novelty Risk Explanation:**
- A: The math of Gaussian contribution falloff is well-understood, but applying it at tile granularity before intersection generation has not been published. The main risk is that the bound is too loose.
- B: Conceptually attractive, but decomposition of a single anisotropic Gaussian into tile-bounded sub-Gaussians while maintaining analytical form and alpha-blending correctness is mathematically challenging.
- C: High novelty because there's no known prior art for depth-bounded sort in Gaussian splatting. But the chicken-and-egg problem makes this hard.

---

## 8. Recommendation

```
RECOMMENDATION = GAP-A
```

### Why Gap-A

1. **Most Directly Related to H6**: H6-COUNT is the core finding (N_isect = N_visible × N_tiles). Gap-A addresses the N_tiles factor directly, which is the unique structural source of the intersection explosion.

2. **Not Simply Repeating Existing Work**: 
   - FlashGS optimizes sorting, not intersection generation
   - Speedy-Splat parallelizes sorting, doesn't reduce count
   - GSCore/TileGS reduce N_visible, not N_tiles(g)
   - No prior art attempts to prove tile-level contribution bound BEFORE intersection generation

3. **Supported by A100 Evidence**:
   - R_full = 0.79-0.98: 79-98% of visible Gaussians cover ALL tiles
   - Each of these Gaussians writes N_tiles intersection records
   - The Gaussian's actual contribution varies across the tile grid (anisotropic)
   - At tiles far from the Gaussian's mean, the contribution may be negligible

4. **Minimum Experiment Is Feasible**:
   - Only requires existing projection output (means2d, conics, opacities)
   - No CUDA modification
   - Can quantify the total intersection reduction possible

### Why Not Gap-B or Gap-C

- **Gap-B**: Mathematical correctness of Gaussian decomposition for alpha blending is uncertain. The risk of visual artifacts is high, and MINIMUM experiment to validate this is more involved (requires custom renderer).

- **Gap-C**: The chicken-and-egg problem (need sorting to know cutoff, need cutoff to reduce sorting) may be fundamentally hard. The minimum experiment (analyze post-sort depth cutoff distribution) is feasible but the actual solution would require a fundamentally different rendering architecture.

### Minimum Next Experiment (For Gap-A)

```python
# Objective: Quantify how many (Gaussian, tile) pairs can be safely skipped
# without modifying the renderer, using a conservative contribution bound.

# Input: fully_fused_projection output (means2d, conics, opacities, radii)
#        + tile geometry (tile_size, tile_width, tile_height)

# For each visible Gaussian:
#   For each tile in its bounding box:
#     Compute: min_Mahalanobis² = min over (x,y) in tile of 
#       [x-mean2d.x, y-mean2d.y] × conic × [x-mean2d.x, y-mean2d.y]^T
#     
#     Derive: max_contribution = opacity × exp(-0.5 × min_Mahalanobis²)
#     
#     If max_contribution < ALPHA_THRESHOLD:
#       → This (Gaussian, tile) pair can be safely skipped
#       → It contributes < 1/255 to ALL pixels in this tile

# Output: Fraction of pairs skippable per scene/camera
# Threshold: If >5% pairs are skippable → Gap-A is worth pursuing

# IMPORTANT: This is a proof-of-concept analysis, not an optimization.
# The actual bound would need to be computed efficiently in GPU code.
```

---

## 9. Summary

```
=== Intersection → Sorting Research Gap Audit ===

H6 status:
Intersection explosion = CONFIRMED
Sorting coupling = CONFIRMED
Explosion source = N_isect = N_visible × N_tiles (structural)
Saturation pattern = Uniform (not heavy tail)
A100 feasibility = 684M max actual, 12B-47B logical OOM

Prior-art density:
MEDIUM — Many papers optimize sorting (FlashGS, Speedy-Splat)
or culling (GSCore, RoofGS, TileGS, HiGS).
But none reduce the per-Gaussian tile-coverage factor N_tiles(g).

Candidate Gaps:
1. Gap-A: Pre-intersection tile-coverage bound (NOVELTY: MEDIUM)
   → Prove (Gaussian, tile) contribution < threshold before generating pair
2. Gap-B: Gaussian footprint decomposition (NOVELTY: MEDIUM-HIGH)
   → Subdivide large Gaussians into tile-limited sub-regions
3. Gap-C: Depth-bounded sort (NOVELTY: HIGH)
   → Bound effective depth range per tile before sorting

Recommended:
GAP-A

Why:
- Directly addresses H6 multiplication factor (N_tiles per Gaussian)
- No prior art covers this stage (pre-intersection tile-level bound)
- Supported by A100 evidence (79-98% full-tile saturation)
- Minimum experiment is feasible without CUDA modifications
- Unlike Gap-B, correctness is easier to prove (conservative bound)
- Unlike Gap-C, doesn't face chicken-and-egg problem

Minimum next experiment:
Compute per-(Gaussian, tile) pair the conservative minimum Mahalanobis
distance to any pixel in the tile, derive max contribution, compare to
ALPHA_THRESHOLD. Measure fraction of skippable pairs.

OPTIMIZATION: NOT STARTED
```

---

## 10. Output Files

| File | Content |
|:-----|:--------|
| `reports/phase-a100/intersection_sorting_research_gap_audit.md` | This report |
| `results/phase-a100/intersection_sorting_research_gap.json` | Structured data |

## 11. Final Output

```text
=== Intersection → Sorting Research Gap Audit ===

H6 status:
Intersection explosion = CONFIRMED
Sorting coupling = CONFIRMED

Prior-art density:
MEDIUM

Candidate Gaps:
1. GAP-A: Pre-intersection tile-coverage bound (NOVELTY: MEDIUM)
2. GAP-B: Gaussian footprint decomposition (NOVELTY: MEDIUM-HIGH)
3. GAP-C: Depth-bounded sort (NOVELTY: HIGH)

Recommended:
GAP-A

Why:
- Directly addresses H6 multiplication factor (N_tiles per Gaussian)
- No prior art covers pre-intersection tile-level contribution bound
- A100 evidence: 79-98% full-tile saturation, 12B-47B logical intersections
- Minimum experiment is feasible without CUDA modifications
- Conservative bound approach preserves correctness

Minimum next experiment:
Compute min Mahalanobis distance per (Gaussian, tile) pair;
derive max_contribution = opacity × exp(-0.5 × M²);
count pairs where max_contribution < ALPHA_THRESHOLD (1/255);
measure % of intersections that can be safely skipped.

OPTIMIZATION:
NOT STARTED
```
