# C38: Training-Time Projection/Intersection Reuse Validity Study — Final Report

## Verdict: **DROP**

K=0 in all 499 consecutive iteration pairs. No exact-valid reuse window exists.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Scene | room (Mip-NeRF 360) |
| Resolution | 1080p (1920×1080) |
| Camera | Fixed camera 0 |
| Checkpoint | iter 5000 (1,000,684 Gaussians, SH degree 3) |
| Optimizer | Fresh Adam (no state from checkpoint) |
| lr_xyz | 1.6e-4 × 46.64 = 7.46e-3 |
| Densification | **DISABLED** (densification_start=999999) |
| Pruning | **DISABLED** (prune_start=999999) |
| Topology | Fixed (N=1,000,684 throughout) |
| Warmup | 50 iterations |
| Recorded iterations | 500 |
| tile_size | 16 |
| packed | True |
| eps2d | 0.1 |

---

## Q1: Adjacent training iteration state — exact stable?

**NO.** State changes every single iteration.

| Metric (K=1) | Result |
|--------------|--------|
| Intersection exact equality (flatten_ids + isect_offsets) | **0/499 (0.0%)** |
| Radii exact equality | **0/499 (0.0%)** |
| flatten_ids exact equality | **0/499 (0.0%)** |
| isect_offsets exact equality | **0/499 (0.0%)** |

**Every single pairwise comparison across 499 consecutive iteration pairs failed exact equality on ALL intersection state fields.**

### Root cause: discrete threshold crossings

With 1,000,684 Gaussians, even though individual parameter changes are small, the probability that **at least one** Gaussian crosses a discrete boundary per iteration approaches 1.0:

| Discrete event | Measured frequency |
|----------------|-------------------|
| Visibility change (radii 0↔>0) | **74.2 per iter** (median=75, max=142) |
| Tile boundary crossing (n_isects change) | **4,874 per iter** (median |Δn_isect|=2,402) |
| As % of total intersections | 0.196% per iter |

Even 1 boundary crossing out of 1M Gaussians invalidates the entire `flatten_ids` array. With 74 visibility changes and ~2400 intersection changes per iteration, exact equality is impossible.

---

## Q2: Average reusable iterations?

**K = 0.** No iterations are reusable.

| K | Valid pairs / total | % valid |
|---|---------------------|---------|
| 1 | 0/499 | 0.0% |
| 2 | 0/498 | 0.0% |
| 4 | 0/496 | 0.0% |
| 8 | 0/492 | 0.0% |
| 16 | 0/484 | 0.0% |

**P50 = 0, P90 = 0, P99 = 0, max K = 0.**

---

## Q3: Validity window vs parameter update magnitude?

**No stable relationship exists** because invalidation is driven by discrete threshold crossings, not continuous deltas.

| Metric | Value |
|--------|-------|
| Min |dxyz| when invalid | 2.967e-05 |
| Max |dxyz| when invalid | 6.649e-05 |
| Min total |Δθ| when invalid | 2.835 |
| Max total |Δθ| when invalid | ~50 |

ALL pairs are invalid regardless of parameter delta magnitude. There is **no ε threshold** below which the state remains exact-valid. Even the smallest parameter update (|dxyz|=2.967e-05) causes at least 1 Gaussian to cross a discrete boundary among 1M Gaussians.

### Why C37-C's analytical estimate was wrong

C37-C estimated ">99.9% unchanged" based on:
```
Δmeans = lr_xyz × spatial_lr_scale × ||grad|| ≈ 4.5e-4 units/step
Relative change: 4.5e-4 / 2.5 = 0.018% per step
```

This was correct for **mean** per-Gaussian change but wrong for **max** change and for **discrete state** validity:

| What C37-C assumed | What C38 measured |
|--------------------|-------------------|
| means2d max delta < 0.05 pixels | **means2d max delta p50 = 18.28 pixels** |
| 99.9% of state unchanged | **0% of intersection state unchanged** |
| Copy vs recompute: 0.01ms vs 1.7ms | **Forward cost = 5.81ms (not 1.7ms)** |
| "Unchanged at output level" | **74 GS change visibility per iter** |

The means2d max delta of 18+ pixels occurs because:
1. Adam momentum amplifies gradients for actively-optimized Gaussians
2. Some Gaussians near camera have large screen-space projections
3. The **max** (not mean) delta determines discrete state invalidation
4. Mean delta is 0.29 pixels (small), but max is 18+ pixels (large)

---

## Q4: Fixed camera K-step reuse window?

**No reuse window.** With fixed camera 0 for 500 iterations, K=0 throughout.

The means2d max delta grows with K (as expected):
| K | means2d max delta p50 | means2d max delta p90 |
|---|----------------------|----------------------|
| 1 | 18.28 | 65.45 |
| 2 | 34.79 | 134.7 |
| 4 | 65.04 | 240.3 |
| 8 | 117.6 | 430.1 |
| 16 | 202.5 | 746.4 |

This confirms state diverges monotonically — no convergence to a stable point within 500 iterations.

---

## Q5: Camera change → immediate state invalidation?

**YES, but it doesn't matter — state is already invalid from parameter updates alone.**

| Adjacency type | Exact equality |
|----------------|---------------|
| Same-camera (A→A) | 0/40 (0.0%) |
| Cross-camera (A→B or B→A) | 0/9 (0.0%) |

Camera changes add additional invalidation (different projection, different tile assignments), but parameter-update-driven invalidation already makes K=0 for same-camera. Camera change is not the bottleneck.

---

## Q6: Projection vs intersection stability?

**Projection is continuous (changes every iter but small mean delta).**
**Intersection is discrete (changes every iter, exactly invalidated).**

| State type | K=1 behavior | Stability |
|------------|-------------|-----------|
| means2d (continuous) | mean delta = 0.29 px, max delta = 18+ px | Changes every iter (float) |
| conics (continuous) | max delta = 1.6e-6 | Changes every iter (float) |
| depths (continuous) | max delta = 1.5e-2 | Changes every iter (float) |
| radii (discrete) | 74 GS change visibility | **Never exact equal** |
| flatten_ids (discrete) | 2402 intersections change | **Never exact equal** |
| isect_offsets (discrete) | tile counts change | **Never exact equal** |

Projection state (means2d, conics, depths) is **never bit-exact** because floating-point arithmetic produces different results for any parameter change. But the changes are small in mean.

Intersection state (flatten_ids, isect_offsets) is **never exact-equal** because discrete threshold crossings (tile boundaries, visibility) occur every iteration.

---

## Q7: Reuse projection only (not intersection)?

**No independent value.** The projection state (means2d, conics, depths) is never bit-exact either. Even if we accepted approximate projection reuse, the intersection state depends on projection — if means2d changes by even 0.001 pixels, the tile intersection test may produce different results for GS near tile boundaries.

With 1M Gaussians and 8160 tiles, there are always GS within 0.001 pixels of a tile boundary. Any parameter change, no matter how small, will push some of them across.

---

## Per-Iteration Detail (first 20 pairs)

```
iter   n_isect    Δn_isect  vis_chg    m2d_max   depth_max  flat_eq   off_eq  isect_eq     |dxyz|
   1   4260658     -13243       11    9.196e+02   1.522e-02        N        N        N   2.967e-05
   2   4247415     -15233       12    8.562e+02   1.506e-02        N        N        N   3.232e-05
   3   4232182     -16091       20    7.976e+02   1.520e-02        N        N        N   3.470e-05
   4   4216091     -13769       17    7.392e+02   1.531e-02        N        N        N   3.651e-05
   5   4202322     -13036       22    6.852e+02   1.495e-02        N        N        N   3.790e-05
```

Key observations:
- **n_isects monotonically decreasing** (4.26M → 3.98M in 20 iters): model is converging, Gaussians shrinking
- **visibility changes**: 11-31 GS per iter become invisible (opacity decreasing)
- **means2d max delta decreasing** (919→186 px): Adam momentum building, then stabilizing
- **|dxyz| increasing** (2.97e-05 → 5.01e-05): Adam momentum ramping up

---

## Cost Measurement

| Metric | Value |
|--------|-------|
| Forward (project + intersect + rasterize) | **mean=5.81ms, median=5.38ms, p90=6.15ms** |

This is the **actual reference recomputation cost**. C37-C estimated 1.7ms for projection alone; the full forward is 5.81ms. Even if reuse were possible (it isn't), the maximum savings would be 5.81ms/iter.

---

## Why This Fails: The Curse of Dimensionality

The fundamental problem is statistical:

```
P(no GS crosses boundary in 1 iter) = (1 - p_cross)^N

where:
  p_cross = probability that 1 GS crosses a tile boundary per iter
  N = 1,000,684 Gaussians
```

Even if p_cross = 10^-6 (one in a million per GS per iter):
```
P(no crossing) = (1 - 10^-6)^1000684 ≈ e^(-1) ≈ 0.37
```

With p_cross = 74/1000684 ≈ 7.4e-5 (measured):
```
P(no crossing) = (1 - 7.4e-5)^1000684 ≈ e^(-74) ≈ 0
```

**With 1M+ Gaussians, exact state reuse is mathematically impossible** unless ALL gradients are exactly zero (convergence) or the number of Gaussians is drastically reduced.

---

## Correction to C37-C

C37-C claimed:
- ">99.9% per iteration unchanged"
- "Copy: 0.01ms vs Recompute: 1.7ms (170x cheaper)"
- "means2d changes by <<0.05 pixels"

C38 measured:
- **0% of intersection state unchanged** (not 99.9%)
- Forward cost = **5.81ms** (not 1.7ms)
- means2d max delta = **18.28 pixels** p50 (not <<0.05 pixels)
- Visibility changes = **74 per iter** (not ~0)

**C37-C's KEEP verdict is overturned by C38's exact measurement.**

The error in C37-C was using **mean** per-Gaussian change (0.018%) to predict **discrete** state stability. Discrete state requires **all** Gaussians to not cross any boundary — the max, not the mean, determines validity.

---

## Final Verdict

| Criterion | Result |
|-----------|--------|
| K >= 2 observed? | **NO** — K=0 in all 499 pairs |
| K >= 1 observed? | **NO** — 0/499 valid |
| Any exact-valid window? | **NO** |
| Projection reuse viable? | **NO** — never bit-exact |
| Intersection reuse viable? | **NO** — 74 boundary crossings per iter |
| Camera A/B recurrence? | **NO** — 0% for both same and cross-camera |

### **DROP**

No exact-valid reuse window exists. The C37-C state reuse hypothesis is refuted by measurement. With 1M+ Gaussians, discrete threshold crossings occur every iteration, making exact state equality mathematically impossible.

---

## Artifacts

- Experiment script: `scripts/phase-c31/c38_validity_study.py`
- Raw data: `results/phase-c31/c38_validity_data.json`
- Checkpoint: `results/epic05/phase7/phase7_room_30k_16/phase7_room_30k_16_iter5000.pt`
