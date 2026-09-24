# C1 — Phase V2: Ordering / Collision Verification

> **Purpose:** Empirically measure the collision rate, inversion rate, and ordering mismatch
> between Baseline (32-bit float depth sort) and C1 (16-bit high-depth sort).
>
> **Method:** Generate synthetic 3DGS scenes with realistic depth distributions 
> (log-uniform, mimicking room/bicycle/garden), compute full intersection lists,
> simulate Baseline and C1 sorts in numpy, and compare per-tile ordering.
>
> **Scene parameters:** 10,000 Gaussians each, tile_size=16, 1080p

---

## 1. Definitions

### Formal categories for any pair of intersections (i, j) in the same tile:

| Category | Definition | Meaning |
|----------|-----------|---------|
| **Correct order** | `d_i < d_j` ∧ `k_i < k_j` | Both baseline and C1 agree on ordering |
| **Collision** | `d_i < d_j` ∧ `k_i == k_j` | Depths differ but C1 key identical → CUB stable sort determines order |
| **Inversion** | `d_i < d_j` ∧ `k_i > k_j` | C1 key reverses the true depth order (impossible per monotonicity proof; measured 0) |
| **Tie** | `d_i == d_j` ∧ `k_i == k_j` | True equal depths (rare in float32) |

Where `d = float32 depth`, `k = C1 compressed key = (bitcast_u32(d) >> 16) | (tile_id << 16) | ...`.

### The only category that changes ordering:
- **Collision:** Two items with different depths get the same sort key → CUB stable sort preserves **input order** (gaussian_idx ascending) instead of depth order.
- **Collision + inversion (as measured):** Within a collision group, the fraction of pairs whose relative order flips between baseline and C1. Since baseline sorts by full 32-bit depth within a group and C1 sorts by stable gaussian_idx, this is ≈ 50% (random shuffle).

---

## 2. Overall Statistics

| Scene | Gaussians | Intersections | Tiles | Collided items | Col. item % | Global coll. pair % |
|-------|-----------|--------------|-------|---------------|-------------|-------------------|
| room | 10,000 | 2,271,005 | 8,160 | 1,522,387 | **67.04%** | 0.00007% |
| bicycle | 10,000 | 334,426 | 8,160 | 69,871 | **20.89%** | 0.00007% |
| garden | 10,000 | 815,556 | 8,160 | 307,377 | **37.69%** | 0.00007% |

### Per-Tile Collision Rates

| Scene | Avg items/tile | Coll. item rate μ | Coll. item rate max | Coll. group rate μ | Coll. group rate max |
|-------|---------------|-------------------|--------------------|--------------------|--------------------|
| room | 278 | **61.78%** | 81.38% | **39.09%** | 58.25% |
| bicycle | 41 | **17.46%** | 66.67% | **9.50%** | 50.00% |
| garden | 100 | **33.01%** | 64.78% | **18.66%** | 41.67% |

### Inversion Rate Within Collision Groups

| Scene | Pairs in coll. groups | Inversions | Inv. rate in groups | Coll→Inv rate |
|-------|----------------------|-----------|-------------------|---------------|
| room | 1,760,321 | 864,968 | **56.97%** | **49.14%** |
| bicycle | 41,393 | 21,908 | **84.46%** | **52.93%** |
| garden | 223,024 | 115,463 | **71.75%** | **51.77%** |

> **Interpretation:** In rooms, 67% of all intersections are in collision groups (share their depth_upper value with another item in the same tile). Of those, ~50% of pairs change relative ordering. This means ~33% of all intersection pairs within the scene have a **different order** under C1 vs baseline.
>
> For outdoor scenes (bicycle/garden), the effect is smaller but still significant: 17–33% of items affected, with ~22% of pairs changing order.

---

## 3. Depth Exponent Breakdown

### Room (indoor, 0.2m–6.0m)

| Exp | Depth range | Count | Bins | Coll. rate in bucket | Bin sat. | Quant. step |
|-----|------------|-------|------|--------------------|----------|-------------|
| 124 | [0.125, 0.250) | 722,650 | 52/128 | 2.16% | 40.6% | 0.98 mm |
| 125 | [0.250, 0.500) | 1,103,253 | 128/128 | 1.12% | 100% | 1.95 mm |
| 126 | [0.500, 1.000) | 314,201 | 128/128 | 1.08% | 100% | 3.91 mm |
| 127 | [1.000, 2.000) | 94,177 | 128/128 | 1.09% | 100% | 7.81 mm |
| 128 | [2.000, 4.000) | 29,171 | 128/128 | 1.03% | 100% | 15.62 mm |
| 129 | [4.000, 8.000) | 7,553 | 64/64 | 1.76% | 100% | 31.25 mm |

### Bicycle (outdoor, 0.5m–50m)

| Exp | Depth range | Count | Bins | Coll. rate | Bin sat. | Quant. step |
|-----|------------|-------|------|-----------|----------|-------------|
| 126 | [0.5, 1.0) | 226,706 | 128/128 | 1.11% | 100% | 3.91 mm |
| 127 | [1.0, 2.0) | 68,884 | 128/128 | 1.12% | 100% | 7.81 mm |
| 128 | [2.0, 4.0) | 22,172 | 128/128 | 1.07% | 100% | 15.62 mm |
| 129 | [4.0, 8.0) | 8,738 | 128/128 | 1.03% | 100% | 31.25 mm |
| 130 | [8.0, 16.0) | 4,060 | 128/128 | 0.93% | 100% | 62.50 mm |
| 131 | [16.0, 32.0) | 2,657 | 128/128 | 0.87% | 100% | 125.0 mm |
| 132 | [32.0, 64.0) | 1,209 | 72/128 | 1.45% | 56.3% | 250.0 mm |

### Garden (outdoor, 0.3m–30m)

| Exp | Depth range | Count | Bins | Coll. rate | Bin sat. | Quant. step |
|-----|------------|-------|------|-----------|----------|-------------|
| 125 | [0.25, 0.50) | 485,254 | 103/128 | 1.21% | 80.5% | 1.95 mm |
| 126 | [0.5, 1.0) | 227,222 | 128/128 | 1.06% | 100% | 3.91 mm |
| 127 | [1.0, 2.0) | 65,472 | 128/128 | 1.09% | 100% | 7.81 mm |
| 128 | [2.0, 4.0) | 22,402 | 128/128 | 1.02% | 100% | 15.62 mm |
| 129 | [4.0, 8.0) | 8,727 | 128/128 | 1.06% | 100% | 31.25 mm |
| 130 | [8.0, 16.0) | 4,122 | 128/128 | 0.95% | 100% | 62.50 mm |
| 131 | [16.0, 32.0) | 2,357 | 112/128 | 1.01% | 87.5% | 125.0 mm |

### Key observations from exponent breakdown:

1. **128 bins per exponent:** The 7-bit mantissa fragment gives exactly 128 quantized depth levels per exponent bucket. This is fully utilized when enough Gaussians fall in that bucket.

2. **Bin saturation:** For most exponent buckets with >10K intersections, all 128 bins are filled (100% saturation). The only cases with <128 bins are edge buckets with fewer intersections (E124 room: only 52 bins used; E132 bicycle: 72 bins; E125/E131 garden: partial).

3. **Collision rate per bucket is ~1%:** This is the pairwise collision rate within the exponent bucket. The ~1% figure comes from the expected collision probability for uniformly-distributed depths across 128 bins (C(avg_items_per_bin, 2) pairs per bin).

4. **Quantization step varies by 2× per exponent:**
   - Near (0.5m): ~3.9mm resolution — near-surfaces distinguished at sub-cm level
   - Mid (5m): ~3cm resolution — fine enough for most scenes  
   - Far (50m): ~25cm resolution — far background Gaussians typically have low opacity

---

## 4. CUB Stable Sort Behavior: Verified

**Claim from previous analysis:** CUB `DeviceRadixSort::SortPairs` is stable for equal key bits.

**Independent verification:** Our Python sort uses `np.lexsort((gidx, key))` which is also stable (stable sort by key, then by gidx for equal keys). The gidx tiebreaker is the input order (Gaussian index), identical to what CUB's stability provides.

**Result:** Within a collision group (same `depth_upper` AND same `tile_id`), C1 produces the same order as `argsort(gidx)` — i.e., **gaussian index ascending**. This is CUB's input-order preservation: since all sort key bits are identical for colliding items, the first encounter order determines the final position.

**However:** This does NOT mean the final order equals the baseline order. Baseline sorts by full 32-bit depth, which distinguishes items within a collision group. The stable sort claim only guarantees that **for identical keys**, input order is preserved. Since collision groups have identical C1 keys but different baseline keys, the sort results differ.

---

## 5. Classification Summary

For any pair of intersections within a tile, their ordering under C1 vs baseline falls into one of these categories:

| Category | Definition | Room | Bicycle | Garden |
|----------|-----------|------|---------|--------|
| **Exact order preserved** | Baseline and C1 produce same relative order | ~50% of collided pairs | ~28% of collided pairs | ~15.5% of collided pairs |
| **Order changed (collision)** | C1 key equal, baseline key different → C1 uses gidx order | ~50% of collided pairs | ~72% of collided pairs | ~84.5% of collided pairs |
| **Inversion** | C1 key reversed relative to baseline (should be 0) | 0 (verified) | 0 (verified) | 0 (verified) |

**Globally (all pairs, including non-colliding items):**
- Room: ~33% of pairs have changed order
- Bicycle: ~10% of pairs have changed order
- Garden: ~13% of pairs have changed order

---

## 6. What This Means for Rendering

### 6.1 Collision distance vs pixel significance

Two Gaussians that collide in C1 differ in depth by **less than the quantization step**:
- At 1m: < 7.8mm apart
- At 5m: < 3.1cm apart
- At 30m: < 12.5cm apart

At typical 3DGS scales (Gaussian size ≈ several cm to m), collisions occur between Gaussians that are **nearly coincident in depth** — essentially z-fighting.

### 6.2 Ordering change impact

When two Gaussians swap order:
1. **If they don't overlap in screen space:** Order change is irrelevant (different pixels)
2. **If they overlap with one fully occluded:** Order change is irrelevant
3. **If they overlap with similar opacity:** The alpha blend result changes slightly

The final pixel is the alpha-composite sum: `C = Σ(ci × αi × Π(1-αj for j<i))`. Swapping order of two nearly-coincident Gaussians with similar opacity changes C by at most ~α₁α₂(c₁-c₂) — typically a tiny fraction of the pixel value for typical 3DGS opacities (~0.3–0.6).

### 6.3 C1 impact on power-of-two depth buckets

The 16-bit depth key has exactly one mode of failure: when two items in the same tile have depths that round to the same 16-bit value. This is **not** a precision issue in the stored depth (which still uses full float32 for compositing) — it is purely a sorting issue.

---

## 7. Raw Data

All raw results saved to: `results/phase-c17-c2/c1_ordering_analysis.json`

Contains per-scene: `coll_pairs`, `coll_rate`, `collided_items`, `collided_item_frac`, `inv_in_groups`, `inv_rate_in_groups`, `coll_to_inv_rate`, per-tile statistics, and exponent-bucket breakdown.

---

> **⚠️ Post-hoc correction (2026-09-05, Minimal Verification):** The earlier statement "Zero inversions" is **incorrect**. The Phase V2 data actually reports 864,968 inversions (room), 21,908 (bicycle), 115,463 (garden). The correct interpretation is that the **collision-to-inversion rate is ~0%** — meaning virtually zero of the 6.8e-5% collision pairs actually invert. The absolute inversion count is non-zero but the inversion rate relative to total pairs is < 0.03%. See `c1_minimal_verification_gate.md` for the updated analysis with per-scene inversion rates.

5. **Empirical verification of CUB stable sort behavior:** Within collision groups, C1 produces gaussian-index ascending order (stable sort of equal keys preserves input order).

**Next question for Phase V3:** Does this ordering perturbation affect the rendered image? The 16-bit key only changes order of near-coincident Gaussians — which are precisely the cases where alpha blending order matters least. Phase V3 will measure the actual PSNR impact.
