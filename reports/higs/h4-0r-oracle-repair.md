# H4-0R: Pixel-Support Oracle Repair & HiGS 32-G Mask Feasibility

**Status:** Complete — Gate: **H4_0R_WEAK**
**Run ID:** 20260922T002711
**Host:** bms-39468022-001 (mx)
**GPU:** NVIDIA A100-PCIE-40GB (CUDA 3)
**Torch:** 2.9.1+cu128 | **CUDA:** 12.8
**Config:** max_long_side=2048, tile=16, seed=4200
**Elapsed:** 48.4 s (3 scenes)

---

## 1. What H4-0 Got Wrong

H4-0 reported **96.6–98.4% V-savings** for B16/B4 block masks. H4-0R corrects
three accounting errors that inflated these numbers by ~2×:

| Error | H4-0 | H4-0R (correct) | Impact |
|-------|------|-----------------|--------|
| Savings formula | `(U − K) / V` | `(V − K) / V` | U ≈ 1.25×V, so savings inflated by ~25% of U/V |
| Support condition | `σ ≥ 0` (trivially true) | `σ ≤ log(255·op)` (alpha gate) | H4-0 counted ALL pixels in support; H4-0R counts only alpha-accepted |
| Visited count | All (pixel, G) pairs | Respects termination (T ≤ 1e-4) | H4-0 overcounted V by including post-termination pairs |

**Net effect:** H4-0's "97% savings" became **31–54% savings** (B16). The
block mask approach is still beneficial, but far from transformative.

## 2. Corrected Baseline Work Counts

### Per-Scene

| Metric | room | bicycle | garden | **Pooled** |
|--------|------|---------|--------|------------|
| Visible Gaussians | 44,908 | 181,525 | 24,483 | 250,916 |
| F4 tile-G entries | 953,144 | 1,412,193 | 533,928 | 2,899,265 |
| **U** (n_t × 256) | 242,309,104 | 360,275,328 | 136,598,640 | **739,183,072** |
| **V** (visited) | 180,621,763 | 282,310,715 | 107,900,702 | **570,833,180** |
| V/U ratio | 74.54% | 78.36% | 78.99% | 77.22% |
| σ evaluations | 180,621,763 | 282,310,715 | 107,900,702 | 570,833,180 |
| σ rejects | **0** | **0** | **0** | **0** |
| **exp evaluations** | **180,621,763** | **282,310,715** | **107,900,702** | **570,833,180** |
| α rejects | 78,984,315 (43.7%) | 178,480,277 (63.2%) | 40,079,732 (37.2%) | 297,544,324 (52.1%) |
| **α accepted (A)** | **101,637,448** | **103,830,438** | **67,820,970** | **273,288,856** |
| Composited | 100,017,704 | 103,000,699 | 66,926,072 | 270,944,475 |
| Terminated pixels | 1,619,744 | 829,739 | 894,898 | 3,344,381 |
| Support size (occ) | 144,165,760 | 134,169,986 | 82,343,868 | 360,679,614 |

### Key Correction: Baseline exp Calls

Since σ_neg = 0 for all scenes:

```
baseline_exp_calls = V - σ_neg = V
```

Every visited (pixel, Gaussian) pair evaluates `exp(-σ)`. The α-threshold
check happens AFTER exp. Therefore:

- **exp evaluations = V = 570.8M** (pooled)
- **α accepted = A = 273.3M** (47.9% of exp calls)
- **α rejected = 297.5M** (52.1% of exp calls)

H4-0 incorrectly reported `exp = 19.4M` (pooled), which was actually the
count of (visited ∧ support) pairs using the wrong support condition.

## 3. Correct Support Condition

### Exact Alpha Gate

```
alpha = min(0.999, opacity · exp(-σ))
accept  iff  alpha ≥ 1/255
```

For the lower gate (ignoring the min with 0.999, which rarely binds):

```
opacity · exp(-σ) ≥ 1/255
⟹  exp(-σ) ≥ 1/(255 · opacity)
⟹  -σ ≥ -log(255 · opacity)
⟹  σ ≤ log(255 · opacity)          [valid when opacity ≥ 1/255]
```

When `opacity < 1/255`: no pixel can be accepted (max alpha = opacity < 1/255).

### Relation to Conic Quadratic

```
σ = 0.5(A·dx² + C·dy²) + B·dx·dy
```

The support region `σ ≤ log(255·op)` is an **ellipse** in (dx, dy) space
centered at the Gaussian's projected center (gx, gy), with shape determined
by the conic coefficients (A, B, C). This is the same ellipse that Speedy-Splat
analytically computes for tile-level culling, but H4-0R evaluates it
per-pixel for exact counting.

### Why σ ≥ 0 Is Not a Useful Criterion

The conic matrix `[[A, B/2], [B/2, C]]` is **positive-definite** for all
valid projected Gaussians (A > 0, C > 0, AC > B²/4), because it is the
inverse of the 2D covariance matrix of a valid 3D Gaussian projection.
Therefore σ ≥ 0 for **all** pixel positions, not just those within the
projected ellipse. The `σ < 0` check in the kernel is a safety net for
degenerate cases (NaN, zero-scale Gaussians) that never occur in practice.

## 4. Sigma-Negative Explanation

**Finding: σ_neg = 0 across all 3 scenes, 570.8M total evaluations.**

**Cause (two independent guarantees):**

1. **Positive-definite conic invariant:** For any valid 3D Gaussian with
   positive scales, the projected 2D conic (A, B, C) is positive-definite.
   This means σ = 0.5(A·dx² + C·dy²) + B·dx·dy ≥ 0 for ALL (dx, dy), with
   equality only at the Gaussian center. This is a mathematical invariant,
   independent of any clipping.

2. **gsplat radius/tile clipping:** `fully_fused_projection` clips each
   Gaussian's 2D radius to the tile grid, so only pixels within the
   projected support are assigned to tiles. This is redundant with
   guarantee #1 for σ ≥ 0, but ensures the (pixel, Gaussian) pair set is
   bounded.

**Implication:** Removing `σ < 0` from the production fast path is
**semantically safe** for all valid projected Gaussians. The check can be
moved to a debug/assert path. (Not done in this oracle; production change
deferred to H4-H.)

## 5. Corrected Block Scheme Savings

### Per-Scene (saved_vs_visited = (V − K) / V)

| Scheme | room saved | bicycle saved | garden saved | Pooled saved |
|--------|-----------|--------------|-------------|-------------|
| B64 (4 blocks) | 25.90% | 39.50% | 21.80% | **31.85%** |
| **B16 (16 blocks)** | **36.34%** | **53.79%** | **30.74%** | **43.91%** |
| B4 (64 blocks) | 41.31% | 60.20% | 35.04% | **49.47%** |

### Comparison with H4-0 (inflated)

| Scheme | H4-0 pooled | H4-0R pooled | Ratio |
|--------|------------|-------------|-------|
| B64 | 97.35% | 31.85% | 3.1× inflation |
| B16 | 97.53% | 43.91% | 2.2× inflation |
| B4 | 97.62% | 49.47% | 2.0× inflation |

The H4-0 numbers were inflated by 2–3× due to the accounting errors in §1.

## 6. Mask Storage Model (Corrected)

The support mask is associated with **(tile, Gaussian-entry)**, not just tile.
Using exact F4 intersection counts:

| Scheme | bits/entry | Packed storage | Aligned storage |
|--------|-----------|---------------|-----------------|
| B64 | 4 | 1.45 MB | 2.90 MB |
| **B16** | **16** | **5.80 MB** | **5.80 MB** |
| B4 | 64 | 23.19 MB | 23.19 MB |

Total F4 entries: 2,899,265 (room 953,144 + bicycle 1,412,193 + garden 533,928).

### Alternative: On-the-Fly Mask Generation (No Persistent Storage)

Instead of storing masks persistently, compute the 16-bit B16 mask **once per
(tile, 32-Gaussian batch) inside F5**, then reuse across the tile's pixel
threads via shared memory or register shuffle:

- **Storage cost:** 0 bytes persistent (16 bits × 32 Gaussians = 64 bytes in
  shared memory per batch, transient)
- **Generation cost:** ~512 bit-ops per 32-G batch per tile (see §7)
- **Benefit:** Eliminates the 5.80 MB persistent mask memory; the mask is
  reconstructed on-demand and discarded after the tile completes.

**Recommendation:** On-the-fly generation is preferable. The 5.80 MB persistent
mask is small in absolute terms, but the on-the-fly approach avoids a memory
bandwidth pass to load/store the mask and simplifies the kernel interface.
The generation cost is negligible relative to the work it avoids (39–67×
ratio, §7).

## 7. B16 False-Positive Efficiency (K/A Ratio)

K/A measures how close the block mask is to pixel-perfect support:
- K/A = 1.0 → every retained pair is α-accepted (no false positives)
- K/A > 1.0 → some retained pairs are α-rejected (false positives within live blocks)

| Scheme | room K/A | bicycle K/A | garden K/A | Pooled K/A |
|--------|---------|------------|-----------|-----------|
| B64 | 1.317 | 1.645 | 1.244 | **1.423** |
| **B16** | **1.131** | **1.257** | **1.102** | **1.172** |
| B4 | 1.043 | 1.082 | 1.034 | **1.056** |

**Interpretation:**
- B16 has **13–26% false-positive overhead** (K/A = 1.10–1.26). Most retained
  pairs are α-accepted, but a significant fraction are α-rejected within
  blocks that contain at least one accepted pixel.
- B4 reduces false positives to **3–8%** (K/A = 1.03–1.08), at the cost of
  4× the mask bits (64 vs 16 per entry).
- **B4 vs B16 tradeoff:** B4 saves an additional 5.6% of V over B16 (pooled:
  49.47% vs 43.91%) while reducing false positives by ~10%. The marginal
  benefit of B4 over B16 is modest.

## 8. HiGS 32-G Transpose Mask Oracle

### Proposed Structure

```
32 depth-ordered Gaussians × 16 B16 blocks
    ↓  per-Gaussian 16-bit mask (which blocks are live)
    ↓  transpose (WarpBitTranspose or equivalent)
16 × uint32  (each uint32: which of 32 Gaussians are live in that block)
```

### Validation of Properties

| Property | Status | Evidence |
|----------|--------|----------|
| 1. ffs preserves depth order | ✅ | Bit 0 = nearest Gaussian; `__ffs` returns lowest set bit = first (nearest) live Gaussian in depth order |
| 2. Zero mask skips entire block | ✅ | If uint32 = 0, all 32 Gaussians are dead in that 4×4 block → skip 16 pixel evaluations entirely |
| 3. No persistent global mask needed | ✅ | Mask generated once per (tile, 32-G batch) in shared memory; discarded after tile |
| 4. Mask gen cost per batch, not per pair | ✅ | 512 bit-ops per 32-G batch vs 16×32 = 512 pixel-G pairs evaluated |
| 5. WarpBitTranspose reusable | ⚠️ Needs verification | HiGS codebase has `warp_bit_transpose` in `higs/csrc/...`; needs confirmation that it supports 32×16 → 16×32 layout |

### Cost/Benefit (Real Scene Counts)

| Scene | Pairs avoided | Mask gen bit-ops | Ratio (avoided/gen) |
|-------|-------------|-----------------|---------------------|
| room | 65,631,268 | 15,250,432 | **43×** |
| bicycle | 151,845,476 | 22,595,584 | **67×** |
| garden | 33,167,852 | 8,543,232 | **39×** |

**Interpretation:** For every 1 bit-op spent generating the mask, 39–67
pixel-G evaluations are avoided (at ~10 FLOPs each). The mask generation
overhead is **negligible** relative to the work avoided. The 32-G transpose
representation is arithmetically favorable.

## 9. Gate Decision

### Criteria

| Criterion | Threshold | room | bicycle | garden | Pass? |
|-----------|----------|------|---------|--------|-------|
| B16 saves ≥90% of V on 3/3 scenes | ≥90% each | 36.34% | 53.79% | 30.74% | ❌ **FAIL** |
| K_B16/A ≤ 1.25 on ≥2/3 scenes | ≤1.25 | 1.131 ✅ | 1.257 ❌ | 1.102 ✅ | ✅ 2/3 |
| Mask gen plausible advantage | >1× | 43× | 67× | 39× | ✅ |

### Decision: **H4_0R_WEAK**

Criterion 1 fails dramatically. B16 removes 30.7–53.8% of V, far below the
90% threshold. The H4-0R_STRONG gate was designed for a scenario where block
masks eliminate the vast majority of the sequential walk; the corrected
numbers show this is not the case.

### Why H4-0 Was Wrong (Root Cause Analysis)

The H4-0 oracle had three compounding errors:

1. **Savings formula used U instead of V:** `(U−K)/V` instead of `(V−K)/V`.
   Since U ≈ 1.25×V, this alone inflated savings by ~20–25 percentage points.

2. **Support condition was `σ≥0` instead of `σ≤log(255·op)`:** Every pixel
   was in the support set, so the block mask "liveness" was determined by
   which pixels are in the tile, not which pixels would actually accept the
   Gaussian. This made the masks nearly all-ones, and the "savings" reflected
   tile geometry, not Gaussian sparsity.

3. **V included post-termination pairs:** The sequential walk's termination
   (T ≤ 1e-4) was not respected in H4-0's V count, inflating V and further
   distorting the savings ratio.

## 10. Implications for H4-H

### What H4-0R Validates

- ✅ The 32-G transpose representation is arithmetically sound (39–67× ratio)
- ✅ B16 block masks provide a real 31–54% reduction in per-pixel evaluations
- ✅ On-the-fly mask generation is preferable to persistent storage
- ✅ K/A ≈ 1.10–1.26 for B16 (near pixel-perfect at 4×4 granularity)
- ✅ σ < 0 check can be removed from the fast path (safe)

### What H4-0R Does NOT Validate

- ❌ A standalone macro rasterizer that replaces F5 entirely
- ❌ A ≥90% reduction in per-pixel work
- ❌ That the 30–54% savings translate to a meaningful E2E speedup
- ❌ That the implementation complexity is justified by the savings

### Recommendation

**H4-H production prototype is NOT authorized** under the H4_0R_STRONG gate.

However, the 30–54% B16 savings are still meaningful. Two paths forward:

1. **Integrate as F5 pre-filter:** Add B16 block masks to the existing F5
   kernel as a pre-filter (skip entire 4×4 blocks where no Gaussian is live)
   rather than building a separate macro rasterizer. This is a smaller,
   lower-risk change that captures most of the B16 benefit.

2. **Re-scope the gate:** If the research lead wants to proceed with a
   moderate-gain candidate, re-scope the H4 gate from "≥90% V reduction" to
   "≥30% V reduction with K/A ≤ 1.25" and re-evaluate. The 32-G transpose
   representation remains attractive regardless of the gate.

## 11. Provenance

| Item | Value |
|------|-------|
| Oracle script | `scripts/h4/h4_0r_oracle.py` (local) → `/tmp/h4_0r_oracle.py` (mx) |
| Fixture module | `/tmp/h3_fwd_0_al.py` on mx (derived from H3 `make_fixture`) |
| gsplat .so | `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so` (frozen H1) |
| F4 intersection counts | room=953,144; bicycle=1,412,193; garden=533,928 (from H3) |
| Output (mx) | `/tmp/higs_h4_0r/h4_0r_summary.json` |
| Output (local) | `artifacts/higs-h4-0r/` (7 JSONs + prior_art_boundary.md) |
| H4-0 (superseded) | `artifacts/higs-h4-0/` + `reports/hig-h4/h4-0-pixel-sparsity.md` |

## 12. Deliverables

```
artifacts/higs-h4-0r/
    h4_0r_summary.json       # Full oracle output (per-scene + pooled)
    corrected_counts.json    # §2: baseline work counts
    exp_accounting.json      # §2: exp calls per scheme
    support_threshold.json   # §3: exact alpha-support equation + sigma invariant
    mask_storage.json        # §5: storage model (per tile-G entry)
    block_efficiency.json    # §7: K/A ratios per scheme
    higs32_mask_oracle.json  # §8: 32-G transpose cost/benefit
    prior_art_boundary.md    # §9: prior-art boundary statement
    final_gate.json          # Gate decision + reasoning
reports/higs/
    h4-0r-oracle-repair.md   # This report
```
