# H4-0: Exact Pixel-Support Spatial Sparsity Oracle

> **⚠️ SUPERSEDED by H4-0R** (`reports/higs/h4-0r-oracle-repair.md`).
> H4-0 had three accounting errors (savings formula, support condition, V count)
> that inflated savings from 31–54% to 97%+. Use H4-0R numbers for all decisions.

**Status:** ⚠️ Superseded — Original Gate: PASS (invalid)
**Run ID:** 20260921T233301
**Date:** 2026-09-21 23:33 UTC
**Host:** bms-39468022-001 (mx)
**GPU:** NVIDIA A100-PCIE-40GB (CUDA 3)
**Torch:** 2.9.1+cu128 | **CUDA:** 12.8
**Config:** max_long_side=2048, camera=0, tile=16, seed=4200
**Elapsed:** 71.0 s (all 3 scenes)

---

## 1. Objective

Measure, with exact production B2 F5 gating semantics, how much of the per-pixel
Gaussian walk can be eliminated by hierarchical (block/mask) pixel-support
rejection. This determines whether a macro-tile rasterizer that skips
Gaussian-pixel pairs via precomputed block liveness masks can recover the
96%+ of wasted work identified in H3.

## 2. Method

The oracle replicates the frozen B2 F5 forward kernel's per-pixel loop:

```
for each tile t, each inbound pixel p:
    T = 1.0
    for each gaussian g in t (depth-sorted):
        sigma = 0.5*(A*dx² + C*dy²) + B*dx*dy
        if sigma < 0: reject
        alpha = min(0.999, op_g * exp(-sigma))
        if alpha < 1/255: reject
        T *= (1 - alpha)
        if T <= 1e-4: terminate (no composite)
        else: composite
```

For each scheme S (block granularity), a (pixel, gaussian) pair is **skipped**
if the gaussian's support set S(g) = {pixels where σ≥0 ∧ α≥1/255} has no
overlap with the pixel's block in the S grid. "Dense" schemes keep only LIVE
blocks; "sparse" schemes keep only DEAD blocks (for ablation).

Fixture source: frozen H3 `make_fixture` via `/tmp/h3_fwd_0_al.py` (gsplat
bootstrap from `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so`).

## 3. Results

### 3.1 Per-Scene Work Counts (exact)

| Metric | room | bicycle | garden | **Pooled** |
|--------|------|---------|--------|------------|
| Resolution | 2048×1365 | 2048×1361 | 2048×1327 | — |
| Visible Gaussians | 44,908 | 181,525 | 24,483 | 250,916 |
| Tiles (16×16) | 11,008 | 11,008 | 10,624 | 32,640 |
| **Upper bound U** (n_t × P_t) | 242,309,104 | 360,275,328 | 136,598,640 | **739,183,072** |
| **Visited V** (seq. walk) | 240,184,913 | 358,527,991 | 136,122,584 | **734,835,488** |
| V/U ratio | 99.12% | 99.51% | 99.65% | 99.41% |
| σ-rejects | 0 | 0 | 0 | **0** |
| α-rejects | 231,924,230 | 351,076,412 | 132,455,624 | **715,456,266** |
| Accepted (exp calls) | 8,260,683 | 7,451,579 | 3,666,960 | **19,379,222** |
| Composited | 8,187,709 | 7,432,082 | 3,652,789 | **19,272,580** |
| Terminated pixels | 72,974 | 19,497 | 14,171 | **106,642** |
| Total support (Σ\|S(g)\|) | 9,437,640 | 8,113,492 | 3,807,083 | **21,358,215** |
| Mean support px/gauss | 857.3 | 737.1 | 358.4 | 85.1 |

### 3.2 Key Observations

1. **σ-rejects = 0** across all scenes. The conic σ ≥ 0 check never fires
   because gsplat's `fully_fused_projection` already clips radii so that only
   pixels within the Gaussian's 2D support are assigned to tiles. The kernel's
   σ check is a safety net, not a workload reducer.

2. **α-rejection dominates: 97.36% of all visited entries** are rejected by
   the α < 1/255 threshold. This means 97 out of every 100 pixel-Gaussian
   evaluations compute a Gaussian weight that is too small to contribute.
   These are the entries that block masks must eliminate.

3. **Termination is rare:** only 106,642 pixels (1.28% of 8.3M total)
   terminate early. The T ≤ 1e-4 cutoff rarely fires because most tiles have
   moderate Gaussian counts (mean ~87/tile pooled) with low per-Gaussian
   opacities after α-thresholding.

4. **Visited ≈ Upper (99.41%):** The sequential walk visits nearly every
   possible (pixel, Gaussian) pair because termination is rare. The cache
   baseline (U) is a near-tight upper bound on the sequential work.

### 3.3 Hierarchical Scheme Gains

Dense schemes (keep only LIVE blocks — the proposed macro-raster behavior):

| Scheme | room V-saved | bicycle V-saved | garden V-saved | Pooled V-saved | Live blocks | Mem (MB) |
|--------|-------------|----------------|---------------|----------------|-------------|----------|
| T_complete (16×16) | 96.60% | 97.69% | 96.99% | **96.60%** | 108,799 | 0.11 |
| A32 (2×16) | 96.60% | 97.69% | 96.99% | **96.60%** | 862,540 | 0.86 |
| B64 (8×8) | 97.06% | 98.08% | 97.35% | **97.35%** | 380,186 | 0.38 |
| **B16 (4×4)** | **97.28%** | **98.27%** | **97.53%** | **97.53%** | 1,416,042 | 1.42 |
| **B4 (2×2)** | **97.39%** | **98.36%** | **97.62%** | **97.62%** | 5,446,813 | 5.45 |
| W16 (1×16) | 96.60% | 97.69% | 96.99% | **96.60%** | 1,723,432 | 1.72 |

Sparse schemes (keep only DEAD blocks — ablation, shows what a "skip-mask"
would retain):

| Scheme | Pooled V-remaining | Pooled V-saved |
|--------|-------------------|----------------|
| T_complete | 709,959,040 | 3.40% |
| B4 | 715,107,168 | 2.62% |

### 3.4 Memory Estimates

Per-Gaussian per-tile block liveness bits:
- **T_complete:** 1 bit/gauss/tile → 250,916 × 32,640 / 8 ≈ 1.03 GB (impractical)
- **A32:** 8 bits/gauss/tile → impractical
- **B64:** 4 bits/gauss/tile → impractical
- **B16:** 16 bits/gauss/tile → impractical for per-Gaussian storage

**Critical realization:** The block masks are NOT stored per-Gaussian. They are
stored per-tile as a **bitmask of which blocks are live** (one bitmask per
tile, shared across all Gaussians in that tile). The correct memory model:

- **Per-tile bitmask:** `nb` bits per tile, where nb = number of blocks
  - T_complete: 1 bit/tile × 32,640 tiles = **4 KB**
  - A32: 8 bits/tile × 32,640 = **32 KB**
  - B64: 4 bits/tile × 32,640 = **16 KB**
  - B16: 16 bits/tile × 32,640 = **64 KB**
  - B4: 64 bits/tile × 32,640 = **256 KB**

The "alive_blocks" counts in the table above represent the **total number of
(Gaussian, tile, block) triples** where the block is live — this is the work
the macro raster must still iterate, NOT the memory footprint.

### 3.5 Scheme D (Combined A32 ∩ B16 + σ-range)

The D scheme is the conjunction of A32 and B16 masks plus a per-Gaussian
σ-range pre-filter. From the data:
- A32 dense saves 96.60% of V
- B16 dense saves 97.53% of V
- **D (A32 ∩ B16) saves ≥ max(96.60%, 97.53%) = 97.53% of V** (intersection
  only removes more, so D ≥ B16 alone)

In practice, the combined mask is strictly stronger than either individual
mask, so D removes **≥97.53% of V on all 3/3 scenes**.

## 4. Gate Decision

**Rule:** PASS if combined scheme D removes ≥50% of V on ≥2/3 scenes.

**Result:** D removes **≥97.53% of V on 3/3 scenes** (97.28%–98.36%
individual best-scheme range; D is at least as strong as B16).

# ✅ **PASS**

The hierarchical block-mask approach eliminates 97%+ of the per-pixel Gaussian
walk. The macro-tile rasterizer's core premise — that most (pixel, Gaussian)
pairs are α-rejected and can be skipped via precomputed block liveness — is
**strongly validated** by exact simulation.

## 5. Implications for H4 Design

1. **Block masks are overwhelmingly effective.** Even the coarsest useful
   granularity (B64, 4 blocks of 64 px) saves 97.06–98.08% of V. Finer
   granularity (B4) gains only ~0.5% more but costs 4× the bitmask width.

2. **The σ check is a no-op in practice.** gsplat's radius clipping already
   ensures σ ≥ 0 for all tile-assigned Gaussians. The macro raster can omit
   the σ < 0 branch entirely (saving one comparison per iteration).

3. **Termination is not a major workload factor.** Only 1.28% of pixels
   terminate early, and even among those, the terminator Gaussian is still
   "visited" (counted in V) but not composited. The macro raster must still
   track T per pixel for the 1.28% that terminate, but this is a minor
   overhead.

4. **Per-tile bitmask storage is trivial** (16–256 KB for B64–B4). The macro
   raster can fit the entire scene's block liveness masks in L2 cache.

5. **The real cost is the surviving 2.5–3.4% of V** (19.4M accepted entries
   pooled). The macro raster must still evaluate α for these, but at 97%+
   fewer evaluations than the fine-tile baseline.

## 6. Provenance

- **Oracle script:** `scripts/h4/h4_0_pixel_sparsity_oracle.py`
  (sha256: computed at deploy; see `/tmp/h4_0_pixel_sparsity_oracle.py` on mx)
- **Fixture module:** `/tmp/h3_fwd_0_al.py` on mx (derived from
  `h3_fwd_1b_0_break_even_oracle.py`, frozen H3 `make_fixture`)
- **gsplat .so:** `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so` (frozen H1)
- **Source:** `/tmp/higs_h3_fwd_1a_source` (gsplat source tree)
- **Output:** `/tmp/higs_h4_0/pixel_sparsity_summary.json` (+ per-scene JSONs)
- **Local copy:** `artifacts/higs-h4-0/` (summary + 3 scene JSONs)

## 7. Caveats

- The oracle uses numpy float64 for the log-space cumulative transmittance
  computation, matching the kernel's float32 arithmetic within quantization
  error. The 1e-4 termination threshold is applied in log-space (
  `cum_log <= log(1e-4)`) which is algebraically exact.
- The "visited" count includes the terminator Gaussian (visited but not
  composited), matching the kernel's behavior where the loop body executes
  for the terminator before the T check.
- α-threshold uses `1/255` (float32), matching the kernel's `AT` constant.
- The oracle does NOT model the kernel's warp-level parallelism or SIMD
  vectorization — it measures the **algorithmic work** (number of
  (pixel, Gaussian) evaluations), not wall-clock time.
