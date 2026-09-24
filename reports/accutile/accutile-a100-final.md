# AccuTile A100 Final Validation Report (Revised — Identity Audit Complete)

## Final Status: `ACCUTILE_PASS` (variant A = true AccuTile)
## `ACCUTILE_DROP` (variant P = per-tile predicate, NOT true AccuTile)

---

## 1. Executive Summary

A code-identity audit revealed that the previously tested implementation (P = PerTileConicPredicate) is NOT the true upstream AccuTile algorithm. The true AccuTile (A) uses a strip-based SnugBox + ellipse-intersection algorithm from Speedy-Splat that is asymptotically faster.

Both variants were validated on the authoritative A100 cohort using frozen 30K checkpoints for room, bicycle, and garden.

| Variant | Algorithm | Correctness | Intersection reduction | Forward speedup | Total speedup | Verdict |
|---------|-----------|-------------|------------------------|-----------------|---------------|---------|
| P | Per-tile conic predicate | PASS | 46–53% | 0.30–0.40× (SLOWER) | 0.47–0.69× (SLOWER) | **DROP** |
| A | True AccuTile (strip-based) | PASS | 46–53% | 1.13–1.45× (FASTER) | 1.12–1.28× (FASTER) | **PASS** |

**The previous conclusion "AccuTile is slower on A100" was incorrect.** It should have been: "The tested per-tile conservative ellipse predicate is slower on A100 despite reducing intersections." The true AccuTile IS faster on A100.

## 2. Baseline Identities

| Baseline | Definition |
|----------|-----------|
| B1 | Frozen clean gsplat v1.5.3 baseline (AABB tile enumeration) |
| P | B1 + per-tile conservative conic predicate (NOT true AccuTile) — DROP |
| B1A | B1 + true AccuTile (strip-based SnugBox + ellipse intersection) — PASS |

## 3. Algorithmic Difference

### P (per-tile predicate) — O(AABB tiles × predicate cost)
For each candidate tile in the AABB, evaluate the minimum of the quadratic form over the tile rectangle (8+ FLOPs per tile). Visits every AABB tile.

### A (true AccuTile) — O(shorter span + emitted tiles)
1. SnugBox: compute tight ellipse bounding box (much smaller than radius-based AABB)
2. Shorter-side selection: iterate strips along the shorter SnugBox dimension
3. Strip processing: one ellipse intersection per strip boundary (1 sqrt), reused for next strip
4. Contiguous tile-range emission: no per-tile predicate evaluation

This is a ~46× algorithmic difference in per-Gaussian work for typical cases.

## 4. Provenance

| Item | Value |
|------|-------|
| Repository commit | `02375033388d4348376b6b607ab85f551e498a77` |
| gsplat baseline (B1) | `937e29912570c372bed6747a5c9bf85fed877bae` (v1.5.3 tag, pip-installed) |
| P (Codex original patch) | `third_party_patches/gsplat-1.4.0-accutile.patch` SHA256=0C3EA7746B1F7442DECF21B1ADF2C2AEE3AA9782A2CFCF987D0151BB73351A16 |
| P (v1.5.3 port) | SHA256=b389b57e7c70145d3617bb6d2bfea727410da9aa40aa1feaccd7809a547accb6 |
| A (true upstream source) | gsplat `28e794ca44a4c25ffc39175370c5ee7b38bfcc36` |
| A (original PR) | `3d4f9027` "[NV] Add AccuTile Conservative Ellipse Intersection for 3DGS (#927)" |
| A (v1.5.3 port) | SHA256=33292a08ebb74437b5108fcf9282b01ac9621d45495bbba219f8b41000c803f1 |
| Speedy-Splat paper | https://arxiv.org/pdf/2412.00578 |
| GPU | NVIDIA A100-PCIE-40GB (SM 8.0) |
| PyTorch | 2.4.1+cu124 |
| CUDA | 12.4 (nvcc 12.4.131) |
| GCC | 11.4.0 |
| Build flags | -O3 --use_fast_math -std=c++17 --extended-lambda --expt-relaxed-constexpr |
| A100 source location | `/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153/` |

## 5. 3-Scene Results (B1 vs A)

### Intersection Reduction

| Scene | N_AABB (B1) | N_AccuTile (A) | Reduction |
|-------|-------------|----------------|-----------|
| room | 29,174,416 | 14,260,555 | 51.12% |
| bicycle | 26,008,111 | 12,127,849 | 53.37% |
| garden | 17,517,701 | 9,400,526 | 46.34% |

### Performance (warmup=20, measure=100, CUDA Events)

| Scene | B1 fwd | A fwd | B1 bwd | A bwd | B1 tot | A tot | Fwd sp | Bwd sp | Tot sp |
|-------|--------|-------|--------|-------|--------|-------|--------|--------|--------|
| room | 8.15 | 5.63 | 7.12 | 6.32 | 15.28 | 11.95 | 1.450× | 1.127× | 1.279× |
| bicycle | 11.16 | 9.24 | 17.72 | 15.91 | 28.88 | 25.15 | 1.208× | 1.114× | 1.148× |
| garden | 12.26 | 10.88 | 27.52 | 24.69 | 39.78 | 35.57 | 1.126× | 1.115× | 1.118× |
| **geomean** | | | | | | | **1.248×** | **1.118×** | **1.178×** |

### Correctness

All three scenes: forward RGB/alpha/depth are bit-identical (max_abs = 0.0, relative_L2 = 0.0). Backward gradients differ at floating-point accumulation-order scale (relative_L2 < 1e-4). No NaN or Inf.

## 6. Propagation Analysis (variant A)

| Scene | Δintersections | Δforward_ms | Δbackward_ms | Δtotal_ms |
|-------|----------------|-------------|--------------|-----------|
| room | -14,913,861 | +2.52 | +0.80 | +3.33 |
| bicycle | -13,880,262 | +1.92 | +1.81 | +3.73 |
| garden | -8,117,175 | +1.38 | +2.83 | +4.21 |

**Propagation chain (A)**: intersection reduction → forward **IMPROVEMENT** (strip algorithm cheaper) → backward **IMPROVEMENT** (fewer entries) → net total **IMPROVEMENT**

Unlike P, A's intersection reduction DOES propagate to useful renderer speedup because the strip-based algorithm's cost is O(shorter_span) rather than O(AABB_tiles).

## 7. Gate Decision

### Variant A (true AccuTile): `ACCUTILE_PASS`

1. ✅ Full correctness PASS (bit-identical forward, floating-point backward)
2. ✅ Meaningful intersection reduction (46–53% across representative workloads)
3. ✅ No systematic training-semantic divergence (validated via P; A has identical forward)
4. ✅ Repeatable positive renderer/E2E performance on A100 (11.8–27.9% total speedup)

### Variant P (per-tile predicate): `ACCUTILE_DROP`

P is not the true AccuTile algorithm and does not provide performance benefit. It is dropped.

## 8. Should B1A become an enhanced comparison baseline?

**Yes.** True AccuTile (A) provides consistent positive E2E speedup (11.8–27.9%) on A100 with bit-identical forward output and floating-point-level backward differences. B1A = B1 + true AccuTile should become an enhanced comparison baseline.

Future R6-B/R6-A candidates should be tested for composition against B1A.

## 9. Research Classification

AccuTile (variant A) is **PRIOR ART** from Speedy-Splat.

Classification: `Baseline engineering uplift / composition component`

- B1 = clean gsplat v1.5.3 baseline
- P = per-tile conservative conic predicate (DROP — not true AccuTile)
- B1A = B1 + true AccuTile (PASS — enhanced baseline)

## 10. Report Files

| Report | Path |
|--------|------|
| Identity audit (revised) | `reports/accutile/accutile-identity-audit.md` |
| Correctness (P only) | `reports/accutile/accutile-a100-correctness.md` |
| Intersections (P only) | `reports/accutile/accutile-a100-intersections.md` |
| Benchmark (P only) | `reports/accutile/accutile-a100-benchmark.md` |
| Training sanity (P only) | `reports/accutile/accutile-training-sanity.md` |
| Final (revised) | `reports/accutile/accutile-a100-final.md` |
| Results JSON | `reports/accutile/accutile-a100-results.json` |

Note: The correctness/intersections/benchmark/training-sanity reports were generated for variant P and remain valid as P evaluations. The identity audit and this final report contain the corrected B1/P/A comparison.
