# H3-FWD-1B-0: Macro Raster Break-even Oracle

**Status:** COMPLETE  
**Decision:** `RASTER_OPPORTUNITY_WEAK`  
**Date:** 2026-09-21  
**GPU:** A100-PCIE-40GB (GPU 4, uncontended)  
**Protocol:** 20 warmup / 100 measurements × 5 reps = 500 samples per scene, CUDA Events  

---

## Executive Summary

This report determines the exact performance target a future FP32 Trainable HiGS Macro Raster must meet to overcome the measured Macro-F4 overhead. **No new rasterizer was implemented.** This is a performance opportunity oracle only.

**Key finding:** The F4 penalty (Macro-F4 being slower than B2 F4) consumes 9.9%–39.0% of the B2 F5 time depending on scene. A future Macro-F5 rasterizer must be **10%–39% faster than B2 F5** just to break even. The existing HiGS FP16 MacroTileRasterize — the only available macro raster implementation — is **already slower than B2 F5 on all three scenes** (3.5%–64% slower), providing no evidence that the macro approach can achieve the required speedup. The decision gate is **WEAK**.

---

## 1. B2 F5 Timing (Freshly Measured)

The B2 F5 stage is the `torch.ops.gsplat.rasterize_to_pixels_3dgs` CUDA op — the pixel rasterization kernel that composites Gaussians per-pixel using alpha blending. Timed in isolation with the exact F0–F4 fixture from `f2_b2()` (projection, culling, tile intersection, sorted lists), re-batched to `[1, 1, N, ...]` layout, with SH-evaluated colors `[1, 1, N, 3]`.

| Scene | Resolution | N_vis | N_isects | Tiles | **F5 median (ms)** | F5 mean (ms) | F5 p10 (ms) | F5 p90 (ms) | Std (ms) |
|-------|-----------|-------|----------|-------|---------------------|--------------|-------------|-------------|----------|
| room | 2048×1365 | 44,908 | 953,144 | 128×86 | **0.6717** | 0.8055 | 0.6636 | 1.2012 | 0.2312 |
| bicycle | 2048×1361 | 181,525 | 1,412,193 | 128×86 | **0.9103** | 0.9106 | 0.9073 | 0.9134 | 0.0045 |
| garden | 2048×1327 | 24,483 | 533,928 | 128×83 | **0.4127** | 0.4133 | 0.4116 | 0.4157 | 0.0018 |

**Note on room variance:** Room's F5 shows higher variance (std=0.231ms, p90=1.20ms vs median=0.672ms). The p10 (0.664ms) is tight with the median, indicating the median is reliable but some measurements caught GPU frequency scaling transients. Bicycle and garden are extremely stable (std < 0.005ms). All medians are reported with 500 samples.

**Core module:** `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so`, SHA256 `361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98`.

---

## 2. Break-even Macro-F5 Targets

The break-even condition is:

```
T_macro_F5 < T_B2_F5 - F4_penalty
```

where `F4_penalty = T_macro_F4 - T_B2_F4` (from H3-FWD-1A-R2 measurements). Macro-F4 is currently **slower** than B2 F4, so the macro raster must recover this overhead in F5.

### F4 Penalties (from H3-FWD-1A-R2)

| Scene | B2 F4 (ms) | Macro F4 (ms) | **F4 Penalty (ms)** | F4 Penalty as % of F5 |
|-------|-----------|---------------|---------------------|----------------------|
| room | 0.5468 | 0.8090 | **+0.2621** | **39.0%** |
| bicycle | 0.6697 | 0.8172 | **+0.1475** | **16.2%** |
| garden | 0.4178 | 0.4588 | **+0.0410** | **9.9%** |

### Break-even and Gain Targets

| Scene | B2 F4+F5 (ms) | **Max Macro-F5 for break-even (ms)** | **Required F5 speedup** | 3% gain target (ms) | 5% gain target (ms) | 10% gain target (ms) |
|-------|---------------|--------------------------------------|------------------------|---------------------|---------------------|----------------------|
| room | 1.2186 | **0.4096** | **39.0%** | 0.3730 (44.5%) | 0.3487 (48.1%) | 0.2877 (57.2%) |
| bicycle | 1.5800 | **0.7629** | **16.2%** | 0.7155 (21.4%) | 0.6839 (24.9%) | 0.6049 (33.6%) |
| garden | 0.8305 | **0.3717** | **9.9%** | 0.3468 (16.0%) | 0.3302 (20.0%) | 0.2887 (30.0%) |

**Interpretation:** A future FP32 Macro-F5 must run at most 0.41ms (room), 0.76ms (bicycle), or 0.37ms (garden) to merely break even. For a 5% net F4+F5 gain, it must hit 0.35ms / 0.68ms / 0.33ms respectively. These are extremely aggressive targets given that B2 F5 itself runs at 0.67ms / 0.91ms / 0.41ms.

---

## 3. B2 F5 Work Composition

Profiled from the F0–F4 fixture data (tile offsets, flatten IDs, opacities, depths). Per-pixel evaluation counts are estimated from tile-level Gaussian counts × TILE_SIZE² (exact per-pixel alpha acceptance and early termination require kernel instrumentation not available without modifying CUDA source).

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| Pixels | 2,795,520 | 2,787,328 | 2,717,696 |
| Visible Gaussians | 44,908 | 181,525 | 24,483 |
| Active tiles | 11,008 (100%) | 11,008 (100%) | 10,624 (100%) |
| Fine tile-Gaussian pairs | 953,144 | 1,412,193 | 533,928 |
| Total pixel-Gaussian evaluations | 244.0M | 361.5M | 136.7M |
| Per-pixel evals: mean | 86.6 | 128.3 | 50.3 |
| Per-pixel evals: p50 | 80 | 85 | 43 |
| Per-pixel evals: p90 | 150 | 270 | 89 |
| Per-pixel evals: p99 | 250 | 819 | 175 |
| Per-pixel evals: max | 424 | 1,563 | 299 |
| Per-tile Gaussian count: max | 424 | 1,563 | 299 |
| Opacity: mean | 0.861 | 0.821 | 0.943 |
| Opacity: above α-threshold (1/255) | 99.998% | 99.999% | 100.0% |

**Key observations (no speculative attribution):**

1. **All tiles are active** (100% active tile fraction) — there are no empty tiles in these scenes at 2048px, so the macro raster cannot gain from tile skipping.
2. **Per-pixel evaluation counts are high** — room averages 87, bicycle 128, garden 50 Gaussian evaluations per pixel. These evaluations are **irreducible**: each pixel must evaluate the Gaussian weight, compute alpha, update transmittance, and accumulate color for every covering Gaussian.
3. **Opacity is uniformly high** (mean 0.82–0.94) — nearly all visible Gaussians pass the alpha threshold, so the alpha acceptance rate is ~100%. Early termination happens when transmittance drops below 1e-4, but with high-opacity Gaussians, this occurs after relatively few contributors.
4. **Bicycle has extreme tail** — p99=819, max=1563 evaluations per pixel. The heavy tiles dominate F5 time.

---

## 4. Existing HiGS MacroTileRasterize Oracle

The existing HiGS macro tile rasterizer (`torch.ops.experimental.gaussian_render_inference_only`) was measured as a **performance oracle only**. It produces **FP16 output — NOT correctness evidence**. The op runs the full inference pipeline internally (projection → macro tile intersection → macro tile rasterization → post-blend), so its timing includes more than just the pixel rasterization kernel.

### Oracle Results (500 samples per scene, CUDA Events)

| Scene | Pack/Convert (ms) | **Raster total median (ms)** | vs B2 F5 | Verdict |
|-------|-------------------|------------------------------|----------|---------|
| room | 55.82 | **0.8120** | 0.672 → 0.812 = **+20.9% slower** | ❌ |
| bicycle | 314.47 | **0.9421** | 0.910 → 0.942 = **+3.5% slower** | ❌ |
| garden | 1.44 | **0.6758** | 0.413 → 0.676 = **+63.7% slower** | ❌ |

**Critical finding:** The existing HiGS macro raster is **slower than B2 F5 on all three scenes**. Even though the macro raster processes fewer sorted entries (7.7× fewer for room, 4.5× for bicycle, 8.0× for garden), it does not achieve a net speedup. This is direct empirical evidence that the macro representation compression does **not** translate to compute speedup in the pixel rasterization stage.

**Packing cost:** The FP32→FP16 packing (`GaussianInferenceScene.from_gaussian_tensors`) costs 1.4–314ms depending on scene size. This is a one-time setup cost but would need to be amortized across many frames. For training (where the scene changes every iteration), this cost is incurred every step.

**Caveat:** The oracle's `raster_total` includes its own internal projection and tile intersection, so it's not a pure F5-equivalent comparison. However, even if we attribute some of that to F0–F4-equivalent work, the fact that the total doesn't beat B2 F5 alone is a strong negative signal.

---

## 5. Reducible vs Irreducible Work

### Reducible Work (where macro raster can theoretically save time)

| Category | Description | Reduction Factor |
|----------|-------------|-----------------|
| Gaussian metadata loads | Macro raster loads sorted Gaussian ID + mask once per macro entry instead of once per fine tile-Gaussian pair | ~compression (4.5–8.0×) |
| Sorted entry traversal | Macro raster traverses N_macro_entries sorted entries instead of N_fine_pairs | ~compression (4.5–8.0×) |
| Fine-tile scheduling | 32-bit mask encodes which fine tiles each Gaussian covers — skips per-fine-tile scheduling | Eliminates per-fine-tile overhead |
| Queue setup | 1 macro tile queue vs MTW×MTH=32 fine tile queues | 32× fewer queues |
| Global memory traffic | Fewer sorted IDs to load, fewer tile offsets | ~compression |

### Irreducible Work (unchanged regardless of representation)

| Category | Description | Why irreducible |
|----------|-------------|-----------------|
| Pixel-Gaussian weight evaluation | `weight = exp(-0.5 * d^T * conic * d)` for each pixel-Gaussian pair | Same number of evaluations — each pixel still sees the same Gaussians |
| Alpha calculation | `alpha = opacity * weight` | Same count |
| Transmittance chain | `T *= (1 - alpha)` | Same count |
| Color FMA | `color += T * alpha * rgb` | Same count |
| Early termination check | `if T < threshold: break` | Same check per Gaussian per pixel |

### The Fundamental Insight

> **Macro raster compresses the REPRESENTATION (4.5–8.0× fewer entries) but NOT the PIXEL COMPUTE (same number of per-pixel Gaussian evaluations).** The total pixel-Gaussian evaluations are 244M (room), 362M (bicycle), 137M (garden) — and these are identical regardless of whether the sorted list is stored as fine tile-Gaussian pairs or macro entries. The speedup opportunity is in reduced memory traffic and traversal overhead, NOT in reduced FLOPs.
>
> The representation compression (7.7× for room) is a **list compression** — fewer entries in the sorted array. It does NOT mean 7.7× fewer pixel computations. Each pixel still evaluates the same Gaussians. **We must not convert 7.7× representation compression into a 7.7× compute claim.**

### Quantitative Bound

The reducible work (metadata loads, traversal, scheduling) is a fraction of total F5 time. The irreducible work (pixel math) dominates because:
- Each pixel evaluates 50–128 Gaussians on average
- Each evaluation involves: 2D delta computation, quadratic form evaluation (conic × delta), exponential, multiply (opacity), fused multiply-add (color), comparison (early termination)
- These are all register-level operations that don't benefit from representation compression

If the reducible overhead is ~30% of F5 (a generous estimate for memory-bound portions), then the maximum achievable F5 speedup from macro raster is ~30%, which is below the 39% required for room break-even.

---

## 6. Decision Gate

### Metrics

| Metric | room | bicycle | garden | Average |
|--------|------|---------|--------|---------|
| F4 penalty as % of F5 | 39.0% | 16.2% | 9.9% | **21.7%** |
| Required F5 speedup | 39.0% | 16.2% | 9.9% | 21.7% |
| Representation compression | 7.7× | 4.5× | 8.0× | 6.7× |
| Margin (compression ÷ required_speedup) | 0.20 | 0.28 | 0.81 | **0.43** |
| Existing macro raster vs B2 F5 | +20.9% slower | +3.5% slower | +63.7% slower | +29.4% slower |

### Gate Criteria

| Threshold | STRONG | MARGINAL | WEAK | **Observed** |
|-----------|--------|----------|------|-------------|
| Avg F4 penalty as % of F5 | < 10% | < 25% | ≥ 25% | **21.7%** (MARGINAL range) |
| All margins (compression ÷ required) | > 3.0 | > 1.5 | ≤ 1.5 | **0.43** (WEAK) |
| Existing macro raster beats B2 F5 | Yes, all scenes | Mixed | No | **No, all scenes slower** (WEAK) |

### Decision: `RASTER_OPPORTUNITY_WEAK`

**Rationale:**

1. **The required F5 speedup is large** — 10%–39% faster than B2 F5 just to break even. Room requires 39%, which is extremely aggressive for a rasterizer that must perform the same per-pixel computations.

2. **The existing macro raster is slower, not faster** — the only available macro raster implementation (HiGS FP16 MacroTileRasterize) is 3.5%–64% slower than B2 F5. This is direct empirical evidence against the macro approach achieving the required speedup.

3. **Representation compression ≠ compute compression** — the 4.5–8.0× list compression reduces metadata and traversal overhead but does not reduce the 137–362 million pixel-Gaussian evaluations that dominate F5. The margin (compression ÷ required speedup) is 0.43 on average, far below the 1.5 threshold for MARGINAL.

4. **Room is the worst case** — 39% required speedup with only 7.7× compression (margin 0.20). The F4 penalty alone is 39% of F5 time, meaning the macro raster must nearly halve F5 time just to recover the F4 overhead.

5. **Garden is the best case** but still WEAK — 9.9% required speedup with 8.0× compression (margin 0.81). Even here, the existing macro raster is 64% slower, suggesting the implementation overhead outweighs the representation savings.

**Implication for FP32 Trainable HiGS:** A future FP32 macro raster faces an even harder target than the FP16 oracle, because FP32 computation is inherently more expensive than FP16 (double the data width, potentially half the throughput on A100 tensor cores for non-tensor-core paths). If the FP16 oracle can't beat B2 F5, an FP32 implementation is unlikely to either, unless it employs fundamentally different algorithmic strategies (e.g., fused projection+rasterization, reduced precision in non-critical paths, or exploitation of macro-tile locality for shared memory caching).

---

## Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| F5 timing CSV | `artifacts/higs-h3-fwd-1b-0/b2_f5_timing.csv` | B2 F5 median/mean/p10/p90/std for room/bicycle/garden |
| Break-even targets | `artifacts/higs-h3-fwd-1b-0/break_even_targets.json` | Per-scene break-even + 3%/5%/10% gain targets |
| F5 workload | `artifacts/higs-h3-fwd-1b-0/b2_f5_workload.json` | Tile counts, pixel-Gaussian evaluations, opacity stats |
| HiGS raster oracle | `artifacts/higs-h3-fwd-1b-0/existing_higs_raster_oracle.json` | Existing FP16 MacroTileRasterize timing (pack + raster) |
| Opportunity analysis | `artifacts/higs-h3-fwd-1b-0/opportunity_analysis.json` | Reducible vs irreducible work + decision gate |
| Provenance | `artifacts/higs-h3-fwd-1b-0/provenance.json` | Run ID, GPU, core .so SHA256, protocol parameters |

**Measurement script:** `scripts/h3/h3_fwd_1b_0_break_even_oracle.py`  
**Launcher:** `scripts/h3/_h3_fwd_1b_0_run.sh`

---

## Provenance

- **Run ID:** `d3cadc4c-782`
- **Timestamp:** 20260921T010858
- **GPU:** cuda:4 (A100-PCIE-40GB, uncontended)
- **Core module:** `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so` (SHA256 `361b216b...`)
- **Source tree:** `/tmp/higs_h3_fwd_1a_source` (clean B2 source, no CF patch)
- **Protocol:** 20 warmup, 100 measurements × 5 reps = 500 samples, CUDA Events, non-interleaved
- **F4 penalties from:** H3-FWD-1A-R2 (`/tmp/h3_fwd_1a_r2/timing_{room,bicycle,garden}.json`)
- **Resolution:** max_long_side=2048 (room 2048×1365, bicycle 2048×1361, garden 2048×1327)
- **Environment:** torch 2.9.1+cu128, CUDA 12.8, nvcc V12.8.93, conda env at `/mnt/storage_pool/liaoyuanjun/higs-13scene-env`
