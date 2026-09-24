# C1 — Comparative Experiment Design

> **Purpose:** Define minimal prototype scope and evaluation criteria before any CUDA rebuild or benchmark. 
> This document replaces the earlier "large-scale A/B" design with a focused gate-compliant plan.
>
> **Key constraint:** C1 is NOT a novel mechanism; it is a differentiated engineering adaptation. 
> Do not claim novelty. Do not run 30K training. Isolate only what matters for renderer optimization evidence chain.
>
> **Date:** 2026-09-05

---

## 1. Minimal Prototype Scope

### What to verify (in priority order)

1. **C1 compiles and runs** without crashes in gsplat differentiable pipeline
2. **Intersection count unchanged** between baseline and C1 (same Gaussians → same tile intersections)
3. **Tile grouping correctness** — each sorted intersection maps to the correct tile (offset kernel works)
4. **Rasterizer produces output** — no black pixels, no NaN, correct alpha accumulation
5. **PSNR/SSIM equivalence** — measure numerical difference from baseline rendering
6. **Sort time reduction** — isolated CUB sort timing via CUDA events

### What to skip

- ❌ Training (forward only)
- ❌ Backward / gradient validation (Phase V4 if forward passes)
- ❌ Multi-scene comparison (single representative scene)
- ❌ IEEE-vs-linear quantization comparison (would require separate implementation)
- ❌ A100 or multi-GPU benchmark
- ❌ 4K rendering (1080p is sufficient for correctness verification)

### Minimum passing criteria for Phase 3

| Criterion | Minimum | Ideal |
|-----------|---------|-------|
| Compilation | No errors | All tests pass |
| Intersection count | Match baseline (±0) | Match baseline |
| Tile grouping | All tiles correct | Same as baseline |
| PSNR vs baseline | > 45 dB | > 55 dB |
| SSIM vs baseline | > 0.999 | > 0.9999 |
| Max pixel error | < 0.05 (in [0,1]) | < 0.01 |
| NaN/Inf pixels | 0 | 0 |

**If PSNR < 40 dB or max pixel error > 0.1:** Stop C1 immediately.
**If PSNR > 50 dB and max pixel error < 0.01:** Proceed to Phase 4 (ordering check).

---

## 2. Scene Selection

Single scene: **room-style** (indoor, 0.2m–6m depth range, many near Gaussians)

**Synthetic scene parameters:**
- Resolution: 960×540 (half 1080p)
- Gaussians: 10,000
- Depth range: 0.2m–6.0m (log-uniform)
- Tile size: 16

**Why this scene:**
- Phase V2 showed highest collision rate (67% of items in collision groups)
- Indoor scenes have dense near-range Gaussians where ordering matters most
- 960×540 reduces CUDA memory pressure for debug builds

---

## 3. CUDA Build Strategy

### Current state
- `IntersectTile.cu` patched with C1
- `IntersectTile.cu.baseline` saved as backup
- gsplat JIT build (ninja) fails due to CUDA 13.3 + MSVC + CCCL compatibility
- Build.ninja patched with `-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` and `/Zc:preprocessor`

### Build status and approach

**CUDA fast-path (attempted, blocked):**
gsplat JIT build (`_backend.py` line 177) hardcodes `-Wno-attributes` (GCC-only flag), which MSVC v19.44 rejects. Additional failures from CUDA 13.3 CCCL requiring `/Zc:preprocessor` and conflicting namespace resolution in MSVC. The JIT-generated `build.ninja` must be patched after each regeneration, creating a fragile workflow. Full CUDA rebuild is deferred per project guidance.

**Selected approach: Python-only simulation (Option B)**
Use baseline gsplat for intersection + tile generation, then replicate C1 key construction and sort in numpy:
1. Run gsplat baseline → capture `flatten_ids`, `tile_offsets`, per-tile Gaussian indices, depths, intersection counts  
2. In Python: reconstruct isect_ids from tile_id + depth → encode C1 key → numpy stable sort
3. Compare ordering: baseline vs C1 per-tile sequences
4. Reorder rasterized output within collision groups → estimate PSNR impact

**Caveats (must report):**
- ❌ Cannot measure CUDA sort time or end-to-end time
- ❌ Cannot verify CUB's actual stable-sort behavior on GPU (numpy stable sort is the best approximation)
- ✅ Can measure collision rate, inversion rate, per-tile ordering changes
- ✅ Can estimate PSNR impact from reordering within collision groups
- ✅ Phase V2 analysis script already provides collision statistics, extended with rendering estimate

---

## 4. Measurement Protocol (Option A — CUDA)

### Step 1: Warmup
```python
for _ in range(3):
    renders, alphas = rasterization(...)
```

### Step 2: Intersection count check
```python
# Patch rasterization to return isect_count
# Compare n_isects between baseline and C1 (must match)
```

### Step 3: Tile offsets check
```python
# Extract tile_offsets from intersect_offset
# Verify per-tile range_start == range_end consistency
```

### Step 4: Rendering comparison
```python
# Run baseline → get baseline_img
# Restore baseline, rebuild → run C1 → get c1_img
metrics = compute_metrics(baseline_img, c1_img)
```

### Step 5: CUDA event timing
```python
# Time CUB sort only (requires kernel-level events)
# Time forward pass (rasterization call)
# N runs with warmup → mean ± std
```

---

## 5. Fallback Protocol (Option B — Python-only)

If CUDA build fails:
- Use baseline gsplat rasterizer to generate `flatten_ids` and `tile_offsets`
- In Python, reconstruct isect_ids from tile_id + depth → simulate C1 key encoding
- Simulate stable sort (numpy lexsort) → compare ordering
- Measure collision/inversion rates (already done in Phase V2)
- Render quality estimation: use baseline renderer but reorder within collision groups
- **Cannot measure sort time or end-to-end time**

This approach already provides sufficient evidence for a Gate decision.

---

## 6. Phase Gate Criteria

| Phase | Input | Output | Gate |
|-------|-------|--------|------|
| **P1: Prior-Art Audit** | Literature search | Prior-art comparison | ✅ DONE |
| **P2: Source Audit** | gsplat source | Source-verified facts | ✅ DONE |
| **P3: Minimal Prototype** | P1+P2 | Rendered image + PSNR | ⏳ NEXT |
| **P4: Ordering Check** | P3 pass | Collision/inversion stats | PENDING |
| **P5: Performance Check** | P4 pass | CUDA event timing | PENDING |
| **P6: Gate Decision** | P1–P5 | REJECT / CONDITIONAL / CONTINUE | PENDING |

### Gate decision outcomes

| Decision | Condition | Action |
|----------|-----------|--------|
| **REJECT** | PSNR < 40 dB, or max pixel error > 0.1, or build is infeasible | Archive C1. Document in evidence chain as "not viable" |
| **CONDITIONAL CONTINUE** | PSNR > 45 dB, but sort time not measurable (Python-only) | Document as "shows promise, CUDA build required for timing" |
| **CONTINUE TO EVIDENCE CHAIN** | PSNR > 50 dB, measurable sort time reduction > 5% | Proceed to full evidence chain (backward, gradient, quality, training) |

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUDA build fails on RTX 5070 (CCCL/MSVC compat) | **High** | Fall back to Python-only simulation (Option B) |
| PSNR indistinguishable from baseline | **High** | This is expected — report as positive result |
| PSNR measurable but small | **Medium** | Acceptable if < 0.1 dB loss — most renderer studies accept this |
| Sort time reduction is measurable but end-to-end isn't | **Medium** | Record as negative finding; sort may not be the bottleneck |
| Python simulation cannot verify CUDA CUB sort behavior | **Low** | numpy.stable_sort mimics CUB stable sort behavior |
