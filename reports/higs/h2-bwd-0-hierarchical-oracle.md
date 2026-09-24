# H2-BWD-0 — HiGS Hierarchical Backward Oracle

**Status:** COMPLETE  
**Date:** 2026-09-21  
**Question:** Does the HiGS macro-tile hierarchy provide enough real gradient-aggregation and execution opportunity to justify implementing a hierarchical backward kernel?

---

## 0. Frozen source identity

```text
Base:       77ab983ffe43420b2131669cb35776b883ca4c3c
B2 patch:   74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c
```

Fixture: room/cam0, tile_size=16, 2048×1365, N_visible=44,908, n_isects=953,144

Macro-tile geometry (from `IntersectMTConfig.h`):
```text
FUSED_MACRO_TILE_WIDTH  = 8
FUSED_MACRO_TILE_HEIGHT = 4
fine_tiles_per_macro    = 32
FUSED_GAUSS_BATCH_SIZE  = 1024
```

---

## 1. Macro-tile pair oracle

| Metric | Value |
|--------|-------|
| N_tile_gaussian_pairs | 953,144 |
| N_macro_gaussian_pairs | 124,017 |
| **R_macro** | **7.69** |

R_macro = 953,144 / 124,017 = 7.69. Each (macro-tile, Gaussian) pair aggregates 7.69 fine-tile pairs on average. This **strongly passes** the R_macro ≥ 2.0 gate.

---

## 2. Distribution of fine-tile reuse inside macro tiles

| Statistic | Value |
|-----------|-------|
| mean | 7.69 |
| p50 | 5 |
| p75 | 10 |
| p90 | 18 |
| p95 | 26 |
| p99 | 32 |
| max | 32 |

| Fine tiles touched | Count | % |
|--------------------|------:|------:|
| 1 | 13,387 | 10.8% |
| 2 | 18,324 | 14.8% |
| 3–4 | 25,976 | 20.9% |
| 5–8 | 30,486 | 24.6% |
| 9–16 | 21,727 | 17.5% |
| 17–24 | 7,037 | 5.7% |
| 25–32 | 7,080 | 5.7% |

**55% of (macro, Gaussian) pairs touch ≥ 5 fine tiles.** 5.7% touch all 32 fine tiles — these are large Gaussians spanning entire macro tiles. The aggregation opportunity is real and substantial.

---

## 3. Macro-tile workload distribution

| Metric | mean | p50 | p90 | p95 | p99 | max |
|--------|-----:|----:|----:|----:|----:|----:|
| N_unique_gaussians | 352 | 307 | 640 | 779 | 1,027 | 1,319 |
| N_fine_tile_gaussian_pairs | 2,708 | 2,521 | 4,482 | 5,354 | 6,395 | 9,426 |
| N_active_fine_tiles | 31.3 | 32 | 32 | 32 | 32 | 32 |
| N_1024_batches | 1.01 | 1 | 1 | 1 | 1.49 | 2 |

**348 of 352 macro tiles fit in a single 1024-Gaussian batch.** Only 4 macro tiles need 2 batches. The p99 workload (1,027 Gaussians) is implementable with bounded 1024-Gaussian batching. This **passes** the implementability gate.

---

## 4. 32-Gaussian mini-batch feasibility

| Component | Bytes |
|-----------|------:|
| Gradient state (32 × 9 × FP32) | 1,152 |
| Metadata (IDs + masks + scratch) | 180 |
| **Total per mini-batch** | **1,332** |

A100 SM80: 164KB shared memory per SM, 65,536 registers per SM.

| Constraint | Limit | Actual | Blocks/SM |
|------------|------:|--------|-----------:|
| Registers (64/thread × 128) | 8 | 8,192 | 8 |
| Shared memory (1,332 bytes) | 126 | 1,332 | 126 |
| Threads (128/block) | 16 | 128 | 16 |
| **Min** | | | **8** |

**Feasibility: SAFE.** 8 blocks/SM estimated occupancy. Shared memory is negligible (1.3KB out of 164KB).

---

## 5. Current cross-warp scatter multiplicity

The kernel `higs_blend_bwd_px_kernel<3,2>` uses block=(16,8,1)=128 threads=4 warps. For each intersection, up to 4 warp leaders do atomicAdd (one per warp with valid pixels).

| Warp coverage | Count | % |
|---------------|------:|------:|
| 0 warps | 1,334 | 0.1% |
| 1 warp | 14,548 | 1.5% |
| 2 warps | 22,620 | 2.4% |
| 3 warps | 28,114 | 2.9% |
| **4 warps** | **886,528** | **93.0%** |

| Metric | Value |
|--------|-------|
| N_current_scatter_groups | 3,690,242 |
| **R_crosswarp** | **3.87** |

**93% of intersections scatter through all 4 warps.** The mean warp coverage is 3.87, near the theoretical maximum of 4.0.

---

## 6. Combined structural reduction opportunity

| Metric | Value |
|--------|-------|
| R_macro | 7.69 |
| R_crosswarp | 3.87 |
| **R_total** | **29.76** |

R_total = 3,690,242 / 124,017 = 29.76. This is the gradient-scatter compression ratio: the current kernel produces 29.76× more atomic operations than a macro-tile + cross-warp hierarchical approach would. **This is NOT expected speedup** — it is only the reduction in atomic operation count.

---

## 7. Atomic-free upper-bound experiment (ATOMIC_FREE_ORACLE)

**Method:** Two CUDA microbenchmark kernels with identical compute (Gaussian weight eval, VJP math, warpSum reduction, early-exit, same grid/block/batch structure) but different scatter:
- **Variant A (atomic):** atomicAdd to contended global gradient buffers (same as real kernel)
- **Variant B (atomic-free):** uncontended scratch write (same compute, no atomic)

Measured with CUDA Events, 20 warmup / 100 measure, GPU 3 (uncontended A100-PCIE-40GB).

| Metric | Value |
|--------|-------|
| T_blend_current (atomic) | 2.091 ms |
| T_blend_atomic_free | 1.965 ms |
| **O_atomic_max** | **0.126 ms** |
| O_atomic_max / T_blend | **6.0%** |
| O_atomic_max / T_F+B (H2-1 ref: 4.512ms) | **2.8%** |

**The atomic scatter accounts for only 6.0% of blend_bwd time.** The remaining 94% is compute: Gaussian weight evaluation (`expf`, multiply), VJP math (9 gradient components per pixel per intersection), and warp reductions. The kernel is **compute-bound, not atomic-bound**.

---

## 8. Macro-hierarchy transformation cost estimate

| Component | Estimated cost |
|-----------|---------------:|
| A. Reuse fine-tile masks (from isect_offsets) | ~0 µs |
| B. WarpBitTranspose | ~0 µs (register-level) |
| C. Mini-batch shared gradient reduction | ~20 µs |
| D. Block synchronization | ~5 µs |
| E. Macro-local final global scatter | ~30 µs |
| F. Compact batch state write (28.3 MB) | ~25 µs |
| **Total** | **~80 µs** |

Transformation cost / atomic-free opportunity = 80 / 126 = **63%**. This **exceeds** the 50% gate.

---

## 9. Batch count / state expansion

| Metric | Value |
|--------|-------|
| N_active_pixel_batches | 2,828,288 |
| Prefix/adjoint state (2×FP32) | 22.63 MB |
| last_local_id (uint16) | 5.66 MB |
| **Total batch state** | **28.28 MB** |

28.3 MB is acceptable. For bicycle/cam0 (larger scene): 41.5 MB, also acceptable.

---

## 10. Batch VJP correctness derivation

**Proposed formulas:**
```
dL/dC_b = P_b * g_rgb
dL/dT_b = P_b * lambda_{b+1}
lambda_b = g_rgb · C_b + T_b * lambda_{b+1}
lambda_B = -g_alpha  (terminal adjoint from alpha loss)
```

**Validation:** 10 random tests (N=20–100, n_batches=2–8) comparing autograd on batched forward vs single forward.

| Metric | Value |
|--------|-------|
| Tests passed | 10/10 |
| Max forward error | 3.33×10⁻¹⁶ |
| Max gradient error | 6.78×10⁻²¹ |
| Min cosine similarity | 1.0 |

**The batch VJP is EXACT.** The batch composition (C = Σ P_b·C_b, T = Π T_b) is mathematically identical to standard front-to-back rendering, so all per-Gaussian gradients match exactly.

---

## 11. Gate decision

| Gate | Criterion | Value | Pass? |
|------|-----------|-------|:-----:|
| 1 | R_macro ≥ 2.0 | 7.69 | ✅ |
| 2a | Atomic-free ≥ 10% of blend_bwd | 6.0% | ❌ |
| 2b | Atomic-free ≥ 5% of F+B | 2.8% | ❌ |
| 3 | p99 workload implementable (1024 batching) | p99=1027, max 2 batches | ✅ |
| 4 | Transformation cost ≤ 50% of opportunity | 63% | ❌ |
| 5 | Batch VJP exact | 10/10 pass, cosine=1.0 | ✅ |
| 6 | State memory acceptable | 28.3 MB | ✅ |

### Verdict: **KEEP_CANDIDATE**

**Reason:** The macro-tile hierarchy provides massive structural compression (R_macro=7.69, R_total=29.76) and the batch VJP is exact. However, two gates fail:

1. **Gate 2 (atomic oracle):** The atomic-free experiment proves that global scatter is only 6.0% of blend_bwd and 2.8% of F+B. The kernel is compute-bound, not atomic-bound. The structural compression targets a small cost.

2. **Gate 4 (transformation cost):** The hierarchical transformation overhead (80 µs) consumes 63% of the 126 µs atomic-free opportunity, leaving only 37% (46 µs) as net benefit — just 1.0% of F+B.

The mechanism is real but the opportunity it targets is too small. NOT promoted to CUDA implementation.

---

## 12. Comparison: C11-R vs H2-BWD-MT

| Candidate | Scatter compression | Structural change | Atomic-free upper bound | Transformation cost | Net opportunity | % blend_bwd | % F+B | Risk | Verdict |
|-----------|--------------------:|------------------:|------------------------:|--------------------:|----------------:|------------:|------:|------|---------|
| **C11-R** (cross-warp reduction) | 3.87× | low | 126 µs | ~30 µs | ~96 µs | 4.6% | 2.1% | low | KEEP_CANDIDATE |
| **H2-BWD-MT** (macro-tile hierarchical) | 29.76× | medium/high | 126 µs | ~80 µs | ~46 µs | 2.2% | 1.0% | medium/high | KEEP_CANDIDATE |

Both candidates share the same atomic-free upper bound (126 µs) because both target the same atomic scatter cost. C11-R achieves 3.87× compression with minimal structural change (add block-level shared memory accumulation + block sync within existing fine-tile blocks). H2-BWD-MT achieves 29.76× compression but with much higher structural complexity and transformation cost.

**C11-R is the better candidate** because:
- Same upper bound (126 µs) with lower transformation cost (30 µs vs 80 µs)
- Higher net opportunity (96 µs vs 46 µs)
- Lower risk (no new state buffers, no batch VJP, no macro-tile mapping)
- Simpler implementation (modify existing kernel, not new kernel)

However, both are KEEP_CANDIDATE because the atomic-free oracle shows the opportunity is only 6% of blend_bwd.

---

## 13. Optional second fixture: bicycle/cam0

| Metric | room/cam0 | bicycle/cam0 |
|--------|----------:|-------------:|
| N_visible | 44,908 | 181,525 |
| n_isects | 953,144 | 1,412,189 |
| R_macro | 7.69 | 4.53 |
| R_crosswarp | 3.87 | 3.56 |
| R_total | 29.76 | 16.12 |
| p99 unique Gaussians | 1,027 | 4,509 |
| max 1024-batches | 2 | 6 |
| Batch state | 28.3 MB | 41.5 MB |

Bicycle shows the same pattern: strong R_macro (4.53), high warp coverage (3.56), all macro tiles active. Room is not structurally unusual — it has higher R_macro due to larger Gaussians relative to tile size, but both scenes confirm the mechanism is real. Bicycle's higher Gaussian count per macro (p99=4509) requires up to 6 batches, still implementable but more complex.

---

## 14. Deliverables

```text
reports/higs/h2-bwd-0-hierarchical-oracle.md           (this file)

artifacts/higs-h2-bwd-0/
    macro_pair_stats.json                               (section 1)
    macro_pair_histogram.csv                            (section 2)
    macro_workload_stats.csv                            (section 3)
    scatter_multiplicity.json                           (sections 5-6)
    atomic_free_oracle.json                             (section 7)
    batch_state_memory.json                             (sections 8-9)
    batch_vjp_validation.json                           (section 10)
    candidate_comparison.csv                            (section 12)
    analysis.json                                       (complete analysis)
    room_cam0_structural.json                           (raw structural data)
    bicycle_cam0_structural.json                        (raw structural data)
    room_cam0_histogram.csv                             (raw histogram)
    room_cam0_macro_workload.csv                        (raw per-macro data)
    bicycle_cam0_histogram.csv                          (raw histogram)
    bicycle_cam0_macro_workload.csv                     (raw per-macro data)
    room_cam0_atomic_oracle.json                        (raw oracle timing)

scripts/h2/
    h2_bwd_0_structural.py                              (structural analysis)
    h2_bwd_0_atomic_oracle.py                           (atomic-free oracle)
    h2_bwd_0_vjp_validation.py                          (batch VJP validation)
    _h2_bwd_0_run.sh                                    (mx launcher)
```

---

## 15. Final answer

1. **N_tile_gaussian_pairs** = 953,144
2. **N_macro_gaussian_pairs** = 124,017
3. **R_macro** = 7.69
4. **Fine-tile reuse distribution**: mean=7.69, p50=5, p75=10, p90=18, p95=26, p99=32, max=32. 55% touch ≥5 fine tiles.
5. **Macro unique-Gaussian distribution**: mean=352, p50=307, p90=640, p95=779, p99=1027, max=1319
6. **1024-batch distribution**: 348 macros with 1 batch, 4 macros with 2 batches. p99=1.49, max=2.
7. **N_current_scatter_groups** = 3,690,242
8. **R_crosswarp** = 3.87 (93% of intersections hit all 4 warps)
9. **R_total** = 29.76
10. **Current blend backward time** = 2.091 ms (microbenchmark, same structure as real kernel)
11. **Atomic-free oracle time** = 1.965 ms
12. **Atomic-removal upper bound** = 0.126 ms (6.0% of blend_bwd, 2.8% of production F+B)
13. **Hierarchical temporary-state memory** = 28.28 MB (room/cam0), 41.47 MB (bicycle/cam0)
14. **Batch-VJP correctness result** = EXACT (10/10 tests pass, cosine=1.0, max error 6.78×10⁻²¹)
15. **C11-R oracle**: scatter compression 3.87×, net opportunity ~96 µs (4.6% blend_bwd, 2.1% F+B), low risk
16. **H2-BWD-MT oracle**: scatter compression 29.76×, net opportunity ~46 µs (2.2% blend_bwd, 1.0% F+B), medium/high risk
17. **Gate decision: KEEP_CANDIDATE**
18. **Exact reason**: R_macro=7.69 (strongly passes), batch VJP exact, workload implementable, state memory acceptable. BUT atomic-free oracle shows only 6.0% of blend_bwd is atomic overhead (below 10% gate) and 2.8% of F+B (below 5% gate) — the kernel is compute-bound, not atomic-bound. Transformation cost (63% of opportunity) exceeds 50% gate. The mechanism is real (29.76× scatter compression) but targets a small cost. Two gates fail (gate 2 and gate 4).
19. **Report/artifact paths**: `reports/higs/h2-bwd-0-hierarchical-oracle.md` + `artifacts/higs-h2-bwd-0/` (9 required files + raw data)
