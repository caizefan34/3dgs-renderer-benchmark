# Phase 8D — Forward Superlinear Scaling Hypothesis Matrix

**Date:** 2026-08-22 (updated with Phase 8E per-kernel timing)  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU  
**Status:** ✅ COMPLETED (with per-kernel CUDA timing from Phase 8E)

---

## Legend

| Status | Meaning |
|:-------|:--------|
| **SUPPORTED** | Evidence is consistent with hypothesis |
| **PARTIAL** | Evidence partially supports, but not fully explanatory |
| **FALSIFIED** | Evidence contradicts hypothesis |
| **INCONCLUSIVE** | Cannot determine from available evidence |
| **SUPERSEDED** | Hypothesis became irrelevant after new evidence (Phase 8E per-kernel timing) |
| **BLOCKED** | Required measurement tool unavailable (Nsight Compute on WDDM) |

---

## Hypothesis Matrix

### H1: Launch / Organization Overhead

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | tile32 has 4× fewer block launches (2040 vs 8160). If launch overhead is the dominant factor, the speedup should be ~4× and should NOT increase with workload. |
| **Required metric** | Block launch time, kernel launch latency measurement |
| **Experiment** | Source-code analysis shows 8160 vs 2040 blocks, identical kernel binary. |
| **Observed result** | tile16/tile32 ratio varies from 3.3× to 25.5× with workload — far beyond 4×. The ratio increases as workload grows. |
| **Status** | **FALSIFIED** — Launch overhead is constant per block, cannot explain workload-dependent ratio growth. |

### H2: Work Granularity

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | tile16 has 4× more blocks each with 1/4 the threads. If finer granularity causes inefficiency (e.g., warp divergence), the effect should be constant or decrease with more work. |
| **Required metric** | Warp occupancy, divergence ratio per kernel |
| **Experiment** | Both tile sizes have 100% tile occupancy (every Gaussian in every tile). No empty tiles to balance. |
| **Observed result** | 3.3×–25.5× ratio, increasing with workload. NOT constant. |
| **Status** | **FALSIFIED** — Granularity is fixed by tile geometry; ratio varies with workload. |

### H3: Memory Reuse / Arithmetic Intensity

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | tile32 loads 4× more Gaussian data per shared memory load (1024 vs 256 per batch). Each Gaussian is reused across 4× more pixel evaluations, increasing arithmetic intensity. |
| **Required metric** | Shared memory loads per Gaussian, FLOPs per byte (arithmetic intensity) |
| **Experiment** | Source-level analysis confirms: batch_size = tile_size². tile32 processes 4× more Gaussians per batch, so each global→shared load is reused 4× more. |
| **Observed result** | Consistent — tile32's larger batch enables better amortization of memory loads. However, this should be a constant factor (~4×), not a growing one (~25×). |
| **Status** | **SUPPORTED (partial)** — Explains the baseline 4×-ish advantage. Does not explain superlinear growth. |

### H4: Occupancy / Latency Hiding

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | tile16 has block_size=256, allowing more blocks/SM (higher occupancy). If occupancy differences matter, the higher-occupancy config (tile16) should be faster, or the difference should decrease when memory-bound. |
| **Required metric** | SM occupancy, achieved occupancy, warp stall reasons |
| **Experiment** | RTX 5070 has ~40 SMs with 64K registers and 128KB shared memory each. tile16: 204 blocks/SM possible? No — limited by register pressure. |
| **Observed result** | tile16 is slower despite potentially higher occupancy. The gap grows with workload, not shrinks. |
| **Status** | **FALSIFIED** — Higher occupancy (tile16) does not translate to better performance; the gap is inverse of occupancy prediction. |

### H5: Scene / Workload Interaction

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | Real trained Gaussians have structured spatial distribution (clustered), causing more overlapping footprints. Synthetic random Gs don't cluster → no superlinearity. |
| **Required metric** | Screen-space Gaussian density distribution, overlap statistics |
| **Experiment** | Phase 7C synthetic (random) shows ~1× ratio. Phase 8B real room shows 3–25× ratio. The definitive difference is tile occupancy: synthetic ~1 Gs/tile, real ~21K Gs/tile. |
| **Observed result** | Structured scene with 100% tile occupancy → superlinear tile16 scaling. Random scene with ~0% tile occupancy → near-linear scaling. |
| **Status** | **SUPPORTED** — The effect is definitively scene-dependent. Clustered Gaussians with large footprints are required. |

### H6: Tile/Gaussian Intersection Structure

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | tile16 generates 4× more tile-Gaussian pairs. If intersection processing is the bottleneck, the ratio should be exactly 4×. |
| **Required metric** | n_isects ratio |
| **Experiment** | All checkpoints show exactly 4.00× intersection ratio (145M→176M vs 36M→44M). |
| **Observed result** | Intersection ratio is exactly 4× (geometric constant). Forward runtime ratio is 3–25×. |
| **Status** | **SUPPORTED (partial)** — Explains the 4× baseline. Does not explain why ratio exceeds 4×. |

### H7: Rasterization Kernel Batch Overhead ⚠️ **FALSIFIED by Phase 8E**

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | tile16 requires ~4× more batch iterations per tile (batch_size=256 vs 1024). Each batch has synchronization and cooperative loading overhead that is fixed per batch, not per-Gaussian. As Gs/tile grows, the number of batches grows (1.21×), but the efficiency PER batch degrades due to increased thread divergence. |
| **Required metric** | **Direct measurement: rasterization kernel wall time** |
| **Experiment** | **Phase 8E** — `torch.cuda.Event` wrapped around `rasterize_to_pixels()` call. N_REPEAT=10, WARMUP=3. |
| **Observed result** | **Rasterization takes < 1% of total forward time** (0.4–0.7ms out of 120–200ms for tile16). The 15.5× batch launch disparity in a sub-millisecond kernel **cannot possibly explain** a ~200ms total time difference. |
| **Status** | **FALSIFIED** — The earlier "SUPPORTED (strong)" status was a methodological error: a large relative workload disparity (15.5× batch count) in an absolutely negligible kernel (0.5ms) was misinterpreted as the root cause. Only per-kernel CUDA timing could resolve this. |

### H8: SIMT / Warp Divergence (NEW)

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | For tile16 (block=256, 8 warps), divergent `done` flags within a warp cause serialized execution. For tile32 (block=1024, 32 warps), same issue but diluted by 4× more threads per block + 4× fewer blocks. The interaction: with 21K Gs/tile and varying pixel-level occlusion, warps diverge significantly → reduced throughput. |
| **Required metric** | Warp execution efficiency, warp divergence ratio, `__syncthreads_count` overhead measurement |
| **Experiment** | Cannot measure without Nsight Compute. Source code shows divergent paths: `if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;` and `if (next_T <= 1e-4f) { done = true; break; }`. |
| **Observed result** | Inference only — no direct measurement. |
| **Status** | **INCONCLUSIVE** — Plausible mechanism, requires Nsight Compute profiling to confirm. |

### H9: Radix Sort as Dominant Cost ✅ **CONFIRMED as Primary Mechanism**

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | The CUB radix sort on 175M elements (tile16) vs 44M elements (tile32) should be the dominant forward cost. |
| **Required metric** | CUB radix sort wall time, isolated from other pipeline stages |
| **Experiment** | **Phase 8E** — Sort isolation via `isect_tiles(sort=False)` vs `isect_tiles(sort=True)` with median-difference estimation. 5s GPU cooldown between phases. |
| **Observed result** | **Sort accounts for 60-80% of total forward time.** Sort time ratio is approximately linear: 3.2-4.0× across all checkpoints, closely matching the 4× input size ratio. |
| **Status** | **SUPPORTED** — Radix sort is the primary forward cost. However, the sort ratio is approximately linear (4× input → ~4× time), NOT superlinear. The forward disparity is ~4× because the dominant stage scales ~4× with input size. |

### H10: Pipeline Backpressure (NEW)

| Aspect | Detail |
|:-------|:-------|
| **Prediction** | The sequential forward pipeline amplifies any slowdown: if projection is slightly slower (larger output → larger isect → larger sort → larger offset → more tile blocks), the pipeline's total time is the SUM of these stages. The cumulative 4× differences compound. |
| **Required metric** | Per-stage wall time for all 7 forward kernel groups |
| **Experiment** | Source-level analysis: only sorting/offset/rasterization have 4× work difference. Projection and SH eval are tile_size-independent. |
| **Observed result** | Without per-kernel timing, cannot quantify each stage's contribution. The cumulative sum of 4×-sized stages and 1×-sized stages should give a total ratio between 1× and 4× for those stages. The observed 25× ratio exceeds what pipeline backpressure alone can explain (bounded by the slowest stage). |
| **Status** | **FALSIFIED** — Cumulative pipeline effect is bounded by the sum of stage ratios; cannot explain 25× when most stages are 1×–4×. |

---

## Winner: Phase 8E Revised Explanation

**⚠️ Phase 8E per-kernel CUDA timing fundamentally revised the mechanism understanding.**

### H7 Falsified: Rasterization is Negligible

Phase 8D identified H7 (Rasterization Kernel Batch Overhead) as the primary mechanism, based on source-level analysis showing 15.5× batch launch volume. **Phase 8E directly measures the rasterization kernel at < 1% of total forward time** (0.4–0.7ms out of 120–200ms). A 15.5× disparity in a 0.5ms kernel cannot explain a 200ms total difference. H7 is falsified.

### H9 Confirmed: Radix Sort is the Primary Cost

**CUB radix sort accounts for 60-80% of total forward time.** The sort input size scales geometrically with tile_size: tile16 produces exactly 4× more tile-Gaussian intersections than tile32 (e.g., 176M vs 44M at iter30000). This drives a ~4× sort time difference (3.2–4.0×). The sort ratio is approximately linear — no superlinear sort effect.

### Phase 8C/8D Ratio Correction

Phase 8C claimed 15-25× forward disparity. Phase 8E shows the true per-kernel ratio is 4-6×. The earlier larger ratios were likely inflated by autograd/metadata overhead that scales with the 4× larger intersection count in the monolithic `rasterization()` call. Phase 8E avoids this by directly calling CUDA wrapper functions.

### Corrected Mechanism Chain

```
Real Gaussians (large footprints)
  → 100% tile occupancy for both tile sizes
    → tile16 has 4× more tiles (= geometric ratio)
      → tile16 generates 4× more tile-Gaussian intersections (n_isects)
        → CUB radix sort input is 4× larger
          → Sort time is ~4× longer (dominant forward cost)
            → tile16 forward total is ~4× tile32 forward total
```

The modest ratio growth from ~4× to ~6× with training is a small second-order effect, likely from one of: (a) CPU cumsum overhead variation, (b) CUB sort memory pressure for 175M elements, (c) thermal artifacts. This requires isolated sort micro-benchmarks.

**H3: Memory Reuse** is irrelevant to the dominant mechanism (sort is not memory-reuse limited).

**H5: Scene/Workload Interaction** is the precondition — without real Gaussians with large screen-space footprints, the 4× intersection ratio would not exist.

---

## Competition Hypothesis Assessment

| Hypothesis | Status | Explanation Power | Evidence |
|:-----------|:------:|:-----------------:|:--------|
| H1 Launch Overhead | FALSIFIED | None (constant) | Ratio varies with workload |
| H2 Work Granularity | FALSIFIED | None (constant) | Ratio varies with workload |
| H3 Memory Reuse | SUPPORTED (partial) | Baseline 4× | Source code analysis |
| H4 Occupancy | FALSIFIED | None (inverse) | tile16 higher occ, slower |
| H5 Scene Interaction | SUPPORTED | Precondition | Synthetic vs real comparison |
| H6 Intersection Structure | SUPPORTED (partial) | Baseline 4× | Fixed 4.00× intersection ratio |
| **H7** Batch Overhead | **FALSIFIED** (Phase 8E) | **None** | Rasterization < 1% of forward time |
| H8 Warp Divergence | **SUPERSEDED** | **Irrelevant** | Rasterization negligible |
| **H9 Radix Sort** | **SUPPORTED (NEW PRIMARY)** | **Primary (60-80% of total)** | Phase 8E per-kernel CUDA timing |
| H10 Pipeline Backpressure | FALSIFIED | None (bounded sum) | Can't explain 25× |

---

## UNBLOCKED Measurements (Phase 8E)

Phase 8E successfully performed per-kernel CUDA event timing by wrapping each gsplat CUDA function call with `torch.cuda.Event`. This isolated the overall time spent in each of the 5 major pipeline stages (projection, SH eval, intersect+sort, offset, rasterize) without requiring Nsight Compute. Result summary:

| Stage | tile16 (iter30000) | tile32 (iter30000) | Ratio | % of Forward |
|:------|:-----------------:|:-----------------:|:----:|:----------:|
| projection | 0.31ms | 0.34ms | 0.92× | < 1% |
| sh_eval | 0.09ms | 0.08ms | 1.18× | < 1% |
| **intersect+sort** | **193.7ms** | **31.2ms** | **6.21×** | **93-97%** |
| offset | 5.58ms | 1.17ms | 4.77× | 3-4% |
| rasterize | 0.44ms | 0.56ms | 0.80× | < 1% |
| **Total** | **200.1ms** | **33.4ms** | **6.00×** | **100%** |

Sort isolation (separate phase with 5s GPU cooldown):

| Metric | tile16 | tile32 | Ratio |
|:-------|:------:|:------:|:-----:|
| intersect_no_sort (pass1+cumsum+pass2) | 26.1ms | 6.1ms | 4.28× |
| intersect_with_sort | 119.6ms | 35.2ms | 3.40× |
| sort_estimated by difference | 93.5ms | 29.1ms | 3.22× |
| sort % of intersect_sort | 78% | 83% | — |

## Remaining BLOCKED Measurements

The following hardware-level measurements are BLOCKED by Nsight Compute unavailability (WDDM driver on RTX 5070 Laptop GPU):

| Measurement | Needed For |
|:------------|:-----------|
| Warp execution efficiency | H8 quantification |
| Shared memory bank conflicts | Memory efficiency |
| L2 cache hit/miss ratio | Memory pressure from large sorts |
| Register spilling | Occupancy analysis |
| Stalled cycles (waiting, memory, sync) | Sync overhead quantification |
| Achieved occupancy | Confirming/falsifying H4 |
| Per-kernel CUDA event timing | Isolating each kernel's contribution to total forward time |

**Alternative:** Run per-kernel CUDA events from Python by wrapping each gsplat CUDA function call. This can isolate overall time spent in (projection+SH+intersect+sort+offset+rasterize) groups without requiring Nsight.
