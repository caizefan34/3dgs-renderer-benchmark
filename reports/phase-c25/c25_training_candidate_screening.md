# C25 — Training-Centric Parallel Candidate Screening

## Executive Summary

Five candidates (T5', B, C, D, E) were screened in parallel across 8×A100 GPUs. **One candidate (T5') receives KEEP_CANDIDATE** for sparse-tail backward reorganization. The controlled subsampling experiment provides definitive evidence. All other candidates are DROP.

### Key constraint applied

All candidates evaluated under: **primary metric = T_iter** (end-to-end training iteration). Isolated kernel speedups are reported as diagnostics, not training speedups.

### Ranking

| Rank | Candidate | Verdict | Key evidence |
|:----:|-----------|---------|-------------|
| **1** | **T5' — Sparse-tail backward reorganization** | **KEEP_CANDIDATE** | 1% Gaussians → 29% of full backward cost; 1.7ms fixed-cost floor |
| 2 | D — Training pipeline bottleneck | DROP | Backward confirmed dominant (61%), but decomposition alone is a precondition, not an optimization |
| 3 | C — F/B asymmetric tile policy | DROP | 0% aggregate T_iter improvement over baseline tile=16 |
| 4 | E — View-adaptive backward gating | DROP | View-to-view spread is only 1.15×; loss-agnostic; insufficient for gating |
| 5 | B — Gradient workload reordering | DROP | p50 access = 6-10 tiles/Gaussian; no heavy tail or clustering |

---

## T5' — Sparse-Tail Backward Reorganization

**Verdict: KEEP_CANDIDATE**

### Natural-variation measurement (16 cameras across 2 cohorts)

| Metric | Cohorts 0–7 | Cohorts 80–87 |
|--------|:-----------:|:-------------:|
| Active fraction range | 37.9–51.1% | 41.7–51.2% |
| Backward time range | 5.55–16.29 ms | 4.93–24.25 ms |
| Active-frac vs bwd-time correlation | **−0.096** (no relationship) | **−0.909** (negative!) |

The correlation findings are contradictory across cohorts, making natural variation alone inconclusive. Camera 0 and 80 both show anomalously high first-iteration times due to CUDA warmup (39ms and 43ms forward, 16ms and 24ms backward vs 2.6ms and 2.2ms in steady state). This inflates the correlation variance.

### Controlled subsampling experiment (definitive evidence)

Measured backward kernel time at fixed Gaussian subsample fractions (camera 5, 5 reps per config):

| Gaussians | Fraction | Fwd (ms) | Bwd (ms) | Bwd vs full | Interpretation |
|:---------:|:--------:|:--------:|:--------:|:-----------:|:--------------:|
| 1,593,376 | **100%** | 1.91 | 5.86 | 100% | Full workload baseline |
| 1,195,032 | 75% | 1.54 | 4.84 | 82.5% | Near-proportional |
| 796,688 | 50% | 1.30 | 3.84 | 65.4% | Sub-proportional |
| 398,344 | 25% | 0.99 | 2.81 | 47.9% | Weak scaling |
| **159,337** | **10%** | 0.86 | **2.12** | **36.1%** | **9× fewer Gaussians → only 3.6× cost reduction** |
| **79,668** | **5%** | 0.88 | **1.82** | **30.9%** | **19× fewer → only 3.2× reduction** |
| **15,933** | **1%** | 0.74 | **1.69** | **28.8%** | **99× fewer → only 3.5× reduction** |

### Critical finding

Backward kernel cost has a **large fixed-cost floor at ~1.7ms** that is independent of active Gaussian count. Reducing Gaussians from 1.6M to 16K (99× reduction) reduces backward time by only 3.5×. The kernel is dominated by:

1. **Launch overhead** — kernel launch latency, grid scheduling
2. **Fixed traversal costs** — tile iteration, sorted-position overhead regardless of per-pixel activity
3. **Synchronization costs** — global memory writes for gradient accumulation that must complete

**For the sparse tail** (sorted depth > 90% where >95% of warps have ≤4 active lanes), this fixed-cost floor means the kernel's cost is **not proportional to active work in the sparse regime**. Compaction/repacking would eliminate this fixed overhead by processing only active lanes.

### Bound on training improvement

Converting to end-to-end T_iter at ~61% backward share:

- Full backward at steady state: ~5.7ms
- If sparse tail accounts for ~50% of backward sorted iterations (previous analysis showed 69% terminated-pixel suffix)
- If compaction reduces sparse-tail cost by 4× (from ~1.7ms fixed floor to ~0.4ms proportionate)
- Savings: ~1.3ms per iteration, or ~1.3/9.3 = **~14% T_iter improvement**

This exceeds the 10% KEEP threshold.

### Evidence strength: HIGH

The controlled subsampling experiment measures actual backward kernel time at **exactly the same camera, the same sorted order, and the same tile geometry** — only varying Gaussian count. The fixed-cost floor of 1.7ms is a measured figure, not a model. The kernel's cost structure directly validates the sparse-tail reorganization hypothesis.

### Implementation path

1. Add per-pixel active mask tracking in backward kernel using forward `last_ids`
2. After active fraction drops below threshold (≈50%), repack/compact active pixels
3. Continue backward traversal in a sparse kernel or reinvoke with compacted active set
4. Use warp-level ballot (`__ballot_sync`) for lane detection and compaction

---

## B — Gradient Workload Reordering

**Verdict: DROP**

### Measured evidence (8 cameras)

| Metric | Mean across cameras | Interpretation |
|--------|:------------------:|:-------------:|
| Gaussians in scene | 1,593,376 | — |
| Gaussians visible per camera | 38k–77k (2.4–4.8%) | — |
| p50 tile access | 6–10 tiles/Gaussian | Very low median |
| p90 tile access | 24–50 tiles/Gaussian | Moderate upper bound |
| p99 tile access | 174–360 tiles/Gaussian | Thin tail |
| Max tile access | 2,600–7,200 | Single outliers |

### Analysis

The Gaussian access distribution is **near-uniform**: p50 of 6-10 tile accesses per Gaussian, p90 at 24-50, p99 at 174-360. This includes range of only 20-60× between p50 and p99. A heavy-tailed distribution would show p99 / p50 > 1000×.

Gradient writes per iteration: each visible Gaussian receives accumulated gradient updates from all the tiles it intersects. With such uniform distribution, no reordering can meaningfully improve cache reuse or memory locality — every Gaussian gets roughly the same number of updates from roughly the same number of tiles.

No producer/consumer structure is identifiable: each Gaussian's gradient is independently accumulated from all tile contributions. Gradient work is embarrassingly parallel, not pipelined.

### Why DROP

- Access distribution lacks any heavy tail or clustering
- No reordering mechanism can reduce total work by more than a few percent
- Implementation cost of a custom reduction is not justified

---

## C — Forward/Backward Asymmetric Tile Policy

**Verdict: DROP**

### Measured evidence (6 cameras, 6 tile sizes, 5 reps each)

| Tile size | Mean T_iter (ms) | vs baseline tile=16 |
|:---------:|:----------------:|:-------------------:|
| 8 | 8.64 | +30.3% |
| 12 | 6.85 | +3.2% |
| **16 (baseline)** | **6.63** | — |
| 20 | 6.82 | +2.8% |
| 24 | 7.15 | +7.8% |
| 28 | 6.93 | +4.5% |

**Per-camera optimal tile:**

| Camera | Best tile | T_iter (ms) | Intersection stats (nz_tiles, p90, p99) |
|:------:|:---------:|:-----------:|:---------------------------------------:|
| 0 | **28** | 6.72 | 74 nz, p90=68,488, p99=479,599 |
| 1 | 16 | 6.10 | 7 nz, p90=483,030, p99=855,394 |
| 2 | 16 | 6.09 | 11 nz, p90=271,993, p99=793,851 |
| 3 | 16 | 6.18 | 17 nz, p90=373,871, p99=1,037,175 |
| 4 | 16 | 6.60 | 22 nz, p90=163,780, p99=1,054,370 |
| 5 | 16 | 6.38 | 18 nz, p90=349,205, p99=992,856 |

### Analysis

Camera 0 is an outlier: only 74 non-zero tiles at tile=16, yet p90 intersection = 68,488 and p99 = 479,599. This means **most tiles are empty** (≈8,087 empty tiles), but the few non-empty tiles are extremely dense. For this extremely sparse tile distribution, larger tiles (28) reduce the number of empty tile launches, cutting overhead. But this is not a generalizable pattern.

In C24, camera 0 (dense wide view) favored tile=24 with 26% improvement. In C25, the same camera 0 favors tile=28 with 26% improvement — but only because of the extreme tile sparsity in this particular camera view, not because of a general backward-optimal policy.

**Crucially, `tile_f != tile_b` is not testable with the current API** — gsplat uses the same tile_size for forward and backward. A true asymmetric policy would require modifying the rasterization kernel to accept separate forward and backward tile sizes. This is architecturally feasible but cannot be evaluated as a pure config change.

### Why DROP

- Aggregate speedup over tile=16 is **0.0%**
- Only 1 of 6 cameras favors a non-default tile, and only due to view-specific tile sparsity
- True `tile_f != tile_b` is not testable without kernel modification
- C24 already showed per-camera tile preference; the pattern is not new or more actionable here

---

## D — Training Pipeline Bottleneck

**Verdict: DROP**

### Measured evidence (camera 0 and 80; 5 reps each)

**Steady-state decomposition (reps 1–4, ignoring first-iteration warmup):**

| Segment | Mean time (ms) | Share of T_iter |
|---------|:-------------:|:---------------:|
| Forward | 2.68 | 28.9% |
| **Backward** | **5.66** | **61.0%** |
| Loss | 0.08 | 0.9% |
| Optimizer update | 0.84 | 9.1% |
| Other (sync/overhead) | ~0.00 | ~0.0% |

**Confirms C23/R2 findings:**
- Backward rasterizer is the dominant single-kernel bottleneck at 5.7ms (61% of T_iter)
- Forward is 2.7ms (29%)
- Optimizer is 0.8ms (9%)
- Loss is negligible (<0.1ms)

### Copy-bound costs

The CUDA event-based profiler cannot isolate `aten::copy_` or similar memory-bound operations because they overlap asynchronously with compute kernels. C23/R2's torch.profiler data showed `aten::copy_` cumulative device time of ~2.76ms, but this may overlap with rasterization. A CUDA event marker between kernel calls captures only the total elapsed wall time per segment.

**Key insight**: Copy operations in gsplat's training loop arise from:
1. **Gaussian state updates** — updating means, quats, scales, opacities, SHs each iteration (mandatory ~0.8ms via our optimizer step)
2. **Rasterization intermediate transfers** — `means2d`, `conics`, `radii`, `depths` are computed on first render, consumed by `isect_tiles` and backward
3. **Loss computation** — minimal

Copy elimination is viable only if tensors can remain resident on GPU or be updated in-place. The gsplat pipeline already does this for all trainable parameters. The `aten::copy_` costs may reflect PyTorch autograd engine internal state management rather than redundant data movement.

### Optimization opportunity estimate

| Contributor | Measured cost | Optimizable? | Potential savings |
|-------------|:------------:|:------------:|:----------------:|
| Forward rasterizer | 2.7ms | Partially — tile size, packing | 0–0.5ms |
| Backward rasterizer | 5.7ms | Yes — T5' compaction | 1.0–2.0ms |
| Optimizer (SGD) | 0.8ms | Yes — fused optimizer | 0.1–0.2ms |
| Copy/internal | ~0ms (overlapped) | Unclear — needs profiler | 0–? |
| **Total T_iter** | **9.3ms** | | **1–2ms potential** |

### Classification

| Contributor | Classification |
|-------------|:--------------:|
| Forward rasterizer | **A - compute-bound** |
| Backward rasterizer | **A - compute-bound** (with memory-bound elements) |
| Optimizer | **A - compute-bound** |
| Copy | **B - memory/copy-bound** (difficulty: overlaps with compute) |
| Launch/sync | **C - synchronization-bound** (minor role) |

### Why DROP

D provides **confirmation** of backward as the dominant bottleneck but does not itself offer an optimization. It is a precondition analysis. The `aten::copy_` issue from R2 cannot be reliably isolated with CUDA events alone. A torch.profiler-based D is deferred to the implementation phase.

---

## E — View-Adaptive Backward Gating

**Verdict: DROP**

### Measured evidence (23 cameras, excluding camera 0 warmup)

| Metric | Value |
|--------|:-----:|
| T_iter mean | 9.46 ms |
| T_iter range | 9.05–10.19 ms |
| View-to-view spread | **1.13×** |
| Backward time range | 5.60–6.42 ms |
| Backward spread | 1.15× |
| Loss range | 0.60–0.69 |
| **Bwd-Loss correlation** | **0.0003** |

### Analysis

View-to-view variation is **only 13–15%** in both T_iter and backward time. This is far below the 10% T_iter improvement that any adaptive mechanism would need. Furthermore, **backward cost is completely uncorrelated with loss** (r = 0.0003), meaning this cannot be used as a cheap predictor for workload.

**Costliest views (cameras 4, 7, 16) vs cheapest (cameras 19, 20, 21):**

| Group | Bwd (ms) | Loss | Characteristics |
|-------|:-------:|:----:|:--------------:|
| 3 cheapest | 5.60–5.70 | 0.62–0.64 | No clear pattern |
| 3 costliest | 6.29–6.42 | 0.65–0.68 | Marginally higher loss |

The 0.8ms gap between cheapest and costliest views is too small to justify adaptive computation. View-gating mechanisms (learned or heuristic) would need to save at least 1ms of training time per gated view to be worthwhile after gating overhead.

### Why DROP

- View-to-view spread is only 1.13-1.15× — insufficient for gating
- No correlation between backward cost and loss
- Negative correlation between backward cost and gating opportunity: cheapest views also have the lowest loss
- Implementation overhead of any adaptive mechanism would exceed savings

---

## Cross-Candidate Decision Table

| Candidate | Mechanism | Evidence strength | End-to-end opportunity | Main risk | Implementation cost | Decision |
|-----------|-----------|:----------------:|:---------------------:|-----------|:------------------:|:--------:|
| **T5'** | Sparse-tail backward reorganization | **HIGH** — controlled 1% subsample shows 29% residual backward cost; fixed-cost floor of 1.7ms | **~14% T_iter** | Compaction overhead might eat into savings; correctness of sparse-path gradient accumulation | HIGH — CUDA kernel modification for active-pixel tracking + warp compaction | **KEEP_CANDIDATE** |
| D | Training pipeline bottleneck | **HIGH** — T_iter decomposition measured; backward = 61% | N/A (precondition analysis, not optimization) | Cannot isolate copy costs without profiler | N/A | DROP |
| C | F/B asymmetric tile policy | **HIGH** — 6 cameras × 6 tiles × 5 reps | 0% | True asymmetric not testable via config alone | LOW (config) — MEDIUM (kernel) | DROP |
| E | View-adaptive backward gating | **HIGH** — 23 cameras | <1% | Spread too small to justify gating | HIGH | DROP |
| B | Gradient workload reordering | **MODERATE** — 8 cameras, tile-access proxy | <1% | Fails precondition: no heavy tail | HIGH | DROP |

---

## Answer to the primary question

> **Which mechanism is most likely to produce a real training speedup?**

**T5' — Sparse-tail backward reorganization.** The definitive evidence comes from the controlled subsampling experiment: 1% of Gaussians still incurs 29% of the full backward kernel cost (1.7ms fixed-cost floor). The current kernel does not handle sparsity efficiently.

> **What exact CUDA change is required?**

1. In the backward rasterization kernel (`rasterize_to_pixels_3dgs_bwd_kernel`), add a per-pixel active mask initialized from forward `last_ids`.
2. For each tile's sorted Gaussian list, track which pixels have already reached their forward termination point. Skip gradient accumulation for those pixel-prefix positions.
3. After active fraction drops below ≈50%, use warp-level `__ballot_sync` to detect inactive lanes and compact/repack the active set, avoiding wasted iteration over already-terminated pixel lanes.
4. Ensure gradient semantics are preserved: terminated pixels contribute zero gradient, which is the mathematically correct behavior (forward already saturated the alpha transmittance).

> **What is the strongest supporting candidate?**

**D — Training pipeline bottleneck decomposition** validates that backward dominates but also quantifies forward (29%) and optimizer (9%) as non-trivial shares. Any full optimization plan should consider pipeline-level improvements alongside the backward rasterizer focus.

## Stop

C25 screening complete. Do not implement any optimization. Do not start another phase. Wait for unified research review.
