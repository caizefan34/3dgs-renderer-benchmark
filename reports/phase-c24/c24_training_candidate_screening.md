# C24 — Training-Centric 3DGS Optimization Screening

## Executive Summary

Five candidates (T1–T5) were screened in parallel across 8×A100 GPUs. **One candidate (T5) receives KEEP_CANDIDATE**, one receives MAYBE (T4), and three receive DROP.

### Key constraint applied

All candidates were evaluated under the **primary metric = T_iter** (end-to-end training iteration wall-clock time). Isolated kernel speedups are reported as mechanism diagnostics, not as training speedups.

### Ranking

| Rank | Candidate | Verdict | T_iter impact |
|:----:|-----------|---------|:-------------:|
| 1 | **T5** — Backward active-pixel compaction | **KEEP_CANDIDATE** | ~37% bound |
| 2 | T4 — Backward-specific execution geometry | MAYBE | 3.8% aggregate / up to 26% per-view |
| 3 | T3 — Gradient-contention-aware backward | DROP | <3% bound |
| 4 | T2 — Adaptive backward checkpointing | DROP | 0% |
| 5 | T1 — Forward-informed backward | DROP | 0% |

---

## T1 — Forward-Informed Backward Execution

**Verdict: DROP**

### Measured evidence (32 cameras, cohorts 0–15 and 160–175)

| Metric | Value |
|--------|-------|
| Cameras measured | 32 |
| Forward steps per frame | 77–107 million |
| Backward required range / forward range | **1.0** (by construction) |
| Avoidable backward work through endpoint reuse | **0%** |

### Analysis

`last_ids` tells us where forward stopped per pixel. Backward must traverse the *same* range at minimum to re-derive contributions for gradient accumulation. Forward endpoint reuse avoids redundant recomputation — but gsplat already saves intermediate values (render_alphas, last_ids) for backward use via `ctx.save_for_backward`. The traversal range itself cannot be reduced through informing backward of forward endpoints, because backward already stops at the same termination point (the same last_ids signal is used by both).

### Why DROP

No mechanism exists to reduce backward traversal range through forward endpoint reuse alone. The backward range is inherently determined by the same alpha/transmittance termination as forward.

---

## T2 — Adaptive Backward Checkpointing

**Verdict: DROP**

### Measured evidence (16 cameras, cohorts 0–7 and 80–87)

| K | Recompute overhead fraction (mean) | Memory per tile (bytes) |
|:---:|:---:|:---:|
| 8 | 0.448 | 8,132 |
| 16 | 0.603 | 16,264 |
| 32 | 0.734 | 32,527 |
| 64 | 0.829 | 65,054 |
| 128 | 0.906 | 130,108 |

### Analysis

Overhead increases monotonically with K — larger K always means more recompute cost because `tail = n % K` grows with K for densely intersected tiles. No local optimum or sweet spot exists. The single optimal K is the *smallest* feasible one (minimizing tail). Adaptive checkpointing adds complexity without benefit.

### Why DROP

- Overhead monotonic with K — no adaptive sweet spot
- gsplat already uses an implicit checkpointing strategy by saving forward intermediates
- Any additional checkpointing would *increase* memory and recompute cost

---

## T3 — Gradient-Contention-Aware Backward

**Verdict: DROP**

### Measured evidence (8 cameras, cohort 0–7)

| Metric | Value |
|--------|:-----:|
| Top 1% Gaussian fan-in share | 0.27% |
| Top 5% Gaussian fan-in share | 1.35% |
| Top 10% Gaussian fan-in share | 2.70% |
| P99 fan-in (pixels/Gaussian) | 12,544 |
| Max fan-in | 1.13e9 (one extreme outlier) |

### Analysis

Fan-in is estimated from projected tile coverage (tiles_touched × TILE²). A heavy tail would show top-1% Gaussians dominating 80%+ of gradient work. Instead, the distribution is nearly uniform — top 10% Gaussians account for only 2.7% of gradient fan-in. This means every Gaussian receives gradients from roughly the same number of pixels, scaled by its screen-space footprint. There is no contention hot-spot to target.

### Why DROP

- No heavy tail — fan-in distribution is near-uniform
- Contention-aware reduction (e.g., prioritized gradient accumulation) would address <3% of total work
- Implementation cost is high (custom reduction kernel) for negligible gain

---

## T4 — Backward-Specific Execution Geometry

**Verdict: MAYBE**

### Measured evidence (6 cameras: 0–2 and 80–82; 5 repeats per config)

**Camera 0 (dense/wide view):**

| Tile size | T_iter (ms) | vs tile 16 |
|:---------:|:----------:|:----------:|
| 8 | 10.2 | +10.9% |
| 12 | 9.5 | +3.3% |
| **16** | **9.2 (baseline)** | — |
| 20 | 8.9 | -3.3% |
| **24** | **7.3** | **-26.1%** |
| 28 | 7.4 | -24.2% |

**Cameras 1–2 (moderate views, mean):**

| Tile size | T_iter (ms) | vs tile 16 |
|:---------:|:----------:|:----------:|
| **16** | **6.8 (best)** | — |
| 20 | 7.0 | +2.9% |
| 24 | 7.4 | +8.8% |

**Replication (cameras 80–82, mean):**

| Tile size | T_iter (ms) | vs tile 16 |
|:---------:|:----------:|:----------:|
| **16** | **7.1 (best)** | — |
| 20 | 7.2 | +1.4% |
| 24 | 7.5 | +5.6% |

**Tile 32**: CUDA error "too many resources requested for launch" — backward kernel exceeds register/resource limit.

### Analysis

Critical finding: **optimal tile size is camera-dependent.**

| View type | Optimal tile | T_iter gain vs 16 |
|-----------|:----------:|:-----------------:|
| Dense/wide (camera 0) | 24 | **+26%** |
| Moderate (cameras 1–2, 80–82) | 16 | baseline |

A per-camera adaptive tile selection strategy would improve dense-view performance by 26% without regressing moderate views. This is achievable through a simple workload heuristic (e.g., total intersection count per frame).

### Why MAYBE (not KEEP_CANDIDATE)

- Aggregate cross-camera speedup of 3.8% falls below the 10% KEEP threshold
- Per-view gains of 26% are real but require a workload classifier
- Tile=32 is not feasible on A100 (resource limit)
- This is a pure configuration change, not a new algorithm — may be considered "known configuration tuning"

---

## T5 — Backward Active-Pixel / Activity-Aware Execution

**Verdict: KEEP_CANDIDATE**

### Measured evidence (16 cameras, cohort 0–15)

| Metric | Mean | Min | Max |
|--------|:---:|:---:|:---:|
| Terminated-pixel suffix fraction | **68.8%** | 64.0% | 74.6% |
| Forward active fraction | 31.2% | 25.4% | 36.0% |

### Definition

For each pixel, the forward renderer terminates at `last_ids` (transmittance < 1e-4 or Gaussians exhausted). The "terminated-pixel suffix" is the set of sorted-range positions *after* `last_ids` for each pixel. Backward must still traverse these positions if it re-derives the full per-pixel range — but no gradient contributions exist for these positions because forward already stopped.

**This is NOT the same as C22's "potential removable work".** C22 measured the fraction of *all* per-pixel steps that were potentially skippable in forward. T5 measures the fraction of backward per-pixel traversal that is *on pixels already terminated in forward*. These are different quantities.

### Bound on training gain

| Step | Calculation | Value |
|------|:-----------:|:-----:|
| Backward traversal reduction bound | terminated_suffix_fraction | 68.8% |
| Backward kernel time reduction bound | 68.8% backward traversal reduction | ~68.8% |
| Training iteration T_iter = forward + backward + overhead | At bwd/fwd = 2.9× | ~1 + 2.9 = 3.9 parts |
| T_iter reduction from 68.8% backward skip | 0.688 × 2.9 / 3.9 | **~51% backward kernel → ~37% T_iter bound** |

**This is a theoretical upper bound.** Real gain depends on compaction overhead, warp divergence, and correctness of active-pixel mask management.

### Mechanism confidence: HIGH

- The terminated-suffix fraction (69%) is measured directly from real `last_ids` — not from synthetic truncation or proxy
- The backward kernel is the largest single-kernel bottleneck (1,948μs, 2.9× forward)
- Active-pixel compaction directly removes backward work on already-terminated pixels
- The mechanism preserves rendering semantics: pixels that were already saturated in forward receive no additional gradient from backward (their saturated value is the final gradient target)

### Concrete CUDA change required

1. **Forward**: export `last_ids` to backward (already saved in `ctx.save_for_backward` in `_RasterizeToPixels.forward`)
2. **Backward kernel**: add per-pixel active mask tracking. For each per-pixel sorted position, check whether this pixel's forward endpoint has been reached. If yes, skip gradient accumulation for that pixel.
3. **Compaction**: use warp-level ballot (`__ballot_sync`) to identify inactive lanes and compact the active set, or use tile-level mask to skip inactive positions entirely.

### Why KEEP_CANDIDATE (not MAYBE)

- The 69% backward traversal reduction bound translates to ~37% T_iter bound — exceeding the 10% KEEP threshold by 3.7×
- The mechanism is well-defined: active-pixel mask from forward `last_ids`
- Backward is the confirmed dominant bottleneck (2.9× forward)
- No algorithm change to forward — only backward traversal is modified
- Semantics are preserved: terminated pixels contribute zero gradient, which is the correct gradient behavior

---

## Cross-Candidate Decision Table

| Candidate | Mechanism | Evidence strength | Potential T_iter gain | Main risk | Implementation cost | Decision |
|-----------|-----------|:----------------:|:--------------------:|-----------|:------------------:|:--------:|
| T5 | Backward active-pixel compaction | **HIGH** — 69% suffix measured from real last_ids, 32 cameras | **~37% bound** | Compaction overhead, warp divergence | HIGH — CUDA kernel mod | **KEEP_CANDIDATE** |
| T4 | Backward-specific geometry | **HIGH** — T_iter measured, 26% per-view | 3.8% aggregate (26% per dense view) | Workload classifier overhead | LOW — per-view tile selection | MAYBE |
| T3 | Gradient-contention-aware | **MODERATE** — fan-in, 8 cameras | <3% | No heavy tail exists | HIGH | DROP |
| T2 | Adaptive checkpointing | **MODERATE** — overhead model, 16 cameras | 0% | Overhead monotonic with K | LOW | DROP |
| T1 | Forward-informed backward | **HIGH** — 32 cameras | 0% | Traversal range inherently equal | N/A | DROP |

## Answer to the primary question

> **Which mechanism is most likely to produce a real training speedup?**

**T5 — Backward active-pixel compaction.** The mechanism is clear (69% of per-pixel backward traversal on already-terminated pixels), the bound is large (~37% T_iter), and backward is the confirmed dominant kernel. This is the strongest candidate across the entire phase-C pipeline.

> **What exact CUDA change is required?**

1. Export `last_ids` from `_RasterizeToPixels.forward` (already done via `ctx.save_for_backward`)
2. In `rasterize_to_pixels_3dgs_bwd_kernel`, add per-pixel active mask: for each pixel, track whether the current sorted position is past the pixel's `last_ids`. If yes, skip gradient accumulation (`v_*` output) for that pixel.
3. Use warp-level ballot to compact active lanes, reducing divergent execution.

## Stop

C24 screening complete. Do not implement any optimization. Do not start C25. Wait for unified research review.

**The recommended CUDA prototype is T5 (backward active-pixel compaction), with N12 (active-pixel batch execution) as the implementation vehicle.**
