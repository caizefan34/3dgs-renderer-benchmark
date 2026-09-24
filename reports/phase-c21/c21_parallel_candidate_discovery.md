# C21 — Parallel Candidate Discovery & Feasibility Screening

## Executive Summary

Six candidates were screened in parallel on 8×A100 GPUs. **No candidate receives a definitive KEEP.** N7 (opacity-aware sorted-range truncation) and N3 (transmittance-driven batch skipping) and N6 (sort/raster co-optimization) each receive MAYBE, each with significant caveats. N1, N8, and N10 are dropped.

### Critical findings

- **N10 directly contradicts C6's scheduling hypothesis.** Under 10 repetitions with rotated policy order, static round-robin (mean 8,226 ms) consistently outperformed every LPT policy (best LPT: 8,342 ms). The C20 3.48% LPT gain was a single-run artifact.
- **N7's truncation feasibility is viewpoint-dependent.** One camera group (240–247) reached 88% per-tile pixel saturation and 49% of pixels within 1e‑3 alpha error at 50% depth truncation. Another group (160–167) reached only 63% saturation with 16% of pixels meeting the same threshold. A uniform truncation mechanism cannot work.
- **Per-pixel `last_ids` is unavailable** through the stock gsplat Python API (`rasterize_to_pixels()` returns only `(rendered, alpha)`). All truncation evidence in this phase comes from per-tile depth-fraction replay, which is a bound, not an exact counter.
- **N6's coupled trade-off is confirmed but a stable phase boundary is not.** Tile 8 has 5.47M intersections and 1.74 ms raster; tile 32 has 0.56M intersections and 2.84 ms raster. The crossing point (~tile 12–16) is consistent with C19, but one scene and camera cannot establish workload regime separation.

| Rank | Candidate | Verdict | Evidence |
|:----:|-----------|---------|----------|
| 1 | **N7** — Opacity-aware sorted-range truncation | **MAYBE** | 49% pixels within 1e‑3 alpha error at 50% depth truncation (best view); 16% (worst view). View-dependent only. |
| 2 | **N3** — Transmittance-driven batch skipping | **MAYBE** | 12–18% pixels within 1e‑3 at 50% truncation. Weaker than N7. |
| 3 | **N6** — Sort/raster co-optimization | **MAYBE** | Coupled trade-off confirmed; no phase-map evidence. |
| 4 | N1 — Traversal-aware rasterization | **DROP** | 13.5% pixels within 1e‑3 at 50% truncation; too weak. |
| 5 | N10 — P99-aware multi-GPU scheduling | **DROP** | 10‑run repetition rejects LPT; static is best. |
| 6 | N8 — Hybrid/workload-aware sorting | **DROP** | No isolated radix timing available. |

## Candidate evidence

### N1 — Traversal-Aware Rasterization (GPU0, cameras 0–7)

Per-tile pixel saturation fraction: mean 0.657. 50% depth truncation produces mean alpha MSE 0.113; only 13.5% of pixels have alpha error below 1e‑3. **Drop:** the truncation error is too large to make a mechanism claim without per-pixel tracking that stock metadata cannot provide.

### N3 — Transmittance-Driven Batch Skipping (GPU1, cameras 80–87; GPU6 replication cameras 200–207)

| Metric | Cameras 80–87 | Cameras 200–207 (replication) |
|--------|:------------:|:----------------------------:|
| Per-tile pixel saturation fraction (mean) | 0.712 | 0.607 |
| 50% truncation MSE | 0.192 | 0.112 |
| Pixels within 1e‑3 alpha error at 50% | 17.7% | 12.3% |

The replication confirms a consistent bound: roughly 60–71% of pixel positions are fully saturated, but this does **not** mean 60–71% of intersections are skippable. When 50% of depth-sorted intersections are removed, only 12–18% of pixels stay within 1e‑3 of the correct alpha. The rest are affected by the truncated Gaussians that, while individually small, collectively matter.

### N6 — Sort/Raster Co-Optimization (GPU2, C19 data reuse)

| Tile size | Intersections | Intersect+sort ms | Raster ms | Forward ms |
|:---------:|:------------:|:-----------------:|:---------:|:----------:|
| 8 | 5,465,619 | 2.639 | 1.744 | 5.012 |
| 12 | 2,640,904 | 0.858 | 1.120 | 2.524 |
| **16** | **1,626,135** | **0.984** | **1.938** | **3.498** |
| 20 | 1,125,232 | 0.769 | 2.226 | 3.583 |
| 24 | 849,047 | 0.655 | 2.352 | 3.599 |
| 28 | 684,937 | 0.359 | 1.682 | 2.481 |
| 32 | 564,323 | 0.540 | 2.838 | 3.921 |

The minimum forward time is at tile 28 (2.481 ms), not tile 16 (3.498 ms) — a 29% improvement, paid for by lower intersection count dominating higher per-intersection raster cost. This confirms the coupled trade-off. However, C19 did not isolate radix-sort timing from intersect timing, and a single scene/camera workload cannot establish a stable density-regime phase map.

### N7 — Opacity-Aware Sorted-Range Truncation (GPU3, cameras 160–167; GPU7 replication cameras 240–247)

| Metric | Cameras 160–167 | Cameras 240–247 (replication) |
|--------|:--------------:|:----------------------------:|
| Per-tile pixel saturation fraction (mean) | 0.627 | **0.882** |
| 50% truncation MSE | 0.122 | **0.019** |
| Pixels within 1e‑3 alpha error at 50% | 16.3% | **49.0%** |

This is the strongest individual signal in C21. The camera-240–247 viewpoint (a close-up or dense-view camera) achieves 88% pixel saturation and 49% of pixels within 1e‑3 alpha error when half the sorted Gaussians are removed. The camera-160–167 viewpoint shows much weaker benefit. **The mechanism is viable only if paired with a view classifier** that can distinguish near-saturated views from those where truncation would be visible.

### N8 — Hybrid / Workload-Aware Sorting (GPU4, C19 data reuse)

Sort input changes 9.69× from tile 8 (5.47M) to tile 32 (0.56M) while mean intersections/tile rise from 169 to 277. However, C19 measured combined intersect-tile timing (pass 1 intersect + CUB radix sort), not isolated sort time. Without per-regime radix timing, a workload-dependent sorting policy cannot be evaluated.

### N10 — P99-Aware Multi-GPU Scheduling (GPU5, 10 repetitions, 4×A100)

| Policy | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|--------|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| **Static round-robin** | **8,226** | 7,847 | 9,178 | 9,590 | 9,920 | 10,003 |
| Gaussian-count proxy | 8,592 | 8,106 | 9,956 | 10,244 | 10,475 | 10,533 |
| Intersection LPT | 8,342 | 7,895 | 9,805 | 9,846 | 9,879 | 9,887 |
| Render-time LPT | 9,380 | 8,446 | 11,175 | 14,262 | 16,732 | 17,349 |

**Static round-robin wins on every percentile.** The C20 single-run 3.48% LPT gain was an artifact of order effects (the non-LPT policy was measured when GPU4–7 were cold). With 10 repetitions and rotated policy order, every LPT policy is equal or worse. Render-time LPT is dramatically worse (P99 16,732 vs 9,920 ms), likely because the first repetition pays profile overhead.

## Research questions

### Where is the largest amount of unnecessary computation in the current 3DGS renderer?

The `rasterize_to_pixels_3dgs_fwd_kernel` (55.5% of forward CUDA time) processes many Gaussians after per-pixel transmittance reaches near-zero. Our truncation replay establishes that **for favorable views, up to 49% of pixels experience negligible alpha change when half the depth-sorted intersections are removed**. For unfavorable views, the fraction drops to 12–16%. The exact fraction is unknowable without CUDA-instrumented per-pixel `last_ids` — which stock gsplat does not expose.

The isect_tiles + sort pipeline (combined ~27% of forward time) also contains unnecessary computation via intersection duplication (9.2× at tile 16). But this is an inherent cost of tile-based rasterization, not removable work.

### Which candidate has the highest probability of becoming a real, measurable optimization?

**N7** — opacity-aware sorted-range truncation — **but only if a view/workload predictor can distinguish high-saturation views from low-saturation views.** The camera-240–247 evidence (49% of pixels safe at 50% truncation) is strong enough to justify a CUDA prototype that implements per-pixel `last_ids` tracking, compares full vs truncated rendering, and measures actual speedup. Without the predictor, N7 degenerates into N3/N1 territory (12–18% safe pixels).

N6 is the second candidate (adaptive tile size), but its gain potential is bounded by the coupled trade-off — a 29% forward-time difference between tile 28 and tile 16 is already achievable in stock gsplat by simply selecting tile 28.

### Problem classification

The evidence converges on **B — Traversal/work-amplification problem**, with influence from **A — Execution-geometry problem**.

- **B (dominant):** The rasterizer traverses many Gaussian intersections per tile even after per-pixel transmittance is saturated. N7/N3 evidence bounds the potentially skippable work at 12–49% depending on view.
- **A (secondary):** The tile-CTA coupling constrains which execution geometry can be used. Tile 28's forward time (2.48 ms) is 29% faster than tile 16 (3.50 ms) purely through configuration change — no algorithm change.
- **C (sorting):** Minor contributor. Combined intersect+sort is ~27%; isolated sort is likely smaller. Not a primary target.
- **D (scheduling):** **Rejected.** N10 shows no benefit from workload-aware scheduling on homogeneous 4×A100.
- **E (forward/backward asymmetry):** Not addressed by any C21 candidate.

## Deliverables

- `reports/phase-c21/c21_parallel_candidate_discovery.md` (this file)
- `results/phase-c21/c21_parallel_candidate_discovery.json`
- Raw evidence files under `results/phase-c21/`

## Stop

C21 is complete. Do not begin CUDA implementation. Wait for unified research review.
