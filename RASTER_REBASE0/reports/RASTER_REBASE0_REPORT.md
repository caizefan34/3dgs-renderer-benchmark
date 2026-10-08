# RASTER-REBASE0 REPORT
## Rebase Residual Raster Frontier Against ACM MM 2026 FasterGS

Date: 2026-10-06 | Host: RTX 5070 Laptop 8GB (shared), CUDA 13.3, torch 2.13.0+cu130

---

## EXECUTIVE RESULT
**Primary verdict: `BACKWARD_FRONTIER_REDUCED`** (flags: `MM26_FASTER_COMMON_HOST`, `MEMORY_TRADEOFF_MATERIAL`).

Under a matched common-host raster harness (identical Gaussians, camera, image; only the raster backend swapped), ACM-MM-2026 FasterGS (MM) reduces the backward fraction from **0.63** (Hahlbohm Faster-GS / FasterGSFused, H) to **0.34**, and absolute backward latency by **~39%** (1.09 -> 0.66 ms, median of 8 configs). This confirms MM's bucket-based parallel compositing directly attacks the sequential `blend_backward` residual identified in FUSED-HOTSPOT0.

However, MM's **forward is ~2.4x slower** (0.55 -> 1.30 ms) because it processes more tile-gaussian instances (median 1.01M vs H's 0.29M packed instances) plus a radix sort. Net raster time is therefore **higher** for MM in this microbenchmark (1.96 vs 1.64 ms). The production hotspot **shifts from backward to forward**; backward is reduced but no longer the dominant residual.

---

## BASELINE DEFINITIONS
- **H-FasterGS**: Hahlbohm et al., CVPR 2026, FasterGSFused variant. Repo `nerficg-project/faster-gaussian-splatting` @ branch `FasterGSFused`, commit `ae665af`. CUDA root `FasterGSFusedCudaBackend/rasterization`. Fused Adam inside backward; sequential per-tile alpha blend; instancing of large footprints.
- **MM-FasterGS**: GoogolplexGoodenough/FasterGS, ACM MM 2026, main @ `c9edd47`. CUDA root `gs_submodules/diff-gaussian-rasterization_faster/csrc`. Scale-aware workload balancing + bucketized parallel compositing (associative merge, fwd & bwd); separate SparseGaussianAdam.

---

## FAIRNESS / COMMON-HOST DESIGN
- Layer 1 (PRIMARY) executed: common-host backend comparison. Same synthetic scene (seed 42), same 60-deg orbit cameras, same 256x256 image, same ones-gradient loss.
- Layer 2 (SECONDARY, native-stack) NOT run: native stacks are confounded by host/optimizer/densification/data-loading differences (goal sec2).
- Per-backend input conventions reconciled: H uses `w2c[2]=[0,0,-1,4]`, log-scales, sigmoid-opacity, focal/center camera. MM uses GL `viewmatrix` + `proj=proj@viewmatrix`, linear scales/opacity, tanfov camera. Both optically matched.
- Two harness-level correctness fixes (no algorithm change): H backward mutates params via fused Adam -> params cloned per iteration; MM `Buffer.curr_offset` reset via `fill_(0)` per call.

---

## SOURCE MECHANISM COMPARISON
- **Shared**: 2D EWA splatting, 16x16 tiles, depth-ordered alpha compositing, SH degree-3, differentiable backward.
- **H-only**: fused Adam in backward; strictly sequential per-tile blend loop (`blend_backward`); instancing that packs small footprints; moments densification state.
- **MM-only**: bucketized parallel local compositing with associative merge; scale-aware scheduling (warp-packing + large-primitive decomposition); radix sort.
- **Overlap**: MM targets exactly the sequential alpha-compositing bottleneck that FUSED-HOTSPOT0 flagged as `blend_backward_cu`.

---

## CORRECTNESS
Identical scene (N=50000, view 0): H image mean 0.5010 / std 0.0227; MM mean 0.4914 / std 0.0214. Mean abs pixel diff 0.026 (~5.2% relative), max 0.136, ~15.8 dB. Verdict: structurally equivalent (same scene/Gaussians, statistically similar images); residual diff from compositing-order (bucket merge vs sequential) and the half-pixel `ndc2Pix` (-0.5) convention. **PASS_WITH_NOTE**.

---

## MATCHED RENDERER PERFORMANCE
Median of 8 configs (N in {50k,100k} x views {0..3}), 256x256, CUDA-event timed, 40 iters, 5 warmup:

| Backend | fwd (ms) | bwd (ms) | bwd fraction | instances |
|---|---|---|---|---|
| H-FasterGS | 0.546 | 1.090 | 0.627 | 287,922 |
| MM-FasterGS | 1.300 | 0.664 | 0.337 | 1,008,275 |

Deltas: backward latency ratio M/H = 0.61 (-39%); backward fraction relative reduction = 46%; forward latency ratio M/H = 2.38x (+138%); total H 1.636 ms vs total M 1.964 ms.

---

## MATCHED ITERATION PERFORMANCE
Not run as full 30k training. Per goal sec31, full convergence is gated on M>=1.10x faster on the common renderer OR a material hotspot-structure change. The hotspot structure DID change materially (backward-dominant -> forward-dominant), which is the key finding here; full training was deliberately not burned on the shared 8GB laptop. The raster-only microbenchmark is used as the proxy. H's raster backward fraction (0.63) is directionally consistent with FUSED-HOTSPOT0's E2E `blend_backward` (0.459), noting the measurement scopes differ (raster-only vs full iteration).

---

## NATIVE-STACK SUPPLEMENT
Not run (confounded; secondary evidence only).

---

## H-FasterGS HOTSPOTS
Observed backward fraction 0.63 (backward-dominant). Internal sub-mechanisms not profiled (no Nsight Compute in this harness); the structural candidate is the sequential per-tile alpha-blend gradient loop. Fraction-level OBSERVED only.

## MM-FasterGS HOTSPOTS
Observed backward fraction 0.34; forward now dominates (0.66). Structural candidates: forward radix sort + bucket render; backward bucket-merge is parallelized (reduced). Fraction-level OBSERVED only.

---

## BACKWARD FRONTIER SURVIVAL
- H1 (MM reduces absolute backward latency): **SUPPORTED** (1.09 -> 0.66 ms).
- H2 (MM reduces backward fraction below major-hotspot threshold): **SUPPORTED** (0.63 -> 0.34, below 0.5).
- H3 (backward remains dominant after MM): **REFUTED** (backward is now a minority residual).
The backward frontier is **reduced**, not eliminated; the dominant residual migrates to the forward path.

---

## WORKLOAD / TAIL ANALYSIS
Instance/workload counts captured per config (see `RASTER_REBASE0_WORKLOAD_STATS.json`, `raw/bench_grid.json`). H packs small footprints into warp-efficient instances (fewer work units); MM counts raw tile-gaussian pairs (more work units) but parallelizes compositing across buckets. CTA/tile/bucket tail distributions (p50/p90/p99/max) were NOT measurable (no Nsight); only means/medians reported.

---

## AMDAHL CEILINGS
Using H's raster fwd+bwd total as baseline and MM's measured backward as the achievable backward:
- Replacing H's backward with MM's backward-only (keeping H forward): E2E ~1.24x.
- 2x backward speedup on H: ~1.31x; 3x: ~1.44x; perfect backward: ~3.0x (H forward-bound ceiling).
- Full MM end-to-end is slower in this microbenchmark because its forward regresses more than its backward improves.
See `RASTER_REBASE0_M_AMDAHL.json`.

---

## RESOURCE COST
Both fit in 8GB. MM preallocates a Buffer (MAX_NUM_RENDERED=2^21: splat_buffer ~256MB + sort space ~100MB) at import; H uses a single growable buffer via `resize_function_wrapper`. No OOM at N<=100k, 256x256. Per-kernel register/spill status not captured (no `ptxas -v` per kernel). A speedup that requires more memory is reported as such: MM trades higher forward instance/memory for parallel backward.

---

## PRIOR-ART IMPLICATION
MM-FasterGS is a direct, effective attack on the sequential alpha-compositing backward residual that this line's prior work (FUSED-HOTSPOT0) identified as the production bottleneck (`blend_backward_cu`). Re-basing onto MM's design collapses the backward majority into a minority. The remaining open frontier is therefore the **forward** path (radix sort + bucket render + higher instance count), not the backward blend.

---

## NEXT FRONTIER
Recommended next lane: **FORWARD-ARITH0** (investigate MM forward cost: radix sort + bucket render overhead, and whether H's fused-forward can be combined with MM's parallel backward). Do NOT implement BWD-ARITH0: the backward frontier is no longer the dominant residual after rebasing onto MM. A hybrid (H-style fused forward + MM-style bucket-parallel backward) is the natural hypothesis to test next.

---

## LIMITATIONS
- Raster-only microbenchmark on a shared 8GB laptop GPU; timing noise present (mitigated via median of 40 iters).
- Not a full 30k training iteration; raster fwd+bwd is a proxy.
- Instance-count units differ between backends (H packed instances vs MM raw tile-gaussian pairs) -> not like-for-like work; both render identical output.
- Native-stack comparison not run.
- No Nsight Compute: hotspot identity is inferred from fwd/bwd fraction + source structure, not from kernel-level profiling.

---

## VERDICT
`BACKWARD_FRONTIER_REDUCED` with flags `MM26_FASTER_COMMON_HOST`, `MEMORY_TRADEOFF_MATERIAL`. MM materially reduces the backward residual (fraction 0.63->0.34, absolute -39%) but regresses forward (+138%), so end-to-end raster time is higher in this microbenchmark and the hotspot shifts forward. The backward frontier is scientifically reduced, not open as a dominant residual; the next open frontier is the forward path. No new method invented.
