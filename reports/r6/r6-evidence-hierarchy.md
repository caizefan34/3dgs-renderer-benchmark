# R6 — Evidence Hierarchy

## Mandated hierarchy (highest to lowest)

| Level | Type | Description | When to use |
|-------|------|-------------|-------------|
| 1 | Direct implementation CUDA event | CUDA events placed inside the actual implementation, wrapping the exact operation being measured. No attribution ambiguity. | Gold standard. Used when we control the implementation. |
| 2 | Direct debug instrumentation | Counters or timers inserted into the actual kernel source code (e.g., `__device__` counter incremented at the exact code path). | When we can modify the kernel but cannot place CUDA events around a sub-operation. |
| 3 | torch.profiler attribution | Kernel-level timing from CUPTI, attributed by kernel name matching. Subject to name-matching errors and cross-pass contamination. | When we cannot modify the kernel. Must be validated against level 1/2 where possible. |
| 4 | Hardware throughput model | Analytical model using hardware specs (e.g., A100 L2 atomic throughput) and workload parameters (e.g., n_isects). Upper bound on time, not precise measurement. | For estimating costs that cannot be directly measured (e.g., atomic stall without ncu). |
| 5 | Metadata-derived oracle | Inference from forward-pass metadata (e.g., bounding-box footprints → warp count). Lowest quality — forward metadata may not predict backward behavior. | Only when no direct measurement is possible. Must be validated. |

## Evidence levels used in R6

### R6-B: Zero-init / touched-only gradient buffer

| Measurement | Evidence level | Value | Status |
|-------------|---------------|-------|--------|
| B0 T_clear (CUDA event) | **1** | 0.031–0.137 ms | **AUTHORITATIVE** |
| B1-v2 T_scat (CUDA event) | **1** | 0.129–0.267 ms | **AUTHORITATIVE** |
| B1-v2 T_clear (CUDA event) | **1** | 0.024–0.104 ms | **AUTHORITATIVE** |
| T_zero (torch.profiler) | 3 | 2.36–8.33 ms | **SUPERSEDED** — 30–60× overestimate |
| Conservative E2E (metadata oracle) | 5 | 1.6–4.3% | **SUPERSEDED** — based on wrong T_zero |
| Upper bound E2E (metadata oracle) | 5 | 2.3–6.4% | **SUPERSEDED** — based on wrong T_zero |

**Why level 1 supersedes level 3**: The B0 implementation wraps `cudaMemsetAsync`
in CUDA events directly inside the patched `Rasterization.cpp`. This measures
exactly the gradient buffer zero-fill — no DSSIM intermediates, no autograd
bookkeeping, no cross-pass contamination. The torch.profiler "memset_zero"
category matched by kernel name (`"memset" or "fill"`) captured all memset/fill
operations during backward, most of which are not gradient buffer zero-init.

### R6-A: Block-level gradient aggregation

| Measurement | Evidence level | Value | Status |
|-------------|---------------|-------|--------|
| actual_atomic (debug counter) | **2** | TBD (pending kernel build) | **AUTHORITATIVE** (if build succeeds) |
| R_atomic (pixel-level simulation) | 2 (proxy) | 2.07–2.86 | Replaced by direct counter |
| R_atomic (footprint estimate) | 5 | 5.50–7.61 | **SUPERSEDED** — 2.6× overestimate |
| Conservative E2E (HW throughput model) | 4 | 0.68–1.58% | Depends on R_atomic input |
| T_raster%T_bwd (torch.profiler) | 3 | 41–59% | Validated by CUDA event T_bwd |

**Why level 2 supersedes level 5**: The debug counter increments at the exact
warp-leader atomic execution point (`if (warp.thread_rank() == 0)`), counting
every actual atomicAdd set. The footprint estimate inferred warp multiplicity
from bounding-box radii, which overestimates because Gaussian alpha falls off
exponentially — most pixels in the bounding box have alpha below threshold.

**Why the pixel simulation (level 2 proxy) is intermediate**: The pixel
simulation evaluates alpha at each pixel using forward metadata (means2d,
conics, opacities) and maps to warps. It is a faithful simulation of the
kernel's decision logic, but uses a different alpha threshold (0.01 vs the
kernel's 1/255 ≈ 0.00392) and samples only 200 tiles. The debug counter is
strictly higher quality — it counts in the actual kernel with the actual
threshold on all tiles.

### R6-C: Backward-optimizer fusion

| Measurement | Evidence level | Value | Status |
|-------------|---------------|-------|--------|
| T_optimizer (CUDA event) | **1** | 2.54–13.76 ms | **AUTHORITATIVE** |
| Gradient traffic bytes (metadata) | 5 | 132–934 MB | Direct from tensor shapes |
| T_saved (bandwidth model) | 4 | 0.265–1.868 ms | Depends on bandwidth assumption |
| Conservative E2E | 4 | 0.48–2.21% | **AUTHORITATIVE** (repaired) |
| Original E2E (broken model) | — | 4.8–17.2% | **SUPERSEDED** — T_opt + traffic (adds instead of saves) |

**Why level 4 is acceptable for R6-C**: The fusion savings are purely
bandwidth-limited (eliminating gradient write + read = 2 × grad_bytes). The
A100 memory bandwidth is a well-characterized hardware spec. The model is
simple and physically grounded: `T_saved = 2 × grad_bytes / bandwidth_achieved`.
No kernel attribution is needed.

### R6-1: Backward decomposition

| Measurement | Evidence level | Value | Status |
|-------------|---------------|-------|--------|
| T_bwd (CUDA event) | **1** | 20–83 ms | **AUTHORITATIVE** |
| T_iter (CUDA event) | **1** | 54–158 ms | **AUTHORITATIVE** |
| T_raster (torch.profiler) | 3 | 11–43 ms | Validated by T_bwd consistency |
| T_zero (torch.profiler) | 3 | 2.36–8.33 ms | **SUPERSEDED** by B0 direct |
| T_sh, T_proj (torch.profiler) | 3 | <2% T_bwd | Negligible, not contested |
| T_other / T_DSSIM (torch.profiler) | 3 | 34–44% T_bwd | Not addressable by R6 |

## Cross-validation summary

| Claim | Level 3/4/5 estimate | Level 1/2 measurement | Discrepancy | Resolution |
|-------|---------------------|----------------------|-------------|------------|
| T_zero % T_bwd | 5.5–11.6% (L3) | 0.14–0.36% (L1) | 30–60× | L1 wins — profiler attribution error |
| R_atomic | 5.5–7.6 (L5) | 2.1–2.9 (L2 proxy) | 2.6× | L2 wins — footprint overestimate |
| R_atomic (direct) | TBD | TBD (L2 direct) | TBD | Pending kernel build |
| R6-C E2E | 4.8–17.2% (broken) | 0.48–2.21% (L4 repaired) | 7.8× | L4 wins — original model was physically impossible |

## Principle

When evidence levels conflict, the higher-quality (lower-numbered) evidence
wins. The old claim is not preserved merely because it motivated the
optimization. Direct measurement supersedes attribution, attribution
supersedes modeling, and modeling supersedes metadata inference.
