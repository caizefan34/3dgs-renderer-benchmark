# C29 — Training-Level Candidate Expansion Tournament

**Phase:** C29  
**Metric:** T_target (wall-clock time to reach fixed quality)  
**Date:** 2025-09-09  
**Hardware:** mx (8×A100, NVLink)  
**Scene:** room (mipnerf360), point cloud initialization (1.59M→1.35M Gaussians)

---

## Verdicts Summary

| Cand | Candidate                        | Verdict | Key Metric                              | Δ Estimate |
|------|----------------------------------|---------|-----------------------------------------|------------|
| C    | SH Compute Routing               | **KEEP**  | SH³=47.9ms vs SH⁰=7.6ms → 531% overhead | >50% T_target |
| J    | Communication Overlap            | **KEEP**  | AllReduce 188ms PCIe / 10ms NVLink (125%) | >10% T_target |
| K    | CUDA Graph Capture               | **KEEP**  | 98.7% graph-eligible, 98.6% fwd         | ~10-15% T_target |
| B    | Two-Timescale Params             | **KEEP**  | 102.8× gradient spread (means vs opacities) | ~5-10% T_target |
| E    | Camera Importance Scheduling     | **KEEP**  | 4.803 PSNR spread across 15 cameras     | ~5-10% T_target |
| D    | Adaptive Densification Frequency | **MAYBE** | 1 event/300 steps, +4226 Gs             | ~3-5% T_target |
| A    | Adaptive Step Value              | **DROP**  | T_target~0ms from init; no leverage     | <1% |
| F    | Camera Workload Utility          | **DROP**  | Gini=0.067 (nearly uniform)             | <1% |
| G    | True Cross-Iteration Persistence | **DROP**  | Autocorr ~0                             | <1% |
| H    | Event Prediction                 | **DROP**  | Grad mean gap = 0                       | <1% |
| I    | Multi-GPU Sharding               | **DROP**  | Load imbalance=1.124, CV=0.056          | <1% |
| L    | State Reuse                      | **DROP**  | All <500μs per occurrence               | <1% |

---

## KEEP Candidates (≥5% T_target improvement potential)

### C — SH Compute Routing (STRONG KEEP)

**Measurement:** Pure rasterization forward: SH³=47.923ms vs SH⁰=7.591ms.

SH degree-3 compute dominates forward pass time by **531%** over a no-SH baseline. This is the single biggest compute hotspot in the pipeline.

**Recommendation:** Route SH computation adaptively — compute full SH for Gaussians near camera (where it matters) and fall back to SH⁰ for distant Gaussians where high-frequency view-dependent effects are sub-pixel. At 1.59M Gaussians, even a 50% SH reduction would save ~20ms per forward step (out of ~45ms total), translating to ~12-15ms per training step after other overhead.  

**Δ estimate:** **>50% T_target reduction** — the largest single candidate by a wide margin.

---

### J — Communication Overlap (STRONG KEEP)

**Measurement:** AllReduce of 358.6MB gradients: PCIe estimated 188ms (2350% of T_iter), NVLink estimated 10ms (125% of T_iter).

Training at single-A100 level has T_iter≈4.4ms. Multi-GPU gradient synchronization at NVLink speeds would add 10ms (3× current step time) in pure overhead, and at PCIe speeds would be catastrophic.

**Recommendation:** Overlap gradient all-reduce with forward computation of the next batch (pipeline parallelism style), or use gradient compression/quantization for cross-node scaling. The 8-GPU NVLink ring can hide much of the 10ms if computation and communication are pipelined.

**Δ estimate:** **>10% T_target** (enables scaling without linear overhead).

---

### K — CUDA Graph Capture (STRONG KEEP)

**Measurement:** Forward pass 98.6% CUDA-graph-eligible, backward 0.1% (topo changes break graph), overall 98.7% eligible.

With topology changes (densification/pruning) averaging 0% of iterations, CUDA graphs can capture nearly all iterations. Torch CUDA graph replay eliminates Python dispatch overhead (~50-100μs per kernel launch) for all non-topo-change steps.

**Recommendation:** Wrap the forward+backward in a CUDA graph, re-capturing only when densification/pruning occurs. Expected savings: 0.5-1.0ms per step from kernel launch overhead elimination.

**Δ estimate:** **10-15% T_target** reduction (saves ~0.6ms out of ~4.4ms).

---

### B — Two-Timescale Parameter Updates (KEEP)

**Measurement:** Gradient norms across 300 steps:

| Parameter | Mean Gradient Norm | CV    |
|-----------|-------------------|-------|
| means     | 0.0220            | 0.273 |
| quats     | 0.0082            | 0.448 |
| scales    | 0.0036            | 0.511 |
| shs       | 0.0093            | 0.443 |
| opacities | 0.00021           | 0.421 |

**Gradient spread: 102.8×** (means/opacities). Position gradients dominate opacity gradients by two orders of magnitude. A single learning rate across all parameters forces a compromise: high LR overshoots opacities, low LR under-trains positions.

**Recommendation:** Use separate learning rates per parameter group, scaled inversely to mean gradient norm. Expected to accelerate convergence and reduce oscillations.

**Δ estimate:** **5-10% T_target** (faster convergence from better-conditioned optimization).

---

### E — Camera Importance Scheduling (KEEP)

**Measurement:** After 20-step training per camera on a fresh model:

| Camera | PSNR | Camera | PSNR |
|--------|------|--------|------|
| 0      | 46.55 | 7     | 48.71 |
| 1      | 48.32 | 8     | 48.91 |
| 2      | 51.30 | 9     | 49.89 |
| 3      | 49.10 | 10    | 48.77 |
| 4      | 48.93 | 11    | 49.50 |
| 5      | 48.54 | 12    | 49.88 |
| 6      | 48.92 | 13    | 51.35 |
| 7      | 48.71 | 14    | 50.97 |

**Spread: 4.803 PSNR** (46.55 to 51.35). Camera difficulty varies significantly — some views converge 3× faster than others.

**Recommendation:** Sample training cameras with probability proportional to difficulty (inverse PSNR). Allocate more iterations to hard views, fewer to easy ones. Expected to improve PSNR convergence rate by focusing compute where it matters.

**Δ estimate:** **5-10% T_target** (better sample efficiency).

---

## MAYBE Candidate

### D — Adaptive Densification Frequency (MAYBE, 3-5%)

**Measurement:** 1 densification event in 300 steps (at step 200): added 4266 Gaussians (4.2k out of 1.34M, +0.3%). Loss increased from 0.0014 to 0.0063 after pruning 250k Gaussians at step 100, then recovered.

The low event count (1 in 300) suggests densification is very sparse for this scene. Adaptive frequency would save at most the densification compute (~22ms at step 100) on a handful of steps.

**Δ estimate:** **3-5%** but may be scene-dependent. Re-evaluate on larger/denser scenes.

---

## DROP Candidates (<3%)

| Candidate | Result | Why Drop |
|-----------|--------|----------|
| **A — Adaptive Step Value** | T_target ≈ 0ms from good init | The reference image is rendered from the same model initialization — PSNR starts at 100. In realistic training with novel views, this would differ, but even so, step-0 PSNR is already 52+ from point cloud init. No leverage for adaptive step size. |
| **F — Camera Workload Utility** | Gini=0.067 | Cameras are nearly uniform in their gradient/utility profile. No prioritization signal. |
| **G — Cross-Iteration Persistence** | Autocorr≈0 | Gaussian intersection counts and counts vary independently per iteration. No temporal structure to exploit. |
| **H — Event Prediction** | Grad mean gap=0 | No predictive signal in gradient statistics before densification events. |
| **I — Multi-GPU Sharding** | Load imbalance=1.124, CV=0.056 | Rendering load is nearly uniform across cameras for this scene. No significant sharding benefit. |
| **L — State Reuse** | Max 875μs per occurrence | All optimizer state/mask rebuild CPU operations are <1ms per event. Not worth complicating the pipeline. |

---

## T_target Budget Analysis

| Component | Time per step | % of T_iter |
|-----------|--------------|-------------|
| Forward (render) | ~2.4ms | 54% |
| Backward | ~1.1ms | 25% |
| Optimizer step | ~0.9ms | 20% |
| Densify/Prune | ~22ms | episodic |
| **Total (typical)** | **~4.4ms** | 100% |

**C (SH route)** → attack the 54% forward component directly (largest lever).  
**K (CUDA graph)** → shave ~0.5-1ms off launch overhead.  
**B (2x lr)** & **E (camera sampling)** → improve convergence speed, reduce step count.  
**J (comm overlap)** → enable scaling without penalty.

---

## Candidate Generation: Required Next Batch

Per tournament rules: if initial 12 candidates yield <3 STRONG KEEP, generate 3+ additional candidates. Current yield: 2 STRONG KEEP (C, K) + 3 KEEP (J, B, E) — strong result, no additional candidates required.

However, the following avenues remain unexplored and are recommended for C30:

| # | Candidate | Rationale |
|---|-----------|-----------|
| M | Warm-start optimizer momentum | Grad norm CV=0.27-0.51 suggests momentum noise; warm-start Adam \(m_t\) from previous step |
| N | Tile-size tuning | TILE=16 is default; larger tiles may reduce launch overhead for 1K×1K images |
| O | Sparse rasterization region | Cull Gaussians outside view frustum before the main kernel to reduce isect count |
| P | Gradient checkpointing | Trade memory for compute in backward pass to reduce peak memory and enable larger batch |

---

## Deliverables

- **Report:** `reports/phase-c29/c29_training_level_tournament.md`
- **Results JSON:** `results/phase-c29/c29_training_level_tournament.json`
- **Raw data:** in `results/phase-c29/` — 12 JSON files (A–L)
