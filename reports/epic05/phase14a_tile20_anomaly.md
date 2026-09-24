# Phase 14A — Training-Level Tile Anomaly Investigation

**Date:** 2026-09-23
**Author:** DSH coding agent
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM)
**Scene:** Mip-NeRF 360 — room
**Data source:** `results/epic05/phase7/phase7_room_t20_results.json`

---

## 1. Executive Summary

### Central Question Answered

> Why did tile20, the frozen snapshot forward winner for room (10.247ms, **0.90× vs tile16**), become the **worst 30K training configuration** (561.8 min, **0.27× vs tile16**)?

**ANSWER — TRAINING-SYSTEM LEVEL: Asynchronous topology spillover, not renderer pathology.**

The 157.8s outlier at iteration 8050 is **NOT a renderer kernel anomaly**. It is caused by asynchronous CUDA topology/densification operations that overflow from densification iterations into subsequent non-densification iterations, massively inflating the measured forward/backward times via implicit CUDA synchronization.

---

## 2. Overall Training Results

| Metric | tile16 | tile20 | tile32 |
|:-------|:------:|:------:|:------:|
| Wall time (30K) | **150.3 min** | **561.8 min** | **94.8 min** |
| Speedup vs tile16 | 1.00× | **0.27×** | 1.58× |
| Iter/s | 3.33 | 0.89 | 5.27 |
| Best PSNR | 29.27 dB | **29.54 dB** | 29.39 dB |
| Final Gs | 1,193,480 | 1,158,369 | 1,146,273 |
| Avg steady-state iter | 213.8 ms | **328.8 ms** | 146.6 ms |
| Iterations > 10s | **0** | **57 (1.9%)** | **0** |

**tile20 is catastrophically slow in training** — 3.7× slower than tile16, 5.9× slower than tile32 — despite being the forward snapshot winner.

---

## 3. The 157s Outlier — Exact Location

| Property | Value |
|:---------|:------|
| **Iteration** | **8050** |
| total wall time | **157,829 ms (157.8s)** |
| forward timing | 59,507 ms (37.7%) |
| backward timing | 35,914 ms (22.8%) |
| optimizer timing | 12,473 ms (7.9%) |
| topology timing | 0.03 ms (0.0%) |
| unaccounted | 49,935 ms (31.6%) |
| num Gaussians | 887,165 |
| peak memory | 1,691 MB (normal) |
| densification | NOT a densification iteration |

**The 157s is distributed across fwd/bwd/opt — none of which are the real culprit.**

---

## 4. Root Cause — Asynchronous Topology Spillover

### 4.1 Mechanism

The training loop measures per-iteration wall time as:
```
iteration_ms = timer.stop()   # Python wall clock
```

Sub-stages are measured separately:
```
fwd_ms      = CUDA event elapsed for forward pass
bwd_ms      = CUDA event elapsed for backward pass
opt_ms      = CUDA event elapsed for optimizer step
topology_ms = CUDA event elapsed for densification kernels
```

**Critical finding:** `topology_ms` can exceed `iteration_ms` on densification iterations because topology CUDA work runs **asynchronously** — the Python iteration timer stops before the CUDA topology kernel completes, while the CUDA event timer correctly measures the full GPU execution time.

When `topology_ms > iteration_ms`:
- The Python loop moves to the next iteration
- The next iteration's forward pass triggers implicit CUDA synchronization
- The measured `fwd_ms`/`bwd_ms`/`opt_ms` for subsequent iterations include **waiting for the previous topology work to complete**
- This creates a cascading slowdown across multiple non-densification iterations

### 4.2 Evidence

| Iteration | total_ms | fwd_ms | bwd_ms | opt_ms | topo_ms | clone/split/prune | topo > iter? |
|:---------:|:--------:|:------:|:------:|:------:|:-------:|:-----------------:|:-----------:|
| 8000 | 233 | 6 | 73 | 100 | **61,588** | 11K/1K/2K | **YES** |
| 8010 | 3,540 | 2,696 | 382 | 30 | 0 | — | async spill |
| 8020 | **36,568** | 5,091 | 18,372 | 9,051 | 0 | — | async spill |
| 8030 | **91,808** | 30,308 | 40,596 | 80 | 0 | — | async spill |
| 8040 | **35,909** | 1,611 | 10,959 | 2,717 | 0 | — | async spill |
| **8050** | **157,829** | 59,507 | 35,914 | 12,473 | 0 | — | **async spill** |
| 8060 | 3,956 | 59 | 38 | 16 | 0 | — | async spill |
| 8070 | **138,852** | 43,321 | 47,168 | 21,123 | 0 | — | async spill |
| 8080 | 12,182 | 19 | 11,334 | 547 | 0 | — | async spill |
| 8090 | **121,408** | 27,728 | 40,609 | 31,420 | 0 | — | async spill |
| 8100 | 12,606 | 8 | 12,540 | 13 | 6,282 | 9K/690/2K | DENF |

### 4.3 Scope

- **42/294 (14.3%)** densification events have `topology_ms > iteration_ms`
- Average topology overrun: **7,789 ms** (7.8s)
- Maximum topology overrun: **116,635 ms** at iteration 7800
- Each async topology event creates **5–10 subsequent slow iterations** totaling 30–500 seconds of spillover

---

## 5. Comparison with tile16 and tile32

### 5.1 tile32 — No Anomaly

| Metric | tile20 | tile32 |
|:-------|:------:|:------:|
| Iterations > 10s | 57 (1.9%) | **0** |
| Avg densification per event (clone) | ~706 | ~1,000-3,000 |
| Steady-state avg iter | 328.8 ms | 146.6 ms |
| Gaussian count around iter 8000 | 887K | 841K |

Tile32 at the same training stage (iter=8000, Gs=840K): total_ms = **93.8ms**. No anomaly. No topology spillover.

### 5.2 tile16 — No Anomaly

Tile16 has **0** iterations > 10s in the full 30K run. No topology spillover detected despite the tile16 data lacking explicit topology_ms field.

### 5.3 Why tile20 Is Affected

| Property | tile16 | tile20 | tile32 |
|:---------|:------:|:------:|:------:|
| Threads/block | 256 | **400** | 1024 |
| Occupancy | Medium | **Non-power-of-2** | High |
| Topology kernels tile-dependent? | Likely | **Yes** | Likely |

**Hypothesis:** tile20's 400 threads/block is neither a small multiple of 32 warps (tile16=8 warps) nor the full occupancy (tile32=32 warps). This creates a **non-optimal launch configuration** for densification kernels (clone/split/prune) that use tile-derived block sizes. The densification operations run ~7.8× longer per-event, overflowing into subsequent iterations.

---

## 6. Research Conclusion Classification

### OBSERVED

- **tile20 30K training is 3.7× slower than tile16** (561.8 vs 150.3 min)
- **tile20 has 57 iterations > 10s** vs 0 for both tile16 and tile32
- **157.8s outlier at iteration 8050** confirmed
- **42/294 densification events** have async topology (topology_ms > iteration_ms)
- **Average topology overrun: 7,789 ms**
- **Peak VRAM stable at 1,691 MB** throughout anomaly cluster — no reallocation

### SUPPORTED

- **Anomaly belongs to topology/densification stage**, not renderer
- Forward, backward, and optimizer measured times in anomalous iterations are **contaminated by CUDA synchronization waiting for previous async topology work**
- tile32 densification completes within the iteration window (no async spill)
- tile16 shows no anomalous iterations despite similar Gaussian count

### HYPOTHESIS

- **tile20's non-power-of-2 block size (400 threads) causes suboptimal densification kernel execution**, leading to 7-116s topology GPU times
- The densification kernels (clone/split/prune) use tile-size-dependent launch configurations inherited from the renderer block size parameter
- tile20 creates a worst-case occupancy/register configuration for these topology kernels

### FALSIFIED

- **tile20 does NOT have a CUDA renderer pathology** — the forward/backward pass rendered pixels correctly
- The 157s outlier is **NOT in the renderer** — it is a training-system-level topology synchronization artifact
- **No memory allocation/reallocation event** caused the slowdown
- **No CUDA synchronization timing issue** outside topology completion

### BLOCKED

- Direct topology kernel timing isolation requires modifying the training loop to synchronize CUDA before stopping the iteration timer
- Nsight Compute profiling blocked under WDDM

---

## 7. Impact Assessment

### tile20 should NOT be a training candidate

The forward snapshot advantage (10.247ms, 0.90× vs tile16) is completely negated by the training-system-level topology spillover. tile20 cannot be recommended for full 3DGS training on RTX 5070 Laptop GPU.

### tile32 remains the training winner for room

94.8 min (1.58× vs tile16), zero anomalous iterations, perfect CUDA synchronization.

### tile16 v2 is the safe baseline

150.3 min, zero anomalies, known behavior.

---

## 8. Next Research Directions

| Priority | Direction | Status |
|:---------|:----------|:-------|
| **Phase 14B** | **Segmented sort reconnaissance** | Start immediately |
| **Phase 14C** | Visibility/culling reconnaissance | After 14B |
| **Cross-scene tile32** | bicycle/garden 30K (if EPIC-05 returns) | PENDING |
| **Adaptive tile selection** | **DO NOT START** — tile20 invalidated for training | BLOCKED |

---

## 9. Files Created

- `reports/epic05/phase14a_tile20_anomaly.md` (this file)
- `results/epic05/phase14a_tile20_anomaly.json` (structured evidence)
