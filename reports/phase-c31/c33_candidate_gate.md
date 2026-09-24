# C33 Candidate Gate 0/1/2 — Final Report

## Track A: Host CUDA Metadata Query Overhead — **DROP**

### Pre-registration (from C32-A trace)
| CUDA API Call | Calls/iter | CPU Time/iter | Avg latency |
|---|---|---|---|
| cudaDeviceGetAttribute | 186 | 0.057ms | 0.3μs |
| cudaFuncGetAttributes | 45 | 0.096ms | 2.1μs |
| cudaOccupancyMaxActiveBlocks | 51 | 0.040ms | 0.8μs |
| cudaFuncSetAttribute | 17 | 0.023ms | 1.3μs |
| cudaPeekAtLastError | 61 | 0.010ms | 0.2μs |
| cudaStreamGetPriority | 8 | 0.007ms | 0.6μs |
| **Total metadata queries** | **368** | **0.233ms** | — |
| All other CUDA API (launch+memcpy+memset) | 420 | 5.595ms | — |

### Experiment: Cold vs Hot Block Timing
- **Cold** (no warmup, 10 iters): T_iter = **205.8ms**
- **Hot** (50 warmup + 100 measured): T_iter = **103.3ms**
- Difference: **102.5ms (99%)** — but this is **caching allocator behavior**, not metadata queries.

The cold run performed 15 `cudaMalloc` calls (allocating 1.1GB of GPU memory for the first time). The hot run performed 0 `cudaMalloc` calls (PyTorch caching allocator reused all memory from warmup).

### Verdict
- Metadata queries total **0.23ms/iter** (0.22% of 103ms T_iter)
- Even if cached to **zero latency**, speedup = 0.22%
- **STOP condition: <5% → DROP**
- The "57ms CPU overhead" from C32-B is NOT from metadata queries. It is from:
  - Autograd engine scheduling (350 ops → trace creates autograd graph each iter)
  - Python dispatch overhead
  - PyTorch caching allocator internal bookkeeping
  - Set-to-none zero_grad + tensor shape propagation

---

## Track D: Iteration-to-Iteration Workload Predictability — **DROP**

### Results summary (933 iters, 3 full camera cycles, room scene)

#### D1: W_t vs W_(t+1) Correlation
| Metric | Pearson r | Spearman ρ |
|--------|-----------|------------|
| n_visible | **0.912** | **0.903** |
| n_intersections | **0.891** | **0.899** |
| fwd_ms | 0.054 | **0.900** |
| bwd_ms | 0.508 | **0.888** |
| total_render_ms | 0.113 | **0.894** |

Forward render time has **near-zero Pearson correlation** with previous iter (r=0.054), despite high Spearman rank correlation. This means:
- The MONOTONIC ordering is preserved (cold→warm→stable)
- But the **magnitude** is unpredictable due to cold-start outlier (604ms fwd on iter 0, then stable at 4-6ms)

#### D2: Autocorrelation
| Metric | lag-1 | lag-2 | lag-3 | lag-5 |
|--------|-------|-------|-------|-------|
| n_visible | **0.912** | **0.847** | **0.789** | **0.672** |
| n_intersections | **0.892** | **0.799** | **0.728** | **0.574** |
| total_render_ms | **0.012** | 0.011 | 0.008 | 0.011 |

**Critical finding**: Render time has **zero autocorrelation** at all lags. Workload count metrics are highly correlated, but render time is NOT.

#### D3: Camera vs Training-State Locality
| Predictor | Median error | P90 error |
|-----------|-------------|-----------|
| Same camera (cycle 1 → cycle 2) | **3,634 Gs** | **8,902 Gs** |
| Adjacent camera | 36,834 Gs | — |
| Random camera | 331,773 Gs | — |
| **Same/adjacent ratio: 0.10** | | |

**Camera locality dominates**. Same camera's workload across cycles is 10x more predictable than the next camera in the same cycle. This means:
- Workload is determined by **which part of the scene the camera sees**, not by training state
- A camera-based predictor would be highly accurate
- But EMA/last-value are poor predictors because camera transitions dominate the sequence

#### D4: Predictor Comparison (visible GS count)
| Predictor | Mean error | Median error | P90 error | Relative (median) |
|-----------|-----------|-------------|-----------|-------------------|
| **Camera-mean (cycle 1 → cycle 2)** | **4,484** | **3,634** | **8,902** | **~0.5%** |
| Last-value (naive) | 71,152 | 34,062 | 173,114 | 6.1% |
| EMA(α=0.3) | 104,967 | 78,586 | 232,742 | 10.5% |

**Camera-mean is 10x better than last-value.** But this doesn't help because...

#### D5: Does workload variation affect render time?

**No.** After warmup (iter > 1):
- fwd_ms: flat at **4-6ms** regardless of visible count (119K to 1.15M)
- bwd_ms: flat at **11-18ms**, only r=0.52 with n_visible
- total_render: 17-20ms regardless of 10x workload variation

The n_visible→fwd_ms correlation is **r=0.046** (Pearson). Visible GS count explains **<0.2% of forward render time variance**.

#### D6: Topology Breakpoints
- Densification at steps 500, 600, 700, 800, 900
- Median post-denf error: comparable to global median
- **No significant breakpoint effect** — topology changes ~770 Gs change per event on 1.59M base

### Key Question Answers

**Q1: Is next-iteration workload predictable?**
YES for visible GS count (Pearson r=0.91). NO for render time (autocorrelation ≈ 0).

**Q2: Prediction error?**
Camera-mean: 3,634 Gs median (0.5%). Last-value: 34,062 Gs median (6.1%).

**Q3: Camera or training-state locality?**
**Camera locality dominates.** Same camera across cycles is 10x more predictable than adjacent camera.

**Q4: Can prediction support pre-allocation/configuration?**
**No.** Render time is insensitive to workload variation. Forward time plateau at 4-6ms regardless of 10x visible-GS variation. The renderer's bottleneck is fixed overhead (kernel launch, allocator), not data-dependent work.

**Q5: Topology breakpoints invalidating predictor?**
No significant breakpoint effect. The data-independent bottleneck dominates.

**Q6: Prediction horizon?**
Camera-mean works across entire training (cycle 1→cycle 2). Last-value degrades fast (4-step: 17.3% error).

### Stop Condition Assessment
- **STOP condition**: If simple last-value predictor error is large → DROP.
- **Practical verdict**: Workload is predictable, but **render time is NOT workload-sensitive**. Prediction cannot improve execution because the bottleneck (~83ms of T_iter) is fixed overhead unaffected by workload.

---

## Final Candidate Verdict

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| **C33-A**: Host CUDA Query Overhead | **DROP** | 0.23ms/iter (0.22%). 100x below 5% threshold |
| **C33-B**: Parameter-group update cadence | **DROP** | Already frozen per protocol |
| **C33-C**: Gaussian birth scheduling | **DROP** | Already frozen per protocol |
| **C33-D**: Workload prediction | **DROP** | Workload IS predictable (camera-based, 0.5% error) but render time is NOT workload-sensitive. No exploitable coupling. |
| **C33-E**: Renderer policy switching | **DROP** | Already frozen per protocol |
| **C33-F**: Training/rendering co-design | **DROP** | Already frozen per protocol |

### Why Track D drops despite strong predictability:
1. Forward render time is **flat at 4-6ms** for 119K-1.15M visible Gs — the renderer is not GS-count-bound
2. total_render autocorrelation ≈ 0 at all lags — render time is a **memory/allocator-bound constant**, not a workload function
3. Even perfect workload prediction cannot improve a bottleneck that is **insensitive to workload**

### The remaining ~83ms bottleneck (after removing render's 18ms)
This is the **unidentified fixed overhead** per iteration. Candidate sources:
- Autograd engine graph construction (350 ops → autograd creates and caches graph each iter)
- PyTorch TensorImpl/ScalarType dispatch overhead
- Python-to-C++ boundary crossing (350 ops × function call overhead)
- C10 observer/callback hooks

**This is NOT a CUDA-level problem.** It's a PyTorch-level dispatch overhead problem. CUDA-level analysis (Track A) and workload prediction (Track D) cannot address this.

## Raw Artifacts
- Track A: `results/phase-c31/c33_a_query_overhead.json`
- Track D: `results/phase-c31/c33_d_workload_data.json`
- Track D analysis: `scripts/phase-c31/c33_d_analyze.py`
- Track A script: `scripts/phase-c31/c33_a_query_overhead.py`
- Track D script: `scripts/phase-c31/c33_d_workload_obs.py`
- Launcher: `scripts/phase-c31/c33_launcher.sh`
- Launcher log: `logs/phase-c31/c33_launcher.log` (mx)
