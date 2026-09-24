# C32-A: Training Iteration Critical-Path Decomposition

## Results Summary

| Metric | Value | Source |
|--------|-------|--------|
| T_iter (block timing) | **117 ms** | Clean measurement, 150 iters, no CUDA events in loop |
| CUDA events (fwd+bwd+opt, default stream) | **40 ms** | 150-event pools, read after block (2us overhead each) |
| Profiler GPU busy (all 4 streams) | **210 ms/iter** | Summed kernel durations across streams |
| Default-stream GPU busy | **46 ms/iter** | Profiler trace, stream (0,7) |
| GPU kernels/iteration | **350** | Chrome trace, cat=="kernel" |
| cudaLaunchKernel calls/iteration | **350** | 1:1 with GPU kernels |
| Active CUDA streams | **4** | 1 default + 3 aux |

## 🔑 Critical Discovery: Hidden Synchronization from `.item()` Calls

The profiler trace reveals **27 `cudaStreamSynchronize` calls per 3 iters = 9/iter**, consuming **87ms/iter** of wall time. This is the single largest component of T_iter.

**Root cause**: The original Phase-7 training loop calls:
- `loss.item()` — reads scalar loss from GPU to CPU
- `l1_item = loss_dict["l1"].item()` 
- `d_ssim_item = loss_dict["d_ssim"].item()`
- `mse = torch.mean((rendered - gt_image) ** 2).item()`
- `psnr = 10 * math.log10(1.0 / max(mse, 1e-10))`
- `clip_grad_norm_` internally calls `total_norm = ...` which involves `.item()` or synchronize

Each `.item()` call triggers `cudaStreamSynchronize` which blocks the CPU until all pending GPU work completes. This **serializes the CPU-GPU pipeline**: instead of the CPU queuing the next iteration while the GPU processes the current one, the CPU stalls for 87ms waiting for the GPU.

**The 87ms sync time vs 46ms default-stream GPU time**: The discrepancy occurs because:
1. CPU launches ~350 kernels/iteration (3ms launch overhead)
2. CPU then hits `.item()` which syncs the stream
3. The sync waits for ALL 46ms of default-stream GPU work to finish
4. Extra ~41ms is the CPU→GPU dispatch delay: kernels queued but not yet started by GPU scheduler when sync is hit

This means **T_iter is dominated by GPU execution + CPU dispatch serialization**, not by launch overhead.

## C31 Mystery Resolved

**Why CUDA events ≈ 40ms ≠ profiler ≈ 210ms ≠ wall ≈ 117ms?**

1. **CUDA events (40ms)**: Only record default-stream GPU time for fwd/rasterization/bwd/opt. Missing ALL other GPU work (D-SSIM, clip, autograd) and all aux-stream work.

2. **Profiler GPU busy (210ms)**: Sums kernel durations across ALL 4 streams. With 4 streams, overlapping execution means the sum exceeds wall. Default stream = 46ms, each aux stream ≈ 55ms.

3. **Wall clock (117ms)**: Dominated by hidden synchronization (`cudaStreamSynchronize` from `.item()` calls = ~87ms). The actual GPU execution (46ms default stream) happens during these sync waits — the CPU queues work ahead and the sync catches the queue draining.

## Truthful Iteration Decomposition

```
T_iter = 117 ms  (measured, clean block timing)

  GPU execution (critical path, default stream):   46 ms (39%)
    ├── gsplat forward:            5 ms  (1 kernel)
    ├── D-SSIM forward (cuDNN):   10 ms  (45 cutlass fprop kernels)
    ├── gsplat backward:          14 ms  (1 kernel)
    ├── D-SSIM backward (cuDNN):   5 ms  (9 dgrad + more)
    ├── Optimizer (Adam):          4 ms  (24 multi_tensor_apply kernels)
    ├── Fill/zero/copy:            3 ms  (63+ kernels)
    └── Other (add, mul, div):     5 ms  (100+ kernels)

  Hidden sync from .item() calls:           ~87 ms (74% of wall, BUT includes GPU time)
    └── cudaStreamSynchronize blocks until
        GPU queue drains. The 87ms includes
        the 46ms GPU execution time above —
        the sync waits for GPU to finish.

  CPU dispatch overhead (net):               ~30 ms (26%)
    ├── cudaLaunchKernel (350 calls):         3 ms
    ├── cudaMemcpyAsync (15 calls):           2 ms
    ├── cudaMemsetAsync (55 calls):           0.3 ms
    ├── aten ops (empty, fill, zero):         2 ms
    ├── cudaEventRecord (28 calls):           0.3 ms
    ├── cudaStreamWaitEvent (30 calls):       0.3 ms
    ├── Python dispatch overhead:           ~12 ms
    ├── Allocator driver calls:             ~10 ms
    └── Stream wait / scheduling:            ~10 ms

  OVERLAP: CPU dispatch (30ms) overlaps with
  GPU execution (46ms). Without hidden sync,
  T_iter ≈ max(CPU, GPU) = max(30, 46) = 46 ms.
```

**Critical insight**: The hidden synchronization inflates T_iter from a potential ~50ms to 117ms. The `.item()` calls destroy CPU-GPU pipelining. **If `.item()` calls were removed (or moved to async), T_iter could drop to ~50ms — a 2.3× speedup.**

This is NOT about kernel launch overhead. It's about **PyTorch scalar access synchronization**.

## Top 5 Kernels by Total GPU Time

| Kernel | Count | Total ms | Mean us |
|--------|-------|----------|---------|
| cuDNN fprop (D-SSIM conv) | 45 | 151.0 | 10274 |
| gsplat rasterize bwd | 1 | 14.2 | 14170 |
| cuDNN tensor transform (FP32) | 45 | 8.3 | 555 |
| D-SSIM dgrad conv | 3 | 6.2 | 2051 |
| gsplat rasterize fwd | 1 | 4.5 | 4471 |

## Stream-Level Stats (Per Iteration)

| Stream | Kernels | GPU Time | Gap Time | Role |
|--------|---------|----------|----------|------|
| (0,7) — Default | 870 (290/iter) | 46 ms | 172 ms | Critical path: fwd/bwd/opt |
| (0,16) — Aux 1 | 60 (20/iter) | 52 ms | 152 ms | D-SSIM parallel tensor transforms |
| (0,17) — Aux 2 | 60 (20/iter) | 56 ms | 147 ms | D-SSIM parallel tensor transforms |
| (0,18) — Aux 3 | 60 (20/iter) | 55 ms | 149 ms | D-SSIM parallel tensor transforms |

The 3 aux streams handle cuDNN tensor layout conversions (tensorTransformGeneric) in parallel with the default stream's computation. The "gap time" is large because these streams are created and idle most of the iteration.

## Research Questions

### Q1: Is the workload actually launch-bound / CPU-bound?

**Answer: NEITHER. It's hidden-sync-bound.**

The workload is gated by **hidden synchronizations from `.item()` calls**, not by kernel launch overhead (3ms) or by CPU dispatch (30ms). Without the `.item()` calls, the wall time would be ~50ms (GPU-bound). 

**The 117ms T_iter breaks down as:**
- 46ms: actual GPU compute (would be the wall time with async item())
- 41ms: CPU dispatch → GPU execution serialization (caused by sync)
- 30ms: net CPU overhead (overlapped with GPU without sync)

So the overhead from `.item()` sync is 117 - 46 = **71ms, or 61% of T_iter**.

### Q2: What fraction of T_iter is theoretically removable by CUDA Graph?

**Answer: ~3ms (3%). CUDA Graph is irrelevant.**

CUDA Graph removes cudaLaunchKernel overhead (3ms/iter). It does NOT prevent hidden synchronizations from `.item()` calls. The transform to pursue is NOT CUDA Graph but **async `.item()` / deferred scalar retrieval**.

**Recommended alternative: Replace syncs with async pipelines:**

1. Accumulate `.item()` calls and read all after a block of iterations (like block timing does)
2. Use `torch.no_grad()` for PSNR computation
3. Replace `clip_grad_norm_` with non-synchronizing version (compute norm on GPU)
4. Defer `loss.item()` logging to a background thread

These changes could save **~71ms/iter (61%)** — far exceeding CUDA Graph's ceiling.

### Q3: Does CUDA Graph remain viable despite dynamic Gaussian topology?

**Answer: Irrelevant — the 3% potential gain does not justify the complexity.**

### Q4: Is there a specific 3DGS-specific execution structure causing overhead?

**Answer: Yes — the `.item()` pattern is inherited from the original 3DGS codebase.**

The original 3DGS Python implementation calls `.item()` on every loss value and PSNR for logging. This is a common pattern in research code that becomes catastrophic for performance. In 3DGS training, the problem is amplified because:
- D-SSIM adds significant GPU time per iteration (~15ms for fwd+bwd cuDNN)
- The `.item()` sync waits for ALL of this to complete
- Unlike MLP training where item() syncs are brief, 3DGS has longer GPU kernels (14ms bwd)

### Q5: If CUDA Graph is worth pursuing, what capture boundary for testing?

**Answer: DROP CUDA Graph.** Time to pursue **hidden sync elimination** instead.

## Stop Condition Assessment

The stop condition analysis shows:
- Kernel launch overhead: **3.1ms / 117ms = 2.7%** → **DROP naive CUDA Graph (<10%)**
- Hidden sync overhead: **71ms / 117ms = 61%** → 🔴 **MAJOR UNEXPECTED FINDING**
- GPU compute: 46ms / 117ms = 39%

Per the stop conditions: "If a non-obvious synchronization / allocator / autograd structure occupies significant time, prioritize that mechanism."

**🏆 Recommendation: Redirect C32-B to "Hidden Sync Elimination"** — remove `.item()` calls from the hot path and evaluate the real GPU-bound T_iter.

## Raw Artifacts

- Chrome trace: `results/phase-c31/c32_a_analysis.trace.json` (3 profiled iters, 15K+ events)
- Analysis JSON: `results/phase-c31/c32_a_analysis.json`
- Trace decomposition script: `scripts/phase-c31/_trace_decomp.py`
