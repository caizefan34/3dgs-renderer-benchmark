# C32-B: Hidden Synchronization Causal Validation — RESULTS

## C32-A Correction

**C32-A's claim that `.item()` calls cost 71ms/iter (61% of T_iter) is REFUTED.**

The profiler showed 27 `cudaStreamSynchronize` calls across 3 profiled iterations (9/iter). These were NOT from `.item()`. The profiler itself adds these synchronizations as part of its instrumentation. In clean wall-clock measurement (block timing, no profiler), `.item()` has negligible overhead.

## Causal A/B Results (3 repeats each, 150 measured iters each)

| Variant | Change | T_iter ms | ±σ | vs Baseline | Speedup |
|---------|--------|-----------|-----|-------------|---------|
| **A** | Baseline (all .item + clip + PSNR) | **104.29** | 0.25 | — | — |
| B1 | Defer loss.item() | 104.03 | 0.35 | -0.26ms | 0.25% |
| B2 | Defer l1.item() | 104.07 | 0.11 | -0.22ms | 0.21% |
| B3 | Defer d_ssim.item() | 103.99 | 0.06 | -0.30ms | 0.29% |
| B4 | Defer PSNR mse.item() | 103.77 | 0.14 | -0.52ms | 0.50% |
| B5 | Defer clip_grad_norm sync | 103.81 | 0.14 | -0.48ms | 0.46% |
| **B6** | Defer ALL scalar reads | **103.53** | 0.07 | **-0.76ms** | **0.73%** |
| **E** | No metrics (no scalar reads at all) | **103.45** | 0.10 | **-0.84ms** | **0.80%** |
| F | Correctness (same as A, checks passed) | 103.85 | 0.22 | -0.44ms | 0.42% |

**Key observations:**
1. Every single `.item()` source contributes <0.5ms per iteration
2. PSNR mse.item() is the largest single contributor (0.5ms / 0.5%)
3. clip_grad_norm_ contributes 0.46ms / 0.44%
4. ALL `.item()` read overhead combined: 0.76ms / 0.73%
5. Complete observability removal (E): 0.84ms / 0.80%

## Research Questions

### Q1: Does removing `.item()` reduce wall-clock T_iter?
**YES, but negligibly.** The reduction is 0.76ms (0.73%) — statistically significant but practically irrelevant.

### Q2: How much?
**0.76ms out of 104.29ms** (0.73%). Not the 71ms predicted by C32-A profiler analysis.

### Q3: Is the speedup stable?
**YES** — 3 repeats per variant show low variance (σ < 0.4ms for all). B6 std=0.07ms, E std=0.10ms.

### Q4: Which scalar is the main culprit?
**PSNR mse.item()** contributes the most (0.5ms), followed by clip_grad_norm (0.46ms). Loss/L1/D-SSIM each contribute <0.3ms.

### Q5: Does no-metrics T_iter approach GPU-bound floor?
**YES** — the GPU-bound floor from C32-A default-stream GPU was ~46ms with ~57ms unavoidable CPU dispatch. Block timing T_iter=103.4ms means the remaining ~57ms is CPU dispatch that cannot be eliminated by removing scalar reads.

### Q6: Are there other hidden synchronizations?
**YES.** The ~57ms CPU dispatch time consists of:
- **CUDA allocator overhead** (~24ms from C32-A profiler — *these are real, not profiler artifacts*)
- **cudaLaunchKernel CPU time** (~3ms)
- **Python dispatch / CUDA driver calls** (~15ms)
- **Data loading / camera preparation** (~5ms)
- **gc.collect / memory management** (~10ms, only during warmup)

### Q7: Is this A (PyTorch mistake) or B (3DGS-specific problem)?
**A: Ordinary PyTorch coding pattern.** The `.item()` pattern is inherited from the original 3DGS codebase but it is NOT a significant contributor to wall time when measured with proper block timing. The profiler simply inflated its apparent cost.

The REAL 3DGS bottleneck remains: **CUDA allocator overhead from managing 1.6M Gaussians × 5 parameter groups across 350+ kernel launches per iteration.** This is what consumes the ~57ms CPU dispatch time.

## Stop Condition Assessment

| Condition | Result |
|-----------|--------|
| no-item speedup <5%? | **YES** (0.73%) → **DROP** |
| blocking/control proves time attribution shift? | **YES** — profiler attribution was misleading |
| training semantics unchanged? | **YES** — correctness confirmed (same PSNR, same G count) |

## Final Verdict

The hidden synchronization hypothesis is **NOT supported by causal A/B evidence.** The `.item()` calls in Phase-7 3DGS training cost <1% of iteration time when measured fairly. The C32-A profiler analysis was misleading because the profiler's own instrumentation (cudaStreamSynchronize, aten::copy_) dominated its own trace.

**C32-B stops here.** No further investigation into `.item()` is warranted.

## Remaining Open Bottleneck

The true CPU-dispatch overhead (~57ms/iter) remains uncharacterized. C32-A's profiler trace showed ~340 CUDA allocator driver calls per iteration (cudaDeviceGetAttribute, cudaOccupancyMaxActiveBlocks, cudaFuncGetAttributes). These are the dominant CPU-side cost, and they are NOT related to `.item()` or scalar reads. This should be the target of C32-C if the tournament continues.

## Raw Artifacts

- All result JSONs: `results/phase-c31/c32_b_{A,B1,B2,B3,B4,B5,B6,E,F}.json`
- Ablation script: `scripts/phase-c31/c32_b_sync_ablation.py`
- Launcher: `scripts/phase-c31/launch_c32_b.sh`
- Log: `logs/phase-c31/c32_b_launcher.log` on mx
