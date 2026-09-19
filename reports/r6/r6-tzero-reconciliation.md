# R6-B — T_zero Profiling Reconciliation

## The discrepancy

| Source | Evidence level | T_zero (ms) | % of T_bwd | Method |
|--------|---------------|-------------|------------|--------|
| Original R6-1 (torch.profiler) | 3 (profiler attribution) | 2.36–8.33 | 5.5–11.6% | Kernel-name matching: `"memset" or "fill"` |
| R6-B B0 direct (CUDA events) | 1 (direct implementation) | 0.031–0.137 | <0.2% | B0 `T_clear` timer wraps `cudaMemsetAsync` only |
| **Discrepancy factor** | — | **30–60×** | — | The original estimate was 30–60× too high |

## Root cause of the overestimate

The original `r6_1_bwd_decompose.py` (line 168) used:

```python
elif "memset" in name_lower or "fill" in name_lower:
    kernel_categories["memset_zero"].append(cuda_self)
```

This category captured **every** kernel whose name contains "memset" or "fill"
during the backward pass — not just the 9 gradient buffer zero-init calls that
R6-B targets. The `r6-2-zero-init-oracle.md` report (lines 9–18) then attributed
this entire category specifically to "at::zeros_like (allocation +
cudaMemsetAsync) for all gradient buffers," listing 9 buffers. **This
attribution was wrong.**

### What the "memset_zero" category actually contained

The torch.profiler `memset_zero` category included:

1. **Gradient buffer zero-init** (the R6-B target): 9 `at::zeros_like` calls
   → 9 `cudaMemsetAsync` launches. On A100, these complete in 0.031–0.137 ms
   total (directly measured by B0's CUDA event timer). This is <0.2% of the
   20–38 ms backward.

2. **DSSIM (SepSSIM) loss backward intermediates**: The SepSSIM loss uses
   15-channel separable Gaussian conv2d. The backward pass allocates and
   zero-fills multiple intermediate tensors (gradient buffers for the separable
   filter, padded inputs, etc.). These `at::zeros_like` / `aten::fill` calls
   dominate the "memset_zero" category.

3. **L1 loss backward intermediates**: Elementwise allocation for the L1
   gradient buffer.

4. **Autograd internal bookkeeping**: `torch.autograd` allocates temporary
   buffers for the backward graph traversal. These may appear as "fill" kernels.

5. **Potential async work from previous kernels**: The torch.profiler
   `self_device_time_total` attribute reports GPU time, but the profiler's
   event-based attribution can include kernels that were queued in a prior
   forward pass but executed during the backward profiling window. Since the
   profiler context wraps only `loss.backward()`, any kernel still in the CUDA
   stream from the forward pass would be attributed to the backward.

### Why the R6-2 oracle misattributed

The R6-2 report listed 9 gradient buffers totaling 21N–69N floats (84N–276N
bytes), and claimed the `T_zero` measurement matched this. But the measured
`T_zero` (2.36–8.33 ms) is 30–60× larger than the actual gradient zero-fill
time (0.031–0.137 ms). The extra time came from items 2–5 above, none of which
are addressable by R6-B.

### Why direct measurement is higher quality

The B0 implementation wraps `cudaMemsetAsync` in CUDA events:

```cpp
cudaEventRecord(clear_start, stream);
cudaMemsetAsync(v_means2d_ptr, 0, bytes, stream);
// ... 5 buffers ...
cudaEventRecord(clear_end, stream);
cudaEventSynchronize(clear_end);
T_clear = clear_start.elapsed_time(clear_end);
```

This measures **only** the gradient buffer zero-fill — no loss backward
intermediates, no autograd bookkeeping, no cross-pass attribution. Evidence
level 1 (direct implementation CUDA event) > evidence level 3 (torch.profiler
attribution).

## Corrected T_zero for R6-B

| Scene | Stage | N | Grad buffer (MB) | T_clear B0 (ms) | T_bwd (ms) | T_clear % T_bwd |
|-------|-------|---|-----------------:|----------------:|-----------:|----------------:|
| room | 5K | 560,632 | 155 | 0.031 | 21.71 | 0.14% |
| room | 15K | 933,590 | 258 | 0.040 | 21.60 | 0.19% |
| room | 30K | 933,590 | 258 | 0.041 | 20.16 | 0.20% |
| bicycle | 5K | 1,933,177 | 534 | 0.071 | 27.07 | 0.26% |
| bicycle | 15K | 3,957,041 | 1092 | 0.130 | 38.09 | 0.34% |
| bicycle | 30K | 3,957,041 | 1092 | 0.130 | 36.05 | 0.36% |
| garden | 5K | 1,779,185 | 491 | 0.067 | 23.47 | 0.29% |
| garden | 15K | 2,610,559 | 721 | 0.091 | 28.45 | 0.32% |
| garden | 30K | 2,610,559 | 721 | 0.092 | 27.41 | 0.34% |

**Corrected T_zero = 0.031–0.137 ms = 0.14–0.36% of T_bwd.**

The original claim of 5.5–11.6% was a profiler attribution error. The true
gradient buffer zero-init cost is <0.4% of backward.

## Corrected touched definitions

### What "touched" means in the R6-B context

- **flatten_ids [n_isects]**: The tile–Gaussian intersection list with
  duplicates. NOT a unique touched Gaussian-row list. A Gaussian that intersects
  5 tiles appears 5 times in flatten_ids.

- **N_intersected** (original r6_1): `len(torch.unique(flatten_ids))` — unique
  Gaussian IDs that appear in at least one tile intersection. This is the
  correct "touched" set for selective-clear purposes.

- **r_touch** (original): `N_intersected / N_total` = 0.18–0.33 (from R6-1
  metadata). But the R6-B benchmark measured r_touch ≈ 0.6–0.8 for the B1-v2
  selective clear. **The discrepancy**: R6-1 used a single camera (idx 0); the
  benchmark also used camera 0 but with a different training state. More
  importantly, the B1-v2 selective clear cost (T_clear = 0.024–0.104 ms) is
  nearly identical to B0's full clear (0.031–0.137 ms) — confirming that most
  rows are touched regardless.

### What "touched" means in the R6-A context

- **flatten_ids [n_isects]**: Each entry is a (tile, Gaussian) intersection
  pair. The backward kernel processes one such pair per batch slot.

- **n_isects**: Total intersection pairs (with cross-tile duplicates). This is
  the atomic operation count after block aggregation (1 atomic per pair).

- **actual_atomic** (direct kernel counter): The actual number of warp-leader
  atomicAdd execution sets. Each set = 11 `gpuAtomicAdd` calls (3 v_colors + 3
  v_conics + 2 v_means2d + 2 v_means2d_abs + 1 v_opacity).

- **R_atomic_direct = actual_atomic / n_isects**: The within-tile warp
  multiplicity. If every (tile, Gaussian) pair is processed by exactly 1 warp,
  R_atomic = 1. If processed by 2 warps, R_atomic = 2.

## Summary

| Metric | Original (torch.profiler) | Corrected (direct CUDA event) | Error |
|--------|--------------------------|-------------------------------|-------|
| T_zero range | 2.36–8.33 ms | 0.031–0.137 ms | 30–60× overestimate |
| T_zero % T_bwd | 5.5–11.6% | 0.14–0.36% | 30–60× overestimate |
| B-GATE-1 (≥5% T_bwd) | PASS 9/9 | FAIL 0/9 | All false positives |
| B-GATE-2 (≥3% T_iter) | PASS 7/9 | FAIL 0/9 | All false positives |
| Conservative E2E | 1.6–4.3% | <0.1% | ~40× overestimate |

The R6-B gate was a false positive caused by profiler attribution error. The
direct implementation measurement (B0 T_clear) proves that gradient buffer
zero-init is <0.4% of backward on A100 — not a bottleneck.
