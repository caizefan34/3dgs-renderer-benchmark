# R20 independent review of R19 timing

Status: STATIC_AUDIT_COMPLETE; root cause NOT_ESTABLISHED; GPU timing replay
NOT_EXECUTED. Date: 2026-10-09.

Inputs are actual `r19_src/r19_trainer.py`, `r19_collect.py`,
`baseline/reference_v1/gaussian_model.py` and exported HiGS wrapper/kernel.
The requested signed-mode pairs (64.83/17.28 ms Bicycle; 47.30/19.21 ms Room)
are user-supplied observations. This review does not invent raw samples or p99.

## Event boundaries

`r19_trainer.py:332-405` creates six reusable timing events:

```text
e0 -> render -> e1 -> L1/SSIM/loss -> e2 -> loss.backward -> e3
   -> radii/statistics/optional clone-split-prune/reset -> e4
   -> optimizer.step + zero_grad -> e5 -> torch.cuda.synchronize
backward = e2.elapsed_time(e3)
densify  = e3.elapsed_time(e4)
optimizer= e4.elapsed_time(e5)
```

There is no direct densification/optimizer overlap in the backward event interval.
Reading elapsed times after synchronize makes event reuse valid. Events record
on the current stream; any kernels on auxiliary streams must rejoin it before
the corresponding end event. The frozen std_ll source package contains stream
utilities, but runtime stream traces are required to prove complete join behavior.
Default `torch.cuda.synchronize()` synchronizes the current device; it does not
establish cross-device measurement semantics.

An event interval can include GPU stream idle time while Python autograd submits
work, not just summed kernel execution. Gradient hooks, backward allocations,
parameter accumulation, SH/projection VJPs and frees are inside this interval.
No synchronize-before-e2 is used, but stream FIFO ordering separates earlier
loss kernels. A separate synchronized replay is useful diagnostically and changes
the schedule; keep original and synchronized modes as distinct measurements.

## Lifetime and memory evidence

Both signed arms retain means2d gradients and register a hook allocating
`grad.detach().abs()`. The hook closure captures the tensor that owns its hook;
this forms a Python reference cycle. The absolute attribute and retained gradient
can live until garbage collection. It is a concrete lifetime concern, not proof
that it explains the observed mean/median discrepancy. HiGS backward additionally
allocates FP32 master and compact intermediates; std_ll has its own allocations.
Python wrapper boolean checks/.item(), dynamic indexing and allocator slow paths
can cause host synchronization. Their contribution requires traces.

`GaussianModel.densify_and_prune` calls `torch.cuda.empty_cache()` after topology
changes. This is recorded in the densify interval, but can change allocator/cache
behavior in subsequent iterations. R19 only records the cumulative maximum
allocated VRAM. It does not reset peaks per phase/step, record reserved bytes,
allocator retries, GPU free-memory history or GC times. Thus memory spikes cannot
be localized to backward from the saved summary. No allocator root cause is asserted.

## Aggregation limitations

`r19_trainer.py:421-426` computes mean, median, minimum and maximum from ALL
iterations. There is no warmup exclusion, per-SH-degree/camera/Gaussian-count
stratification, p99, outlier IDs, or saved raw phase array. `per_iter` only samples
loss/N every 100 steps. Changes in SH degree, camera visibility and topology can
make the population nonstationary. A much larger mean than median establishes
skew/long-tail cost, but cannot identify its origin from these summaries.

`r19_collect.py:307-325` sums stage medians for its iteration decomposition.
`median(A)+median(B)` generally differs from `median(A+B)`. Dividing stage-median
deltas by their sum can yield ratios above one; it is not a causal time attribution.
Use each iteration's stage sum and paired fixed-state means for an additive check.
Reported renderer time includes densify, whose contents also include trainer
statistics/topology/reset work. E2E wall time includes dataset loading, LR/SH
updates, evaluations and host work outside event boundaries. Final evaluation is
performed after `total_time` is fixed, but periodic evaluations are included.
Those definitions must accompany any renderer/E2E speedup claim.

P99 cannot be reconstructed from mean/median/min/max. R20's microbench stores
all iteration records and uses NumPy percentile `method='linear'`, including
sample count, max and top outlier indices; no trimming is done silently.

## R20 diagnostics for DSH

1. Run fixed-state std_ll signed/native-absolute and R20 HiGS OFF/ON in separate
   processes, equal camera/SH/N/cotangents, warmup 20, samples 200, three repeats.
2. Save raw events, host wall time, allocated/reserved before/after, per-step
   peaks. Compare mean/median/p90/p99/max and annotate iteration outliers.
3. Instrument an R19 replay with iteration, camera ID, SH degree, N, previous
   topology/reset event, GC counters and allocator retry counters. Preserve the
   historical script; keep diagnostics in a new file.
4. Trace a short outlier window with Nsight Systems (CUDA API/kernel + NVTX +
   OS runtime) to distinguish kernel duration, host submission gaps, allocation
   waits and concurrent GPU work. Verify stream joins in the trace.
5. Compare CUDA-event phase sums against explicitly defined wall-time windows.
   Do not turn a median-only decomposition into a causal conclusion.

Additional source confound: std_ll passes eps2d=0.1; R19 HiGS omits it and the
actual wrapper default is 0.3. R20 test/trainer calls explicitly use 0.1 in both
arms. No renderer default or frozen R19 result is rewritten. This can affect
projected footprint and gradients; its quantitative effect is NOT measured here.
