# R6-B — Benchmark

The CUDA-event benchmark entry point is
`experiments/r6/r6_b/benchmark_r6b.py`. Run baseline, B0, and B1 separately
for each fixed R6 workload with `--warmup 20 --measure 100`; it emits mean,
standard deviation, p50, and p95 for `T_bwd`, `T_iter`, and (for B0/B1) the
rasterizer preparation CUDA-event interval.

The required nine checkpoint workloads are absent from this local workspace.
Consequently there are no fabricated speedups, variances, or E2E claims.
