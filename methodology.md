# Methodology

The benchmark separates four classes of evidence:

1. Synthetic Stress Benchmark
2. Real Scene Quality Benchmark
3. Real Scene Speed Benchmark
4. Pareto Analysis

Rows are comparable only when scene, renderer inputs, camera trajectory,
resolution, timing protocol, and quality reference are compatible.

Raw metrics remain primary: FPS, latency percentiles, VRAM, PSNR, SSIM, LPIPS,
renderer metadata, hardware metadata, and reproducibility hashes. Derived
metrics such as Scene Difficulty Score, quality factor, effective FPS,
stability score, recommendations, and Pareto membership are additive.

Design rationale:

- synthetic workloads expose scheduling and overlap bottlenecks, but they do
  not establish GT quality;
- missing quality remains `null`;
- leaderboard generation consumes JSON artifacts instead of hard-coded tables;
- schemas validate artifact shape without changing historical measurements;
- optional hardware profiling and roofline analysis are separate from core
  benchmark timing to avoid biasing baseline results.

Migration notes:

- existing renderer-keyed `benchmark_results.json` outputs remain readable;
- new runs include `benchmark_suite_version` and additional metadata fields;
- generated leaderboards should be produced from benchmark JSON, not edited by
  hand.

Regression checks compare candidate JSON against stored baselines:

```text
python src/scripts/check_regressions.py \
  --baseline data/results/rtx5070_laptop_2026-07-13.json \
  --candidate results/new_run/benchmark_results.json \
  --output results/regression/regression_report.json
```
