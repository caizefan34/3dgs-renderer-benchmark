# Evaluation methodology

All new metrics are additive. Raw FPS, latency, PSNR, SSIM, LPIPS, VRAM, and
renderer metadata remain the primary evidence.

## Scene Difficulty Score (experimental)

`geometric_mean_v1` is normalized to `[0, 10]`:

```text
factor_i = min(measurement_i / saturation_i, 1)
difficulty_score = 10 * (visible * overlap * tile_density * depth_complexity)^(1/4)
```

Default saturation values are 1,000,000 visible Gaussians, overlap ratio 8,
average tile density 128, and depth complexity 16. Measurements must aggregate
the same camera trajectory used by every renderer. The JSON stores raw inputs,
normalization scales, `schema_version`, and `formula_id`, so another formula
can be added without changing old results. If the measurements do not exist,
the score is `null`; Gaussian count alone is not silently substituted.

## Quality-adjusted efficiency (experimental)

Only regressions relative to the selected GT-scored reference are penalized:

```text
penalty = wp * max(0, reference_psnr - psnr)
        + ws * max(0, reference_ssim - ssim)
        + wl * max(0, lpips - reference_lpips)
quality_factor = exp(-penalty)
effective_fps = fps * quality_factor
```

Defaults are `wp=0.25`, `ws=25`, and `wl=10`; CLI coefficients are
configurable. Improvements do not raise the factor above 1. Missing quality
produces `null`, never 1. This is an experimental decision aid, not a
replacement for raw metrics or hard quality gates.

## Stability

- `std_latency_ms`: population standard deviation of measured frame latency.
- `coefficient_of_variation = std / mean`.
- `jitter_pct = 100 * coefficient_of_variation` is retained for compatibility.
- `stability_score = min(median_latency / p99_latency, 1)`; higher is steadier.

## Deterministic recommendations

Rules are versioned as `deterministic_v1`:

| Category | Rule |
|---|---|
| Best Absolute Speed | Highest raw FPS |
| Best Quality Preserving | Highest quality factor among quality-eligible rows |
| Best Balanced | Highest effective FPS among quality-eligible rows |
| Best Memory Efficiency | Lowest peak VRAM |
| Best Low-Latency | Lowest P99 latency |
| Best Pareto Candidate | Highest effective FPS on the Pareto frontier |

Renderer id is the final alphabetical tie-break. A published recommendation
requires one compatible cohort; current website artifacts intentionally remain
empty because the existing quality and speed snapshots use different
resolutions.

## Commands and artifacts

Attach measured difficulty inputs to a benchmark:

```powershell
python src/run_benchmark.py `
  --scene data/scene.ply --camera-path circle --renderers gsplat `
  --difficulty-metrics data/examples/difficulty_metrics.json `
  --benchmark-type synthetic_stress --output results/example
```

Analyze a compatible real-scene cohort:

```powershell
python src/scripts/analyze_results.py `
  --input data/examples/evaluation_records.json `
  --output-dir results/evaluation `
  --reference-renderer reference_renderer
```

The analyzer writes `evaluation_records.json`, `pareto_frontier.json`,
`recommendations.json`, and `pareto_frontier.html`.
