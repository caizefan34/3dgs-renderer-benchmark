# Regression Baselines

Use committed benchmark JSON files as baselines for regression checks. The
default threshold policy is stored in `regression_thresholds.json`.

Example:

```text
python src/scripts/check_regressions.py \
  --baseline data/results/rtx5070_laptop_2026-07-13.json \
  --candidate results/new_run/benchmark_results.json \
  --output results/regression/regression_report.json \
  --fail-on-regression
```

Baselines must be matched by renderer, benchmark type, scene/cohort, and
Gaussian count when available.

