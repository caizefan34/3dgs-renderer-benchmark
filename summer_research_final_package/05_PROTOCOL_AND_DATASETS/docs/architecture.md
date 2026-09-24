# Evaluation framework architecture

```text
scene + fixed cameras
        |
        +-- Synthetic Stress run ------> benchmark_results.*
        |
        +-- Real Scene Speed run ------> benchmark_results.*
        |
GT images + renderer output ----------> quality JSON
        |
compatible normalized cohort
        +-- quality adjustment --------> evaluation_records.json
        +-- Pareto analysis -----------> pareto_frontier.json/html
        +-- deterministic rules -------> recommendations.json

benchmark JSON artifacts
        +-- schema validation ---------> CI pass/fail
        +-- leaderboard generator -----> leaderboard.json/md/html
        +-- regression checker --------> regression_report.json
        +-- plot generator ------------> PNG/PDF figures
```

Key modules:

- `src/benchmark/difficulty.py`: versioned difficulty calculation.
- `src/benchmark_framework/metrics.py`: raw timing, stability, optional
  difficulty, quality, and adjusted metrics.
- `src/analysis/efficiency.py`: experimental effective FPS.
- `src/analysis/pareto.py`: multi-objective dominance.
- `src/analysis/recommendations.py`: deterministic category rules.
- `src/analysis/visualization.py`: standalone Pareto HTML.
- `src/analysis/regression.py`: threshold-based performance regression checks.
- `src/analysis/plots.py`: optional matplotlib publication figures.
- `src/analysis/hardware.py`: optional hardware-profiler normalization.
- `src/analysis/roofline.py`: optional roofline-style classification.
- `src/scripts/analyze_results.py`: normalization and artifact CLI.
- `src/leaderboard/generator.py`: speed, quality, memory, and Pareto
  leaderboard generation.
- `src/schema_validation.py`: lightweight JSON Schema validation used by CI.

The original renderer-keyed `benchmark_results.json` envelope, old metric
names, adapters, and result files remain valid. New analysis outputs use their
own schema versions.
