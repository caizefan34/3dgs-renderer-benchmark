# Source Layout (`src/`)

`src/` is a flat Python namespace (see `pyproject.toml`). The `benchmark` CLI is
defined by `benchmark_cli.py`; the other top-level modules are higher-level
drivers or libraries called by it.

## Entry modules
- `benchmark_cli.py` — the `benchmark` command (`prepare` / `run` / `report` / ...); `benchmark.py` at the repo root forwards here.
- `benchmark_matrix.py` — Matrix v2 validation, cohort aggregation, ranking, and publication.
- `benchmark_suite.py` — legacy suite loader (reads `benchmark_suite/suite.json`); kept for older callers.
- `run_benchmark.py` / `run_full_benchmark.py` — batch drivers for matrix/EPIC-05 runs.
- `compression_artifact.py` — compression artifact utilities.
- `schema_validation.py` — JSON schema validation helpers.

## Packages
- `benchmark_framework/` — core config/metrics plumbing behind the CLI.
- `benchmark/` — small helpers (e.g. scene difficulty metrics).
- `renderers/` — executable renderer adapters.
- `adapters/` — renderer integration base classes and quality adapters.
- `analysis/` — efficiency, hardware, Pareto, regression, roofline, visualization.
- `leaderboard/` — generated leaderboard construction.
- `datasets/` — official training dataset definitions.
- `schemas/` — embedded JSON schemas for specific result types.
- `scripts/` — Python workers (dataset prep, collection, validation, report/plot generation).

## Related locations
- Shell/env scripts: repo root `scripts/` (Linux/EPIC-05).
- Tests: `tests/`.
