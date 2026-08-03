# Schemas

- `schemas/` (repo root) — top-level, shared JSON schemas:
  `benchmark_result.schema.json`, `leaderboard.schema.json`, `regression_report.schema.json`.
- `benchmark/schemas/` — Matrix v2 result schemas (result, compression, temporal, training).
- `src/schemas/` — embedded schemas for analysis/resolution/speed artifact types.

Validation entry points: `src/schema_validation.py`, `src/scripts/validate_artifacts.py`.
