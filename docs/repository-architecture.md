# Repository Architecture

```text
benchmark/                   Suite, protocol, renderer registry, datasets, schemas
benchmark_suite/             Compatibility layer for older callers
src/
  benchmark_cli.py           prepare/run/report command implementation
  benchmark_matrix.py        validation, cohort aggregation, ranking, publication
  renderers/                 executable renderer adapters
  scripts/                   workers, asset preparation, collection, validation
datasets/                    ignored raw and processed asset cache
results/
  measured/                  published Tier A JSON evidence
  reproduced/                Tier B official implementation reproductions
  paper/                     Tier C citation records
  quarantine/                non-rankable legacy or incomplete evidence
docs/
  README.md                  documentation navigation
  comparison-analysis.md     current human-readable conclusions
  leaderboard/               generated public JSON/CSV/Markdown/SVG artifacts
reports/
  README.md                  current report navigation
  benchmark_report.md        current run outcome
  machine_report.md          current immutable cohort
  reproducibility.md         reproduction and resume procedure
  dataset_report.md          canonical source and asset hashes
  archive/                   historical machine-specific investigations
tests/                       CPU-runnable contract, schema, and workflow tests
```

## Source of truth

- Protocol and registry data come from `benchmark/`.
- Published measurements come from `results/*/metrics.json` and their adjacent
  raw evidence.
- Numeric ranking artifacts are generated into `docs/leaderboard/`.
- `docs/comparison-analysis.md` interprets the generated result without
  replacing it.

## Publication boundaries

Render PNGs, build outputs, downloaded datasets, local environments, temporary
logs, and failed run caches are not committed as benchmark evidence. Historical
reports are preserved under `reports/archive/` so old machine limitations are
not confused with current repository status.

Evidence tiers remain physically and logically separate. A new hardware or
software cohort cannot silently replace or mix with the EPIC-05 A100 baseline.
