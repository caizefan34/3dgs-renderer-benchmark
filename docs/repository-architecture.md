# Repository Architecture Proposal

```text
benchmark/                  immutable suite, protocol, renderer registry, schemas
  datasets/                 acquisition manifests
  schemas/                  strict Matrix v2 result schema
datasets/                   ignored download/raw/processed cache
scripts/                    user-facing command layout documentation
src/
  benchmark_cli.py          benchmark prepare/run/report
  benchmark_matrix.py       tier-safe validation, aggregation, Pareto, reports
  renderers/                executable in-process adapters
  scripts/                  workers, dataset download/staging, collection
results/
  measured/                 Tier A local executions
  reproduced/               Tier B official implementation reproductions
  paper/                    Tier C citation records
  quarantine/               non-rankable legacy/incomplete evidence
reports/                    generated JSON/CSV/Markdown summaries
plots/                      generated publication plots
docs/                       methodology, hardware, datasets, rankings, migration
```

Raw measurements are immutable inputs.
Reports and plots are disposable derived artifacts.
The website and README consume generated Tier A-first outputs rather than hand-edited benchmark numbers.
