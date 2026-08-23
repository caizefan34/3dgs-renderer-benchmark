# `benchmark_suite/` — DEPRECATED

Legacy compatibility copy of the Matrix v2 suite configuration
(`suite.json` + dataset definitions). It diverged from the canonical config and
is kept only because `src/benchmark_suite.py` and
`tests/test_benchmark_suite.py` still load it.

**Use `benchmark/suite.json` instead** (see `benchmark/README.md`). Remove this
directory after migrating those two callers.
