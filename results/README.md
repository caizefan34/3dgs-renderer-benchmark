# Results

Raw and normalized results are separated by evidence provenance.
Generated reports never combine these directories into one ranking.

- `measured/`: Tier A, produced by `benchmark run` through this repository.
- `reproduced/`: Tier B, produced from a pinned official implementation with deviations recorded.
- `paper/`: Tier C, citation-backed transcription only.
- `quarantine/`: legacy or incomplete artifacts that cannot satisfy the current matrix.
- `diagnostics/`: controlled investigations that explain anomalies but are
  excluded from rankings because they are non-canonical or post-hoc.

Diagnostic runs must state why they are excluded from Tier A. See the
[TC-GS variance re-test](diagnostics/tcgs-variance-20260805/README.md) for an
example that separates follow-up evidence from the pre-declared leaderboard.

Tier A layout:

```text
results/measured/<renderer>/<dataset>/<scene>/<run-id>/
  metrics.json
  raw_samples.json
  <renderer>/speed/benchmark_results.json
  <renderer>/quality/quality_gt.json
```
