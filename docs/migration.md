# Matrix v2 Migration Status

## Phase 1 - Stop publishing invalid comparisons (complete)

- Replaced tables that joined synthetic FPS with unrelated quality values.
- Made Matrix v2 the only public leaderboard generator.
- Preserved the empty state until a complete comparable cohort existed.
- Replaced the GitHub Pages empty state with the complete Tier A baseline.

## Phase 2 - Quarantine legacy artifacts (complete)

Legacy results remain auditable and are never rewritten to look compliant.
Historical Windows/WDDM reports now live under
`reports/archive/windows-rtx5070-2026-07/`, separated from the current Linux
baseline. Legacy result records remain classified by provenance and cannot enter
the default ranking.

## Phase 3 - Establish canonical assets (complete)

Suite v3.1.0 pins:

- official per-scene transport identities and archive checksums;
- official iteration-30000 checkpoint members and source cameras;
- deterministic 100-view ordering, 16:9 crop, and GT mapping;
- checkpoint, camera, and GT manifest SHA-256 values for all five cases.

## Phase 4 - Gather Tier A baseline (complete)

- Measured original 3DGS, gsplat, gsplat HiGS, Speedy-Splat, and TC-GS on the
  same EPIC-05 A100 host.
- Completed all five cases with coupled speed/quality records and strict NVML
  process peaks.
- Published 25 results, five eligible overall rows, one cohort, and no rejected
  files.

## Phase 5 - Continuous publication (active)

- Run CPU validation and dry-run planning in ordinary CI.
- Re-run GPU benchmarks on labeled self-hosted runners when a new cohort is
  intentionally created.
- Keep raw result directories immutable and review generated changes before
  promotion.
- Never mix new hardware/software cohorts with the EPIC-05 baseline.
- Add automated documentation consistency checks so current conclusions cannot
  drift from generated `ranking.json`.

## Current repository layout

```text
benchmark/                 Protocol, suite, renderer, dataset, and schema definitions
src/                       Benchmark, adapters, collection, ranking, and reporting code
results/measured/          Published Tier A JSON evidence
docs/leaderboard/          Generated public ranking artifacts
docs/comparison-analysis.md Human-readable interpretation and decision guide
reports/                   Current machine, dataset, reproduction, and outcome reports
reports/archive/           Historical machine-specific investigations
```

Compatibility shims under `benchmark_suite/`, legacy schemas, and historical
documents remain until their callers are explicitly retired. They are not the
publication path.
