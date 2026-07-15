# Migration Plan

## Phase 1 — Stop publishing invalid comparisons

- Replace the README table that joins synthetic FPS with unrelated quality.
- Regenerate the public leaderboard from Matrix v2 only.
- Preserve the honest empty state until complete comparable results exist.
- Label `docs/index.html` and the current `docs/leaderboard/*` as legacy or replace them with generated v2 outputs.

## Phase 2 — Quarantine legacy artifacts

Do not rewrite old JSON to look compliant.
Inventory `data/results/**` under `results/quarantine/legacy-index.json` with explicit reasons such as mismatched resolution, missing renderer commit, missing protocol hash, synthetic workload, missing GT quality, or incomplete suite coverage.

## Phase 3 — Establish canonical assets

Completed in suite v3.1.0:

- Pin official per-scene transport identities and the T&T archive SHA-256.
- Pin official iteration-30000 checkpoint members and source camera exports.
- Define deterministic 100-view ordering, selection, 16:9 crop, and GT mapping.
- Publish checkpoint, canonical camera, and GT manifest SHA-256 for all five cases.

## Phase 4 — Gather Tier A baseline

- Measure original 3DGS and gsplat on the same reference host.
- Require all five cases and coupled speed/quality records.
- Add Speedy-Splat, TC-GS, and one fixed HiGS configuration after build provenance passes.

## Phase 5 — Continuous publication

- Run CPU validation and dry-run planning in ordinary CI.
- Run the GPU benchmark on a labeled self-hosted runner on demand/nightly.
- Upload raw result directories as immutable artifacts.
- Generate reports from `results/measured/**/metrics.json`.
- Require review before promoting GPU-run artifacts into the published baseline.

## File actions

Add:

- `benchmark/suite.json`, `benchmark/protocol.json`, `benchmark/renderers.json`;
- `benchmark/datasets/*.json`, `benchmark/schemas/result.schema.json`;
- `src/benchmark_cli.py`, `src/benchmark_matrix.py`;
- `src/scripts/prepare_datasets.py`, `src/scripts/collect_matrix_result.py`;
- `src/scripts/stage_dataset_case.py`;
- `results/{measured,reproduced,paper,quarantine}/`;
- `docs/{methodology,hardware,datasets,ranking,repository-architecture,renderer-integration,migration}.md`.

Reorganize:

- `benchmark_suite/` into compatibility shims pointing at `benchmark/` after v1 callers are migrated;
- `data/results/` into a read-only legacy archive plus explicit quarantine index;
- duplicate `schemas/` and `src/schemas/` into `benchmark/schemas/` by version;
- generated plots/reports out of source-data directories.

Remove after compatibility sunset:

- hand-maintained numeric ranking tables in `README.md` and `docs/index.html`;
- stale v1 leaderboard artifacts;
- duplicated protocol defaults in code and documentation;
- paper-reported numbers from any default ranking input.

Compatibility files should not be deleted until old result readers and CI checks are migrated.
