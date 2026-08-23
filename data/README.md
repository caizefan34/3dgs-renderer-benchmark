# Curated Data (`data/`)

Committed, curated inputs used by the pipeline and CI:
- `camera_presets/` — reusable camera trajectories (circle, flythrough, spiral, random_walk).
- `examples/` — difficulty metrics, evaluation records, Pareto frontier examples.
- `datasets/official_training_datasets.json` — official training dataset policy.
- `results/` — CI fixture result JSONs (e.g. `rtx5070_laptop_2026-07-13.json`).

Do not confuse with `datasets/` (repo root): that is the **local download and
processing cache** (mostly git-ignored), described in `datasets/README.md`.
