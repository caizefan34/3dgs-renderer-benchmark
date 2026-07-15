# 3DGS Renderer Evaluation Framework

[![Tests](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml)
[![Pages](https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/deploy-pages.yml/badge.svg)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research-grade, quality-gated benchmarking for CUDA 3D Gaussian Splatting
renderers. The platform asks which renderer is most efficient under explicit
quality constraints, not merely which renderer is fastest on one workload.

Synthetic stress results, real-scene quality results, real-scene speed results,
and Pareto analysis are kept separate. Synthetic speed is never treated as
ground-truth quality evidence.

## Project Overview

The benchmark platform provides:

- isolated renderer adapters for gsplat, HiGS, Speedy-Splat, original 3DGS,
  TC-GS, and registered experimental renderers;
- fixed scenes, camera trajectories, timing protocol, and reproducibility
  metadata;
- official-dataset training policy for quality-bearing benchmark submissions;
- PSNR, SSIM, and LPIPS quality gates for real-scene validation;
- Scene Difficulty Score, stability metrics, effective FPS, Pareto analysis,
  and deterministic recommendations as additive metrics;
- generated leaderboard artifacts, schema validation, regression checks,
  Docker scaffolding, and GitHub Pages outputs.

## Key Results

The existing reported measurements are preserved unchanged.

Synthetic stress timing on an RTX 5070 Laptop at 1920x1080:

| Scene | Renderer | GPU mean | P99 | FPS | Peak VRAM | GT quality |
|---|---|---:|---:|---:|---:|---|
| 50K | HiGS tile16 | 1.99 ms | 2.45 ms | 502.7 | 147 MB | N/A |
| 200K | HiGS tile16 | 6.34 ms | 7.23 ms | 157.8 | 391 MB | N/A |
| 400K | HiGS tile8 | 15.96 ms | 23.22 ms | 62.7 | 1057 MB | N/A |

Paired-reference quality audit on the official Train model:

| Renderer | PSNR | SSIM | LPIPS | Status |
|---|---:|---:|---:|---|
| original 3DGS | 24.9319 | 0.865773 | 0.223592 | reference |
| gsplat dense | 24.3061 | 0.858717 | 0.226278 | -0.6257 dB; not equivalent |
| TC-GS | 24.9138 | 0.865044 | 0.222874 | equivalent at configured thresholds |

The pretrained model archive does not prove that those 38 reference images
were excluded from training, so these are renderer-fidelity results rather
than a held-out reconstruction leaderboard.

## Quick Start

```text
git clone https://github.com/caizefan34/3dgs-renderer-benchmark.git
cd 3dgs-renderer-benchmark
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

GPU smoke test after installing a CUDA-enabled PyTorch build and at least one
renderer backend:

```text
python src/scripts/generate_scene.py --gaussians 50000 --output data/scene.ply
python src/run_benchmark.py --list-renderers
python src/run_benchmark.py --scene data/scene.ply --camera-path circle --renderers gsplat --frames 100 --warmup 30 --repeats 3 --output results/quickstart
```

Generate leaderboards from committed benchmark JSON:

```text
python src/scripts/generate_leaderboard.py --inputs data/results/rtx5070_laptop_2026-07-13.json data/results/rtx5070_train_reference_summary_2026-07-14.json --output-dir results/leaderboard
```

List official training dataset sources:

```text
python src/scripts/download_datasets.py --list-official
python src/scripts/validate_official_training.py
```

## Leaderboard

Committed GitHub Pages artifacts live in [`docs/leaderboard`](docs/leaderboard).
Local generated artifacts should be written to `results/leaderboard`.

The generator produces:

- `leaderboard.json`
- `leaderboard.md`
- `leaderboard.html`

See [leaderboard documentation](docs/leaderboard.md).

## Supported Renderers

- `speedy_splat`
- `original_3dgs` / `diff_gaussian`
- `gsplat`
- `gsplat_dense`
- `gsplat_higs`
- `gsplat_higs_tile16`
- `gsplat_higs_sh32`
- `gsplat_higs_sh16`
- `gsplat_higs_auto`
- `tcgs`
- `fast_gauss` (registered, unavailable in the current local Windows policy)

Renderer source, commit, and reproducibility notes are tracked in
[the renderer survey](docs/renderer_survey.md).

## Documentation Links

- [Benchmark taxonomy](docs/benchmark_taxonomy.md)
- [Methodology](docs/methodology.md)
- [Evaluation formulas](docs/evaluation_methodology.md)
- [Benchmark suite](docs/benchmark_suite.md)
- [Official dataset training](docs/official_dataset_training.md)
- [Renderer discovery](docs/renderer_discovery.md)
- [Synthetic Stress Suite](docs/synthetic_stress_suite.md)
- [Leaderboard pipeline](docs/leaderboard.md)
- [Reproducibility](docs/reproducibility.md)
- [Architecture](docs/architecture.md)
- [Research extensions](docs/research_extensions.md)
- [Summary report](docs/summary_report.md)
- [Contributing](CONTRIBUTING.md)
