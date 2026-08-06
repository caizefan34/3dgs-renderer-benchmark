# 3DGS Renderer Benchmark and Research Suite

<p align="center">
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml"><img src="https://github.com/caizefan34/3dgs-renderer-benchmark/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2563eb" alt="MIT license"></a>
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/stargazers"><img src="https://img.shields.io/github/stars/caizefan34/3dgs-renderer-benchmark?style=flat&label=Stars&color=f59e0b" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="https://caizefan34.github.io/3dgs-renderer-benchmark/"><strong>Results explorer</strong></a> |
  <a href="paper/README.md"><strong>Paper evidence</strong></a> |
  <a href="https://github.com/caizefan34/3dgs-renderer-benchmark/issues/new?template=result_submission.yml"><strong>Submit results</strong></a>
</p>

A reproducible research suite for 3D Gaussian Splatting (3DGS), organized
around three independently publishable questions:

1. Which recent rendering, training, and storage methods are reproducible and
   comparable?
2. Can HiGS support a correct native backward and faster end-to-end training?
3. Which storage format is smallest under a declared lossless or near-lossless
   contract?

The repository shares datasets, provenance, quality metrics, and artifact
validation across these tracks. Results are valid only inside their declared
hardware and protocol cohort; they are not universal renderer, training, or
codec rankings.

## Research tracks

| Track | What is available | Current scientific boundary | Start here |
| --- | --- | --- | --- |
| Reproducible 3DGS survey | Source-pinned registry, evidence tiers, integration status, five-renderer A100 matrix | A systematic/latest-survey claim still needs a frozen search and screening audit | [Survey protocol](docs/survey-protocol.md) |
| Differentiable HiGS | Native CUDA backward, dynamic topology, sparse/progressive training studies, positive and negative ablations, frozen 177-job from-scratch matrix | Trainability and mean peak-memory reduction are measured; quality-preserving training speedup is not supported (final PSNR lower on 10 of 11 scenes) | [HiGS paper plan](docs/higs-paper-plan.md) |
| Storage compression | Bit-exact and same-checkpoint near-lossless round trips across five scenes | Learned retraining codecs and decode/deployment cost require a separate completed cohort | [Compression protocol](docs/compression-protocol.md) |

The [research program](docs/research-program.md) explains why these tracks share
one artifact but should not be presented as three co-equal contributions in one
paper.

## Supported evidence

### Renderer comparison

The repository has complete Tier A coverage for its declared measured cohort:
five renderer configurations across five fixed 1920x1080 cases on one NVIDIA
A100-SXM4-80GB. Within that cohort,
`gsplat_higs` has a 5.671x speed index and 696.91 aggregate FPS. It also has a
small measured quality delta, so the result is a throughput finding rather than
a universal quality-preserving claim.

- [Comparison analysis](docs/comparison-analysis.md)
- [Generated leaderboard](docs/leaderboard/ranking.md)
- [Protocol](docs/protocol.md)
- [Hardware cohort rules](docs/hardware.md)

## Tier A comparison charts

The charts below are generated from the same 25 accepted runs and remain inside
the frozen A100 cohort.

| Throughput | Quality |
| --- | --- |
| [![FPS ranking](docs/leaderboard/measured-fps-ranking.svg)](docs/leaderboard/measured-fps-ranking.svg) | [![PSNR ranking](docs/leaderboard/measured-psnr-ranking.svg)](docs/leaderboard/measured-psnr-ranking.svg) |
| [![VRAM ranking](docs/leaderboard/measured-vram-ranking.svg)](docs/leaderboard/measured-vram-ranking.svg) | [![SSIM ranking](docs/leaderboard/measured-ssim-ranking.svg)](docs/leaderboard/measured-ssim-ranking.svg) |
| [![Speed versus LPIPS](docs/leaderboard/measured-speed-vs-lpips.svg)](docs/leaderboard/measured-speed-vs-lpips.svg) | [![LPIPS ranking](docs/leaderboard/measured-lpips-ranking.svg)](docs/leaderboard/measured-lpips-ranking.svg) |

### Differentiable HiGS

The implementation provides:

- a correctness baseline using standard gsplat recomputation;
- a native HiGS CUDA backward for blend, projection, and SH gradients;
- frozen and dynamic topology paths with versioned scene state;
- finite-difference, `gradcheck`, mixed-precision, multi-camera, background,
  depth-mode, and topology-lifecycle coverage;
- hierarchy-aware tile sampling, progressive resolution, and visibility-aware
  optimizer experiments.

The research log reports quality-prioritized 3000-step A100 configurations that
reduce per-step time by 8.6% to 21.7% against the same-backend full-resolution
control on four high/mid Gaussian-count scenes. Train is a documented exception.
The frozen 177-job A100 matrix (original_3dgs 33 = 11 scenes x 3 seeds;
gsplat/HiGS 48 each, with the five cross-hardware scenes run twice, 30k steps)
has now been executed from SfM initialization with zero failed jobs. The
proposed method (visibility-masked Adam + progressive resolution) trains to a
complete scene with 21.6% lower mean peak GPU memory per job than the gsplat
control (2.79 vs 3.72 GiB aggregate), but final PSNR is lower on 10 of 11
scenes (tied on stump), mean wall time is faster on 7 of 11, and aggregate
time-to-quality is 4.3% higher, so no quality-preserving speedup is claimed.
Machine-readable aggregates live in
[`paper/higs/tables/matrix-summary.json`](paper/higs/tables/matrix-summary.json).
The short-horizon 1.8x-2.5x numbers are not full-convergence results.

- [Implementation report](reports/higs-trainability-implementation.md)
- [Training research and negative results](reports/higs-training-speedup-research-2026-08-03.md)
- [Submission design](docs/higs-paper-plan.md)

### Storage compression

The storage study keeps bit-exact, same-checkpoint near-lossless, and
retraining-required codecs in separate cohorts. On the frozen five-scene
same-checkpoint cohort, SPZ 8/8 passes every near-lossless gate at 5.572x to
6.072x compression and under 0.02 dB absolute PSNR change. XZ is the bit-exact
option with substantially smaller storage savings.

- [Expanded qualification](reports/epic05-expanded-compression-qualification-2026-07-24.md)
- [Machine-readable evidence](reports/generated/compression-expanded-final/compression-results.json)
- [Terminology and deployment protocol](docs/compression-protocol.md)

## Quick start

Requirements depend on the selected CUDA backend. CPU-only validation and
artifact inspection work without renderer extensions.

```bash
git clone https://github.com/caizefan34/3dgs-renderer-benchmark.git
cd 3dgs-renderer-benchmark
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Prepare a canonical benchmark case and run an installed renderer:

```bash
python -m pip install -r requirements-benchmark.txt
python benchmark.py prepare mipnerf360 --scene garden
python benchmark.py prepare-case small-garden-1080p
python benchmark.py run gsplat_higs --dataset garden
```

The preparation commands download and verify official assets. Renderer-specific
CUDA extensions are installed separately; unavailable backends fail or skip
explicitly rather than producing placeholder measurements.

## Academic evidence gates

Validate the frozen benchmark-paper manifest:

```bash
python src/scripts/validate_paper_evidence.py
```

Validate the independent survey, HiGS, and compression paper tracks:

```bash
python src/scripts/validate_research_program.py
```

Validate and expand the HiGS full-training submission matrix:

```bash
python src/scripts/validate_higs_paper_protocol.py \
  --output-plan artifacts/higs-paper/experiment-plan.json
```

Audit the pinned gsplat source and emit one from-SfM training command:

```bash
python src/scripts/prepare_higs_paper_source.py --variant official
python src/scripts/prepare_higs_paper_source.py --variant higs

python src/scripts/build_higs_training_command.py \
  --method gsplat --scene mipnerf360/garden --seed 0 \
  --data-dir /datasets/360_v2/garden \
  --result-dir results/paper/higs/gsplat-garden-s0
```

Each `supported` claim is pinned to Git-tracked JSON evidence by SHA-256 and
executable assertions. `blocked` claims name the missing experiment. A blocked
claim must not enter an abstract, result table, or conclusion.

Build a deterministic release artifact with:

```bash
python src/scripts/build_release_bundle.py --output 3dgs-renderer-benchmark-<version>.zip
```

See [paper/README.md](paper/README.md) for the release and DOI workflow.

## Repository map

```text
benchmark/       Frozen suites, protocols, registries, schemas, and training configs
src/             Adapters, evaluation, validation, statistics, and release tooling
tests/           CPU-safe unit tests plus opt-in CUDA regression tests
docs/            Current methodology, research protocols, and generated leaderboard
paper/           Machine-readable claims and independent paper-track gates
reports/         Auditable run reports and chronological research appendices
results/         Raw, measured, generated, and historical result artifacts
scripts/higs/    Reproduction scripts for HiGS training experiments
patches/         Differentiable HiGS source patch and integration notes
```

Current documentation starts at [docs/README.md](docs/README.md). Historical
Windows/WDDM results remain under `reports/archive/` and are not mixed with the
current A100 cohort.

## Contributing results

Use the structured [benchmark result submission](https://github.com/caizefan34/3dgs-renderer-benchmark/issues/new?template=result_submission.yml)
before changing a published cohort. Literature additions should include a
stable paper URL, source URL and commit, task taxonomy, license, and evidence
tier. HiGS or compression results must include the exact recipe, seeds, raw
JSON, environment, and all failures.

See [CONTRIBUTING.md](CONTRIBUTING.md) for review requirements.

## Citation

Citation metadata is maintained in [CITATION.cff](CITATION.cff). Until an
archival DOI is issued, cite the repository URL and exact release or commit.
Do not invent a DOI in advance.

## License

Repository code is available under the [MIT License](LICENSE). Upstream
renderers, datasets, checkpoints, and codecs retain their own licenses and
must be reviewed independently.
