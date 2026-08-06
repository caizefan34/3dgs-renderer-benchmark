# Benchmark Reports

## Current EPIC-05 baseline

- [Benchmark outcome](benchmark_report.md)
- [Machine and software cohort](machine_report.md)
- [Reproduction and resume instructions](reproducibility.md)
- [Canonical dataset and asset hashes](dataset_report.md)
- [Human-readable comparison analysis](../docs/comparison-analysis.md)
- [Generated leaderboard](../docs/leaderboard/ranking.md)

`docs/leaderboard/` is the only publication path for generated ranking JSON,
CSV, Markdown, and SVG charts. Tier A is preferred when available; Tier B and
Tier C remain separate.

## Differentiable HiGS research

- [Submission-oriented paper plan](../docs/higs-paper-plan.md) — authoritative scientific question, contribution boundary, and missing full-training experiment.
- [Implementation report](higs-trainability-implementation.md) — chronological appendix for native CUDA backward, correctness verification, and performance engineering.
- [Trainability source analysis](higs-trainability-analysis-2026-07-24.md)
- [Training optimization log](higs-training-speedup-research-2026-08-03.md) — chronological positive and negative results; not the manuscript narrative.
- Dated EPIC-05 studies: [Tier A baseline](epic05-tier-a-baseline-2026-07-20.md), [HiGS calibrated](epic05-higs-calibrated-2026-07-24.md), [HiGS p99/LPT](epic05-higs-p99-lpt-2026-07-25.md)
- Historical sweep: [final-conclusions.md](final-conclusions.md)

## Storage compression research

- [Compression protocol and terminology](../docs/compression-protocol.md)
- [SPZ qualification](epic05-spz-qualification-2026-07-24.md)
- [Expanded same-checkpoint qualification](epic05-expanded-compression-qualification-2026-07-24.md)
- [Artifact encoding](epic05-compression-artifact-encoding-2026-07-23.md)
- [FCGS rate curve](epic05-fcgs-rate-curve-2026-07-24.md)
- [FCGS five-scene study](epic05-fcgs-five-scene-2026-07-24.md)

## Historical reports

The pre-Linux Windows RTX 5070/WDDM investigation is preserved under
[`archive/windows-rtx5070-2026-07/`](archive/windows-rtx5070-2026-07/). Its
empty rankings and NVML blocker describe that historical machine only and are
not current repository status.
