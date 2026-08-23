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

## HiGS research

- [Implementation report](higs-trainability-implementation.md) — round-by-round log (Rounds 1-30) of making HiGS differentiable: native CUDA backward, correctness verification, performance engineering, EPIC-05 recovery.
- [Trainability source analysis](higs-trainability-analysis-2026-07-24.md)
- [Research plan: significant training speedup](higs-training-speedup-research-2026-08-03.md) — current direction (tile-sampled training with quality guarantees).
- Dated EPIC-05 studies: [Tier A baseline](epic05-tier-a-baseline-2026-07-20.md), [HiGS calibrated](epic05-higs-calibrated-2026-07-24.md), [HiGS p99/LPT](epic05-higs-p99-lpt-2026-07-25.md)
- [TC-GS variance re-test](epic05-tcgs-variance-2026-08-05.md) - three independent five-scene passes documenting heavy-tail latency without replacing the pre-declared leaderboard batch.
- Compression studies: [SPZ qualification](epic05-spz-qualification-2026-07-24.md), [expanded compression](epic05-expanded-compression-qualification-2026-07-24.md), [artifact encoding](epic05-compression-artifact-encoding-2026-07-23.md), [FCGS rate curve](epic05-fcgs-rate-curve-2026-07-24.md), [FCGS five scene](epic05-fcgs-five-scene-2026-07-24.md)
- Historical sweep: [final-conclusions.md](final-conclusions.md)

## Historical reports

The pre-Linux Windows RTX 5070/WDDM investigation is preserved under
[`archive/windows-rtx5070-2026-07/`](archive/windows-rtx5070-2026-07/). Its
empty rankings and NVML blocker describe that historical machine only and are
not current repository status.
