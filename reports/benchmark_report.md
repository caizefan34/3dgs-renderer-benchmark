# EPIC-05 Tier A Benchmark Report

## Outcome

The Linux Tier A matrix completed all 25 renderer/case combinations on one
EPIC-05 NVIDIA A100 cohort. The final report contains five eligible overall
rows and no rejected inputs.

| Requirement | Result |
| --- | --- |
| Renderer/case coverage | 5 x 5 = 25 complete results |
| Evidence tier | Tier A measured |
| Hardware cohorts | 1 |
| Strict NVML process peaks | Positive samples in every result |
| Report rejected files | 0 |
| Overall eligible renderers | 5 |
| Local project tests | 115 passed, 1 skipped |
| GitHub CI | Python 3.10 and 3.12 passed |

## Main findings

- gsplat HiGS is the throughput and efficiency winner at 696.91 aggregate FPS
  and a 5.671x reference speed index.
- Speedy-Splat is the balanced recommendation: 2.385x reference speed,
  reference-level aggregate quality, and 4,276 MiB maximum process VRAM.
- TC-GS leads aggregate PSNR and LPIPS while reaching a 2.048x speed index.
- gsplat packed/dense has the lowest full-suite peak VRAM at 4,206 MiB.
- Original 3DGS remains the scientific baseline.

The detailed interpretation is in
[`docs/comparison-analysis.md`](../docs/comparison-analysis.md). Generated
numeric outputs are in [`docs/leaderboard/`](../docs/leaderboard/).

## Evidence layout

Each published run contains only the reviewable JSON evidence needed to verify
the result:

```text
results/measured/<renderer>/<dataset>/<scene>/<run-id>/
  metrics.json
  raw_samples.json
  <renderer>/speed/benchmark_results.json
  <renderer>/speed/nvml_samples.json
  <renderer>/speed/pareto_frontier.json
  <renderer>/speed/recommendations.json
  <renderer>/quality/quality_gt.json
```

Render PNGs, failed attempts, local suite reports, environments, build trees,
and temporary credentials are not publication evidence and were not committed.
