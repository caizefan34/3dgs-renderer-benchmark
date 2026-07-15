# Benchmark Methodology

## Scientific question

The primary benchmark asks: **which renderer executes the same 3D Gaussian representation fastest without changing its output quality?**

The primary `common_representation` track fixes the checkpoint, Gaussian count, SH degree, camera order, resolution, color space, background, ground-truth images, protocol, hardware cohort, and software identity.
A renderer-native track may evaluate pruning, retraining, reduced precision, or approximation, but it is a separate experiment and ranking.

## Evidence priority

1. **Tier A — Measured:** produced by this repository's canonical local runner with hashed raw samples.
2. **Tier B — Reproduced:** produced from a pinned official implementation with raw samples and deviations.
3. **Tier C — Paper Reported:** citation-backed values with table/figure/page location.

Recommendations use the highest available tier in that order.
Ranking tables never combine tiers.
Paper transcription is fallback context, not the benchmark's primary data source.

## Comparable cohort

Rows may share a table only when all of these match:

- evidence tier;
- suite, version, track, and case;
- dataset manifest and evaluation split;
- checkpoint hash, Gaussian count, and SH degree;
- camera and ground-truth manifest hashes;
- resolution, color space, background, and antialiasing policy;
- protocol hash;
- hardware profile, driver, CUDA runtime, and clock policy;
- renderer semantic configuration and metric implementation versions.

Renderer identity is not part of the cohort key; otherwise every renderer would form a one-row cohort.

## Execution protocol

The canonical values live only in `benchmark/protocol.json`.
Documentation and code must read that file rather than repeat defaults.

- Launch each renderer/case in a fresh process.
- Rotate renderer order deterministically by case index; the canonical five-case all-renderer run forms a balanced Latin rotation.
- Warm up 30 frames.
- Measure 100 frames per repeat for 5 repeats.
- Synchronize GPU work at measurement boundaries.
- Retain ordered per-frame GPU and end-to-end wall samples.
- Compute throughput as total frames divided by total synchronized wall time.
- Sample NVML process memory; framework allocated/reserved memory is secondary.

Cold-path metrics are distinct:

- parent launch to child ready, plus renderer initialization;
- shared scene read and parse;
- renderer-specific preparation/upload;
- time to first valid frame.

Shared scene loading is not presented as a renderer-specific optimization.

## Performance metrics

- FPS from synchronized end-to-end samples;
- arithmetic mean, P95, and P99 frame time;
- peak process VRAM and framework peaks;
- renderer startup time;
- scene read/parse time;
- renderer preparation/upload time;
- time to first correct frame.

GPU-kernel latency may be reported as a diagnostic but does not replace end-to-end FPS.

## Quality metrics

Every timed camera belongs to the same ordered evaluation manifest used for quality:

- PSNR: macro-mean of per-view RGB PSNR;
- SSIM: macro-mean of per-view RGB SSIM;
- LPIPS: macro-mean of per-view LPIPS-VGG.

Missing GT metrics exclude a row from speed-quality, quality, efficiency, and Pareto rankings.
Synthetic stress scenes are diagnostics only.

## Aggregation

Overall rankings require every mandatory suite case.
Per-scene tables remain primary evidence.

- Speed index: geometric mean of `renderer FPS / original_3dgs FPS` per case.
- FPS display: geometric mean across required cases.
- PSNR, SSIM, LPIPS: equal-scene macro means.
- VRAM: maximum across required cases.
- Startup and load: median across required cases.

The report also publishes quality deltas against the reference renderer.

## Efficiency

The efficiency score is a transparent decision aid, not a universal scientific constant:

```text
Qpsnr  = clamp((PSNR - 20) / 20, 0, 1)
Qssim  = clamp((SSIM - 0.8) / 0.2, 0, 1)
Qlpips = clamp((0.4 - LPIPS) / 0.4, 0, 1)
quality_utility = geometric_mean(Qpsnr, Qssim, Qlpips)
resource_cost = frame_time_ms * peak_VRAM_GiB
efficiency = quality_utility / resource_cost
```

Constraint rankings such as “fastest within 0.1 dB of reference and under 4 GiB” should be preferred when making deployment decisions.

## Pareto analysis

Each exact cohort and each complete-suite aggregate receives separate frontiers:

1. FPS vs PSNR: maximize both.
2. FPS vs LPIPS: maximize FPS and minimize LPIPS.
3. Combined: maximize FPS, PSNR, SSIM and minimize LPIPS.

The frontier drawn on a 2D chart is computed in that 2D space, not projected from the combined frontier.
The practical-tolerance policy in `benchmark/protocol.json` treats near-equal measurements as ties when robust frontiers are added.

## Failure policy

OOM, unsupported, failed, and incomplete attempts are retained without fabricated metrics.
They rank after successful rows and remain visible in coverage reports.
