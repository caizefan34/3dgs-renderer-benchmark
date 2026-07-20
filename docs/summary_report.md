# Project Status Summary

## Current capability

- Canonical five-case Matrix v2 suite with pinned checkpoints, cameras, GT
  manifests, protocol, renderer commits, and result schema.
- Complete EPIC-05 Tier A baseline for original 3DGS, gsplat, gsplat HiGS,
  Speedy-Splat, and TC-GS.
- Strict per-process NVML peak collection with retained raw samples.
- Coupled speed and 100-view GT quality measurements for every published row.
- Generated JSON, CSV, Markdown, and Pareto SVG outputs with evidence tiers kept
  separate.
- Human-readable comparison analysis and renderer decision guide derived from
  the generated ranking.

## Architectural state

- `benchmark/` owns protocol and registry data.
- `src/` owns execution, collection, validation, aggregation, and publication.
- `results/measured/` contains immutable JSON evidence, not generated images or
  local build artifacts.
- `docs/leaderboard/` is the sole public generated ranking location.
- `reports/` contains current run reports; historical machine investigations
  are isolated under `reports/archive/`.

## Current conclusions

- gsplat HiGS: maximum throughput and efficiency.
- Speedy-Splat: best balanced recommendation with reference-level quality.
- TC-GS: best aggregate PSNR and LPIPS.
- gsplat: lowest full-suite peak VRAM.
- Original 3DGS: scientific reference path.

See [`comparison-analysis.md`](comparison-analysis.md) for the complete trade-off
discussion and limitations.

## Next priorities

1. Reproduce the matrix on a second GPU cohort without mixing leaderboards.
2. Add a self-hosted GPU workflow for intentional, reviewed baseline refreshes.
3. Integrate FlashGS, Local-GS, GEMM-GS, and fast-gaussian one at a time.
4. Add temporal consistency and browser/WebGPU tracks before making claims in
   those areas.
5. Add documentation consistency checks against generated ranking metadata.
