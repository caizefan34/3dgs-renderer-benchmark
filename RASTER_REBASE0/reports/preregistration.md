# RASTER-REBASE0 Preregistration

**Date**: 2026-10-06  
**Objective**: Compare Hahlbohm Faster-GS vs ACM-MM-2026 FasterGS backward pass performance

## Baselines

**H (H-FasterGS)**: Hahlbohm et al., CVPR 2026  
- Repo: Hahlbohm/Faster-GS  
- Focus: Faster-GS + FasterGSFused (validated from FUSED-HOTSPOT0)

**M (MM-FasterGS)**: GoogolplexGoodenough/FasterGS, ACM MM 2026  
- Repo: GoogolplexGoodenough/FasterGS  
- Focus: Workload-balanced rasterization

## Key Hypotheses

**H1**: MM workload balancing substantially reduces absolute backward blend latency  
**H2**: MM parallel compositing reduces backward fraction below major-hotspot threshold  
**H3**: Even after MM parallelization, backward remains a dominant production residual

## Required Outputs

- [ ] RASTER_REBASE0_PREREG.json
- [ ] RASTER_REBASE0_ENVIRONMENT.json
- [ ] RASTER_REBASE0_SOURCE_MAP_H.json
- [ ] RASTER_REBASE0_SOURCE_MAP_M.json
- [ ] RASTER_REBASE0_MECHANISM_OVERLAP.json
- [ ] RASTER_REBASE0_COMMON_SEMANTICS.json
- [ ] RASTER_REBASE0_INTEGRATION_PATCH.json
- [ ] RASTER_REBASE0_CORRECTNESS.json
- [ ] RASTER_REBASE0_RENDERER_TIMING.json
- [ ] RASTER_REBASE0_ITERATION_TIMING.json
- [ ] RASTER_REBASE0_NATIVE_SUPPLEMENT.json
- [ ] RASTER_REBASE0_WORKLOAD_STATS.json
- [ ] RASTER_REBASE0_H_HOTSPOT_PROFILE.json
- [ ] RASTER_REBASE0_M_HOTSPOT_PROFILE.json
- [ ] RASTER_REBASE0_M_AMDAHL.json
- [ ] RASTER_REBASE0_RESOURCE_USAGE.json
- [ ] RASTER_REBASE0_FRONTIER_COMPARISON.json
- [ ] RASTER_REBASE0_VERDICT.json
- [ ] RASTER_REBASE0_REPORT.md
- [ ] MANIFEST.sha256

## Phases

- [ ] R0: Source and method audit
- [ ] R1: Environment setup
- [ ] R2: Common-host integration
- [ ] R3: Correctness validation
- [ ] R4: Performance benchmarking
- [ ] R5: Analysis and reporting

## Decision Gates

Before proceeding to expensive operations:
- [ ] Confirm both repos build successfully
- [ ] Verify correctness tests pass
- [ ] Profile at least one common-host run
