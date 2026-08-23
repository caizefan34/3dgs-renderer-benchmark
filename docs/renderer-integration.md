# Renderer Execution and Gap Analysis

Machine-readable build, runtime, input/output, platform, feature, and status specifications live in `benchmark/renderers.json`.

## Automatic now

The repository has callable adapters for:

- original 3DGS / diff-gaussian;
- gsplat packed and dense;
- gsplat HiGS configurations;
- Speedy-Splat;
- TC-GS.

They share the canonical Graphdeco PLY and camera loader and normalize output to an RGB tensor.
All five automatic paths completed the pinned five-scene Tier A matrix on
EPIC-05. Future results must still use prepared assets, same-case quality, and
the strict result collector; new hardware/software combinations form separate
cohorts.

## Environment integration required

- fast-gaussian-rasterization: adapter exists, but requires Linux/WSL2 EGL and correct per-camera framebuffer state.

## Custom adapter required

- FlashGS: checkpoint mapping, camera mapping, output normalization, build container, and differential quality tests.
- Local-GS/TiCoGS: CUDA extension build, same-checkpoint adapter, and quality validation.
- GEMM-GS: architecture/precision policy, build, adapter, and quality validation.

## Separate track required

- StopThePop: distinguish same-checkpoint renderer behavior from retrained/reduced models and add temporal consistency metrics.

## Prioritized roadmap

1. Reproduce the complete matrix on a second GPU cohort without mixing results.
2. Add a Linux EGL image and repair fast-gaussian per-camera state.
3. Integrate FlashGS, Local-GS, and GEMM-GS one at a time with image-difference regression tests.
4. Add StopThePop as a temporal/native pipeline track.
5. Add a verified browser/WebGPU track before making web-viewer recommendations.

Registered aliases and tuning modes must use distinct `config_id` values.
The diagnostic `speedy_splat_raw` mode is not a product recommendation candidate.
