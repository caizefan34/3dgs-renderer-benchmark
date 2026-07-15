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
Tier A still requires the pinned environment, prepared five-scene assets, same-case quality run, and strict result collector.

## Environment integration required

- fast-gaussian-rasterization: adapter exists, but requires Linux/WSL2 EGL and correct per-camera framebuffer state.

## Custom adapter required

- FlashGS: checkpoint mapping, camera mapping, output normalization, build container, and differential quality tests.
- Local-GS/TiCoGS: CUDA extension build, same-checkpoint adapter, and quality validation.
- GEMM-GS: architecture/precision policy, build, adapter, and quality validation.

## Separate track required

- StopThePop: distinguish same-checkpoint renderer behavior from retrained/reduced models and add temporal consistency metrics.

## Prioritized roadmap

1. Complete reproducible original 3DGS and gsplat builds and all five Tier A cases.
2. Pin Speedy-Splat and repair the TC-GS container so it actually installs the extension.
3. Qualify one fixed HiGS configuration; keep adaptive and SH/tile ablations out of the default recommendation table.
4. Add a Linux EGL image and repair fast-gaussian per-camera state.
5. Integrate FlashGS, Local-GS, and GEMM-GS one at a time with image-difference regression tests.
6. Add StopThePop as a temporal/native pipeline track.

Registered aliases and tuning modes must use distinct `config_id` values.
The diagnostic `speedy_splat_raw` mode is not a product recommendation candidate.
