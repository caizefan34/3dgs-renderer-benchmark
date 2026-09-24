# H1-B2 source freeze

## Result

Trainable HiGS source reconstruction, isolated build, ABI gate, and room/cam0 regression passed. No optimization or AccuTile changes were introduced. The authoritative cohort is not complete because the target A100 remained saturated by unrelated jobs; consequently this report makes no performance claim.

## Frozen identity

- Base: `77ab983ffe43420b2131669cb35776b883ca4c3c` (tree `0a8487ad41b2ccfb4dc066fd7ffcb82f2a21eb86`).
- Research repository: `02375033388d4348376b6b607ab85f551e498a77`.
- B2 patch: [`patches/higs-trainable-authoritative.patch`](../../patches/higs-trainable-authoritative.patch), SHA256 `74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c`.
- `B2_SOURCE_ID = 77ab983ffe43420b2131669cb35776b883ca4c3c + 74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c`.
- 15 required files are frozen in the patch. The other 14 modified tracked files in the research checkout are classified `EXPERIMENTAL_UNRELATED`; six untracked entries were inventoried and none is an unexplained runtime source dependency. See `artifacts/h1-b2-freeze/source_delta.json` and `patch_manifest.json` for file-level records.
- Existing `patches/higs-differentiable.patch` is `PATCH_STALE`: `git apply --check` failed against the exact base. The new patch applies to the exact base.

The one compatibility adjustment in the frozen B2 Python wrapper omits the optional `tile_mask=None` keyword when using Full mode (`tile_sampling_ratio=1.0`), because the validated immutable B1A core wrapper does not expose that keyword. Ratios below 1 fail closed rather than changing core APIs. This preserves the validated AccuTile/core path and Full-mode behavior; it is not an algorithm change.

## Dependency/build provenance

Base tree GLM gitlink is `2d4c4b4dd31fde06cfffad7915c2b3006402322f`. The retained payload at `/mnt/storage_pool/liaoyuanjun/higs-glm-2d4c4b4` has no Git metadata, but its 1,665 payload files matched the copied GLM tree file-for-file by SHA256; sorted-path manifest SHA256 is `dec4471246f76ec522a126f12bc43ba5773ce1b40c0547125d0c95654c5317a3`.

Build passed from `/tmp/h1_b2_authoritative_source` into isolated cache `/tmp/h1_b2_authoritative`, using Torch `2.9.1+cu128`, CUDA 12.8, nvcc `/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/nvcc` V12.8.93, and `compute_80,code=sm_80`. Build.ninja source paths and CUDA 12.8 include paths were checked. Three cache-built extensions are recorded in `artifacts/h1-b2-freeze/build_manifest.json`:

- `gsplat_cuda.so`: SHA256 `361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98`.
- `experimental_gaussian_render_inference_scene_cuda.so`: SHA256 `00d7899d29e81269a98a409b83b441c805b4118a33d4764c61deef349df3f38a`.
- `gsplat_scene_cuda.so`: SHA256 `7389105286de5709054edba25f5af519762f9955ef0830530e750f71dbc3c031`.

## Gates

ABI gate passed. Runtime schemas and Python/C++/CUDA parameter order are recorded in `artifacts/h1-b2-freeze/abi_audit.json`; the former positional shift (`Python position 4=viewmats` against a binary position 4=`opacities`) is absent. The wrapper explicitly maps optional opacities into schema position 4.

Room/cam0 B2 regression passed: RGB relative L2 `3.057e-8`, alpha exact; means/quats/scales/opacity/SH gradients passed project tolerance (maximum relative L2 `6.1253e-4`, cosine at least `0.99999988`, no zero/nonzero disagreement, NaN, or Inf). See `b2_regression.json`.

B1A regression passed against a direct replay of the previously validated B1A binary: both yield 44,844 visible and 953,085 intersections, 11,008 active tiles. The earlier smoke record’s 44,908/953,144 values correspond to B2 metadata, not B1A raster structural counts; this provenance discrepancy is documented in `b1a_regression.json`. The pathological billion-intersection scale is absent.

## Cohort status

No 3-scene × 3-camera timing run was started. Final GPU snapshot: 38,143 MiB used / 2,299 MiB free, 100% utilization, with unrelated PIDs 220824 and 295572 using 19,318 and 18,810 MiB. They were not interrupted. Cohort CSVs therefore say `NOT_RUN`; speedups/geomeans are null and current-performance classification is `NOT_CLASSIFIED`.
