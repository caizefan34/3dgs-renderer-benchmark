# R20 actual source audit

Status: SOURCE_AUDIT_COMPLETE; renderer source patch implemented;
MX complete-build/parity/training NOT_EXECUTED. Date: 2026-10-09.

Task: TRAINABLE-HIGS-ABSGRAD-PARITY-R20-MX.
Repository: [3dgs-renderer-benchmark](https://github.com/caizefan34/3dgs-renderer-benchmark).
Local base commit: `387a7b31f35fa0b9ab1d160fa12181e1a60c77c2`.
Branch created: `feature/r20-higs-true-absgrad`. Existing unrelated dirty files
and submodules were preserved. No frozen source/binary/checkpoint was edited.

## Actual implementation map

All gsplat paths below have the prefix `HIGS_SOURCE_HANDOFF_ACTUAL/source/`.

| Requested implementation | Actual file and evidence |
|---|---|
| Native raster backward | gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu:233-316; formula inline, fabs before warpSum, atomicAdd_system |
| Native allocation/accumulator | gsplat/cuda/csrc/Rasterization.cpp:502-545; conditional zeros_like means2d; native wrapper gsplat/cuda/_wrapper.py:1880-1950 copies backward result into holder |
| HiGS backward | gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu:60-553,1154-1593; dense and PX kernels; compute_abs=false, seven outputs |
| HiGS binding | gsplat/experimental/render/kernels/cuda/ext.cpp:60-96 plus csrc/gaussian_inference/HigsNativeBackward.h:47-95 |
| HiGS autograd | gsplat/experimental/render/functional/gaussian_inference.py:1620-1995,2220-2380,2509-2531; captures actual forward tensors, native backward, visible-ID scatter, proxy gradient |
| Densification adapter | r19_src/r19_trainer.py:60-74,200-220; attach_absgrad_hook uses grad.detach().abs(); signed std_ll hook at 190-199 |
| Training consumer | baseline/reference_v1/gaussian_model.py:260-294,449-599; prefer .absgrad, W/2,H/2, norm, accumulator/denom, clone/split threshold >=, prune after topology changes |
| Thresholds | baseline/reference_v1/config.py:43-57; 0.0008, percent_dense .01, min_opacity .005; R20 does not modify it |
| Build selection | gsplat/experimental/render/kernels/cuda/build.py:70-116; HigsNativeBackward.cu explicitly included |
| Timing | r19_src/r19_trainer.py:332-426; six events and synchronize; r19_src/r19_collect.py:307-325 sums medians |

The CUDA native gradient is with respect to pixel-coordinate means2d, summed
over pixels after componentwise fabs. It is not a vector norm, channelwise
absolute gradient or abs of an already summed gradient. See the semantics file.

## Provenance and missing dependencies

`HIGS_SOURCE_HANDOFF_ACTUAL/SOURCE_MANIFEST.json` identifies the source root:
`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx`.
All 92 exported files were locally SHA256-verified: zero mismatches.
It records frozen main binary SHA256
`40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`
and frozen experimental SHA256
`b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3`.
This associates the export with a stated binary identity; it does NOT prove
recompiling these sources reproduces those binaries or their runtime behavior.

The REAL HiGS .cu, .h, binding and Python wrapper are present. Thus a concrete
kernel patch is supported. However the 92-file export is not a standalone build
tree despite its `missing_required=[]` field. The following exact dependencies
are absent under the exported `source/` root:

- gsplat/cuda/include/Common.h
- gsplat/cuda/include/Utils.cuh
- gsplat/cuda/include/Utils.h
- gsplat/cuda/csrc/Config.h
- gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh
- gsplat/cuda/csrc/third_party/glm/glm/gtc/type_ptr.hpp and the GLM include tree
- gsplat/_lazy_backend.py
- gsplat/__init__.py
- gsplat/rendering.py

The complete MX tree must supply these unchanged files. Preparation checks
critical dependencies and records their hashes, verifies four exact source
preimages, then copies into a new candidate tree. A mismatch stops preparation
and requires re-audit, rather than patching a guessed implementation.

For local compile/scoped GPU checks only, shared includes came from the actual
`public_baselines/gsplat/gsplat/cuda/{include,csrc}` checkout (commit
`512d366b67073d77ca099ede742683c165dfc23b`) and the existing GLM tree at
`wv-hier-r1-mx/evidence/raw/evidence/gsplat_source_snapshot/cuda/csrc/third_party/glm`.
These are not asserted to be the frozen MX dependency hashes. The local shared
blend helper was read and its formula agrees with the exported native inline
formula; the patch retains its existing signed invocation and does not invent
or replace this helper. See `R20_SOURCE_MANIFEST.json` for exact patch pre/post hashes.

## Deliverable and validation boundaries

The overlay under `source/gsplat/...` contains complete patched copies of four
real source files. `R20_SOURCE_PATCH.diff` is a directly applicable unified diff
against that same MX export. Its completion state is SOURCE_PATCH_COMPLETE;
the patch has an informational status preamble accepted by git apply, and the
manifest records status/hashes. R19 compatibility training is a separate `R20_TRAINER.py`.

The eight requested artifacts are present. Supporting tools prepare an isolated
MX tree, build only the experimental extension, generate a hashed real fixture,
and retain validation evidence. No production change is made to the frozen main
extension. CPU tests execute actual modified wrapper methods with a mock native
extension; scoped GPU tests extract actual native/HiGS pixel kernel bodies.
They do not simulate a CUDA parity PASS against an unavailable frozen binary.

Local platform: Windows, PyTorch 2.13.0+cu130, CUDA toolkit 13.3,
RTX 5070 Laptop 8 GiB, MSVC 2022. The Linux frozen MX binaries cannot be loaded
on this Windows host. Frozen MX full-render/raw-gradient parity, A100 timing,
large checkpoint replay and full training remain NOT_EXECUTED. Full status and
commands are in R20_LOCAL_VALIDATION.json and R20_DSH_BUILD_INSTRUCTIONS.md.
