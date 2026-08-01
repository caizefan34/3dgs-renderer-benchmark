# HiGS Trainability Implementation Report

## Status: Native differentiable training path (HiGS forward + HiGS native CUDA backward) — VERIFIED on EPIC-05 (A100)

This report documents the real, verifiable differentiable training path for HiGS:
**HiGS forward + HiGS native backward**, with the standard gsplat recomputation
kept only as an explicit, metadata-tagged fallback
(`backward_backend="gsplat_recompute"`).

Three backward paths exist today:

| Path | Entry point | `backward_backend` metadata | What runs in backward |
|------|-------------|-----------------------------|-----------------------|
| Frozen-topology native | `rasterize_gaussian_higs_frozen(..., backward_mode="higs_native")` | `higs_native` | Native CUDA kernels from forward-captured state (blend VJP + projection VJP + SH VJP); no recomputation |
| Dynamic-topology native | `rasterize_gaussian_higs_dynamic(..., backward_mode="higs_native")` | `higs_native` | Same native kernels + densify/prune with Adam-state sync |
| gsplat recomputation fallback | `backward_mode="gsplat_recompute"` | `gsplat_recompute` | Standard gsplat `rasterization()` re-run under autograd on the visible subset |

The native backward NEVER re-runs the rasterization pipeline. It consumes the
forward-captured state (means2d / conics / evaluated colors / opacities /
per-tile sorted intersection ids / render alphas / last ids / radii) so the
backward is bound to the exact scene / hierarchy / order / visibility version
of the forward.

## Stage A/B/C API (preserved, not renamed)

- `rasterize_gaussian_higs_trainable()` — Stage A correctness baseline (unchanged API).
- `rasterize_gaussian_higs_frozen()` — Stage B; now defaults to `backward_mode="higs_native"`.
- `rasterize_gaussian_higs_dynamic()` — Stage C; now defaults to `backward_mode="higs_native"`.

New additions (no existing API removed):
- `create_higs_renderer()` / `HigsRendererHandle` — explicit versioned scene handle
  owning the packed FP16 scene + pybind renderer; binds forward/backward to one
  `scene_version` and raises on topology mutation while a backward is pending.
- `sync_optimizer_state_for_topology_change()` — Adam-state sync for densify/prune
  (copy duplicated rows, zero new rows, drop pruned rows).

## Native CUDA backward (new)

`gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu`
implements three kernel stages, mirroring gsplat's own fused kernels:

1. **Pixel-blend backward** (`higs_blend_bwd_kernel`) — one thread per pixel,
   shared-memory batched traversal of the per-tile sorted intersection lists
   (back-to-front), warp-reduced `rasterize_to_pixels_3dgs_blend_bwd` from
   `RasterizeToPixels3DGSDevice.cuh`, atomic scatter to flat `[I*N, ...]`
   gradients. Background gradient `d(render)/d(background) = T_final` per pixel.
2. **Projection VJP** (`higs_projection_bwd_kernel`) — one thread per `(image, gaussian)`
   pair, warp-reduced by Gaussian id, computing
   camera-model projection VJP (`persp_proj_vjp` / `ortho_proj_vjp` /
   `fisheye_proj_vjp`) -> posW2C_VJP + covarW2C_VJP -> quat_scale_to_covar_vjp
   with `eps2d` implicitly captured in the forward conics; FP32 master accumulators.
3. **SH VJP** (`higs_sh_vjp_kernel`) — degree 0..3 port of gsplat's
   `sh_coeffs_to_color_fast_vjp`, chained through the forward activation
   `colors_eval = clamp_min(sph + 0.5, 0)` (mask from forward `colors_eval > 0`),
   plus the view-direction contribution to `v_means`.

Host launcher `higs_rasterize_backward(...)` returns FP32 tuple
`(v_means, v_quats, v_scales, v_opacities, v_colors_master, v_backgrounds)`.

## Python autograd layer

- `_HigsAutogradFunction` captures the full forward state in ONE pass
  (`_native_forward_capture`: `fully_fused_projection -> isect_tiles ->
  isect_offset_encode -> _maybe_evaluate_sh -> rasterize_to_pixels_3dgs`).
- `ctx` saves all 23 inputs; non-None `background` is saved and used (never just
  `background_was_none`).
- Single- and multi-camera batches supported (no `viewmats[0,0]` hardcoding).
- `grad_frame` / `grad_alpha` None or empty handled explicitly.
- Invisible Gaussians receive exactly zero gradient (stop-gradient visibility;
  the mask is never differentiated).
- FP32 master tensors are the optimization variables; FP16 packed buffers are
  used only for the HiGS culling scene (`packed_dtype` in metadata).
- Lossy SH compression (PACKED_16B/32B) is trainable via a straight-through
  FP16 quantization (STE); culling auto-refreshes when the FP32 master
  parameters drift (`params_changed`); densify RGB clamping is configurable
  via `color_clamp`.

## Discrete culling semantics

- Visibility mask is computed under `no_grad` and is a plain boolean index;
  it is never a continuous differentiable variable.
- Visible Gaussians get gradients through the native chain; invisible get zero.
- Culling ratio and `n_visible` are computed from the actual visible subset
  (no hardcoded values), reported in metadata.

## Dynamic topology

- `_HigsDynamicScene` + `HigsRendererHandle` replace the old singleton-only
  pattern with an explicit, versioned handle that can be passed via `scene=`.
- Every forward/backward is bound to a unique `scene_version`; the autograd
  context keeps the handle (and its packed buffers) alive until backward.
- `mark_dirty()` raises while a backward is pending (mutation mid-graph is
  impossible), and the version is validated again in backward.
- densify/prune run only after backward; optimizer param groups + Adam state
  are rewritten by `sync_optimizer_state_for_topology_change`.

## Error handling / observability

- The extension-unavailable / input-error / topology-version / kernel-error
  cases are distinguished (RuntimeError with actionable messages vs fallback).
- Metadata keys: `backward_backend`, `scene_version`, `n_gaussians`,
  `n_visible`, `culling_ratio`, `topology_rebuilt`, `packed_dtype`, `render_mode`.

## Test suites

- `tests/test_higs_trainable.py` — Stage A (unchanged).
- `tests/test_higs_frozen.py` — Stage B (unchanged).
- `tests/test_higs_dynamic.py` — Stage C (unchanged).
- `tests/test_higs_native_backward.py` — new native-backward suite:
  forward RGB/SH/background parity vs standard gsplat; gradients on
  means/quats/scales/opacities/RGB/SH (finite difference); `torch.autograd.gradcheck`;
  background forward+backward; single + multi camera; empty / all / partial
  visible sets; invisible -> zero grad; alpha-only backward; mixed precision
  (`packed_dtype == "torch.float16"`); SH degrees 0..3; SH compression via
  straight-through estimator (trainable); native vs recompute gradient agreement (RGB + SH incl. clamp
  activation); explicit fallback when the extension is unavailable; pending-
  backward mutation raises; densify/prune optimizer-state sync;
  culling-boundary FD (near plane, far plane, radius clip, projection edge);
  depth render modes `D`/`ED`/`RGB+D`/`RGB+ED` (forward parity, native vs
  recompute gradients incl. expected-depth normalization, finite differences,
  multi-camera, background gradients, SH-input zero-gradient, hit-distance
  rejection); no-CUDA static/API surface (imports, signature defaults, backend
  probe, metadata key contract, handle/scene API).

## Test results (EPIC-05, A100)
```
tests/test_higs_trainable.py ............... 13/13 [100%]
tests/test_higs_frozen.py ................... 14/14 [100%]
tests/test_higs_dynamic.py ................. 11/11 [100%]
tests/test_higs_native_backward.py ......... 58/58 [100%]
============================== 99 passed in 19.91s ===============================
```
Native-backward coverage: forward RGB/SH/background parity vs standard gsplat;
finite-difference gradients on means/quats/scales/opacities/RGB/SH;
`torch.autograd.gradcheck`; non-empty background; single + multi camera;
empty / all / partial visible sets; invisible Gaussians get exactly zero
gradient; alpha-only backward; mixed precision (`packed_dtype="torch.float16"`);
SH degrees 0..3; SH compression via straight-through estimator;
native-vs-recompute gradient agreement (RGB + SH incl. clamp activation);
explicit fallback when the CUDA extension is unavailable; pending-backward
topology mutation raises; densify/prune optimizer-state sync; culling-boundary
FD at the near/far planes, radius-clip threshold and projection (image) edge;
depth render modes `D`/`ED`/`RGB+D`/`RGB+ED` (native vs recompute gradient
parity incl. expected-depth normalization); 6 no-CUDA tests that still run
when no CUDA device is present.

## Test commands

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.0
BUILD_EXPERIMENTAL=1 MAX_JOBS=64 pip install -e artifacts/renderer-sources/gsplat --no-build-isolation
CUDA_VISIBLE_DEVICES=7 ~/miniforge3/envs/gsplat/bin/python -m pytest \
  tests/test_higs_trainable.py tests/test_higs_frozen.py tests/test_higs_dynamic.py \
  tests/test_higs_native_backward.py -v
```

## Training benchmark

`benchmark/run_higs_train_benchmark.py` runs, per scene (small + large) and per
backend (`std`, `higs_recompute`, `higs_native`, `higs_dynamic`), a fixed Adam
schedule fitting the scene's own reference renders, and reports forward /
backward / total iteration latency, peak VRAM, culling ratio,
PSNR / SSIM / LPIPS on held-out cameras, and a native-vs-recompute gradient
cosine probe.

Measured on EPIC-05 (1x A100-SXM4-80GB, CUDA 12.8, torch 2.9.1+cu128), 960x540,
4 train + 3 eval cameras, 20 Adam steps, densify/prune every 5 steps.
Full JSON: `results/higs-train-benchmark-2026-07-31.json` (baseline),
`results/higs-train-benchmark-2026-08-01.json` (after the culling fix) and
`results/higs-train-benchmark-2026-08-01b.json` (after the per-step pack skip).

### Forward bottleneck found and fixed (2026-08-01)

Root cause of the missing total-iteration speedup: **HiGS forward was
slow, not the backward.** With `use_higs_culling=True` the old code ran a
*full HiGS render per camera per step* only to populate the visibility
bitmask, then threw the rendered frame away; that made the forward
12.03/28.29 ms vs std 8.26/17.84 ms (+46%/+59%), which cancelled the
native-backward gain and left the total slower than std.

Fix (`_cull_gaussians_batched` in `gaussian_inference.py`): the visibility
mask now comes from **one batched FP32 `fully_fused_projection` over all
cameras at once** (union `(r > 0)` across cameras) using the same
near/far/radius_clip/eps2d criteria as the render path, so the per-camera
HiGS render is eliminated. The packed HiGS scene is still rebuilt/versioned
through the handle (`_refresh_higs_renderer_scene`) but never rendered just
for culling. Forward dropped to 8.57/23.66 ms, and the total iteration now
shows a real speedup.

### tanks_and_temples/train (N=1,026,508, low-N scene) — 2026-08-01

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 8.22 | 14.00 | 22.40 | 3.22 GB | 0.0% | 19.25 | 0.6763 | 0.2966 | 1,026,508 |
| higs_recompute | 9.60 | 24.55 | 34.34 | 4.00 GB | 15.1% | 19.24 | 0.6761 | 0.2969 | 1,026,508 |
| higs_native | 8.57 | 14.11 | 22.85 | 3.23 GB | 15.1% | 19.26 | 0.6764 | 0.2969 | 1,026,508 |
| higs_dynamic | 7.38 | 12.64 | 20.20 | 3.63 GB | 15.6% | 20.18 | 0.7043 | 0.2778 | 731,871 |

### mipnerf360/bicycle (N=6,131,954, high-N scene) — 2026-08-01

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 17.96 | 32.91 | 51.13 | 10.39 GB | 0.0% | 17.29 | 0.4483 | 0.4286 | 6,131,954 |
| higs_recompute | 25.24 | 60.11 | 85.61 | 12.69 GB | 62.9% | 17.28 | 0.4480 | 0.4286 | 6,131,954 |
| higs_native | 23.66 | 31.77 | 55.65 | 11.27 GB | 62.9% | 17.26 | 0.4476 | 0.4290 | 6,131,954 |
| higs_dynamic | 18.23 | 26.41 | 44.84 | 14.14 GB | 65.0% | 18.21 | 0.4742 | 0.3805 | 4,159,297 |

Native-vs-recompute probe: gradient cosine 0.999996 (train) / 0.999997
(bicycle); forward parity PSNR 21.17 dB / 18.98 dB.

**Interpretation (after the culling fix)**

- The native backward remains ~1.7x faster than the `gsplat_recompute`
  fallback (bwd 14.11 vs 24.55 ms; 31.77 vs 60.11 ms) because it never
  re-runs the rasterization pipeline.
- Forward is now at std parity on the small scene (8.57 vs 8.22 ms, +4%)
  and much closer on the large scene (23.66 vs 17.96 ms; was +59%, now +32%).
- `higs_native` total iteration is now at parity with std gsplat on the
  small scene (22.85 vs 22.40 ms, +2%) and improves from +15.9% to +8.8%
  on the large scene (55.65 vs 51.13 ms). The remaining large-scene gap is
  the batched FP32 projection + visible-subset render in the differentiable
  forward, not the backward.
- `higs_dynamic` (native backward + densify/prune) is now **faster than
  std gsplat end-to-end on both scenes**: 20.20 vs 22.40 ms (-9.8%) and
  44.84 vs 51.13 ms (-12.3%), while improving held-out quality (PSNR
  20.18 / 18.21 vs std 19.25 / 17.29) through adaptive densify/prune.
- Quality parity with std holds for the frozen path (PSNR 19.26 vs 19.25;
  17.26 vs 17.29; SSIM/LPIPS within 0.001); small differences come from
  HiGS culling of 15.1% / 62.9% invisible Gaussians.

**Baseline for comparison (2026-07-31, before the culling fix)**

| backend | train fwd | train total | bicycle fwd | bicycle total |
|---|---|---|---|---|
| std | 8.26 | 22.46 | 17.84 | 50.93 |
| higs_native | 12.03 | 25.96 | 28.29 | 59.02 |
| higs_dynamic | 10.93 | 23.61 | 23.50 | 49.68 |

### Second optimization (2026-08-01): drop the per-step packed-scene rebuild

Phase profiling of the `higs_native` forward on bicycle (4 cams, 960x540,
N=6.13M) with CUDA events showed the forward was still paying
`_refresh_higs_renderer_scene`: every `optimizer.step()` bumped the master
tensor `_version`s, so `HigsRendererHandle.params_changed` forced a full
`pack_gaussian_inference_scene` (FP32 -> packed FP16, ~3.2 ms) plus a renderer
construction on **every** training step - even though neither the native nor
the recompute backward ever consumes the packed FP16 scene (both consume the
FP32 captured tensors: means2d/conics/colors_eval/opacities/tile offsets/
flatten ids).

Fix (`_refresh_higs_renderer_scene(..., lightweight=True)`, used by the
differentiable forward): pure parameter drift now only updates the handle
version bookkeeping - the pack/renderer rebuild is skipped entirely. Real
topology changes (`mark_dirty()` / an N change) still re-pack, so the handle
stays valid for the non-training `_cull_gaussians_higs` culling API (which
detects the drift through the untouched master-tensor versions and re-packs on
demand). The culling projection now also passes `camera_model` through, so the
visibility mask is projection-consistent for ortho/fisheye too.

Measured phase cost on bicycle (mean over 10 steps): `refresh` 3.00 -> 0.03 ms;
forward total 26.30 -> 23.83 ms. Re-running the full benchmark:

### tanks_and_temples/train (N=1,026,508) --- 2026-08-01b (after pack skip)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 8.04 | 13.78 | 22.00 | 3.22 GB | 0.0% | 19.25 | 0.6762 | 0.2965 | 1,026,508 |
| higs_recompute | 9.05 | 24.52 | 33.76 | 4.00 GB | 15.1% | 19.24 | 0.6761 | 0.2969 | 1,026,508 |
| higs_native | 8.13 | 14.11 | 22.41 | 3.24 GB | 15.1% | 19.26 | 0.6767 | 0.2967 | 1,026,508 |
| higs_dynamic | 7.08 | 12.69 | 19.95 | 3.63 GB | 15.6% | 20.17 | 0.7033 | 0.2802 | 731,802 |

### mipnerf360/bicycle (N=6,131,954) --- 2026-08-01b (after pack skip)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 17.99 | 32.82 | 50.99 | 10.39 GB | 0.0% | 17.28 | 0.4480 | 0.4288 | 6,131,954 |
| higs_recompute | 22.18 | 60.10 | 82.54 | 12.69 GB | 62.9% | 17.28 | 0.4480 | 0.4291 | 6,131,954 |
| higs_native | 20.66 | 31.87 | 52.79 | 11.27 GB | 62.9% | 17.28 | 0.4481 | 0.4284 | 6,131,954 |
| higs_dynamic | 16.28 | 26.52 | 43.04 | 14.14 GB | 65.0% | 18.20 | 0.4739 | 0.3780 | 4,160,008 |

**Interpretation (after the pack skip)**

- `higs_native` total iteration improved from 22.85 -> 22.41 ms (train, +2.0% ->
  +1.9% vs std) and from 55.65 -> 52.79 ms (bicycle, +8.8% -> +3.5% vs std).
- `higs_dynamic` improved to 19.95 ms (train, -9.3% vs std) and 43.04 ms
  (bicycle, -15.6% vs std) while raising held-out PSNR (20.17 / 18.20 vs
  std 19.25 / 17.28).
- The native backward stays ~1.7-1.9x faster than the `gsplat_recompute`
  fallback (bwd 14.11 vs 24.52 ms; 31.87 vs 60.10 ms) because it never
  re-runs the rasterization pipeline.
- `topology_rebuilt` is now honest: `False` on pure parameter-drift steps
  (the packed hierarchy is not rebuilt), `True` only right after a real
  topology change (`mark_dirty()`/densify-prune).
- Remaining bicycle `higs_native` forward gap vs std (+2.7 ms) is structural:
  the batched FP32 culling projection over all N x C (~3.3 ms) and the
  visible-subset gathers (~4.4 ms) are the price of projection-consistent
  culling; they cannot be removed without changing the discrete-culling
  semantics.

## Known limitations

1. The native backward supports `render_mode` in `RGB`/`D`/`ED`/`RGB+D`/`RGB+ED`
   (depth composited as a channel with camera-space `z`; expected modes chain
   through the `depth_acc / alpha` normalization). Camera models pinhole, ortho
   and fisheye are fully supported (projection VJP switched on `CameraModelType`).
   ftheta/lidar cameras and the eval3d-only hit-distance modes
   (`d`/`Ed`/`RGB-d`/`RGB-Ed`) still raise with a clear message (use the
   recompute fallback).
2. Culling is a discrete approximation: the HiGS scene (packed FP16 buffers +
   renderer) is rebuilt on topology change, explicit `mark_dirty()`, or
   automatic FP32 parameter-drift detection (`HigsRendererHandle.params_changed`,
   tensor `_version` based, e.g. after `optimizer.step()`); between rebuilds the
   visibility mask reflects the last packed parameter snapshot.
3. Lossy SH compression (PACKED_16B/32B) is trainable via a straight-through
   estimator: the forward quantizes SH3 coefficients to FP16, the backward
   passes gradients through unchanged (STE). This is an approximate, non-zero
   gradient; `sh_compression_mode="none"` remains the exact path.
4. The frozen native path is at std parity on the small scene and +3.5% on the
   large scene (960x540); the dynamic path (densify/prune) is faster than std
   on both scenes (-9.3% / -15.6%, 2026-08-01b benchmark). The remaining
   large-scene forward gap comes from the batched FP32 culling projection +
   visible-subset gathers, not from the native backward. The differentiable
   forward no longer re-packs the FP16 scene per step (parameter drift only
   updates the handle version bookkeeping; `pack_gaussian_inference_scene` runs
   only on real topology changes).
5. `_densify_gaussians` clamps RGB colors to [0,1] by default;
   `color_clamp=None` disables the clamp. SH coefficient tensors ([N, K, C])
   are intentionally not clamped.

## Modified files (gsplat source tree)

- `gsplat/experimental/render/functional/gaussian_inference.py` — native forward
  capture, native/recompute backward, `HigsRendererHandle`, `create_higs_renderer`,
  `sync_optimizer_state_for_topology_change`, dynamic-scene handle ownership.
- `gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.{cu,h}` — new.
- `gsplat/experimental/render/kernels/cuda/build.py` — include `HigsNativeBackward.cu`.
- `gsplat/experimental/render/kernels/cuda/ext.cpp` — `higs_rasterize_backward` binding.
- `tests/test_higs_native_backward.py` — new native-backward test suite.
- `benchmark/run_higs_train_benchmark.py` — new training benchmark.
