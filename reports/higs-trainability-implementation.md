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
tests/test_higs_native_backward.py ......... 61/61 [100%]
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

Current EPIC-05 environment (CUDA 12.9, torch 2.9.1+cu128, GPU 1). Two environment
caveats were hit on 2026-08-02 and are required for a green run:

```bash
cd /root/3dgs-roadmap-matrix
mv artifacts/renderer-sources/gsplat/pytest.ini artifacts/renderer-sources/gsplat/pytest.ini.bak
PATH=/root/miniforge3/envs/gsplat/bin:/usr/local/cuda-12.9/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.9 TORCH_DONT_CHECK_COMPILER_ABI=1 CUDA_VISIBLE_DEVICES=1 \
/root/miniforge3/envs/gsplat/bin/python -m pytest \
  tests/test_higs_native_backward.py tests/test_higs_frozen.py \
  tests/test_higs_dynamic.py tests/test_higs_trainable.py -q
mv artifacts/renderer-sources/gsplat/pytest.ini.bak artifacts/renderer-sources/gsplat/pytest.ini
```

- `TORCH_DONT_CHECK_COMPILER_ABI=1` is required: torch 2.9.1's
  `WRONG_COMPILER_WARNING` has 6 `%s` placeholders but is logged with 4 args,
  so the resulting logging `TypeError` is swallowed in plain python but
  propagates under pytest's log-capture handler and fails the JIT load of
  `gsplat_scene_cuda` / the experimental extension (observed as 11 spurious
  failures; verified 99 passed with the env var set).
- The gsplat `pytest.ini` `addopts = -p pytest_check` references a plugin that
  is not installed in this env; temporarily moving `pytest.ini` aside avoids
  the import error.

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
`results/higs-train-benchmark-2026-08-01b.json` (after the per-step pack skip) and
`results/higs-train-benchmark-2026-08-01c.json` (after the native visible-subset gather) and
`results/higs-train-benchmark-2026-08-01e.json` (after the SH VJP / scatter optimization),
`results/higs-train-benchmark-2026-08-01f.json` (after the blend VJP launch-bounds fix)
and `results/higs-train-benchmark-2026-08-01h.json` (after the C++ direct-master
gradient scatter). `results/higs-train-benchmark-2026-08-01i.json` captured a
single-pass projection experiment that was measured as a regression and
reverted; `results/higs-train-benchmark-2026-08-01j.json` confirms the reverted
state matches 08-01h. `results/higs-train-benchmark-2026-08-01l.json` captured the
round-8 camera-row slice experiment (fused mask + slice in tree, regression vs the
fused-mask-only state, slice reverted); `results/higs-train-benchmark-2026-08-01m.json` is the
fused union-visibility-mask final result. 
esults/higs-train-benchmark-2026-08-02-shvjp.json and

esults/higs-train-benchmark-2026-08-02-train.json are the round-9
fixed-grid SH VJP results (bicycle / train).

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
- Remaining bicycle `higs_native` forward gap vs std (+2.7 ms) was dominated
  by the visible-subset gathers (~4.4 ms) plus the batched FP32 culling
  projection over all N x C (~3.3 ms). The gather part is now eliminated by
  the native gather kernel (third optimization below); the culling
  projection remains the price of projection-consistent discrete culling.


### Third optimization (2026-08-01): native visible-subset gather kernel

Phase profiling of the `higs_native` forward on bicycle (4 cams, 960x540,
N=6.13M) still showed a 4.42 ms cost for gathering the visible-subset master
rows (`means[vis_ids]`, `quats[vis_ids]`, ...). An inner-dimension sweep of
PyTorch's row gather found the root cause: the CUDA index-select dispatches a
vectorized path whenever the row width is a multiple of four floats, and that
path is pathologically slow for random row indices on large tensors - ~1.70 ms
for `quats [N,4]` and ~1.70 ms for `colors [N,16,3]` (48 floats/row), vs
0.05-0.28 ms for non-multiple-of-4 widths (measured on EPIC-05, 2.27 M visible
rows). `torch.gather`, `index_select` and int32 indices all hit the same path.

Fix: a new single-purpose compact-copy kernel `higs_gather_visible`
(`GatherVisible.cu`/`.h`) copies the five FP32 master tensors (means/quats/
scales/opacities/colors, arbitrary trailing shape) into fresh contiguous
visible-subset tensors with one element per thread, bypassing the bad PyTorch
dispatch. Python falls back to the plain PyTorch gather when the extension is
absent or the tensors are not CUDA FP32. Measured: 5-gather 3.79-4.42 ms ->
0.93 ms (quats 1.70 -> 0.10 ms; colors 1.70 -> 0.65 ms); values bit-identical
to `t[vis_ids].contiguous()`. Forward dropped from 23.83 -> 20.14 ms on
bicycle (`higs_native`). Re-running the full benchmark:

### tanks_and_temples/train (N=1,026,508) --- 2026-08-01c (after gather kernel)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 8.16 | 14.17 | 22.51 | 3.22 GB | 0.0% | 19.24 | 0.6763 | 0.2966 | 1,026,508 |
| higs_recompute | 8.08 | 24.48 | 32.74 | 4.00 GB | 15.1% | 19.24 | 0.6763 | 0.2966 | 1,026,508 |
| higs_native | 7.12 | 14.11 | 21.40 | 3.24 GB | 15.1% | 19.26 | 0.6765 | 0.2969 | 1,026,508 |
| higs_dynamic | 6.24 | 12.65 | 19.07 | 3.63 GB | 15.6% | 20.19 | 0.7039 | 0.2806 | 731,783 |

### mipnerf360/bicycle (N=6,131,954) --- 2026-08-01c (after gather kernel)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 17.85 | 33.21 | 51.26 | 10.39 GB | 0.0% | 17.28 | 0.4478 | 0.4289 | 6,131,954 |
| higs_recompute | 19.30 | 59.86 | 79.34 | 12.69 GB | 62.9% | 17.29 | 0.4481 | 0.4284 | 6,131,954 |
| higs_native | 17.74 | 31.64 | 49.56 | 11.27 GB | 62.9% | 17.27 | 0.4478 | 0.4286 | 6,131,954 |
| higs_dynamic | 14.07 | 26.38 | 40.62 | 14.14 GB | 65.0% | 18.22 | 0.4746 | 0.3792 | 4,159,824 |

**Interpretation (after the gather kernel)**

- `higs_native` total iteration is now **faster than std gsplat end-to-end on
  both scenes**: 21.40 vs 22.51 ms (train, -4.9%) and 49.56 vs 51.26 ms
  (bicycle, -3.3%). The bicycle forward is at std parity (17.74 vs 17.85 ms)
  and the native backward is faster than std's own backward (31.64 vs 33.21 ms).
- `higs_dynamic` improves to 19.07 ms (train, -15.3%) and 40.62 ms (bicycle,
  -20.8%) vs std while keeping the held-out PSNR gains (20.19 / 18.22 vs
  std 19.24 / 17.28).
- Native-vs-recompute probe stays at gradient cosine 0.999996 / 0.999997;
  quality metrics are unchanged (frozen path bit-level parity aside from the
  culled 15.1% / 62.9% invisible Gaussians), so the gather kernel is a pure
  data-movement optimization.
- All 99 tests pass with the new kernel; the PyTorch gather remains the
  automatic fallback whenever the CUDA extension is not available.


### Backward profiling and SH VJP optimization (2026-08-01e)

With the forward at std parity, the remaining target was the native backward
itself. Profiling the 6.1M-Gaussian bicycle backward (4 cams, 960x540, L1
loss) with `torch.profiler` + CUDA events split the ~37 ms backward into:

| stage | before | after | notes |
|---|---|---|---|
| blend VJP (+ flat-buffer alloc) | 21.6 ms | 21.6 ms | same algorithm as std `rasterize_to_pixels_3dgs_bwd` (19-20 ms) |
| SH VJP | 8.3 ms | 5.9 ms | visible-pair compaction + one thread per (pair, channel) |
| projection VJP | 1.6 ms | 1.6 ms | |
| Python-side scatter (`zeros_like` + `index_add_`) | 5.4 ms | 4.5 ms | `index_add_` -> `index_copy_` (visible ids are duplicate-free) |

Three experiments pinned down the SH VJP fix:

1. `atomicAdd_system` -> `atomicAdd` (device atomics): no change (reverted).
2. Compacting the visible (camera, gaussian) pairs so the VJP launches only
   the visible subset: no change -- the full-grid prologue was already cheap;
   the VJP is bound by the ~109M coefficient atomics, not the launch shape.
3. Splitting the 3 output channels across a 2D (pair, channel) grid:
   **slower** (12.8 ms), because a Gaussian's coefficient atomics then land
   in different blocks / SMs.

The winning layout matches gsplat's `spherical_harmonics_bwd`: a 1D grid
ordered as (visible pair, channel) so the D=3 channel threads of one pair are
adjacent in the same warp, keeping the 48 coefficient atomic updates of a
Gaussian on the same cache lines. Combined with the pair compaction
(`higs_vis_block_counts_kernel` + `cub::exclusive_sum` +
`higs_vis_pairs_kernel`) this dropped the SH VJP from 8.3 to 5.9 ms
(std: 4.8 ms). The `index_copy_` swap removed the read-modify-write reduction
on the duplicate-free visible ids and saved another ~1 ms.

### tanks_and_temples/train (N=1,026,508) --- 2026-08-01e (after backward opt)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 7.8 | 13.6 | 21.6 | 3.22 GB | 0.0% | 19.25 | 0.6762 | 0.2967 | 1,026,508 |
| higs_recompute | 8.2 | 24.5 | 32.8 | 4.00 GB | 15.1% | 19.25 | 0.6763 | 0.2966 | 1,026,508 |
| higs_native | 7.1 | 13.0 | 20.4 | 3.24 GB | 15.1% | 19.24 | 0.6764 | 0.2966 | 1,026,508 |
| higs_dynamic | 6.3 | 11.8 | 18.3 | 3.63 GB | 15.6% | 20.32 | 0.7059 | 0.2771 | 731,367 |

### mipnerf360/bicycle (N=6,131,954) --- 2026-08-01e

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 17.8 | 32.8 | 50.8 | 10.39 GB | 0.0% | 17.28 | 0.4480 | 0.4291 | 6,131,954 |
| higs_recompute | 19.3 | 59.9 | 79.4 | 12.69 GB | 62.9% | 17.29 | 0.4481 | 0.4288 | 6,131,954 |
| higs_native | 17.8 | 28.3 | 46.3 | 11.27 GB | 62.9% | 17.28 | 0.4479 | 0.4286 | 6,131,954 |
| higs_dynamic | 14.1 | 23.9 | 38.1 | 14.14 GB | 65.0% | 18.21 | 0.4733 | 0.3788 | 4,160,370 |

**Interpretation (after the backward optimization)**

- `higs_native` total iteration is now **-8.9% on bicycle** (46.3 vs 50.8 ms)
  and **-5.6% on train** (20.4 vs 21.6 ms) vs std gsplat. The native backward
  alone is -14% vs std's own backward on bicycle (28.3 vs 32.8 ms) and ~2.1x
  faster than the `gsplat_recompute` fallback (28.3 vs 59.9 ms).
- `higs_dynamic` improves further to 18.3 ms (train, -15.3%) and 38.1 ms
  (bicycle, -25.0%) vs std while keeping the held-out PSNR gains
  (20.32 / 18.21 vs std 19.25 / 17.28).
- The blend VJP (~21.6 ms, ~69% of the kernel bundle) is already the same
  algorithm as std's `rasterize_to_pixels_3dgs_bwd` (19-20 ms); the residual
  gap is launch/measurement overhead, not an algorithmic difference. The
  remaining backward headroom is small: SH VJP 5.9 ms (std 4.8 ms) plus
  ~4.5 ms of Python-side scatter.
- Native-vs-recompute probe stays at gradient cosine 0.99999998-1.000000 and
  quality metrics are unchanged; all 99 tests pass.

### Fifth optimization (2026-08-01f): blend VJP register/occupancy fix

The remaining blend VJP (~21.6 ms, ~69% of the kernel bundle) looked
structurally identical to std `rasterize_to_pixels_3dgs_bwd`, so the earlier
report attributed the 2.5 ms gap to launch/measurement overhead. This round
proved the gap was real and pinned it down:

- **Same data, same isects**: both kernels process 14.13M isects on bicycle
  (HiGS n_isects on the 2.27M visible subset is 14,125,785 vs std 14,123,181
  on the full set; the native capture runs `isect_tiles` over the visible
  subset with identical radius/eps2d criteria).
- **Cross-kernel A/B on identical inputs**: feeding the same captured tensors
  through std's `rasterize_to_pixels_3dgs_bwd` and through
  `higs_blend_bwd_kernel` gave std 19.1-19.2 ms vs HiGS 21.6-21.7 ms
  (interleaved, stable across reps) - a real kernel-level difference.
- **SASS + resource dump**: identical math (`rasterize_to_pixels_3dgs_blend_bwd`
  is `__forceinline__` and matches the std inlined body), same build flags
  (-O3, -use_fast_math), but HiGS compiled to **REG:40 -> 6 blocks/SM** while
  std uses **REG:48 -> 5 blocks/SM** on sm_80. Experiments ruled out the
  atomic scope (`atomicAdd_system` swap: no change) and lower occupancy
  (`__launch_bounds__(256, 4)` at 64 regs: 20.6 ms, worse).

Fix: `__launch_bounds__(256, 5)` on `higs_blend_bwd_kernel` lets nvcc use
the same 48-register budget as std's kernel. Blend VJP 21.6 -> 19.1 ms
(matches std exactly), backward bundle 33.8 -> 31.3 ms in same-process
profiling.

### tanks_and_temples/train (N=1,026,508) --- 2026-08-01f (after blend fix)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 8.1 | 14.2 | 22.5 | 3.22 GB | 0.0% | 19.25 | 0.6762 | 0.2967 | 1,026,508 |
| higs_recompute | 8.1 | 24.5 | 32.8 | 4.00 GB | 15.1% | 19.25 | 0.6763 | 0.2965 | 1,026,508 |
| higs_native | 7.1 | 12.3 | 19.6 | 3.24 GB | 15.1% | 19.25 | 0.6763 | 0.2965 | 1,026,508 |
| higs_dynamic | 6.3 | 10.9 | 17.4 | 3.63 GB | 15.6% | 20.31 | 0.7058 | 0.2779 | 731,614 |

### mipnerf360/bicycle (N=6,131,954) --- 2026-08-01f

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 17.9 | 32.9 | 51.0 | 10.39 GB | 0.0% | 17.28 | 0.4479 | 0.4289 | 6,131,954 |
| higs_recompute | 19.5 | 60.1 | 79.8 | 12.69 GB | 62.9% | 17.28 | 0.4481 | 0.4288 | 6,131,954 |
| higs_native | 18.0 | 27.2 | 45.5 | 11.27 GB | 62.9% | 17.28 | 0.4479 | 0.4290 | 6,131,954 |
| higs_dynamic | 14.2 | 22.3 | 36.6 | 14.14 GB | 65.0% | 18.19 | 0.4739 | 0.3799 | 4,160,121 |

**Interpretation (after the blend fix)**

- `higs_native` total iteration vs std gsplat: **-12.9% on train** (19.6 vs
  22.5 ms) and **-10.8% on bicycle** (45.5 vs 51.0 ms). The native backward
  alone is -13.4% (12.3 vs 14.2 ms) / -17.3% (27.2 vs 32.9 ms) vs std's own
  backward, and **~2.0x / ~2.2x faster than `gsplat_recompute`** (24.5 / 60.1
  ms) - the recompute fallback re-runs the rasterization pipeline inside
  backward.
- `higs_dynamic` improves to 17.4 ms (train, -22.7%) and 36.6 ms (bicycle,
  -28.2%) vs std, keeping the held-out PSNR gains (20.31 / 18.19 vs std
  19.25 / 17.28).
- Blend VJP is now bit-for-bit the same algorithm as std at the same register
  budget; quality metrics and the gradient-cosine probe are unchanged
  (1.0), 99 tests pass.

### Sixth optimization (2026-08-01h): C++ direct-master gradient scatter

The kernel bundle was already at std parity, so this round attacked the
remaining Python-side scatter. The native backward used to return
visible-subset gradients and let autograd scatter them into the FP32 master
tensors with five `torch.zeros_like` + five `index_copy_` calls (about
1.1 ms of `index_copy_` plus the per-call launch/alloc overhead, and an
extra ~435 MB temporary `v_colors_master` allocation on bicycle).

`higs_rasterize_backward` now receives the pre-zeroed master gradient tensors
plus `visible_ids` (the strictly increasing, duplicate-free culling mask) and
writes every gradient directly into the master rows:

- `higs_projection_bwd_kernel` / `higs_sh_vjp_kernel` scatter their
  `atomicAdd` targets through `visible_ids[g]` (means/quats/scales/SH
  coefficients);
- `higs_reduce_master_kernel` writes the per-view color/opacity reductions
  at `visible_ids[g]`;
- the Python side keeps the five `zeros_like` allocations (invisible rows must
  stay zero) but drops the five `index_copy_` scatter launches.

Measured effect (bicycle, 4 cams, 960x540, same-process kernel profile):
`index_copy_` and the extra fill calls disappear (-1.3 ms), partially offset
by worse atomic locality on the sparse master rows (SH VJP 5.4 -> 6.0 ms,
projection bwd 1.6 -> 2.0 ms). Net: native backward 28.9 -> 28.5 ms and total
46.1 -> 45.5 ms in profiling; the full benchmark shows backward 27.0 -> 26.6 ms
and total 45.0 -> 44.6 ms on bicycle (train 12.3 -> 11.9 ms), with peak VRAM
down 0.13 GB. Gradient cosine stays 1.000000 and all 99 tests pass.

### tanks_and_temples/train (N=1,026,508) --- 2026-08-01h (direct-master scatter)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 7.9 | 13.6 | 21.7 | 3.22 GB | 0.0% | 19.25 | 0.6764 | 0.2967 | 1,026,508 |
| higs_recompute | 8.2 | 24.6 | 32.9 | 4.00 GB | 15.1% | 19.25 | 0.6763 | 0.2967 | 1,026,508 |
| higs_native | 7.1 | 11.9 | 19.2 | 3.18 GB | 15.1% | 19.24 | 0.6762 | 0.2969 | 1,026,508 |
| higs_dynamic | 6.3 | 10.6 | 17.1 | 3.59 GB | 15.6% | 20.34 | 0.7063 | 0.2785 | 731,576 |

### mipnerf360/bicycle (N=6,131,954) --- 2026-08-01h

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 18.0 | 32.9 | 51.1 | 10.39 GB | 0.0% | 17.28 | 0.4479 | 0.4287 | 6,131,954 |
| higs_recompute | 19.4 | 60.0 | 79.5 | 12.69 GB | 62.9% | 17.28 | 0.4480 | 0.4286 | 6,131,954 |
| higs_native | 17.8 | 26.6 | 44.6 | 11.14 GB | 62.9% | 17.28 | 0.4480 | 0.4288 | 6,131,954 |
| higs_dynamic | 14.1 | 22.0 | 36.3 | 14.14 GB | 65.0% | 18.16 | 0.4736 | 0.3795 | 4,159,381 |

**Interpretation (2026-08-01h)**

- `higs_native` total iteration vs std: **-11.5% on train** (19.2 vs 21.7 ms)
  and **-12.7% on bicycle** (44.6 vs 51.1 ms); vs `gsplat_recompute` it is
  **-42% / -44%** (32.9 / 79.5 ms).
- Why a 2x backward speedup over `gsplat_recompute` does not become a 2x
  total speedup: the 2x only holds against the recompute fallback, which
  re-runs the whole rasterization pipeline inside backward (24.6 / 60.0 ms).
  Against the real baseline (std's own backward, 13.6 / 32.9 ms) the native
  backward is only -13% / -19%, and backward is ~60% of the iteration, so the
  ceiling on total speedup vs std is bounded by that ~5-7 ms saving (11-13%
  measured). The forward side is isect-bound (n_isects is identical to std's
  because culling only removes Gaussians that had no intersections), so it
  cannot improve via culling; the remaining total-time reduction must come
  from the backward and per-gaussian memory savings (packed buffers, fewer
  sorts, no index_add backward).
- `higs_dynamic` reaches 17.1 ms (train, -21%) / 36.3 ms (bicycle, -29%) with
  the densify/prune PSNR gains unchanged.

### Attempted (2026-08-01i, reverted): single-pass culling+projection merge

The forward ran two `fully_fused_projection` passes per step: one over ALL
Gaussians to derive the culling mask (union over cameras) and a second one on
the gathered visible subset inside `_native_forward_capture`. A single
all-N projection whose per-camera outputs are sliced to the visible subset was
implemented and benchmarked:

- kernel bundle: `projection_ewa_3dgs_fused_fwd_kernel` went from 2 launches
  (~1.75 ms aggregate) to 1 (~1.26 ms), but slicing the four per-camera
  outputs with PyTorch advanced indexing (`t[:, :, visible_ids]`) added a
  0.95 ms gather.
- Full benchmark (08-01i): `higs_native` forward regressed 17.8 -> 18.5 ms on
  bicycle (total 44.6 -> 45.5 ms); train was flat. The fused projection
  kernel is launch/occupancy-bound rather than work-bound, so the second pass
  was nearly free and the merge cannot win even with a native gather kernel
  (gather cost ~= saved projection). Reverted; 08-01j matches 08-01h
  (forward 18.0 / backward 26.8 / total 45.0 on bicycle, 99 tests pass).

### Seventh optimization (2026-08-01m): fused union-visibility mask kernel

A phase profile of the `higs_native` forward on bicycle (4 cams, 960x540,
N=6.13M, 2.27M visible) decomposed the ~18.0 ms forward into: rasterize 10.4,
isect 2.9, SH 1.9, batched FP32 culling projection 2.3, visible-subset gathers
0.5, second projection over the visible subset 0.55, `where` 0.09, and the
Python visibility mask `(r > 0).all(-1).any(0)` at **0.78 ms**. The mask was a
double reduction over ~24.5M bools (C x N): an and-reduce over the two radius
components (~0.28 ms) plus an or-reduce over cameras (~0.07 ms) plus the
temporary bool allocations - a hidden Python-side cost on the critical path.

Fix: `higs_union_visible_mask` (`GatherVisible.cu/.h`) - one thread per
Gaussian, loop over cameras with early exit,
`mask[n] = any_c((r[c,n,0] > 0) && (r[c,n,1] > 0))`, templated on int32/FP32
radii (the culling projection returns int32 radii; the kernel covers both),
emitting a single `[N]` bool tensor. `_cull_gaussians_batched` now calls it via
`_union_visible_mask_native()`, keeping the original PyTorch expression as the
automatic fallback when the extension is absent. Measured: 0.78 ms -> 0.136 ms
(~-0.65 ms), matching the full-benchmark forward drop of -0.74 ms.

The same round re-attempted the round-7 "single-pass projection" idea with a
native per-camera row gather (`higs_gather_camera_rows` slicing the all-N
projection outputs to the visible subset). The sliced projection outputs are
bit-identical to a re-projection, but the native gather still costs 0.93 ms vs
the ~0.55 ms the second projection pass saves - confirmed again as a regression
and fully reverted (the tree contains no `higs_gather_camera_rows`). Two
details matter for any future slicing attempt: `fully_fused_projection` radii
are int32 (not FP32) and depths are 3-D `[1, C, N]`.

### tanks_and_temples/train (N=1,026,508) --- 2026-08-01m (after fused mask)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 8.16 | 14.20 | 22.55 | 3.22 GB | 0.0% | 19.24 | 0.6762 | 0.2967 | 1,026,508 |
| higs_recompute | 8.01 | 24.52 | 32.71 | 4.00 GB | 15.1% | 19.25 | 0.6765 | 0.2966 | 1,026,508 |
| higs_native | 7.02 | 11.84 | 19.03 | 3.18 GB | 15.1% | 19.25 | 0.6763 | 0.2964 | 1,026,508 |
| higs_dynamic | 6.30 | 10.60 | 17.07 | 3.59 GB | 15.6% | 20.30 | 0.7060 | 0.2771 | 731,698 |

### mipnerf360/bicycle (N=6,131,954) --- 2026-08-01m (after fused mask)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 17.92 | 32.81 | 50.94 | 10.39 GB | 0.0% | 17.28 | 0.4480 | 0.4291 | 6,131,954 |
| higs_recompute | 18.83 | 59.87 | 78.89 | 12.69 GB | 62.9% | 17.28 | 0.4480 | 0.4287 | 6,131,954 |
| higs_native | 17.27 | 26.61 | 44.05 | 11.14 GB | 62.9% | 17.28 | 0.4479 | 0.4288 | 6,131,954 |
| higs_dynamic | 13.68 | 22.00 | 35.86 | 14.14 GB | 65.0% | 18.21 | 0.4738 | 0.3792 | 4,160,369 |

**Interpretation (after the fused mask)**

- vs the 08-01j best baseline, bicycle `higs_native` forward 18.01 -> 17.27
  (-0.74 ms) and total 45.04 -> 44.05 (-0.99 ms); train forward 7.12 -> 7.02.
  The forward is now clearly faster than std (17.27 vs 17.92, -3.6%) and the
  native backward is -18.9% vs std (26.61 vs 32.81) and 2.25x faster than the
  recompute fallback (59.87 ms).
- End-to-end vs std: `higs_native` -15.6% (train) / -13.5% (bicycle);
  `higs_dynamic` -24.3% / -29.6% with the densify/prune PSNR gains unchanged
  (train 20.30 / bicycle 18.21 vs std 19.24 / 17.28).
- 08-01l captured the intermediate state with the camera-row slice still in the
  tree (bicycle `higs_native` fwd 17.61 / total 44.45): a small net gain over
  08-01j but a regression vs the fused-mask-only state, so the slice was
  reverted and 08-01m is the final result. All 99 tests pass.

### Eighth optimization (2026-08-02): fixed-grid SH VJP (no compaction, no device->host sync)

A backward profiler on bicycle (4 cams, 960x540, N=6.13M, 2.27M visible
pairs) showed `higs_sh_vjp_kernel` at 5.97 ms vs std's
`spherical_harmonics_bwd_kernel` at 4.89 ms - the only HiGS backward kernel
slower than its std counterpart (std additionally pays ~9.8 ms of
`indexing_backward` that HiGS eliminates). The old SH VJP sized its grid from
a device-side visible-pair compaction
(`higs_vis_block_counts` / `higs_vis_total` / `higs_vis_pairs` +
`exclusive_sum` + `n_vis.to(cpu).item()`), forcing a device->host sync in
every backward.

Fix: replace the compaction pipeline with a fixed `I*N*D` grid and a per-pair
radii mask (std-style `masks` filtering), deleting the three compaction
kernels, the `HIGS_SH_VIS_BLOCK` macro and the CUB includes. The kernel body
is unchanged apart from the thread mapping. Measured:

- `higs_sh_vjp_grid_kernel` self time unchanged at 5.97 ms - the compaction
  and the host sync were not the kernel bottleneck. Full benchmark is neutral:
  bicycle `higs_native` 17.37/26.52/44.07 ms vs 08-01m 17.27/26.61/44.05
  (same run: std 22.06/35.82/58.10, recompute 18.82/60.30/79.32; std's
  absolute numbers vary run-to-run with neighbor-GPU load, the same-run
  relative ranking is unchanged).
- Structural win: no per-backward device->host sync, 3 fewer kernel launches,
  no CUB dependency, no per-step visible-pair buffer allocations. All 99 tests
  pass.

An isolation build that hardcoded `cam_pos = 0` (skipping the per-thread
`-R^T t` direction recompute) still measured 5.96 ms, so the 1.1 ms gap vs std
is not the direction math. It is attributed to the master-buffer `v_means`
atomics (up to 12-way contention for Gaussians visible in all 4 cameras vs
std's per-camera `v_dirs` 3-way) plus the `colors_eval` ReLU-mask load.
Closing it would need std-style per-camera `dirs` capture ([B, N, 3] = 294 MB
on bicycle) or a per-camera `v_dirs` intermediate plus a reduction pass - a
real memory/launch cost for ~1 ms on a 44 ms iteration, so it is left as a
documented trade-off.

Topology-rebuild cost (bicycle): `pack_gaussian_inference_scene` 3.2 ms,
renderer construction 0.35 ms; a `mark_dirty()`-forced rebuild step measures
56.3 ms fwd+bwd vs 47.3 ms without rebuild (+9.0 ms). In the 20-step benchmark
(densify every 5, 3 rebuilds) that is ~+1.4 ms/step amortized.

**Answer to "backward is 2x faster than recompute, why is the total not
faster?"** The observation was accurate for the early state (08-01b:
`higs_native` bicycle total 52.8 ms vs std 51.0 ms - the backward was already
faster than recompute, but the forward paid a full per-camera HiGS render just
to build the culling mask and cancelled the gain). The forward-side rounds
(batched culling projection, per-step pack skip, visible-subset gather, fused
union mask) plus the backward rounds brought the total to 44.05 ms
(-13.5% vs std, -44% vs recompute on bicycle). The absolute backward is still
dominated by `higs_blend_bwd_kernel` (19.1 ms), the same
pixel/isect-throughput-bound kernel std uses (19.1 ms); the native path wins by
not re-running the pipeline (recompute bwd = fwd + bwd = 60 ms) and by
eliminating std's `indexing_backward` + gather costs.

### tanks_and_temples/train (N=1,026,508) - 2026-08-02 (after fixed-grid SH VJP)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 7.59 | 13.17 | 20.94 | 3.22 GB | 0.0% | 19.25 | 0.6761 | 0.2966 | 1,026,508 |
| higs_recompute | 8.05 | 24.54 | 32.78 | 4.00 GB | 15.1% | 19.25 | 0.6764 | 0.2965 | 1,026,508 |
| higs_native | 7.12 | 11.82 | 19.11 | 3.15 GB | 15.1% | 19.35 | 0.6820 | 0.2931 | 1,026,508 |
| higs_dynamic | 6.66 | 11.14 | 18.11 | 3.57 GB | 15.6% | 20.28 | 0.7070 | 0.2755 | 730,112 |

### mipnerf360/bicycle (N=6,131,954) - 2026-08-02 (after fixed-grid SH VJP)

| backend | fwd ms | bwd ms | total ms | peak VRAM | culling | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|---|
| std | 22.06 | 35.82 | 58.10 | 15.58 GB | 0.0% | 17.28 | 0.4481 | 0.4287 | 6,131,954 |
| higs_recompute | 18.82 | 60.30 | 79.32 | 17.89 GB | 62.9% | 17.28 | 0.4481 | 0.4287 | 6,131,954 |
| higs_native | 17.37 | 26.52 | 44.07 | 16.26 GB | 62.9% | 17.28 | 0.4479 | 0.4294 | 6,131,954 |
| higs_dynamic | 13.74 | 21.83 | 35.75 | 19.33 GB | 65.0% | 18.20 | 0.4740 | 0.3798 | 4,159,001 |

**Interpretation (round 9)**

- The fixed-grid SH VJP is benchmark-neutral (bicycle total 44.07 vs 08-01m
  44.05, train 19.11 vs 19.03) while removing the per-backward device->host
  sync; train `higs_native` stays -8.7% vs std (19.11 vs 20.94) and bicycle
  -24.2% vs std in the same run (std's absolute numbers are inflated by
  neighbor-GPU load; the -13.5% figure from the 08-01m quiet-machine run is
  the more representative cross-run comparison).
- PSNR/SSIM/LPIPS parity with std holds (train 19.35 vs 19.25; bicycle 17.28);
  `higs_dynamic` keeps its densify/prune quality gain (train 20.28, bicycle
  18.20).

### Structural analysis (2026-08-02, round 10): blend bwd is irreducible; the add_/fill cluster is autograd semantics shared with std

Two profiling probes on bicycle (4 cams, 960x540) settle where the remaining
time is and why it cannot be reduced further:

1. **Intersection count is identical, so `higs_blend_bwd_kernel` is
   throughput-bound, not culling-bound.** A probe counted `n_isects =
   14,013,675` for both std and higs (2,274,065 visible Gaussians). Culled
   Gaussians produce zero intersections, so the visibility mask already
   removes exactly the isects std processes; `higs_blend_bwd_kernel`
   (19.1 ms) and std `rasterize_to_pixels_3dgs_bwd_kernel` (19.2 ms) do the
   same per-isect math. Further culling cannot shrink it - it is the same
   pixel/isect-throughput wall std hits.

2. **The 2.47 ms vectorized-add cluster is autograd leaf accumulation,
   shared with std - and mostly a profiling artifact.** A round-13 probe
   (2026-08-02) counted AccumulateGrad `add_` kernel executions in a
   benchmark-style step using `opt.zero_grad(set_to_none=True)`: the engine
   performs **direct assignment** when the leaf grad is None, so the
   1,940 us `vectorized_elementwise CUDAFunctor_add` attributed to
   `torch::autograd::AccumulateGrad` -> `aten::add_` (adding the returned
   master grad into the `colors` leaf `[6,131,954, 16, 3]`, ~1.2 GB) does
   NOT exist in real training - it only appears in profile scripts that
   leave stale grads behind, and the benchmark itself uses set_to_none.
   Real-training backward is therefore ~1.9 ms cheaper than the profiled
   figure for both std and higs. The 643 us FillFunctor is `aten::zeros_like`
   inside `_native_backward` pre-zeroing the master buffers the kernels
   scatter into (required dense-tensor semantics; std pays the same or more
   under load: std FillFunctor 9.66 ms + add 7.84 ms vs higs 4.9 ms +
   7.3 ms).

Fresh quiet-GPU benchmark (GPU1 idle, 2026-08-02): std 52.3 /
higs_recompute 79.3 / higs_native 44.0 / higs_dynamic 35.8 ms on bicycle.
The native backward is 2.27x faster than recompute (26.4 vs 59.9 ms) but the
total is only 1.80x (44.0 vs 79.3) because the forward (~17-19 ms, ~40% of
the total) is shared and unchanged; vs std the total is -15.9% and the
backward -21%.

**Conclusion:** the frozen native per-step path is at its structural floor on
bicycle. The top three costs (blend bwd ~19.5 ms / rasterize fwd ~8.7 ms /
SH VJP ~6 ms) are the same kernel math std executes, and the remaining
HiGS-specific items (batched culling projection ~2.3 ms, gather ~0.9 ms,
sort ~1.6 ms, amortized pack ~1.4 ms/step in dynamic mode) are already below
std's corresponding full-scene costs. No further culling or autograd-side
change can move the total more than ~1-2 ms without algorithmic work (e.g.
tile LOD / fewer isects) or benchmark-level stream overlap.
### Speed/quality knob: radius_clip curve (2026-08-02, benchmarked)

The only remaining speed lever on the exact path is the `radius_clip`
parameter (HiGS default 0.0 = max quality, applied consistently to culling
and render). Benchmarked end-to-end (20 training steps, train+eval at the
same clip, GPU1 quiet):

**bicycle (N=6.13M, detail-heavy)** - native / dynamic total, PSNR/SSIM/LPIPS:

| clip | std tot | native tot | dynamic tot | native PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| 0.0 | 52.3 | 44.5 | 35.8 | 17.27 | 0.4479 | 0.4288 |
| 3.0 | 32.7 | 32.8 (-26%) | 28.3 (-21%) | 17.46 | 0.4591 | 0.4188 |
| 5.0 | 24.5 | 25.6 (-42%) | 22.4 (-37%) | 17.18 | 0.4257 | 0.4834 |

**tanks_and_temples/train (N=1.03M, coarse)**:

| clip | std tot | native tot | dynamic tot | native PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| 0.0 | 20.94 | 19.11 | 18.11 | 19.35 | 0.6820 | 0.2931 |
| 3.0 | 17.6 | 15.6 (-18%) | 13.9 (-23%) | 19.44 | 0.6859 | 0.2873 |
| 5.0 | 13.7 | 12.7 (-33%) | 11.5 (-37%) | 19.59 | 0.6872 | 0.2898 |

Findings:

- At the 20-step horizon (train and eval rendered at the same clip), clip=3.0
  is a near-free speedup on both scenes: -18% to -26% total with equal-or-
  better PSNR/SSIM/LPIPS (removing tiny noisy splats helps the short-horizon
  renders). clip=5.0 stays quality-neutral on the coarse train scene but
  degrades on detail-heavy bicycle (SSIM 0.4257 / LPIPS 0.4834 vs clip 0).
- The knob is a renderer-level parameter: it speeds up std and HiGS alike.
  HiGS's culling advantage shrinks as the clip tightens (at clip 5.0 native
  25.6 ms is marginally slower than std 24.5 ms because the union culling
  pass over all N stays fixed while the shared isect-bound kernels shrink);
  at clip 0.0 HiGS keeps its full -15.9% vs std.
- Caveat: 20-step metrics are short-horizon; a fully converged model at
  clip>=3 caps fine-detail content that small splats would otherwise provide,
  so the long-run ceiling on detail-heavy scenes is still expected to sit
  below clip 0. The knob is a legitimate product-level quality/speed
  operating point, now exposed in the benchmark as `--radius-clip`.

### Round 13 (2026-08-02): SH VJP atomic-contention probe refuted; AccumulateGrad add_ is a profile artifact

Two probes on bicycle (4 cams, 960x540, N=6.13M, GPU1 idle) closed the last
two open cost-model questions:

1. **SH VJP master `v_means` atomics are NOT the ~1.1 ms gap vs std.** An
   A/B rebuild of `higs_sh_vjp_grid_kernel` replaced the three master
   `atomicAdd(v_means + m_id*3 + {0,1,2}, v_dir)` (12-way cross-camera
   contention) with per-camera flat `atomicAdd(v_means_flat + idx*3 +
   {0,1,2})` (3-way within-warp contention, std-style). Same-session
   measurements: master 5972.8 us vs flat 5672.3 us (-0.30 ms, -5%);
   total backward 25.55 -> 25.38 ms and total iteration 41.92 -> 41.99 ms
   (flat, within noise; the flat variant also needs a 294 MB zeroed buffer).
   Conclusion: the gap vs std's 4.89 ms `spherical_harmonics_bwd_kernel` is
   the coefficient-atomic work itself (~109M atomics) plus the ReLU-mask
   load, not the means scatter. The std-style per-camera `v_dirs` + reduce
   redesign is not worth its memory cost; kept as documented trade-off.
2. **AccumulateGrad `add_` (1,940 us) does not run in real training.** The
   benchmark's `opt.zero_grad(set_to_none=True)` makes leaf grads None, so
   the autograd engine assigns the first backward result directly - the
   probe counted 0 `CUDAFunctor_add` kernels under AccumulateGrad. The
   profiled 26.6 ms backward includes ~1.9 ms that real training never pays
   (for std and higs alike); the cuda-event backward here is 25.4 ms.


### Round 14 (2026-08-02): fused Adam + honest train-time metric

The benchmark's `total_ms` always excluded `opt.step()`, so the "total
training time" was never measured. A fresh phase profile on bicycle (GPU1
idle) showed why that matters: the 5-group Adam step over the 6.13M-Gaussian
masters (means/quats/scales/opacities/SH, ~362M floats) costs **14.9 ms** with
the default foreach path - the same order as the entire HiGS forward (17.3 ms)
and half of the native backward (26.5 ms). It dilutes the backward speedup in
real training and is identical for every backend.

Fix in `benchmark/run_higs_train_benchmark.py`:

- `make_optimizer` now defaults to `fused=True`: torch's single-kernel fused
  Adam runs each param group's whole update in one pass. Standalone A/B on the
  bicycle shapes: **14.88 -> 6.78 ms/step (2.2x)**. Inside the full loop the
  optimizer part drops from ~18.0 to ~7.4 ms/step (CUDA-event wall time), a
  ~10-12 ms/step saving on bicycle and ~1.9 ms/step on the 1.03M train scene.
  A `--no-fused-adam` flag plus an automatic non-fused fallback (CPU /
  unsupported platforms) preserves the old behavior; fused Adam state
  (`exp_avg` / `exp_avg_sq`) is shape-identical to foreach state, so the
  dynamic-mode densify/prune `sync_optimizer_state_for_topology_change` works
  unchanged.
- The benchmark now also reports `train_ms`: the CUDA-event wall time of the
  full training step **including** `opt.step()` (and densify/prune for
  `higs_dynamic`), recorded after `zero_grad`. This is the honest per-step
  training time.

Round-14 benchmark (EPIC-05 A100, GPU1 idle, fused Adam, 960x540, 4 train +
3 eval cams, 20 steps, densify every 5):

### tanks_and_temples/train (N=1,026,508) - 2026-08-02 round-14

| backend | fwd ms | bwd ms | total ms | train ms | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|
| std | 8.1 | 14.0 | 22.3 | 24.2 | 19.25 | 0.6762 | 0.2967 | 1,026,508 |
| higs_recompute | 8.1 | 24.7 | 33.0 | 34.7 | 19.24 | 0.6763 | 0.2968 | 1,026,508 |
| higs_native | 7.1 | 11.8 | 19.1 | 20.8 | 19.25 | 0.6763 | 0.2967 | 1,026,508 |
| higs_dynamic | 6.2 | 10.6 | 17.1 | 20.1 | 20.36 | 0.7057 | 0.2764 | 731,688 |

### mipnerf360/bicycle (N=6,131,954) - 2026-08-02 round-14

| backend | fwd ms | bwd ms | total ms | train ms | PSNR | SSIM | LPIPS | final N |
|---|---|---|---|---|---|---|---|---|
| std | 17.8 | 32.9 | 50.9 | 58.3 | 17.29 | 0.4483 | 0.4286 | 6,131,954 |
| higs_recompute | 18.8 | 60.0 | 79.0 | 86.5 | 17.28 | 0.4480 | 0.4283 | 6,131,954 |
| higs_native | 17.3 | 26.5 | 44.0 | 51.4 | 17.28 | 0.4479 | 0.4286 | 6,131,954 |
| higs_dynamic | 13.7 | 21.9 | 35.8 | 48.0 | 18.20 | 0.4736 | 0.3796 | 4,159,597 |

With the optimizer included, `higs_native` is -13.9% / -11.8% and
`higs_dynamic` -16.9% / -17.7% vs std on train/bicycle. The shared optimizer
cost shrinks the gap vs the optimizer-excluded `total_ms` figures, but the
absolute per-step training time is ~10-12 ms faster than before for every
backend on bicycle.

Notes:

- With the old foreach optimizer the same bicycle loop measured std `train_ms`
  70.0 / native 61.5 ms (round-14-nofused JSON) vs 58.3 / 51.4 ms fused:
  fused Adam is worth ~10-12 ms/step of real training time on the large scene
  for all backends.
- Quality is unchanged within run noise (PSNR/SSIM/LPIPS match the
  foreach-Adam runs to ~0.01 dB / 0.0005). The dynamic-path densify/pack cost
  is now visible in `train_ms` (bicycle +12.2 ms/step vs the +7.4 ms shared
  optimizer on the frozen paths).
- One transient anomaly: a `higs_recompute` train-scene bwd of 41.5 ms in the
  first round-14 pass (vs the stable 24.5-24.7 ms) disappeared on re-run
  (24.7 ms); not reproducible, attributed to allocator/neighbor-GPU state.


### Round 15 (2026-08-02): fast row-gather for densify/prune + Adam-state sync

The dynamic-path densify/prune and Adam-state sync used PyTorch's boolean-mask
row gather and masked assignment on the FP32 masters / state tensors. On row
widths divisible by four floats (`colors [N,16,3]`, `quats [N,4]`) those
dispatch to the same pathologically slow vectorized path that motivated the
round-3 `higs_gather_visible` kernel. Micro-benchmark at 4.8M rows / 20%
duplication: the five-tensor state-sync pattern (`rows[valid] =
value[idx[valid]]`) cost **15.4 ms vs 5.6 ms** with `higs_gather_visible` +
`index_copy_` (bit-identical), and the five `t[mask]` gathers cost 2.24 ms vs
0.57 ms with the kernel.

Fix (all in `gaussian_inference.py` + the experimental extension):

- New single-tensor binding `higs_gather_rows(src, row_ids)` in
  `GatherVisible.{cu,h}` / `ext.cpp`, reusing the element-wise compact-copy
  kernel for arbitrary trailing shapes.
- `_densify_gaussians` / `_prune_gaussians`: the masked row gathers go through
  the new `_gather_rows_fast` helper (kernel when available, `t[ids]`
  fallback otherwise).
- `sync_optimizer_state_for_topology_change`: the per-state-tensor update
  becomes `rows.index_copy_(0, row_idx, _gather_rows_fast(value, v_idx))`
  instead of the boolean-mask gather+scatter (identical semantics for the
  duplicate-free row index map).

Measured on the dynamic path (bicycle, GPU1 idle, random-reference probe with
heavy duplication): densify-event tails **31-73 ms -> 12-18 ms**. Full
benchmark (GT references, 20 steps, densify every 5), round-15:
`higs_dynamic` `train_ms` **48.0 -> 46.6 ms** (bicycle) and **20.1 -> 19.2 ms**
(train scene); frozen paths and quality unchanged (bicycle native train 51.46
vs 51.45 ms, PSNR within 0.01 dB); all 99 tests pass. In heavy-duplication
training regimes the per-event saving is much larger (the slow gather scales
with the duplicated-row count).

### Round 16 (2026-08-02): 1080p benchmark - resolution-independent culling widens the HiGS lead

Re-ran the round-15 configuration at 1920x1080 (4 train + 3 eval cameras, 20
Adam steps, fused Adam, densify every 5, GPU1 idle) to verify the speedup at
higher resolution. The culling decision is resolution-independent (one
O(N x C) visibility mask; the 15.1% / 62.9% culling ratios are identical to
960x540), while the per-pixel blend cost grows 4x - so the relative and the
absolute per-step HiGS advantage should both grow.

Full JSON: `results/higs-train-benchmark-2026-08-02-round15-1080p.json`.

#### tanks_and_temples/train (N=1,026,508) - 1080p

| backend | fwd ms | bwd ms | total ms | train ms | peak VRAM | culling | PSNR | final N |
|---|---|---|---|---|---|---|---|---|
| std | 11.7 | 24.7 | 36.9 | 38.8 | 3.52 GB | 0.0% | 19.02 | 1,026,508 |
| higs_recompute | 11.6 | 38.5 | 50.6 | 52.2 | 4.54 GB | 15.1% | 19.01 | 1,026,508 |
| higs_native | 11.7 | 20.9 | 33.1 | 34.7 | 3.57 GB | 15.1% | 19.02 | 1,026,508 |
| higs_dynamic | 9.7 | 19.1 | 29.3 | 31.5 | 3.99 GB | 15.5% | 20.05 | 741,584 |

#### mipnerf360/bicycle (N=6,131,954) - 1080p

| backend | fwd ms | bwd ms | total ms | train ms | peak VRAM | culling | PSNR | final N |
|---|---|---|---|---|---|---|---|---|
| std | 27.9 | 51.6 | 79.9 | 87.4 | 10.89 GB | 0.0% | 16.76 | 6,131,954 |
| higs_recompute | 28.7 | 88.5 | 117.7 | 125.1 | 13.66 GB | 62.9% | 16.76 | 6,131,954 |
| higs_native | 26.3 | 42.6 | 69.4 | 76.8 | 11.67 GB | 62.9% | 16.75 | 6,131,954 |
| higs_dynamic | 21.2 | 37.8 | 59.4 | 68.0 | 14.92 GB | 62.9% | 17.61 | 4,477,802 |

**Interpretation**

- Native-vs-recompute gradient cosine 0.999995 (train) / 1.000000 (bicycle),
  quality within noise of the round-15 probe.
- The native backward is 1.84x (train) / 2.08x (bicycle) faster than
  `gsplat_recompute` at 1080p (20.9 vs 38.5 ms; 42.6 vs 88.5 ms): the
  recompute fallback re-runs the full rasterization pipeline per backward and
  scales with resolution, while the native VJP only touches the captured
  visible-subset state.
- Speedup vs std holds and grows with resolution (train_ms): `higs_native`
  -10.6% (train) / -12.1% (bicycle); `higs_dynamic` **-18.9% / -22.2%**
  (vs -19.0% / -20.1% at 960x540). On bicycle the absolute saving is now
  **-19.4 ms per training step** (68.0 vs 87.4 ms; -11.7 ms at 960x540).
- Why the advantage grows: culling is resolution-independent, so the
  N-proportional work saved by skipping 62.9% of the 6.13M Gaussians is
  constant in ratio but its absolute value scales with the
  resolution-proportional blend/isect work it avoids.
- Quality conclusions unchanged: frozen paths within 0.01 dB of std
  (16.75 vs 16.76); `higs_dynamic` still ahead on held-out PSNR (20.05 vs
  19.02 train; 17.61 vs 16.76 bicycle).
- VRAM: `higs_native` stays near std (+0.05 GB train / +0.8 GB bicycle);
  `higs_dynamic` peaks at 14.92 GB on bicycle (+4.0 GB vs std) - densify
  temporaries plus resolution-scaling visibility lists; well within the
  80 GB A100, but the dynamic-path overhead does grow with resolution.

### Round 17 (2026-08-02): fused zero-neg row gather for the Adam-state sync

The dynamic-path `sync_optimizer_state_for_topology_change` built each new
Adam state tensor with `torch.zeros_like(new_t)` (allocation + memset), a
`higs_gather_rows` copy of the valid rows, and an `index_copy_` scatter - up
to ten state tensors per densify event (5 params x exp_avg / exp_avg_sq).

Fix: `higs_gather_rows` gained a `zero_on_neg` mode - row ids of -1
(brand-new Gaussians) write zeros inside the same gather kernel, so the
output is allocated uninitialized and every row is written by a single
launch. The new helper `_gather_rows_new_zeros` routes the sync through it
(zeros + indexed-copy fallback without CUDA).

Micro-benchmark (4.8M rows, 25% duplication + 5% brand-new, 5 state tensors,
GPU1 idle): **8.36 ms -> 2.97 ms per event** (-5.4 ms, 2.8x), bit-identical.

Full benchmark (960x540, 20 steps, densify every 5, GPU1 idle; round-17 vs
round-15 with the same config):

| backend | train total/train ms | bicycle total/train ms |
|---|---|---|
| std | 21.5 / 23.4 (was 21.8 / 23.7) | 50.8 / 58.3 (was 50.9 / 58.3) |
| higs_dynamic | 16.9 / 18.8 (was 16.9 / 19.2) | 35.7 / 43.1 (was 35.7 / 46.6) |

`higs_dynamic` `train_ms` improves from -19.0% to **-19.7%** (train) and from
-20.1% to **-26.1%** (bicycle) vs std gsplat. Frozen paths are unchanged
(bicycle native train 51.4 vs 51.5 ms, PSNR within 0.01 dB) and dynamic
held-out quality is unchanged (PSNR 18.19 vs 18.20). All 99 tests pass.
Dynamic-bicycle peak VRAM did not increase (13.42 vs 14.15 GB - the fused
path allocates one output instead of zeros + gather temp, though allocator
run-to-run variance plays a role).

Full JSON: `results/higs-train-benchmark-2026-08-02-round17.json`.

### Round 18 (2026-08-02): defer the packed-scene rebuild on training topology changes

The dynamic training forward derives its visibility mask from a batched FP32
projection (`_cull_gaussians_batched`) and both backends consume the FP32
captured tensors - the handle's packed FP16 scene is never touched by
training. But a real topology change (densify/prune -> `mark_dirty()` / an N
change) still triggered `pack_gaussian_inference_scene` (FP32 -> FP16) plus
renderer construction on every densify step (~3.25 ms for 6.13M Gaussians,
measured directly).

Fix: `_refresh_higs_renderer_scene` on the lightweight (training) path now
defers the pack for real topology changes too - only the version bookkeeping
advances (`version += 1`, `n_gaussians` updated, `topology_rebuilt = True`)
and the new `packed_stale` flag is set. `_cull_gaussians_higs` (the
non-training culling API) treats `packed_stale` as a rebuild trigger, so the
packed scene is re-created on demand the first time it is actually needed. The
explicit flag is required because densify/prune creates brand-new tensors
whose `_version` can collide with the captured ones, so `params_changed`
alone cannot detect the stale scene.

Benchmark (960x540, 20 steps, densify every 5, GPU1 idle; round-18 vs
round-17, same config):

| backend | bicycle train ms | vs std |
|---|---|---|
| std | 58.2 (was 58.3) | - |
| higs_dynamic | 42.4 (was 43.1) | **-27.2%** (was -26.1%) |

The train scene shows no measurable change (its pack is only ~0.5 ms; run
noise dominates there). Frozen paths unchanged (native bicycle 51.5 ms),
quality within noise (dynamic PSNR 18.16 vs 18.19), `topology_rebuilt_frac`
unchanged (0.75), peak VRAM unchanged. All 100 tests pass; one new test
verifies that a training topology change defers the pack while the
non-training culling API still re-packs on demand.

Full JSON: `results/higs-train-benchmark-2026-08-02-round18.json`.

### Round 19 (2026-08-02): final 1080p numbers + tile-LOD feasibility measurement

Re-ran the 1920x1080 benchmark (4 train + 3 eval cameras, 20 Adam steps,
densify every 5, GPU1 idle) with all accumulated optimizations (rounds
15-18), giving the authoritative end-state numbers:

| backend (bicycle) | fwd ms | bwd ms | total ms | train ms | peak VRAM | PSNR | final N |
|---|---|---|---|---|---|---|---|
| std | 27.7 | 51.6 | 79.7 | 87.2 | 10.89 GB | 16.75 | 6,131,954 |
| higs_native | 26.4 | 42.6 | 69.4 | 76.9 | 11.67 GB | 16.75 | 6,131,954 |
| higs_dynamic | 20.8 | 37.7 | 59.0 | 66.2 | 14.67 GB | 17.60 | 4,477,957 |

| backend (train) | fwd ms | bwd ms | total ms | train ms | peak VRAM | PSNR | final N |
|---|---|---|---|---|---|---|---|
| std | 11.8 | 24.6 | 37.1 | 39.0 | 3.52 GB | 19.02 | 1,026,508 |
| higs_native | 11.7 | 20.9 | 33.1 | 34.8 | 3.57 GB | 19.02 | 1,026,508 |
| higs_dynamic | 9.7 | 19.1 | 29.2 | 31.2 | 4.02 GB | 20.12 | 741,775 |

vs std `train_ms`: `higs_native` -10.7% (train) / -11.8% (bicycle);
`higs_dynamic` **-19.9% / -24.0%** (round-16 was -18.9% / -22.2%, so rounds
17+18 added ~1.8 ms/step on bicycle). Frozen paths within 0.01 dB of std,
dynamic held-out PSNR still ahead.

Full JSON: `results/higs-train-benchmark-2026-08-02-round19-1080p.json`.

**Tile-LOD feasibility (the last remaining lever, now measured).** On bicycle,
4 train cameras, 1920x1080, rendering just the culled-visible subset (2.27M of
6.13M Gaussians, 37.1%):

- standard gsplat forward on the subset: **28.99 ms** (essentially the same as
  rendering the full 6.13M scene - the 62.9% culled Gaussians generate ~zero
  intersections because they are off-screen or sub-pixel, so culling saves
  only the per-Gaussian projection overhead, not the isect-bound blend work);
- HiGS renderer (macro-tile intersect) on the same subset: **8.61 ms**;
- potential forward saving: **20.4 ms/step (3.37x)**.

This confirms tile LOD / HiGS-native rendering is the one remaining
algorithmic lever: it reduces the intersection work itself rather than the
constant per-Gaussian overhead. Realizing it means making the differentiable
forward render the visible subset with the HiGS renderer and having the
native backward consume HiGS's captured state (or emitting the standard-layout
capture tensors from the HiGS pipeline) - a substantial forward+backward
rework, not a micro-optimization.

### Round 20 (2026-08-02): honest low-overhead std baseline (std_ll) + tile-LOD economics corrected

**Fairness fix.** The benchmark's `std` backend calls the high-level
`gsplat.rendering.rasterization()` wrapper, which carries Python/alloc
overhead on top of the same CUDA kernels. Added a `std_ll` backend that runs
the raw kernels the HiGS capture path uses (identical pipeline, no culling) as
the apples-to-apples baseline. In the warmed training loop at 1920x1080 x 4
cameras the wrapper overhead is ~1.3 ms fwd + ~4.2 ms bwd on bicycle (a cold
standalone probe showed ~9 ms, i.e. most of the round-19 "3.37x" probe gap was
wrapper overhead that the training loop does not actually pay).

Re-ran the 1080p benchmark (4 train + 3 eval cameras, 20 Adam steps, densify
every 5, GPU1 idle) with backends `[std, std_ll, higs_native, higs_dynamic]`:

| backend (bicycle) | fwd ms | bwd ms | total ms | train ms | peak VRAM | PSNR |
|---|---|---|---|---|---|---|
| std | 28.1 | 52.1 | 80.7 | 88.3 | 16.03 GB | 16.75 |
| std_ll | 26.8 | 47.9 | 75.2 | 82.7 | 15.88 GB | 16.75 |
| higs_native | 26.9 | 42.7 | 70.0 | 77.5 | 17.06 GB | 16.76 |
| higs_dynamic | 21.2 | 37.8 | 59.4 | 67.1 | 19.81 GB | 17.61 |

| backend (train) | fwd ms | bwd ms | total ms | train ms | peak VRAM | PSNR |
|---|---|---|---|---|---|---|
| std | 11.0 | 23.5 | 34.9 | 36.5 | 10.50 GB | 19.02 |
| std_ll | 11.0 | 21.7 | 33.2 | 34.8 | 10.20 GB | 19.02 |
| higs_native | 11.8 | 20.9 | 33.1 | 34.8 | 10.55 GB | 19.02 |
| higs_dynamic | 9.7 | 19.1 | 29.3 | 31.2 | 10.99 GB | 20.06 |

Honest margins (train_ms): vs the low-overhead `std_ll` baseline,
`higs_native` is -6.3% (bicycle) / ~0% (train) and `higs_dynamic` is
**-18.9% / -14.5%**; vs the `std` wrapper the margins stay -12.2%/-4.9% and
-24.0%/-14.6%. The native forward is **not** faster than the raw std forward
(culling+gather overhead offsets the subset-render savings; the 62.9% culled
Gaussians generate no isects anyway) - the win is the native backward (bwd
42.7 vs 47.9 on bicycle) and the dynamic path's densify/prune efficiency.

**Tile-LOD economics corrected.** The round-19 probe's 28.99 ms std number
used the high-level `rasterization()` wrapper on the visible subset; the
training loop's capture path is ~17.7 ms of kernels (rasterize 11.3, isect
4.1, SH 1.9, proj 0.55). A realistic tile-LOD forward = pack visible subset
(raw C++ pack 1.23 ms for 2.27M; the 7.8 ms `GaussianInferenceScene.
from_gaussian_tensors` path is avoidable by constructing the scene from the
packed tensors directly) + HiGS renderer with reused output buffers (9.5 ms,
4 cams 1080p) + emitting std-format captures for the native backward (isect
sort + alpha/last_ids + FP32 conversions, ~4-5 ms) ~= 15-17 ms vs the current
20.9 ms forward - a forward-only saving of roughly 4 ms/step, not 20.4. The
dominant training cost is the backward (37.7 ms dynamic / 42.7 native); a
HiGS-format backward (blend VJP over the macro-tile isect structure) remains
the last big lever.

**Rejected experiment.** Reusing the culling projection's rows for the capture
(4 advanced-index gathers) cost 1.05 ms vs 0.55 ms for re-projecting the
2.27M subset - a net regression, reverted (bit-exact either way; tests still
100 passed).

Full JSON: `results/higs-train-benchmark-2026-08-02-round20-1080p.json`.


### Round 21-22 (2026-08-02): backward cost decomposition + PX pixels-per-thread blend VJP

**Why the "2x backward" does not move the total.** CPU+CUDA profiling of one
dynamic step (bicycle 1080p, 4 cams) splits the 37.81 ms backward into
`higs_blend_bwd_kernel` 29.76 ms + `higs_sh_vjp_grid_kernel` 5.96 ms +
`higs_projection_bwd_kernel` 2.04 ms. Culling cannot reduce the blend VJP: the
62.9% culled Gaussians generate zero isects, so the per-isect blend work is
fixed. Round 20's honest `std_ll` baseline (bwd 47.9 vs native 42.7 ms on
bicycle) shows the real native-backward edge is ~5 ms; the standalone "2x"
microbenchmark only compared against the recompute fallback, which pays a
whole extra re-rasterization. The dominant backward cost is the blend VJP
itself.

**Compile-time probe (GPU1 idle).** Building the blend kernel with the warp
reductions + atomic scatters compiled out (`HIGS_BWD_PROBE_NO_ATOMICS`) drops
it 29.76 -> 3.64 ms; removing the per-isect VJP math leaves ~15.3 ms. The
naive reading is "~10.9 ms of reduction/atomic work", but the honest PX A/B
below shows the real lever is much smaller (probe codegen effects), so that
estimate was optimistic.

**PX (pixels-per-thread) blend VJP.** Added a templated
`higs_blend_bwd_px_kernel<CDIM, PX>` where each thread owns PX pixels (rows
`ty + q * (16/PX)` of the 16x16 tile), so the per-isect warp reductions and
atomic scatters scale as 1/PX while the per-pixel math is unchanged. Runtime
selection via `HIGS_PX_RUNTIME` (0/1/2/4; default 2); PX=1 is bit-equivalent
to the original kernel.

**Bug found & fixed during the PX work.** The first shared-accumulator PX=2
implementation produced wrong gradients (max diff ~3e-3 on means/scales for
four near-tile-boundary Gaussians) while PX=1 was bit-exact. Root cause:
`rasterize_to_pixels_3dgs_blend_bwd` ASSIGNS its `v_rgb_local` /
`v_conic_local` / `v_xy_local` / `v_opacity_local` outputs (correct for one
call per (thread, isect)), so calling it once per q-pixel made the later q
overwrite the earlier q's contribution. Verified with a diagnostic "split"
kernel (per-q warp reductions, matches the original to 1e-10) vs the shared
version (1e-3). Fixed by accumulating per-q scratch locals into the shared
per-isect totals before the single warp reduction. PX=1/2/4 now all match the
original kernel to ~1e-10; 100 tests pass with PX=2 as the default.

**Measured.** Paired profiler (bicycle 1080p, GPU1): blend bwd 30.5 -> 29.0 ms
(PX1 -> PX2), full step 122.6 -> 119.6 ms; PX=4 is slower than PX=2
(29.8 / 121.2, register pressure). Round-22 benchmark (same machine,
backends `[std_ll, higs_native, higs_dynamic]`): bicycle native bwd
42.7 -> 40.0 and dynamic bwd 37.8 -> 35.5 ms; train scene native 20.9 -> 19.1
and dynamic 19.1 -> 17.0 ms. Cross-run variance is +-15%, so the paired
profiler numbers are the reliable A/B: the honest gain is ~1.5 ms on the
blend kernel and ~3 ms/step end-to-end. The dominant blend cost is the
per-isect VJP math; the last big lever remains a HiGS-format backward
consuming the macro-tile structure (30M entries vs 330M isects).


### Round 23 (2026-08-02): "why the 2x backward does not move the total" answered with the full backend matrix + CatArrayBatchedCopy red herring cleared

**Direct answer.** On the current code (PX=2 default) the total iteration **is**
speeded up. A full 4-backend run (backends `[std_ll, higs_recompute,
higs_native, higs_dynamic]`, bicycle 1080p x 4 cams, 20 Adam steps, GPU0):

| backend (bicycle) | fwd ms | bwd ms | total ms | train ms | vs recompute | vs std_ll |
|---|---|---|---|---|---|---|
| std_ll | 27.0 | 48.4 | 75.8 | 83.4 | — | — |
| higs_recompute | 28.8 | 88.7 | 118.0 | 125.5 | — | +50% |
| higs_native | 26.4 | 40.0 | 66.9 | 74.4 | **-43%** | **-12%** |
| higs_dynamic | 20.9 | 35.5 | 56.9 | 64.2 | **-52%** | **-25%** |

| backend (train, 1.03M) | fwd ms | bwd ms | total ms | train ms | vs recompute | vs std_ll |
|---|---|---|---|---|---|---|
| std_ll | 11.0 | 21.8 | 33.3 | 34.9 | — | — |
| higs_recompute | 11.6 | 38.6 | 50.6 | 52.3 | — | +50% |
| higs_native | 11.7 | 19.1 | 31.2 | 32.8 | **-38%** | **-6%** |
| higs_dynamic | 9.6 | 17.0 | 27.1 | 29.0 | **-46%** | **-17%** |

The "2x backward" only materialises against `higs_recompute`, whose backward
re-runs the entire forward (rasterization under autograd) inside
`loss.backward()`: 88.7 ms on bicycle vs 40.0 ms native (2.2x) — but that
comparison is not a backward-vs-backward one. Against the honest low-overhead
`std_ll` baseline the native backward edge is 40.0 vs 48.4 ms (-17%) and the
forward is *not* faster (26.4 vs 27.0 ms: batched FP32 culling over all N x C
+ visible-subset gather offset the subset-render savings; the 62.9% culled
Gaussians generate ~zero isects anyway). The pre-PX=2 code (round-22 px1
control: native 26.9/56.1/83.4/91.1) is where "no total speedup" was true:
native total 83.4 ≈ std_ll 82.5 ms. Round 22's PX=2 blend VJP (bwd
56.1 -> 40.0) is exactly what moved the total into speedup territory.

**1-camera check** (bicycle 1080p, `--n-train 1`, in case the question was
measured on a single-view probe): std_ll 7.7/13.4/21.3/29.7; recompute
8.3/29.4/37.9/45.4; native 7.5/11.0/18.6/26.1 — native is -13% vs std_ll and
-51% vs recompute on total, so the speedup holds at 1 camera too.

**CatArrayBatchedCopy red herring cleared.** The nsys fwd profile's
`at::native::CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>, uint32_t, 2,
128, 1, 16>` row showed avg 14.5 ms x 3 launches, which looked like a hidden
per-step forward cost. Per-launch inspection shows the stats were skewed by a
**single 43.56 ms launch** (grid 3456x45, block 128) during scene loading
(cat of the 6.13M-Gaussian tensor group); the two steady-state launches in
the 3-step forward loop are **4.5 us each**. There is no 14.5 ms/step hidden
cat in the forward, and no `torch.cat`/`aten::cat` in the per-step capture
path (torch profiler sees zero cat events; render_mode RGB skips the
colors+depths cat). Nothing to optimize there.

**Where the total actually goes** (bicycle 1080p x 4 cams, native 66.9 ms
total): forward 26.4 ms (culling projection over all N x C ~2.3 ms, capture
rasterize ~11.3 ms + isect ~4.1 ms + SH ~1.9 ms + subset projection ~0.6 ms,
rest launch/alloc overhead), backward 40.0 ms (blend VJP ~29.8 ms + SH VJP
~6.0 ms + projection VJP ~2.0 ms). The blend VJP's per-isect math on ~330M
isects remains the largest single item; the last big lever is still a
HiGS-format backward consuming the macro-tile structure (30M entries vs 330M
isects), not the forward cat path.

Full JSONs: `results/higs-train-benchmark-2026-08-02-round23-1080p.json`,
`results/higs-train-benchmark-2026-08-02-round23-1cam-1080p.json`
(regeneratable; not tracked).

### Round 24 (2026-08-02): blend-VJP ellipse AABB prefilter measured and reverted (interleaved A/B negative)

**Hypothesis.** The blend VJP's inner q-loop calls `eval_gaussian_weight`
(an `__expf` plus alpha-threshold compare) for every (isect, pixel) pair,
and most pairs are far outside the Gaussian's support. A per-isect ellipse
AABB (`|dx| <= ex && |dy| <= ey`, with `ex/ey` derived from the conic and
`opac`/`ALPHA_THRESHOLD`) is a strict superset of the valid set: for
`opac <= 1` the `alpha >= ALPHA_THRESHOLD` region lies inside the
`sigma <= ln(opac / ALPHA_THRESHOLD)` ellipse, and degenerate conics
(`det <= 0`) disable the filter, so skipping `__expf` for out-of-AABB
pixels is bit-exact.

**Implementation.** Added `ex_batch`/`ey_batch` shared-memory arrays
(+2 floats x block size), computed them during the batch load
(`k = 2*ln(opac/ALPHA_THRESHOLD)`, `inv_det = 1/det`,
`ex = sqrt(k*conic.z*inv_det)`, `ey = sqrt(k*conic.x*inv_det)`), and gated
the q-loop with `inside_ellipse = ex<=0 || (|dx|<=ex && |dy|<=ey)`.

**Interleaved A/B** (bicycle 1080p x 4 cams, GPU0, alternating
OLD/NEW/OLD/NEW builds, 2 runs each; kernel column is torch-profiler
self-CUDA-time, WALL columns are CUDA-event timings):

| variant | blend bwd (self) | WALL fwd | WALL bwd | WALL total |
|---|---|---|---|---|
| OLD (PX=2 baseline) | 29.12 / 29.19 / 29.32 / 29.14 ms | 20.4 ms | 41.2 ms | 61.8 ms |
| NEW (AABB prefilter) | 31.42 / 31.30 / 31.29 / 31.33 ms | 20.8 ms | 43.8 ms | 64.6 ms |

**Conclusion: reverted.** The AABB prefilter is a stable ~+2.1 ms (+7%) on
the blend kernel and ~+2.8 ms on the full step across all four interleaved
pairs. The added shared memory, the per-isect `logf/sqrtf/div` in the batch
load and the extra register/branch pressure cost more than the skipped
`__expf` (a cheap, fully parallel instruction); the kernel is bound by
memory/launch-bounds and the per-isect VJP math, not by the exponential.
The source was restored to the round-22 PX=2 baseline: 100 tests pass and
the blend kernel is back to 29.1 ms.

**Next levers (unchanged).** The last big item is a HiGS-format backward
consuming the macro-tile structure (30M entries vs 330M isects). Cheap
kernel-level knobs (`__launch_bounds__`, loop-unroll hints, shared-memory
layout) were already tuned in rounds 21-22 and show no remaining headroom.

### Round 25 (2026-08-02): SH VJP kernel - precomputed camera positions + shuffle-reduced means atomics (-0.6 ms/step)

**Target.** `higs_sh_vjp_grid_kernel` (5.96 ms self) runs the coefficient VJP
for every (camera, visible gaussian, channel) triple. Two structural costs
vs the std `sh_backward` kernel stood out: (1) every thread re-derived the
camera world position `cam_pos = -R^t t` (a mat3 transpose + mat-vec per
thread, 27.2M times), and (2) the D=3 channel lanes of one
(camera, gaussian) each issued their own `atomicAdd` to the same
`v_means` entry - three serialized same-address atomics per output
coordinate (std instead writes a per-camera `v_dirs` buffer with 1-way
contention and chains it later).

**Changes.**
1. New `higs_camera_positions_kernel` computes `cam_pos` once per camera
   per backward into a tiny `[C, 3]` tensor; the VJP kernel loads it instead
   of re-deriving the transpose.
2. The D channel lanes of one (camera, gaussian) are consecutive
   (`c == t % D`), so the kernel reduces their `v_dir` partials with two
   `__shfl_down_sync` and issues **one** atomicAdd per output coordinate
   from the group leader. Partial warps (grid tail) and groups straddling a
   warp boundary fall back to per-lane atomics; masked lanes participate in
   the shuffles with `v_dir = 0`. The reduced sum is deterministic (a+b+c in
   lane order) and bit-compatible with the previous nondeterministic atomic
   order at test tolerance.

**A/B (EPIC-05, bicycle 1080p x 4 cams, torch-profiler self-CUDA-time):**

| kernel | before | after |
|---|---|---|
| higs_sh_vjp_grid_kernel | 5.955 / 5.955 ms | 5.385 / 5.390 ms |
| _HigsAutogradFunctionBackward total | 37.13 ms | 36.57 ms |
| blend bwd (unchanged control) | 29.08 ms | 29.08 ms |

Full paired benchmark (bicycle 1080p x 4 cams, steps 20):
`higs_native` bwd 40.0 -> 39.4 ms, total 66.9 -> 66.3 ms (-0.6 ms/step);
`higs_dynamic` total 56.3 ms. Quality unchanged (native PSNR 16.75 /
SSIM 0.4620 / LPIPS 0.5518, dynamic 17.61 / 0.4833 / 0.4968; grad cosine
1.0; parity PSNR 18.81). All 100 tests pass.

**Boundary of this win.** The remaining ~5.4 ms of the SH VJP is dominated
by the 145M coefficient atomics (16 per thread for K=16), which are
inherent to the visible-subset layout; a round-13 per-camera flat probe
measured only -0.3 ms. The last big lever stays the HiGS-format backward
(30M macro-tile entries vs 330M std isects).
### Round 26 (2026-08-02): per-(camera,gaussian) ellipse AABB prefilter measured and reverted (A/B negative again)

**Hypothesis.** Round-24's AABB prefilter was negative, but its extent math ran
per isect inside the batch load (`logf/sqrtf/div` per isect + 2 extra shared
floats). Round-25 diagnostics decomposed the blend kernel into ~3.8 ms pure
traversal, ~15.8 ms eval + warp-reduce + atomic and ~9.5 ms VJP math (29.1 ms
total). This round kept the same ellipse prefilter but hoisted the extent math
out of the blend kernel entirely: a new `higs_compute_aabb_kernel` computes
`ex/ey = sqrt(k*conic.z/det), sqrt(k*conic.x/det)` (`k = 2*ln(opac/AT)`) once
per (camera, gaussian) pair (9.1M pairs for bicycle 1080p x 4 cams) into a
flat `[I*N, 2]` tensor; both blend kernels load it into a shared
`vec2 aabb_batch` (same 8B/entry footprint as round-24's 2 floats) and gate
the q-loop with `in_ellipse = ex<=0 || (|dx|<=ex && |dy|<=ey)` before
`eval_gaussian_weight`, skipping the eval + accumulated VJP body for
out-of-ellipse pixels.

**Why it still loses.** The prefilter can only skip `eval_gaussian_weight`
(2 FMAs + `__expf` + clamp + compare): the VJP math is already gated by
`gw.valid` in the unfiltered kernel, and the per-isect warp-reduce/atomic
scatters run whenever any lane is valid, which the AABB (a superset of the
valid set) leaves unchanged. So the filter trades a cheap, fully-parallel
`__expf` for one extra scattered 8-byte global load per isect (`aabb[g]`) in
the batch load, the shared write/read of `aabb_batch`, two `fabsf`+compare
branches per (isect, pixel) pair, and extra register pressure - and the blend
kernel is memory/launch-bounds bound. The 15.8 ms eval+reduce+atomic block is
dominated by the reduction/atomic scatter, which no AABB scheme can skip.

**A/B (EPIC-05, bicycle 1080p x 4 cams, torch-profiler self-CUDA-time, PX=2):**

| variant | blend bwd px kernel (self) |
|---|---|
| baseline (ef8fcb3) | 29.08 / 29.18 / 29.23 ms |
| NEW (per-pair AABB) | 31.51 / 31.64 / 31.38 ms |

**Conclusion: reverted.** +2.3~2.6 ms on the blend kernel, reproducing
round-24's +2.1 ms even with the per-isect math removed - the extent-math
location was not the problem, the filter itself is. Source restored to the
ef8fcb3 PX=2 baseline (blend kernel back to 29.1 ms), 100 tests pass. No
further AABB variants are planned; the remaining blend levers are the
per-isect reduce/atomic scatter and a HiGS-format backward consuming the
macro-tile structure (30M entries vs 330M std isects).
### Round 27 (2026-08-02): blend-backward shared-memory slot accumulation measured and reverted (A/B negative)

**Hypothesis.** Round-26 closed the prefilter levers: the blend kernel is
memory/launch-bounds bound and the 15.8 ms eval+reduce+atomic block is dominated
by the per-isect warp-reduce and atomic scatter. This round attacked the scatter
itself. Each isect's warp leader currently emits 9 global atomics
(3 rgb + 3 conic + 2 xy + 1 opacity, CDIM=3) to the flat `[I*N, ...]` gradient
buffers, and the same Gaussian is typically visible in ~4 of the 8 pixel-bins
of a tile, so global atomics are ~4-way contended. The change added a per-batch
shared slot array `v_acc_batch[block_size][CDIM+6]`: warp leaders now do 9
*shared* atomics into their isect's slot, then after the t-loop one
`block.sync()` and a flush pass lets thread `tr < batch_size` scatter slot
`tr` to global with 9 atomics (plus a 9-float zero pass per batch). Global
atomic count drops ~4x; the shared layout (9-float stride, coprime with the
32 banks) is conflict-free.

**Why it still loses.** Measured blend bwd 29.08 -> 35.63/35.69 ms (+6.6 ms,
~23% regression), reproduced twice. The 9 global atomics per (isect, leader)
were not the bottleneck: the kernel is not L2-atomic-throughput bound. Moving
the contention to shared memory kept the same total atomic volume (9 shared +
9 global per slot) while adding a second reduction hop, a second
`block.sync()` per batch and a per-batch zero pass, all on a kernel that is
already launch-bounds/register bound. This closes the atomic-scatter lever with
the same verdict as rounds 24/26: restructuring the per-isect scatter inside
the current blend format does not help; the remaining lever is a HiGS-format
backward consuming the macro-tile structure (30M entries vs 330M std isects),
which changes the reduction shape rather than its location.

**A/B (EPIC-05, bicycle 1080p x 4 cams, torch-profiler self-CUDA-time, PX=2):**

| variant | blend bwd px kernel (self) |
|---|---|
| baseline (ef8fcb3) | 29.08 / 29.18 / 29.23 ms |
| NEW (shared-slot accumulation) | 35.63 / 35.69 ms |
| restored baseline | 29.20 ms |

**Conclusion: reverted.** Source restored to the ef8fcb3 PX=2 baseline,
100 tests pass.

### Round 28 (2026-08-02): HiGS macro-tile backward feasibility quantified and refuted (40.5M vs 11.2M isects; 6.2G per-pixel evals are format-independent)

**Context.** Rounds 24/26/27 closed the AABB-prefilter and shared-atomic-scatter
levers inside the current blend format. The last advertised lever was a
HiGS-format backward consuming the macro-tile structure ("30M vs 330M isects").
This round measured the actual quantities on EPIC-05 (bicycle 1080p x 4 cams,
eps2d=0.3, visible subset 2.27M of 6.13M gaussians) to bound the achievable
gain *before* any rework:

| quantity | value |
|---|---|
| std-format isects (visible subset, tile 16) | 40.5M |
| HiGS macro-tile entries (visible subset, tile 8) | 11.2M (4 cams) |
| HiGS macro-tile entries (visible subset, tile 16) | 9.1M (4 cams) |
| per-pixel eval depth (bin_final - tile_start + 1) | 751 avg |
| total per-pixel evals (format-independent) | 6.23G |

Note the earlier "330M std isects" figure was the full-scene (un-culled) count;
the culled-visible subset at 1080p is 40.5M. The mt format cuts the
*isect-entry* count ~3.6-4.5x (40.5M -> 11.2M/9.1M), but every pixel still
evaluates every sorted gaussian up to its last contributor (751/pixel avg), and
the valid-pixel VJP volume is unchanged. The format only saves (a) per-entry
gaussian data loads (the 3.8 ms traversal of the 29.1 ms blend bwd) and (b) a
share of the per-(isect, warp) loop/reduce overhead; per-pixel eval (~8-10 ms)
and VJP math (~9.5 ms) are identical in both formats. Bounded ceiling: ~4-6 ms
of the blend kernel for a full rework (mt-buffer capture in the forward, a new
backward kernel with per-fine-tile warp masking, depth-cutoff mapping from the
std `last_ids` to mt order, parity/gradcheck suites).

The same economics apply to the forward: the HiGS renderer renders the packed
visible subset in ~2.1 ms/4 cams (tile 8, wall incl. pybind), but a
differentiable tile-LOD forward must still emit std-format capture outputs
(SH eval + render_alphas/last_ids/isect emission, ~4-5 ms per Round 19), so the
forward-side ceiling is also ~4-5 ms/step.

**Fresh baseline locked after the Round-27 revert (bicycle 1080p x 4 cams,
steps 20, 2026-08-02):**

| backend | fwd | bwd | tot | train | PSNR | VRAM |
|---|---|---|---|---|---|---|
| std_ll | 27.0 | 48.0 | 75.5 | 83.1 ms | 16.76 | 15.88 GB |
| higs_recompute | 28.7 | 88.7 | 117.9 | 125.4 ms | 16.74 | 18.80 GB |
| higs_native | 26.4 | 39.4 | 66.2 | 73.7 ms | 16.75 | 16.81 GB |
| higs_dynamic | 20.8 | 35.0 | 56.2 | 63.5 ms | 17.60 | 19.81 GB |

100 tests pass; native-vs-recompute grad cosine = 0.9999999.

**Conclusion.** The remaining theoretical levers (mt-format backward and
tile-LOD differentiable forward) are both quantified at ~4-6 ms/step ceilings,
below the cost/risk of a multi-week rework with high correctness risk. The
native/dynamic differentiable paths are at their practical optimum for the
current architecture; a HiGS-format differentiable pipeline would only pay off
if the format-independent per-pixel work itself could be reduced, which no
restructuring of the current math achieves.

### Round 29 (2026-08-02): mt-format per-pixel-eval assumption re-verified at kernel level (Round 28 conclusion confirmed)

**Context.** Round 28 closed the mt-backward lever with a ~4-6 ms ceiling,
arguing the 6.23G per-pixel evals are format-independent. One residual doubt
remained: if the HiGS rasterize kernel fused per-pixel evals into the
macro-tile structure (e.g., per-row or per-column sharing), an mt-backward
would exceed that bound. This round re-verified the assumption by reading the
actual kernels and re-profiling.

**Verification (kernel-level).**

1. `MacroTileRasterize.cu` `rasterizeGaussian` (phase 2 of
   `macro_tile_rasterize_kernel`): for every isect in a tile queue the
   kernel evaluates one `sigma = (l10*dy + l00*dx)^2 + (l11*dy)^2 - log2(opac)`
   and one `exp2(-sigma)` per pixel (N_PAIRS half2 lanes), then per-pixel
   T-chain multiply and color FMA. The half2 lanes are SIMD vectorization over
   pixel pairs, not amortization -- the exp2 count is one per (isect, pixel),
   identical to the std blend kernel eval_gaussian_weight.
2. The weight contains the cross term `2*l10*l00*dx*dy`, so it cannot factor
   into `f(x)*g(y)`; no row/column sharing exists to compress the per-pixel
   chain mathematically.
3. `_native_forward_capture` (`gaussian_inference.py:1863-1955`) feeds the
   backward the **std per-tile isect list** (40.5M entries via `isect_tiles`)
   plus std `render_alphas`/`last_ids`; `higs_blend_bwd_px_kernel` iterates
   that per-tile list with per-pixel `eval_gaussian_weight` + VJP. An
   mt-backward would consume the native 11.2M-entry list -- exactly the
   load-side saving (3.8 ms traversal) Round 28 credited, nothing more.
4. The backward per-pixel eval count equals
   `sum_pixels(last_ids - tile_start + 1)` = 6.23G (`count_probe2`), fixed by
   geometry/visibility in both formats.

**Fresh profile (2026-08-02, EPIC-05, bicycle 1080p x 4 cams):**
`higs_blend_bwd_px_kernel` = 29.14 ms (stable vs Round-28 29.1 ms); total
native backward 36.6 ms.

**Conclusion.** The mt-format hypothesis is false: the macro-tile structure
compresses the isect-entry list (~3.6-4.5x) and data-load traffic, but the
per-pixel eval+VJP volume (6.23G) is untouched, and no restructuring of the
current math removes it. Round-28 ceiling (~4-6 ms) and "lever closed"
status stand; no revision needed. The realized end-to-end speedups -- 73.7 vs
125.4 ms train vs recompute (-41%), 73.7 vs 83.1 ms vs std_ll (-11%), with a
backward 39.4 vs 88.7 ms vs recompute (2.25x) -- are bounded by the
format-independent per-pixel work that dominates both forward and backward.

**Final two-scene baseline (final code 9bbd720 / ef8fcb3 kernels, fresh run
2026-08-02, EPIC-05 A100, 1920x1080, 4 train cams + 3 eval cams, steps 20):**

| scene | backend | fwd | bwd | tot | train | VRAM | cull | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|
| train (1.03M) | std_ll | 11.3 | 22.3 | 34.0 | 35.9 ms | 3.23 GB | 0% | 19.02 | 0.6885 | 0.3850 |
| train | higs_recompute | 11.8 | 38.6 | 50.8 | 52.4 ms | 4.54 GB | 15.1% | 19.02 | 0.6885 | 0.3851 |
| train | higs_native | 11.7 | 18.9 | 31.0 | 32.6 ms | 3.57 GB | 15.1% | 19.02 | 0.6885 | 0.3853 |
| train | higs_dynamic | 9.7 | 16.8 | 27.0 | 28.9 ms | 4.02 GB | 15.5% | 20.09 | 0.7139 | 0.3567 |
| bicycle (6.13M) | std_ll | 26.8 | 48.0 | 75.2 | 82.7 ms | 10.74 GB | 0% | 16.75 | 0.4621 | 0.5516 |
| bicycle | higs_recompute | 28.7 | 88.7 | 117.9 | 125.4 ms | 13.67 GB | 62.9% | 16.75 | 0.4619 | 0.5517 |
| bicycle | higs_native | 26.4 | 39.5 | 66.3 | 73.8 ms | 11.67 GB | 62.9% | 16.76 | 0.4620 | 0.5518 |
| bicycle | higs_dynamic | 20.8 | 35.0 | 56.3 | 63.5 ms | 14.67 GB | 62.9% | 17.59 | 0.4832 | 0.4956 |

Total-iteration speedup vs std_ll on both scenes: higs_native -9.2% (train) /
-10.8% (bicycle), higs_dynamic -19.5% / -23.2%; native grad cosine vs recompute
0.999994 (train) / 1.000000 (bicycle). This is the completion-audit evidence
for the objective requirement that total iteration (forward + backward) must
actually benefit before claiming a speedup: both native and dynamic paths
satisfy it on both small and large scenes, so the speedup claim stands.

### Round 30 (2026-08-02): EPIC-05 environment recovery + reproducibility (container re-provisioned; all levers re-validated)

**Incident.** The EPIC-05 container was re-provisioned: /root (repo clone,
conda env, rebuilt extensions, /root/epic05-data) was wiped; hostname changed.
The shared storage /mnt/workspace/codex-3dgs-epic05 (raw/processed datasets,
downloads, state) and the local Windows tree (the authoritative gsplat HiGS
sources under rtifacts/renderer-sources/gsplat) survived.

**Recovery (all on the new container, CUDA 12.9.86 toolkit, 8x A100-80GB).**
1. miniforge -> gsplat env: python 3.10.20, torch 2.7.0+cu128 (matches the
   old py310_cu128 JIT cache ABI), plus benchmark/quality deps.
2. Repo restored at /root/3dgs-roadmap-matrix @ 5e5cbd; processed data
   symlinked (/root/epic05-data -> shared datasets/).
3. gsplat tree (73.5 MB, incl. the untracked .cu kernels) re-synced from the
   local tree; CRLF -> LF normalized.
4. Build fix required: gsplat/cuda/csrc/Utils.cpp called the 2-arg
   cudaEventCreate(&e, flags) (a CUDA 12.9 header mismatch that the old flow
   never hit because it only rebuilt the experimental extension against a
   prebuilt gsplat). Fixed to cudaEventCreateWithFlags(&e, flags) (2 sites).
5. Built both extensions in-place (setup.py build_ext --inplace, ninja,
   MAX_JOBS=6): gsplat/csrc.so + gsplat/experimental/render/kernels/csrc.so.

**Validation (fresh).**
- 100 tests pass (	est_higs_native_backward/frozen/dynamic/trainable).
- Two-scene benchmark parity vs the Round-29 locked baseline (train/bicycle,
   steps 20): higs_native 33.0/73.8 ms (baseline 32.6/73.8), higs_dynamic
   29.4/63.6 (baseline 28.9/63.5), PSNR and native-vs-recompute grad cosine
   (0.9999999/1.000000) match. Rebuilt environment reproduces the locked
   numbers within noise.

**Reproducibility safeguards (so this never costs 2h again).**
- scripts/linux/rebuild_higs_csrc.py (tracked) rebuilds both extensions
  incrementally via ninja and clears the stale torch JIT cache; this replaces
  the lost /root/rebuild_higs_csrc.py.
- Full gsplat HiGS source snapshot archived to shared storage
  /mnt/workspace/codex-3dgs-epic05/state/backups/gsplat-higs-source-2026-08-02.tar.gz
  (145 MB; the rtifacts/renderer-sources/ tree is gitignored so it is not
  in the repo).
- The Utils.cpp cudaEventCreateWithFlags fix is included in both the local
  tree and the shared-storage archive.

**Re-verification of optimization headroom (fresh profile, bicycle 1080p x 4
cams, native): blend bwd 26.3 ms, forward capture rasterize 10.4 ms, SH VJP
5.4 ms, fused Adam ~7.5 ms (shared with std), radix sort 2.1 ms, projection
bwd 2.0 ms, projection fwd 1.8 ms, intersect 1.4 ms, SH fwd 1.1 ms, grad-zero
fills 1.0 ms, culling gathers 0.8 ms. No new lever appears; the closed
Round-28/29 conclusions (mt-backward ~4-6 ms ceiling, tile-LOD forward
~4-5 ms ceiling, per-pixel eval volume format-independent) hold on the rebuilt
environment.

### Round 31 (2026-08-03): tile-sampled training - M2 done, M3 partial, M4 negative for r<=0.25 (honest quality bounds for the speedup lever)

**Lever implemented and verified.** Rounds 21-29 established that the HiGS
backward is dominated by per-pixel VJP volume (6.2G evals on bicycle 1080p x 4
cams) and that no backward-format change can reduce it (Round 28: macro-tile
format ceiling ~4-6 ms). Tile-sampled training is the lever that cuts the
volume: an optional `tile_sampling_ratio` (default 1.0 = full frame) plus
`sampling_mode` (`uniform` | `stratified`) was added to the native capture
path (`_HigsAutogradFunction._native_forward_capture`). The isect buffer is
masked to the sampled tiles before the dense-offset re-encode, so forward
intersect + blend and the blend backward only touch the selected tiles. The
sampled-tile mean is an unbiased estimator of the full-frame mean, so the
harness loss needs no 1/r rescale - it masks the L1 loss over the sampled
tiles (`_masked_l1_loss`; new benchmark backends `higs_native_ts` /
`higs_dynamic_ts`, new CLI flags `--tile-sampling-ratio` and
`--sampling-mode`).

**Bug found and fixed (multi-camera isect filtering).** `sampled_ratio` was
computed as `mask.sum() / n_tiles` over a `[C, n_tiles]` mask, so C=2 + r=0.5
gave 1.0 and silently skipped the isect filtering (old stratified runs all
reported `isect_frac=1.0`). Fixed to `float(mask.float().mean())`.
Regression test `test_tile_sampling_ratio_multi_camera_filter` (C=2,
r=0.5/0.25, mask fraction and isect fraction match r) passes on EPIC-05.

**Speed (sequential, uncontended runs, frozen 1080p x 4 cams x 20 steps; the
earlier parallel matrix was discarded - concurrent runs inflated every cell
by 2-4x, e.g. train r=1.0 measured 121 ms vs 32.4 ms when run alone).**

| scene | r=1.0 | r=0.5 | r=0.25 | r=0.125 |
|---|---|---|---|---|
| train | 32.4 / 19.8 | 27.4 / 13.0 | 21.8 / 9.2 | 18.5 / 7.3 |
| truck | 39.0 / 24.6 | 31.5 / 16.2 | 21.3* / 11.5 | 21.3 / 8.7 |
| bicycle | 66.4 / 39.4 | 52.3 / 25.3 | 40.6 / 17.7 | 34.2 / 13.8 |
| bonsai | 26.1 / 16.2 | 22.7 / 10.6 | 17.3 / 7.2 | 14.6 / 5.2 |
| garden | 64.5 / 40.6 | 48.3 / 26.0 | 38.2 / 18.7 | 32.2 / 14.6 |

`total / bwd` ms per step. *truck r=0.25 fwd was a one-off outlier (fwd 30.8
vs ~14 in the other truck cells); bwd 11.5 scales correctly. Total savings
(median, excluding the truck outlier): r=0.5 ~-15..-25%, r=0.25 ~-33..-41%,
r=0.125 ~-43..-50%. The blend backward scales ~linearly (bwd at r=0.5 is
~64% of full, r=0.25 ~45%, r=0.125 ~35%) but a fixed floor of ~5-6 ms
(projection VJP + SH VJP + grad-zero fills + culling) does not shrink, so
total iteration saves are well below 1/r.

**Quality (300-step protocol, seed 0, 4 train + 3 eval cams; PSNR / SSIM /
LPIPS).**

Frozen (`higs_native_ts`):

| r | sampling | train | bicycle |
|---|---|---|---|
| 1.0 | - | 16.013 / 0.6329 / 0.3968 | 16.135 / 0.4364 / 0.4747 |
| 0.5 | uniform | 15.808 / 0.6271 / 0.4264 | 16.262 / 0.4432 / 0.5132 |
| 0.5 | stratified | 15.995 / 0.6290 / 0.4189 | 16.027 / 0.4436 / 0.5120 |
| 0.25 | stratified | 15.812 / 0.6161 / 0.4675 | 15.567 / 0.4291 / 0.5775 |

Dynamic (`higs_dynamic_ts`):

| r | sampling | train | bicycle |
|---|---|---|---|
| 1.0 | - | 17.742 / 0.6466 / 0.4272 | 16.283 / 0.4061 / 0.5835 |
| 0.5 | stratified | 17.371 / 0.6329 / 0.4671 | 16.074 / 0.3972 / 0.6320 |

**Honest quality conclusion.** At r=0.5, PSNR/SSIM stay within single-seed
noise of full for both frozen scenes (train -0.02 dB / SSIM -0.004; bicycle
-0.11 dB / SSIM +0.007) but LPIPS degrades consistently (+0.022 train,
+0.037 bicycle). Stratified clearly beats uniform on train (PSNR -0.02 dB vs
-0.20 dB at r=0.5) but not on bicycle, where uniform r=0.5 even wins PSNR
(+0.13 dB) - single-seed, so the direction is noise-level. r=0.25 fails the
parity bar everywhere (train -0.20 dB, bicycle -0.57 dB; LPIPS +0.07/+0.10),
and dynamic r=0.5 loses 0.21-0.37 dB PSNR plus +0.04-0.05 LPIPS. The repo's
"quality must hold" constraint is therefore satisfied only for frozen r=0.5
PSNR/SSIM (single seed); the lever ships as opt-in with these documented
bounds, not as a default.

**Long-run check (1200 steps, dynamic, r=1.0).** train collapses to N=354K
(16.651 / 0.6019 / 0.4730 vs 17.742 at 300 steps) and bicycle to N=2.34M
(15.946 / 0.3715 / 0.6133) - the 300-step protocol is where the repo's
numbers live, and the N collapse is a protocol property (densify_every=5 /
fixed thresholds over-prune), not a tile-sampling artifact (measured at
r=1.0).

**Status.** M2 complete (blend bwd scales ~linearly, total iteration drops at
every r; r=0.5 total -15..-25%, r=0.25 -33..-41%). M3 partial: stratified
beats uniform on train at r=0.5 (-0.20 -> -0.02 dB); uniform beats stratified
on bicycle (+0.13 vs -0.11 dB PSNR); error-guided sampling and
densify-anchored steps are the obvious next experiments. M4 not met for
r<=0.25 or dynamic; the safe operating point is frozen r=0.5 stratified
(PSNR/SSIM parity, LPIPS +0.02).


### Round 32 (2026-08-03): anchor-densify, multi-seed quality, and error-guided tile sampling (M3 lever; honest LPIPS bound)

Three additions landed on top of Round 31's `tile_sampling_ratio` + uniform/
stratified machinery: (1) anchor-densify (full-res forward+backward on
densify steps), (2) multi-seed quality verification of the frozen r=0.5
operating point, and (3) error-guided tile sampling with an unbiased
importance-weighted loss. Protocol is unchanged: 300 steps, 4 train + 3 eval
cams, 1080p, A100; results below are PSNR / SSIM / LPIPS.

**Anchor-densify (`--anchor-densify`, dynamic backend, single seed 0).**
Densify/prune steps (every 5) run at `sampling_ratio=1.0` so the topology
signal is full-res, while ordinary steps stay at r. On train this closes most
of the PSNR gap: r=0.5 anchor 17.827 / 0.6385 / 0.4596 vs dynamic full
17.742 / 0.6466 / 0.4272 (+0.09 dB, SSIM -0.008, LPIPS +0.032); r=0.25 anchor
17.463 / 0.6220 / 0.5068 (PSNR -0.28 dB vs full, SSIM -0.025, LPIPS +0.080),
vs -0.67 dB for the non-anchor r=0.25 run. On bicycle it does not recover:
r=0.5 anchor 16.080 / 0.3997 / 0.6201 (-0.20 dB, LPIPS +0.037 vs dynamic
full) and r=0.25 anchor 15.656 / 0.3848 / 0.6833 (-0.63 dB, LPIPS +0.100).
Directional only (seed 0): anchor-densify helps train PSNR but never
recovers LPIPS, which stays the binding quality limit.

**Multi-seed frozen r=0.5 vs full (seeds 0, 1, 2; stratified).**

| scene | full r=1.0 | r=0.5 stratified | delta |
|---|---|---|---|
| train | 16.006 / 0.6316 / 0.3987 | 16.017 / 0.6308 / 0.4203 | PSNR +0.01, SSIM -0.001, LPIPS +0.022 |
| bicycle | 16.178 / 0.4363 / 0.4745 | 16.083 / 0.4424 / 0.5117 | PSNR -0.10, SSIM +0.006, LPIPS +0.037 |

So frozen r=0.5 stratified holds PSNR/SSIM parity on both scenes across
seeds; **LPIPS degrades +0.02..+0.04 and is the consistent, honest quality
bound of tile-sampled training at r=0.5.**

**Error-guided sampling (`--sampling-mode error_guided`).** The harness
refreshes an exact per-tile mean-|diff| map every `--error-refresh-every`
(default 25) steps with a full-res forward, then draws k tiles per image with
replacement with p proportional to (err + floor)^alpha (`--error-alpha`). The
explicit `tile_mask` (bool [C, th, tw]) is passed to the rasterizer (new
optional argument on the frozen/dynamic public APIs; applied verbatim to the
isect buffer) and the loss is the exact unbiased importance estimate
`(1/P) sum_t m_t w_t S_t`, `w_t = m_t/(k p_t)`, `P = C*W*H*3`. The estimator
is unbiased for any p>0 (verified by a CPU test: 40 draws mean approx full-frame
mean).

| scene | mode | train | bicycle |
|---|---|---|---|
| train | full r=1.0 (3 seeds) | 16.006 / 0.6316 / 0.3987 | - |
| train | error-guided r=0.5, alpha=1.0 (4 seeds) | **16.816 / 0.6390 / 0.4142** | - |
| train | error-guided r=0.25, alpha=1.0 (3 seeds) | **16.848 / 0.6299 / 0.4455** | - |
| train | error-guided r=0.5, alpha=0.5 (2 seeds) | 15.972 / 0.6254 / 0.4375 | - |
| train | error-guided r=0.25, alpha=0.5 (3 seeds) | 16.535 / 0.6254 / 0.4536 | - |
| bicycle | full r=1.0 (3 seeds) | - | 16.178 / 0.4363 / 0.4745 |
| bicycle | error-guided r=0.5, alpha=0.5 (3 seeds) | - | 15.897 / 0.4356 / 0.5517 |
| bicycle | error-guided r=0.5, alpha=1.0 (2 seeds) | - | 15.691 / 0.4327 / 0.5587 |

On train, alpha=1.0 (the variance-optimal exponent for L1) is the first
sampling mode to beat full-res training on PSNR at r<1: r=0.5 +0.81 dB and
r=0.25 +0.84 dB over the 3-seed full baseline, every seed above every full
seed (all four r=0.5 seeds and all three r=0.25 seeds >= 16.35). SSIM stays at
parity-or-better; LPIPS is +0.016 (r=0.5) / +0.047 (r=0.25). The earlier
single-seed +1.09 dB at r=0.25 alpha=0.5 did not replicate (seed 2 = 15.765,
below the full mean) - alpha=1.0 is the robust setting on train. On bicycle
error-guided is the worst mode (PSNR -0.28..-0.49, LPIPS +0.08): the
importance emphasis on high-error tiles hurts the outdoor scene, so the mode
is scene-dependent and reported as a negative there. Per-step wall-clock at
these settings: train r=0.5 25.2 ms vs full 34.3 ms (-26%), r=0.25 21.7 ms
(-37%); the full-res refresh adds ~14.5 ms every 25 steps (~0.6 ms/step
amortized) on train and ~32.6 ms on bicycle.

**Tests.** New `tests/test_benchmark_tile_sampling.py` (3 CPU tests: exact
border-tile errors, unbiased estimator across 40 draws, mask fraction <= ratio
with finite weights); `tests/test_higs_frozen.py` gains
`test_external_tile_mask_filters_isects` (2-camera CUDA: mask applied
verbatim, isect fraction matches the mask, backward succeeds) and an autouse
`_reset_frozen_tracker` fixture that fixes a module-level Gaussian-count
tracker leak across tests (50 -> 80 was contaminating later assertions).
Remote EPIC-05 suite: 105 passed; local Windows: pytest 158 passed / 103
skipped, unittest 155 OK.

**Round 32 bottom line.** PSNR/SSIM parity at frozen r=0.5 is now verified
across seeds; error-guided alpha=1.0 turns sampled training into a PSNR win
on train at both r=0.5 and r=0.25, but is a loss on bicycle; and **LPIPS
degrades in every r<1 mode on both scenes (+0.02..+0.08) - the honest quality
bound that still prevents M4 from being fully green.**

### Round 33 (2026-08-04): uniform-mix lambda knob + 3000-step horizon probe (M4 partial; LPIPS bound unchanged)

Two harness additions: (1) `--error-lambda` blends the error-guided tile
distribution with the uniform distribution (`p = (1-lambda)/n + lambda*p_err`;
default 1.0 = pure error-guided, 0.0 = uniform), keeping the importance
estimator unbiased for any lambda; (2) `--eval-every N` records an `eval_curve`
(step / PSNR / SSIM / LPIPS / n_gaussians) at full resolution during training,
enabling horizon studies. `tests/test_benchmark_tile_sampling.py` gains 3 CPU
tests for the mix (lambda=0 estimator unbiased, mask fraction and
`m*n/k`-shaped weights, lambda=1 default unchanged); 6/6 pass on local CPU and
on EPIC-05.

**Lambda sweep (bicycle, r=0.5, alpha=1.0, 300-step protocol, seed 0).**
PSNR / SSIM / LPIPS:

| lambda | PSNR  | SSIM   | LPIPS  |
|--------|-------|--------|--------|
| 0.70   | 15.964| 0.4376 | 0.5492 |
| 0.85   | 15.826| 0.4325 | 0.5574 |
| 0.90   | 16.056| 0.4339 | 0.5575 |
| 1.00   | 15.794| 0.4333 | 0.5567 |

lambda=0.70 is best and was repeated on seed 1 (15.756/0.4328/0.5575): the
2-seed mean 15.860/0.4352/0.5534 vs the Round-32 lambda=1.0 2-seed mean
15.691/0.4327/0.5587 is +0.17 dB PSNR and -0.005 LPIPS - a small,
seed-dependent recovery that does **not** close the gap to stratified (0.5117)
or full (0.4745) LPIPS. On train the mix does not help: lambda=0.70 (2 seeds)
16.678/0.6382/0.4183 is -0.14 dB / +0.004 LPIPS vs lambda=1.0 (4 seeds)
16.816/0.6390/0.4142, so pure error-guided remains the train operating point.

**3000-step horizon probe (seed 0, eval every 300 steps).** The frozen
topology + L1-only protocol is **not** a stable long-horizon regime: both modes
collapse on train after ~300 steps, and the sampled variant collapses more
slowly (step-300 / step-3000 = PSNR/SSIM/LPIPS):

| scene   | mode                          | step-300               | step-3000              | train_ms |
|---------|-------------------------------|------------------------|------------------------|----------|
| train   | full (r=1.0)                  | 16.024/0.6318/0.3989   | 12.499/0.5416/0.5638   | 37.0 |
| train   | error-guided r=0.5            | 17.009/0.6390/0.4170   | 13.197/0.5756/0.5454   | 31.6 |
| bicycle | full (r=1.0)                  | 16.162/0.4368/0.4733   | 15.433/0.4023/0.4904   | 97.2 |
| bicycle | error-guided r=0.5 (lambda=.7)| 16.093/0.4379/0.5461   | 14.290/0.4169/0.5424   | 88.6 |

At 3000 steps error-guided is still +0.70 dB / -0.018 LPIPS better than full
on train but -1.14 dB / +0.052 LPIPS worse on bicycle; the step-300 points
reproduce the Round-32 300-step protocol within seed noise (+-0.13 dB),
validating `eval_curve`. The r<1 LPIPS bound is unchanged: the bicycle gap
(0.542-0.558 vs 0.4745 full) is a property of the importance-weighted loss at
any tested lambda and horizon. Timing at 3000 steps (within-session,
sequential): train 37.0 -> 31.6 ms (-14.6%), bicycle 97.2 -> 88.6 ms (-8.9%);
absolute ms are inflated vs the 300-step protocol by sustained-load clock
throttle, and the full-res error refresh (57.2 ms on bicycle every 25 steps)
eats most of the bicycle margin (backward still saves 17 ms/step there).

**Round 33 bottom line.** The uniform-mix lambda knob is a weak, honest
mitigation for bicycle LPIPS (+0.17 dB PSNR / -0.005 LPIPS at lambda=0.70,
seed-dependent) and does not change the train operating point; the 3000-step
probe shows the frozen protocol itself (not sampling) is the quality ceiling
at long horizons, so 30k-step M4 validation requires the full dynamic pipeline
(densify/prune + schedule), which remains open.

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
4. After three forward optimizations (batched culling projection, per-step
   pack skip, native visible-subset gather) plus three backward rounds (SH VJP
   visible-pair compaction + channel-adjacent thread order, blend VJP
   `__launch_bounds__(256, 5)` register fix, and the C++ direct-master
   gradient scatter that removed the Python `index_copy_` scatter, and the fused
   union-visibility mask kernel that removed the Python double reduction), the
   frozen native path is faster than std gsplat end-to-end on both scenes
   (-15.6% train / -13.5% bicycle, 2026-08-01m benchmark) and the dynamic path
   is -24.3% / -29.6%. The remaining structural forward costs are the batched
   FP32 culling projection over all N x C (~2.3 ms on bicycle) and the
   visible-subset second projection (~0.55 ms), the price of
   projection-consistent discrete culling; the native backward contributes no
   recomputation. The differentiable forward no longer re-packs the FP16
   scene per step (parameter drift only updates the handle version
   bookkeeping; `pack_gaussian_inference_scene` runs only on real topology
   changes).
5. `_densify_gaussians` clamps RGB colors to [0,1] by default;
   `color_clamp=None` disables the clamp. SH coefficient tensors ([N, K, C])
   are intentionally not clamped.a6. The SH VJP kernel remains ~1.1 ms slower than std's on bicycle (5.97 vs
   4.89 ms). A round-13 probe (2026-08-02) refuted the master-buffer
   `v_means` atomic-contention hypothesis: a per-camera flat accumulation
   variant (3-way vs 12-way contention) measured 5.67 ms, only -0.3 ms
   (-5%), with total iteration unchanged (41.9 vs 42.0 ms). The remaining
   gap is the ~109M coefficient atomics plus the `colors_eval` ReLU-mask
   load, both of which std shares in different form; a std-style per-camera
   `v_dirs` intermediate + reduction pass would cost a 294 MB buffer and a
   reduce pass for ~0.3 ms, so it is kept as a documented, not-worth-it
   trade-off.
7. A topology rebuild on a 6.13M-Gaussian scene costs ~3.6 ms to pack +
   construct the renderer and ~+9 ms per `mark_dirty()`-forced forward+backward
   step; amortized over densify-every-5 training this is ~+1.4 ms/step.

## Modified files (gsplat source tree)

- `gsplat/experimental/render/functional/gaussian_inference.py` — native forward
  capture, native/recompute backward, `HigsRendererHandle`, `create_higs_renderer`,
  `sync_optimizer_state_for_topology_change`, dynamic-scene handle ownership.
- `gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.{cu,h}` — new.
- `gsplat/experimental/render/kernels/cuda/build.py` — include `HigsNativeBackward.cu`.
- `gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/GatherVisible.{cu,h}` — new
  compact-copy kernel `higs_gather_visible` (visible-subset gather, avoids PyTorch's
  slow vectorized row gather for row widths divisible by four) and the fused
  `higs_union_visible_mask` kernel (union visibility mask over cameras from
  `[C, N, 2]` int32/FP32 projection radii).
- `gsplat/experimental/render/kernels/cuda/ext.cpp` — `higs_rasterize_backward`,
  `higs_gather_visible` and `higs_union_visible_mask` bindings.
- `tests/test_higs_native_backward.py` — new native-backward test suite.
- `benchmark/run_higs_train_benchmark.py` — new training benchmark.
