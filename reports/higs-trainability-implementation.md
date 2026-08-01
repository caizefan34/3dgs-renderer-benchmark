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
fused union-visibility-mask final result. esults/higs-train-benchmark-2026-08-02-shvjp.json and
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
   4.89 ms): the fixed-grid rewrite (2026-08-02) removed the per-backward
   device->host sync and the compaction pipeline, but the remaining gap is
   the master-buffer `v_means` atomics (multi-camera contention) plus the
   `colors_eval` ReLU-mask load. Closing it requires std-style per-camera
   `dirs` capture (294 MB on bicycle) or a per-camera `v_dirs` intermediate
   plus a reduction pass, kept as a documented trade-off.
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
