# HiGS Trainable Implementation (Stage A/B/C)

> **STATUS: SUPERSEDED (2026-07-31).** Stages A/B/C were superseded by the HiGS
> **native CUDA backward** implementation — see [implementation report](../reports/higs-trainability-implementation.md)
> and README. The differentiable HiGS path now uses `backward_backend="higs_native"` by default
> (87 tests passing on EPIC-05, 13.09s), with gsplat recomputation kept only as an explicit
> `backward_backend="gsplat_recompute"` fallback. This page is kept for historical reference.

**Status: DELIVERED — 38 tests passing on EPIC-05 (A100-80GB, 5.53s)**

This page summarizes the work that made gsplat's HiGS inference path **trainable end-to-end**.
Full details: [implementation report](../reports/higs-trainability-implementation.md).

## Why this was needed

HiGS was inference-only for three reasons:

1. `check_inference_grad_mode()` forbids autograd during forward.
2. `GaussianInferenceScene.from_gaussian_tensors()` detaches requires-grad inputs.
3. The CUDA extension registers Autograd fallthrough with no backward kernel.

## What was delivered

| Stage | Public API | Key components | Tests |
|---|---|---|---|
| A. Correctness baseline | `rasterize_gaussian_higs_trainable()` | `differentiable=True/False`; standard gsplat backward as recomputation proxy; no detach / no grad guard | 13 |
| B. Frozen topology | `rasterize_gaussian_higs_frozen()` | `freeze_topology=True/False`; `_HigsAutogradFunction` native autograd backward; HiGS-native culling via `get_visible_mask`; scatter-add grad (packed→original IDs) | 14 |
| C. Dynamic topology | `rasterize_gaussian_higs_dynamic()` | `_HigsDynamicScene` versioned scene buffers; `_densify_gaussians()` / `_prune_gaussians()` between steps; multi-step training smoke test | 11 |

## Verification results

- Gradients flow to **all 5 parameter types** (means, quats, scales, opacities, colors/SH) — finite-difference verified.
- Training dynamics are **identical to standard gsplat**: gradient cosine similarity = 1.000000, loss/PSNR curves match, forward+backward aligned at atol=1e-5.
- All 5 params remain **FP32 master tensors**; lossy SH compression is
  trainable via a straight-through estimator (FP16 cast); the native backward
  supports pinhole/ortho/fisheye camera models.

## Reproduce

```bash
# On EPIC-05 (or any CUDA Linux machine):
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
BUILD_EXPERIMENTAL=1 pip install -e artifacts/renderer-sources/gsplat --no-build-isolation
python3 -m pytest tests/test_higs_trainable.py tests/test_higs_frozen.py tests/test_higs_dynamic.py -v
```

## Known limitations

- All stages use standard gsplat `rasterization()` for the backward pass — no HiGS-native gradient computation yet.
- Python-side bitmask decode in `_cull_gaussians_higs()` is inefficient for large N.
- `_HIGS_DYNAMIC_SCENE` is a module-level singleton requiring `reset()` between test runs.
- HiGS culling ratio is 0% for typical scenes with all Gaussians in view.
- Patch: `patches/higs-differentiable.patch` (applies to gsplat source).
