# HiGS Trainability Implementation Report

## Status: Stage A (Correctness Baseline / Re-computation Proxy)

### Completed
- [x] `check_trainable_grad_mode()` in `_common.py` ！ grad mode guard that allows autograd when `differentiable=True`
- [x] `rasterize_gaussian_higs_trainable()` in `functional/gaussian_inference.py` ！ new entry point with two modes
  - `differentiable=False`: delegates to existing `rasterize_gaussian_inference_scene()` (HiGS path)
  - `differentiable=True`: uses standard `gsplat.rasterization()` for the differentiable pipeline
- [x] API exports through `__init__.py` chain
- [x] Test suite covering:
  - API importability
  - Grad mode guards (both modes)
  - Gradient flow to all 5 parameter types (means, quats, scales, opacities, colors)
  - Forward output alignment with standard gsplat
  - Finite-difference gradient check on means

### Not Yet Implemented (Stage B)
- HiGS native autograd Function with CUDA backward kernel
- `freeze_topology=True` mode
- FP32 master tensors with FP16 packed buffer sync
- Native CUDA/Python unit tests for backward
- End-to-end training smoke test with HiGS-native gradients

### Not Yet Implemented (Stage C)
- Dynamic topology (versioned scene buffers)
- Densify/prune/clone with backward safety
- Multi-step topology mutation smoke test

### Parameters / Modes Covered
| Parameter | Stage A | Stage B | Stage C |
|-----------|---------|---------|---------|
| means (FP32) | ? (std gsplat) | pending | pending |
| quats (FP32) | ? (std gsplat) | pending | pending |
| scales (FP32) | ? (std gsplat) | pending | pending |
| opacities (FP32) | ? (std gsplat) | pending | pending |
| colors/SH (FP32) | ? (std gsplat) | pending | pending |
| Pre-activated RGB | ? | pending | pending |
| SH coefficients | ? | pending | pending |
| SH compression | ！ (train path default disabled) | pending | pending |
| freeze_topology | N/A | pending | pending |
| dynamic topology | N/A | N/A | pending |

### Numerical Tolerances
Not yet measured (requires Stage B native path).

### Test Commands
```bash
# Run Stage A tests (requires CUDA and gsplat with experimental module)
cd <repo-root>/artifacts/renderer-sources/gsplat
BUILD_EXPERIMENTAL=1 pip install -e .
pytest tests/experimental/render/test_trainable.py -v -s

# Or from repo root with the main repo tests
cd <repo-root>
pytest tests/test_higs_trainable.py -v -s
```

### Benchmark Results
Not yet measured (requires Stage B for native HiGS gradients).

### Known Limitations
1. Stage A uses standard gsplat `rasterization()` for the backward pass ！ no HiGS-native gradient computation.
2. HiGS preview under `torch.no_grad()` is optional; silently skipped if HiGS backend unavailable.
3. No training speed improvement is expected from Stage A (by design).
4. The gsplat source modifications are in the nested git repo at `artifacts/renderer-sources/gsplat/`. Apply `patches/higs-differentiable.patch` to reproduce.

### Modified Files (gsplat source)
1. `gsplat/experimental/render/_common.py` ！ added `check_trainable_grad_mode()`
2. `gsplat/experimental/render/functional/gaussian_inference.py` ！ added `rasterize_gaussian_higs_trainable()`
3. `gsplat/experimental/render/__init__.py` ！ exported new function
4. `gsplat/experimental/__init__.py` ！ exported new function
5. `tests/experimental/render/test_trainable.py` ！ Stage A test suite

### Patch
A unified diff patch is available at `patches/higs-differentiable.patch`.
