# HiGS Trainability Implementation Report

## Status: Stage A + Stage B (Frozen-topology native differentiable path) — VERIFIED on EPIC-05 (A100)

### Stage A: Correctness Baseline / Re-computation Proxy (COMPLETE ✅)
- [x] `check_trainable_grad_mode()` in `_common.py`
- [x] `rasterize_gaussian_higs_trainable()` with `differentiable=True/False`
- [x] API exports through `__init__.py` chain
- [x] Test suite (8 tests) **ALL PASS**

### Stage B: Frozen-topology HiGS native differentiable path (COMPLETE ✅)
- [x] `rasterize_gaussian_higs_frozen()` — differentiable rendering entry point
- [x] `_higs_frozen_forward()` — wrapper with full autograd support
- [x] CUDA extension: `getVisibleMask()` method on `GaussianInferenceRenderer`
- [x] API exports through all `__init__.py` files
- [x] Test suite (3 tests) **ALL PASS**:
  - Function importable
  - Gradients to all 5 parameter types (means, quats, scales, opacities, colors)
  - Forward output aligned with Stage A

### Not Yet Implemented (Stage C) — Dynamic topology
- [ ] Dynamic topology (versioned scene buffers)
- [ ] Densify/prune/clone with backward safety
- [ ] Multi-step topology mutation smoke test
- [ ] CUDA backward kernels for native HiGS gradient computation
- [ ] FP32 master → FP16 packed buffer sync with `packed_to_original_ids` scatter-add

### Parameters / Modes Covered
| Parameter | Stage A | Stage B | Stage C |
|-----------|---------|---------|---------|
| means (FP32) | ✅ (std gsplat) | ✅ (std gsplat) | pending |
| quats (FP32) | ✅ | ✅ | pending |
| scales (FP32) | ✅ | ✅ | pending |
| opacities (FP32) | ✅ | ✅ | pending |
| colors/SH (FP32) | ✅ | ✅ | pending |
| Pre-activated RGB | ✅ | ✅ | pending |
| SH coefficients | ✅ | ✅ | pending |
| SH compression | ⚠️ (default disabled) | ⚠️ | pending |
| freeze_topology | N/A | ✅ | N/A |
| dynamic topology | N/A | N/A | pending |

### Test Results (EPIC-05, 1x A100-SXM4-80GB)
```
tests/test_higs_trainable.py ........ [ 8/11]
tests/test_higs_frozen.py ...         [11/11]
============================== 11 passed in 3.92s ===============================
```

### Test Commands
```bash
# On EPIC-05 (or any CUDA Linux machine):
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
BUILD_EXPERIMENTAL=1 pip install -e artifacts/renderer-sources/gsplat --no-build-isolation
python3 -m pytest tests/test_higs_trainable.py tests/test_higs_frozen.py -v
```

### Environment
- **GPU**: NVIDIA A100-SXM4-80GB (8x)
- **CUDA**: 12.9 (nvcc) + 13.0 (PyTorch runtime)
- **PyTorch**: 2.13.0+cu130
- **gsplat**: 1.5.3 (editable install with BUILD_EXPERIMENTAL=1)

### Known Limitations
1. Stage A and B use standard gsplat `rasterization()` for the backward pass — no HiGS-native gradient computation yet.
2. The CUDA extension changes (ext.cpp, header) are included in the patch but require a rebuild to take effect.
3. No training speed improvement is expected from Stage A (by design). Stage B provides the frozen-topology API structure for future HiGS-native backward integration.
4. SH compression support is disabled in the differentiable path by default.

### Modified Files (gsplat source)
1. `gsplat/experimental/render/_common.py` — added `check_trainable_grad_mode()`
2. `gsplat/experimental/render/functional/gaussian_inference.py` — Stage A + Stage B functions
3. `gsplat/experimental/render/functional/__init__.py` — exports
4. `gsplat/experimental/render/__init__.py` — exports
5. `gsplat/experimental/__init__.py` — exports
6. `gsplat/experimental/render/kernels/cuda/ext.cpp` — added `get_visible_mask` binding
7. `gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/GaussianRenderInferenceScene.h` — added `getVisibleMask()` method

### Patch
A unified diff patch is available at `patches/higs-differentiable.patch`.
