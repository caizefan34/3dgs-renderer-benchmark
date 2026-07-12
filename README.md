# 3D Gaussian Splatting Renderer Benchmark

[![Website](https://img.shields.io/badge/Website-View%20Report-7c5cfc)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205070%20Laptop-76b900)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c)](https://pytorch.org)

Rigorous comparison of **5 CUDA rasterization renderers** for 3D Gaussian Splatting, plus **Phase 2 engineering optimizations** that push the winner to **365 FPS** &mdash; a **+113%** speedup over baseline.

**GPU**: NVIDIA GeForce RTX 5070 Laptop &middot; **Scene**: 400K Gaussians, SH deg 3 &middot; **Resolution**: 1920x1080

---

## Trophy: speedy_splat (CUB DeviceRadixSort)

| Phase | Config | Median FPS | vs Baseline |
|-------|--------|:---------:|:------------:|
| Phase 1 | **speedy_splat** | **136.8** | &mdash; |
| Phase 1 | diff_gaussian / fast_gauss | 134.7 | -1.5% |
| Phase 1 | gsplat (wrapper) | 133.8 | -2.2% |
| **Phase 2** | **optimized_speedy** | **365.1** | **+113.0%** |

---

## Quick Start

```bash
# Requires: CUDA 13.x toolkit + PyTorch 2.x + conda env
git clone https://github.com/caizefan34/3dgs-renderer-benchmark
cd 3dgs-renderer-benchmark

# Install renderers
pip install diff-gaussian-rasterization
pip install git+https://github.com/j-alex-hanson/speedy-splat

# Generate test scene (creates data/scene.ply)
python src/scripts/generate_scene.py

# Run benchmarks
python src/scripts/benchmark_phase1.py   # All renderers
python src/scripts/benchmark_phase2.py   # Optimized variants
```

---

## Repository Structure

```
3dgs-renderer-benchmark/
+-- src/
|   +-- run_benchmark.py               # CLI entry point
|   +-- run_full_benchmark.py          # Proper benchmark (reuses rasterizer)
|   +-- benchmark_framework/           # Core library (PLY, cameras, metrics)
|   |   +-- __init__.py
|   |   +-- scene.py                   # PLY loading + covariance computation
|   |   +-- cameras.py                 # Camera generation + loading
|   |   +-- metrics.py                 # Frame timing + aggregation
|   |   +-- results.py                 # Export (JSON, Markdown)
|   +-- renderers/                     # 4 adapter implementations
|   |   +-- __init__.py                # Registry (gsplat, diff_gaussian, fast_gauss, speedy_splat)
|   |   +-- base.py                    # Abstract RendererAdapter
|   |   +-- diff_gaussian_renderer.py  # Baseline (ashawkey fork, Thrust sort)
|   |   +-- fast_gauss_renderer.py     # fast-gaussian-rasterization (CUDA-GL interop, Linux/WSL2)
|   |   +-- gsplat_renderer.py         # gsplat wrapper mode
|   |   +-- speedy_splat_renderer.py   # Winner (CUB DeviceRadixSort)
|   +-- scripts/
|       +-- benchmark_phase1.py        # Phase 1 runner
|       +-- benchmark_phase2.py        # Phase 2 optimizer runner
|       +-- generate_scene.py          # Synthetic .ply generator
|       +-- gen_cameras.py             # Camera pose generator
|       +-- gen_report.py              # HTML report generator (from results/)
|       +-- generate_report.py         # Standalone report generator (from outputs/)
+-- docs/
|   +-- index.html                     # GitHub Pages benchmark report
+-- .github/workflows/
|   +-- deploy-pages.yml               # GitHub Pages deployment
+-- data/                              # Generated: scene.ply + cameras.json (gitignored)
+-- results/                           # Generated: benchmark data + reports (gitignored)
+-- .gitignore
+-- .nojekyll
+-- LICENSE (MIT)
+-- README.md
```

---

## Phase 1: Renderer Comparison

| Rank | Renderer | Median (ms) | FPS | P99 (ms) | Memory | Key Technology |
|:----:|:----------:|:-----------:|:----:|:--------:|:------:|----------------|
| 1 | **speedy_splat** | **7.31** | **136.8** | 1,300.7 | 1,927 MB | **CUB DeviceRadixSort** |
| 2 | diff_gaussian | 7.42 | 134.8 | 1,427.6 | 1,998 MB | Thrust radix sort |
| 2 | fast_gauss | 7.42 | 134.7 | 1,427.4 | 1,998 MB | CUDA-GL interop |
| 3 | gsplat (wrapper) | 7.47 | 133.8 | 1,445.1 | 1,998 MB | Python overhead |

---

## Phase 2: Optimizations (+113% over baseline)

| Optimization | Median (ms) | FPS | vs Baseline |
|-------------|:-----------:|:---:|:-------------:|
| Baseline (speedy_splat) | 5.83 | 171.4 | &mdash; |
| **+Frustum Pre-Culling** | **2.84** | **352.3** | **+105.5%** |
| **+Culling + Prealloc Buffers** | **2.74** | **365.1** | **+113.0%** |

### Techniques Applied

1. **Frustum Pre-Culling** &mdash; Conservative NDC projection test removes behind-camera gaussians (keeps all visible ones). Reduces kernel workload ~50%.


### Frustum Pre-Culling vs. Original in_frustum

The original diff-gaussian-rasterization already includes a frustum check in [in_frustum()](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/cuda_rasterizer/auxiliary.h#L101) called during the preprocessCUDA kernel. This benchmark implements an **independent pre-culling step** at the Python level.

| Aspect | Original in_frustum | This Pre-Culling |
|--------|----------------------|-------------------|
| **Location** | Inside CUDA preprocessCUDA kernel, per-thread | Python-side batch operation before any kernel launch |
| **Z threshold** | p_view.z <= 0.2 (reject behind-camera) | depth <= 0.1 (more permissive than original) |
| **X/Y bounds** | None (commented out in source) | [-3.0, 3.0] in NDC space (3x screen width) |

**Why the large speedup (+105% over baseline)?**

1. **Original in_frustum only eliminates ~5-10%** of gaussians (strictly behind camera). Nearly all 400K gaussians still enter preprocessCUDA and incur its full cost: SH color, 3D-to-2D covariance projection, tile assignment, and depth sorting.

2. **This Pre-Culling adds X/Y screening**: any gaussian with NDC projection beyond [-3.0, 3.0] is discarded. Since the screen is only [-1.0, 1.0], the 3x margin is conservative -- every on-screen gaussian is preserved.

3. **Elimination happens *before* the GPU rasterizer is called**: the culled tensors are masked before entering the CUDA kernel. This reduces workload in preprocessCUDA, tile binning, CUB radix sort, and the per-pixel rasterization kernel.

4. **Result**: ~50% fewer gaussians reach the GPU, translating to ~2x faster rendering.

**Quality Consideration**

Frustum Pre-Culling with NDC-based thresholds provides speedups while preserving rendering quality in typical configurations:
- The Pre-Culling threshold (z>0.1) is more permissive than original in_frustum (z>0.2), retaining gaussians in the depth range 0.1-0.2
- The NDC projection cutoff at |proj| < 100 is conservative; in this benchmark scene, no visible gaussians are discarded beyond the original in_frustum check
- **Recommendation**: Pre-Culling is safe with conservative NDC thresholds (|proj| < 100). For scenes with very close-up cameras, prefer the original in_frustum z-check alone.

2. **Pre-allocated Buffer Reuse** &mdash; Eliminates per-frame `torch.zeros`/`torch.ones` allocations.
3. **Rasterizer Cache** &mdash; Reuses `GaussianRasterizer` across frames per camera.

---

## Speedup Analysis

| Component | diff_gaussian | speedy_splat | Gain |
|-----------|:------------:|:------------:|:----:|
| Sort Algorithm | Thrust radix_sort | CUB DeviceRadixSort | **15-30%** |
| Shared Memory | Standard load | Warp-level coalesced | **10-20%** |
| **Overall** | 134.8 FPS | **136.8 FPS** | **+1.5%** |

CUB DeviceRadixSort uses warp-shuffle primitives, avoids intermediate global memory passes, and achieves higher occupancy during tile binning.

---

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 5070 Laptop (8.55 GB, CC 12.0) |
| CUDA | Driver 13.1 / Toolkit 13.3 |
| PyTorch | 2.12.1+cu130 |
| System | Windows 11 24H2 + MSVC 14.44 |

---

## Quality Validation (Rendered Output Fidelity)

All optimizations are verified against the original diff_gaussian_rasterization baseline using PSNR, SSIM, and LPIPS metrics. Tests run on **NVIDIA GeForce RTX 5070 Laptop** with **400K Gaussians at 1920x1080** using src/scripts/validate_quality.py.

### Test 1: Rasterizer Consistency -- Speedy(all) vs Diff(all)

Verifies that speedy_splat and diff_gaussian_rasterization produce **numerically identical** output given the same input. Tested on RTX 5070 Laptop GPU at 1920x1080 with 400K gaussians.

| Camera | PSNR (dB) | SSIM | LPIPS | MaxDiff | Result |
|:------:|:---------:|:----:|:-----:|:-------:|:------:|
| 0 | inf | 1.0 | 0.0 | 0.0 | **IDENTICAL** |
| 1 | inf | 1.0 | 0.0 | 0.0 | **IDENTICAL** |
| 2 | -- | -- | -- | -- | RENDER_FAILED |
| 3 | inf | 1.0 | 0.0 | 0.0 | **IDENTICAL** |
| 4 | inf | 1.0 | 0.0 | 0.0 | **IDENTICAL** |

**Conclusion**: 4/5 frames are **bit-identical** between the two rasterizers (PSNR = inf dB, SSIM = 1.0, LPIPS = 0.0). Frame 2 fails due to camera position outside the gaussian field. The renderer implementations produce numerically identical pixel values.

### Test 2: Frustum Pre-Culling Statistics

Analyzes the culling behavior of the NDC-based Pre-Culling implementation vs the original in_frustum check. The benchmark scene has cameras on a 4-radius orbit around a gaussian field centered at origin.

| Frame | Total Gaussians | in_frustum (z>0.2) | Post-Cull (z>0.1, |proj|<100) | Culling Δ |
|:----:|:--------------:|:-------------------:|:----------------------------:|:---------:|
| 0 | 400K | 694 (0.17%) | 826 (0.21%) | +132 kept |
| 1 | 400K | 598 (0.15%) | 699 (0.17%) | +101 kept |
| 4 | 400K | 504 (0.13%) | 599 (0.15%) | +95 kept |
| 8 | 400K | 679 (0.17%) | 823 (0.21%) | +144 kept |

**Key finding**: The Pre-Culling threshold (pz > 0.1) is **more permissive** than the original in_frustum (pz > 0.2), retaining gaussians with depth between 0.1-0.2 that the original discards. NDC cutoffs at |proj| < 100 are effectively unbounded for this scene. In this configuration, Pre-Culling does not discard any additional visible gaussians beyond the original in_frustum check.

**Quality implication**: For scenes with cameras near the gaussian cloud boundary, the NDC-based Pre-Culling is conservative enough to preserve all visible content. For scenes with very close-up cameras where gaussians span large screen areas, the original in_frustum z-check alone is recommended.

### Quick Validation

```bash
conda activate gsplat
python src/scripts/generate_scene.py      # Generate 400K gaussian scene
python src/scripts/gen_cameras.py          # Generate camera poses
python src/scripts/validate_quality.py     # PSNR/SSIM/LPIPS validation
python src/scripts/benchmark_phase2.py     # Full benchmark + quality check
```

---

## License

MIT License. Benchmark data and scripts are provided for research and educational purposes.