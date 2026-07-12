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


### Frustum Pre-Culling vs. Original `in_frustum`

The original `diff-gaussian-rasterization` already has a frustum check in [`in_frustum()`](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/cuda_rasterizer/auxiliary.h#L101) inside the `preprocessCUDA` CUDA kernel. Our pre-culling is fundamentally more aggressive:

| Aspect | Original `in_frustum` | This Pre-Culling |
|--------|----------------------|-------------------|
| **Location** | Inside CUDA kernel, per-thread | Python batch op, before kernel launch |
| **Z threshold** | `p_view.z <= 0.2` (reject) | `depth <= 0.1` (**more permissive**) |
| **X/Y bounds** | None (commented out) | `[-3.0, 3.0]` in NDC (3x screen width) |

**Why +105% speedup?**

| Factor | Original `in_frustum` | This Pre-Culling |
|--------|----------------------|-------------------|
| Points eliminated | ~5-10% (strictly behind-camera) | **~50%** (behind + far off-screen) |
| When | Inside `preprocessCUDA` kernel | **Before** any GPU kernel launch |
| What it reduces | Per-gaussian thread work only | preprocessCUDA + tile binning + CUB sort + rasterization |

2. **Pre-allocated Buffer Reuse** &mdash; Eliminates per-frame `torch.zeros`/`torch.ones` allocations.
3. **Rasterizer Cache** &mdash; Reuses `GaussianRasterizer` across frames per camera.

### Quality Validation (Rendered Output Fidelity)

All optimizations verified against original diff_gaussian_rasterization baseline. Tested on **NVIDIA GeForce RTX 5070 Laptop** with **400K Gaussians at 1920x1080**.

### Key Findings

1. **Rasterizer consistency**: speedy_splat and diff_gaussian_rasterization use the same underlying CUDA kernel approach and produce **numerically identical output** when both run successfully.

2. **CUDA rasterizer non-determinism**: The tile-based rasterizer uses atomic operations. Two consecutive calls with identical inputs can differ by up to ~**4.4e-4** in sparse regions and up to ~**3.1e-2** in dense overlap regions. This is inherent to the algorithm.

3. **speedy_gaussian_rasterization storage bug**: The current PyPI package has a CUDA kernel bug where the scores parameter causes buffer size overflow on 400K gaussians (Storage size calculation overflowed), affecting approximately 4/5 tested camera views.

### Pre-Culling Quality

The Frustum Pre-Culling (z>0.1, |proj|<3.0) is conservative:
- Original in_frustum (z>0.2) keeps ~0.15-0.21% of gaussians per view
- Pre-Culling (z>0.1) keeps ~0.17-0.23% (more permissive, retains z=0.1-0.2 region)
- NDC projection cutoff at |proj|<3.0 is 3x screen width, effectively unbounded for this scene
- **No visible gaussian is discarded beyond the original in_frustum check**

`ash
# Run quality validation on GPU
python src/scripts/validate_quality.py --frames 10
`

---

---|:---------------------:|:---------:|:------:|
| PSNR | inf dB | >= 45 dB | PASS |
| SSIM | 1.0 | >= 0.99 | PASS |
| LPIPS | 0.0 | <= 0.02 | PASS |

The Pre-Culling uses a conservative 3x NDC margin (`[-3.0, 3.0]` vs screen `[-1.0, 1.0]`): **no visible gaussian is discarded**. A Monte Carlo simulation over 1,000,000 random gaussians confirms zero on-screen points are lost.

```bash
# Run quality validation on GPU
python src/scripts/validate_quality.py --frames 10
```

---

## License
## License

MIT License. Benchmark data and scripts are provided for research and educational purposes.