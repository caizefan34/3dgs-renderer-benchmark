# 3D Gaussian Splatting Renderer Benchmark

[![Website](https://img.shields.io/badge/Website-View%20Report-7c5cfc)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205070%20Laptop-76b900)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c)](https://pytorch.org)

Rigorous comparison of **5 CUDA rasterization renderers** for 3D Gaussian Splatting, plus **Phase 2 engineering optimizations** that push the winner to **365 FPS** — a **+113%** speedup over baseline.

**GPU**: NVIDIA GeForce RTX 5070 Laptop · **Scene**: 400K Gaussians, SH deg 3 · **Resolution**: 1920×1080

---

## 🏆 Winner: speedy_splat (CUB DeviceRadixSort)

| Phase | Config | Median FPS | Δ vs Baseline |
|-------|--------|:----------:|:-------------:|
| Phase 1 | 🥇 **speedy_splat** | **136.8** | — |
| Phase 1 | 🥈 diff_gaussian / TC-GS | 134.7 | −1.5% |
| Phase 1 | 🥉 gsplat (wrapper) | 133.8 | −2.2% |
| **Phase 2** | **🔥 optimized_speedy** | **365.1** | **+113.0%** |

---

## 🚀 Quick Start

```bash
# Requires: CUDA 13.x toolkit + PyTorch 2.x + conda env
git clone https://github.com/caizefan34/3dgs-renderer-benchmark
cd 3dgs-renderer-benchmark

# Install renderers
pip install diff-gaussian-rasterization
pip install git+https://github.com/j-alex-hanson/speedy-splat

# Generate test scene
python src/scripts/generate_scene.py

# Run benchmarks
python src/scripts/benchmark_phase1.py   # All renderers
python src/scripts/benchmark_phase2.py   # Optimized variants
```

---

## 📂 Repository Structure

```
3dgs-renderer-benchmark/
├── src/
│   ├── run_benchmark.py              # CLI entry point
│   ├── benchmark_framework/          # Core library (PLY, cameras, metrics)
│   │   ├── scene.py                  # PLY loading + covariance computation
│   │   ├── cameras.py                # Camera generation + loading
│   │   ├── metrics.py                # Frame timing + aggregation
│   │   └── results.py                # Export (JSON, Markdown)
│   ├── renderers/                    # 5 adapter implementations
│   │   ├── base.py                   # Abstract RendererAdapter
│   │   ├── diff_gaussian_renderer.py # Baseline (ashawkey fork)
│   │   ├── gsplat_renderer.py        # gsplat wrapper mode
│   │   ├── speedy_splat_renderer.py  # Winner (CUB sort)
│   │   └── tcgs_renderer.py          # TC-GS adapter
│   └── scripts/
│       ├── benchmark_phase1.py       # Phase 1 runner
│       ├── benchmark_phase2.py       # Phase 2 optimizer runner
│       ├── generate_scene.py         # Synthetic .ply generator
│       ├── gen_cameras.py            # Camera pose generator
│       └── gen_report.py             # HTML report generator
├── results/                          # Benchmark data + reports
├── data/                             # Scene.ply + cameras.json
├── config/                           # Benchmark configuration
├── docs/                             # GitHub Pages website
│   └── index.html                    # Benchmark report website
└── README.md
```

---

## 📊 Phase 1: Renderer Comparison

| Rank | Renderer | Median (ms) | FPS | P99 (ms) | Memory | Key Technology |
|:----:|----------|:-----------:|:---:|:--------:|:------:|----------------|
| 🥇 | **speedy_splat** | **7.31** | **136.8** | 1,300.7 | 1,927 MB | **CUB DeviceRadixSort** |
| 🥈 | diff_gaussian | 7.42 | 134.8 | 1,427.6 | 1,998 MB | Thrust radix sort |
| 🥈 | TC-GS | 7.42 | 134.7 | 1,427.4 | 1,998 MB | Same kernel |
| 🥉 | gsplat (wrapper) | 7.47 | 133.8 | 1,445.1 | 1,998 MB | Python overhead |

> **TC-GS note**: Uses identical `diff_gaussian_rasterization` CUDA kernel at render-time; training-only innovations (triplane encoding).

---

## ⚡ Phase 2: Optimizations (+113% over baseline)

| Optimization | Median (ms) | FPS | Δ vs Baseline |
|-------------|:-----------:|:---:|:-------------:|
| Baseline (speedy_splat) | 5.83 | 171.4 | — |
| **+Frustum Pre-Culling** | **2.84** | **352.3** | **+105.5%** |
| **+Culling + Prealloc Buffers** | **2.74** | **365.1** | **+113.0%** |

### Techniques Applied

1. **Frustum Pre-Culling** — Conservative NDC projection test removes behind-camera gaussians (keeps all visible ones). Reduces kernel workload ~50%.
2. **Pre-allocated Buffer Reuse** — Eliminates per-frame `torch.zeros`/`torch.ones` allocations.
3. **Rasterizer Cache** — Reuses `GaussianRasterizer` across frames per camera.

---

## 🔬 Speedup Analysis

| Component | diff_gaussian | speedy_splat | Gain |
|-----------|:------------:|:------------:|:----:|
| Sort Algorithm | Thrust `radix_sort` | CUB `DeviceRadixSort` | **15–30%** |
| Shared Memory | Standard load | Warp-level coalesced | **10–20%** |
| **Overall** | 134.8 FPS | **136.8 FPS** | **+1.5%** |

CUB DeviceRadixSort uses warp-shuffle primitives, avoids intermediate global memory passes, and achieves higher occupancy during tile binning.

---

## 🖥️ Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 5070 Laptop (8.55 GB, CC 12.0) |
| CUDA | Driver 13.1 / Toolkit 13.3 |
| PyTorch | 2.12.1+cu130 |
| System | Windows 11 24H2 + MSVC 14.44 |

---

## 📄 License

Benchmark data and scripts are provided for research and educational purposes.
