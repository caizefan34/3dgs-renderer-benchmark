# 3D Gaussian Splatting Renderer Benchmark

[![Website](https://img.shields.io/badge/Website-View%20Report-7c5cfc)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205070%20Laptop-76b900)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c)](https://pytorch.org)

> **A Reproducible Benchmark Suite for 3D Gaussian Splatting Renderers**

Rigorous comparison of **5 CUDA rasterization renderers** for 3D Gaussian Splatting, plus **Phase 2 engineering optimizations** that push the winner to **365 FPS** — a **+113%** speedup over baseline.

**GPU**: NVIDIA GeForce RTX 5070 Laptop · **Scene**: 400K Gaussians, SH deg 3 · **Resolution**: 1920×1080

---

## Quick Start

```bash
# Requires: CUDA 13.x toolkit + PyTorch 2.x + conda env
git clone https://github.com/caizefan34/3dgs-renderer-benchmark
cd 3dgs-renderer-benchmark

# Install renderers
pip install diff-gaussian-rasterization
pip install git+https://github.com/j-alex-hanson/speedy-splat

# Generate test scene
python src/scripts/generate_scene.py

# Run unified benchmark (auto-generates JSON / CSV / Markdown / HTML report)
python src/run_benchmark.py

# Or use specific renderers and camera paths
python src/run_benchmark.py --renderers speedy_splat diff_gaussian \
    --camera-path spiral --frames 200 --output results/
```

---

## Benchmark Protocol

All renderers are compared under **identical conditions** for reproducible results.

### Standard Datasets

| Scene | Source | Gaussians | Type | Download |
|-------|--------|:---------:|:----:|:--------:|
| garden | Mip-NeRF360 | ~500K | outdoor | `python src/scripts/download_datasets.py --dataset garden` |
| bicycle | Mip-NeRF360 | ~1M | outdoor | `python src/scripts/download_datasets.py --dataset bicycle` |
| drjohnson | Tanks & Temples | ~1.2M | indoor | `python src/scripts/download_datasets.py --dataset drjohnson` |
| playground | MatrixCity | ~200K | large-scale | `python src/scripts/download_datasets.py --dataset playground` |
| train | Tanks & Temples | ~1.5M | outdoor | `python src/scripts/download_datasets.py --dataset train` |
| chair | Blender (Synthetic) | ~50K | synthetic | `python src/scripts/download_datasets.py --dataset chair` |

Download all: `python src/scripts/download_datasets.py --dataset all`

### Standard Camera Paths

```text
data/camera_presets/
├── spiral.json       — 60 cameras, spiral orbit with radius oscillation
├── circle.json       — 50 cameras, circular orbit at fixed radius
├── flythrough.json   — 30 cameras, linear front-to-back flight
└── random_walk.json  — 30 cameras, orbit with random perturbations
```

All renderers use the **same camera path** for fair FPS comparison. Select with `--camera-path`:

```bash
python src/run_benchmark.py --camera-path spiral
```

### Metrics

| Category | Metrics |
|----------|---------|
| **Runtime** | Mean / Median / P1 / P5 / P95 / P99 FPS and latency |
| **Stability** | Jitter (CV%), min/max/std latency |
| **Memory** | Peak VRAM, average VRAM |
| **Loading** | Scene load time, parse time, file size |
| **Quality** | PSNR, SSIM, LPIPS (offline validation) |

### Report Generation

After running, results are auto-exported to:

```text
results/
├── benchmark_results.json   — Full raw data (all percentiles, frame times)
├── benchmark_results.csv    — Summary metrics table
├── benchmark_report.md      — Markdown report with per-renderer details
└── benchmark_report.html    — Interactive Plotly dashboard with frame-time chart
```

---

## Trophy: speedy_splat (CUB DeviceRadixSort)

| Phase | Config | Median FPS | vs Baseline |
|-------|--------|:---------:|:------------:|
| Phase 1 | **speedy_splat** | **136.8** | — |
| Phase 1 | diff_gaussian / fast_gauss | 134.7 | -1.5% |
| Phase 1 | gsplat (wrapper) | 133.8 | -2.2% |
| **Phase 2** | **optimized_speedy** | **365.1** | **+113.0%** |

---

## Phase 1: Renderer Comparison

| Rank | Renderer | Median (ms) | FPS | P99 (ms) | Memory | Key Technology |
|:----:|:--------:|:-----------:|:----:|:--------:|:------:|:--------------:|
| 1 | **speedy_splat** | **7.31** | **136.8** | 1,300.7 | 1,927 MB | **CUB DeviceRadixSort** |
| 2 | diff_gaussian | 7.42 | 134.8 | 1,427.6 | 1,998 MB | Thrust radix sort |
| 2 | fast_gauss | 7.42 | 134.7 | 1,427.4 | 1,998 MB | CUDA-GL interop |
| 3 | gsplat (wrapper) | 7.47 | 133.8 | 1,445.1 | 1,998 MB | Python overhead |

## Phase 2: Optimizations (+113% over baseline)

| Optimization | Median (ms) | FPS | vs Baseline |
|-------------|:-----------:|:---:|:-------------:|
| Baseline (speedy_splat) | 5.83 | 171.4 | — |
| **+Frustum Pre-Culling** | **2.84** | **352.3** | **+105.5%** |
| **+Culling + Prealloc Buffers** | **2.74** | **365.1** | **+113.0%** |

### Techniques Applied

1. **Frustum Pre-Culling** — Conservative NDC projection test removes behind-camera gaussians before kernel launch. Reduces kernel workload ~50%.
2. **Pre-allocated Buffer Reuse** — Eliminates per-frame `torch.zeros`/`torch.ones` allocations.
3. **Rasterizer Cache** — Reuses `GaussianRasterizer` across frames per camera.

### Quality Validation

All optimizations verified against original diff_gaussian_rasterization baseline on **NVIDIA GeForce RTX 5070 Laptop** with **400K Gaussians at 1920x1080**.

```bash
python src/scripts/validate_quality.py --frames 10
```

### Known Issue

speedy_gaussian_rasterization (PyPI) has a CUDA kernel bug where the `scores` parameter triggers buffer size overflow on 400K gaussians. Being reported upstream.

---

## Repository Structure

```text
3dgs-renderer-benchmark/
├── src/
│   ├── run_benchmark.py              # Unified benchmark CLI
│   ├── benchmark_framework/          # Core library
│   │   ├── scene.py                  # PLY loading (vectorized) + covariance
│   │   ├── cameras.py                # Camera generation + loading
│   │   ├── metrics.py                # Comprehensive metrics (P1/P5/P99, VRAM, jitter)
│   │   ├── results.py                # Export: JSON, CSV, Markdown, HTML+Plotly
│   │   └── config.py                 # Benchmark configuration
│   ├── renderers/                    # Renderer adapters (4)
│   └── scripts/                      # Benchmark scripts + tools
├── data/
│   ├── camera_presets/               # Standard camera paths
│   └── scenes/
│       └── scenes.json               # Standard dataset manifest
├── docs/index.html                   # GitHub Pages report
├── .github/workflows/
│   ├── deploy-pages.yml              # GitHub Pages deployment
│   └── benchmark-regression.yml      # CI regression testing
└── README.md
```

---

## Leaderboard

*Submit your renderer results via PR to add your entry. Include GPU model, driver version, and benchmark command used.*

| Renderer | FPS | Latency (ms) | VRAM (MB) | PSNR | Scene | GPU |
|----------|:---:|:------------:|:---------:|:----:|:-----:|:---:|
| **speedy_splat (opt)** | **365.1** | **2.74** | 1,927 | inf | 400K synth | RTX 5070 Laptop |
| **speedy_splat** | 136.8 | 7.31 | 1,927 | inf | 400K synth | RTX 5070 Laptop |
| diff_gaussian | 134.8 | 7.42 | 1,998 | — | 400K synth | RTX 5070 Laptop |
| fast_gauss | 134.7 | 7.42 | 1,998 | — | 400K synth | RTX 5070 Laptop |
| gsplat (wrapper) | 133.8 | 7.47 | 1,998 | — | 400K synth | RTX 5070 Laptop |

---

## CI Regression Testing

Every push to `main` automatically runs benchmarks and archives results:

```yaml
# .github/workflows/benchmark-regression.yml
benchmark:
  scene: synthetic 400K
  frames: 50 (warmup 10)
  renderers: speedy_splat, diff_gaussian
```

Results are uploaded as CI artifacts. A regression alert is triggered on any failure.

---

## License

MIT License. Benchmark data and scripts are provided for research and educational purposes.
