# 3D Gaussian Splatting Renderer Benchmark

[![Website](https://img.shields.io/badge/Website-View%20Report-7c5cfc)](https://caizefan34.github.io/3dgs-renderer-benchmark/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205070%20Laptop-76b900)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c)](https://pytorch.org)

> **A Reproducible Benchmark Suite for 3D Gaussian Splatting Renderers**

Rigorous comparison of **4 CUDA rasterization renderers** for 3D Gaussian Splatting, plus **Phase 2 engineering optimizations** that push the winner to **365 FPS** 鈥?a **+113%** speedup over baseline.

**GPU**: NVIDIA GeForce RTX 5070 Laptop 路 **Scene**: 400K Gaussians, SH deg 3 路 **Resolution**: 1920脳1080

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

# Run unified benchmark (default: warmup=100, measured=500, repeats=5)
python src/run_benchmark.py --seed 42

# Or use specific renderers and camera paths
python src/run_benchmark.py --renderers speedy_splat diff_gaussian \
    --camera-path spiral --frames 500 --warmup 100 --repeats 5 \
    --seed 42 --output results/

# Optional runtime knobs (default-safe: disabled)
python src/run_benchmark.py --mixed-precision --amp-dtype float16
python src/run_benchmark.py --compile --compile-mode reduce-overhead

# Parameter sweep (writes results/raw/*.json and results/summary.csv)
python scripts/run_sweep.py --renderers speedy_splat diff_gaussian \
    --gs-counts 100000,400000 --resolutions 1280x720,1920x1080 --seed 42
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

### FPS Measurement Methodology

All FPS results are collected under a **strictly controlled protocol** to ensure reproducibility:

| Parameter | Value |
|-----------|-------|
| Warmup frames | 100 |
| Measured frames | 500 |
| Repeats | 5 |
| GPU Clock Lock | Yes |
| CUDA Synchronization | Measurement boundary only (`cuda.Event`) |
| Timing Method | `torch.cuda.Event.elapsed_time()` |
| Seed | CLI `--seed` (Python / NumPy / Torch / CUDA) |
| Reported Metrics | Mean ± SEM (Standard Error of Mean) |

Results are reported as **mean ± standard error of the mean (SEM)** across all repeats. The median and P1/P5/P95/P99 percentiles are also provided for tail-latency analysis.

### Standard Camera Paths

```text
data/camera_presets/
鈹溾攢鈹€ spiral.json       鈥?60 cameras, spiral orbit with radius oscillation
鈹溾攢鈹€ circle.json       鈥?50 cameras, circular orbit at fixed radius
鈹溾攢鈹€ flythrough.json   鈥?30 cameras, linear front-to-back flight
鈹斺攢鈹€ random_walk.json  鈥?30 cameras, orbit with random perturbations
```

All renderers use the **same camera path** for fair FPS comparison. Select with `--camera-path`:

```bash
python src/run_benchmark.py --camera-path spiral
```

### Benchmark Results

#### Multi-Scene Comparison (Real-World Datasets)

| Dataset | Scene | Gaussians | Renderer | Mean FPS | Median (ms) | VRAM (MB) | PSNR |
|---------|-------|:---------:|:--------:|:--------:|:-----------:|:---------:|:----:|
| Synthetic | 400K | 400K | speedy_splat | 136.8 | 7.31 | 1,927 | inf |
| Synthetic | 400K | 400K | diff_gaussian | 134.8 | 7.42 | 1,998 | - |
| Synthetic | 400K | 400K | fast_gauss | 134.7 | 7.42 | 1,998 | - |
| Synthetic | 400K | 400K | gsplat | 133.8 | 7.47 | 1,998 | - |
| Mip-NeRF360 | garden | 500K | speedy_splat | TBD | TBD | TBD | TBD |
| Mip-NeRF360 | bicycle | 1M | speedy_splat | TBD | TBD | TBD | TBD |
| Tanks & Temples | drjohnson | 1.2M | speedy_splat | TBD | TBD | TBD | TBD |
| Tanks & Temples | train | 1.5M | speedy_splat | TBD | TBD | TBD | TBD |

> **Note:** Real-world dataset results marked as TBD are pending GPU time. Run `python src/scripts/download_datasets.py --dataset <name>` to download and `python src/run_benchmark.py` to benchmark.

#### Multi-Scale Performance (Gaussian Count Scaling)

| Gaussians | speedy_splat | diff_gaussian | fast_gauss | gsplat |
|:---------:|:-----------:|:-------------:|:----------:|:------:|
| 50K | TBD | TBD | TBD | TBD |
| 200K | TBD | TBD | TBD | TBD |
| 500K | TBD | TBD | TBD | TBD |
| 1M | TBD | TBD | TBD | TBD |
| 2M | TBD | TBD | TBD | TBD |

> **Note:** Multi-scale results are generated by `python src/scripts/generate_scene.py --gaussians <N>`.

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
鈹溾攢鈹€ raw/*.json               鈥?Per-repeat raw artifacts (one file per renderer run)
鈹溾攢鈹€ summary.csv              鈥?Aggregate comparison table (mean/median/std FPS + p95 latency)
鈹溾攢鈹€ benchmark_results.json   鈥?Full benchmark summary with frame-time series
鈹溾攢鈹€ benchmark_results.csv    鈥?Legacy summary metrics table
鈹溾攢鈹€ benchmark_report.md      鈥?Markdown report with per-renderer details
鈹斺攢鈹€ benchmark_report.html    鈥?Interactive Plotly dashboard with frame-time chart
```

`summary.csv` fields: `renderer`, `warmup_iters`, `measured_iters`, `repeats`, `seed`, `mean_fps`, `median_fps`, `std_fps`, `p95_latency_ms`, `std_mean_fps_across_runs`, `peak_vram_mb`.

Fair comparison guidance: keep renderer set, scene, camera path, resolution, warmup/measured/repeats, and seed identical across runs; compare speed (FPS/latency) together with quality metrics (PSNR/SSIM/LPIPS) when changing runtime knobs (mixed precision / compile).

---

## Trophy: speedy_splat (CUB DeviceRadixSort)

| Phase | Config | Median FPS | vs Baseline |
|-------|--------|:---------:|:------------:|
| Phase 1 | **speedy_splat** | **136.8** | 鈥?|
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
| Baseline (speedy_splat) | 5.83 | 171.4 | 鈥?|
| **+Frustum Pre-Culling** | **2.84** | **352.3** | **+105.5%** |
| **+Culling + Prealloc Buffers** | **2.74** | **365.1** | **+113.0%** |

### Techniques Applied

1. **Frustum Pre-Culling** 鈥?Conservative NDC projection test removes behind-camera gaussians before kernel launch. Reduces kernel workload ~50%.
2. **Pre-allocated Buffer Reuse** 鈥?Eliminates per-frame `torch.zeros`/`torch.ones` allocations.
3. **Rasterizer Cache** 鈥?Reuses `GaussianRasterizer` across frames per camera.

### GPU Profiling Insights

The speedup of `speedy_splat` comes from replacing **Thrust radix sort** with **CUB DeviceRadixSort** in the tile binning pipeline:

| Operation | diff_gaussian (Thrust) | speedy_splat (CUB) | Improvement |
|-----------|:---------------------:|:------------------:|:-----------:|
| Sort Time (per frame) | ~0.8 ms | ~0.3 ms | **-62%** |
| Raster Time | ~6.6 ms | ~7.0 ms | +6% |
| Memory Copy | ~0.1 ms | ~0.1 ms | - |
| Kernel Occupancy | 45% | 48% | +3 pp |
| SM Utilization | 38% | 42% | +4 pp |

> **Note:** Profile data collected via Nsight Systems on RTX 5070 Laptop, 400K Gaussians, 1920x1080. Results may vary by GPU architecture.

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
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ run_benchmark.py              # Unified benchmark CLI
鈹?  鈹溾攢鈹€ benchmark_framework/          # Core library
鈹?  鈹?  鈹溾攢鈹€ scene.py                  # PLY loading (vectorized) + covariance
鈹?  鈹?  鈹溾攢鈹€ cameras.py                # Camera generation + loading
鈹?  鈹?  鈹溾攢鈹€ metrics.py                # Comprehensive metrics (P1/P5/P99, VRAM, jitter)
鈹?  鈹?  鈹溾攢鈹€ results.py                # Export: JSON, CSV, Markdown, HTML+Plotly
鈹?  鈹?  鈹斺攢鈹€ config.py                 # Benchmark configuration
鈹?  鈹溾攢鈹€ renderers/                    # Renderer adapters (4)
鈹?  鈹斺攢鈹€ scripts/                      # Benchmark scripts + tools
鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ camera_presets/               # Standard camera paths
鈹?  鈹斺攢鈹€ scenes/
鈹?      鈹斺攢鈹€ scenes.json               # Standard dataset manifest
鈹溾攢鈹€ docs/index.html                   # GitHub Pages report
鈹溾攢鈹€ .github/workflows/
鈹?  鈹溾攢鈹€ deploy-pages.yml              # GitHub Pages deployment
鈹?  鈹斺攢鈹€ benchmark-regression.yml      # CI regression testing
鈹斺攢鈹€ README.md
```

---

## Leaderboard

*Submit your renderer results via PR to add your entry. Include GPU model, driver version, and benchmark command used.*

| Renderer | FPS | Latency (ms) | VRAM (MB) | PSNR | Scene | GPU |
|----------|:---:|:------------:|:---------:|:----:|:-----:|:---:|
| **speedy_splat (opt)** | **365.1** | **2.74** | 1,927 | inf | 400K synth | RTX 5070 Laptop |
| **speedy_splat** | 136.8 | 7.31 | 1,927 | inf | 400K synth | RTX 5070 Laptop |
| diff_gaussian | 134.8 | 7.42 | 1,998 | 鈥?| 400K synth | RTX 5070 Laptop |
| fast_gauss | 134.7 | 7.42 | 1,998 | 鈥?| 400K synth | RTX 5070 Laptop |
| gsplat (wrapper) | 133.8 | 7.47 | 1,998 | 鈥?| 400K synth | RTX 5070 Laptop |

---

## CI Regression Testing

Every push to `main` automatically runs benchmarks across **three scene tiers** and archives results:

| Tier | Gaussians | Frames | Renderers | Regression Threshold |
|:----:|:---------:|:------:|:---------:|:-------------------:|
| Small | 50K | 100 | all 4 | >5% FPS drop |
| Medium | 200K | 100 | all 4 | >5% FPS drop |
| Large | 500K | 100 | all 4 | >5% FPS drop |

Results are uploaded as CI artifacts. A **regression alert** is triggered if any renderer's FPS drops >5% relative to the stored baseline.

```yaml
# .github/workflows/benchmark-regression.yml
strategy:
  matrix:
    scene-tier: [small, medium, large]
```

To set a new baseline, delete the cached baseline files in `results/ci/baselines/` and re-run the workflow.

---

## License

MIT License. Benchmark data and scripts are provided for research and educational purposes.
