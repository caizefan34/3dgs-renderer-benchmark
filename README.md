# 3D Gaussian Splatting Renderer Benchmark

Comprehensive benchmark comparing 5 candidate CUDA rasterization renderers for 3D Gaussian Splatting, tested on **NVIDIA GeForce RTX 5070 Laptop GPU** with **CUDA 13.3 + MSVC 14.44**.

## Tested Renderers

| Renderer | Repo | Stars | Status | Speed |
|---|---|---|---|---|
| [speedy-splat](https://github.com/j-alex-hanson/speedy-splat) | j-alex-hanson/speedy-splat | 347 | ✅ Tested | **224 FPS** 🥇 |
| [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization) | graphdeco-inria (ashawkey fork) | 487 | ✅ Tested | 205 FPS |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | nerfstudio-project/gsplat | 5,363 | ⚠️ Wrapper mode | 178 FPS |
| [TC-GS](https://github.com/timwang2001/TC-GS) | timwang2001/TC-GS | 75 | ❌ Not installed | Same kernel as speedy-splat |
| [fast-gaussian-rasterization](https://github.com/dendenxu/fast-gaussian-rasterization) | dendenxu/fast-gaussian-rasterization | 1,186 | ❌ Linux only | Needs EGL/GL |

## 🏆 Winner: **speedy-splat** (CUB DeviceRadixSort)

- **224 FPS** median at 1920×1080 with 400K Gaussians (SH degree 3)
- **9.4% faster** than baseline (diff-gaussian-rasterization with Thrust sort)
- Core reason: **CUB DeviceRadixSort** replaces Thrust's radix sort - CUB uses warp-level primitives with more efficient shared memory patterns

## Hardware

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop (8.55 GB, Compute 12.0) |
| CUDA | Driver 13.1 / Toolkit 13.3 / PyTorch 13.0 |
| PyTorch | 2.12.1+cu130 |
| System | Windows 11 + MSVC 14.44 |

## Benchmark Scene

| Property | Value |
|---|---|
| File | data/scene.ply |
| Gaussians | 400,000 |
| SH Degree | 3 (48 coefficients each) |
| Resolution | 1920 × 1080 |
| Camera Poses | 50 fixed orbit views |

## Quick Start

`ash
# Clone and enter
git clone https://github.com/caizefan34/3dgs-renderer-benchmark
cd 3dgs-renderer-benchmark

# Activate environment (requires CUDA toolkit)
conda create -n gsplat python=3.10
conda activate gsplat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install renderers
pip install diff-gaussian-rasterization@https://github.com/ashawkey/diff-gaussian-rasterization
pip install git+https://github.com/j-alex-hanson/speedy-splat

# Generate scene
python work/scripts/generate_scene.py 400000

# Run benchmark
python work/run_full_benchmark.py
`

## Results

### Render-Only Performance (median of 50 frames)

| Renderer | Latency (ms) | FPS | vs Baseline |
|---|---|---|---|
| **speedy_splat** | **4.47** | **223.9** | — (FASTEST) |
| diff_gaussian | 4.89 | 204.6 | +9.4% slower |
| gsplat (wrapper) | 5.61 | 178.4 | +25.5% slower |

See outputs/benchmark_report.html for interactive charts, or outputs/benchmark_results.md for full per-frame logs.

## Optimization Roadmap (Phase 2)

1. CUB DeviceRadixSort — 15-30% ✅ (already in speedy-splat)
2. Shared Memory Bank Conflict — 10-20%
3. FP16 Parameter Loading — 20-40%
4. CUDA Graph — 5-15%
5. Screen-Space Culling — 10-25%
6. Async Double-Buffer Pipeline — 10-20%

## Repository Structure

`
├── outputs/
│   ├── benchmark_report.html    # Interactive HTML report
│   ├── benchmark_results.json   # Structured benchmark data
│   ├── benchmark_results.md     # Full test log
│   └── README.md                # Summary
├── work/
│   ├── benchmark_framework/     # Benchmark library (PLY loading, metrics)
│   ├── renderers/               # 5 renderer adapters (unified interface)
│   ├── scripts/                 # Scene generator, report generator
│   ├── config/                  # Benchmark configuration
│   ├── run_benchmark.py         # Main entry point
│   └── run_full_benchmark.py    # Full 50-frame benchmark
├── data/                        # Scene data (see scene generation)
└── config/                      # Benchmark configs
`

## License

This project is for research and benchmarking purposes.
