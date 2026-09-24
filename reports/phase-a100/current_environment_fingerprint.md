# Current A100 Environment Fingerprint

**Date:** 2026-09-06  
**Host:** bms-39468022-001 (mx)  
**User:** liaoyuanjun

---

## 1. Hardware Profile

| Property | Value |
|----------|-------|
| **GPU Model** | **NVIDIA A100-PCIE-40GB** (×8) |
| **Form Factor** | PCIe 4.0 |
| **Compute Capability** | 8.0 |
| **VRAM** | 40,441 MiB per GPU (~40 GB) |
| **Memory Clock** | 1215 MHz |
| **Graphics Clock (Base/Boost)** | 765 MHz / 1410 MHz |
| **Power Limit** | 250 W |
| **Current Power Draw (idle)** | ~32–46 W per GPU |
| **CPU** | Unknown (Ubuntu 22.04) |
| **RAM** | 378 GB (/dev/shm) |
| **OS** | Ubuntu 22.04.4 LTS (kernel 5.15.0-181-generic) |

## 2. Software Profile

| Property | Value |
|----------|-------|
| **NVIDIA Driver** | 595.71.05 |
| **CUDA Driver** | 13.2 |
| **CUDA Runtime (conda)** | 11.8 |
| **NVCC** | Cuda compilation tools, release 12.4, V12.4.131 |
| **GCC** | gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| **Python** | 3.10.19 |
| **PyTorch** | **2.7.1+cu118** |
| **Build CUDA for PyTorch** | 11.8 |
| **cuDNN** | 90100 |
| **gsplat** | **1.5.3** (pip wheel, JIT-compiled backend) |
| **gsplat location** | `/home/liaoyuanjun/.local/lib/python3.10/site-packages` |
| **CUB backend** | Bundled with gsplat (`gsplat/cuda/csrc/`) — `cub::DeviceRadixSort::SortPairs` |
| **Conda env** | `anysplat` |

## 3. GPU Occupancy (During Profiling)

GPU 0 was used for profiling. All other GPUs were idle.

| GPU | Memory Used (peak) | Notes |
|:---:|:------------------:|:------|
| 0 | Up to 23,803 MiB (garden t16) | Baseline profiling |
| 1–7 | 0 MiB | Idle |

## 4. Environment Notes

- gsplat 1.5.3 had a minor compatibility issue with PyTorch 2.7.1+cu118:
  duplicate `with_sycl=None` argument in `_backend.py` — fixed by removing line 91.
- CUDA runtime is 11.8 (from conda PyTorch) but system NVCC is 12.4.131.
- gsplat CUDA extension was JIT-compiled on first use.
