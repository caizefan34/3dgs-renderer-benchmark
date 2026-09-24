# A100 Environment Fingerprint

**Date:** 2026-09-04  
**Host:** mx (bms-39468022-001)  
**User:** liaoyuanjun

---

## 1. Hardware Profile

| Property | Value |
|----------|-------|
| **GPU Model** | **NVIDIA A100-PCIE-40GB** (×8) |
| **Form Factor** | PCIe 4.0 (vs EPIC-05 SXM) |
| **Compute Capability** | 8.0 |
| **VRAM** | 40,960 MiB per GPU (vs EPIC-05 81,920 MiB) |
| **Memory Clock** | 1215 MHz (vs EPIC-05 1593 MHz SXM) |
| **Graphics Clock** | 1410 MHz |
| **Power Limit** | 250 W (vs EPIC-05 400 W SXM) |
| **Current Power Draw** | ~130-195 W (GPU 0-4 under vLLM load) |
| **CPU** | Unknown (EPIC-05: 2× Xeon Platinum 8369B) |
| **RAM** | 378 GB (/dev/shm) |
| **OS** | Ubuntu 22.04.4 LTS |
| **Uptime** | 78 days |

## 2. Software Profile

| Property | Value | vs EPIC-05 |
|----------|-------|:----------:|
| **NVIDIA Driver** | 595.71.05 | **DIFFERENT** (580.105.08) |
| **CUDA Driver** | 13.2 | **DIFFERENT** (12.x) |
| **NVCC (conda)** | 12.4.131 (anysplat env) | **DIFFERENT** (12.9) |
| **NVCC (system)** | 11.5.119 (broken path) | N/A |
| **Python** | 3.10.12 (system) / conda envs | **DIFFERENT** (3.10.20) |
| **PyTorch** | **2.4.1+cu124** (anysplat env) | **DIFFERENT** (2.9.1+cu128) |
| **gsplat** | **1.5.3+pt24cu124** | **SAME VERSION, DIFFERENT BUILD** |
| **GCC** | 11.4.0 (Ubuntu) | DIFFERENT |
| **Conda** | miniforge3 | NOT present on EPIC-05 |
| **CUDA Toolkit** | 12.4 (conda nvcc) | DIFFERENT |

## 3. Available Conda Environments

| Environment | PyTorch | CUDA | gsplat | Notes |
|:-----------:|:-------:|:----:|:------:|:------|
| **anysplat** | **2.4.1+cu124** | **12.4** | **1.5.3** | **Primary target** |
| behavior | 2.6.0+cu124 | 12.4 | — | Available |
| vllm | (active, serving) | — | — | Two vLLM instances running |
| MVimgNet | unknown | — | — | — |
| vomp | unknown | — | — | — |
| base | (no torch) | — | — | — |

## 4. GPU Occupancy (During Experiments)

During the Phase A100 validation experiments (2026-09-04), all vLLM instances were stopped to free GPUs:

| GPU | Memory Used (at experiment start) | Memory Used (peak) | Process |
|:---:|:----------:|:----------:|:--------|
| 0 | 0 MiB | 2,801 MiB | 3DGS training (room t16 30K) |
| 1 | 0 MiB | 2,709 MiB | 3DGS training (room t20 30K) |
| 2 | 0 MiB | 2,755 MiB | 3DGS training (room t24 30K) |
| 3 | 0 MiB | 4,695 MiB | 3DGS training (bicycle t24 30K) |
| 4 | 0 MiB | 4,679 MiB | 3DGS training (bicycle t16 30K) |
| 5 | 0 MiB | 4,581 MiB | 3DGS training (bicycle t20 30K) |
| 6 | 0 MiB | 2,363 MiB | 3DGS training (garden t16 30K) |
| 7 | 0 MiB | 2,097 MiB | 3DGS training (garden t20 30K) |

> **Note**: vLLM was stopped for the duration of experiments. GPU memory usage during 30K training peaked at ~5.8 GB (bicycle scenes with 2.5M+ Gaussians), well within the 40 GB VRAM capacity. Room and garden required < 2.3 GB peak.

## 5. Storage

| Mount | Size | Used | Avail | Use% |
|:-----|:----:|:----:|:-----:|:---:|
| `/` (root) | 439 GB | 380 GB | 42 GB | **91%** |
| `/mnt/storage_pool` | 3.5 TB | 2.6 TB | **915 GB** | 75% |

## 6. Performance Characteristics (from inference benchmarks)

Previous A100 inference benchmark data (2026-07) from garden scene:
- **EPIC-05 A100-SXM4-80GB**: gsplat baseline 492 FPS (2.03ms)
- **PCIe 40GB** expected to be **slower** due to:
  - Lower memory bandwidth (PCIe 4.0 x16 ~64 GB/s vs SXM ~2 TB/s)
  - Lower power limit (250W vs 400W) → lower sustained clocks
  - Same compute capability (8.0) so kernel caps are identical

## 7. Environment Compatibility Assessment

| Dimension | EPIC-05 (old A100) | New A100 (mx) | Verdict |
|:----------|:------------------:|:-------------:|:-------:|
| GPU Architecture | A100-SXM4-80GB | A100-PCIE-40GB | **DIFFERENT** (PCIe vs SXM) |
| VRAM | 80,000 MiB | 40,000 MiB | **DIFFERENT** |
| DRAM Bandwidth | ~2 TB/s (HBM2e SXM) | ~1.6 TB/s (HBM2e PCIe) | **DIFFERENT** |
| TDP/Power Limit | 400 W | 250 W | **DIFFERENT** |
| NVIDIA Driver | 580.105.08 | 595.71.05 | **DIFFERENT** |
| CUDA Toolkit | 12.9 | 12.4 (conda) | **DIFFERENT** |
| PyTorch | 2.9.1+cu128 | 2.4.1+cu124 | **DIFFERENT** |
| gsplat | 1.5.3 (pt128 build) | 1.5.3 (pt24cu124 build) | **DIFFERENT BUILD** |
| Python | 3.10.20 | 3.10.12 | DIFFERENT (minor) |
| OS | Ubuntu 22.04.5 al8 kernel | Ubuntu 22.04.4 | DIFFERENT (minor) |

**Overall Verdict: ENVIRONMENTS ARE DIFFERENT**

Direct cross-hardware speed comparison between old and new A100 is **NOT VALID** without environment normalization. Results on new A100 should be treated as:
- **Single-machine validation** (for evidence chain completeness)
- **Not directly comparable** to old A100 or RTX 5070 for raw speed
- **Comparable for binary outcomes** (training stability, NaN/Inf, PSNR quality)
