# Machine report

## Hardware

| Item | Value |
| --- | --- |
| CPU | Intel Core Ultra 9 275HX, 24 cores / 24 logical processors |
| RAM | 33,752,997,888 bytes (31.44 GiB) |
| NVIDIA GPU | NVIDIA GeForce RTX 5070 Laptop GPU, Blackwell, compute capability 12.0 |
| NVIDIA UUID | `GPU-ddda4ab2-d9b9-0efc-a5f9-f438ff9214b9` |
| NVIDIA VRAM | 8,151 MiB reported by NVML |
| Integrated GPU | Intel(R) Graphics, driver 32.0.101.8331 |

Only CUDA device 0, the NVIDIA GPU, was selected with `CUDA_VISIBLE_DEVICES=0`. No benchmark process selected the Intel GPU.

## Software

| Item | Value |
| --- | --- |
| OS | Windows 11 Home Chinese, 64-bit, 10.0.26200 build 26200 |
| NVIDIA driver | 592.01 |
| Driver model | WDDM |
| CUDA reported by driver | 13.1 |
| CUDA toolkit used for builds | 13.3 (`nvcc 13.3.73`) |
| PyTorch CUDA runtime | 13.0 |
| Benchmark Python | 3.10.20 |
| PyTorch | 2.12.1+cu130 |
| Benchmark base commit | `b46e8f27fbc3beea89a12f25c35ce8b296f24cd9` |
| Protocol SHA-256 | `892e18890501c408dc6746af69f17e16973604f0c02a3caddf1954d3bf1fede2` |

The benchmark changes add raw NVML sample retention, render-output export, complete failure evidence, external canonical camera handling, and adapter output/API compatibility. They do not modify `benchmark/protocol.json`, `benchmark/suite.json`, ranking rules, workloads, camera data, resolution, or metric definitions.

Raw environment export: `generated/machine_environment.json`.
