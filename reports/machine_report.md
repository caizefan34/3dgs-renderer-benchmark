# Machine report

## Hardware

| Item | Value |
| --- | --- |
| Host | EPIC-05 |
| CPU | 2 x Intel Xeon Platinum 8369B at 2.90 GHz |
| CPU topology | 64 physical cores / 128 logical processors |
| RAM | 2,163,293,777,920 bytes (about 1.97 TiB) |
| NVIDIA GPUs | 8 x NVIDIA A100-SXM4-80GB, compute capability 8.0 |
| Selected GPU | Physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`) |
| Selected GPU UUID | `GPU-12b6b703-727b-0c6e-1433-4e6161c54938` |
| Selected GPU VRAM | 81,920 MiB reported by `nvidia-smi`; 81,152.8 MiB reported by PyTorch |
| Power limit | 400 W |

All published metrics identify the selected physical GPU UUID. The earlier
GPU-metadata attempt was isolated and is not part of the published matrix.

## Software

| Item | Value |
| --- | --- |
| Userland OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | `5.10.134-16.3.al8.x86_64` |
| NVIDIA driver | 580.105.08 |
| CUDA toolkit used for builds | 12.9 (`nvcc 12.9.86`) |
| PyTorch CUDA runtime | 12.8 |
| Benchmark Python | 3.10.20 |
| PyTorch | 2.9.1+cu128 |
| Benchmark evidence commit | `dc9bb4e9231ae2fdf90fa9c40bcd6e0dbd7d104f` |
| Protocol SHA-256 | `892e18890501c408dc6746af69f17e16973604f0c02a3caddf1954d3bf1fede2` |

## Published matrix

- 25 complete Tier A results: five renderers by five canonical cases.
- 30 warmup frames and 5 x 100 measured frames per speed run.
- 100 ordered GT views per quality run with PSNR, SSIM, and LPIPS-vgg.
- Strict positive NVML process-memory samples for every result.
- One hardware/software cohort and no rejected report inputs.

The generated Tier A overall table contains all five renderers. Its aggregate
process peak range is 4,206-8,234 MiB; raw per-run NVML evidence remains adjacent
to each published `metrics.json`.
