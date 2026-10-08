# REPRODUCIBILITY — Environment

**Generated:** 2026-10-02  |  Inventory only; no environment mutated.

## Local (this workstation — analysis/build only, NOT the GPU authority)
- Python 3.13.13
- PyTorch 2.13.0+cu130 (CUDA 13.0 runtime)
- nvcc 13.3 (V13.3.73)
- GPU: NVIDIA GeForce RTX 5070 Laptop (local; used for source audit/assembly only)

## Remote GPU authority — EPIC-05 A100 cluster (all paper GPU results)
- Host: 8.130.30.251:1024
- GPU: 8x NVIDIA A100-SXM4-80GB (CUDA_VISIBLE_DEVICES selects device)
- CUDA: 12.8
- Python: 3.12.13 (/root/miniforge3/envs/gsplat/bin/python)
- PyTorch: 2.9.1+cu128 (TFC/DFC) or 2.10.0+cu128 (DFC1-R1/R2, PUB0 current cohort)
- gsplat: 1.5.3 fork `gsplat-b1-clean-v153`
- numpy 2.5.0
- OS: Linux (dsw container)

## K50-B1 historical cohort (DIFFERENT — do not merge)
- A100-PCIE-40GB (8 GPUs SM80, 108 SMs); gsplat 1.5.3; CUDA 11.8 build
- C51_STAGE4A: A100-PCIE-40GB (config-buggy 5K). C51_STAGE4B: A100-80GB (clean 30K).

## VOS cohort (different software stack)
- NVIDIA A100-SXM4-80GB (single device); torch 2.8.0+cu128; graphdeco-b0-pub; sparse_adam.

## PFC0 static-analysis source
- `_mx_h8/pack/h8/gsplat/cuda/csrc` (ProjectionEWA3DGSFused.cu, IntersectTile.cu); scene room_iter10000, 1920x1080, tile_size=32; RTX 5070 Laptop capture for per-kernel ratios.

## Environment-variable / config notes
- `CUDA_VISIBLE_DEVICES` selects the A100 device per run.
- EPIC-05 remote-only: no local CUDA for paper results.
- DFC_FC0 / DFC2 require an ISOLATED EPIC-05 node (no PBR0/campaign sharing).
