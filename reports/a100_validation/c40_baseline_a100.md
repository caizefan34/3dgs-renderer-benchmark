# C40 A100 Revalidation: Training Bottleneck Profiling

## Executive Summary

**D-SSIM is an even MORE dominant bottleneck on A100 than on RTX 5070.**

| Metric | RTX 5070 (exploratory) | A100 (final validation) | Change |
|--------|----------------------|------------------------|--------|
| D-SSIM % of GPU time | 40.7% | **78.6%** | +93% relative |
| D-SSIM time (ms/iter) | 35.8 | **81.4** | 2.27x |
| Total GPU time (ms/iter) | 88.1 | **103.5** | 1.18x |
| Total iteration (wall, ms) | 85.3 | **95.1** | 1.11x |
| Checkpoint PSNR | 12.03 (diverged) | **20.56 (normal)** | — |

**Answer to the key question: Yes, D-SSIM is a significant bottleneck on A100 — in fact it is the dominant bottleneck at 78.6% of GPU kernel time.**

---

## 1. Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-PCIE-40GB |
| Compute Capability | 8.0 (SM80) |
| VRAM | 42.4 GB |
| SMs | 108 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| gsplat | 1.5.3 |
| Host | mx (bms-39468022-001), Ubuntu 22.04 |

## 2. Checkpoint

| Parameter | Value |
|-----------|-------|
| File | `a100_30k_room_t16_16_iter5000.pt` |
| Iteration | 5000 |
| Gaussians | 896,969 |
| SH degree | 3 |
| PSNR | 20.56 dB |
| Scene | room (Mip-NeRF 360) |
| Resolution | 1080p (1920×1080) |
| Tile size | 16 |

**Note**: This checkpoint was trained on the same A100-PCIE-40GB. It is a normal, non-diverged model — unlike the RTX 5070 exploration which used `phase7_room_30k_16_iter5000.pt` (PSNR=12.03, diverged).

---

## 3. Method 1: CUDA Events (20 iterations)

| Stage | Mean (ms) | Std (ms) | P90 (ms) | % of Total |
|-------|-----------|----------|----------|------------|
| **Forward** (render) | 3.38 | 0.02 | 3.39 | **3.6%** |
| **Loss** (L1 + D-SSIM) | 75.17 | 1.12 | 76.58 | **79.4%** |
| **Backward** | 11.70 | 0.03 | 11.75 | **12.4%** |
| **Optimizer** (Adam) | 3.63 | 0.00 | 3.64 | **3.8%** |
| **Total** | 94.70 | 1.13 | 96.11 | 100% |

**The loss computation (dominated by D-SSIM) accounts for 79.4% of total iteration time.**

The CUDA events measure wall time around each Python operation. The "loss" stage includes both L1 (cheap) and D-SSIM (expensive), so D-SSIM alone accounts for nearly all of the 75.17 ms.

## 4. Method 2: Block Timing (150 iterations, wall clock)

| Metric | Value |
|--------|-------|
| Mean T_iter | 95.11 ms |
| Std | 1.56 ms |
| Min | 90.99 ms |
| Max | 100.50 ms |

Block timing (no CUDA events in loop) confirms Method 1: ~95 ms/iter, consistent with CUDA event total of 94.70 ms. No hidden synchronization overhead — the `.item()` sync issue from C32-A is not present here because this script doesn't call `.item()` in the hot loop.

## 5. Method 3: torch.profiler Kernel Breakdown (10 iterations)

| Category | Time/iter (ms) | % of Total | Kernels/iter |
|----------|---------------|------------|--------------|
| **loss_dssim** | **81.40** | **78.6%** | **63** |
| adam | 7.88 | 7.6% | 56 |
| rasterize_bwd | 5.54 | 5.3% | 1 |
| elementwise_misc | 3.87 | 3.7% | 134 |
| rasterize_fwd | 1.79 | 1.7% | 1 |
| memory | 1.04 | 1.0% | 90 |
| other | 0.93 | 0.9% | 21 |
| sorting | 0.73 | 0.7% | 23 |
| reduction | 0.16 | 0.2% | 7 |
| projection_bwd | 0.13 | 0.1% | 1 |
| projection_fwd | 0.05 | 0.0% | 1 |
| **Total** | **103.51** | **100%** | **399** |

**D-SSIM consumes 81.40 ms/iter — 78.6% of total GPU kernel time.** This is 2.27x the RTX 5070 measurement (35.83 ms) in absolute terms, and nearly double the percentage (78.6% vs 40.7%).

## 6. RTX 5070 vs A100 Comparison

### Why D-SSIM is MORE dominant on A100

| Factor | RTX 5070 (SM120) | A100 (SM80) | Effect |
|--------|-----------------|-------------|--------|
| D-SSIM (cuDNN conv) | 35.8 ms | 81.4 ms | A100 cuDNN is **slower** for this workload |
| Rendering (fwd+bwd) | ~7+41=48 ms | ~3.4+11.7=15 ms | A100 is **3.2x faster** at rendering |
| Adam | 17.5 ms | 7.9 ms | A100 is **2.2x faster** at optimizer |
| Elementwise | 14.9 ms | 3.9 ms | A100 is **3.8x faster** at elementwise |
| **Total** | 88.1 ms | 103.5 ms | A100 is **1.18x slower** overall |

**Root cause**: A100's cuDNN convolution for the 11×11 Gaussian blur on 1920×1080×3 images is significantly slower than RTX 5070's. The A100's SM80 architecture has different cuDNN kernel selection that is less efficient for this specific conv configuration. Meanwhile, A100 is much faster at everything else (rendering, optimizer, elementwise), which means D-SSIM's relative share grows from 40.7% to 78.6%.

**Checkpoint quality matters**: The RTX 5070 used a diverged checkpoint (PSNR=12.03) while A100 uses a normal checkpoint (PSNR=20.56). A normal model produces images closer to GT, which means the SSIM intermediate values are in a different numerical regime. However, this does not explain the 2.27x absolute difference — that is primarily an architecture/cuDNN effect.

### D-SSIM kernel count

| Platform | D-SSIM kernels/iter |
|----------|-------------------|
| RTX 5070 | 9 |
| A100 | **63** |

The A100 cuDNN library dispatches 63 kernels per iteration for the same D-SSIM computation — 7x more than RTX 5070's 9. This suggests cuDNN on SM80 uses a different algorithm decomposition (possibly more fine-grained tiling) that is less efficient for this small-kernel workload.

## 7. Implications for C42

The A100 validation confirms that **D-SSIM optimization is even more critical on A100 than the RTX 5070 exploration suggested**:

1. **D-SSIM is 78.6% of GPU time** — optimizing it has the highest possible impact
2. **Rendering is only 3.6%** — the gsplat renderer is NOT the bottleneck
3. **Backward is 12.4%** — secondary to D-SSIM
4. **Total iteration is 95 ms** — reducing D-SSIM can dramatically cut this

Any D-SSIM speedup translates nearly 1:1 to total iteration speedup because D-SSIM dominates so completely.

## 8. Data Provenance

| Item | Location |
|------|----------|
| Script | `scripts/phase-c42/c40_baseline_a100.py` |
| A100 JSON | `results/a100/validation-c40-c42/c40_baseline_a100.json` |
| Hardware metadata | `results/a100/hardware_metadata.json` |
| RTX 5070 exploratory data | `results/exploratory/rtx5070/c40/` (NOT for final claims) |

---

**Status**: D-SSIM bottleneck CONFIRMED on A100. Proceed to C42 downsampled SSIM validation.
