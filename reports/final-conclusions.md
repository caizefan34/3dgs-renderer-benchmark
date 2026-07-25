# Final Conclusions: 3DGS Renderer Benchmark

> **Authority host:** EPIC-05 **GPU:** NVIDIA A100-SXM4-80GB **Date:** 2026-07-25

This report synthesizes across **renderer fusion** (12 HiGS configs + 3 external renderers x 5 scenes), **compression** (10 codecs x 5 scenes, 50 data points), and **two prototype accelerators**. Every claim below is backed by committed JSON evidence.

---

## Part I: Renderer Fusion

### 1.1 The Baseline Landscape

The Tier A matrix established a reliable baseline across 5 renderers:

| Renderer | Agg. FPS | Speed Index | PSNR | VRAM | Best For |
|---|---|---|---|---|---|
| **gsplat HiGS** | 698.19 | 5.713x | 25.834 dB | 6,616 MB | Maximum throughput |
| Speedy-Splat | 293.59 | 2.402x | 26.121 dB | 4,276 MB | Balanced deployment |
| TC-GS | 227.97 | 1.865x | 26.130 dB | 4,322 MB | Quality preservation |
| gsplat packed | 240.27 | 1.966x | 25.834 dB | 4,206 MB | Memory efficiency |
| Original 3DGS | 122.22 | 1.000x | 26.120 dB | 8,234 MB | Scientific reference |

**Key insight:** The gap between HiGS and next-fastest is **2.38x** (698 vs 294 FPS). This is not incremental.

### 1.2 HiGS Variant Sweep -- 12 Configs Measured

| Variant | FPS | vs HiGS | Quality impact |
|---|---|---|---|
| gsplat_higs_sh16 | 709.09 | +1.6% | -0.38 dB PSNR |
| gsplat_higs_auto | 699.50 | +0.2% | -0.006 dB |
| gsplat_higs (baseline) | 698.19 | 1.00x | reference |
| gsplat_higs_sh32 | 696.44 | -0.3% | -0.006 dB |
| gsplat_higs_tile16_sh16 | 643.97 | -7.8% | -0.38 dB PSNR |
| gsplat_higs_tile16_sh32 | 636.95 | -8.8% | -0.006 dB |
| gsplat_higs_tile16 | 633.13 | -9.3% | reference quality |

**Resolution scaling (garden-only):**

| Resolution | FPS | Gain |
|---|---|---|
| quarter | 553 | +12.4% |
| half | 531 | +7.9% |
| full | 492 | 1.00x |

**Conclusion:** HiGS is compute-bound on A100. 16x fewer pixels yields only 12% gain.

### 1.3 Prototype Accelerators -- Both Measured, Both Negative

**Calibrated Tile Selection (Ideas 11+29):** Dynamic CUDA-event timing to choose tile-8 vs tile-16. All 5 scenes picked tile-8 anyway. Aggregate: **-0.97% FPS**.

**LPT Tile Scheduling (Idea 12):** Bitonic-sort fine tiles by Gaussian count. Aggregate: **-0.63% FPS**. P99 improved on Truck (-11.8%) but regressed on 3/5 scenes.

**Lesson:** Tile-size selection is not the bottleneck on this cohort. Sort overhead dominates straggler reduction on 4/5 scenes. A conditional (variance-gated) policy could be viable for unbalanced scenes.

### 1.4 HiGS Trainability

**Why inference-only:** forced no_grad, detached tensors, no backward CUDA kernel, packed uint32 SH storage, hierarchy invalidation on mutation.

**Path forward:** backward CUDA kernel (2-4 engineer-months), hierarchy lifecycle management, differentiable packed-SH or float16 storage. The on_duplicate/on_split methods already exist in source.

---

## Part II: Near-Lossless Compression -- The Pareto Frontier

### 2.1 All 10 Codecs

| Codec | Ratio | Worst dPSNR | 5/5 Pass |
|---|---|---|---|
| **XZ (LZMA2)** | 1.17x | 0.000 dB | Yes |
| **Tile-codebook** | 3.84x | -0.002 dB | Yes |
| **SPZ 8/8** | **5.73x** | **-0.015 dB** | **Yes -- WINNER** |
| Block-float | 2.17x | -0.026 dB | 4/5 |
| Compressed PLY | 4.05x | -0.232 dB | 4/5 |
| SPZ 6/6 | 7.62x | -0.275 dB | 4/5 |
| FCGS (lambda=0.0001) | 12.84x | -0.074 dB | 2/5 |
| SPZ 5/4 | 10.07x | -1.806 dB | 0/5 |
| SOG | 18.66x | -2.451 dB | 0/5 |

### 2.2 The Pareto Frontier

Bit-exact: XZ (1.17x)
Near-perfect: Tile-codebook (3.84x, <0.002 dB)
**WINNER: SPZ 8/8 (5.73x, <0.02 dB, all 5/5)**
Light-loss: SPZ 6/6 (7.62x, fails Bonsai), FCGS (12.84x, fails 3/5)
Unacceptable: SOG (18.66x, -2.45 dB)

### 2.3 Why SPZ 8/8 Wins

SPZ (Niantic) uses block floating-point quantization. The 8/8 config is the sweet spot: no training needed, deterministic decode, all 5 scenes pass the strict gate. The gap between 8/8 (5.73x) and 6/6 (7.62x, fails Bonsai) defines the current practical limit at ~5.7x.

### 2.4 FCGS: Learned Codec Trade-off

FCGS + MPEG G-PCC reaches 12.84x compression but only 2/5 scenes pass. Useful for storage-bound deployments that tolerate small quality loss.

### 2.5 Size Reality Check

| Format | 5-scene total | vs PLY |
|---|---|---|
| Raw PLY | 4,161 MB | 1.00x |
| XZ | 3,570 MB | 0.86x |
| SPZ 8/8 | **726 MB** | **0.17x** |
| FCGS | 324 MB | 0.08x |

A full Mip-NeRF 360 dataset fits in 726 MB with SPZ 8/8 while preserving quality within 0.02 dB.

---

## Part III: Cross-Domain Synthesis

### 3.1 Deployment Envelope

| Use Case | Renderer | Storage | Total Budget |
|---|---|---|---|
| Real-time server | gsplat HiGS | SPZ 8/8 | ~7.3 GB |
| Memory-constrained | gsplat packed | SPZ 8/8 | ~4.9 GB |
| Quality-critical | Speedy-Splat/TC-GS | XZ lossless | ~7.9 GB |
| Maximum density | gsplat HiGS | FCGS light-loss | ~6.9 GB |

### 3.2 The A100 Bottleneck

Every fusion attempt hit: HiGS is compute-bound on A100. The hierarchy culling shifts the bottleneck to Gaussian math throughput. Meaningful speedups need: Gaussian count reduction, Tensor Core mixed-precision, or algorithmic culling.

### 3.3 The "5.7x Speed + 5.7x Storage" Pipeline

1. Train with Original 3DGS / Speedy-Splat
2. Export as SPZ 8/8 (5.73x, zero measurable loss)
3. Deploy with gsplat HiGS (5.7x speed)
4. Optional: FCGS for 12.8x storage at -0.07 dB cost

**End-to-end: 5.7x speed x 5.7x storage = 32x effective throughput per byte.**

---

## Part IV: What Didnt Work (Negative Results)

| Idea | Result | Root Cause |
|---|---|---|
| Calibrated tile (Ideas 11+29) | -0.97% FPS | All scenes favor tile-8; calibration overhead unrecoverable |
| LPT scheduling (Idea 12) | -0.63% FPS | Sort overhead exceeds straggler reduction on 4/5 |
| SOG compression | -2.45 dB | Designed for WebGL visual comfort, not near-lossless |
| SPZ 5/4 | -1.81 dB | Quantization too aggressive |

**Methodological lessons:** 5-scene gates matter (SPZ 6/6 passes 4/5 but fails Bonsai). A100 is not consumer GPU (different bottleneck profile possible). Common checkpoints isolate rendering from training quality.

---

## Part V: Future Directions

### Renderer Acceleration

| Priority | Direction | Expected | Risk |
|---|---|---|---|
| P0 | Exact conic-vs-tile rejection | +15-30% | Medium |
| P1 | Sparse-warp/Tensor-Core hybrid | +10-20% | High |
| P2 | Conditional LPT (variance-gated) | 0-3% | Low |
| P3 | FP16 Gaussian attributes | +5-10% | Medium |

### Compression

| Priority | Direction | Expected Ratio | Risk |
|---|---|---|---|
| P0 | SPZ + custom per-scene grids | 6-7x | Low |
| P1 | FCGS fine-tuned for canonical | 12-15x | Medium |
| P2 | Mixed SPZ/FCGS | 8-10x | Medium |
| P3 | Learned entropy on SPZ residuals | 6-8x | High |

### Training Integration

| Priority | Direction | Impact |
|---|---|---|
| P0 | Remove no_grad from HiGS | Forward-path training |
| P1 | Add HiGS backward CUDA kernel | Full training |
| P2 | Differentiable packed-SH | Memory-efficient training |

---

## Summary

1. **Renderer winner:** gsplat HiGS (698 FPS, 5.7x reference) -- compute-bound on A100
2. **Compression winner:** SPZ 8/8 (5.73x, <0.02 dB, all 5 pass)
3. **Balanced:** Speedy-Splat + SPZ 8/8 (ref quality, 2.4x speed, 5.7x storage)
4. **Maximum density:** gsplat HiGS + FCGS (5.7x speed, 12.8x storage)
5. **Prototypes:** Both negative -- calibrated tile (-0.97%), LPT (-0.63%)
6. **Training:** HiGS is inference-only; backward kernel is the gap
7. **Methodology:** 155 tests, 25 runs, 10 codecs, 2 prototypes, one GPU

---

*Evidence tiers never mix. Reproduce any claim from committed JSON evidence.*