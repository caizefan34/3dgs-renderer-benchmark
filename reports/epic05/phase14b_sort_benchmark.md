# Phase 14B — Segmented Sort Benchmark Results

**Date:** 2026-09-23
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM)
**Scene:** Mip-NeRF 360 — room (1,593,376 Gaussians, state-30000, 1080p)
**Protocol:** 10 forward passes (torch.inference_mode), CUDA synchronized, median timing
**Status:** COMPLETE

---

## 1. Benchmark Results

| Tile Size | Mode | Avg fwd (ms) | Std (ms) | n_isects | Peak Mem (MB) | Correct? |
|:---------:|:----:|:------------:|:--------:|:--------:|:-------------:|:--------:|
| **16** | baseline | **7.63** | 0.41 | 1,626,135 | 894 | — |
| **16** | segmented | **34.51** | 2.46 | 1,626,135 | 979 | ✓ (bit-exact) |
| **20** | baseline | **12.43** | 0.56 | 1,125,232 | — | — |
| **20** | segmented | **24.41** | 0.78 | 1,125,232 | — | ✓ (bit-exact) |
| **32** | baseline | **9.25** | 0.43 | 564,323 | — | — |
| **32** | segmented | **17.71** | 0.61 | 564,323 | — | ✓ (bit-exact) |

## 2. Speedup Ratios

| Tile Size | Ratio (segmented/baseline) | Result |
|:---------:|:-------------------------:|:-------|
| 16 | **4.52× slower** | ❌ NOT BENEFICIAL |
| 20 | **1.96× slower** | ❌ NOT BENEFICIAL |
| 32 | **1.91× slower** | ❌ NOT BENEFICIAL |

## 3. Conclusion

| Question | Answer |
|:---------|:-------|
| Is segmented sort faster? | **NO — 1.9–4.5× slower** in all configurations |
| Does it preserve pixel output? | **YES — bit-exact** (max diff = 0.0) |
| Does it preserve gradients? | **YES** (finite gradients) |
| Is it a training candidate? | **NO** |
| Next step | **Close Phase 14B** — move to visibility culling reconnaissance |

## 4. Analysis

The gsplat documentation warns explicitly (line 176 of `rendering.py`):

> *"Segmented radix sort performs sorting in segments, which is more efficient for the sorting operation itself. However, since it requires offset indices as input, **additional global memory access is needed, which results in slower overall performance in most use cases**."*

The benchmark confirms this warning:

1. **Single-camera (I=1):** Segmented sort degenerates to 1 segment (same as global) but adds offset computation + CUB segmented sort overhead. No benefit from narrower key width.

2. **Multi-camera (I>1):** Would segment per-image, potentially reducing sort key width. However, even in single-camera case, the 4.5× slowdown at tile16 shows the overhead dominates.

3. **Key width analysis:** The global sort sorts `32 + tn + in` bits (where `in = log2(I) = 0` for I=1). The segmented sort sorts `32 + tn` bits. For I=1, the key widths are identical, so there's no theoretical sorting work reduction.

## 5. Files Created

- `reports/epic05/phase14b_sorting_recon.md` — Source trace + design
- `reports/epic05/phase14b_sorting_source_trace.md` — Detailed CUDA source trace
- `reports/epic05/phase14b_sort_benchmark.md` — This file
- `results/epic05/phase14b_sorting.json` — Raw benchmark data
