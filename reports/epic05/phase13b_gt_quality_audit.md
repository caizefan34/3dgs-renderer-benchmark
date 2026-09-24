# Phase 13B — Ground-Truth Quality Audit

**Date:** 2026-08-28  
**Author:** DSH coding agent  
**Status:** COMPLETE

---

## 1. Executive Finding

> **Tile size has zero impact on rendered image quality.** All tile sizes {4, 8, 12, 16, 20, 24, 28, 32} produce pixel-identical output (max_abs_diff = 0.0, mean_abs_diff = 0.0, changed_pixel_ratio = 0.0) across all three test scenes.

> This finding is **expected but important** — it confirms that tile_size is purely a performance/compute knob, not a quality knob. The tile size only affects how the rasterizer partitions the image and sorts intersections, not the numerical computation of pixel values.

---

## 2. Methodology

### 2.1 Layer A: Candidate vs Baseline (tile16)
- Compare rendered image at each tile_size vs tile_size=16
- Metrics: max_abs_diff, mean_abs_diff, changed_pixel_ratio (|diff| > 1e-5)
- Coverage: all 8 tile_sizes × 3 scenes = 21 comparisons (tile16 excluded as self-reference)

### 2.2 Layer B: Candidate vs Real GT
- Compare rendered image at each tile_size vs ground truth image
- Metrics: PSNR, SSIM, LPIPS (AlexNet)
- Coverage: all 8 tile_sizes × 3 scenes = 24 quality measurements
- GT source: Mip-NeRF 360 official dataset images

### 2.3 Protocol
- Resolution: 1920×1080
- SH degree: 3 (full)
- Packed mode: True
- Camera: First real camera from each scene (matches GT image index)

---

## 3. Layer A Results — Pixel Equivalence

### 3.1 All vs tile16 Baseline

| tile_size | Room | Bicycle | Garden |
|:---------:|:----:|:-------:|:------:|
| 4 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |
| 8 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |
| 12 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |
| 16 | (self) | (self) | (self) |
| 20 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |
| 24 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |
| 28 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |
| 32 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 | max=0.0, mean=0.0, changed=0.0 |

**Conclusion:** All tile sizes produce bit-exact output. The tile size does not change the numerical computation of the rasterizer — it only changes the parallelization and sorting strategy.

---

## 4. Layer B Results — Quality vs Real GT

### 4.1 PSNR (dB, higher = better)

| tile_size | Room | Bicycle | Garden |
|:---------:|:----:|:-------:|:------:|
| 4 | 30.00 | 21.55 | 22.65 |
| 8 | 30.00 | 21.55 | 22.65 |
| 12 | 30.00 | 21.55 | 22.65 |
| 16 | 30.00 | 21.55 | 22.65 |
| 20 | 30.00 | 21.55 | 22.65 |
| 24 | 30.00 | 21.55 | 22.65 |
| 28 | 30.00 | 21.55 | 22.65 |
| 32 | 30.00 | 21.55 | 22.65 |
| **Mean** | **30.00** | **21.55** | **22.65** |
| **Variance** | **0.00** | **0.00** | **0.00** |

### 4.2 SSIM (higher = better)

| tile_size | Room | Bicycle | Garden |
|:---------:|:----:|:-------:|:------:|
| 4 | 0.9892 | 0.9266 | 0.9442 |
| 8 | 0.9892 | 0.9266 | 0.9442 |
| 12 | 0.9892 | 0.9266 | 0.9442 |
| 16 | 0.9892 | 0.9266 | 0.9442 |
| 20 | 0.9892 | 0.9266 | 0.9442 |
| 24 | 0.9892 | 0.9266 | 0.9442 |
| 28 | 0.9892 | 0.9266 | 0.9442 |
| 32 | 0.9892 | 0.9266 | 0.9442 |

### 4.3 LPIPS (AlexNet, lower = better)

| tile_size | Room | Bicycle | Garden |
|:---------:|:----:|:-------:|:------:|
| 4 | 0.191 | 0.304 | 0.139 |
| 8 | 0.191 | 0.304 | 0.139 |
| 12 | 0.191 | 0.304 | 0.139 |
| 16 | 0.191 | 0.304 | 0.139 |
| 20 | 0.191 | 0.304 | 0.139 |
| 24 | 0.191 | 0.304 | 0.139 |
| 28 | 0.191 | 0.304 | 0.139 |
| 32 | 0.191 | 0.304 | 0.139 |

---

## 5. Quality-Preserving Classification

| tile_size | Classification | Rationale |
|:---------:|:--------------|:----------|
| 4 | **A. Quality-preserving** | Pixel-identical output, identical metrics |
| 8 | **A. Quality-preserving** | Pixel-identical output, identical metrics |
| 12 | **A. Quality-preserving** | Pixel-identical output, identical metrics |
| 16 | **A. Quality-preserving** | Reference baseline |
| 20 | **A. Quality-preserving** | Pixel-identical output, identical metrics |
| 24 | **A. Quality-preserving** | Pixel-identical output, identical metrics |
| 28 | **A. Quality-preserving** | Pixel-identical output, identical metrics |
| 32 | **A. Quality-preserving** | Pixel-identical output, identical metrics |

> **All tile sizes are classified "A — Quality-preserving."** No tile size changes the numerical output or reconstruction quality.

---

## 6. Per-View Variance

Not applicable — all tile sizes produce identical per-view output. Variance across views is determined solely by the training process, not by tile size during evaluation.

---

## 7. Theoretical Explanation

The tile size only affects:
1. **Tile grid partitioning** — how the image is divided into tiles
2. **Intersection detection** — which Gaussians overlap which tiles
3. **Sorting granularity** — how intersections are sorted within tiles
4. **Parallelization** — how many threads/block are used

It does NOT affect:
1. The projection of 3D Gaussians to 2D
2. The evaluation of Gaussian weights/splats
3. The alpha-compositing math
4. The color computation from SH coefficients

Therefore, as long as the kernel correctly handles all tile_size values (confirmed in the compatibility audit), the rendered output is mathematically identical.

---

## 8. Caveat: Training Quality

This audit covers **snapshot quality** — evaluating a frozen checkpoint at different tile sizes. **Training quality** (how well the model converges) could theoretically differ if:
1. The backward pass uses tile-size-dependent numerical approximations
2. Gradient accumulation varies due to different kernel paths

However, since:
- Forward pass is bit-identical
- Backward pass uses the same mathematical formulation (just different partition)
- Quality vs GT is identical across tile sizes for frozen checkpoints

The training quality should also be identical. This was already confirmed for room scene in Phase 7 (30K training: tile16 vs tile32 achieved equivalent final PSNR).

---

*Report generated 2026-08-28*
