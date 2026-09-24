# Phase 13A — Tile-Size Selection Oracle

**Date:** 2026-08-28
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU
**Status:** COMPLETE

---

## Executive Summary

Measured **15** workloads via **real camera** across **3** scenes (bicycle, garden, room).

- **Real camera:** tile16 winner=4, tile32 winner=11
- **Synthetic camera (for comparison):** tile16 winner=1, tile32 winner=14

## 1. Data Collected — Real Camera

### 1.1 Workload Matrix

| Label | Scene | Iter | Source | N (Gs) | Visible (K) | Intersect (M) | t16 fwd(ms) | t32 fwd(ms) | Speedup | Winner |
|:------|:-----:|:----:|:------:|:------:|:-----------:|:-------------:|:-----------:|:-----------:|:-------:|:------:|
| room_room_t16_iter10000 | room | 10000 | training_checkpoint | 1,004,935 | 345.9 | 9.16 | 9.29 | 5.27 | 1.76x | tile32 |
| room_room_t16_iter15000 | room | 15000 | training_checkpoint | 1,219,406 | 395.6 | 13.47 | 13.13 | 5.92 | 2.22x | tile32 |
| room_room_t16_iter20000 | room | 20000 | training_checkpoint | 1,207,872 | 385.7 | 13.33 | 11.84 | 7.12 | 1.66x | tile32 |
| room_room_t16_iter25000 | room | 25000 | training_checkpoint | 1,199,627 | 377.3 | 13.04 | 13.07 | 7.10 | 1.84x | tile32 |
| room_room_t16_iter30000 | room | 30000 | training_checkpoint | 1,193,480 | 370.8 | 12.81 | 11.40 | 6.73 | 1.69x | tile32 |
| room_room_t16_iter5000 | room | 5000 | training_checkpoint | 899,729 | 208.7 | 4.68 | 9.30 | 8.42 | 1.10x | tile32 |
| room_room_t32_iter10000 | room | 10000 | training_checkpoint | 985,142 | 341.3 | 7.93 | 12.08 | 11.39 | 1.06x | tile32 |
| room_room_t32_iter15000 | room | 15000 | training_checkpoint | 1,167,393 | 382.9 | 12.59 | 16.19 | 8.75 | 1.85x | tile32 |
| room_room_t32_iter20000 | room | 20000 | training_checkpoint | 1,158,324 | 372.5 | 12.07 | 14.92 | 8.14 | 1.83x | tile32 |
| room_room_t32_iter25000 | room | 25000 | training_checkpoint | 1,152,120 | 364.1 | 11.94 | 13.42 | 9.19 | 1.46x | tile32 |
| room_room_t32_iter30000 | room | 30000 | training_checkpoint | 1,146,273 | 354.6 | 11.80 | 14.27 | 9.18 | 1.55x | tile32 |
| room_room_t32_iter5000 | room | 5000 | training_checkpoint | 895,240 | 207.4 | 4.53 | 8.54 | 9.02 | 0.95x | tile16 |
| bicycle_sfm | bicycle | 0 | sfm | 6,131,954 | 1806.2 | 6.08 | 25.84 | 38.09 | 0.68x | tile16 |
| garden_sfm | garden | 0 | sfm | 5,834,784 | 2250.3 | 6.13 | 27.99 | 40.89 | 0.68x | tile16 |
| room_sfm | room | 0 | sfm | 1,593,376 | 389.8 | 3.22 | 9.70 | 12.61 | 0.77x | tile16 |

### 1.2 Real vs Synthetic Camera Comparison

| Label | Real Winner | Syn FWD Winner | Syn FB Winner | Note |
|:------|:-----------:|:---------------:|:-------------:|:-----|
| room_room_t16_iter10000 | tile32 | tile32 | tile32 |  |
| room_room_t16_iter15000 | tile32 | tile32 | tile32 |  |
| room_room_t16_iter20000 | tile32 | tile32 | tile32 |  |
| room_room_t16_iter25000 | tile32 | tile32 | tile32 |  |
| room_room_t16_iter30000 | tile32 | tile32 | tile32 |  |
| room_room_t16_iter5000 | tile32 | tile32 | tile32 |  |
| room_room_t32_iter10000 | tile32 | tile32 | tile32 |  |
| room_room_t32_iter15000 | tile32 | tile32 | tile32 |  |
| room_room_t32_iter20000 | tile32 | tile32 | tile32 |  |
| room_room_t32_iter25000 | tile32 | tile32 | tile32 |  |
| room_room_t32_iter30000 | tile32 | tile32 | tile32 |  |
| room_room_t32_iter5000 | tile16 | tile32 | tile32 | ⚠️ Synthetic disagrees with real |
| bicycle_sfm | tile16 | tile32 | tile32 | ⚠️ Synthetic disagrees with real |
| garden_sfm | tile16 | tile16 | tile16 |  |
| room_sfm | tile16 | tile32 | tile16 | ⚠️ Synthetic disagrees with real |

## 2. Feature → Winner Correlation (Real Camera)

### 2.1 Top Features by Effect Size

| Rank | Feature | Effect Size | Direction | Threshold | t16_mean | t32_mean |
|:----:|:--------|:-----------:|:----------|:---------:|:--------:|:--------:|
| 1 | tpg_std | 4.53 | t32_winner_higher | 135.12 | 38.59 | 231.65 |
| 2 | empty_tile_ratio_estimate | 4.49 | t16_winner_higher | 1.00 | 1.00 | 1.00 |
| 3 | coverage_fraction | 4.49 | t32_winner_higher | 0.00 | 0.00 | 0.00 |
| 4 | p99_intersections_per_tile | 3.68 | t32_winner_higher | 281.59 | 97.00 | 466.18 |
| 5 | tpg_mean | 3.50 | t32_winner_higher | 20.04 | 9.05 | 31.02 |
| 6 | p95_intersections_per_tile | 3.06 | t32_winner_higher | 58.92 | 27.75 | 90.09 |
| 7 | mean_intersections_per_tile | 3.01 | t32_winner_higher | 990.07 | 611.61 | 1368.52 |
| 8 | gaussians_per_tile_mean | 3.01 | t32_winner_higher | 990.07 | 611.61 | 1368.52 |
| 9 | total_intersections | 3.01 | t32_winner_higher | 8078954.76 | 4990776.25 | 11167133.27 |
| 10 | sort_input_count | 3.01 | t32_winner_higher | 8078954.76 | 4990776.25 | 11167133.27 |

### 2.2 Detailed Feature Analysis

**tpg_std:**
- Effect size: 4.53
- Direction: t32_winner_higher (t16 winners avg=38.59, t32 winners avg=231.65)
- Threshold: 135.12

**empty_tile_ratio_estimate:**
- Effect size: 4.49
- Direction: t16_winner_higher (t16 winners avg=1.00, t32 winners avg=1.00)
- Threshold: 1.00

**coverage_fraction:**
- Effect size: 4.49
- Direction: t32_winner_higher (t16 winners avg=0.00, t32 winners avg=0.00)
- Threshold: 0.00

**p99_intersections_per_tile:**
- Effect size: 3.68
- Direction: t32_winner_higher (t16 winners avg=97.00, t32 winners avg=466.18)
- Threshold: 281.59

**tpg_mean:**
- Effect size: 3.50
- Direction: t32_winner_higher (t16 winners avg=9.05, t32 winners avg=31.02)
- Threshold: 20.04

## 3. Oracle 1: Simple Threshold Predictor

**Rule:** if `tpg_std` > 135.1193 → tile32, else tile16

- **Accuracy (all real camera data):** 0.933 (14/15)
- Confusion: {'t16_correct': 4, 't16_wrong': 0, 't32_correct': 10, 't32_wrong': 1}

### 3.1 Baseline Comparisons

| Strategy | Wrong choices | Error rate |
|:---------|:-------------:|:----------:|
| Always-tile16 | 11 | 73% |
| Always-tile32 | 4 | 27% |
| Oracle 1 | 1 | 7% |

## 4. Leave-One-Scene-Out Validation

**Feature:** tpg_std
**Overall accuracy:** 1.0 (2 preds)

| Held-Out | n_train | n_test | Threshold | Accuracy |
|:---------|:-------:|:------:|:---------:|:--------:|
| bicycle | 14 | 1 | 139.95 | 1.000 |
| garden | 14 | 1 | 140.36 | 1.000 |
| room | ? | ? | N/A | no_variation |

**Preliminary evidence of cross-scene generalizability.**

## 5. Checkpoint-Level Diversity (Room)

| Iteration | Pipeline | Winner | Speedup | Visible Gs | Intersections | t16_fwd | t32_fwd |
|:---------:|:--------:|:------:|:-------:|:----------:|:-------------:|:-------:|:-------:|
| 5000 | room_t16 | tile32 | 1.10x | 208.7K | 4.68M | 9.30 | 8.42 |
| 5000 | room_t32 | tile16 | 0.95x | 207.4K | 4.53M | 8.54 | 9.02 |
| 10000 | room_t16 | tile32 | 1.76x | 345.9K | 9.16M | 9.29 | 5.27 |
| 10000 | room_t32 | tile32 | 1.06x | 341.3K | 7.93M | 12.08 | 11.39 |
| 15000 | room_t16 | tile32 | 2.22x | 395.6K | 13.47M | 13.13 | 5.92 |
| 15000 | room_t32 | tile32 | 1.85x | 382.9K | 12.59M | 16.19 | 8.75 |
| 20000 | room_t16 | tile32 | 1.66x | 385.7K | 13.33M | 11.84 | 7.12 |
| 20000 | room_t32 | tile32 | 1.83x | 372.5K | 12.07M | 14.92 | 8.14 |
| 25000 | room_t16 | tile32 | 1.84x | 377.3K | 13.04M | 13.07 | 7.10 |
| 25000 | room_t32 | tile32 | 1.46x | 364.1K | 11.94M | 13.42 | 9.19 |
| 30000 | room_t16 | tile32 | 1.69x | 370.8K | 12.81M | 11.40 | 6.73 |
| 30000 | room_t32 | tile32 | 1.55x | 354.6K | 11.80M | 14.27 | 9.18 |

**Conclusion:** Tile-size preference CHANGES across training iterations.

## 6. Research Questions

### Q1: Which workload feature best predicts tile winner?
A: **tpg_std** (effect_size=4.53)

### Q2: Does a simple threshold exist?
A: Yes — **`tpg_std` > 135.1193 → tile32**

### Q3: Are room/bicycle/garden separable?
- **room:** tile16, tile32
- **bicycle:** tile16
- **garden:** tile16
Within-scene variation exists (checkpoint-dependent).

### Q4: Does training checkpoint change the winner?
A: **Yes** — winner changes across checkpoints.

### Q5: Oracle regret vs always-t16 / always-t32?
A: Among 15 real camera workloads:
   - Always-tile16 wrong: 11/15 (73%)
   - Always-tile32 wrong: 4/15 (27%)
   - Oracle wrong: 1/15 (7%)

### Q6: Can renderer-level oracle predict training-level winner?
A: Forward vs fwd+bwd winner agreement on synthetic data: 14/15 (93%)
   Note: synthetic camera is a poor proxy for real training workload.

### Q7: Sufficient evidence for adaptive tile-size?
A: **CONDITIONAL** — Feature-based prediction is possible (high effect size), but:
   - Within-scene variation EXISTS → adaptive kernel SUPPORTED

## 7. Key Insight: Synthetic vs Real Camera

The synthetic camera is fundamentally misleading for workload analysis:

- **room_room_t32_iter5000:** Real cam says `tile16` (t16=8.5ms, t32=9.0ms), but synthetic cam says `tile32` (t16=4.3ms, t32=3.1ms)
- **bicycle_sfm:** Real cam says `tile16` (t16=25.8ms, t32=38.1ms), but synthetic cam says `tile32` (t16=3.9ms, t32=3.7ms)
- **room_sfm:** Real cam says `tile16` (t16=9.7ms, t32=12.6ms), but synthetic cam says `tile32` (t16=3.0ms, t32=2.9ms)

**Root cause:** Synthetic camera sees <0.3% of Gaussians on outdoor scenes. Real workload characterizations MUST use real cameras.

---
*Report generated 2026-08-28 14:16:27*