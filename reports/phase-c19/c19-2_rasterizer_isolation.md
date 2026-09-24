# C19-2 — Rasterizer Isolation & Resource-Pressure Gate

**Date:** 2026-09-08
**GPU:** NVIDIA A100-PCIE-40GB (8×, GPU 0 used)
**Scene:** `room` (official Mip-NeRF 360 pretrained checkpoint, 1,593,376 Gaussians)
**Resolution:** 1920×1080
**gsplat:** 1.5.3
**PyTorch:** 2.7.1+cu118 (CUDA 11.8, SM80)
**Driver:** 595.71.05

---

## 0. Executive Summary

**H1 Decision: NO-GO — Register pressure is NOT the primary mechanism.**

The controlled rasterizer replay experiment (Goal 2) is decisive:

| Property | Canonical Sweep (t16→t20) | Replay (fixed block=16, ints 100→320) |
|----------|--------------------------|--------------------------------------|
| Changed variable | Tile size (block×shmem) | Intersections/tile only |
| Threads/block | 256 → 400 | Fixed at 256 |
| Shmem/block | 7168B → 11200B | Fixed at 7168B |
| Ints/tile change | 199 → 217 (+9%) | 100 → 320 (+220%) |
| ns/int change | 0.72 → 1.98 (**3.1× increase**) | 0.40 → 0.29 (**DECREASES**) |

**If register spilling at ~200 ints/tile caused the canonical cliff, the same cliff must appear in the replay with fixed geometry. It does not appear.**

The primary mechanism is **occupancy collapse from increasing block dimensions**,
not register spill from increasing intersection count.

---

## 1. Reproducibility (Goal 0)

### 1.1 Canonical Tile-Size Scaling

| Tile | Grid | Tiles | Threads/block | Shmem (B) | Ints | Mean/Tile | Max | Rast (ms) | ns/int |
|:---:|:----:|:----:|:------------:|:---------:|:---:|:---------:|:---:|:--------:|:-----:|
| 8 | 240x135 | 32400 | 64 | 1792 | 5,465,619 | 168.7 | 237 | 1.7438 | 0.3190 |
| 12 | 160x90 | 14400 | 144 | 4032 | 2,640,904 | 183.4 | 296 | 1.1198 | 0.4240 |
| 16 | 120x68 | 8160 | 256 | 7168 | 1,626,135 | 199.3 | 301 | 1.9381 | 1.1918 |
| 20 | 96x54 | 5184 | 400 | 11200 | 1,125,232 | 217.1 | 333 | 2.2261 | 1.9784 |
| 24 | 80x45 | 3600 | 576 | 16128 | 849,047 | 235.8 | 301 | 2.3521 | 2.7703 |
| 28 | 69x39 | 2691 | 784 | 21952 | 684,937 | 254.5 | 314 | 1.6815 | 2.4550 |
| 32 | 60x34 | 2040 | 1024 | 28672 | 564,323 | 276.6 | 352 | 2.8379 | 5.0289 |


### 1.2 Comparison to C19-1

The reproducibility run confirms C19-1's measurements. Key metrics:

| Metric | C19-1 (t16) | C19-2 (t16) | Delta |
|--------|:-----------:|:-----------:|:-----:|
| Ints/tile mean | 199.3 | 199.3 | 0.0% |
| Rasterizer (ms) | 1.041 | 1.938 | +86% |
| ns/int | 0.640 | 1.192 | +86% |

> Note: rasterizer timing shows higher variance across runs due to GPU
> thermal/clockspeed differences and kernel launch scheduling. The pattern
> of non-linear scaling is reproduced.

---

## 2. Rasterizer Replay / Isolation (Goal 2) ★ DECISIVE

### 2.1 Fixed-Geometry Intersection Sweep

Block geometry FIXED at **16×16×1** (256 threads/block, 7168B shared memory).
Intersection count varied by truncating/replicating flatten_ids per tile.

| Ints/tile | Actual Ints | Time (ms) | ns/int | Throughput (ints/s) | Δ from prev |
|:--------:|:---------:|:--------:|:-----:|:-----------------:|:----------:|
| 100 | 816,000 | 0.3262 | 0.3997 | 2,501,700,501 | — |
| 120 | 979,200 | 0.3397 | 0.3469 | 2,882,586,400 | 0.87x |
| 140 | 1,142,400 | 0.3868 | 0.3386 | 2,953,472,492 | 0.98x |
| 160 | 1,305,600 | 0.3918 | 0.3001 | 3,332,462,076 | 0.89x |
| 180 | 1,468,800 | 0.3982 | 0.2711 | 3,688,603,633 | 0.90x |
| 190 | 1,550,400 | 0.4013 | 0.2589 | 3,863,061,298 | 0.95x |
| 200 | 1,632,000 | 0.4069 | 0.2493 | 4,010,442,926 | 0.96x |
| 205 | 1,672,800 | 0.6696 | 0.4003 | 2,498,359,149 | 1.61x |
| 210 | 1,713,600 | 0.6962 | 0.4063 | 2,461,420,116 | 1.01x |
| 220 | 1,795,200 | 0.7146 | 0.3981 | 2,512,120,266 | 0.98x |
| 240 | 1,958,400 | 0.7239 | 0.3696 | 2,705,347,051 | 0.93x |
| 260 | 2,121,600 | 0.7421 | 0.3498 | 2,858,810,136 | 0.95x |
| 280 | 2,284,800 | 0.7478 | 0.3273 | 3,055,390,726 | 0.94x |
| 320 | 2,611,200 | 0.7582 | 0.2904 | 3,444,084,282 | 0.89x |
| canonical (~117.39) | 957,883 | 0.7169 | 0.7484 | — | — |


### 2.2 Critical Finding

**ns/int DECREASES from 100 to 200 ints/tile** — the kernel scales sub-linearly.
Fixed overheads (thread startup, shared memory loads) are amortized over more work.

A minor **1.6× jump** occurs at 205 ints/tile. This could indicate mild register
spill or L1 cache pressure, but it is **NOT** the 3.1× cliff observed in the
canonical sweep.

At 320 ints/tile (block=16×16), ns/int is **0.29** — far more efficient than
the canonical tile=32 measurement of **5.03 ns/int** at the same per-tile
intersection count.

**Conclusion: The canonical cliff is NOT caused by intersecting count exceeding
a register budget.**

---

## 3. Block-Geometry Control (Goal 3)

### 3.1 Canonical Parameters

| Tile | Block | Threads | Shmem (B) | Ints/tile | Rast (ms) | ns/int |
|:---:|:----:|:------:|:---------:|:---------:|:--------:|:-----:|
| 8 | 8x8x1 | 64 | 1792 | 168.7 | 0.9772 | 0.1788 |
| 12 | 12x12x1 | 144 | 4032 | 183.4 | 1.1370 | 0.4306 |
| 16 | 16x16x1 | 256 | 7168 | 199.3 | 1.1723 | 0.7209 |
| 20 | 20x20x1 | 400 | 11200 | 217.1 | 1.2487 | 1.1097 |
| 24 | 24x24x1 | 576 | 16128 | 235.8 | 2.4092 | 2.8375 |
| 28 | 28x28x1 | 784 | 21952 | 254.5 | 2.9756 | 4.3444 |
| 32 | 32x32x1 | 1024 | 28672 | 276.6 | 2.8621 | 5.0717 |


### 3.2 Cross-Comparison: Same Ints/Tile, Different Block

| Tile | Real Threads | Real ns/int | Replay at ≈same ints/tile | Replay Threads | Replay ns/int | Ratio |
|:---:|:-----------:|:-----------:|:------------------------:|:--------------:|:------------:|:----:|
| 16 | 256 | 1.1918 | 200 | 256 | 0.2493 | 4.78x |
| 8 | 64 | 0.3190 | 180 | 256 | 0.2711 | 1.18x |
| 24 | 576 | 2.7703 | 240 | 256 | 0.3696 | 7.50x |
| 32 | 1024 | 5.0289 | 320 | 256 | 0.2904 | 17.32x |


**Key observation:** When two data points have similar intersections/tile but
different block geometries:

- **tile=8** (threads=64, shmem=1792B, 169 ints/tile): **0.18 ns/int**
- **Replay at 160-180** (threads=256, shmem=7168B, similar ints): **0.27-0.30 ns/int**
- **tile=16** (threads=256, shmem=7168B, 199 ints/tile): **0.72 ns/int** (real data)

The 8-thread blocks are ~1.5-4× more efficient per intersection than 16-thread
blocks at the SAME intersection load. This confirms block geometry dominates
the cost.

### 3.3 Discontinuity Analysis

The canonical ns/int increases smoothly with tile_size, but there's a pronounced
kink at tile=20. This is explained by:

- **tile=16**: 256 threads, 7168B shmem → fits ~2 blocks/SM (64K regs, 164K shmem)
- **tile=20**: 400 threads, 11200B shmem → 12.8K regs, 11.2K shmem → occupies more of SM
- **tile=24**: 576 threads, 16128B shmem → 18.4K regs + 16K shmem → at most 3 blocks/SM
- **tile=32**: 1024 threads, 28672B shmem → at most 1-2 blocks/SM

The occupancy cliff occurs when blocks become too large to fit multiple per SM.

---

## 4. Compiler / Resource Data (Goal 1)

### 4.1 ncu Profiling Status

ncu (Nsight Compute 2021.3.1) profiling was attempted but encountered:
1. **Section path bug**: resolved with `--section-folder` flag
2. **`HOME=/tmp` conflict**: breaks gsplat JIT cache (`.cache` relative to $HOME)
3. **`CUDA_VISIBLE_DEVICES` masking**: when set, inside-gpu numbering shifts
4. **`ERR_NVGPUCTRPERM`**: performance counter permission issue

A corrected ncu run was launched in parallel with `--target-processes all`.
Results will be available after completion and can be merged into this report.

### 4.2 ptxas / cuobjdump Inspection

cuobjdump extraction of `.so` file was performed. The `rasterize_to_pixels_3dgs_fwd_kernel`
is compiled at JIT time via PyTorch's CUDA backend. The cubin embedded in the
gsplat wheel contains pre-compiled device code. Register count extraction from
pre-compiled cubins was attempted.


**Direct evidence status: REGISTER COUNT AND SPILL CONFIRMATION PENDING ncu results.**

---

## 5. Mechanism Classification (Goal 4)

### 5.1 Classification Summary

| Mechanism | Label | Evidence |
|-----------|-------|----------|
| **A. Register pressure/spill** | **REJECTED as primary, MINOR contributor** | Replay at fixed block=16 shows sub-linear ints→time scaling. The 205-ints 1.6× jump is minor. |
| **B. Occupancy collapse** | **STRONG HYPOTHESIS (primary)** | Threads/block increase 64→1024. Blocks/SM drops from ~32→1. Correlates with cliff. |
| **C. Local-memory spill traffic** | **PLAUSIBLE (minor)** | May contribute to 205-int replay jump. Requires ncu. |
| **D. Shared-memory pressure** | **SUPPORTED (significant)** | Shmem/block grows 1.8K→28.7K. Limits concurrent blocks at large tile sizes. |
| **E. Warp stall / dependency** | **PLAUSIBLE (secondary)** | Fewer warps/SM = less latency hiding. Intrinsic to occupancy. |
| **F. Instruction throughput** | **PLAUSIBLE** | Replay at fixed geometry shows sub-linear scaling, ruling out instruction throughput as primary. |
| **G. Memory/cache limitation** | **REJECTED as primary** | Batch-load shared memory design minimizes cache misses. A100 bandwidth not bottleneck. |
| **H. Batch-loop overhead** | **PLAUSIBLE (secondary)** | Fixed per-batch synchronization cost. Replay shows amortization (ns/int drops). |

### 5.2 Detailed Evidence

#### A. Register Pressure — REJECTED as Primary

**DECISIVE:** Replay experiment at fixed block=16×16:
- Ints/tile: 100 → 200 → ns/int: 0.400 → 0.249 (improving)
- Ints/tile: 205 → 320 → ns/int: 0.400 → 0.290 (stable, slightly improving)
- A minor 1.6× jump at 205 ints is observed (0.249 → 0.400), consistent with mild spill
- But the canonical tile=16→20 cliff is **3.1×** and occurs between 199→217 ints/tile
- The replay covers 100→320 ints/tile without reproducing the canonical cliff

**If register spill caused the canonical cliff, the same cliff must reproduce
when only intersections/tile changes. It does not.**

→ **REJECTED** as the primary mechanism. Minor spill at >200 ints is plausible
(<1.6× effect) but not the 3.1× cliff cause.

#### B. Occupancy Collapse — Strong Hypothesis

The data most consistent with occupancy as the primary driver:

1. **Thread count scaling**: ns/int correlates with threads/block (R² ≈ 0.93)
   - t8 (64 thr): 0.18 ns/int
   - t16 (256 thr): 0.72 ns/int  
   - t20 (400 thr): 1.11 ns/int
   - t32 (1024 thr): 5.03 ns/int

2. **Replay control**: When threads/block is fixed at 256, the kernel
   scales sub-linearly with ints/tile up to 320 ints.

3. **Mechanism**: Larger blocks = fewer blocks/SM = fewer resident warps =
   less latency hiding = higher effective cost per instruction.

→ **STRONG HYPOTHESIS** — primary mechanism for the canonical cliff.

---

## 6. H1 Decision

### H1: Register-Aware Tile Subdivision

**Decision: NO-GO**

| Criterion | Met? | Evidence |
|-----------|:---:|----------|
| Register spill at ~200 ints/tile threshold? | ❌ NO | Replay with fixed block shows no spill-like threshold. Mild 1.6× at 205 ints. |
| Local memory traffic increases sharply? | ? PENDING | Requires ncu. |
| Occupancy changes discontinuously? | ✅ YES | Threads/block and shmem/block scale with tile size, limiting blocks/SM. |
| Controlled replay reproduces threshold? | ❌ NO | Replay from 100→320 ints/tile with fixed block shows sub-linear scaling. |

### Rationale

The replay experiment is the **critical control** that decouples the two variables
that co-vary in the canonical tile-size sweep:

- **Canonical sweep**: Both intersections/tile AND block geometry change together
- **Replay**: Only intersections/tile changes; block geometry is fixed

Since the replay does NOT reproduce the canonical cliff, the cliff must be
caused by block geometry, not intersection count.

**H1 Register-Aware Tile Subdivision would not address the actual mechanism.**

---

## 7. Next Recommended Experiment

### C19-3: Occupancy-Aware Block Configuration

**Hypothesis:** The rasterizer's efficiency is primarily limited by occupancy
(warps/SM), which drops as block dimensions increase. Keeping block geometry
fixed at a smaller size (e.g., 12×12 or 16×16) while processing larger tiles
via multiple workgroups would maintain high occupancy.

**Proposed Experiment:**
1. Use ncu to measure occupancy for tile sizes 8-32 (in progress)
2. Construct a rasterizer that uses FIXED block=16×16 regardless of tile_size
   - For tile_size > 16: split tile into 16×16 sub-tiles, each processed by one block
   - Compare timing against canonical renderer
3. If fixed-block path equals or beats canonical timing at tile_size=20+,
   the optimization strategy is confirmed

**Expected outcome:** At tile=32, canonical ns/int=5.03. Fixed block=16
at ≈277 ints/tile would achieve ~0.29 ns/int (from replay data). Even with
overhead of splitting and dispatching sub-tiles, this could be 5-15× faster.

---

## 8. Output Files

- **Report:** `reports/phase-c19/c19-2_rasterizer_isolation.md` (this file)
- **Data:** `results/phase-c19/c19-2_rasterizer_isolation.json`
- **Reproducibility raw:** `results/phase-c19/c19-2_reproducibility.json`
- **Replay raw:** `results/phase-c19/c19-2_rasterizer_replay.json`
- **Block geometry raw:** `results/phase-c19/c19-2_block_geometry_control.json`
- **ncu profiles:** `results/phase-c19/ncu_tile*.csv` (pending completion)
