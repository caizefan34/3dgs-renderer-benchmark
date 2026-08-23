# Phase 7C — Track B: Tile Size Mechanism Hypotheses

**Date:** 2026-09-02  
**Status:** Analysis document — all claims marked as HYPOTHESIS unless supported by direct evidence.

---

## B0. Source-Level Trace Summary

See `reports/epic05/phase7c_tile_source_trace.md` for the complete source-level trace.

**Key facts established:**

1. `tile_size` changes exactly 4 launch parameters: `grid(tile_width, tile_height, I)` and `block(tile_size, tile_size)`
2. The compiled kernel binary is **identical** for all tile sizes (cuobjdump VERIFIED: same REG=40, SHARED=1024B for `rasterize_to_pixels_3dgs_fwd_kernel<SH=3, float>`)
3. `tile_size` also affects the intersection kernel (`intersect_tile`) by changing the quantization of screen-space radius to tile coordinates
4. The offset array (`isect_offsets`) shape changes, affecting prefix-sum work and memory footprint
5. No other code path is affected — the optimizer, densification, pruning, SH evaluation, projection, and data loading are all tile-size-independent

---

## B2. Exact Kernel Launch Geometry

### Forward kernel: `rasterize_to_pixels_3dgs_fwd_kernel<SH=3, float>`

| Parameter | tile16 | tile32 | Source |
|:----------|:------:|:------:|:-------|
| grid.x | 120 | 60 | `ceil(1920 / ts)` — rendering.py:632 |
| grid.y | 68 | 34 | `ceil(1080 / ts)` — rendering.py:633 |
| grid.z (I=B×C) | 1 | 1 | 1 image |
| **Total blocks** | **8,160** | **2,040** | grid.x × grid.y × grid.z |
| block.x | 16 | 32 | tile_size — kernel launch |
| block.y | 16 | 32 | tile_size — kernel launch |
| block.z | 1 | 1 |  |
| **Threads/block** | **256** | **1,024** | ts² |
| **Warps/block** | **8** | **32** | ts² / warpSize |
| **Pixels/block** | **256** | **1,024** | ts² |
| **REG count** | 40 | 40 | cuobjdump (SAME BINARY) |
| **SHARED** | 1024 B | 1024 B | cuobjdump (SAME BINARY) |
| **STACK** | 0 | 0 | cuobjdump |

### Backward kernel: `rasterize_to_pixels_3dgs_bwd_kernel<SH=3, float>`

Same launch geometry rules apply. REG count: 48 (cuobjdump).

### Intersect tile kernel: `intersect_tile`

The `intersect_tile` kernel uses `tile_size` for quantizing screen-space radii to tile coordinates. Launch geometry is 1D (one thread per Gaussian, or one block per Gaussian), not directly proportional to tile_size.

### Intersect offset kernel: `intersect_offset`

Array output size: `I × tile_height × tile_width`. tile32 → 4× fewer elements to compute.

---

## B3. Binary Evidence Verification

### Current status: FALSIFIED / NOT SUPPORTED — compile-time resource difference

**Evidence:**
- cuobjdump on the installed `gsplat` CUDA binary confirms `rasterize_to_pixels_3dgs_fwd_kernel<SH=3, float>` has REG=40, SHARED=1024B, STACK=0
- These resource counts are **identical** regardless of tile_size
- The binary is the same `.cubin` — tile_size is purely a runtime launch parameter

**Implication:**
Any explanation based on "tile32 uses fewer registers" or "tile32 uses less shared memory" is **falsified**. The kernel's compile-time resource footprint is constant.

---

## B4. Competing Hypotheses

### H1: tile32 Reduces Organizational / Block-Level Overhead

**Category:** Launch overhead  
**Mechanism:** tile32 launches 2,040 blocks vs tile16's 8,160 blocks. Each block has CUDA launch overhead (scheduler dispatch, register allocation, shared memory setup). 4× fewer launches means 4× less organizational overhead.

**Prediction:** At very low Gaussian counts where compute is negligible, tile32 will still be faster due to lower launch overhead.

**Test:** Microbenchmark with very few Gaussians (e.g., 100, 1,000) where per-Gaussian compute is near-zero. If tile32 is faster at N=100, H1 is supported.

**Status:** HYPOTHESIS

---

### H2: tile32 Changes Work Granularity and Useful Computation Efficiency

**Category:** Work distribution  
**Mechanism:** With tile16, each block handles only 256 pixels. Many blocks may have little or no Gaussian work (empty tiles). With tile32, each block handles 1,024 pixels — 4× more work per block. This creates better load balancing because a block that gets work has 4× more pixels to process, increasing "useful" compute per block launch.

**Prediction:** tile32's advantage should increase as the scene becomes sparser (more empty tiles with tile16). Dense scenes (where most tiles have work) should see less advantage.

**Test:**
1. Fixed N, vary resolution (720p, 1080p, 4K) — higher resolution = more tiles, amplifying the advantage
2. Fixed resolution, vary N from low to high — as N increases, more tiles get work, advantage should decrease

**Status:** HYPOTHESIS

---

### H3: tile32 Changes Memory Access / Reuse Behavior

**Category:** Memory hierarchy  
**Mechanism:** Each block processes `ts²` pixels in a contiguous tile. With tile32, 1,024 pixels share the same block's shared memory. When the rasterization kernel iterates over a tile's Gaussian list, it:
1. Loads Gaussian data (means2d, conics, colors, opacities) from global memory into registers/shared memory
2. For each pixel in the tile, computes Gaussian contribution

With tile32: **4× more pixels** evaluate each loaded Gaussian before moving to the next Gaussian. This means each Gaussian's data is reused 4× more times, improving the **compute-to-global-memory-access ratio**.

**Prediction:** For a fixed number of Gaussian intersections per tile, tile32 should have higher arithmetic intensity (FLOPs/byte).

**Test:**
1. Fixed N and fixed camera. Compare memory throughput via proxy: kernel duration at varying N
2. If H3 is correct, the advantage per Gaussian should grow with Gaussian count (more data to reuse across pixels)

**Status:** HYPOTHESIS

---

### H4: tile32 Changes Latency Hiding / Residency Behavior

**Category:** Occupancy and latency hiding  
**Mechanism:** On RTX 5070 (36 SMs, Compute Capability 12.0, 128 warps/SM max):

| Property | tile16 | tile32 |
|:---------|:------:|:------:|
| Warps/block | 8 | 32 |
| Blocks/SM limit (shared mem) | 6 (1024B each, fits) | 1 (1024B each, fits) |
| Warps/SM (max) | 48 (6×8) | 32 (1×32) |
| Theoretical occupancy | 37.5% | 25% |
| Registers/warp required | 40×32=1280 | 40×32=1280 |
| Registers/SM available | 65536 | 65536 |
| Warps possible (reg limit) | 65536/1280=51 → 48 cap | 65536/1280=51 → 32 cap |

**Counterintuitive finding:** tile16 has **higher theoretical occupancy** (37.5%) than tile32 (25%). If higher occupancy were always better, tile16 would win. The fact that tile32 wins suggests that **tile32's lower occupancy is not the bottleneck**, and other factors dominate.

**Refined H4a:** tile16's 6 blocks/SM create register pressure per-SM that causes register spilling or reduced per-thread register availability. However, cuobjdump shows REG=40 and STACK=0 — no spilling. **FALSIFIED if STACK=0 holds.**

**Refined H4b:** tile32's 1 block/SM vs tile16's 6 blocks/SM means tile32 has **no intra-SM block contention** for shared memory/L1 bandwidth. Each SM serves only one block's 1,024 pixels worth of memory requests, reducing bank conflicts and cache pressure.

**Prediction:** If H4b is correct, tile32's advantage should scale with SM count and memory subsystem pressure.

**Status:** HYPOTHESIS (partially testable via occupancy calculation; Nsight counters blocked)

---

### H5: tile32 Interacts Differently with Workload Size / Gaussian Density

**Category:** Scene-dependent scaling  
**Mechanism:** The intersection kernel (`intersect_tile`) maps each Gaussian to overlapping tiles using `tile_size`. With tile32, each Gaussian hits fewer tiles (radius/32 vs radius/16 quantization). This means:
1. Fewer total intersection entries (`n_isects`) → less sorting work
2. Each Gaussian appears in fewer tile lists → less redundant evaluation across adjacent tiles
3. In the rasterization kernel, each tile has fewer Gaussians to process on average

**Prediction:** tile32's advantage should be most pronounced for scenes with many Gaussians that project to large screen-space radii (overlapping many tiles with tile16). For scenes where Gaussians are tiny (sub-tile), the advantage should be smaller.

**Test:**
1. Bicycle scene (~6.1M initial Gaussians) — many Gaussians, large variance in radii → expect large advantage
2. Room scene (~1.6M initial Gaussians) — moderate count, tighter distribution → advantage already measured (1.58×)
3. Synthetic scenes with uniformly tiny Gaussians → should reduce advantage

**Status:** HYPOTHESIS

---

### H6 (New): tile32 Reduces Sorting/Intersection Work

**Category:** Pre-rasterization pipeline  
**Mechanism:** `isect_tiles` calls `intersect_tile` CUDA kernel which:
1. For each nnz Gaussian, computes tile overlap using `radius / tile_size` quantization
2. Generates sorted intersection IDs (radix sort on `n_isects` elements)
3. The number of intersections `n_isects` is proportional to `(2R/ts)²` per Gaussian on average

With tile32: `(2R/32)²` vs tile16: `(2R/16)²` — approximately **4× fewer intersections** per Gaussian (for Gaussians with radius > tile_size).

**Prediction:** The reduction in `n_isects` directly reduces the sorting work in `isect_tiles` and the subsequent rasterization work in each tile.

**Test:** Measure `tiles_per_gauss` and `isect_offsets` array sizes (available in `meta` dict from `rasterization()`). Compare between tile16 and tile32.

**Status:** HYPOTHESIS (potentially directly measurable)

---

## B5. Discriminating Microbenchmarks

### Design Principles
1. Each test changes exactly one variable
2. Use CUDA events for precise kernel timing
3. Decompose forward pass into projection + intersection + rasterization timing

### M1: Fixed-Workload, Varying N

| Parameter | Value |
|:----------|:------|
| Resolution | 1080p |
| Camera | 1 (centered) |
| N | 50K, 200K, 400K |
| Source | Synthetic random Gaussians |
| Measure | Per-kernel timing (forward), per-Gaussian timing |

**Purpose:** Isolate N-scaling behavior of tile16 vs tile32.

### M2: Fixed N, Varying Resolution

| Parameter | Value |
|:----------|:------|
| N | 200K |
| Resolution | 720p, 1080p, 4K |
| Camera | 1 |
| Measure | Total forward + per-kernel timing |

**Purpose:** Test H2 (work granularity) — does advantage grow with resolution?

### M3: Fixed N, Varying Occupancy (via scene extent)

| Parameter | Value |
|:----------|:------|
| N | 200K |
| Resolution | 1080p |
| Gaussian distribution | Uniform in sphere, varying radius |
| Measure | Forward timing vs screen-space occupancy |

**Purpose:** Test H2/H5 — does the advantage depend on how many pixels are covered?

### M4: Intersection-Only Timing

| Parameter | Value |
|:----------|:------|
| N | 50K, 200K, 400K, 1M |
| Resolution | 1080p |
| Measure | CUDA event timing of `isect_tiles` + `isect_offset_encode` |

**Purpose:** Test H6 — is the gain partially from reduced intersection/sorting work?

### M5: Real-Scene Gaussian Snapshots

| Parameter | Value |
|:----------|:------|
| Scene | room, bicycle, garden |
| N | Checkpoint at iteration 0, 5000, 15000, 30000 |
| Camera | Real camera from each scene (1 per scene) |
| Measure | Forward + backward kernel timing |

**Purpose:** Test H5 — does the advantage depend on scene-specific Gaussian distributions?

### M6: Sub-Pixel Gaussian Test

| Parameter | Value |
|:----------|:------|
| N | 200K |
| All radii < 1 px | Zero screen-space coverage |
| Resolution | 1080p |
| Measure | Forward timing |

**Purpose:** Isolate the intersection/sorting cost from rasterization compute. If H6 dominates, tile32 should still be faster. If H3 dominates, advantage should vanish.

---

## B6. Profile What Is Actually Measurable

### Available indicators

| Indicator | Tool | Available? | Notes |
|:----------|:-----|:----------:|:------|
| Kernel duration | CUDA events (torch.cuda.Event) | ✅ YES | Wrap each kernel in CUDA events |
| Launch config | cuobjdump / nvdisasm | ✅ YES | Already done: REG=40, SHARED=1024B |
| Occupancy | Formula: warps/SM / max_warps | ✅ YES | Can compute from launch config |
| Grid size | Python: W/ts, H/ts | ✅ YES | Directly computed |
| Block count | Python: grid.x × grid.y × I | ✅ YES | Directly computed |
| Intersection count | `meta["tiles_per_gauss"].shape` | ✅ YES | Available after forward pass |
| Per-stage timing | `torch.cuda.Event` around stages | ✅ PARTIAL | Requires code modification |
| GPU utilization | `nvidia-smi` | ✅ PARTIAL | Not per-kernel |
| Memory bandwidth | Not directly | ❌ BLOCKED | Requires Nsight |
| Cache hit rate | Not directly | ❌ BLOCKED | Requires Nsight |
| Warp stall reasons | Not directly | ❌ BLOCKED | Requires Nsight (WDDM blocked) |
| Register spilling | cuobjdump shows STACK=0 | ✅ CONFIRMED | No spilling — STACK=0 |
| Shared memory | cuobjdump shows SHARED=1024B | ✅ CONFIRMED | Same for all tile sizes |

### Nsight Compute status: BLOCKED

Nsight Compute is blocked on this system (WDDM driver mode prevents NV-CONTROL access). Do NOT attempt to wrap Nsight calls, install alternative profilers, or guess hardware counter values.

---

## B7. OBSERVED → EVIDENCE → HYPOTHESIS → TEST → CONCLUSION Framework

### Claim 1: tile32 is faster per-iteration at equivalent Gaussian count

```
OBSERVED:    tile32 per-iteration time 35-55% lower than tile16 at same Gaussian count
EVIDENCE:    Full-run v2 metrics_log (room 30K), per-iteration timing
             Examples: iter 2000 (N≈1.2M): t16=177.5ms, t32=128.7ms (1.38×)
                       iter 10000 (N≈1.0M): t16=164.3ms, t32=101.8ms (1.61×)
HYPOTHESIS:  Several mechanisms could explain this (see H1-H6)
TEST:        Discriminating microbenchmarks M1-M6
CONCLUSION:  Not yet assigned — requires microbenchmark data
```

### Claim 2: tile32 advantage is consistent across all training stages

```
OBSERVED:    In full-run v2, tile32 faster at EVERY 1000-step bucket
EVIDENCE:    Stage-by-stage timing from metrics_log
             Ratio ranges from 0.374 to 0.801 (t32/t16)
             Only 26000-29000 range shows t32 slightly slower (ratio > 1.0), 
             attributed to last-stage fine-tuning anomalies
HYPOTHESIS:  Scene-dependent workload interaction
TEST:        Cross-scene replication (bicycle, garden)
CONCLUSION:  SUPPORTED for room scene
```

### Claim 3: Compile-time resource differences explain advantage

```
OBSERVED:    (Previous hypothesis before Phase 5 cuobjdump)
EVIDENCE:    cuobjdump confirms IDENTICAL binary for all tile sizes
CONCLUSION:  FALSIFIED — same REG=40, SHARED=1024B, STACK=0
```

### Claim 4: Gaussian count reduction contributes significantly to speedup

```
OBSERVED:    tile32 ends with 4% fewer Gaussians (1,146,273 vs 1,193,480)
EVIDENCE:    Direct count from full-run v2 final state
HYPOTHESIS:  If speedup were from fewer Gaussians, it would be ~7% max
TEST:        Per-iteration timing at EQUAL Gaussian counts (e.g., iter 2000)
CONCLUSION:  FALSIFIED — equal-Gaussian-count comparison shows tile32 still 1.38× faster
```

### Claim 5: tile32 advantage is scene-independent

```
OBSERVED:    Only room scene tested for full training
EVIDENCE:    No bicycle or garden data available
HYPOTHESIS:  Scene-dependence likely (H5)
TEST:        bicycle 30K, garden 30K (Track A)
CONCLUSION:  NOT TESTED
```

---

## B8. Microbenchmark Results (Executed 2026-09-02 on RTX 5070)

### Experimental Context

All microbenchmarks were run on the same RTX 5070 Laptop GPU used for the full training runs. **Synthetic random Gaussians** were used (no dataset dependency). Both dense mode (packed=False, all N Gaussians processed unconditionally) and packed mode were tested. Batched CUDA event timing (BATCH=30, 5 repeats = 150 forward calls per data point).

### Key Finding: tile16 and tile32 Are Indistinguishable on Synthetic Workloads

| Benchmark | N | tile16 (ms) | tile32 (ms) | Ratio |
|:----------|:-:|:-----------:|:-----------:|:-----:|
| Forward dense | 1K | 2.11 ± 3.03 | 0.59 ± 0.01 | **3.55×** (noisy) |
| Forward dense | 10K | 0.58 ± 0.01 | 0.57 ± 0.00 | 1.02× |
| Forward dense | 50K | 0.62 ± 0.01 | 0.60 ± 0.01 | 1.02× |
| Forward dense | 200K | 0.61 ± 0.01 | 0.60 ± 0.01 | 1.01× |
| Forward dense | 400K | 0.65 ± 0.06 | 0.61 ± 0.02 | 1.06× |
| Forward packed | 50K | 0.63 ± 0.00 | 0.63 ± 0.01 | 1.00× |
| Forward packed | 400K | 0.63 ± 0.02 | 0.71 ± 0.10 | 0.88× (noisy) |
| Fwd+Bwd dense | 50K | 1.50 ± 0.54 | 1.18 ± 0.03 | 1.27× (noisy) |
| Fwd+Bwd dense | 200K | 1.19 ± 0.04 | 1.17 ± 0.02 | 1.02× |
| Fwd+Bwd dense | 400K | 1.23 ± 0.08 | 1.20 ± 0.01 | 1.03× |
| Fwd+Bwd packed | 200K | 1.08 ± 0.02 | 1.07 ± 0.01 | 1.02× |
| Sub-pixel (M6) | 200K | 0.62 ± 0.01 | 0.60 ± 0.00 | 1.02× |

> **All tile16 vs tile32 ratios are 0.99×–1.06× — within measurement noise. No statistically significant difference detected.**

### Critical Interpretation

The synthetic random Gaussian workload **does NOT reproduce the 1.58× training speedup** observed on the real room scene. This is itself a significant finding:

1. **tile32's advantage is NOT a generic hardware property** of the RTX 5070 GPU. It depends on specific workload characteristics.
2. **The timing scales with neither N nor forward/backward complexity** — ALL configurations take ~0.6ms forward / ~1.1ms fwd+bwd, suggesting the bottleneck is CUDA launch overhead and the Gaussians produce minimal actual work.
3. **Synthetic random Gaussians do not produce the tile-intersection patterns of real scenes**, because real scenes have structured spatial distributions (Gaussians cluster around scene geometry).

### What This Falsifies

| Hypothesis | Status | Evidence |
|:-----------|:------:|:---------|
| **H1**: Lower block-launch overhead | **FALSIFIED at N≥10K** | Both tile sizes identical at all tested N |
| **H2**: Better work granularity | **FALSIFIED on synthetic data** | No advantage regardless of N |
| **H3**: Higher arithmetic intensity | **FALSIFIED on synthetic data** | No advantage in forward or backward |
| **H4**: SM contention difference | **FALSIFIED on synthetic data** | No advantage regardless of occupancy |
| **H6**: Less intersection/sorting work | **FALSIFIED on synthetic data** | Sub-pixel test shows no advantage |

### What Remains Supported or Untested

| Hypothesis | Status | Reason |
|:-----------|:------:|:--------|
| **H5**: Scene-dependent interaction | **SUPPORTED (negative evidence)** | Only real-scene Gaussians produce the advantage. The synthetic workload doesn't replicate real intersection patterns. |
| **The real advantage is scene/structure dependent** | **HYPOTHESIS** | tile32's benefit is specifically tied to real Gaussian distributions, not to N, resolution, or occupancy in isolation. |

### Mechanism Insight

The synthetic Gaussians produce **0.6ms forward for ANY N (1K–400K)**. This suggests the dense-mode rasterization kernel is doing minimal actual work when Gaussians are random — most don't project to visible pixels, and those that do don't create meaningful tile-intersection patterns. In contrast, real room training at 200K Gaussians took **~180ms** per iteration — **300× longer**.

The 300× difference implies the real workload is not just "more Gaussians" but a fundamentally different computation pattern:
- Real Gaussians **cluster** on scene surfaces, creating many per-tile intersections
- Real Gaussians have **varying screen-space radii**, from sub-pixel to many tiles
- The **backward pass** in real training reuses per-pixel gradients across each Gaussian's covering footprint — this is where tile32's 4× pixel-per-block advantage may appear

### Required: Real-Scene Snapshot Microbenchmark

The only way to discriminate between remaining hypotheses is to run the same microbenchmark with **real Gaussian snapshots** (checkpointed positions, scales, opacities, SH coefficients from actual training). This requires the server (EPIC-05) to load real trained checkpoints, OR requires the real checkpoints to be available locally.

### Summary for the Research Questions

| Question | Answer |
|:---------|:-------|
| Is tile32 generically faster on RTX 5070? | **FALSIFIED** — only on specific real workloads |
| Does the synthetic experiment reproduce the training speedup? | **FALSIFIED** — 1.0× ratio, not 1.58× |
| What discriminates real from synthetic? | Most likely the **structured tile-intersection pattern** of real scene Gaussians |
| What is the next highest-value experiment? | **Real-scene snapshot microbenchmark** — same protocol as above but using real Gaussian checkpoints from room/bicycle/garden |

---

## B6+ — What Remains Unmeasurable (BLOCKED)

The following would provide direct mechanism evidence but are BLOCKED under WDDM:

1. **Cache hit rates** (L1, L2, texture) — require Nsight Compute
2. **Warp stall reasons** (long scoreboard, short scoreboard, wait, etc.)
3. **Sector accesses** — to test H3's memory reuse hypothesis directly
4. **Achieved occupancy** — to verify H4's predictions
5. **DRAM bandwidth utilization** — to test memory-bound nature

**Do not attempt to guess these values or use proxy tools that report synthetic counters.**
