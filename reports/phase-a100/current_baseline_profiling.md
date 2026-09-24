# Current A100 Baseline Profiling Report

**Date:** 2026-09-06  
**Objective:** Establish current A100 PCIe 40GB baseline per-kernel forward timing  
**Status:** `MEASURED`

---

## 1. Environment Fingerprint

```json
{
  "host": "bms-39468022-001 (mx)",
  "gpu_model": "NVIDIA A100-PCIE-40GB",
  "gpu_count": 8,
  "compute_capability": "8.0",
  "vram_per_gpu_mib": 40441,
  "memory_clock_mhz": 1215,
  "graphics_clock_mhz": 1410 (boost),
  "power_limit_w": 250,
  "nvidia_driver": "595.71.05",
  "cuda_runtime": "11.8 (conda nvcc: 12.4.131)",
  "pytorch": "2.7.1+cu118",
  "cudnn": 90100,
  "gsplat": "1.5.3 (prebuilt wheel, JIT-compiled backend)",
  "gcc": "11.4.0 (Ubuntu)",
  "python": "3.10.19",
  "os": "Ubuntu 22.04.4 LTS (kernel 5.15.0-181-generic)"
}
```

**Full details:** `reports/phase-a100/current_environment_fingerprint.json`

---

## 2. Protocol

| Parameter | Value |
|-----------|-------|
| Timing | `torch.cuda.Event` (enable_timing=True) |
| Warmup | 30 iterations |
| Measured | 100 iterations |
| Repeats | 1 (per scene×tile combination) |
| Cooldown | 3s between scene changes, 5s within sort isolation |
| Resolution | 1920×1080 (pinhole, fov=50°) |
| SH degree | 3 (from checkpoint) |
| Camera | Single, fixed |
| Packed | True |

---

## 3. Checkpoints Used

All checkpoints are **30K-step full-trained** from the current A100 environment:

| Scene | tile_size | Gaussians | Path |
|-------|:---------:|----------:|------|
| room | 16 | 1,105,873 | `a100_30k_room_t16_16_latest.pt` |
| room | 24 | 1,157,886 | `a100_30k_room_t24_24_latest.pt` |
| bicycle | 16 | 2,589,484 | `a100_30k_bicycle_t16_16_latest.pt` |
| bicycle | 24 | 2,790,221 | `a100_30k_bicycle_t24_24_latest.pt` |
| garden | 16 | 874,019 | `a100_30k_garden_t16_16_latest.pt` |
| garden | 24 | 739,103 | `a100_30k_garden_t24_24_latest.pt` |

---

## 4. Per-Kernel Timing Results

### 4.1 Summary Table

| Scene | Tile | Gaussians | Intersections | Fwd(ms) | Proj(ms) | SH(ms) | Pass1(ms) | CUB Sort(ms) | Offs(ms) | Rast(ms) | Sort% |
|:------|:----:|----------:|--------------:|--------:|---------:|-------:|----------:|-------------:|---------:|---------:|:-----:|
| bicycle | 16 | 2,589,484 | 607,234,821 | 107.93 | 0.28 | 0.06 | 30.72 | **72.87** | 3.78 | 0.19 | 67.5% |
| bicycle | 24 | 2,790,221 | 273,067,261 | 48.61 | 0.30 | 0.05 | 13.69 | **32.60** | 1.71 | 0.22 | 67.1% |
| garden | 16 | 874,019 | 683,654,546 | 124.47 | 0.16 | 0.06 | 37.46 | **82.46** | 4.26 | 0.14 | 66.2% |
| garden | 24 | 739,103 | 293,729,512 | 53.45 | 0.14 | 0.07 | 16.25 | **35.06** | 1.84 | 0.19 | 65.6% |
| room | 16 | 1,105,873 | 174,446,870 | 31.88 | 0.16 | 0.10 | 9.69 | **20.60** | 1.10 | 0.13 | 64.6% |
| room | 24 | 1,157,886 | 77,873,444 | 14.46 | 0.16 | 0.12 | 4.25 | **9.25** | 0.51 | 0.20 | 63.9% |

**Columns:**
- `Pass1` = isect_tiles(sort=False) = intersect pass 1 + CPU cumsum + intersect pass 2
- `CUB Sort` = median(intersect_with_sort) − median(intersect_no_sort) = estimated CUB `SortPairs` time
- `Sort%` = CUB Sort / Forward total

### 4.2 Detailed Per-Kernel Breakdown

#### room tile16 (1.1M Gaussians, 174M intersections)

| Kernel | Median(ms) | Mean±Std(ms) | CV | % of Forward |
|--------|:----------:|:------------:|:--:|:-----------:|
| projection | 0.157 | 0.158±0.005 | 0.030 | 0.5% |
| sh_eval | 0.099 | 0.100±0.004 | 0.037 | 0.3% |
| intersect_pass1 (no sort) | 9.687 | 9.682±0.033 | 0.003 | 30.4% |
| **cub_sort** | **20.604** | (estimated) | — | **64.6%** |
| intersect_sort (total) | 30.395 | 30.385±0.040 | 0.001 | 95.3% |
| intersect_offset | 1.100 | 1.102±0.009 | 0.009 | 3.5% |
| rasterization | 0.131 | 0.132±0.007 | 0.057 | 0.4% |
| **FORWARD TOTAL** | **31.881** | | | **100%** |

#### room tile24 (1.2M Gaussians, 78M intersections)

| Kernel | Median(ms) | Mean±Std(ms) | CV | % of Forward |
|--------|:----------:|:------------:|:--:|:-----------:|
| projection | 0.164 | 0.166±0.005 | 0.033 | 1.1% |
| sh_eval | 0.116 | 0.117±0.004 | 0.031 | 0.8% |
| intersect_pass1 (no sort) | 4.253 | 4.252±0.017 | 0.004 | 29.4% |
| **cub_sort** | **9.247** | (estimated) | — | **63.9%** |
| intersect_sort (total) | 13.471 | 13.469±0.017 | 0.001 | 93.1% |
| intersect_offset | 0.514 | 0.515±0.006 | 0.012 | 3.6% |
| rasterization | 0.198 | 0.198±0.008 | 0.041 | 1.4% |
| **FORWARD TOTAL** | **14.462** | | | **100%** |

#### bicycle tile16 (2.6M Gaussians, 607M intersections)

| Kernel | Median(ms) | Mean±Std(ms) | CV | % of Forward |
|--------|:----------:|:------------:|:--:|:-----------:|
| projection | 0.285 | 0.290±0.033 | 0.115 | 0.3% |
| sh_eval | 0.056 | 0.060±0.007 | 0.112 | 0.1% |
| intersect_pass1 (no sort) | 30.719 | 30.686±0.088 | 0.003 | 28.5% |
| **cub_sort** | **72.873** | (estimated) | — | **67.5%** |
| intersect_sort (total) | 103.630 | 103.626±0.192 | 0.002 | 96.0% |
| intersect_offset | 3.777 | 3.774±0.028 | 0.007 | 3.5% |
| rasterization | 0.186 | 0.192±0.012 | 0.064 | 0.2% |
| **FORWARD TOTAL** | **107.935** | | | **100%** |

#### bicycle tile24 (2.8M Gaussians, 273M intersections)

| Kernel | Median(ms) | Mean±Std(ms) | CV | % of Forward |
|--------|:----------:|:------------:|:--:|:-----------:|
| projection | 0.296 | 0.297±0.005 | 0.017 | 0.6% |
| sh_eval | 0.055 | 0.057±0.006 | 0.101 | 0.1% |
| intersect_pass1 (no sort) | 13.687 | 13.688±0.016 | 0.001 | 28.2% |
| **cub_sort** | **32.600** | (estimated) | — | **67.1%** |
| intersect_sort (total) | 46.328 | 46.318±0.131 | 0.003 | 95.3% |
| intersect_offset | 1.711 | 1.713±0.011 | 0.006 | 3.5% |
| rasterization | 0.219 | 0.221±0.008 | 0.034 | 0.5% |
| **FORWARD TOTAL** | **48.609** | | | **100%** |

#### garden tile16 (0.9M Gaussians, 684M intersections)

| Kernel | Median(ms) | Mean±Std(ms) | CV | % of Forward |
|--------|:----------:|:------------:|:--:|:-----------:|
| projection | 0.156 | 0.158±0.008 | 0.049 | 0.1% |
| sh_eval | 0.065 | 0.067±0.007 | 0.105 | 0.1% |
| intersect_pass1 (no sort) | 37.459 | 37.454±0.031 | 0.001 | 30.1% |
| **cub_sort** | **82.463** | (estimated) | — | **66.2%** |
| intersect_sort (total) | 119.850 | 119.922±0.237 | 0.002 | 96.3% |
| intersect_offset | 4.259 | 4.253±0.030 | 0.007 | 3.4% |
| rasterization | 0.137 | 0.141±0.010 | 0.067 | 0.1% |
| **FORWARD TOTAL** | **124.467** | | | **100%** |

#### garden tile24 (0.7M Gaussians, 294M intersections)

| Kernel | Median(ms) | Mean±Std(ms) | CV | % of Forward |
|--------|:----------:|:------------:|:--:|:-----------:|
| projection | 0.143 | 0.146±0.006 | 0.039 | 0.3% |
| sh_eval | 0.066 | 0.067±0.004 | 0.061 | 0.1% |
| intersect_pass1 (no sort) | 16.251 | 16.244±0.049 | 0.003 | 30.4% |
| **cub_sort** | **35.059** | (estimated) | — | **65.6%** |
| intersect_sort (total) | 51.209 | 51.297±0.185 | 0.004 | 95.8% |
| intersect_offset | 1.836 | 1.836±0.007 | 0.004 | 3.4% |
| rasterization | 0.191 | 0.194±0.009 | 0.046 | 0.4% |
| **FORWARD TOTAL** | **53.446** | | | **100%** |

---

## 5. Sorting-Specific Measurements

### 5.1 Sort Key Structure (from source code analysis)

From `gsplat/cuda/csrc/IntersectTile.cu`:

```cpp
// isect_ids format: 64-bit integer
// Bits 63–32: image_id | tile_id (packed)
// Bits 31–0:  depth (bit-level reinterpret of float32)
//
// Encoding:
//   isect_id = (image_id << (32 + tile_n_bits)) | (tile_id << 32) | depth_id_enc
//
// Sort key width used by CUB:
//   begin_bit = 0
//   end_bit   = 32 + tile_n_bits + image_n_bits
```

### 5.2 Sort Parameters by Tile Size

| Parameter | tile16 | tile24 |
|-----------|:------:|:------:|
| Tile grid (W×H) | 120×68 | 80×45 |
| Number of tiles | 8,160 | 3,600 |
| `tile_n_bits` | 14 | 13 |
| `image_n_bits` | 1 | 1 |
| **Key width (total)** | **47 bits** | **46 bits** |
| Key dtype | `int64_t` | `int64_t` |
| Effective key bits used | 47/64 | 46/64 |
| CUB `begin_bit` | **0** | **0** |
| CUB `end_bit` | **47** | **46** |
| CUB sort function | `DeviceRadixSort::SortPairs` | same |
| Buffer strategy | `DoubleBuffer` | same |
| Temporary storage | Auto-allocated via `CUB_WRAPPER` | same |

### 5.3 Intersection Scaling

| Scene | Tile | Gaussians | Intersections | TPG mean | TPG median | TPG max |
|:------|:----:|----------:|--------------:|:--------:|:----------:|:-------:|
| room | 16 | 1,105,873 | 174,446,870 | 8,127 | 8,160 | 8,160 |
| room | 24 | 1,157,886 | 77,873,444 | 3,588 | 3,600 | 3,600 |
| bicycle | 16 | 2,589,484 | 607,234,821 | 5,840 | 8,160 | 8,160 |
| bicycle | 24 | 2,790,221 | 273,067,261 | 2,599 | 3,600 | 3,600 |
| garden | 16 | 874,019 | 683,654,546 | 7,950 | 8,160 | 8,160 |
| garden | 24 | 739,103 | 293,729,512 | 3,510 | 3,600 | 3,600 |

The median TPG being exactly 8160 (tile16) or 3600 (tile24) across all scenes indicates **Gaussian coverage saturates the entire image** — every visible Gaussian projects onto all tiles. This is consistent with the camera being placed at a fixed distance (-5 in Z) pointing at the center of the point cloud.

> **Note:** CUB internal radix policy (pass count, bits per pass) is `UNKNOWN` — cannot observe without `nvtx` or `ncu` profiling.

---

## 6. A100 Sorting Share

### Sort Ratio (CUB Sort / Forward)

| Scene | Tile | $$R_{sort}$$ |
|:------|:----:|:------------:|
| room | 16 | 64.6% |
| room | 24 | 63.9% |
| bicycle | 16 | **67.5%** |
| bicycle | 24 | **67.1%** |
| garden | 16 | 66.2% |
| garden | 24 | 65.6% |

**CUB SortPairs accounts for 64–68% of total forward time on A100 PCIe 40GB.**

### Sort Pipeline Ratio (pass2 + CUB sort + offset) / Forward

Note: pass2 already includes pass1 time, and `intersect_sort` (the full pipeline) already includes CUB sort. So the sort pipeline share is simply `intersect_sort + offset` divided by forward total:

| Scene | Tile | $$R_{sort-pipeline}$$ |
|:------|:----:|:---------------------:|
| room | 16 | 98.8% |
| room | 24 | 96.7% |
| bicycle | 16 | **99.5%** |
| bicycle | 24 | 98.8% |
| garden | 16 | **99.7%** |
| garden | 24 | 99.2% |

**The intersection pipeline (pass1 + cumsum + pass2 + CUB sort + offset) consumes 96–100% of forward time.**

---

## 7. Scaling Analysis

### 7.1 Tile Size Scaling (tile16 → tile24)

| Scene | Forward speedup | Intersection reduction |
|:------|:--------------:|:---------------------:|
| room | 2.20× | 2.24× |
| bicycle | 2.22× | 2.22× |
| garden | 2.33× | 2.33× |

The speedup is nearly linear with intersection count reduction, confirming that the forward pipeline is dominated by intersection work.

### 7.2 Intersections vs Sort Time (all measurements)

| Metric | Observation | Classification |
|--------|------------|:--------------:|
| $$N_{intersection}$$ vs $$T_{sort}$$ | Linear relationship: more intersections → proportionally more sort time | **MEASURED** |
| $$N_{intersection}$$ vs $$T_{forward}$$ | Nearly linear (96-99% pipeline share) | **MEASURED** |
| Intersect pass1 (no sort) correlation | Pass1 scales with intersections (b/w 27-30% of sort) | **MEASURED** |
| Memory bandwidth limitation | SortPairs on A100 likely bandwidth-limited; 47-bit key requires 2-pass CUB radix sort | **HYPOTHESIS** |
| CUB radix pass count | Cannot determine without `ncu`; 47-bit key typically requires 2-3 passes at 16 bits/pass | **UNKNOWN** |

---

## 8. GPU Hardware State

### Baseline State (idle, all GPUs)

| Metric | Value |
|--------|:-----:|
| Temperature | 32–34°C per GPU |
| Power draw | 32–46W per GPU |
| Graphics clock | 210–765 MHz (idle boost varies) |
| Memory clock | 1215 MHz (constant) |
| GPU utilization | 0% |
| Memory utilization | 0% |

### Under Load (during profiling)

| Metric | room t16 | bicycle t16 | garden t16 |
|--------|:--------:|:-----------:|:----------:|
| Max temperature | 42°C | 51°C | 54°C |
| Max power draw | 61W | 67W | 67W |
| Graphics clock | 1410 MHz | 1410 MHz | 1410 MHz |
| Clock stability | No throttling | No throttling | No throttling |

**GPU stability assessment: STABLE.** Despite 54°C peak on garden tile16 (the heaviest workload with 684M intersections), the graphics clock remained at the maximum boost of 1410 MHz throughout. No thermal throttling was detected. Power consumption stayed well within the 250W TDP.

---

## 9. Comparison with Historical Data

> **Caution:** Cross-hardware comparison is marked `HISTORICAL CONTEXT` — not a speedup claim. The old RTX 5070 data was collected on a Windows machine with different driver, PyTorch, CUDA, and gsplat build.

| Metric | RTX 5070 (historical) | New A100 (measured) | Interpretation |
|--------|:---------------------:|:-------------------:|---------------|
| Forward (room t16) | ~25–35 ms | 31.88 ms | Comparable magnitude (different GPU class) |
| CUB Sort share | ~50–65% | **64–68%** | Sort **remains primary bottleneck** on A100 |
| Intersect pass1 share | ~30–35% | ~28–30% | Consistent pass1 fraction |
| Raster share | ~1–2% | ~0.4% | Slightly lower on A100 (compute-bound stage) |
| Peak VRAM (room t16) | ~4 GB | 6,330 MiB | Higher due to larger intersections buffer |
| Intersection count scaling | Similar | Similar | Consistent behavior |

**Key finding:** The bottleneck profile has **not changed** from the RTX 5070 to the A100 PCIe. CUB `SortPairs` remains the dominant forward time contributor.

---

## 10. Bottleneck Reclassification

```
PRIMARY BOTTLENECK:
CUB DeviceRadixSort::SortPairs
  - 64.6–67.5% of forward time across all scenes
  - 607M intersections → 72.9ms sort time (bicycle t16, worst case)
  - 684M intersections → 82.5ms sort time (garden t16, worst case)
  
  Key properties (A100):
    - 64-bit key, effective width: 47 bits (tile16) / 46 bits (tile24)
    - begin_bit=0, end_bit=47 (or 46)
    - SortPairs with DoubleBuffer
    - CUB internal radix pass count: UNKNOWN (need ncu)
    - Memory bandwidth-limited operation on A100 PCIe (1.6 TB/s HBM2e)

SECONDARY BOTTLENECK:
Intersect Pass 1 (tile intersection kernel) + CPU cumsum + Pass 2
  - 28.2–30.4% of forward time
  - 30.7ms (bicycle t16) worst case
  - Partial overlap with sort preparation

NON-BOTTLENECK:
  - Projection: 0.1–1.1% — negligible
  - SH evaluation: 0.1–0.8% — negligible
  - Intersect offset encoding: 3.4–3.6% — small, but non-trivial
  - Rasterization: 0.1–1.4% — negligible for forward pass

UNKNOWN:
  - CUB internal radix sort pass count (needs NSight Compute)
  - Exact CUB temporary storage allocation size
  - Memory bandwidth utilization during sort
  - Whether sort is bound by DRAM bandwidth or computation
```

---

## 11. Output Files

- **Report:** `reports/phase-a100/current_baseline_profiling.md` (this file)
- **Data (JSON):** `results/phase-a100/current_baseline_profiling.json`
- **Environment fingerprint:** `reports/phase-a100/current_environment_fingerprint.json`

---

## 12. Final Output

```
CURRENT A100 BASELINE PROFILING COMPLETE

PRIMARY BOTTLENECK:
CUB DeviceRadixSort::SortPairs (64.6–67.5% of forward time)

SECONDARY BOTTLENECK:
Intersect tile kernel pass1 + cumsum + pass2 (28–30% of forward time)

CUB SORT SHARE:
64.6–67.5% of forward time (MEASURED)

FORWARD TIME:
31.88ms (room tile16) to 124.47ms (garden tile16) at 1920×1080

INTERSECTION COUNT:
77M (room tile24) to 684M (garden tile16) — linear with sort time

KEY WIDTH:
47 bits (tile16) / 46 bits (tile24) — int64_t key, begin_bit=0, end_bit=47/46
CUB internal pass count: UNKNOWN (requires ncu)

GPU STABILITY:
STABLE — max 54°C, no throttling (GPU clock sustained at 1410 MHz)

HISTORICAL PROFILE DIFFERENCE:
Bottleneck profile UNCHANGED from RTX 5070 — CUB SortPairs remains the
primary bottleneck on A100 PCIe. Sort share slightly higher (64-68% vs 50-65%).

NEXT RESEARCH QUESTION:
Given that CUB SortPairs accounts for ~2/3 of forward time on A100 with
47-bit keys, what fraction of sort time is spent in each CUB radix pass?
Is the 2-pass (typical for 47-bit at 32-bit/radix) or 3-pass scenario?
This requires NSight Compute profiling to answer definitively.

OPTIMIZATION:
NOT STARTED
```

---

## Data Quality Notes

- All timing data uses `torch.cuda.Event` with `enable_timing=True`
- **Warmup:** 30 iterations before measurement begins
- **Measurement:** 100 timed iterations per kernel
- **CV values:** All primary stages have CV < 0.005 (excellent consistency)
- **Sort isolation:** Separate no-sort / with-sort phases with 5s GPU cooldown to avoid thermal state differences
- **GPU state monitoring:** Temperature, power, and clock recorded per scene
- **Peak VRAM:** Recorded via `torch.cuda.max_memory_allocated()`
- **Allocation overhead:** Offsets and sorted ID buffers allocated per iteration (part of realistic profile)
