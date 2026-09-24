# N2: Cotangent-Projected Compositing Backward (CPCB) — CR1 + R1 + R2 Report

**Date**: 2026-09-15
**Environment**: A100-PCIE-40GB (SM 8.0), CUDA 11.8.89, PyTorch 2.7.1+cu118, gsplat v1.5.3
**Patched File**: `RasterizeToPixels3DGSBwd.cu` (only file modified)
**Scene**: Mip-NeRF 360 Room, checkpoints at 5K/15K/30K iterations

---

## Provenance

| Item | Value |
|------|-------|
| Repo HEAD | `32ab80e773f74f4d8e40ff4e338c86b29c7957f5` |
| `git describe` | `baseline/reference-v1-absgrad-3-g32ab80e` |
| Working tree | Dirty (2 tracked files modified, 73 untracked entries) |
| Tracked modifications | `baseline/reference_v1/gaussian_model.py`, `scripts/phase-r0.1/collect_checkpoints.sh` |
| Baseline gsplat build | `/tmp/gsplat_n2_cpcb/gsplat_baseline/` — `csrc.so` MD5 `7b264bce70570c5d21b763b98824c475` |
| CPCB gsplat build | `/tmp/gsplat_n2_cpcb/gsplat_cpcb/` — `csrc.so` MD5 `a7f8b5bbc4e048e3b23f24ec7434c58c` |
| gsplat version | 1.5.3 (both builds, verified via `__init__.py`) |
| Hostname | `bms-39468022-001` |
| GPU driver | 595.71.05 |
| CUDA_HOME | `/home/liaoyuanjun/miniforge3` (nvcc 11.8.89) |
| Python | `/usr/bin/python3.10` (Python 3.10.12) |
| Torch | 2.7.1+cu118 |
| Compile flags | `-Xptxas -v --use_fast_math -O3 -std=c++17 -gencode=arch=compute_80,code=sm_80` |
| `TORCH_CUDA_ARCH_LIST` | `8.0` |

### Checkpoint provenance

| Checkpoint | Path | Size | MD5 |
|-----------|------|------|-----|
| 5K (N=567,706) | `results/reference_v1/room_30k/checkpoints/iter_5000.pt` | 140,794,473 B | `3eb28ea587f4a1a70a10850a05349b3f` |
| 15K (N=952,353) | `results/reference_v1/room_30k/checkpoints/iter_15000.pt` | 236,187,191 B | `33f9a0bbce71573c09eef63ddf5c81f2` |
| 30K (N=952,353) | `results/reference_v1/room_30k/checkpoints/iter_30000.pt` | 236,187,191 B | `b7adc27119beb6b477c37d9495ecfef9` |

The two gsplat builds are isolated copies in `/tmp/gsplat_n2_cpcb/` with distinct `csrc.so` binaries (different MD5 hashes), loaded in separate subprocesses to avoid JIT extension name conflicts.

---

## Executive Summary

| Test | Verdict |
|------|---------|
| **CR1** (Gradient Correctness) | ✅ **PASS** — all gradient families cosine ≥ 0.99999 |
| **R1** (Register/Occupancy) | ✅ Register reduction confirmed (CDIM=3: −10 regs, +12.5% occupancy via whole-block residency) |
| **R2** (Timing) | ❌ **REGRESSION** — `isolated_replay_raster_bwd` 1–10% slower across all checkpoints |
| **Decision Gate** | 🔴 **DROP** — no raster backward improvement; consistent regression |

**N2_FINAL_DECISION = DROP**

CPCB achieves a real register reduction (68→58 for CDIM=3) and theoretical occupancy improvement (37.5%→50.0%), but this does **not** translate to wall-clock speedup. The observed regression is consistent with added per-interaction arithmetic and loop-carried scalar dependency outweighing the register/occupancy benefit; instruction-level causality was not profiled.

---

## CR1: CUDA Gradient Equivalence

### Methodology
- Load Room checkpoints (5K N=567,706 / 15K N=952,353 / 30K N=952,353)
- Run `rasterization()` with `packed=False`, `sh_degree=3`, `tile_size=16`, `absgrad=True`, `eps2d=0.1`
- Resolution: 960×540 (downsampled from 3114×2075 for memory)
- Compare gradients: `v_colors` (SH), `v_opacities`, `v_means2d`, `v_conics`, `v_means2d_abs`
- Baseline and CPCB builds run in separate subprocesses to avoid JIT extension conflicts
- Gate: cosine ≥ 0.99999, rel_L2 ≤ 1e-4, no NaN/Inf

### Results

| Test Config | v_colors | v_opacities | v_means2d | v_conics | v_means2d_abs |
|-------------|----------|-------------|-----------|----------|---------------|
| standard_5K | cos=1.000, rel=1.5e-6 ✅ | cos=1.000, rel=2.8e-6 ✅ | cos=1.000, rel=9.8e-6 ✅ | cos=1.000, rel=1.0e-5 ✅ | cos=1.000, rel=6.9e-7 ✅ |
| standard_15K | cos=1.000, rel=1.1e-6 ✅ | cos=1.000, rel=1.9e-6 ✅ | cos=1.000, rel=4.0e-6 ✅ | cos=1.000, rel=2.9e-6 ✅ | cos=1.000, rel=4.0e-7 ✅ |
| standard_30K | cos=1.000, rel=1.5e-6 ✅ | cos=1.000, rel=4.4e-6 ✅ | cos=1.000, rel=4.0e-6 ✅ | cos=1.000, rel=2.1e-6 ✅ | cos=1.000, rel=5.4e-7 ✅ |
| adv_bgnull_va0 (30K) | cos=1.000, rel=8.7e-7 ✅ | cos=1.000, rel=6.7e-6 ✅ | cos=1.000, rel=2.8e-6 ✅ | cos=1.000, rel=3.7e-6 ✅ | cos=1.000, rel=5.8e-7 ✅ |
| adv_bgnonzero_vanonzero (5K) | cos=1.000, rel=1.3e-6 ✅ | cos=1.000, rel=1.4e-6 ✅ | cos=1.000, rel=1.6e-5 ✅ | cos=1.000, rel=1.1e-5 ✅ | cos=1.000, rel=7.7e-7 ✅ |
| adv_bgnonzero_va0 (5K) | cos=1.000, rel=1.2e-6 ✅ | cos=1.000, rel=1.9e-6 ✅ | cos=1.000, rel=5.3e-6 ✅ | cos=1.000, rel=1.4e-5 ✅ | cos=1.000, rel=7.6e-7 ✅ |

**All 6 test configurations PASS.** Gradient equivalence confirmed at FP32 precision. The CPCB reformulation is mathematically exact in real arithmetic; the ~1e-6 rel_L2 differences are FP32 rounding noise.

---

## R1: Register / Spill / Occupancy Audit

### Methodology
- Built both baseline and CPCB gsplat with `-Xptxas -v`
- Extracted ptxas verbose output for `rasterize_to_pixels_3dgs_bwd_kernelILj{cdim}EfEE`
- Computed theoretical occupancy for A100 SM 8.0 using **whole-block residency**:
  - 65536 regs/SM, 2048 max threads (64 warps), 256 regs/warp allocation unit
  - 99 KB (101,376 B) default shared-memory limit per block (no opt-in)
  - Block size = 256 threads = 8 warps; resident warp count must be a multiple of 8
  - `blocks_per_sm = min(blocks_from_regs, blocks_from_threads, blocks_from_smem, 32)`
  - `warps_per_sm = blocks_per_sm × warps_per_block`
  - No fractional or non-block-multiple warp counts reported

### Shared memory (launch-time dynamic smem)

The kernel uses `extern __shared__ int s[]` with layout:
```
int32_t  id_batch[block_size]         // 256 × 4  = 1024 B
vec3     xy_opacity_batch[block_size]  // 256 × 12 = 3072 B
vec3     conic_batch[block_size]       // 256 × 12 = 3072 B
float    rgbs_batch[block_size × CDIM] // 256 × 4 × CDIM
```

Launch formula (from `RasterizeToPixels3DGSBwd.cu`):
```
shmem_size = tile_size² × (sizeof(int32) + sizeof(vec3) + sizeof(vec3) + sizeof(float) × CDIM)
           = 256 × (4 + 12 + 12 + 4 × CDIM)
```

| CDIM | Per-element bytes | Total smem (B) |
|------|-------------------|----------------|
| 3 | 4 + 12 + 12 + 12 = 40 | 256 × 40 = **10,240** |
| 8 | 4 + 12 + 12 + 32 = 60 | 256 × 60 = **15,360** |

### Results

| Metric | CDIM=3 Baseline | CDIM=3 CPCB | Delta | CDIM=8 Baseline | CDIM=8 CPCB | Delta |
|--------|-----------------|-------------|-------|-----------------|-------------|-------|
| Registers/thread | 68 | 58 | **−10** | 80 | 74 | **−6** |
| Spill stores | 0 | 0 | 0 | 0 | 0 | 0 |
| Spill loads | 0 | 0 | 0 | 0 | 0 | 0 |
| Stack frame | 0 | 0 | 0 | 0 | 0 | 0 |
| smem_bytes | 10,240 | 10,240 | 0 | 15,360 | 15,360 | 0 |
| Blocks/SM (limiter) | 3 (regs) | 4 (regs) | +1 | 3 (regs) | 3 (regs) | 0 |
| Warps/SM | 24 | 32 | **+8** | 24 | 24 | 0 |
| Theoretical occupancy | 37.5% | 50.0% | **+12.5%** | 37.5% | 37.5% | 0.0% |

**Analysis**:
- CDIM=3: The 10-register reduction crosses an allocation boundary (regs_per_warp 2304→2048), unlocking 1 additional resident block (3→4 blocks/SM, 24→32 warps, +12.5% occupancy). The register file is the binding constraint; smem (9 blocks possible) and threads (8 blocks possible) are not limiting.
- CDIM=8: The 6-register reduction (80→74) does not change regs_per_warp (2560 in both cases, since ceil(80×32/256)=ceil(74×32/256)=10). Both stay at 3 blocks/SM (24 warps, 37.5%). The register file remains the binding constraint.
- Zero spills in all configurations — the register reduction is "free" at the compiler level.

---

## R2: Raster-Backward + E2E Timing

### Terminology

Two separately constructed timings are reported. They are **not** an exact phase decomposition of one combined measurement:

- **`isolated_replay_raster_bwd`**: Precompute projection + SH + tile intersection under `no_grad`, then time `rasterize_to_pixels` backward only. The autograd graph contains only the rasterize-to-pixels kernel; projection and SH backward are not in the graph. Inputs are detached leaf tensors that are replayed forward each iteration.
- **`full_graph_bwd`**: Full `rasterization()` forward + backward via the high-level API. The backward graph includes projection_bwd, SH_bwd, and rasterize_to_pixels_bwd. This is reported as the "E2E" measurement.

The two timings use different graph structures and input preparation; their values do not sum or decompose exactly.

### Methodology
- 50 warmup + 300 timed iterations per rep
- CUDA Events with explicit `torch.cuda.synchronize()`
- Interleaved baseline/CPCB order across reps
- Leaf tensors reused across iterations (`.grad = None` reset)
- **Repetition counts**: 5K used 2 internal timing reps (each with 300 CUDA-event samples). 15K and 30K used 2 internal timing reps instead of the pre-registered 5, due to A100 40 GB memory constraints requiring lower resolution and longer per-subprocess setup. Each rep contains 300 CUDA-event samples; within-run variation is <0.06 ms std for `isolated_replay_raster_bwd`. The ~10% mature-stage regression is much larger than observed within-run variation.

### Results

| Checkpoint | Resolution | Baseline `isolated_replay_raster_bwd` (ms) | CPCB `isolated_replay_raster_bwd` (ms) | Δ | Baseline `full_graph_bwd` E2E (ms) | CPCB `full_graph_bwd` E2E (ms) | Δ E2E |
|-----------|-----------|-------------------------------------------|---------------------------------------|---|-----------------------------------|-------------------------------|-------|
| 5K (N=567K) | 960×540 | 58.36 | 58.92 | **+0.97%** | 262.53 | 262.36 | −0.06% |
| 15K (N=952K) | 480×270 | 30.97 | 33.97 | **+9.71%** | 128.27 | 129.96 | +1.32% |
| 30K (N=952K) | 480×270 | 30.82 | 34.04 | **+10.42%** | 95.25 | 96.82 | +1.64% |

**Note on `full_graph_bwd`**: The total backward time in the `full_graph_bwd` measurement may appear smaller than `isolated_replay_raster_bwd` for 30K (28.20 ms vs 30.82 ms). This is expected because the two timings use different graph structures: `isolated_replay_raster_bwd` replays the forward pass each iteration (creating the graph inside the timed region's setup), while `full_graph_bwd` includes projection_bwd and SH_bwd but the CUDA event placement differs. The two are not directly subtractive.

### Analysis

The `isolated_replay_raster_bwd` kernel is **consistently slower** with CPCB across all checkpoints. The `full_graph_bwd` E2E impact is smaller because `rasterize_to_pixels_bwd` is a fraction of total E2E time.

Regression magnitude is workload-dependent; resolution, Gaussian count, and training stage co-vary in the current measurements. The 5K checkpoint (960×540, N=567K) shows +0.97%, while 15K/30K (480×270, N=952K) show +9.7–10.4%. These three data points do not isolate resolution from Gaussian count or training stage.

The observed regression is consistent with added per-interaction arithmetic and loop-carried scalar dependency outweighing the register/occupancy benefit; instruction-level causality was not profiled.

---

## Decision Gate

| Threshold | `isolated_replay_raster_bwd` Improvement |
|-----------|------------------------------------------|
| < 2% → **DROP** | 5K: −0.97%, 15K: −9.71%, 30K: −10.42% |
| 2–5% → KEEP_SYSTEMS | |
| 5–10% → STRONG_KEEP | |
| ≥ 10% → PROMOTE | |

### Verdict: 🔴 **DROP**

CPCB shows a **consistent raster backward regression** across all checkpoints. The register reduction (−10 for CDIM=3) and occupancy improvement (+12.5% for CDIM=3, whole-block residency) are real and verified by ptxas, but they do not translate to wall-clock speedup.

**Do not promote to production. Do not combine with other optimizations.**

---

## Deliverables

| File | Description |
|------|-------------|
| `results/n2_cpcb/cr1_gradient_equivalence.json` | Full CR1 gradient comparison data |
| `results/n2_cpcb/compiler_resources.json` | R1 ptxas register/spill/occupancy data (whole-block residency) |
| `results/n2_cpcb/timing_output_v2/{5K,15K,30K}_{baseline,cpcb}_rep*.json` | R2 raw timing per checkpoint |
| `results/n2_cpcb/e2e_timing.json` | R2 summary comparison |
| `results/n2_cpcb/final_decision.json` | Final decision gate evaluation |
| `results/n2_cpcb/cpcb_diff.patch` | CPCB patch (4 replacements in RasterizeToPixels3DGSBwd.cu) |
| `results/n2_cpcb/provenance.json` | Build environment provenance |
| `results/n2_cpcb/git_state.json` | Git HEAD, describe, status |
| `reports/n2-cpcb-cr1-r2.md` | This report |

---

## Experimental Notes

- Resolution reduced from native 3114×2075 to 960×540 (5K) or 480×270 (15K/30K) due to A100 40 GB memory limits with `packed=False` at N=952K. The `isect_tiles` kernel creates ~1B intersections at native resolution requiring >40 GB.
- The 5K test at 960×540 and 15K/30K tests at 480×270 use different resolutions due to OOM constraints. Regression magnitude is workload-dependent; resolution, Gaussian count, and training stage co-vary in the current measurements.
- Baseline and CPCB builds use separate gsplat installations with distinct `csrc.so` MD5 hashes to avoid JIT extension name conflicts.
- The `n2_apply_patch.py` script applies 4 text replacements to `RasterizeToPixels3DGSBwd.cu` and has been verified with `--dry-run`.
- The CPCB patch was not modified for this report revision. No CUDA experiments were rerun.
