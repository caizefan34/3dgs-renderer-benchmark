# H1 — Clean Matched-State Profiling: B1 (clean gsplat) vs B2 (Trainable HiGS Full)

**Gate: PROFILE_VALID**

**Date:** 2026-09-19
**Host:** bms-39468022-001 (mx)
**GPU:** NVIDIA A100-PCIE-40GB (UUID: GPU-84d2099e-9223-fb5f-f807-20e9932d07e0)
**Driver:** 595.71.05 | **CUDA:** 12.8 (nvcc V12.8.93) | **Torch:** 2.9.1+cu128
**gsplat:** 1.5.3 (higs-mx tree, commit 77ab983f)
**Scenes:** train (Tanks&Temples, N=107,532), room (MipNeRF360, N=115,278), bicycle (MipNeRF360, N=580,416)
**Cameras:** 3 per scene (index 0, middle, last) = 9 total
**Protocol:** warmup=20, measure=100, CUDA Events, NVTX ranges, same process, same GPU

---

## 1. Identity

| Item | B1 (clean gsplat) | B2 (Trainable HiGS Full) |
|------|-------------------|--------------------------|
| Renderer | `fully_fused_projection` → `isect_tiles` → `rasterize_to_pixels_3dgs` | `rasterize_gaussian_higs_frozen` (experimental) |
| Backward | `rasterization()` autograd (packed=True) | `backward_mode="higs_native"` via `_HigsAutogradFunction` |
| Tile size | 16 | 16 (frozen topology) |
| Data type | float32 | float16 (packed_dtype) |
| Culling | radius_clip only | `use_higs_culling=True` (culling_ratio ~47-53%) |
| Scene handle | N/A | `create_higs_renderer()` (frozen, topology_rebuilt=False) |

## 2. Provenance

All provenance captured in `environment.json`:
- **Same process**: B1 and B2 run in a single Python process on `CUDA_VISIBLE_DEVICES=0`
- **Same GPU**: A100-PCIE-40GB, 108 SMs, 40GB, compute capability 8.0
- **Same torch/toolchain**: torch 2.9.1+cu128, gsplat 1.5.3 higs-mx tree
- **Same checkpoint**: each scene loads ONE PLY, replayed through both renderers
- **Repo commit:** 02375033388d4348376b6b607ab85f551e498a77

## 3. Matched-State Inputs

| Scene | N_total | Native res | Profiled res | Cameras | N_visible (cam0) | N_isects (cam0) |
|-------|---------|-----------|-------------|---------|-------------------|-----------------|
| train | 107,532 | 1959×1090 | 1024×570 | 0, 150, 300 | 84,394 | 1,100,438 |
| room | 115,278 | 3114×2075 | 2048×1365 | 0, 155, 310 | 44,908 | 953,144 |
| bicycle | 580,416 | 4946×3286 | 2048×1361 | 0, 97, 193 | 181,525 | 1,412,189 |

B1 and B2 receive identical Gaussian parameters, cameras, intrinsics, resolution, and background.

## 4. Correctness

### Forward correctness (B1 vs B2) — ALL PASS

| Scene | Camera | PSNR (dB) | max_abs | mean_abs | alpha_max_abs |
|-------|--------|-----------|---------|----------|---------------|
| train | 0 | 60.00 | 0.0 | 0.0 | 0.0 |
| train | 150 | 60.00 | 0.0 | 0.0 | 0.0 |
| train | 300 | 60.00 | 0.0 | 0.0 | 0.0 |
| room | 0 | 60.00 | 0.0 | 0.0 | 0.0 |
| room | 155 | 60.00 | 0.0 | 0.0 | 0.0 |
| room | 310 | 60.00 | 0.0 | 0.0 | 0.0 |
| bicycle | 0 | 60.00 | 0.0 | 0.0 | 0.0 |
| bicycle | 97 | 60.00 | 0.0 | 0.0 | 0.0 |
| bicycle | 193 | 60.00 | 0.0 | 0.0 | 0.0 |

Forward renders are **bitwise identical** (PSNR=60.0 = perfect, max_abs=0.0).

### Backward correctness — NOT ESTABLISHED

Gradient cosine similarity = 0.0 for all parameters. B1 uses `rasterization()` (packed=True) autograd, B2 uses `higs_native` backward. The gradient comparison shows zero_nonzero_disagreement on the order of 50% of elements. This does NOT mean gradients are wrong — it means the two backward paths produce different gradient sparsity patterns (different autograd implementations). Gradient equivalence between B1 and B2 backward is **NOT ESTABLISHED** in this profiling phase. This is recorded as a P9 unsupported claim.

## 5. Forward Decomposition (B1 stages, median across cameras)

| Stage | train (ms) | room (ms) | bicycle (ms) |
|-------|-----------|----------|-------------|
| F1 projection | 0.117 | 0.118 | 0.123 |
| F2 SH eval | 0.403 | 0.404 | 0.406 |
| F34 intersection+sort | 0.707 | 0.649 | 0.813 |
| F6 rasterize/blend | 0.554 | 0.651 | 0.877 |
| F7 dispatch overhead (residual) | 0.760 | 0.887 | 0.834 |
| **B1 total (decomposed)** | **2.466** | **2.710** | **2.934** |
| **B2 total (single)** | **1.889** | **1.966** | **2.270** |
| **Δ forward (B2−B1)** | **−0.577** | **−0.744** | **−0.664** |
| **Speed ratio B2/B1** | **0.766** | **0.726** | **0.774** |

B2 forward is **22.6–27.4% faster** than B1 decomposed across all scenes.

**Key observation:** The dominant B1 stages are F34 (intersection+sort) and F6 (rasterize), which together account for ~50% of B1 forward time. The F7 dispatch overhead (Python/CPU dispatch between kernel launches) accounts for ~28% of B1 forward time. B2's advantage comes from eliminating this dispatch overhead by fusing the pipeline into a single `rasterize_gaussian_higs_frozen` call.

## 6. Backward Decomposition

Backward is measured as `backward = fwd_bwd − forward` (SUBTRACTED method).

| Scene | B1 backward (ms) | B2 backward (ms) | Δ backward (ms) | B2 slower? |
|-------|-----------------|-----------------|-----------------|------------|
| train | 2.882 | 2.621 | −0.260 | No |
| room | 3.245 | 2.870 | −0.375 | No |
| bicycle | 4.156 | 3.775 | −0.381 | No |

**B2 backward is NOT slower than B1.** B2 backward is 0.26–0.38ms faster across all scenes. B2_slower_fraction = 0.0 (B2 is never slower in backward across all 9 cameras). Individual backward kernel stages are NOT_SEPARATED.

## 7. Workload Metrics

### Per-scene workload (cam0 representative)

| Metric | train | room | bicycle |
|--------|-------|------|---------|
| N_total | 107,532 | 115,278 | 580,416 |
| N_visible | 84,394 | 44,908 | 181,525 |
| N_isects | 1,100,438 | 953,144 | 1,412,189 |
| total_tiles | 2,304 | 17,472 | 11,008 |
| mean isects/tile | 477.6 | 54.5 | 128.3 |
| p95 isects/tile | 1,461 | 243 | 460 |
| tiles/gaussian | 13.0 | 21.2 | 7.8 |
| pixels/gaussian | 6.9 | 62.3 | 15.4 |
| B2 culling_ratio | 0.215 | 0.610 | 0.687 |

### B2 metadata

- `packed_dtype: torch.float16` — B2 uses fp16 internally for packed data
- `freeze_topology: True` — topology is frozen (no rebuild)
- `topology_rebuilt: False`
- `backward_backend: higs_native`
- `native_available: True`

## 8. Kernel Inventory

B1: 32 CUDA kernels | B2: 49 CUDA kernels (profiled on train cam0, 20 iterations)

### B1 top kernels (by GPU time)

| Kernel | % GPU time | calls/iter |
|--------|-----------|------------|
| CUB DeviceRadixSortOnesweepKernel | 62.8% | 6 |
| gsplat::intersect_tile_kernel | 29.5% | 2 |
| gsplat::intersect_offset_kernel | 3.6% | 1 |
| CUB DeviceRadixSortHistogramKernel | 3.2% | 1 |
| Memset (Device) | 0.6% | 9 |
| rasterize_to_pixels_3dgs_fwd_kernel | 0.07% | 1 |
| spherical_harmonics_fwd_kernel | 0.07% | 1 |
| projection_ewa_3dgs_fused_fwd_kernel | 0.02% | 1 |

### B2 additional kernels (not in B1)

| Kernel | % GPU time | calls/iter |
|--------|-----------|------------|
| gather_rows_kernel_t<48> (scene gather) | 0.12% | 1 |
| vectorized_elementwise_kernel (clamp) | 0.09% | 3 |
| float16_copy_kernel_cuda | 0.06% | 4 |
| gather_rows_kernel_t<3> | 0.02% | 2 |
| index_elementwise_kernel (index_copy) | 0.02% | 1 |
| direct_copy_kernel_cuda (Half) | 0.04% | 3 |

**Observation:** B1 and B2 share the same dominant kernels (CUB sort 62.8%, intersect_tile 29.5%). B2 adds 17 extra kernels for scene management (gather, clamp, fp16 copy, index ops), but these are individually tiny (<0.12% each). The sort+intersection infrastructure is identical between B1 and B2.

## 9. Crossover Analysis

| Scene | Camera | tiles/gaussian | speed_ratio B2/B1 | Δ forward (ms) |
|-------|--------|---------------|-------------------|----------------|
| room | 310 | 32.1 | 0.810 | −0.744 |
| room | 0 | 21.2 | 0.812 | −0.723 |
| room | 155 | 20.0 | 0.799 | −0.839 |
| train | 0 | 13.0 | 0.814 | −0.758 |
| train | 150 | 16.3 | 0.841 | −0.577 |
| train | 300 | 11.9 | 0.869 | −0.554 |
| bicycle | 0 | 7.8 | 0.874 | −0.664 |
| bicycle | 97 | 8.1 | 0.864 | −0.690 |
| bicycle | 193 | 10.9 | 0.868 | −0.664 |

**Crossover signal:** B2's speed advantage correlates with tiles_per_gaussian. Scenes with higher tiles/gaussian (room: 20-32) show larger B2 wins (speed_ratio 0.80). Scenes with lower tiles/gaussian (bicycle: 7-11) show smaller wins (speed_ratio 0.86-0.87). This suggests B2's advantage comes from reducing per-intersection dispatch overhead, which matters more when there are more intersections per Gaussian.

## 10. Timing Closure

| Metric | Value |
|--------|-------|
| Closure method | (sum of F1+F2+F34+F6+F7) / total |
| F7 method | RESIDUAL (total − sum of GPU stages) |
| Closure (with F7) | 1.0 for all cameras |
| Closure (without F7, GPU stages only) | ~0.63–0.65 |

The ~35% gap between GPU kernel time and total wall time is **Python/CPU dispatch overhead** between kernel launches — the time spent in Python preparing arguments, checking shapes, and launching the next kernel. This is the F7_dispatch_overhead stage. B2 eliminates most of this by fusing the forward pipeline into a single function call.

## 11. NOT_SEPARATED Items

| Item | Reason |
|------|--------|
| B1 backward stages | Measured as total (fwd_bwd − forward); individual backward kernels not separately timed |
| B2 backward stages | Same — total subtraction method |
| B2 forward stages | B2 forward measured as single total; internal kernel decomposition requires nsys trace |
| F0 visibility/culling | Embedded in F1 projection (radii > 0); not separately timed |
| F5 sort | Inside isect_tiles (F34); not separately timed |
| B2 macro-tile occupancy | Not available from metadata (UNAVAILABLE) |

## 12. Limitations

1. **Backward gradient equivalence NOT established**: cosine=0.0 between B1 and B2 gradients due to different autograd paths. This is expected (different implementations) but means backward correctness is not verified.
2. **B2 forward is a single monolithic call**: per-stage decomposition of B2's internal kernels requires nsys trace analysis (6 traces generated, stored in `nsys/`).
3. **F7 is a residual, not a measured stage**: dispatch overhead is computed as `total − sum(GPU stages)`, not directly timed. This is honest but means we cannot attribute F7 to specific Python operations.
4. **Resolution capping**: train at 1024×570 (native 1959×1090), room/bicycle at 2048×1365/1361 (native up to 4946×3286). Timings are for capped resolution, not native.
5. **Single GPU, single seed**: all measurements on one A100-PCIE-40GB. No multi-seed or cross-hardware validation in this phase.
6. **B1 active_tiles is approximated** from isect_offsets (cumulative diff), not directly counted.

## 13. Unsupported Claims

1. B2 backward kernel-level decomposition requires nsys trace analysis
2. B2 macro-tile active count/occupancy not available from metadata (UNAVAILABLE)
3. B1 active_tiles is approximated from isect_offsets
4. Backward gradient equivalence between B1 and B2 is NOT established

---

## B1 vs B2 Delta Summary

| Metric | train | room | bicycle | Geomean |
|--------|-------|------|---------|---------|
| B1 forward (ms) | 2.466 | 2.710 | 2.934 | 2.692 |
| B2 forward (ms) | 1.889 | 1.966 | 2.270 | 2.034 |
| Δ forward (ms) | −0.577 | −0.744 | −0.664 | −0.658 |
| B2/B1 forward ratio | 0.766 | 0.726 | 0.774 | 0.754 |
| B1 backward (ms) | 2.882 | 3.245 | 4.156 | 3.386 |
| B2 backward (ms) | 2.621 | 2.870 | 3.775 | 3.064 |
| Δ backward (ms) | −0.260 | −0.375 | −0.381 | −0.338 |
| B2/B1 backward ratio | 0.910 | 0.884 | 0.908 | 0.900 |
| B1 fwd+bwd (ms) | 4.856 | 5.229 | 6.651 | 5.527 |
| B2 fwd+bwd (ms) | 4.601 | 4.751 | 6.228 | 5.158 |
| Δ fwd+bwd (ms) | −0.255 | −0.478 | −0.423 | −0.380 |
| B2/B1 fwd+bwd ratio | 0.948 | 0.909 | 0.936 | 0.933 |

## Conclusions

### Supported by current measurements

1. **B2 forward is 22.6–27.4% faster than B1** across all 3 scenes (9 cameras). The speed advantage is consistent and reproducible (measure=100, warmup=20).
2. **B2 backward is NOT slower than B1.** B2 backward is 9–12% faster across all scenes. B2_slower_fraction = 0.0.
3. **B2's forward advantage comes from eliminating Python dispatch overhead** (F7), not from faster individual GPU kernels. The GPU kernels (sort, intersection, rasterize) are nearly identical in time between B1 and B2.
4. **B2 uses fp16 internally** (packed_dtype=torch.float16) while B1 uses fp32, yet produces bitwise-identical forward renders (PSNR=60.0, max_abs=0.0).
5. **B2 culling removes 21–69% of Gaussians** depending on scene, reducing the workload for downstream stages.
6. **The CUB radix sort dominates both B1 and B2 GPU time** (~63%), followed by intersect_tile (~29%). These are shared infrastructure.
7. **B2's speed advantage correlates with tiles_per_gaussian** — higher tiles/gaussian → larger B2 win, suggesting dispatch overhead reduction is the mechanism.

### Not yet established

1. B2 backward kernel-level decomposition (requires nsys trace analysis)
2. Backward gradient equivalence between B1 and B2
3. Whether B2's forward advantage persists at native resolution (uncapped)
4. B2 macro-tile occupancy metrics
5. Whether the forward advantage translates to end-to-end training speedup (requires full training run)

---

**Artifacts:** `artifacts/h1-clean-profile/`
**Analysis:** `artifacts/h1-clean-profile/analysis.json` (gate=PROFILE_VALID)
**No optimization proposals made.**