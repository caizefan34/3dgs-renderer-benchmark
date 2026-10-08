# P18-P0 Stage-0: C0 Baseline Inner Loop Audit

## Source Identity

All four frozen hashes cross-verified against ≥3 prior-phase records (P12→P16). **PASS**.

| Artifact | Expected SHA256 | Status |
|----------|----------------|--------|
| trainer | `675b0267...9050c5` | ✓ cross-verified (P15, P16, P13X, P14, P13Y) |
| C0 extension | `d26cccf4...a513f11` | ✓ cross-verified |
| C0 core | `935ea6c0...1ff37e` | ✓ cross-verified |
| fused SSIM | `dd6a4c79...83d8ebb` | ✓ cross-verified |

Build: `sm_80` (A100), `-O3 -DNDEBUG --use_fast_math -std=c++20`, CTA=320 (raster), CTA=128 (bwd PX=2).

---

## 1. Forward Raster Inner Loop

**Kernel:** `macro_tile_rasterize_kernel<16>` — `MacroTileRasterize.cu:192`
**Config:** TILE_SIZE=16, GAUSS_PER_WARP=1, N_PAIRS=4, CTA=320 (10 warps)

### 1.1 Gaussian Evaluation — `rasterizeGaussian` (lines 287–330)

```
sigma = u0² + u1² − log2(opac)        // fp16, fused FMA chain
  u0  = l00·dx + l10·dy               // Cholesky row 0
  u1  = l11·dy                        // Cholesky row 1
  dx  = mean_x − pixel_x              // __hsub2, half2, 2 px/lane
  dy  = mean_y − pixel_y              // __hsub2, half2, 2 px/lane
```

- **Conic source:** `smem_conic_raw[idx]` — 4×half = `{l00, l10, l11, −log2(opac)}`, scaled by `CHOL_SCALE = √(ln2/2)·16 ≈ 13.589`.
- **Exp:** `__h2exp2_approx(−sigma)` — **half2 MUFU.EX2** (hardware approximate). NOT `expf`, NOT `__expf`, NOT inline PTX.
- **Fusion:** `alpha = opac · exp2(−(u0²+u1²))` — opacity folded into the exponent via `−log2(opac)`. No separate multiply.

### 1.2 Alpha Semantics

| Quantity | Value | Source |
|----------|-------|--------|
| ALPHA_THRESHOLD | 1/255 ≈ 0.003922 | `Common.h:97` |
| sigma_thresh (log2 domain) | log2(255) = 7.9943534f | `MacroTileRasterize.cu:83` |
| MAX_ALPHA | 0.99 (forward: **not clamped**, only in bwd) | `Common.h:105` |
| MIN_TRANSPARENCY | 1e-4f | `MacroTileRasterize.cu:85` |
| TRANSMITTANCE_THRESHOLD | 1e-4f | `Common.h:106` |

**Flow per (gaussian, pixel-pair):**
1. `sigma = hfma2(u1,u1, hfma2(u0,u0, neg_log_opac))`
2. `ok = sigma < 7.994` (per-half mask)
3. `alpha = h2exp2_approx(−sigma)`; if `!ok` → alpha = 0
4. `vis = alpha × T` (T = running transmittance, `acc[pp][3]`)
5. `acc_R += R·vis; acc_G += G·vis; acc_B += B·vis`
6. `T = max(T − vis, 0)` (`__hsub2_sat`)
7. **Early exit:** if ALL 8 pixels done (`T < 1e-4`) for ALL 32 lanes → `__ballot_sync == ~0u` → break

### 1.3 Thread→Pixel Mapping (tile16, GAUSS_PER_WARP=1)

```
lane l → (thread_x = l & 15, thread_y = l >> 4)
         → 8 pixels via 4 row-pairs:
           pair pp → rows (thread_y + 2·pp, thread_y + 2·pp + 1)
```

| Lanes | Row y | Columns x |
|-------|-------|-----------|
| 0–15  | 0     | 0–15      |
| 16–31 | 1     | 0–15      |

Each lane covers rows `thread_y, thread_y+2, thread_y+4, thread_y+6` (even) and `+1,+3,+5,+7` (odd) → **8 pixels/lane × 32 lanes = 256 = 16×16 tile**. ✓

**Source:** `MacroTileRasterize.cu:269–282` (mapping), `484–491` (offset_y).

### 1.4 Raster Loop Structure (lines 523–673)

1. **Queue fill:** per mini-batch (32 gaussians), each lane checks `smem_masks[mb·32 + tile]` bit. If set → `warp_queue[write_head + popc] = mb·32 + lane`. All lanes see same mask → contiguous per-warp gaussian list.
2. **Drain:** 32-entry batches. `UNROLL_FACTOR=8`: uint4 LDS loads 8 indices → 8× `rasterizeGaussian`. Tail for <8.
3. **Early exit:** per-pair done mask → warp ballot → break if all done.

---

## 2. Backward Inner Loop

**Kernel:** `higs_blend_bwd_px_kernel<3, 2, true, true>` — `HigsNativeBackward.cu:317`
**Config:** CDIM=3, PX=2, SCALAR_ADJOINT=true, H8_MR=true, CTA=128 (4 warps)

### 2.1 Gaussian Evaluation — RECOMPUTED

```
sigma = 0.5·(A·dx² + C·dy²) + B·dx·dy    // FP32, full conic (A,B,C)
vis   = __expf(−sigma)                    // CUDA fast FP32 exp
alpha = min(0.99, opac·vis)
valid = !(sigma < 0 || alpha < 1/255)
```

**Source:** `RasterizeToPixels3DGSDevice.cuh:44–55` (`eval_gaussian_weight`).
**Key difference from forward:** backward uses **full FP32 conic** (A,B,C form) + `__expf`, NOT half2 Cholesky + `__h2exp2_approx`. Same math, different numerical path.

### 2.2 Backward Loop Structure (lines 446–583)

1. **Batch load:** 128 threads load 128 gaussians (back-to-front) into smem: `id_batch`, `xy_opacity_batch`, `conic_batch`, `rgbs_batch`. `block.sync()`.
2. **Per-gaussian iteration (t loop):** ALL 128 threads evaluate gaussian `t` at their own pixel(s):
   - `conic_batch[t]`, `xy_opacity_batch[t]` — uniform smem index `t`
   - Per-thread: `delta = mean − (px[q], py[q])` → `eval_gaussian_weight` → `rasterize_to_pixels_3dgs_blend_bwd`
3. **Warp reduction:** `warpSum(v_rgb, warp)`, `warpSum(v_conic, warp)`, `warpSum(v_xy, warp)`, `warpSum(v_opacity, warp)`
4. **Atomic scatter:** lane 0 of each warp: `atomicAdd(v_colors[3g+k], ...)`, `atomicAdd(v_conics[3g+k], ...)`, `atomicAdd(v_means2d[2g+k], ...)`, `atomicAdd(v_opacities[g], ...)`

### 2.3 Backward Thread→Pixel Mapping (PX=2)

```
warp w (w = 0..3): rows {2w, 2w+1} (and +8 for q=1)
lane l:            column j = tile_x_base + l
```

| Warp | Rows (q=0) | Rows (q=1) | Cols |
|------|-----------|-----------|------|
| 0    | 0, 1      | 8, 9      | 0–15 |
| 1    | 2, 3      | 10, 11    | 0–15 |
| 2    | 4, 5      | 12, 13    | 0–15 |
| 3    | 6, 7      | 14, 15    | 0–15 |

Each thread covers 2 pixels (q=0, q=1). 4 warps × 16 lanes × 2 px = 128 = tile area. ✓

**Difference from forward:** forward lane l → (x=l%16, y=l/16); backward warp w lane l → (x=l, y=2w+q·8). **Mappings differ** but both are "same gaussian, per-lane pixel" patterns.

---

## 3. Shared Gaussian Data Audit

### Forward (per CTA, FUSED_GAUSS_BATCH_SIZE=1024)

| Smem Array | Type | Size/Gauss | Total | Content |
|-----------|------|-----------|-------|---------|
| `smem_xy_mt` | `__half2` | 4B | 4096B | (mean_x, mean_y) in tile units, macro-tile-relative |
| `smem_conic_raw` | `uint2` | 8B | 8192B | {l00, l10, l11, −log2(opac)} × CHOL_SCALE |
| `smem_color` | `ushort4` | 8B | 8192B | {R, G, B, _} fp16 |
| `smem_masks` | `uint32` | — | 4096B | per-(mini-batch, tile) overlap masks |
| **Total** | | | **~24.6 KB** | |

**Can we add `k = exp(dd)` per gaussian?**
- **YES.** Add 2 bytes (one `__half`) per gaussian → +2048B → total ~26.6 KB. Well within 100 KB smem (sm_80).
- **Zero additional global traffic:** `dd` is already in registers during Phase 1 conic processing (it's part of the conic/opacity computation). Store `exp(dd)` alongside the conic in smem.
- Read once per gaussian in `rasterizeGaussian` (alongside `smem_xy_mt`, `smem_conic_raw`, `smem_color`).

### Backward (per CTA, block_size=128)

| Smem Array | Type | Size/Gauss | Total |
|-----------|------|-----------|-------|
| `id_batch` | `int32` | 4B | 512B |
| `xy_opacity_batch` | `vec3` | 12B | 1536B |
| `conic_batch` | `vec3` | 12B | 1536B |
| `rgbs_batch` | `float[3]` | 12B | 1536B |
| **Total** | | | **5120B** |

**Can we add `k`?** YES. +4B/gauss (float) → +512B. Negligible.

---

## 4. Lockstep Analysis (Source-Level Proof)

### Forward (tile16, GAUSS_PER_WARP=1)

**Claim:** `f_lockstep(W) = 1.0` for W ∈ {4, 8, 16}.

**Proof:**
- `MacroTileRasterize.cu:554–559`: drain loop loads `b_idx[UNROLL_FACTOR]` via `uint4` LDS from `warp_queue[base+q]`. This is a **uniform** LDS — all 32 lanes read the same address, get the same values.
- `rasterizeGaussian(b_idx[i], offset_x, offset_y, acc)` is called with the **same** `b_idx[i]` (gaussian index) for all lanes. Only `offset_x`, `offset_y` (derived from `lane_id`) differ per lane.
- **No per-lane branch on gaussian selection.** No `__shfl` for index exchange. No `if (lane < 16)` for different gaussians (that's only for GAUSS_PER_WARP=2 / tile8).
- **Early exit** (`tile_done == DONE_MASK`): triggers when ALL pixels in ALL pairs are done for ALL 32 lanes. Before that, all lanes process the same gaussian sequence. Partial saturation does NOT cause per-lane divergence.
- **Alpha threshold** (`ok_mask`): sub-threshold pixels get alpha=0 but still execute the FMA (contributing 0). No branch.

∴ Every drain-loop iteration processes the **same Gaussian ID** across all 32 lanes. `f_lockstep = 1.0` for any W ≤ 32.

### Backward (PX=2, CTA=128)

**Claim:** `f_lockstep(W) = 1.0` for W ∈ {4, 8, 16}.

**Proof:**
- `HigsNativeBackward.cu:468`: `for (t = ...; t < batch_size; ++t)` — `t` is the loop variable, **identical for all 128 threads** in the CTA.
- `HigsNativeBackward.cu:485–489`: `conic_batch[t]`, `xy_opacity_batch[t]`, `rgbs_batch[t*CDIM]` — smem loads with **uniform index t**. All threads access the same gaussian.
- Per-thread variation: only `px[q]`, `py[q]` (pixel coordinates) and `v_render_c[q]`, `v_render_a[q]`, `T_final[q]` (per-pixel adjoint state).
- `warp.any(valid)` (line 546): if ALL lanes invalid → `continue`. This is **warp-uniform** (all lanes skip together). No per-lane divergence.
- **No per-lane branch on gaussian index.** No scatter/gather within the t loop.

∴ Every t-loop iteration processes the **same Gaussian ID** across all threads. `f_lockstep = 1.0` for any W ≤ 32.

### Forward ↔ Backward Mapping Compatibility

- Forward: lane l → (x=l%16, y=l/16), 8 px/lane
- Backward: warp w, lane l → (x=l, y=2w+q·8), 2 px/thread
- **Mappings differ**, but both satisfy "same gaussian across all lanes in a warp."
- **QSCAN can reuse the same subgroup mapping** in both passes because the lockstep property is identical. QSCAN operates on per-gaussian reductions, not per-pixel coordinates.

---

## 5. QSCAN Feasibility Summary

| Width W | Forward f_lockstep | Backward f_lockstep | Verdict |
|---------|-------------------|--------------------|---------|
| 4       | 1.0               | 1.0                | ✓       |
| 8       | 1.0               | 1.0                | ✓       |
| 16      | 1.0               | 1.0                | ✓       |

**Maximum feasible W = 16.** Both forward and backward are fully lockstep at all subgroup widths. No divergence sources found in the actual kernel code.

**Shared-memory k=exp(dd) load:** feasible in both forward (+2 KB smem, 0 global traffic) and backward (+512 B smem, 0 global traffic).
