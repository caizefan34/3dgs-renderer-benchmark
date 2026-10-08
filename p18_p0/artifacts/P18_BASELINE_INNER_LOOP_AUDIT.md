# P18-P0 — Frozen C0 Raster Inner-Loop Audit

Protocol section 2: audit BEFORE writing any QSCAN CUDA. Gates `P18_MAPPING_BLOCKER`.

## Verdict

**MAPPING_OK.** A contiguous subgroup's lanes process the **same Gaussian on the same image row** at the same raster-loop iteration, in both forward and backward. The quotient reconstruction is feasible on subwarps of 4/8/16 contiguous x-pixels. **`P18_MAPPING_BLOCKER` is NOT triggered.** No rasterizer redesign is required.

## 1. Gaussian exponent expression

### Forward — `MacroTileRasterize.cu` (half, exp2 domain)
Conic stored in smem as `half2[2]` = raw `covar2d_inv` scaled by `CHOL_SCALE²` with `-log2(opac)` fused into the c₁₁ slot, where `CHOL_SCALE = 0.84932180028·TILE_SIZE = √(log2e/2)·TILE` (line 212).

```
u0_base = l00·dx                    (line 308, dx = mean_x − offset_x; varies in x)
u0      = l10·dy + u0_base          (line 314, per pixel-pair pp)
u1      = l11·dy                    (line 315)
sigma   = u1² + (u0² + neg_log_opac)(line 317)
alpha   = __h2exp2_approx(-sigma)   (line 322)   == exp2(−sigma)
```
`alpha(x) = G(x) = exp2(−(a x² + b x + c))`, a 1-D quadratic in x (dy, u1 constant across the row). So `r_j = G(x_{j+1})/G(x_j)`, `k = exp2(−dd)`, `dd = 2a` (constant). Matches protocol §3 in the exp2 domain.

### Backward — `HigsNativeBackward.cu` (PX=2) + `RasterizeToPixels3DGSDevice.cuh` (float, natural domain)
Conic is the **raw** `covar2d_inv` (producer `GatherlessProjectedProducer.cu:100` emits `conics = covar2d_inv`), NOT CHOL-scaled:
```
sigma = 0.5·(c00·dx² + c22·dy²) + c11·dx·dy   (line 47)
vis   = __expf(-sigma)                          (line 48)   == exp(−sigma)
alpha = min(0.99, opac·vis)                     (line 49)
valid = !(sigma < 0 || alpha < 1/255)           (line 53)
```
The weight is **recomputed** in backward (not stored). QSCAN must recompute it via seeds + prefix scan. Forward (CHOL-scaled half) and backward (raw float) conics are related by the constant `CHOL_SCALE²`; both are 1-D quadratics in x, so the same QSCAN machinery applies, each kernel using its own exp intrinsic.

## 2. Actual exp intrinsic / function
- **Forward:** `__h2exp2_approx` (half exp2) — fast-math `exp2f` equivalent. QSCAN forward seeds MUST use the same.
- **Backward:** `__expf` (float exp, fast-math). QSCAN backward seeds MUST use the same.
- Built with `-use_fast_math` (c0_ext `build.ninja`), so both take the approximate path. Using the matching intrinsic for seeds keeps seed semantics identical to baseline (§3).

## 3. Compiler flags (c0_ext build.ninja)
`nvcc=/root/miniforge3/envs/gsplat/bin/nvcc`, `--gencode=arch=compute_80,code=sm_80`, `-use_fast_math --expt-relaxed-constexpr -O3 -DNDEBUG -std=c++20 -fopenmp -DAT_PARALLEL_OPENMP -fPIC`.

## 4. Alpha thresholds
| constant | value | role |
|---|---|---|
| `ALPHA_THRESHOLD` | `1/255` (Common.h:97) | **semantics-changing per-pixel branch**: forward `ok_mask = sigma < log2(255)`; backward `valid = !(sigma<0 \|\| alpha<AT)` |
| `MAX_ALPHA` | `0.99` (Common.h:105) | backward `alpha = min(0.99, opac·vis)` |
| `MIN_TRANSPARENCY` / `TRANSMITTANCE_THRESHOLD` | `1e-4` | warp-wide early-exit on transmittance T (NOT a per-pixel alpha branch) |

γ = 2×10⁻⁶ (fixed). If a QSCAN alpha is within γ of a branch threshold (AT, or the `sigma<0` / `MAX_ALPHA` bounds), that lane falls back to the **DIRECT** expression and re-decides the branch. γ=0 only as supplementary sensitivity.

## 5. Tile size
`TILE_SIZE = 16` (trainer lines 380, 741–744; `rasterize_gaussian_higs_dynamic` default line 3002). Macro-tile = 8×4 = 32 tiles. `FUSED_GAUSS_BATCH_SIZE = 1024`.

## 6. Thread → pixel / warp → row mapping

**Forward** (`macro_tile_rasterize_kernel<16>`, CTA 320 / 10 warps, `GAUSS_PER_WARP=1`, `PIXELS_PER_THREAD=8`, `N_PAIRS=4`):
- `thread_x = lane_id & 15`, `thread_y = lane_id >> 4` (lines 273–274).
- Each lane owns **8 contiguous x-pixels in one row** (pairs pp=0..3 at rows `thread_y + 2pp`, x = `thread_x + 2pp` and `thread_x + 2pp + 1`).
- For a fixed gaussian + row, the warp's lanes cover **32 contiguous x-pixels** of that row. → 1-D quadratic in x.
- QSCAN groups: `QSCAN4`/`8`/`16` = contiguous x-runs of 4/8/16 lanes of one row.

**Backward** (`higs_blend_bwd_px_kernel<3,2,true,true>`, 128 threads/tile, PX=2, `PX_ROWS=8`):
- `i = i0 + ty + q·8`, `j = j0 + tx` (lines 397–398), q∈{0,1}.
- A warp = 16 x-columns × 2 y-rows; each lane owns 2 pixels (same tx, rows ty and ty+8). For a fixed row, **16 contiguous x-lanes** share the gaussian. → 1-D quadratic in x.
- QSCAN16 = the 16 x-lanes of one row; QSCAN8/4 = subgroups.

## 7. Shared Gaussian load path
- **Forward:** phase 1 loads 32-gaussian mini-batches from global into persistent smem (`smem_xy_mt`, `smem_conic_raw`), builds per-tile visibility masks via a 5-stage warp bit-transpose. Phase 2 streams gaussian indices through a ring buffer and calls `rasterizeGaussian` (1 smem load per gaussian, invariant across the 4 pairs). → QSCAN can compute `k = exp2(−dd)` **once per gaussian** (per-gaussian setup before the N_PAIRS loop) and broadcast via shuffle.
- **Backward:** per-batch each thread fetches 1 gaussian (id, xy, opac, conic, rgb) into smem; `block.sync`; the warp iterates `t` over the batch reading `conic_batch[t]`/`xy_opacity_batch[t]`. → QSCAN computes seeds + k per (gaussian t, row) inside the t-loop.

## 8. Backward Gaussian-weight path
Recomputed per (lane, q-pixel, gaussian t) via `eval_gaussian_weight` (line 489). Forward weights are NOT stored/reused. **P18 backward MUST recompute via QSCAN** (protocol forbids storing all Gaussian-pixel weights in HBM). Downstream `v_sigma = −opac·vis·v_alpha` (SCALAR_ADJOINT/H8_MR) is unchanged; only the `vis`/`alpha` source changes.

## 9. P18_MAPPING_BLOCKER
**Not triggered.** Both forward (TILE=16, GPW=1) and backward (PX=2) satisfy: contiguous subgroup lanes + same gaussian + same row + same loop iteration + contiguous x. `q(x)` is 1-D quadratic in x. The only change is per-pixel exp → seed + warp/subwarp **exclusive multiplicative prefix scan** (pixel-parallel, no lane-0 serial walk).

### Caveats
1. Forward row = 32 x-pixels (2 per lane); QSCAN group must stay on one contiguous x-run of one row.
2. Forward exp = half `exp2`; backward exp = float `exp` — each kernel uses the matching intrinsic.
3. `GAUSSIAN_EXTEND=3.33` culling bounds `|x|` per gaussian ⇒ bounded quadratic domain ⇒ well-conditioned 16-step multiplicative scan.
4. The per-gaussian alpha-cull branch (`ok_mask`) and the warp-wide early-exit (`tile_done`/`MIN_TRANSPARENCY`) are preserved; QSCAN changes only how alpha is computed, not when the branch fires (γ guard handles near-threshold lanes).

## 10. Frozen binaries (verified on A100 EPIC-05)
- trainer `675b0267…` `/root/epic05_p12/publication_trainer_p12.py`
- c0_ext `d26cccf4…` `/root/epic05_build_cache/c0_ext/…/experimental_gaussian_render_inference_scene_cuda.so`
- core `935ea6c0…` `/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so`
- fused_ssim `dd6a4c79…` `/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so`
- worktree `/root/epic05_frozen/higs_c0_final30k_worktree`
- ckpts 16k: room `17c1f590…`, bicycle `0c36f5c7…`
- timing GPU: A100-SXM4-80GB GPU 3 (`GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`, clean). Driver 592.01, CUDA 13.1. Local machine (RTX 5070 Laptop) is **not** used for any P18 timing.

## 11. P17 provenance
P17-P0 (multirate gradient oracle) is a **different candidate**, terminated `P17_P0_HARD_DROP`. Its camera set uses N/12 (bicycle N=194: `[8,24,40,56,72,88,105,121,137,153,169,185]`), which does **not** satisfy P18's N/8 count ⇒ P18 recomputes its own 8-camera set and records provenance. P17's frozen binaries are the same four hashes P18 freezes (verified).
