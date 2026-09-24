# S9 — Trainable HiGS Evidence Digest

> Evidence package for the "Trainable HiGS" line of research: making gsplat's
> hierarchical Gaussian-inference renderer (HiGS) end-to-end differentiable,
> verifying correctness, and measuring full-training speed/quality. All claims
> cite file paths in `C:\Users\36570\3dgs-renderer-benchmark`.

---

## Important scope note on requested terms

Two terms in the assignment brief do not appear anywhere in this repository:

- **"T_HIGS_REGRESSION"** — an exhaustive search of all files (`grep` over the
  entire tree), the evidence ledger (`evidence_search`), and the git history
  returned zero matches for `T_HIGS_REGRESSION`, `T_HIGS`,
  `TRAINABLE_HIGS`, or any HiGS-specific named regression test/finding. The
  HiGS work does discuss many measured *regressions* (reverted optimization
  experiments in `reports/higs-trainability-implementation.md` rounds 08-01i,
  08-01l, 24, 26, 27; quality regressions in the ablation matrices), but none
  is labeled `T_HIGS_REGRESSION`. Section 3 documents the closest analogous
  findings.
- **"112/117 test summary"** — no file contains "112 passed", "117 passed",
  "112/117", or "117/117". The actual HiGS test counts are **99 passed**
  (initial native-backward suite), growing to **105 → 109** as tile-sampling
  tests were added, and **272 / 276 passed** for the full repo suite. The
  numbers 112 and 117 appear in the repo only as Gaussian counts
  (e.g. SfM init 112,627 points), tile/grid sizes, and unrelated benchmark
  values — never as test totals. See Section 2.
- **"13-scene HiGS results"** — the HiGS paper matrices use **11 scenes × 3
  seeds** (210-job and 132-job matrices), not 13. The "13-scene" terminology in
  this repo refers to a **separate C42 structural-downsampling benchmark**
  (`reports/speedysplat-c42-13scene-final.md`, `reports/fastgs-c42-13scene-final.md`,
  `reports/fastergs-c42-13scene-final.md`) that is not part of the HiGS
  trainability line. Section 4 reports the actual 11-scene HiGS matrices.

These discrepancies are documented honestly rather than fabricated.

---

## 1. HiGS Architecture

### What HiGS is

HiGS ("Hierarchical Gaussian Splatting") is gsplat's experimental
**Gaussian-inference / macro-tile rendering** path, pinned to upstream gsplat
commit `77ab983ffe43420b2131669cb35776b883ca4c3c`
(`reports/higs-trainability-analysis-2026-07-24.md:8`). The renderer packs
Gaussians into a **FP16-oriented packed scene** with a macro-tile intersect
structure (`MacroTileRasterize.cu`) and a stateful renderer that culls
invisible Gaussians via a per-tile Gaussian mask
(`reports/higs-trainability-implementation.md:1518-1525`). The
`GaussianInferenceScene` stores quaternion/scale/opacity in FP16 layouts with
optional lossy SH compression (PACKED_16B/32B)
(`reports/higs-trainability-analysis-2026-07-24.md:28-31`).

### Why it was inference-only

The pre-trainability analysis (`reports/higs-trainability-analysis-2026-07-24.md:14-31`)
identifies three blockers: (1) `check_inference_grad_mode()` rejects
grad-enabled execution; (2) `GaussianInferenceScene.from_gaussian_tensors()`
detaches all grad-tracked inputs; (3) the CUDA extension installs an Autograd
fallthrough because no backward kernel exists.

### The differentiable pieces (native CUDA backward)

The implemented differentiable path is **HiGS forward + HiGS native CUDA
backward** (`reports/higs-trainability-implementation.md:3-8`), with gsplat
recomputation kept as an explicit fallback (`backward_backend="gsplat_recompute"`).

Three backward paths exist (`reports/higs-trainability-implementation.md:10-16`):

| Path | Entry point | `backward_backend` | Backward work |
|---|---|---|---|
| Frozen-topology native | `rasterize_gaussian_higs_frozen(..., backward_mode="higs_native")` | `higs_native` | Native CUDA kernels from forward-captured state; no recomputation |
| Dynamic-topology native | `rasterize_gaussian_higs_dynamic(..., backward_mode="higs_native")` | `higs_native` | Same native kernels + densify/prune with Adam-state sync |
| gsplat recompute fallback | `backward_mode="gsplat_recompute"` | `gsplat_recompute` | Standard gsplat `rasterization()` re-run under autograd |

The native backward **never re-runs the rasterization pipeline** — it consumes
the forward-captured state (means2d / conics / evaluated colors / opacities /
per-tile sorted intersection ids / render alphas / last ids / radii) so
backward is bound to the exact scene/hierarchy/order/visibility version of the
forward (`reports/higs-trainability-implementation.md:18-22`).

### Forward/backward design

The native CUDA backward is implemented in
`gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu`
with three kernel stages (`reports/higs-trainability-implementation.md:39-58`):

1. **Pixel-blend backward** (`higs_blend_bwd_kernel`) — one thread per pixel,
   shared-memory batched traversal of per-tile sorted intersection lists
   (back-to-front), warp-reduced `rasterize_to_pixels_3dgs_blend_bwd`,
   atomic scatter to flat `[I*N, ...]` gradients.
2. **Projection VJP** (`higs_projection_bwd_kernel`) — one thread per
   (image, gaussian) pair, warp-reduced by Gaussian id, camera-model
   projection VJP (`persp_proj_vjp` / `ortho_proj_vjp` / `fisheye_proj_vjp`)
   → posW2C_VJP + covarW2C_VJP → `quat_scale_to_covar_vjp`.
3. **SH VJP** (`higs_sh_vjp_kernel`) — degree 0..3 port of gsplat's
   `sh_coeffs_to_color_fast_vjp`, chained through the forward activation
   `colors_eval = clamp_min(sph + 0.5, 0)`.

Host launcher `higs_rasterize_backward(...)` returns FP32 tuple
`(v_means, v_quats, v_scales, v_opacities, v_colors_master, v_backgrounds)`
(`reports/higs-trainability-implementation.md:57-58`).

The Python autograd layer `_HigsAutogradFunction` captures the full forward
state in ONE pass (`_native_forward_capture`:
`fully_fused_projection → isect_tiles → isect_offset_encode → _maybe_evaluate_sh
→ rasterize_to_pixels_3dgs`) and saves all 23 inputs
(`reports/higs-trainability-implementation.md:62-69`). FP32 master tensors are
the optimization variables; FP16 packed buffers are used only for the HiGS
culling scene. Lossy SH compression is trainable via a straight-through FP16
quantization (STE) (`reports/higs-trainability-implementation.md:73-76`).

### Supporting kernels

- `GatherVisible.cu/.h` — `higs_gather_visible` (visible-subset gather,
  bypassing PyTorch's slow vectorized row gather) and `higs_union_visible_mask`
  (fused union visibility mask over cameras)
  (`reports/higs-trainability-implementation.md:363-365, 632-650`).
- The PX (pixels-per-thread) blend VJP `higs_blend_bwd_px_kernel<CDIM, PX>`
  lets each thread own PX pixels, scaling per-isect warp reductions as 1/PX
  (`reports/higs-trainability-implementation.md:1241-1258`).

### Discrete culling semantics

The visibility mask is computed under `no_grad` as a plain boolean index —
never a continuous differentiable variable. Visible Gaussians get gradients
through the native chain; invisible get exactly zero (stop-gradient
visibility) (`reports/higs-trainability-implementation.md:80-84, 69-71`).

---

## 2. Correctness: Native Backward vs gsplat Recomputation

### Test suites and files

| File | Tests (function count) | Stage |
|---|---|---|
| `tests/test_higs_trainable.py` | 13 | Stage A correctness baseline |
| `tests/test_higs_frozen.py` | 16 | Stage B frozen topology |
| `tests/test_higs_dynamic.py` | 16 | Stage C dynamic topology |
| `tests/test_higs_native_backward.py` | 53 | Native backward suite |

(Counts from `Select-String 'def test_'` per file; total = 98 functions.)

Additional HiGS test files: `tests/test_higs_paper_protocol.py`,
`tests/test_higs_paper_results.py`, `tests/test_higs_paper_results_assembler.py`,
`tests/test_higs_paper_tables.py`, `tests/test_higs_sparse_pixel_raster.py`,
`tests/test_higs_training_commands.py`.

### Test results

**Initial native-backward suite (EPIC-05, A100):**
```
tests/test_higs_trainable.py ............... 13/13 [100%]
tests/test_higs_frozen.py ................... 14/14 [100%]
tests/test_higs_dynamic.py ................. 11/11 [100%]
tests/test_higs_native_backward.py ......... 61/61 [100%]
============================== 99 passed in 19.91s ===============================
```
(`reports/higs-trainability-implementation.md:127-132`)

Note: the function count (98) vs reported passed (99) differs because some
test functions are parametrized (e.g. `@pytest.mark.parametrize` on
`sh_degree`, `camera_model`, `render_mode`), generating multiple test cases
per function.

**As tile-sampling features were added:**
- Round 32: "Remote EPIC-05 suite: 105 passed; local Windows: pytest 158
  passed / 103 skipped"
  (`reports/higs-trainability-implementation.md:1834`).
- Round 37: "Full HiGS suite 44 passed; full repo suite 272 passed / 1 skipped"
  (`reports/higs-trainability-implementation.md:2143`).
- Round 38: "HiGS 全量 105 passed... 全仓 `pytest tests` 272 passed / 1 skipped"
  (`reports/higs-trainability-implementation.md:2107-2108`).
- Round 39: "HiGS 全量 109 passed；全仓 `pytest tests` 276 passed / 1 skipped"
  (`reports/higs-trainability-implementation.md:2071`).

**Windows local dev:** "276 passed / 1 skipped on the local RTX box"
(`reports/higs-trainability-implementation.md:190`).

**Native densification contract smoke** (`paper/higs/higs-full-runner-readiness.json:95-110`):
CUDA tests `passed: 80, failed: 0` across
`tests/test_higs_native_backward.py` and `tests/test_higs_dynamic.py`.

### What the tests cover

The native-backward suite (`reports/higs-trainability-implementation.md:109-124, 133-145`)
covers: forward RGB/SH/background parity vs standard gsplat; finite-difference
gradients on means/quats/scales/opacities/RGB/SH; `torch.autograd.gradcheck`;
non-empty background; single + multi camera; empty/all/partial visible sets;
invisible Gaussians get exactly zero gradient; alpha-only backward; mixed
precision (`packed_dtype="torch.float16"`); SH degrees 0..3; SH compression
via STE; native-vs-recompute gradient agreement (RGB + SH incl. clamp
activation); explicit fallback when extension unavailable; pending-backward
topology mutation raises; densify/prune optimizer-state sync; culling-boundary
FD (near/far plane, radius clip, projection edge); depth render modes
`D`/`ED`/`RGB+D`/`RGB+ED`; 6 no-CUDA static/API surface tests.

Key test classes in `tests/test_higs_native_backward.py`:
`TestDensificationContract`, `TestForwardParity`, `TestNativeBackwardGradients`,
`TestSHAndCompression`, `TestNativeVsRecompute`, `TestCameraModelBackward`,
`TestDepthRenderModes`, `TestFallbackAndErrors`, `TestDynamicTopology`,
`TestStaticApiSurface`, `TestCullingBoundaryFD`, `TestCullingRefreshAndClamp`,
`TestTileSampledBackward` (4 tests: `test_sampled_backward_matches_masked_full`,
`test_sampled_multi_camera_matches_masked_full`,
`test_unsampled_gaussians_exact_zero_grad`,
`test_unmasked_loss_background_gradient`).

### Gradcheck / finite-difference verification

- `test_finite_difference_means_quats_scales` and `test_gradcheck_small_fully_visible`
  (`tests/test_higs_native_backward.py:337, 390`).
- `test_sh_finite_difference`, `test_ortho_finite_difference`,
  `test_depth_finite_difference` (parametrized over render modes).
- Culling-boundary FD: `test_near_plane_culling_fd`,
  `test_far_plane_culling_fd`, `test_radius_clip_culling_fd`,
  `test_projection_boundary_culling_fd`.

### Native-vs-recompute gradient agreement

Gradient cosine: **0.999996 (train) / 0.999997 (bicycle)**
(`reports/higs-trainability-implementation.md:270-271`), later
**0.99999998-1.000000** (`reports/higs-trainability-implementation.md:484`) and
**0.9999999 / 1.000000** (`reports/higs-trainability-implementation.md:1612-1613`).
Forward parity PSNR 21.17 dB / 18.98 dB.

### Stage A/B/C (historical, superseded)

`docs/higs-trainable-implementation.md` documents the earlier Stage A/B/C
delivery: 38 tests passing, gradient cosine = 1.000000, all 5 param types
FP32 master tensors. Superseded by the native CUDA backward (99 tests).

---

## 3. T_HIGS_REGRESSION

**No finding named `T_HIGS_REGRESSION` exists in this repository.** An
exhaustive search (`grep` for `T_HIGS_REGRESSION`, `T_HIGS`,
`TRAINABLE_HIGS`, `higs.regression` across all files) and the evidence ledger
returned zero matches. The closest analogous findings are the **measured
regressions that were reverted** during optimization, documented in
`reports/higs-trainability-implementation.md`:

### Reverted optimization experiments (performance regressions)

1. **Round 08-01i — single-pass culling+projection merge (REVERTED):**
   "higs_native forward regressed 17.8 -> 18.5 ms on bicycle (total 44.6 ->
   45.5 ms)... Reverted; 08-01j matches 08-01h"
   (`reports/higs-trainability-implementation.md:625-630`).

2. **Round 08-01l — camera-row slice experiment (REVERTED):** "a regression
   vs the fused-mask-only state, so the slice was reverted and 08-01m is the
   final result" (`reports/higs-trainability-implementation.md:689-692`).

3. **Round 24 — blend-VJP ellipse AABB prefilter (REVERTED):** "stable ~+2.1
   ms (+7%) on the blend kernel... The source was restored to the round-22
   PX=2 baseline" (`reports/higs-trainability-implementation.md:1360-1367`).

4. **Round 26 — per-(camera,gaussian) AABB prefilter (REVERTED):** "+2.3~2.6
   ms on the blend kernel, reproducing round-24's +2.1 ms... Source restored
   to the ef8fcb3 PX=2 baseline"
   (`reports/higs-trainability-implementation.md:1452-1458`).

5. **Round 27 — shared-memory slot accumulation (REVERTED):** "Measured blend
   bwd 29.08 -> 35.63/35.69 ms (+6.6 ms, ~23% regression)... Source restored
   to the ef8fcb3 PX=2 baseline"
   (`reports/higs-trainability-implementation.md:1475-1496`).

6. **Round 20 — culling-projection row reuse (REVERTED):** "a net regression,
   reverted (bit-exact either way; tests still 100 passed)"
   (`reports/higs-trainability-implementation.md:1213-1216`).

### Quality regressions in ablation matrices

- The 210-job matrix: `higs_proposed` has **mean PSNR -0.380 dB vs gsplat**
  (`paper/higs/tables/aggregate.md:5`, `paper/higs/tables/aggregate.json:13`).
- `higs_visible_only` fails all quality gates (PSNR CI lo -0.172, SSIM CI lo
  -0.0045, LPIPS CI hi +0.0078) and speed (mean 1.012x, CI lo 0.794)
  (`paper/higs/tables/confirmatory-accel15-summary.json:135-193`).
- Tile-sampling LPIPS regression: "LPIPS degrades in every r<1 mode on both
  scenes (+0.02..+0.08) - the honest quality bound"
  (`reports/higs-trainability-implementation.md:1838-1841`).

**Resolution/status:** All performance regressions were reverted with tests
re-verified at 100/109 passed. The quality regressions are documented as
honest bounds, not bugs. No single finding is labeled `T_HIGS_REGRESSION`.

---

## 4. 11-Scene Training Results (210-job and 132-job Matrices)

> The HiGS paper uses **11 scenes × 3 seeds**, not 13. The 11 scenes are:
> deep_blending/drjohnson, deep_blending/playroom, mipnerf360/{bicycle, bonsai,
> counter, garden, kitchen, room, stump}, tanks_and_temples/{train, truck}.

### 210-job from-scratch A100 matrix (primary)

`paper/higs/tables/matrix-summary.json` — 5 methods × 11 scenes × 3 seeds =
210 complete jobs, 0 failed, 0 missing. Methods: original_3dgs (33 jobs),
gsplat (48), speedy_splat (33), higs_full (48), higs_proposed (48). The 48-job
methods include 15 cross-hardware A100 leg jobs.

**Per-method aggregates** (`paper/higs/tables/matrix-summary.json:17-53`):

| Method | Jobs | PSNR mean (dB) | Wall mean (s) | TTQ mean (s) | Peak VRAM mean (MiB) |
|---|---|---|---|---|---|
| original_3dgs | 33 | 28.576 | 793.5 | 657.7 | 6517.8 |
| gsplat | 48 | 28.288 | 594.0 | 516.3 | 3809.5 |
| speedy_splat | 33 | 27.958 | 1288.7 | 1065.0 | 6094.1 |
| higs_full | 48 | 28.270 | 576.8 | 505.3 | 3780.0 |
| higs_proposed | 48 | 27.872 | 541.9 | 538.7 | 2854.2 |

**Speed/quality geomean** (`paper/higs/tables/aggregate.json`,
`paper/higs/tables/aggregate.md`):

| Comparison | Wall speedup (geomean) | PSNR delta (mean dB) |
|---|---|---|
| gsplat vs higs_proposed | 1.043x | -0.380 |
| higs_full vs higs_proposed | 1.019x | -0.354 |
| gsplat vs speedy_splat | 0.413x | -0.800 |

**Per-scene table** (`paper/higs/tables/summary.md`) — selected scenes:

| Scene | Method | PSNR (dB) | SSIM | LPIPS | Wall (s) | TTQ (s) | Mem (GiB) | Gaussians |
|---|---|---|---|---|---|---|---|---|
| bicycle | gsplat | 25.58±0.05 | 0.774 | 0.160 | 1065±8 | 949±40 | 8.7 | 5.88M |
| bicycle | higs_full | 25.57±0.05 | 0.774 | 0.160 | 921±6 | 817±46 | 8.5 | 5.85M |
| bicycle | higs_proposed | 25.50±0.01 | 0.762 | 0.177 | 670±3 | 664±3 | 6.3 | 4.29M |
| garden | gsplat | 27.74±0.02 | 0.872 | 0.072 | 908±4 | 836±13 | 7.4 | 5.02M |
| garden | higs_full | 27.71±0.08 | 0.871 | 0.072 | 904±2 | 834±7 | 7.4 | 5.02M |
| garden | higs_proposed | 27.32±0.03 | 0.850 | 0.098 | 596±2 | 593±2 | 4.6 | 3.11M |
| train | gsplat | 22.81±0.21 | 0.882 | 0.056 | 442±9 | 432±8 | 0.8 | 500K |
| train | higs_full | 22.68±0.11 | 0.881 | 0.056 | 523±7 | 509±7 | 0.9 | 501K |
| train | higs_proposed | 22.07±0.19 | 0.851 | 0.078 | 674±7 | 669±8 | 0.6 | 338K |
| truck | gsplat | 27.51±0.07 | 0.940 | 0.024 | 365±4 | 354±4 | 1.3 | 840K |
| truck | higs_full | 27.55±0.05 | 0.940 | 0.024 | 424±3 | 411±3 | 1.3 | 832K |
| truck | higs_proposed | 26.51±0.04 | 0.923 | 0.034 | 533±10 | 531±10 | 0.8 | 541K |

**Headline finding:** higs_proposed uses **21.6% lower mean peak GPU memory**
(per-job range -17.0% to +38.6%), but final PSNR is lower on 10/11 scenes
(tied on stump), mean wall time lower on 7/11 scenes, TTQ lower on only 3/11
(`docs/higs-paper-plan.md:92-100`).

### 132-job confirmatory matrix (accel15)

`paper/higs/tables/confirmatory-accel15-summary.json` — 4 methods × 11 scenes
× 3 seeds = 132 jobs, 0 failures. The frozen candidate
`gsplat_30k_fused_prune10_rclip05` (fused renders + opacity pruning every 10th
densification + radius clamp 0.05) **passes all 5 pre-registered gates**
(`paper/higs/README.md:8-16`):

| Gate | Candidate value | Threshold | Pass? |
|---|---|---|---|
| PSNR paired-delta 95% CI lo | -0.022 dB | >= -0.10 dB | ✓ |
| SSIM CI lo | -0.0011 | >= -0.003 | ✓ |
| LPIPS CI hi | +0.0025 | <= +0.005 | ✓ |
| Wall speedup ratio mean | 1.164x | >= 1.111x | ✓ |
| Wall speedup CI lo | 1.034 | > 1.0 | ✓ |
| TTQ | faster (CI lo -85.5 s) | faster | ✓ |

(`paper/higs/tables/confirmatory-accel15-summary.json:76-134`)

Controls that **fail** the gates: `gsplat_25k` (early-stop) fails quality
(PSNR CI lo -0.122, LPIPS CI hi +0.0053); `higs_visible_only` fails quality
and speed (mean 1.012x, CI lo 0.794)
(`paper/higs/tables/confirmatory-accel15-summary.json:17-75, 135-193`). This
proves the gain is neither ordinary early-stop nor the visibility mechanism
alone (`paper/higs/README.md:16-22`).

### 165-job confirmatory formal matrix (earlier)

`paper/higs/tables/confirmatory-summary.json` — 5 methods × 11 scenes × 3
seeds = 165 jobs. `higs_full` passes non-inferiority on all three quality
metrics (PSNR CI lo -0.052, SSIM CI lo +0.00001, LPIPS CI lo -0.00057) but
**fails the speed gate** (wall speedup mean 0.965x, CI lo 0.845)
(`paper/higs/tables/confirmatory-summary.json:76-134`). `higs_current` fails
all quality NI gates and speed. `higs_switch_12k` and `higs_switch_21k` also
fail quality and speed.

### Short-horizon training benchmark (per-step, 20-step protocol)

Final two-scene baseline (EPIC-05 A100, 1920×1080, 4 train + 3 eval cams, 20
steps, final code 9bbd720 / ef8fcb3, `reports/higs-trainability-implementation.md:1598-1616`):

| Scene | Backend | fwd ms | bwd ms | tot ms | train ms | VRAM | cull | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|
| train (1.03M) | std_ll | 11.3 | 22.3 | 34.0 | 35.9 | 3.23 GB | 0% | 19.02 | 0.6885 | 0.3850 |
| train | higs_native | 11.7 | 18.9 | 31.0 | 32.6 | 3.57 GB | 15.1% | 19.02 | 0.6885 | 0.3853 |
| train | higs_dynamic | 9.7 | 16.8 | 27.0 | 28.9 | 4.02 GB | 15.5% | 20.09 | 0.7139 | 0.3567 |
| bicycle (6.13M) | std_ll | 26.8 | 48.0 | 75.2 | 82.7 | 10.74 GB | 0% | 16.75 | 0.4621 | 0.5516 |
| bicycle | higs_native | 26.4 | 39.5 | 66.3 | 73.8 | 11.67 GB | 62.9% | 16.76 | 0.4620 | 0.5518 |
| bicycle | higs_dynamic | 20.8 | 35.0 | 56.3 | 63.5 | 14.67 GB | 62.9% | 17.59 | 0.4832 | 0.4956 |

Total-iteration speedup vs std_ll: higs_native **-9.2% (train) / -10.8%
(bicycle)**, higs_dynamic **-19.5% / -23.2%**; native grad cosine vs recompute
0.999994 / 1.000000 (`reports/higs-trainability-implementation.md:1611-1616`).

### Tile-sampled training (short-horizon speedups)

Round 40 local end-to-end (`reports/higs-training-speedup-research-2026-08-03.md:98-103`):

| config | fwd ms | bwd ms | total ms | vs std | PSNR | LPIPS |
|---|---|---|---|---|---|---|
| std | 22.8 | 55.2 | 78.6 | 1.00x | 19.24 | 0.2968 |
| higs_native (r=1.0) | 26.1 | 42.2 | 69.7 | 1.13x | 19.24 | 0.2970 |
| higs_native_ts (r=0.5) | 18.9 | 27.2 | 48.1 | **1.63x** | 19.04 | 0.3332 |
| higs_native_ts (r=0.25) | 15.1 | 20.5 | 36.9 | **2.13x** | 18.39 | 0.4066 |

Round 41b A100 3-seed (`reports/higs-training-speedup-research-2026-08-03.md:142-150`):

| config (train, 3-seed) | PSNR | SSIM | LPIPS | total_ms/step | speedup |
|---|---|---|---|---|---|
| full r=1.0 | 16.673±0.060 | 0.6267 | 0.3678 | 21.98 | 1.00x |
| error_guided r=0.35 | 17.089±0.107 | 0.6310 | 0.3914 | 12.10 | **1.82x** |
| error_guided r=0.30 | 16.897±0.023 | 0.6298 | 0.3944 | 11.57 | **1.90x** |

Round 42 M5 multi-scene matrix (3 new scenes, 3-seed,
`reports/higs-training-speedup-research-2026-08-03.md:195-199`):

| Scene | full PSNR | eg PSNR | ΔPSNR | ΔLPIPS | speedup |
|---|---|---|---|---|---|
| garden (5.8M) | 18.733 | 17.971 | -0.76 | +0.050 | 2.12x |
| bonsai (1.2M) | 23.128 | 22.721 | -0.41 | +0.020 | 1.74x |
| truck (2.5M) | 18.711 | 19.297 | +0.59 | -0.015 | 1.88x |

### Confirmatory results (full 30k horizon)

`docs/confirmatory-results-2026-08-06.md:28-61` — canonical 5 scenes, A100,
1080p, 30k: pd vs ctrl paired delta — train_ms **+0.739** (pd is slower), PSNR
**+0.612** (pd better), LPIPS **-0.014** (pd better). "Strict per-scene
dominance: 0 of 5 scenes. The 1080p/30k training-speed hypothesis is NOT
confirmed on the canonical five." The robust benefit is final quality with
large Gaussian-count reduction (bicycle 3.12M → 398K).

---

## 5. Dynamic Topology

### scene_version binding

`_HigsDynamicScene` + `HigsRendererHandle` replace the singleton-only pattern
with an explicit, versioned handle passed via `scene=`
(`reports/higs-trainability-implementation.md:88-90`). Every forward/backward
is bound to a unique `scene_version`; the autograd context keeps the handle
(and its packed buffers) alive until backward
(`reports/higs-trainability-implementation.md:90-91`).

### Forward/backward topology consistency

`mark_dirty()` raises while a backward is pending (mutation mid-graph is
impossible), and the version is validated again in backward
(`reports/higs-trainability-implementation.md:92-93`). The `cancel_pending_backward`
method was added for SkipGS per-view backward-gating
(`git log: 032fcea`).

### Optimizer-state migration

`sync_optimizer_state_for_topology_change()` performs Adam-state sync for
densify/prune: copy duplicated rows, zero new rows, drop pruned rows
(`reports/higs-trainability-implementation.md:34-35`). Optimized in Round 15
with `higs_gather_rows` (densify-event tails 31-73 ms → 12-18 ms,
`reports/higs-trainability-implementation.md:986-992`) and Round 17 with
`zero_on_neg` mode (8.36 → 2.97 ms per event, 2.8x, bit-identical,
`reports/higs-trainability-implementation.md:1056-1078`).

### Tests

- `test_pending_backward_mutation_raises` (`tests/test_higs_native_backward.py:1175`)
- `test_densify_optimizer_state_sync` (`:1206`)
- `test_prune_optimizer_state_sync` (`:1259`)
- `TestCullCache` in `tests/test_higs_dynamic.py` (3 tests: cadence counting,
  invalidation on densify, static-param parity)
  (`reports/higs-trainability-implementation.md:2139-2143`).
- Topology-rebuild cost: ~3.6 ms pack + construct on 6.13M Gaussians, ~+9 ms
  per `mark_dirty()`-forced step; amortized ~+1.4 ms/step
  (`reports/higs-trainability-implementation.md:731-734, 2216-2218`).

---

## 6. Known Limitations and Open Questions

1. **Render modes / camera models:** Native backward supports RGB/D/ED/RGB+D/
   RGB+ED and pinhole/ortho/fisheye. ftheta/lidar and eval3d hit-distance
   modes still raise (use recompute fallback)
   (`reports/higs-trainability-implementation.md:2172-2178`).
2. **Culling is a discrete approximation:** The visibility mask reflects the
   last packed parameter snapshot between rebuilds
   (`reports/higs-trainability-implementation.md:2179-2183`).
3. **SH compression STE is approximate:** Non-zero gradient;
   `sh_compression_mode="none"` remains the exact path
   (`reports/higs-trainability-implementation.md:2184-2187`).
4. **SH VJP ~1.1 ms slower than std** (5.97 vs 4.89 ms on bicycle): the
   ~109M coefficient atomics + ReLU-mask load; a std-style per-camera `v_dirs`
   redesign would cost 294 MB for ~0.3 ms — documented trade-off
   (`reports/higs-trainability-implementation.md:2206-2215`).
5. **Blend VJP is the structural floor:** 6.23G per-pixel evals are
   format-independent; the macro-tile backward ceiling is ~4-6 ms, quantified
   and closed (Round 28/29, `reports/higs-trainability-implementation.md:1498-1551`).
6. **210-job higs_proposed quality gap:** PSNR lower on 10/11 scenes; the
   210-job results "prove trainability and memory reduction, not a universal
   quality-preserving training speedup for the visibility-masked HiGS method
   itself" (`docs/higs-paper-plan.md:97-100`).
7. **Tile-sampling LPIPS bound:** LPIPS degrades +0.02..+0.08 at r<1 in every
   mode on both scenes; closed on train with LPIPS regularization (Round 36)
   but persists on bicycle/garden (high-N scenes)
   (`reports/higs-training-speedup-research-2026-08-03.md:71`).
8. **Cross-hardware generalization blocked:** Consumer and second data-center
   GPU cohorts remain blockers (`paper/higs/README.md:31`).
9. **30k convergence at 1080p:** The exploratory 720p/3k speed wins do not
   transfer to 1080p/30k — "the pd cell does not speed up training" at full
   horizon (`docs/confirmatory-results-2026-08-06.md:56-61`).
10. **Frozen protocol ceiling:** Fixed-LR protocols collapse after ~300 steps;
    M4 converged-quality parity requires the full 3DGS recipe (lr schedule,
    densify window, opacity reset) (`reports/higs-trainability-implementation.md:1953-1956`).

---

## 7. Key Quotes

**Main conclusion (implementation report):**
> "the frozen native per-step path is at its structural floor on bicycle. The
> top three costs (blend bwd ~19.5 ms / rasterize fwd ~8.7 ms / SH VJP ~6 ms)
> are the same kernel math std executes... No further culling or autograd-side
> change can move the total more than ~1-2 ms without algorithmic work (e.g.
> tile LOD / fewer isects) or benchmark-level stream overlap."
> — `reports/higs-trainability-implementation.md:817-824`

**Speedup claim (completion audit):**
> "Total-iteration speedup vs std_ll on both scenes: higs_native -9.2% (train)
> / -10.8% (bicycle), higs_dynamic -19.5% / -23.2%; native grad cosine vs
> recompute 0.999994 (train) / 1.000000 (bicycle). This is the
> completion-audit evidence for the objective requirement that total iteration
> (forward + backward) must actually benefit before claiming a speedup: both
> native and dynamic paths satisfy it on both small and large scenes, so the
> speedup claim stands."
> — `reports/higs-trainability-implementation.md:1611-1616`

**Paper writing boundary:**
> "The abstract may claim a native differentiable implementation, a released
> evaluation protocol, a frozen 210-job from-scratch execution in which the
> proposed method reduces mean peak GPU memory relative to gsplat, and a
> pre-registered 132-job 3-seed confirmatory matrix in which the frozen accel15
> candidate passes all quality-preservation and >=10% wall-clock speedup gates
> (1.164x mean, CI lower bound 1.034, TTQ faster). It may not claim
> cross-hardware generalization or superiority to official training baselines."
> — `paper/higs/README.md:62-72`

**Honest negative (210-job):**
> "These 210-job results prove trainability and memory reduction, not a
> universal quality-preserving training speedup for the visibility-masked HiGS
> method itself. The 1.8x-2.5x short-horizon numbers must not be advertised as
> full-convergence results."
> — `docs/higs-paper-plan.md:97-100`

**Tile-sampling honest bound:**
> "LPIPS degrades in every r<1 mode on both scenes (+0.02..+0.08) - the honest
> quality bound that still prevents M4 from being fully green."
> — `reports/higs-trainability-implementation.md:1838-1841`

**Confirmatory negative (1080p/30k):**
> "on the canonical five at full 1080p resolution and the full 30k convergence
> horizon, the progressive-resolution + union-decay cell does not speed up
> training; its robust benefit is final quality (PSNR +0.61 dB, LPIPS -0.014)
> with a large Gaussian-count reduction."
> — `docs/confirmatory-results-2026-08-06.md:56-61`

---

## Key commit hashes (from `git log --oneline --all | grep -iE 'higs|hierar|differentiable'`)

| Hash | Description |
|---|---|
| `73bebec` | feat(higs): native CUDA backward, culling semantics, dynamic topology, tests and benchmark |
| `af7a1b0` / `110622b` / `a36f59d` | feat(higs): ortho/fisheye native backward, SH-compression STE, culling auto-refresh, configurable color clamp |
| `f2b93de` | feat(higs): native backward for depth render modes (D/ED/RGB+D/RGB+ED) |
| `4df0b14` | perf(higs): scatter master gradients directly in native backward kernels |
| `e155a9b` | docs: publish HiGS trainability analysis |
| `364f3d0` | docs: condense HiGS trainability section in README |
| `98f6dc9` | bench(higs): round-39 backward tile compaction |
| `6f48873` | fix(higs-ablation): pin accel12 to dynamic-native-backward renderer contract |
| `99f80fc` | fix(higs-ablation): correct accel15 pre-registration (sh_fp16 → fp32) |
| `c598bed` | feat(higs-ablation): accel15 exploration results (seed-0 gates pass) |
| `1c29fc9` | feat(higs-ablation): freeze accel15 candidate, register 132-job confirmatory matrix |
| `fb1cb7d` | feat(higs-ablation): confirmatory_accel15_11s3 132-job formal matrix results |
| `e7bb1e7` | feat(higs-ablation): commit raw per-job evidence (132 files, sha256-verified) |
| `06cc15a` | docs(higs-ablation): publish accel15 formal confirmatory result |
| `0e047cb` | feat(higs-ablation): accel21 SkipGS backward-gating exploration (honest negative) |
| `fef0b77` | feat(higs-ablation): accel20 low-res-window exploration (honest negative) |
| `2b859d2` / `06cc15a` | docs(higs-ablation): publish accel15 result (duplicate commits) |
| `5d9e3bc` | docs(higs-ablation): record accel22/23/24 honest-negative explorations |
| `167f40e` | feat(higs-ablation): pre-register Phase-3 topology-preserving exploration |
| `a7033e4` | feat(higs-ablation): complete 44-job topology-preserving exploration (honest negative) |
