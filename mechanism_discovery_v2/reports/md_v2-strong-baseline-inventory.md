# MD V2 — Strong Baseline Inventory

> **Purpose**: Factual inventory of available strong baselines for 3DGS research within the current project environment. This informs planning for mechanism-discovery experiments. No speed numbers are claimed unless sourced from a prior locked report.
>
> **Environment snapshot** (verified 2026-09):
> - Python 3.13, torch 2.13.0+cu130, CUDA 13.0 runtime
> - gsplat 1.4.0 (installed), gsplat 1.5.3 snapshot (source-only, `mechanism_discovery_v2/external_source_snapshot/gsplat_1_5_3/`)
> - diff-gaussian-rasterization 0.0.0 (installed, base variant — no SparseGaussianAdam)
> - fast-gauss 0.0.9 (installed, `C:\Users\36570\miniconda3\Lib\site-packages\fast_gauss/`)
> - Hardware target: NVIDIA A100-PCIE-40GB (verified in reference-v1-baseline-lock.md)

---

## 1. REFERENCE_V1_ABSGRAD (Current Baseline)

| Attribute | Value |
|-----------|-------|
| **Code available?** | Yes — `baseline/reference_v1/` (gaussian_model.py, trainer.py, config.py, provenance.py, instrumentation.py, colmap_reader.py) |
| **Code path** | `baseline/reference_v1/trainer.py` |
| **Compatible with current env?** | Yes. Verified: runs under Python 3.13, torch 2.13.0+cu130, gsplat 1.4.0 |
| **Can run on A100?** | Yes — locked run completed on A100-PCIE-40GB in ~28 min (reference-v1-baseline-lock.md) |
| **Renderer backend** | gsplat 1.4.0 `rasterization(..., packed=False, absgrad=True, tile_size=16)` |
| **Densification strategy** | Custom `GaussianModel` class in `gaussian_model.py` — hand-rolled clone/split/prune with persistent Adam state migration |
| **Semantic label** | `REFERENCE_V1_ABSGRAD` (config.py defines `semantic_label`) |
| **Config hash** | `REFERENCE_V1_ABSGRAD` (defined in `baseline/reference_v1/config.py:semantic_label`, tracked in provenance) |
| **Known results** | Room 30K: PSNR=32.30, SSIM=0.9263, N=952K, mean 55.4 ms/iter (locked report) |
| **Deviation from official** | Uses absgrad=True with threshold 0.0008 (4× official 0.0002 signed). 7 documented gsplat adaptations (reference-v1-baseline-lock.md, Q8). |
| **Implementation effort to replicate** | Already built and locked. Zero effort. |
| **Existing benchmarks** | Room 30K trajectory fully instrumented (C49/C50/C53). Checkpoints at 2K, 5K, 10K, 14K, 15K, 30K. |

**Status: LOCKED BASELINE** — this is the sole reference for all paper experiments.

---

## 2. gsplat `DefaultStrategy` Optimized Configuration

| Attribute | Value |
|-----------|-------|
| **Code available?** | Yes — `gsplat.strategy.DefaultStrategy` in installed gsplat 1.4.0 (`C:\Users\36570\miniconda3\Lib\site-packages\gsplat\strategy\default.py`) |
| **Compatible with current env?** | Yes — ships with installed gsplat 1.4.0 |
| **Can run on A100?** | Yes — same CUDA kernel path as the `rasterization()` call used by REFERENCE_V1_ABSGRAD |
| **Default parameters in gsplat 1.4.0** | `prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01, grow_scale2d=0.05, prune_scale3d=0.1, prune_scale2d=0.15, refine_start=500, refine_stop=15000, reset_every=3000, refine_every=100` |
| **Recommended optimized params (AbsGS mode)** | `DefaultStrategy(absgrad=True, grow_grad2d=0.0008, prune_opa=0.005)` — matches reference_v1's threshold and absgrad choice |
| **Packed mode** | `DefaultStrategy` natively uses `packed=True` for `rasterization()`. The strategy's state update in `_update_state()` handles both `packed=True` and `packed=False` code paths (default.py lines 240–250). |
| **Key difference from REFERENCE_V1_ABSGRAD** | DefaultStrategy uses one optimizer PER parameter group (dict keyed by name), not a single Adam with all param groups. Different optimizer-state layout for `duplicate/split/remove` ops (strategy/ops.py). Uses `reset_opa` with value=0.01 vs reference_v1's global min(opacity, 0.01). |
| **Implementation effort** | Low — wrap `DefaultStrategy.check_sanity()` + `step_pre_backward()` + `step_post_backward()` in training loop. Replace hand-rolled GaussianModel topology with strategy calls. |
| **Existing benchmarks** | None in this project's baseline suite. Would be a NEW training configuration. |
| **Config hash** | N/A (no existing locked hash) |

**Note**: The reference-v1-baseline-lock.md states gsplat 1.5.3 was used for the locked run. The environment now has gsplat 1.4.0 installed. The API differences between 1.4.0 and 1.5.3 are minor for the `rasterization()` call signature (same parameters), but the `DefaultStrategy` implementation may differ. The 1.5.3 source snapshot exists at `mechanism_discovery_v2/external_source_snapshot/gsplat_1_5_3/` if needed for diff comparison.

---

## 3. Packed vs Dense Modes in gsplat

| Attribute | Packed Mode (`packed=True`) | Dense Mode (`packed=False`) |
|-----------|----------------------------|-----------------------------|
| **Code available?** | Both modes are in the single `gsplat.rasterization()` function call. No separate code needed. | Same function, different flag. |
| **Current usage** | NOT used by REFERENCE_V1_ABSGRAD (which uses `packed=False`). Used by `GsplatRenderer` in `src/renderers/gsplat_renderer.py` (default `packed=True`). | Used by REFERENCE_V1_ABSGRAD (trainer.py line 119: `packed=False`). Also `GsplatDenseRenderer` class. |
| **Adapter classes** | `GsplatRenderer(packed=True)` | `GsplatRenderer(packed=False)` alias `GsplatDenseRenderer` |
| **Performance impact** | Packed mode flattens per-camera Gaussians into a single list, sorting once. Dense mode keeps per-camera arrays. Packed is typically faster with many cameras. For single-camera training, the performance difference is marginal. | — |
| **Compatible with current env?** | Yes — both modes are in gsplat 1.4.0 | Yes |
| **A100 compatibility** | Yes | Yes (locked run) |
| **Implementation effort** | Zero — flag change | Zero — already the baseline |
| **Existing benchmarks** | None with packed=True. REFERENCE_V1_ABSGRAD is dense. | Room 30K locked. |

**Note**: The `DefaultStrategy._update_state()` has explicit code paths for both packed and dense. Switching the REFERENCE_V1_ABSGRAD from `packed=False` to `packed=True` changes how gradient accumulation indexes Gaussians (packed uses `gaussian_ids` from meta dict; dense uses `radii > 0` selection) — this would affect densification behavior.

---

## 4. SparseGaussianAdam (`SparseGaussianAdam`)

| Attribute | Value |
|-----------|-------|
| **Code available?** | In the *official* Graphdeco repository as part of the accelerated rasterizer. Tried as `from diff_gaussian_rasterization import SparseGaussianAdam` in `reference/graphdeco/train.py` (line 38) with fallback. |
| **Installed in current env?** | **NO** — `diff_gaussian_rasterization` package is installed at version "0.0.0" but does NOT export `SparseGaussianAdam`. Verified: `hasattr(diff_gaussian_rasterization, 'SparseGaussianAdam')` → `False`. The package exposes only `GaussianRasterizationSettings`, `GaussianRasterizer`, and `rasterize_gaussians`. |
| **Install method** | `pip install diff-gaussian-rasterization[3dgs_accel]` per official Graphdeco docs. Not currently installed. |
| **Can run on A100?** | Yes, by design — the accel variant compiles CUDA kernels for A100. Would need installation. |
| **What it does** | Adam optimizer variant that only updates parameters for Gaussians visible in the current view. Redises optimizer memory and computation proportionally to visible count, not total count. Official train.py uses it via `gaussians.optimizer.step(visible, radii.shape[0])` when `use_sparse_adam=True`. |
| **Implementation effort** | Medium — requires installing the accel rasterizer and adapting the optimizer step call. SparseGaussianAdam expects a different optimizer interface: `optimizer.step(visible_mask, total_count)`. |
| **Existing benchmarks** | None in this project. |
| **Config hash** | N/A |

---

## 5. Graphdeco Accelerated Rasterizer

| Attribute | Value |
|-----------|-------|
| **Code available?** | Yes — official Graphdeco codebase at `graphdeco-inria/gaussian-splatting`. The accel variant replaces `diff-gaussian-rasterization` with a version that includes `SparseGaussianAdam` and other optimizations. Installed via `pip install diff-gaussian-rasterization[3dgs_accel]`. |
| **Installed in current env?** | **NO** — the installed `diff-gaussian-rasterization` (0.0.0) is the base variant. The accel variant is a separate install path. |
| **Can run on A100?** | Yes — the official 3DGS paper results were produced on an A100. |
| **What it includes** | (1) `SparseGaussianAdam` optimizer, (2) fused kernels for SH computation, (3) potential minor CUDA optimizations over base rasterizer. |
| **Implementation effort** | Medium — requires (a) uninstalling base diff-gaussian-rasterization, (b) installing the accel variant, (c) adapting the optimizer step. The trainer infrastructure already handles `SparseGaussianAdam` via the `use_sparse_adam` branch in `reference/graphdeco/train.py`. |
| **Existing benchmarks** | None in this project. The official 3DGS paper provides benchmarks. |
| **Config hash** | N/A |

---

## 6. Graphdeco Fused SSIM

| Attribute | Value |
|-----------|-------|
| **Code available?** | Yes — external package `fused_ssim`. Tried as `from fused_ssim import fused_ssim` in `reference/graphdeco/train.py` (line 32) with fallback. |
| **Installed in current env?** | **NO** — `pip list` shows no `fused-ssim` or `fused_ssim` package. Verified: `import fused_ssim` fails. |
| **What it does** | CUDA-optimized SSIM computation that fuses the separable convolution and comparison steps into a single kernel. Provides substantial speedup over the PyTorch-native separable SSIM (factor of ~5–10× depending on resolution, per phase C44 reports). |
| **Compatible with current env?** | Would need installation. Pure CUDA extension — requires CUDA toolkit at build time. |
| **Can run on A100?** | Yes — CUDA kernel, architecture-independent. |
| **Implementation effort** | Low — single import + replace `ssim_fn(pred, gt)` call. The reference_v1 trainer already uses a `SepSSIM` class (trainer.py lines 61–87) that implements separable SSIM in PyTorch. Replacing with `fused_ssim` is a one-line change. |
| **Existing benchmarks** | Phase C44 (`scripts/phase-c44/`) has extensive profiling of fused SSIM. Track B reports show fused SSIM reduces SSIM computation time by ~5–10× at full resolution. |
| **Config hash** | N/A |

---

## 7. Faster-GS

| Attribute | Value |
|-----------|-------|
| **Code available?** | External project. GitHub: not yet determined in this search. |
| **Adapter in src/renderers/ ?** | **NO** — no adapter class present. Available adapters: gsplat, diff_gaussian, speedy_splat, tcgs, flashgs, local_gs, gemm_gs, fast_gauss, stopthepop, experimental. |
| **Installed in current env?** | **NO** — not found in site-packages. |
| **Can run on A100?** | Likely yes if CUDA-based, but not verified. |
| **Implementation effort** | High — requires (a) finding the repository, (b) building/installing the CUDA extension, (c) writing a renderer adapter, (d) adapting the training loop. |
| **Existing benchmarks** | None in this project. |
| **Config hash** | N/A |

**Note**: This entry requires external research to identify the exact repository and verify claims. The name "Faster-GS" may refer to multiple projects. No implementation exists in the current codebase.

---

## 8. FastGS

| Attribute | Value |
|-----------|-------|
| **Code available?** | External project. GitHub: not yet determined in this search. |
| **Adapter in src/renderers/ ?** | **NO** — no adapter class present. |
| **Installed in current env?** | **NO** — not found in site-packages. |
| **Can run on A100?** | Unknown — not verified. |
| **Implementation effort** | High — similar to Faster-GS: requires repository discovery, CUDA build, adapter implementation. |
| **Existing benchmarks** | None in this project. |
| **Config hash** | N/A |

**Note**: Same caveat as Faster-GS. The name "FastGS" is ambiguous and may overlap with "Faster-GS" in the literature. No implementation exists in the current codebase.

---

## Summary Matrix

| # | Baseline / Optimization | Code Available | Installed Now | A100 Compatible | Impl. Effort | Has Benchmark | Ref V1 Config Hash |
|---|------------------------|:--------------:|:-------------:|:---------------:|:------------:|:-------------:|:------------------:|
| 1 | REFERENCE_V1_ABSGRAD | ✅ `baseline/reference_v1/` | ✅ | ✅ (locked) | None | ✅ Room 30K | `REFERENCE_V1_ABSGRAD` |
| 2 | gsplat DefaultStrategy (recommended) | ✅ gsplat 1.4.0 | ✅ | ✅ | Low | ❌ | — |
| 3a | gsplat packed mode | ✅ gsplat 1.4.0 | ✅ | ✅ | Zero (flag) | ❌ | — |
| 3b | gsplat dense mode | ✅ gsplat 1.4.0 | ✅ | ✅ (locked) | Zero (flag) | ✅ (REF_V1) | — |
| 4 | SparseGaussianAdam | ✅ official repo | ❌ | ✅ (needs install) | Medium | ❌ | — |
| 5 | Graphdeco accelerated rasterizer | ✅ official repo | ❌ | ✅ (needs install) | Medium | ❌ | — |
| 6 | Graphdeco fused SSIM | ✅ `fused_ssim` pkg | ❌ | ✅ (needs install) | Low | ✅ Phase C44 | — |
| 7 | Faster-GS | External | ❌ | Unknown | High | ❌ | — |
| 8 | FastGS | External | ❌ | Unknown | High | ❌ | — |

---

## Existing Renderer Adapters in `src/renderers/`

All registered adapters (from `src/renderers/__init__.py`):

| Adapter Name | Class | Backend | Installed? | Notes |
|-------------|-------|---------|:----------:|-------|
| `gsplat` | `GsplatRenderer` | gsplat 1.4.0 `rasterization(packed=True)` | ✅ | Default `packed=True` |
| `gsplat_dense` | `GsplatDenseRenderer` | gsplat 1.4.0 `rasterization(packed=False)` | ✅ | Dense mode |
| `diff_gaussian` | `DiffGaussianRenderer` | diff-gaussian-rasterization 0.0.0 | ✅ | Graphdeco base rasterizer |
| `original_3dgs` | `DiffGaussianRenderer` | same as diff_gaussian | ✅ | Alias |
| `fast_gauss` | `FastGaussRenderer` | fast-gauss 0.0.9 | ✅ | CUDA-GL interop; requires EGL (Linux/WSL2) |
| `speedy_splat` | `SpeedySplatRenderer` | speedy-gaussian-rasterization | ❌ (not installed) | CUB radix sort variant |
| `flashgs` | `FlashGSRenderer` | flash-gaussian-splatting | ❌ (not installed) | InternLandMark/FlashGS |
| `local_gs` | `LocalGSRenderer` | diff-gaussian-rasterization (scores fork) | ❌ | Local-GS / TiCoGS |
| `gemm_gs` | `GemmGSRenderer` | GEMM-GS fork of diff-gaussian | ❌ | Tensor Core blending |
| `tcgs` | `TCGSRenderer` | diff-gaussian-rasterization (scores fork) | ❌ (not installed) | DeepLink-org TensorCore |
| `stopthepop` | `StopThePopRenderer` | — | ❌ | Placeholder only |

**Available and verified working**: `gsplat`, `gsplat_dense`, `diff_gaussian`, `original_3dgs` (all use installed backends). `fast_gauss` is installed but requires EGL (not available on native Windows).

---

## Version Mismatch Note

The locked baseline report (`reference-v1-baseline-lock.md`) was run with **gsplat 1.5.3** and **torch 2.7.1+cu118**. The current environment has **gsplat 1.4.0** and **torch 2.13.0+cu130**. A snapshot of gsplat 1.5.3 source exists at:
`mechanism_discovery_v2/external_source_snapshot/gsplat_1_5_3/`

If exact reproduction of the locked baseline is needed, either:
- Downgrade gsplat to 1.5.3 and torch to 2.7.1+cu118, or
- Validate that gsplat 1.4.0 produces equivalent PSNR/SSIM/N trajectory on Room 30K.

---

## Recommendations (for planning, not decisions)

1. **Lowest-effort new baseline**: gsplat `DefaultStrategy(absgrad=True, grow_grad2d=0.0008)` with packed mode. Leverages all of gsplat's internal state management (zero-effort optimizer migration). Risk: densification behavior may differ from REFERENCE_V1_ABSGRAD due to different per-group optimizer layout.

2. **Lowest-effort speedup**: Install `fused_ssim` and replace the PyTorch `SepSSIM` call. Phase C44 data suggests ~5–10× SSIM speedup. One-line code change.

3. **Medium-effort speedup**: Install `diff-gaussian-rasterization[3dgs_accel]` and use `SparseGaussianAdam`. Changes optimizer step semantics.

4. **Not recommended for current planning**: Faster-GS, FastGS — no implementation, no adapter, unknown compatibility. Would require significant research and development effort before any experiment.

---

*Generated by md_v2-strong-baseline-inventory. All environment checks performed live on the current workspace. Speed claims from REFERENCE_V1_ABSGRAD are sourced from reference-v1-baseline-lock.md. All other assessments are factual availability checks.*
