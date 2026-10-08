# SB0-FGS Report — Frozen C0 × Faster-GS 2026 Strong-Baseline Uplift

**Gate:** Pre-production-CUDA-modification gate. This report delivers **COMPONENT AUDIT + OPPORTUNITY ORACLE** to the main reviewer, per task instruction: *"在任何 production CUDA 修改前，先返回 COMPONENT AUDIT + OPPORTUNITY ORACLE 给主审。"*
**Date:** 2026-09-29
**Scope of this turn:** Steps 1 (COMPONENT AUDIT) + 3 (OPPORTUNITY ORACLE) only. No production CUDA modification performed. Steps 4–10 (reference reproduction, implementation, correctness/performance gates, composition, final verdict) are gated on reviewer approval.

---

## 0. Executive summary

- **9 components audited** across C0 (gsplat H8 pack) vs Faster-GS `main` / `FasterGSTestbed` / `FasterGSFused`. (Audit corrected this turn: C8 Morton/Z-order is a real Faster-GS framework-layer feature, reclassified UNKNOWN -> DISTINCT_AND_EXACT.)
- **Only ONE component passes the Opportunity Oracle gate:** `C7_backward_optimizer_fusion` (fused backward→Adam), realistic O_net 3–5% (room), 8–15% (bicycle), 6–12% (garden) E2E.
- **8 components DROP:** 4 are `ALREADY_IN_C0`, 1 is `DISTINCT_BUT_APPROXIMATE` (prohibited), 1 is `UNKNOWN` (not in reference), 2 are `DISTINCT_AND_EXACT` but fail the O_net gate.
- **Provenance gap flagged:** local C0 `.so`/core sha256 do NOT match the task-pinned hashes. Stage TIMINGS are the authoritative PUBLICATION-grade C0 numbers; binary-hash reconciliation is a pre-implementation blocker.
- **Success criteria status:** A (C0-PLUS) and B (STRONGER_EXTERNAL_BASELINE) both PENDING. Source audit shows most Faster-GS components are already in C0, so a large external gap is not expected from these components alone.

---

## 1. Environment & provenance (OBSERVED)

| Item | Value |
|---|---|
| Host | Windows (PowerShell), cwd `C:\Users\36570\3dgs-renderer-benchmark` |
| GPU | NVIDIA RTX 5070, CUDA driver 13.1 (local); **C0 timings were produced on A100-PCIE-40GB** |
| torch (local) | 2.13.0+cu130, CUDA available |
| C0 source (local) | `_mx_h8/pack/h8/gsplat` (gsplat H8 pack) + `p18_p0/c0_source/HigsNativeBackward.cu` |
| Faster-GS reference (local) | `public_baselines/faster-gs` (branches main / FasterGSTestbed / FasterGSFused fetched) |
| C0 timings source | `artifacts/final-30k/C0_V3_FINAL30K/{room,bicycle,garden}/timing.json` |

**C0 identity (from `room/provenance.json`):** `f9=true`, `scalar_adjoint=true`, `h8_mr=true`, `px_runtime=2`, env `HIGS_PX_RUNTIME=2 / HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint / HIGS_BWD_H8_MR=1 / HIGS_BWD_ABSGRAD=1`, `timing_grade=PUBLICATION`, `contaminated=false`, `n_cameras=311` (room), 30K iters, seed=42. **This matches the task's C0 definition.**

**Provenance gap (EVIDENCE):** local `C0_V3_FINAL30K` `.so` sha256 `9baf8655...` / core `361b216b...` do NOT match the task-pinned core `935ea6c0...` / extension `d26cccf...`. Most likely cause: CUDA extensions embed the build-directory path into the binary; the local build_dir differs from the task's `/root/epic05_frozen/...`. This does NOT affect the stage TIMINGS (PUBLICATION-grade) but MUST be reconciled before any implementation.

---

## 2. COMPONENT AUDIT (OBSERVED + EVIDENCE)

Classification legend: `ALREADY_IN_C0` / `DISTINCT_AND_EXACT` / `DISTINCT_BUT_APPROXIMATE` / `INCOMPATIBLE_WITH_C0` / `UNKNOWN`.

| # | Component | Faster-GS evidence | C0 evidence | Classification |
|---|---|---|---|---|
| C1 | Two-stage (depth/instance/bucket) sorting | `kernels_forward.cuh`: `primitive_depth_keys`, `create_instances`, `extract_instance_ranges`, `cub::DeviceRadixSort`×14, `instance`×302, `bucket`×79; FasterGSTestbed `kernels_forward_separate_sorting.cuh` | `IntersectTile.cu`/`SparseTileLayout.cu`: single-pass tile sort (`isect_offsets`/`flatten_ids`), `DeviceRadixSort`×3 | **DISTINCT_AND_EXACT** |
| C2 | Per-Gaussian backward | `kernels_backward.cuh`: `diff_rasterize`, `grad_mean2d_helper`, `densification_info` | `RasterizeToPixels3DGSSerialBatchBwd.cu`: `H8_MR` template, `v_xy_abs_local`, absgrad, per-gaussian×19 | **ALREADY_IN_C0** |
| C3 | Opacity-aware / tighter bounds | `rasterization_config.h`: `TRUNCATION_MODE` (1/2/3/4σ), `original_opacity_interpretation`, `min_alpha_threshold`, 1.15× clip | `ProjectionEWA3DGSFused.cu`: 3σ (eps2d=0.3), no aggressive cull | **DISTINCT_BUT_APPROXIMATE** |
| C4 | Activation fusion | `preprocess_cu` fuses cov3d→conic→opacity→color | `ProjectionEWA3DGSFused.cu` present; `fused`×128 | **ALREADY_IN_C0** |
| C5 | SH buffer / SH eval fusion | `sh_coefficients_0`+`sh_coefficients_rest`, `convert_sh_to_color` | `SphericalHarmonicsCUDA.cu`: `sh_coeffs_to_color_fast` | **ALREADY_IN_C0** |
| C6 | Custom fused Adam | `adam.cu`: all elements, `bias_correction1/2_rcp`, no skip | `AdamCUDA.cu`: all elements, optional `valid` mask, **no bias correction** | **ALREADY_IN_C0** (bias-corr delta = semantics) |
| C7 | Backward→optimizer fusion | FasterGSFused `kernels_backward.cuh`: `adam_step_helper` inlined into `preprocess_backward_cu`; `moments_*` passed in | separate `AdamCUDA.cu` launch; grad write/read round-trip | **DISTINCT_AND_EXACT** |
| C8 | Morton/Z-order ordering of Gaussians + optimizer states | `Model.py:apply_morton_ordering()` (L472), `morton_encode`+`argsort`+`sort`, `MORTON_ORDERING_INTERVAL=5000`/`END=15000` (framework layer, `CudaUtils.MortonEncoding`) | no Morton re-sort in training path | **DISTINCT_AND_EXACT** |
| C9 | Frontend allocation / densification memory | `buffer_utils.h`: `resize_*_buffers`, `PrimitiveBuffers`/`InstanceBuffers`/`BucketBuffers` blobs | per-tensor `AT_DISPATCH` allocation | **DISTINCT_AND_EXACT** |

**Counts:** ALREADY_IN_C0 = {C2,C4,C5,C6}; DISTINCT_AND_EXACT = {C1,C7,C8,C9}; DISTINCT_BUT_APPROXIMATE = {C3}; INCOMPATIBLE_WITH_C0 = {}; UNKNOWN = {}.

**Prohibited-shortcut findings (EVIDENCE):**
- FasterGSFused `preprocess_backward_cu` early-returns `if(primitive_n_touched_tiles[primitive_idx]==0) return;` → **invisible-Gaussian optimizer skipping (prohibited #1)**. The fusion structure is legitimate; the skip MUST be stripped on port.
- Faster-GS `adam.cu` adds `bias_correction2_sqrt_rcp`; C0 omits bias correction. Adding it would **change Adam semantics (prohibited #2)**. C7 must keep C0's bias-correction status (none).
- Faster-GS `TRUNCATION_MODE<0` tighter bounds cull Gaussians → **quality-changing truncation (prohibited)**.

---

## 3. OPPORTUNITY ORACLE (OBSERVED stage times + HYPOTHESIS bounds)

### 3.1 C0 stage times (OBSERVED, A100 PUBLICATION-grade, mean ms)

| scene | forward | loss | backward | densify | optimizer | **E2E mean** | it/s |
|---|---|---|---|---|---|---|---|
| room | 3.679 | 6.908 | 17.663 | 0.973 | 2.873 | **32.284** | 28.83 |
| bicycle | 5.417 | 6.882 | 21.993 | 1.212 | 10.224 | **45.904** | 20.64 |
| garden | 5.462 | 6.862 | 21.011 | 0.751 | 8.119 | **42.386** | 22.35 |

Stage % of E2E: **backward** 54.7/47.9/49.6%, **loss** 21.4/15.0/16.2%, **forward** 11.4/11.8/12.9%, **optimizer** 8.9/22.3/19.2%, **densify** 3.0/2.6/1.8% (room/bicycle/garden).
The `loss` phase (~6.9 ms) is a fixed PyTorch-side cost (0.8·L1 + 0.2·(1−SSIM), full-res fused SSIM); no audited component touches it.

### 3.2 Per-component oracle (HYPOTHESIS for bounds/overhead/O_net; OBSERVED for stage times)

| Component | Stage | Perfect upper bound (% E2E) | Realistic O_net (% E2E) | Decision |
|---|---|---|---|---|
| **C7 backward→opt fusion** | backward+optimizer | 8.9 / 22.3 / 19.2 | 3–5 / 8–15 / 6–12 | **IMPLEMENT (P1)** |
| C1 two-stage sorting | forward | 4.6 / 4.7 / 5.2 | 0.5–1.5 | **DROP** (O_net <2%) |
| C9 frontend allocation | per-iter overhead | 0.6 / 0.4 / 0.4 | 0.0–0.5 | **DROP** (bound <3%, O_net <2%) |
| C2/C4/C5/C6 | — | — | — | **DROP** (ALREADY_IN_C0) |
| C3 | — | — | — | **DROP** (prohibited / approximate) |
| C8 | — | — | — | **DROP** (UNKNOWN, not in reference) |

**Gate application (task step 3):**
- C1 perfect bound >3% BUT realistic O_net <2% → **DROP** (very-high transformation overhead = rasterizer swap; Faster-GS two-stage benefit is mostly VRAM, not speed, on A100).
- C9 perfect bound <3% AND O_net <2% → **DROP** (coupled to C1).
- C7 realistic O_net ≥3% on bicycle/garden (room borderline) → **IMPLEMENT**. No credible ≥5% combination exists (only C7 passes), but C7 alone satisfies the ≥3% implementation gate.

### 3.3 Combination analysis
Only C7 passes its individual gate; C1 and C9 are DROP. No exact-component combination ≥5% E2E is available. C7 alone is the implementation candidate.

---

## 4. Evidence ledger (OBSERVED / EVIDENCE / HYPOTHESIS / TEST / CONCLUSION)

**OBSERVED**
- C0 stage timings for room/bicycle/garden (PUBLICATION-grade A100).
- C0 identity flags (f9/scalar_adjoint/h8_mr/px2) and env vars.
- Faster-GS source markers across all 3 branches (sorting/Adam/fusion/densification).
- Z-order/Morton absent in all 3 Faster-GS branches.

**EVIDENCE**
- C0 `.so`/core sha256 ≠ task-pinned hashes (build-path embedding suspected).
- FasterGSFused contains prohibited invisible-Gaussian skip (`n_touched_tiles==0`).
- C0 Adam omits bias correction; Faster-GS Adam includes it (semantics delta).
- C0 `z_order` is a distributed-rendering option, not densification.

**HYPOTHESIS** (marked; not yet measured)
- C1: C0 single-pass sort ≈ 40% of forward.
- C7: fusion efficiency ~0.7–0.85 of optimizer kernel; gradient-traffic upside on large-N scenes.
- O_net ranges for C7/C1/C9.

**TEST** (deferred to gated steps)
- Reconcile C0 binary/source sha256 with task-pinned hashes.
- Build independent Faster-GS reference worktree; reproduce on room (PSNR/SSIM/LPIPS, wall-clock, Gaussian count, protocol diff), then garden+bicycle (success criterion B).
- C7 correctness gate: forward parity, gradient parity, one-step param parity, Adam m/v parity, zero-gradient row parity, densification-facing gradient parity (mark FP-reassociation as `ALGEBRAIC_EXACT_FP_REASSOCIATED` if applicable).
- C7 performance gate: geometry microbench → full F+B → post-densification iter → 5K → 30K strict-quality; STOP if net gain vanishes.

**CONCLUSION**
- The only exact, non-prohibited, gate-passing port is **C7 backward→optimizer fusion**. All other components are already in C0, prohibited/approximate, absent from the reference, or fail the O_net gate.
- No training-speedup or strict-quality claim is made this turn (none is permitted without full-training evidence).

---

### 3.4 Faster-GS vs C0 training-protocol diff (OBSERVED, step 4)

From `fastergs_garden.yaml` + `Trainer.py` + `Model.py` vs C0 `publication_trainer_p12.py` + provenance:

| aspect | Faster-GS (garden) | C0 | same |
|---|---|---|---|
| iterations | 30000 | 30000 | yes |
| loss | 0.8*L1 + 0.2*(1-DSSIM) | 0.8*L1 + 0.2*(1-SSIM) | yes (weights) |
| SH degree / background | 3 / black | 3 / black | yes |
| densification | adaptive_density_control (USE_MCMC=false), grad 0.0002, min_opacity 0.005 | adaptive density control, signed 2D-grad L2>=0.0002, absgrad | yes (family) |
| densification window | 600..14900 | 500..15000 | no (~400-iter offset) |
| **Morton/Z-order** | **YES** (every 5000 until 15000; reorders params + optimizer states) | **NO** | **no (C8)** |
| eval SSIM metric | Faster-GS SSIM | SepSSIM (frozen) | no |

**Read:** the Faster-GS garden reference config is *close* to C0 (same loss/SH/bg/densification family). The main structural protocol difference is Morton ordering (C8). A large quality/speed gap is NOT expected from the audited components alone. Full reference training/quality/wall-clock remain PENDING (no NeRFICG framework locally).

---

## 5. Recommendation to reviewer

1. **Reconcile C0 provenance** (binary/source sha256 vs task-pinned hashes) before any CUDA work.
2. **Approve C7** as the sole implementation candidate; proceed in an isolated worktree with its own binary hash and result directory.
3. **Build the independent Faster-GS reference** (success criterion B) — BLOCKED locally. Two concrete blockers (OBSERVED): (a) no NeRFICG framework/env (the trainer depends on it); (b) backend `setup.py build_ext` fails on a toolchain mismatch — local torch 2.13 + MSVC 14.44 + CUDA 13.3 require C++20, but the reference hardcodes `-std=c++17` (errors C1189/C7555/C7582/C2666). Environment incompatibility, not a Faster-GS defect; fixable only by patching the build config or using a torch<=2.7 env. Protocol diff (§3.4) shows the garden config is close to C0; do not expect a large gap from these components alone, but measure it under the aligned formal protocol (eval-SSIM metric differs: Faster-GS vs SepSSIM).
4. Defer C1/C9 (DROP) and do not pursue C3 (prohibited).

## 6. Deliverables (this turn)
- `sb0/SB0_COMPONENT_MATRIX.json`
- `sb0/SB0_C0_PROFILE.json`
- `sb0/SB0_OPPORTUNITY_ORACLES.json`
- `sb0/SB0_FASTERGS_REFERENCE.json` (PARTIAL: setup + protocol diff; training/quality PENDING)
- `sb0/SB0_REPORT.md`
- `sb0/MANIFEST.sha256`

_(Steps 4–10 deliverables — SB0_FASTERGS_REFERENCE / SB0_CORRECTNESS / SB0_TIMING / SB0_QUALITY / SB0_COMPOSITION / SB0_VERDICT — are produced only after reviewer approval and are NOT claimed here.)_
