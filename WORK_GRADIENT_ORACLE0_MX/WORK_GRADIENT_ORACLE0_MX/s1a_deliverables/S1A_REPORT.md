# S1A Report — Cost-Optimal Gradient Supervision: Math Correctness & Fixed-State Gradient Variance

**Round**: S1A (pilot). **Node**: MX `bms-39468022-001`. **GPU**: A100-PCIE-40GB cc[8,0] (cuda:6). **Torch**: 2.9.1+cu128, CUDA 12.8, Python 3.10.21. **gsplat upstream** `77ab983` + HiGS patch. **Authorization**: S1A only — NO S1B, NO 3K/30K, NO CUDA rebuild.

---

## 0. TL;DR / VERDICT

| Gate | State |
|---|---|
| **S1A-0 HT estimator math** | **PASS** |
| **S1A-2 cost calibration** | **PASS** |
| **§2 render sanity (RGB/alpha)** | **FAIL** |
| **S1A-1 gradient value** | **DERIVED only** (estimator-efficiency math; render sanity failed) |
| **Overall scientific state** | **S1A_MATH_PASS + render sanity FAIL → value comparison & GO BLOCKED (S1A_BLOCKED_INPUT)** |

**Conclusion**: The HT gradient estimator is mathematically correct and unbiased (S1A-0 PASS). The sparse tile kernel delivers real structural cost savings (73%/59% isect reduction). HOWEVER, the bicycle 5K checkpoint does **not** reconstruct the GT image with the available gsplat-HiGS renderer (corr ≈ 0.08, PSNR ≈ 10 — error larger than signal). Per task §2, the formal image-reconstruction gradient-value comparison and any GO are **blocked**. S1A-1 is delivered as **DERIVED estimator-efficiency math only** and does **not** support a GO.

---

## 1. Environment Freeze (OBSERVED)

Inherited from S0-R2 / S1A_ENV_FREEZE.json. Source commit `77ab983ffe43420b2131669cb35776b883ca4c3c`; prebuilt `gsplat_scene_cuda.so` sha256 `e4b0bc5b…90627c8` (236744 bytes). `PYTHONNOUSERSITE=1`, `TMPDIR=/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp`, `TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext`, `XDG_CACHE_HOME=$TMPDIR/xdg`. Every HiGS script sets `sys.path.insert(0, "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda")` so the prebuilt ext loads (JIT fallback unavailable — no ninja on host).

---

## 2. Checkpoint Scientific Validity (FAILURE — OBSERVED)

**Checkpoint**: FasterGSFused `fp1_bicycle_2026-10-06-01-01-33/0005000.pt` (N=2,456,697, iter=5000). Secondary `0010000.pt` (N=3,849,060, iter=10000).

### 2.1 Conventions verified (OBSERVED — all standard gsplat)
- **Activations**: `scales=exp`, `opacities=sigmoid`, `quats=normalize` — match FasterGS Model.py and gsplat.
- **SH ordering**: `cat([sh_coefficients_0, sh_coefficients_rest])` → `[N,16,3]`, sh_degree=3.
- **Color/SH**: verified against FasterGS CUDA source `convert_sh_to_color` (`sh_utils.cuh:34`) — it is **exactly standard gsplat SH** (`0.5 + C0*sh_dc` + standard higher-order basis, view dir `normalize(position - cam_position)`). **No lighting model.** (The `sh_coefficients_0`=DIFFUSE / `sh_coefficients_rest`=SPECULAR naming in `Renderer.py:98-99` is just standard DC + higher-order SH.)
- **No PCA**: no `apply_pca`/inverse/PCA keys anywhere in checkpoint or Python/CUDA source. `APPLY_PCA` is a dead config field.
- **Camera**: COLMAP/OpenCV convention; projection test shows 75.1% in-frustum, 26.9% on-screen, on-screen gaussians span full image, median cam-z 4.28. Camera pose **correct**.

### 2.2 Render sanity (FAILURE — OBSERVED)
Render of real camera 0 (`_DSC8679`) vs real GT (`eval_images`):

| Config | PSNR | corr | rend_std | GT_std |
|---|---|---|---|---|
| 5K, native 0.25 (1236×821) | 10.45 | 0.084 | 0.236 | 0.211 |
| 10K, native 0.25 | 10.08 | 0.087 | — | — |
| 5K, small 0.08 (395×262) | 10.74 | 0.079 | — | — |
| **benchmark's own `_std_ll_forward`** | 10.45 | 0.084 | — | — |

Multi-camera: corr 0.086 / 0.178 / 0.106 / 0.253 (cams 0/2/5/8). **Gamma correction does not help.** rend_std ≈ GT_std but corr ≈ 0.08 → the render has the right global statistics but **wrong spatial structure** (noise-like). PSNR 10 ⇒ MSE 0.1 > GT variance 0.04 ⇒ error larger than signal.

**Ruled out**: camera pose (projection correct; **16-combo axis-flip grid search over means/camera axes — best corr only 0.296 with `flip_z`, far below usable threshold**; 76% of gaussians in front, OpenCV +z; `eval_cameras.json` confirmed standard COLMAP/OpenCV via `_mx_colmap_to_json.py` + FasterGS `Datasets/Base.py`), color convention (standard gsplat SH, verified vs FasterGS source), activations, PCA (absent), aliasing (gaussians ~10–15px, not sub-pixel), resolution/sh_degree/flip variants. The benchmark's **own reference render path** (`_std_ll_forward`) gives the same corr 0.08, so this is **not** a bug in the HiGS render call.

**Root cause (HYPOTHESIS, unconfirmed)**: the checkpoint's gaussians do not reconstruct these particular GT images with the available renderer — likely undertrained at iter 5K/10K for a "fused/condensed" representation, or `eval_images` are held-out views not in the FasterGS training split. Cannot be resolved without rebuilding/rerunning the FasterGS training pipeline (out of scope).

### 2.3 Consequence (per task §2 fallback)
Render sanity FAILED ⇒ **formal image-reconstruction gradient-value comparison and GO are blocked.** Pure-math HT smoke test (S1A-0) proceeds. Garden excluded (camera provenance unresolved; no stub COLMAP substitution allowed).

---

## 3. S1A-0 — HT Gradient Estimator Validation (PASS — OBSERVED)

Fixed real 5K checkpoint, camera 0, additive full-image-normalized L1, `L_j = Σ_{tile j}|RGB−GT|/(3HW)`. 128×128 render, 64 tiles, N=120,000 on-screen gaussians. HT estimator `ĝ = Σ_j (z_j/p_j) grad(L_j)`, `z_j~Ber(p_j)`.

| # | Check | Result | Gate |
|---|---|---|---|
| 1 | p=1 (all z_j=1) == dense | rel_err ~1e-7 (all 5 groups) | PASS |
| 2 | Partition additivity Σ_j grad(L_j) == dense | rel_err ~1e-7 (all 5 groups) | PASS |
| 3 | MC consistency p=0.25 (300 trials) | cos 0.995–0.999 (5 groups) | PASS |
| 3 | MC consistency p=0.40 (300 trials) | cos 0.996–0.999 (5 groups) | PASS |
| 4 | Loss normalization Σ_j L_j == L_full | rel_err 8e-8 | PASS |
| 4 | Boundary tile pixel counts | corners=256, edges=256, interior=256 (128×128, tile=16 → 8×8 uniform) | recorded |
| 5 | 5 param groups receive grad | means/quats/scales/opacities/colors all nonzero | PASS |
| 6 | signed vs absgrad vs densification | signed_dense_norm=4.70e-2, absgrad=7.70e-2 (distinct, not mixed) | PASS |
| 7 | Unbiasedness + independence | MC mean ≈ dense within sampling CI; analytic Var = Σ(1−p_j)/p_j²‖g_j‖² | PASS |

**Not** substituted by single cosine≈1: unbiasedness verified via MC mean convergence + analytic HT variance, cos reported per group. Dense grad norms: means=4.70e-2, quats=6.27e-3, scales=8.82e-2, opac=4.02e-2, colors=1.28e-2. **S1A_0 overall PASS.**

---

## 4. S1A-1 — Cost-Aware Gradient Oracle (DERIVED — NOT value)

**LABEL: DERIVED.** SUBSET_ORACLE: 64-tile analysis subset, N=120,000. Per-tile gradient norms `‖g_j‖`, per-tile L1 error `|L_j|`. Analytic HT variance (independent Bernoulli + fixed objective — premises hold). **Render sanity failed ⇒ this is estimator-efficiency math on a fixed objective, NOT training value, does NOT support GO.**

### 4.1 Cost model (from S1A-2, OBSERVED)
`C(ratio) = C_fixed + C_var·ratio`. Fit: C_fixed=0.042ms, C_var=0.008ms/ratio. **Fixed overhead = 84% of total** (projection + sort setup + memory alloc dominate at this small scale).

### 4.2 Primary NMSE (means group), p=0.25 / p=0.40

| Strategy | p=0.25 NMSE | p=0.40 NMSE | Δ vs U (p=0.25) |
|---|---|---|---|
| U — Uniform Bernoulli | 13.84 | 4.32 | — |
| E — Error-guided | 23.51 | 7.88 | **+70% (worse)** |
| G — Gradient-only (`p_j ∝ ‖g_j‖`) | 8.57 | 2.72 | **−38%** |
| O — Cost-aware oracle (`p_j ∝ ‖g_j‖/√c_j`) | 8.57 | 2.72 | **−38%** |

### 4.3 Interpretation (DERIVED / HYPOTHESIS)
- **O == G identically**: because 84% of cost is fixed overhead, the `√c_j` per-tile cost term is negligible → cost-awareness provides no differentiation at this scale. The task's requirement to test the oracle candidate under real shared projection/sorting/rasterization overhead is **satisfied**: the candidate does NOT help beyond gradient-only here.
- **G/O beat U by ~38% NMSE reduction**: gradient concentration is real — top 25% tiles capture 72% of `‖g_means‖²`. Sampling gradient-heavy tiles reduces estimator variance.
- **E is worse than U**: per-tile render error does NOT track per-tile gradient magnitude (because render is noise, corr 0.08). Error-guided inclusion is counterproductive on this objective.
- **Caveat**: these NMSE values are large (>1) because the objective itself is noise-like (render sanity failed). The 38% reduction is a relative estimator-efficiency result, not an absolute quality claim.

---

## 5. S1A-2 — Real GPU Cost Calibration (PASS — OBSERVED)

Same ckpt/camera/kernel/loss. CUDA events + host wall. 30 warmup, 100 timing iters, 3 paired repeats, alternating order. 396×264, N=150,000.

| Config | fwd (ms) | bwd (ms) | F+B (ms) | n_isects | tile_ratio | F+B speedup | isect reduction |
|---|---|---|---|---|---|---|---|
| full | 0.016 | 0.034 | 0.050 | 260,221 | 1.000 | 1.00× | — |
| sparse p=0.25 | 0.014 | 0.031 | 0.044 | 71,387 | 0.249 | 1.14× | 72.6% |
| sparse p=0.40 | 0.014 | 0.032 | 0.047 | 107,386 | 0.400 | 1.10× | 58.7% |

n_isects scale linearly with tile ratio. **Modest F+B speedup** (1.10–1.14×) because at this small scale fixed per-call overhead dominates over sparse savings. The **structural win is the 73%/59% isect reduction**; runtime speedup will be larger at production scale (more gaussians/tiles, where variable cost dominates). Stable across 3 paired repeats. Mark OBSERVED (not STRUCTURAL_ORACLE — profiling completed).

---

## 6. GO / DROP Assessment (CONCLUSION)

**PROMISING conditions** (task §6): Ideal Oracle vs strongest comparable baseline at equal expected budget + comparable real GPU work: aggregate MSE ↓≥30%, no key-group degradation, direction consistent on ≥2 bicycle states × ≥2 cameras, no hidden extra runtime cost.

**Assessment**:
- Oracle (O) vs Uniform (U): −38% NMSE at p=0.25 and p=0.40 (≥30% threshold met) **on the DERIVED estimator math**.
- BUT: (a) render sanity FAILED → the objective is noise-like, so the gradient is not meaningful for training; (b) the 38% is estimator-efficiency, not demonstrated training-value; (c) only ONE bicycle state (5K), ONE camera analyzed in S1A-1 (the SUBSET_ORACLE is 64 tiles of one camera); (d) O provides no advantage over G (cost-awareness negligible due to fixed overhead).
- The task requires "至少两个真实 bicycle 状态和两个训练 camera 上方向一致" — **not met** (single state, single camera in S1A-1).

**Verdict: S1A_MATH_PASS (S1A-0) + render sanity FAIL → S1A_BLOCKED_INPUT.**

The cost-aware gradient sampling **estimator** is correct and the sparse kernel saves real isects, but this round **cannot determine whether there is a worth-exploiting cost-aware gradient sampling ADVANTAGE for training**, because the checkpoint does not reconstruct the GT image with the available renderer. The formal value comparison is blocked by the render sanity gate (§2).

**Recommendation**: Do NOT issue S1B based on this round. To unblock: (1) obtain a checkpoint/renderer combination that passes the RGB/alpha render sanity test (e.g., confirm `eval_images` are in-training views, or rerun FasterGS training to a converged state with the gsplat-HiGS renderer), then re-run S1A-1/S1A-2 on ≥2 bicycle states × ≥2 cameras. The S1A-0 estimator math and S1A-2 cost calibration remain valid and reusable.

---

## 7. Provenance Labels

- **OBSERVED**: env freeze, convention checks, render sanity numbers, S1A-0 gates, S1A-2 timing/isects, S1A-1 NMSE table.
- **DERIVED**: S1A-1 analytic HT variance, NMSE reductions, cost model fit, gradient concentration.
- **HYPOTHESIS**: render-sanity root cause (undertrained / held-out views); why speedup is modest at small scale.
- **FAILURE**: §2 render sanity (corr 0.08, PSNR 10).
- **CONCLUSION**: S1A_MATH_PASS + render sanity FAIL → value comparison & GO blocked (S1A_BLOCKED_INPUT).

All numbers traceable to `raw/S1A_HT_CORRECTNESS.json`, `raw/S1A_COST_CALIBRATION.json`, `raw/S1A_GRADIENT_VALUE.json`, and the CSVs. No synthetic Gaussians used for formal experiments; no old data reused as fresh.
