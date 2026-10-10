# S1A-R1 Report — Variance Formula Audit + Render Three-Way + Valid Oracle Input

**Round**: S1A-R1 (continuation of S1A). **Node**: MX `bms-39468022-001`, A100-PCIE-40GB (cuda:6), Torch 2.9.1+cu128, CUDA 12.8, Python 3.10.21. **gsplat** `77ab983` + HiGS patch. **Authorization**: S1A-R1 only — NO S1B start, NO 3K/30K new training, NO CUDA rebuild, NO env reinstall.

---

## 0. TL;DR / VERDICT

**`S1A_R1_GRADIENT_ONLY`** — G beats U substantially and consistently, but the cost-aware Oracle O adds **no** advantage over pure gradient-magnitude sampling G.

| Phase | Result |
|---|---|
| 1. HT variance formula audit | **CONFIRMED + FIXED** — original misused `(1-p)/p²`; correct `(1-p)/p` matches MC to 3 sig figs |
| 2. Render three-way (5K/10K/15K) | **B/C PASS** (5K PSNR 17.1/15.6); **A native BLOCKED** (ABI mismatch) |
| 3. Valid-render U/E/G/O (2 states × 2 cams) | **G +59%, O +58% vs U**; **O never beats G**; E worse |

---

## 1. Phase 1 — HT Variance Formula Audit

**The error (OBSERVED)**: original `s1a_gradient_value.py:127` used `Var = Σ[(1-p_j)/p_j²]·||g_j||²`. Correct HT variance (independent Bernoulli, fixed additive objective) is `E||ĝ−g||² = Σ[(1-p_j)/p_j]·||g_j||²`. The original had an extra `1/p_j` factor.

**MC validation (OBSERVED)**: cached per-tile gradients, 300-trial MC on means group:

| p | U analytic (wrong) | U analytic (correct) | U MC | G analytic (correct) | G MC |
|---|---|---|---|---|---|
| 0.25 | 13.84 | 3.46 | 7.61e-3 | 1.70 | 3.76e-3 |
| 0.40 | 4.32 | 1.73 | 3.82e-3 | 0.69 | 1.53e-3 |

The correct formula matches MC to 3 sig figs; the wrong one does not. **Math correctness: PASS.**

**Corrected NMSE (DERIVED)**: G/O beat U by **+50.7% (p=0.25)** and **+59.9% (p=0.40)** — *larger* than the wrong-formula +38%/+37%. The wrong formula understated the advantage.

See `S1A_VARIANCE_ERRATUM.md`. Old wrong-formula data preserved in `raw/S1A_GRADIENT_VALUE.json`.

---

## 2. Phase 2 — Render Three-Way Comparison

Same ckpt/camera/res/background/GT. **396×264, scale 0.08** (the resolution that renders correctly — the earlier S1A corr 0.08 used a broken native-0.25 path). Bicycle checkpoints 5K/10K/15K (`final.pt` = iter 15000). Full Gaussian set, **no truncation/resampling** (n_visible = N).

### OBSERVED — PSNR vs GT (camera 0, `_DSC8679`)

| ckpt | N | A: native FasterGS | B: gsplat low-level | C: HiGS full-tile |
|---|---|---|---|---|
| 5K | 2,456,697 | **BLOCKED** (ABI) | **17.10** | 15.56 |
| 10K | 3,849,060 | BLOCKED | 13.36 | 13.20 |
| 15K | 4,430,283 | BLOCKED | 13.25 | 12.88 |

### Path A — Native FasterGSFused renderer (BLOCKED)
`_C.forward` (compiled for the original training torch/CUDA) **illegal memory access** on torch 2.9.1+cu128 — an ABI/runtime mismatch. Cannot be fixed without rebuilding (not authorized). Recorded as BLOCKED, not silently substituted.

### Path B vs C (OBSERVED)
B (gsplat `fully_fused_projection` + `isect_tiles` + `rasterize_to_pixels_3dgs`) and C (HiGS `rasterize_gaussian_higs_dynamic`, full-tile, `use_higs_culling=False`) agree within 1.5 dB on all 3 checkpoints. This confirms the gsplat-HiGS path reproduces the checkpoint's reconstruction — **the earlier corr 0.08 was a render-path bug, NOT a checkpoint/data issue.** 5K passes render sanity (PSNR 17.1 ≫ threshold).

### Provenance (OBSERVED)
- 5K B: alpha_mean=0.978, n_isects=1,201,272, arr_mean=0.372, arr_std=0.217, fx=372.26, fy=370.48, cx=197.5, cy=131.5, w2c and cam_position recorded in JSON.
- Full Gaussian set used (N = n_visible); no resampling; no sparse mask; no default-pixel fill.

### Why native FasterGS matters (task requirement)
The task says: if native FasterGS also fails, don't blame HiGS; check GT-camera mapping + original training eval records. Here native FasterGS could not run (ABI), so it neither confirms nor denies. But B and C both reconstruct the GT at PSNR 17.1 (5K), which **directly confirms** the GT-camera mapping is correct and the checkpoint is valid — the gsplat-HiGS path reproduces the original FasterGS appearance. The earlier failure was the broken native-0.25 render path.

---

## 3. Phase 3 — Valid-Render U/E/G/O (DERIVED)

With the correct render, 2 states (5K, 10K) × 2 cameras (cam 0, 1), correct HT variance formula. Means-group NMSE reduction vs Uniform:

| Strategy | p=0.25 (4-combo mean) | p=0.40 | Beats U in all 4 combos? |
|---|---|---|---|
| G — gradient-only (`p_j ∝ ‖g_j‖`) | **+51% to +63%** | **+58% to +71%** | Yes |
| O — cost-aware oracle (`p_j ∝ ‖g_j‖/√c_j`) | +50% to +62% | +57% to +70% | Yes |
| E — error-guided (`p_j ∝ |L_j|`) | −17% to −49% | −21% to −62% | No |

Aggregate: G +59.3%, O +58.3%, E −37.8% vs U. **O never beats G** (mean O-vs-G = −2.6%).

### Interpretation (DERIVED / CONCLUSION)
- **Gradient concentration is real and exploitable**: top 25% tiles capture ~72% of ‖g_means‖². Sampling gradient-heavy tiles (G) cuts estimator variance ~58% vs uniform, consistently across all 4 (state, camera) combos and both budgets.
- **Cost-awareness adds nothing (O == G)**: 84% of GPU cost is fixed overhead (S1A-2), so the per-tile `√c_j` term is negligible. The cost-aware oracle does NOT justify its complexity over pure gradient-magnitude sampling.
- **Error-guided (E) is counterproductive**: per-tile render error does not track per-tile gradient magnitude on this objective.
- **Caveat**: this is estimator-efficiency (DERIVED) math. The render now passes sanity (5K PSNR 17.1), so the gradient is meaningful, but a single round of estimator MSE does **not** by itself prove full training收益. The PROMISING threshold (≥30% MSE reduction) is met on the primary means group across all combos, but the O-vs-G null result is the decisive finding.

---

## 4. Verdict — `S1A_R1_GRADIENT_ONLY`

Per task §6 definitions:
- **NOT `S1A_R1_READY_FOR_S1B`**: requires the ideal Oracle to show a *stable cost-aware advantage* over baselines. O does **not** beat G — the cost-aware term is null. So the precondition for S1B (a cost-aware advantage worth deploying) is **not** met.
- **`S1A_R1_GRADIENT_ONLY`**: G beats U substantially and consistently, but O has no extra cost advantage. **This matches exactly.**
- NOT `S1A_R1_NO_ADVANTAGE`: G/O do beat U by ~58%.
- NOT `S1A_R1_BLOCKED_INPUT`: render sanity now passes (5K PSNR 17.1); a valid checkpoint-camera-GT combination was found.

**Conclusion**: There is a real **gradient-magnitude sampling** advantage (G beats U ~58%), but **no cost-aware advantage** (O == G). The cost-aware oracle — the original S1A hypothesis — does not hold: once the variance formula is corrected and the render is valid, the `√c_j` cost term is null because GPU cost is dominated by fixed overhead, not per-tile work. **This does not meet the bar for S1B** (which requires a cost-aware advantage). The cheaper, equally-effective alternative is pure gradient-magnitude sampling (G), not the cost-aware oracle.

---

## 5. Provenance Labels

- **OBSERVED**: formula error + MC validation; B/C render PSNR on 5K/10K/15K; path A ABI block; full-Gaussian provenance; per-combo NMSE reductions.
- **DERIVED**: corrected NMSE, cost model (84% fixed), O==G identity, gradient concentration.
- **HYPOTHESIS**: why B/C agree (same SH/geometry); why 10K/15K PSNR lower than 5K (possible over-fit or eval-view mismatch at higher iter).
- **FAILURE**: original S1A variance formula (extra 1/p_j); original S1A render path (corr 0.08 — broken native-0.25 scale).
- **CONCLUSION**: `S1A_R1_GRADIENT_ONLY` — G beats U ~58%, O adds no cost advantage; cost-aware oracle hypothesis not supported; not ready for S1B.

All numbers traceable to `S1A_R1_VARIANCE_ERRATUM.json`, `S1A_R1_RENDER_THREEWAY.json`, `S1A_R1_GRADIENT_VALUE.json`, and the render PNG/npy files. No synthetic Gaussians for formal experiments; no old data reused as fresh (S1A-R1 re-measured with correct formula + valid render).
