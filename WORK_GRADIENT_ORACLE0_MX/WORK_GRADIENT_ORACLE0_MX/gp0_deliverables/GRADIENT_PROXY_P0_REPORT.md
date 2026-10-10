# GRADIENT-PROXY-P0 Report — Cheap Tile Gradient Importance Estimators

**Round**: GRADIENT-PROXY-P0 (continuation of S1A-R1, current state `S1A_R1_GRADIENT_ONLY`). **Node**: MX `bms-39468022-001`, A100-PCIE-40GB (cuda:6), Torch 2.9.1+cu128, CUDA 12.8, Python 3.10.21. **gsplat** `77ab983` + HiGS patch. **Authorization**: P0 feasibility only — NO 3K/30K training, NO CUDA modification, NO runtime rebuild. NOT a Cost-Aware S1B pass.

---

## 0. VERDICT — `DROP` (all three proxy candidates)

| Proxy | NMSE reduction vs U (p=0.25 / p=0.40) | Held-out | Meets ≥25%? | Overhead vs sparse |
|---|---|---|---|---|
| G (true-grad oracle, NOT deployable) | **+49% / +60%** | +44% / +55% | n/a (upper bound) | **317×** (info cost) |
| H1 Historical Gradient | −9% / −9% (all), +10% / +12% (held-out) | mixed | **NO** | cache EMA (cheap) |
| H2 Geometry-Jacobian | **−373% / −546%** | −363% / −535% | **NO** | **34×** |
| H2b (residual×√count) | −78% / −89% | −80% / −93% | **NO** | — |
| H3 Hybrid (H1+H2+floor) | −18% / −16% | −6% / −1% | **NO** | — |
| E Error-guided (baseline) | −31% / −35% | −27% / −32% | **NO** | — |

**No deployable proxy meets the ≥25% NMSE-reduction threshold on the primary (means) group, consistently across states/cameras/held-out.** Per task §GO/DROP: "失败则 DROP 该代理，不得因为理想 G 很强就放宽条件." G is strong but **not deployable** (its true per-tile gradient needs 425 separate backwards = 1389 ms = 317× a sparse iteration). The cheap proxies cannot capture the gradient structure. **DROP H1, H2, H3.**

---

## 1. Phase 1 — Scientific Consistency (OBSERVED)

Same ckpt/camera/res/bg/full-Gaussian set; path B (gsplat low-level) vs path C (HiGS full-tile), 5K/10K/15K × cam0/cam1, 396×264 (scale 0.08).

**B vs C render**: near-identical. cam1 byte-identical (MAE 0.0, PSNR 120); cam0 MAE 0.020 / PSNR 32.3; 10K cam0 MAE 0.003 / PSNR 48.5. The two renderers reproduce the same image.

**B vs C alpha** (OBSERVED, supplement): cam1 byte-identical (MAE 0.0); cam0 MAE ≤0.0029, max error ≤0.10 (5K/10K/15K). Alpha通道同样高度一致。

**B vs C dense gradient (5-param cosine + relative L2)** (OBSERVED): near-ORTHOGONAL (means 0.005, quats 0.006, scales 0.07–0.17, opac 0.01–0.05, colors 0.02–0.06). Relative L2 between B and C gradients is large (≈1, i.e. near-orthogonal). Despite identical renders/alpha, the gradients disagree strongly → **the per-tile gradient is highly renderer-implementation-sensitive.** Consequence: all Phase 2/3 work uses HiGS (path C) consistently, since it is the deployable sparse renderer.

**5K→10K→15K PSNR drop**: the earlier "5K>10K>15K" narrative was an artifact of a wrong PSNR number (S1A-R1's 17.10 did not reproduce; recomputed from the saved .npy it is **10.72**). True PSNR vs GT at 396×264: 5K cam0 10.72, 10K cam0 10.32, 15K cam0 10.42; cam1 ~9.66 flat. **No monotonic drop** — PSNR is roughly flat/slightly noisy across iterations.

**Root cause of low PSNR + unclear source identity** (OBSERVED from training records): `training_config.yaml` shows `RUN_VALIDATION: false`, `RENDER_VALSET: false`, `RENDER_TESTSET: false` — **no eval PSNR was ever recorded**; `IMAGE_SCALE_FACTOR: 0.25` (training scale) ≠ 0.08 (this eval scale); `APPLY_PCA: true`. With no ground-truth training eval and a scale mismatch, source data identity is genuinely unclear. Per task §Phase1.5: math-proxy validation is allowed, but **no full training-quality benefit is claimed.**

**Visible vs total gaussians** (OBSERVED): N_total 2.46M/3.85M/4.43M; on-screen (this cam) 686K/1.25M/1.45M; visible with opacity>0.01 663K/1.21M/1.40M. **~27% of the model is on-screen for a single camera.** opacity_mean 0.237, median 0.151 (many low-opacity gaussians). Training ran 15000 iters @ 284 ms/iter, final 4.43M gaussians, peak VRAM 5.86 GiB.

---

## 2. Phase 2 — Cheap Proxies (DERIVED / OBSERVED)

Per-tile true gradient (HiGS per-tile backward, offline oracle G) + forward-only H2 geometry, for 9 combos (5K/10K/15K × cam0/1/2), on-screen subset (150K cap for memory; full set used for H2 projection). 425 tiles (16px, 396×264).

- **H1 — Historical Gradient Proxy**: per-tile importance = EMA of the TRUE per-tile `‖g_means‖` from PRIOR states (α=0.5), updated only after a state is processed (design states 5K,10K update history; 15K consumes 5K+10K history, never its own). **How each tile's independent gradient is obtained**: the true per-tile gradient is computed by rendering the full tile set and back-propagating each tile's image slice separately (`L_j = diff[:,tile].sum()`), so `g_j = ∂L_j/∂θ` is the genuine per-tile contribution — NOT an aggregate of per-Gaussian gradients. **Unsampled tiles are NOT zeroed**: H1 carries the historical EMA forward with an exploration floor (FLOOR=0.02), so a tile never sampled keeps its prior importance.
- **H2 — Geometry-Jacobian Proxy**: `importance_j = residual_j × conic_mass_j`, where `residual_j` = mean |render−GT| in tile j (from HiGS forward) and `conic_mass_j` = Σ over gaussians touching tile j of the per-Gaussian conic (2D inverse-covariance) norm, from a **single forward projection** (gsplat `fully_fused_projection`). **No dense backward** — reuses forward projection + render only.
- **H3 — Hybrid Proxy**: `0.5·normalize(H1) + 0.5·normalize(H2)` with exploration floor (FLOOR=0.02) so every tile has non-zero inclusion probability.

---

## 3. Phase 3 — Offline Oracle (OBSERVED)

Correct formula `E‖ĝ−g‖² = Σ_j[(1−p_j)/p_j]‖g_j‖²`; `NMSE = Var/‖g_means‖²`; budget-matched Bernoulli-HT (`p_j ∝ importance_j`, clipped to (FLOOR,1], renormalized to budget). Held-out: H1 designed on 5K/10K only; evaluation includes 15K (all cams) + cam2 (all states).

**Correlation diagnostic** (corr between true `‖g_j‖` and each cheap signal):

| combo | residual | opac_mass | cnt | conic_mass | resid×conic |
|---|---|---|---|---|---|
| 5K cam0 | −0.18 | 0.10 | 0.12 | 0.10 | 0.08 |
| 5K cam1 | −0.06 | 0.57 | 0.61 | 0.59 | 0.51 |
| 5K cam2 | −0.04 | 0.43 | 0.39 | 0.32 | 0.35 |
| 10K cam0 | −0.00 | 0.02 | 0.04 | 0.02 | 0.02 |
| 10K cam1 | 0.02 | 0.55 | 0.57 | 0.51 | 0.41 |
| 15K cam0 | −0.05 | 0.05 | 0.08 | 0.00 | 0.00 |
| 15K cam1 | 0.04 | 0.54 | 0.57 | 0.57 | 0.53 |
| 15K cam2 | −0.12 | 0.39 | 0.38 | 0.38 | 0.32 |

Cheap forward signals correlate at most **0.4–0.6** with the true per-tile gradient (and **≈0 on cam0**, where the gradient oracle G gives its largest advantage). The image `residual` is even **anti-correlated** on cam0. A linear correlation of 0.5 (R²=0.25) is insufficient to align the second-moment NMSE — hence the proxies underperform uniform.

---

## 4. Phase 4 — Deployability (OBSERVED, MX A100, CUDA events, 30 warmup / 100 timing)

| quantity | median (ms) | ratio vs sparse fwd+bwd |
|---|---|---|
| proxy H2 (projection + per-tile conic mass) | 74.4 | 17× |
| mask generation | 74.3 | 17× |
| **scheduling overhead (proxy+mask)** | **148.7** | **34×** |
| sparse forward (HiGS full-tile) | 1.53 | 0.35× |
| sparse fwd+bwd | 4.38 | 1.0× |
| **oracle info (true per-tile grad, 425 backwards)** | **1389** | **317×** |
| **VRAM increment (proxy projection, 150K subset)** | **+4.96 MiB** | **+2.7%** (on 184 MiB baseline) |

- The H2 proxy's 34× overhead is dominated by the Python per-tile loop (425 tiles). Even vectorized, it re-runs a projection; in a fully integrated pipeline it could reuse the renderer's forward projection, but the marginal cost would still be non-trivial.
- **The oracle's information-acquisition cost (1389 ms = 317× a sparse iteration) is reported explicitly, NOT hidden** — this is why G, despite +49–60% NMSE reduction, is not deployable.
- **VRAM increment is modest (+4.96 MiB, +2.7%)** — the proxy itself is memory-cheap; its problem is compute time, not memory.
- **No dense backward is used by any proxy** (H1 uses cached past gradients; H2/H3 use forward projection + render only). Confirmed `dense_backward_used_by_proxy=false`.
- VRAM delta not reliably captured (peak-stats reset timing); baseline allocation dominated by the 150K Gaussian tensors.

---

## 5. GO/DROP Evaluation vs Task Criteria

| # | Criterion | Result |
|---|---|---|
| 1 | proxy vs U, primary group NMSE ↓ ≥25% | **FAIL** (best deployable H1 +10–12% held-out but −9% on design; H2/H3 negative) |
| 2 | direction consistent across states/cameras | **FAIL** (H1 positive held-out / negative design; H2/H3 negative) |
| 3 | no severe key-group degradation | **FAIL** for H2/H2b/H3 (NMSE worse than U); H1 mixed |
| 4 | proxy overhead ≤5% of sparse iter | **FAIL** (H2 34×) |
| 5 | sampling prob / HT loss / renderer budget consistent | PASS (all budget-matched, same HT estimator) |
| 6 | no dense backward / future-grad / held-out GT leakage | PASS for proxies (H1 uses only past states; H2 forward-only; E uses current-training GT residual which is available in training) |

**Decision: DROP H1, H2, H3.** Criteria 1, 2, 3, 4 all fail for at least the leading candidates. The ideal G is strong but its 317× info cost disqualifies it (criterion 4 by a wide margin, and it is non-deployable by construction). Per the task, the conditions are **not relaxed** because G is strong.

---

## 6. Provenance Labels

- **OBSERVED**: B/C render MAE/PSNR; B/C dense-gradient near-orthogonality; PSNR vs GT (recomputed); visible/total counts; training config (RUN_VALIDATION=false, scale 0.25, 15000 iters); Phase 3 NMSE table; correlation diagnostic; Phase 4 timings.
- **DERIVED**: H1/H2/H3 proxy definitions; cost model; budget-matching.
- **CONCLUSION**: cheap forward-only proxies cannot predict the HiGS per-tile gradient well enough (corr ≤0.6, ~0 on cam0) to beat uniform by ≥25%; the gradient oracle advantage is real but its information cost (317×) is prohibitive. **DROP.**
- **FAILURE**: original S1A-R1 PSNR 17.10 (did not reproduce; true 10.72); H2 proxy formulation (opacity-based, then conic-based, both fail); H2 Python per-tile loop overhead.

## 7. Implication for S1B

The original Cost-Aware S1B hypothesis is **not** rescued by these cheap proxies. The gradient-only oracle advantage (G vs U ≈ +50–60%) is real but (a) requires the non-deployable true per-tile gradient, and (b) the cheap surrogates (historical / geometry-Jacobian / hybrid) do not capture it. **No P0→S1B promotion on this evidence.**

All numbers traceable to `GP0_PHASE1_CONSISTENCY.json`, `GP0_PHASE3_ORACLE.json`, `GP0_PHASE4_TIMING.json`, `GP0_SOURCE_IDENTITY.json`, `gp0_pertile_*.npz`, `gp0_pertile_meta.json`, and the source scripts. No synthetic Gaussians for formal results; held-out states/cameras used for proxy design; Oracle info cost reported, not hidden.

---

## 8. Source / Binary Identity (OBSERVED)

| item | value |
|---|---|
| gsplat source git | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| gsplat_scene_cuda.so sha256 | `e4b0bc5bf358ec43875a3b7e0d84591763968021b7c7eabcc6d60572490627c8` |
| gsplat_scene_cuda.so bytes | 236744 |
| torch | 2.9.1+cu128 |
| CUDA | 12.8 |
| Python | 3.10.21 |
| NVIDIA driver | 595.71.05 |
| GPU | A100-PCIE-40GB (cuda:6), sm_80 |

Recorded in `GP0_SOURCE_IDENTITY.json`. The prebuilt `gsplat_scene_cuda.so` is the S0-R2-frozen runtime; not rebuilt (per authorization).
