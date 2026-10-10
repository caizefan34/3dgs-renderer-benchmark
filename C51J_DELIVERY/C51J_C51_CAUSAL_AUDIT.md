# C51J_C51_CAUSAL_AUDIT.md

**Track:** CVPR27-C51-JOINT-MX — C51 Causal Evidence Audit (Stage 5 four-arm + R0.2 gradient Top-K)
**Generated:** 2026-10-10 (UTC)
**Source of record:** `reports/phase-c51-stage5.md`, `reports/phase-r0.2-c51-final-mechanism-gate.md`, `p0_c51_cd0_20261008/P0_VERDICT_TERMINAL.json`

---

## 0. Purpose

Provide Codex with a **real, self-consistent cost table** for the C51 mechanism, quantifying:
- A. full backward + full densification (Baseline)
- B. sparse backward + sparse densification (K50-B1)
- C. sparse backward + full densification signal
- D. full backward + masked optimizer

and re-verify the R0.2 historical gradient Top-K prediction-failure conclusion. **This is NOT the 132-job protocol** — see §6 for the hard boundary.

---

## 1. The four-arm design (verbatim from Stage 5)

All four arms use identical canonical 30K training (C45 moderate config). The only differences:

| Arm | Backward | Densification signal | Optimizer update |
|-----|----------|----------------------|------------------|
| **A Baseline** | Full | Full | All Gaussians |
| **B K50-B1** | Sparse (CUDA skip) | Sparse (masked=0) | Selected 50% only |
| **C Sparse+FullDens** | Sparse (CUDA skip) | Full (`compute_densify_grad=True`) | Selected 50% only |
| **D FullBwd+MaskedOpt** | Full | Full | Selected 50% only (zeroed) |

- **C** computes `v_means2d` for masked Gaussians → full `xyz.grad` for ALL → densification sees a baseline-like signal; masked `xyz.grad` is zeroed before `optimizer.step()`.
- **D** full backward, then Python-side grad zeroing after `accumulate_positional_gradient()`, before `optimizer.step()`. Isolates "computation skipped" (B/C) from "computed then discarded" (D).

---

## 2. Real cost table — Room (2.16M GS, 30K, A100-PCIE-40GB)

| Component (ms) | A | B | C | D | C−A | D−A |
|----------------|----|----|----|----|------|------|
| Fwd+Bwd | 49.49 | 46.49 | 48.62 | 49.10 | +0.86 | +0.39 |
| Optimizer | 7.31 | 5.29 | 7.75 | 7.68 | −0.44 | −0.37 |
| Densification | 0.11 | 0.07 | 0.10 | 0.12 | 0.00 | −0.01 |
| Mask overhead | 0.00 | 0.678 | 0.715 | 0.704 | −0.715 | −0.704 |
| **Total (ms)** | **57.25** | **52.90** | **56.96** | **57.48** | **+0.30** | **−0.23** |

**Decomposition (Room):**
- B total speedup **+8.2%**; C intrinsic **+0.5%**; D **−0.4%**.
- **Densification-induced (B−C) = +7.7% = 94% of B's speedup.**
- CUDA-skip benefit (C vs D) = **0.52 ms** = pure computation saved.
- Mask overhead (0.715 ms) consumes **83%** of the 0.86 ms kernel saving.

---

## 3. Cross-scene cost table (the arm C never passes the gate)

| Scene | A (ms) | B (ms) | C (ms) | B speedup | C intrinsic | C passes 5%? | C GS ratio |
|-------|--------|--------|--------|-----------|-------------|--------------|------------|
| Room (2.16M) | 57.25 | 52.90 | 56.96 | +8.2% | +0.5% | ❌ | 94.6% |
| Bicycle (11.05M) | 120.48 | 96.82 | 128.93 | +24.4% | **−6.6%** | ❌ | **111.9%** |
| Garden (11.40M) | 119.45 | 92.70 | 121.49 | +28.9% | **−1.7%** | ❌ | 99.7% |

**Two independent reasons C fails:**
1. **Mask overhead dominates** at small N (Room): 0.715 ms mask vs 0.86 ms saved.
2. **B2 path (`compute_densify_grad=True`) overhead exceeds kernel savings** at large N (Bicycle/Garden): C becomes *slower* than A.

**C also over-densifies on Bicycle (111.9% of baseline GS)** — the B2 gradient signal diverges enough to trigger extra densification, contaminating the "population preserved" premise on that scene.

---

## 4. Population trajectory (Room) — the causal link

| Iter | A | B | C | D |
|------|------|---------|---------|---------|
| 0 | 1,593,376 | 1,593,376 | 1,593,376 | 1,593,376 |
| 5000 | 1,436,880 | 1,533,388 | 1,616,358 | 1,615,787 |
| 10000 | 1,678,255 | 1,508,125 | 1,775,254 | 1,780,830 |
| 15000 | 2,255,579 | 1,587,849 | 2,147,720 | 2,151,147 |
| 30000 | 2,155,755 | 1,505,069 | 2,039,821 | 2,042,608 |

- B diverges after iter 5000 and plateaus at ~70% of A. C/D track A closely (94.6%/94.8%) → **semantic equivalence of C and D** confirmed by trajectory overlap.
- **C vs D equivalence (Room):** PSNR 29.94 = 29.94; SSIM 0.8882 vs 0.8883; GS Δ −2,787 (0.1%); Clone Δ −2,587; Split Δ −1,465; time Δ −0.52 ms. The only difference is the CUDA skip.
- Densification events: C retains 68.1% clone / 88.4% split vs baseline (vs B's 14.9% / 45.8%). The residual gap is the EMA predictor's imperfect recall (recall@50 = 95.7%, not 100%).

---

## 5. R0.2 — historical gradient Top-K prediction failure (re-verified)

**Conclusion: historical C51 (previous-iteration xyz-gradient Top-K masking) is definitively unsupported.**

| Predictor | K=50% coverage of current G_opt mass | vs Random |
|-----------|:-------------------------------------:|:---------:|
| Oracle (TopK of G_opt(t)) | 1.000 | — |
| **Previous (TopK of G_opt(t−1))** | **0.528** | +0.028 |
| Random | 0.500 | — |
| xyz_norm previous (historical C51 actual scalar) | 0.521 | +0.021 |

- Mechanism: camera visibility changes between iterations. The previous Top-50% is ~50% visible + ~50% invisible; invisible contributes zero G_opt at the current camera, so previous coverage collapses to ≈ random.
- **G_dens (means2d.absgrad), a same-iteration byproduct, is the only strong predictor:** Top-50% by G_dens covers **92.7%** of G_opt mass (Spearman 0.858, oracle 0.957). But it is available **after** the dominant rasterization backward (75% of backward, 30% of total E2E).
- **Amdahl ceilings:** G_dens-gated SH-backward + optimizer = **15.8% E2E ceiling**; forward-time signal (opacity/tiles_per_gauss) = 45.7% ceiling but **untested G_opt correlation**; historical C51-in-raster = 12.0% ceiling.

This is the exact reason Stage-5 condition C uses the B2 path (`compute_densify_grad=True`) — to reconstruct a full densification signal — and why that path's overhead then erases the speedup.

---

## 6. PROTOCOL BOUNDARY — do NOT merge with 132-job (mandatory)

| Dimension | C51 Stage-5 (this audit) | 132-job confirmatory |
|-----------|--------------------------|----------------------|
| Scenes | 3 (room/bicycle/garden) | 11 |
| Seeds | 1 | 3 |
| Purpose | mechanism attribution (A/B/C/D ablation) | pre-registered confirmation |
| Candidate | K50-B1 joint sparse-bwd+dens | `gsplat_30k_fused_prune10_rclip05` |
| Result | +8.2/+24.4/+28.9% (94% densification-induced) | 1.164× paired |
| Hardware | A100-PCIE-40GB | A100 (40/80GB per report) |

**Rule enforced:** the C51 K50-B1 speedup is **NOT** independent confirmation of the 132-job 1.164×. Different scenes, seeds, candidate, and gate structure. The two live in separate evidence ladders (skill §29). The `C51J_WORKLOAD_COST.csv` tags every row with its source stage.

---

## 7. Novelty disposition (from P0 terminal verdict, carried forward)

`P0_VERDICT_TERMINAL.json` → **`P0_NO_NOVELTY`** for the *specific* C51-CD0 co-design hypothesis (joint gradient-spend + capacity-growth coupling):
- Backward lever falsified: C vs A = +0.5% / −6.6% / −1.7% (Stage 5).
- Capacity control = FastGS prior art; topology/capacity scheduling = CLOSED (VTS0 / ReBudgetGS).
- Scope: this C51-CD0 design only, **not** the whole 3DGS-training-acceleration direction.

---

## 8. What remains uncertain / open

1. **No CUDA kernel profiling** — Stage-5 fwd+bwd numbers are Python-level timings; occupancy/registers/memory-transactions not measured.
2. **Condition D only on Room** — C-vs-D semantic equivalence on bicycle/garden not run (GPU-constrained at the time). Bicycle over-densification (111.9%) makes that scene's C-arm attribution the least clean.
3. **Quality-improvement mechanism of C (Room +0.74 dB)** is not formally explained; hypothesized regularization effect, untested.
4. **LPIPS not available** in the C51 Stage-5 framework — no perceptual-quality claim for C/B.
5. **LPIPS regression source in the 132-job** (prune vs rclip) cannot be localized — per-scene LPIPS raw data absent (see baseline audit §2.3).

---

## 9. Verdict for Codex

- Use `C51J_WORKLOAD_COST.csv` as the canonical cost table.
- The **only** reproducible speed in the C51 design is the **densification-induced population reduction** (arm B), not the backward kernel. Any future candidate must beat the **B2-path overhead** and the **mask overhead** to make the intrinsic sparse-backward worthwhile — currently a −0.715 ms tax for +0.86 ms of kernel saving, net +0.30 ms (Room) and net-negative at large N.
- The R0.2 ceiling stands: a same-iteration forward signal with verified G_opt correlation is the only path to a large (>15.8%) E2E gain.
