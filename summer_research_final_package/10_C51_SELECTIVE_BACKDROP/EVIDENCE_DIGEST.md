# S5B — C51 Selective / Sparse Backward Pass: Evidence Digest

**Candidate**: C51 — selective/sparse backward pass (train only a subset of Gaussians)
**Project period**: 2026-07 .. 2026-09 (repo HEAD `b562562`, 299 commits)
**Compiled from**: reports + results under `reports/`, `results/a100/`, `scripts/phase-c51*`
**Scope note**: All C51 markdown reports and result JSONs are **untracked in git** (`git status` shows them as `??`). The baseline commit `32ab80e` cited inside the R0.x reports is **not resolvable** in this local clone (`git rev-parse` → "malformed object name"), and no commit message in `git log --all` mentions `c51`. The `reports/r3_1/candidate-c-r3-1-final.md` file is about **Candidate C** (certificate repair), **not C51** — it does not cover C51.

---

## 1. What C51 Is: Hypothesis, Mechanism, and Difference from C25

### Hypothesis (C49→C50→C51 chain)
C51 builds on two upstream findings (`reports/phase_c51_sparse_backward.md` §1):
- **C49**: The per-Gaussian gradient distribution is highly concentrated — top 32% of Gaussians account for ~90% of gradient mass; oracle top-10% filtering costs only −0.06 dB.
- **C50**: A *previous-iteration* gradient-norm predictor is highly accurate in the original measurement (Recall@32% = 0.977, Coverage@32% = 0.912 on the `room` scene).

C51's question: **Can predicted sparse backward actually reduce computation while preserving training correctness and quality?** (`reports/phase_c51_sparse_backward.md` §1, line 10).

### Mechanism — which Gaussians get backward
C51 loads a per-Gaussian `uint8_t importance_mask[N]` into the gsplat 1.5.3 backward kernel `rasterize_to_pixels_3dgs_bwd` (`reports/phase-c51-stage4a.md` §3.1; `candidate_c_source_audit/historical_c51_reference/patch_notes_c51.md` §2). For a masked Gaussian (`mask=0`) the kernel **skips gradient computation** (v_rgb, v_conic, v_xy, v_opacity, the warpSum reduction, and the atomicAdd) but **still updates the transmittance T and the running buffer** so that rendering-correctness-dependent quantities (which all *selected* Gaussians depend on) remain correct (`reports/phase-c51-stage4a.md` §3.1; `reports/phase_c51_sparse_backward.md` §3.1). The mask is predicted **before** the kernel from the **previous iteration's** gradient norm via `torch.topk(prev_grad_norm, N * keep_fraction)` (`reports/phase-c51-stage4b.md` §2; `scripts/phase-c51-stage4b/canonical_training.py:380`).

### How C51 differs from C25
**C25** ("sparse-tail backward reorganization") was an earlier *measurement/feasibility* candidate, not an implementation (`reports/phase-c25/c25_training_candidate_screening.md`; `reports/backward_candidate_analysis.md` §4). Its finding: even subsampling to 1% of Gaussians still consumes ~29% of full backward cost, revealing a **~1.7 ms fixed-cost floor** in the CUDA kernel — backward cost is *not* proportional to active-lane count in the sparse regime (`reports/phase_c49_gaussian_lifecycle_research.md` line 239). C25 bounded the *opportunity* at ~14% of T_iter and was verdict **KEEP_CANDIDATE** but **not implemented** — it only screened feasibility.

**C51** is the *implementation* descendant: it actually modifies the CUDA kernel to skip per-Gaussian gradient work, targeting the C25-bounded opportunity. C25 said "the kernel cost is dominated by the top 1-5% of contributors, the tail is already skipped by tile structure" (`reports/backward_candidate_analysis.md` §4, lines 110-114); C51 inverts that to ask whether the *bottom* K% by predicted importance can be skipped *inside* the fused kernel while preserving correctness. C51-R frames the CUDA target as Design B from the C51 Stage-1 source audit, citing the C25 bound: "C25 bound: ~14% T_iter for backward, backward is 39.6% of T_iter → ~5.5% net iteration speedup if 20% of backward arithmetic is skipped" (`reports/phase_c51r_sparse_backward.md` line 309).

---

## 2. Gates A / B / C / D

The four acceptance gates were formalized in Stage 4A and carried through Stage 5 (`reports/phase-c51-stage4a.md` §2):

| Gate | Criterion | Purpose |
|------|-----------|---------|
| **A — Correctness** | Cosine ≥ 0.999 for *selected* Gaussians; skipped = exactly 0 (B1/B3) or v_means2d-only (B2); forward bit-exact (max diff = 0.0) | Verify the CUDA skip does not corrupt gradients of kept Gaussians |
| **B — Quality** | ΔPSNR < 0.2 dB AND ΔSSIM < 0.005 (later reinterpreted as "no degradation", ΔSSIM > −0.005, since SSIM *improved* on all scenes) | Training quality preserved vs full-backward control |
| **C — Speedup** | E2E training speedup > 5% | Practical value |
| **D — Densification** | Clone/split agreement > 50% of baseline | No densification failure |

**Gating criteria tested** (the signals used to decide keep/skip): the *only* scalar actually used by every historical C51 implementation was **`model.xyz.grad.detach().norm(dim=-1)`** — the L2 norm of the xyz (position) parameter gradient (`reports/phase-r0.2-c51-final-mechanism-gate.md` §C, lines 109-114, auditing `scripts/phase-c51/simulated_sparse_backward.py:254`, `scripts/phase-c51r/run_experiment.py:217`, `scripts/phase-c51-stage4b/canonical_training.py:380`). Stage 4A added an **EMA** (decay=0.9, epsilon=1e-6) for B1/B3 to prevent feedback lockout (`reports/phase-c51-stage4a.md` §3.3). Stage 4B/Stage 5 additionally tested (as oracle/ablation, not deployed masks): **G_dens** (means2d.absgrad — densification gradient, C49's original signal), **oracle current-iteration G_opt**, and **FreezeMask** (full backward + Python zero). R0.2 evaluated forward-time signals (alpha/contribution, opacity, tiles_per_gauss) as candidate gates but found their G_opt correlation **untested** (`reports/phase-r0.2-c51-final-mechanism-gate.md` §E).

---

## 3. Variants B1 / B2 / B3

Defined in Stage 4A §1 (`reports/phase-c51-stage4a.md` lines 13-16) and §3.3:

- **B1 (previous-gradient densification)**: Skip **ALL** gradient computation for masked Gaussians. Densification uses the *sparse* gradient (masked Gaussians get xyz.grad = 0 → never exceed the densify threshold). Mask from EMA of previous xyz gradient norm. **Experiment**: 5K training, K∈{50,60,70,80} on `room` (`results/a100/phase-c51-stage4a/training_5k_k{50,60,70,80}_b1.json`); 30K `room`/`bicycle`/`garden` K50 (`results/a100/phase-c51-stage4b/training_{room,bicycle,garden}_k50_b1.json`).
- **B2 (scalar current-gradient norm path, `compute_densify_grad=True`)**: Compute **only `v_means2d`** (→ xyz.grad) for masked Gaussians; skip all other gradients. Densification sees the **full** xyz gradient. Mask from direct gradient norm (no EMA feedback loop needed). **Experiment**: 5K `room` K80 (`results/a100/phase-c51-stage4a/training_5k_k80_b2.json`); reused in Stage 5 Condition C (Sparse+FullDens) on all 3 scenes.
- **B3 (delayed densification ablation)**: **Same CUDA kernel as B1**; tests whether delayed densification feedback is acceptable. **Experiment**: 5K `room` K80 (`results/a100/phase-c51-stage4a/training_5k_k80_b3.json`).

Stage 4A §4 microbenchmark confirms the kernels behave as designed (`results/a100/phase-c51-stage4a/kernel_microbenchmark.json`): B1 skipped gradients are exactly zero (max_abs=0.0, is_zero=true for all 5 param groups); B2 skipped xyz has max_abs=0.002, norm=0.0124, is_zero=**false**, while opacity/scales/rotations/shs skipped are exactly zero.

---

## 4. Core Results: 5K and 30K Training

### 4.1 C51 simulation (Stage 2, no CUDA) — `room`, 5K, predictor = previous-grad-norm
Source: `results/a100/phase-c51/final_comparison.json`; `reports/phase_c51_sparse_backward.md` §4.

**V1 (mask BEFORE densification — BROKEN)**: baseline 26.23 dB; K50 −2.39, K32 −2.60, K20 −3.04 dB. Clone collapse (58,814 → 8,308–9,706). Root cause: masking gradients before `accumulate_positional_gradient()` starved densification.

**V2 (densification decoupled — accumulate BEFORE mask)**: K50 −0.55 (25.68), K32 −0.73 (25.50), K20 −0.95 (25.28) dB. Gradient cosines all >0.99 (xyz 0.9932–0.9946). **Gates**: cosine PASS, PSNR **FAIL** (all >0.2 dB), speedup DEFERRED. Decision: **MODIFY** (`results/a100/phase-c51/final_comparison.json` decision="MODIFY").

### 4.2 C51-R (error-control, simulation) — `room`, 8 configs, 5K + one 30K
Source: `results/a100/phase-c51r/final_comparison.json`; `reports/phase_c51r_sparse_backward.md` §2.

| Config | K | Iters | PSNR | dPSNR | Cosine | Gate B |
|--------|---|-------|------|-------|--------|--------|
| baseline | 1.0 | 5K | 26.235 | — | — | — |
| k90 | 0.9 | 5K | 26.094 | **−0.141** | 0.9999986 | PASS |
| k80 | 0.8 | 5K | 26.051 | **−0.185** | 0.9999660 | PASS |
| refresh50 | 0.5 | 5K | 25.617 | −0.618 | — | FAIL |
| refresh100 | 0.5 | 5K | 25.585 | −0.650 | — | FAIL |
| refresh200 | 0.5 | 5K | 25.708 | −0.527 | 0.9999983 | FAIL |
| refresh500 | 0.5 | 5K | 25.702 | −0.533 | — | FAIL |
| post_dens | 0.5 | 30K | 25.124 | **−0.009** (vs C45 25.133) | 0.9999999 | PASS |

**Mechanism findings**: M1 (higher K helps) — **supported** (K50→K90 +0.42 dB). M2 (periodic refresh resets drift) — **NOT supported** (refresh50-500 all ≈ no-refresh; gap plateaus by iter 1000, not accumulating). M3 (post-densification) — **supported** (post-dens K50 30K −0.01 vs from-start K50 −0.56). Decision: **KEEP → CUDA**, recommended CUDA candidate **K=80%** (−0.18 dB, cosine 1.0).

### 4.3 C51 Stage 4A (CUDA kernel) — `room`, 5K, 7 configs
Source: `reports/phase-c51-stage4a.md` §5; `results/a100/phase-c51-stage4a/training_5k_*.json`.

| Config | PSNR@5K | ΔPSNR | ΔSSIM | E2E speedup | Clone% | Split% | ALL gates |
|--------|---------|-------|-------|-------------|--------|--------|-----------|
| baseline | 29.11 | — | — | — | 100 | 100 | — |
| **K50-B1** | 29.08 | **−0.03** | −0.0027 | **+6.7%** | 54.2 | 68.2 | **✅ ALL** |
| K60-B1 | 29.20 | +0.09 | −0.0020 | +4.6% | 59.4 | 72.2 | ❌ (C) |
| K70-B1 | 29.09 | −0.01 | −0.0012 | +4.3% | 58.9 | 76.2 | ❌ (C) |
| K80-B1 | 29.21 | +0.10 | −0.0001 | +2.9% | 72.2 | 82.6 | ❌ (C) |
| K80-B2 | 28.61 | **−0.49** | **−0.0054** | +0.8% | 98.1 | 98.4 | ❌ (B,C) |
| K80-B3 | 29.24 | +0.13 | −0.0011 | +3.2% | 65.8 | 83.7 | ❌ (C) |

**Only K50-B1 passes all four gates at 5K.** Kernel microbenchmark (`results/a100/phase-c51-stage4a/kernel_microbenchmark.json`): B1 kernel speedup linear in skip fraction — K50 10.16%, K70 6.85%, K80 4.62%, K90 2.25%; B2 consistently ~40% less (K50 6.59%, K80 2.68%). Transfer ratio E2E/kernel ≈ 63-66% (backward ≈ 50% of iteration).

### 4.4 C51 Stage 4A — 30K validation (K50-B1 vs baseline, `room`)
Source: `reports/phase-c51-stage4a.md` §6.2; `results/a100/phase-c51-stage4a/training_5k_baseline_30k.json`, `training_5k_k50_b1_30k.json`.

| Metric | Baseline 30K | K50-B1 30K | Delta | Gate |
|--------|-------------|------------|-------|------|
| PSNR | 26.49 | 27.08 | **+0.59** | B ✅ |
| SSIM | 0.8589 | 0.8589 | 0.0000 | B ✅ |
| Mean time | 17.88 ms | 16.59 ms | **+7.8%** | C ✅ |
| Clone | 6,798 | 1,657 | 24.4% | D ⚠️ |
| Split | 43,881 | 25,924 | 59.1% | D ✅ |

Clone agreement drops below the 50% Gate D threshold at 30K, but **quality is 0.59 dB better** than baseline — interpreted as implicit regularization from reduced densification, not failure.

### 4.5 C51 Stage 4B — canonical 30K, 3 scenes (K50-B1)
Source: `results/a100/phase-c51-stage4b/multi_scene_analysis.json`; `reports/phase-c51-stage4b.md` §9.1.

| Scene | Baseline PSNR | K50-B1 PSNR | ΔPSNR | ΔSSIM | E2E speedup | Final GS ratio |
|-------|--------------|-------------|-------|-------|-------------|----------------|
| Room | 29.209 | 30.239 | **+1.030** | +0.0058 | +8.23% | 69.8% |
| Bicycle | 21.244 | 21.742 | **+0.498** | +0.0101 | +24.43% | 69.4% |
| Garden | 23.064 | 23.101 | **+0.037** | +0.0099 | +28.86% | 64.3% |

All 3/3 scenes pass quality and speedup gates. Quality *improves* on every scene. **Decision: KEEP (with caveats)** (`reports/phase-c51-stage4b.md` §14).

### 4.6 C51 Stage 5 — densification decoupling (the critical experiment)
Source: `reports/phase-c51-stage5.md` §2; result files referenced in §11.

Four conditions on `room` 30K (A=baseline, B=K50-B1, C=Sparse+FullDens [B2 path], D=FullBwd+MaskedOpt):

| Condition | PSNR | Speedup vs A | Final GS ratio |
|-----------|------|--------------|----------------|
| A: Baseline | 29.21 | — | 100% |
| B: K50-B1 | 30.24 | **+8.2%** | 69.8% |
| C: Sparse+FullDens | 29.94 | **+0.5%** | 94.6% |
| D: FullBwd+MaskedOpt | 29.94 | **−0.4%** | 94.8% |

**The single most important C51 number**: **intrinsic sparse-backward speedup = +0.5%** (Condition C vs A, Room) when Gaussian population is preserved. **94% of K50-B1's E2E speedup comes from altered densification** (B−C = +7.7%), not from the CUDA kernel skip. On outdoor scenes C is *negative* (Bicycle −6.6%, Garden −1.7%) because the B2 densification-path overhead exceeds kernel savings. C vs D (semantic equivalence): PSNR 29.94=29.94, SSIM Δ=−0.0001, final GS Δ=−2,787 — confirming CUDA sparse ≡ full backward + Python masking, with only +0.52 ms timing difference.

---

## 5. Densification Agreement

Measured in Stage 4A §5.5 and Stage 5 §3. **B2 preserves densification near-perfectly** (clone 98.1%, split 98.4% at 5K) because xyz.grad is correct for all Gaussians. **B1/B3 partially decouple** (clone 54-72%, split 68-84% at 5K; clone 14.9% at Room 30K) — masked low-gradient Gaussians get xyz.grad=0 and never exceed the densify threshold. **Prune agreement is ~100% for all** because pruning is opacity-based and unaffected by gradient skipping. At 30K, K50-B1 clone agreement on Room drops to 24.4% (below 50% Gate D) but quality is +0.59 dB better — interpreted as the reduced densification acting as implicit regularization rather than failure. Stage 5's OracleDens ablation (`reports/phase-c51-stage4b.md` §5.1) at 5K shows that **preserving densification eliminates the quality difference** (OracleDens 15.52 dB vs B1 13.73 dB, both vs baseline 16.11) — confirming the +dB is a densification side effect, not a sparse-backward benefit.

---

## 6. Why B2 Failed and Why B1/B3 Survived

**Documented in** `reports/phase-c51-stage4a.md` §5.1 (hypothesis) and RQ6 (line 296). B2 gives **position updates (xyz.grad) without appearance updates** (opacity/scales/rotations/shs.grad = 0) to masked Gaussians → a **partial-update inconsistency**: Gaussians *move* to new positions but *retain old appearance*, degrading quality (−0.49 dB, −0.0054 SSIM, both failing gates). B1/B3 **freeze all parameters** of masked Gaussians → consistency preserved → quality maintained (ΔPSNR −0.03 to +0.13). This is reported as a **non-obvious finding**: the "densification-safe" design (B2) is *worse* for quality than the "freeze-everything" design (B1/B3). Additionally B2's `v_means2d` computation for masked Gaussians adds overhead, yielding ~40% less kernel speedup than B1 at the same K (`reports/phase-c51-stage4a.md` §4.3, RQ7). Stage 5 §9 limitation 1 notes the B2 path overhead on 10M+ Gaussian scenes makes Condition C slower than baseline — an implementation artifact, not a fundamental limitation.

---

## 7. Final Verdict on C51

The verdict **evolved across four phases**:

1. **C51 Stage 1-3** (`reports/phase_c51_sparse_backward.md` §8): **MODIFY** — promising but gates not met (PSNR 0.55-0.95 dB, garden coverage FAIL).
2. **C51-R** (`reports/phase_c51r_sparse_backward.md` §6; `results/a100/phase-c51r/final_comparison.json` decision="KEEP"): **KEEP → proceed to CUDA Stage 4**. Recommended CUDA config Design B with K=80%.
3. **C51 Stage 4A** (`reports/phase-c51-stage4a.md` §6.1, §6.2): K50-B1 passes all 4 gates at 5K; at 30K Gate D clone drops below 50% but quality +0.59 dB.
4. **C51 Stage 4B** (`reports/phase-c51-stage4b.md` §14): **KEEP (with caveats)** — 3/3 scenes pass; caveat that +dB and much of the speedup are densification side effects, not sparse-backward benefits.
5. **C51 Stage 5** (`reports/phase-c51-stage5.md` §10): **KEEP but reframe as joint mechanism** — pure sparse-backward speedup is +0.5% (fails 5% gate); the mechanism's value is as a *joint sparse-backward + densification-modification* technique.
6. **R0.1** (`reports/phase-r0.1-evidence-correction.md` §9): **C51_MODIFY** — G_opt concentration is strong (Top-50% = 95.4%, Gini 0.751) but lag-1 predictability insufficient (Top-50% Jaccard median 0.648 < 0.80 gate, Pearson 0.033). Previous-gradient masking unsupported.
7. **R0.2 — FINAL** (`reports/phase-r0.2-c51-final-mechanism-gate.md` Final Decision): **C51_SYSTEMS_COMPONENT**. Exact wording (line 7): `**Decision**: **C51_SYSTEMS_COMPONENT**`. Rationale: historical previous-xyz-gradient masking is **dead** (Top-50% captures only 52.8% of current G_opt mass — 2.8 points above random; xyz_norm signal 52.1%, even worse). G_dens is a strong *current-iteration* predictor (Top-50% coverage 92.7%, Spearman 0.858) but is available **after** the dominant backward kernel, capping its E2E opportunity at **15.8%**. The CUDA sparse-backward infrastructure from Stage 4A can be **repurposed**: use current-iteration G_dens (not previous xyz gradient) to gate SH backward + optimizer.

**Commit hashes**: No C51-specific commit hash exists in `git log --all`. The R0.x reports cite baseline commit `32ab80e` (tag `baseline/reference-v1-absgrad`) but this is **not resolvable** in the local clone (`git rev-parse 32ab80e` → "malformed object name"). Repo HEAD is `b562562` (299 commits). **All seven C51 markdown reports and all `results/a100/phase-c51*` JSONs are untracked** (`git status` → `??`).

---

## 8. Contradictions / Gaps

1. **C50 recall contradiction**: The original C50/C51 reports claim previous-gradient Recall@32% = 0.977, Coverage@32% = 0.912 (`reports/phase_c51_sparse_backward.md` §1, §5.1). R0.1/R0.2 **refute** this: corrected lag-1 Top-50% Jaccard median = 0.648 (R0.1 §4.1) / 0.487 (R0.2 §A corrected), and previous-Top-50% captures only 52.8% of current G_opt mass (R0.2 §B). The original C50 was measured across **distant checkpoints**, not consecutive iterations (R0.1 §2.1 correction #1). This is the central reason historical C51 was killed.
2. **C49 signal confusion**: R0.1 §2.3 clarifies that historical C49 measured **G_dens** (means2d.absgrad, the densification gradient), **not G_opt** (the optimization gradient C51 actually needs). C51's mask used xyz.grad.norm — a G_opt component — which R0.2 §C confirms is the *worst*-predictable parameter (previous coverage 52.1%).
3. **The +dB "quality improvement" is a densification artifact**: Stage 4B reports K50-B1 *improves* PSNR by +0.04 to +1.03 dB across 3 scenes, but Stage 4B §13 and Stage 5 §4 attribute this to altered densification (30-36% fewer Gaussians → less overfitting), **not** to sparse backward. The 5K OracleDens ablation eliminates the quality difference when densification is preserved (Stage 4B §5.1). A 30K OracleDens confirmation was **not run** (Stage 4B §12 limitation 1).
4. **Post-densification mode contradicts C51-R**: C51-R recommended post-densification K50 as the *best* result (−0.01 dB at 30K, `reports/phase_c51r_sparse_backward.md` §2.1). Stage 4B §4.4 and Stage 5 §7 contradict: post-densification K50 shows **no speedup** (−0.1%) because after densification the gradient landscape is uniform and mask overhead (0.389 ms) exceeds kernel savings. Sparse backward is only effective *during* densification.
5. **5K ablation quality unreliable**: Stage 4B §5 / §12 limitation 1 states the 5K OracleDens/FreezeMask/Oracle ablation quality collapsed due to an opacity-reset bug at iter 3000 — only timing/densification behavior are informative from 5K.
6. **Gate B SSIM criterion mis-specified**: The original `abs(ΔSSIM) < 0.005` *fails* when SSIM improves by >0.005 (all 3 scenes). Stage 4B §9.1 reinterprets as "no degradation" (ΔSSIM > −0.005) — a post-hoc criterion change.
7. **Single-scene → multi-scene**: Stages 4A and the 5K ablations are `room`-only; the 30K multi-scene (4B/5) covers only Room/Bicycle/Garden. No Bonsai/Kitchen. Stage 3 multi-scene *predictor* validation showed **garden fails** coverage criteria (Coverage@32% = 0.834 < 0.85, `reports/phase_c51_sparse_backward.md` §5.1) — a generalizability warning.
8. **No git provenance**: C51 reports and results are untracked; the cited baseline commit `32ab80e` is unresolvable. Reproducibility relies on file contents alone, not git history. The `reports/r3_1/candidate-c-r3-1-final.md` file requested in the task brief is about Candidate C (certificate repair), **not C51** — it should not be cited as a C51 source.
