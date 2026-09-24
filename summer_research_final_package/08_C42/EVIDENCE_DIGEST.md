# S2 — C42 Evidence Digest (3DGS Optimization)

**Scope:** Every piece of evidence about candidate **C42** found in the
`3dgs-renderer-benchmark` repo (project period 2026-07 .. 2026-09).
Every claim cites its source file path. Note up front: the task brief
hypothesized C42 was "scale/gaussian quantization or compression." It is
**not**. C42 is a **Downsampled-SSIM (D-SSIM) training-speed optimization**:
the structural-similarity (SSIM/DSSIM) term of the 3DGS loss is computed on
area-downsampled rendered/GT images, while the L1 term stays at full
resolution. The "scale" in C42 is the **SSIM image-resolution downsample
factor**, not a Gaussian scale or bit-width.

---

## 1. C42 Formulation

### 1.1 What C42 is

C42 modifies the 3DGS training loss so that the D-SSIM component is evaluated
at reduced spatial resolution. The rendering (rasterization) and the L1 term
are untouched; only the SSIM convolution operates on smaller images.

**Canonical definition (frozen in `reports/phase-s2.2-c42-generalization.md` §1):**

```python
pred_ds   = F.interpolate(pred,   scale_factor=0.5, mode="area")
target_ds = F.interpolate(target, scale_factor=0.5, mode="area")
dssim     = SepSSIM(pred_ds, target_ds)
loss      = (1 - λ) * L1 + λ * dssim,   λ = 0.2
```
- L1 remains **full resolution**.
- SepSSIM: window=11, sigma=1.5, C1=0.0001, C2=0.0009, padding=5, zero border.
- `reports/phase-s2.2-c42-generalization.md` §1; `reports/a100_validation/final_c42_decision.md` §5.

**Implementing code (verbatim), `scripts/rtx5070/c42_downsampled_ssim_benchmark.py` lines 41–81:**

```python
def d_ssim_downsampled(pred, target, scale=1.0, window_size=11, sigma=1.5, data_range=1.0):
    """D-SSIM with optional downsampling before computation."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    def blur(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu_pred = blur(pred); mu_target = blur(target)
    mu_pred_sq = mu_pred ** 2; mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    sigma_pred_sq = blur(pred ** 2) - mu_pred_sq
    sigma_target_sq = blur(target ** 2) - mu_target_sq
    sigma_pred_target = blur(pred * target) - mu_pred_target
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return 1.0 - ssim_map.mean()

def combined_downsampled(pred, target, scale=1.0, lambda_dssim=LAMBDA_DSSIM):
    l1 = F.l1_loss(pred, target)  # L1 always at full resolution
    dsim = d_ssim_downsampled(pred, target, scale=scale)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim
```
(`LAMBDA_DSSIM = 0.2`, line 34.)

The same modification is stated in `reports/c42-completion-batch.md` §1.1:
"C42 modification: `loss = (1-lambda)*L1 + lambda*d_ssim_downsampled(scale)`,
lambda=0.2; `d_ssim_downsampled`: `F.interpolate(pred, scale_factor=s,
mode='area')` then SepSSIM."

### 1.2 Why it speeds things up (root cause)

`reports/rtx5070/c42_dssim_audit.md` and `reports/rtx5070/c42_hypothesis_matrix.md`
establish that D-SSIM is the dominant training bottleneck: on A100 it consumes
**78.6% of GPU kernel time (81.4 ms/iter)** vs 42.5% on RTX 5070
(`reports/a100_validation/final_c42_decision.md` §1, Table). The cause is
cuDNN dispatching a pathological `conv2d_grouped_direct` algorithm for the
3-channel (RGB) depthwise 11×11 Gaussian convolution — 63 kernels on A100 vs
9 on RTX 5070 — achieving only 0.36% of peak compute
(`reports/rtx5070/c42_dssim_audit.md` §5–6). Downsampling reduces pixel count
by s², so SSIM conv cost scales as s² while L1 is unchanged.

---

## 2. Reference V1 vs C42 — Baseline Anchoring

C42 is anchored to the **Reference V1** official 3DGS semantics, not a custom
trainer. Per `reports/c42-completion-batch.md` §1.1 and Provenance (§7):

- **Training codebase:** `baseline/reference_v1/` — official Graphdeco 3DGS
  semantics with persistent Adam optimizer (state migration across topology
  changes), view-space mean2D gradient for densification (absgrad=True,
  threshold=0.0008), fixed SH tensor with active_sh_degree progression.
- **GaussianModel:** `baseline/reference_v1/gaussian_model.py`, hash
  `68731e375013a6f2`.
- **Config:** `baseline/reference_v1/config.py`, hash `ca76d4f36059839e`,
  base `REFERENCE_V1_ABSGRAD`.
- **Git HEAD (completion batch):** `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`;
  git describe `baseline/reference-v1-absgrad-3-g32ab80e`.
- **Canonical commit (S2.2):** `ad18916` (tag `baseline/reference-v1-absgrad`),
  per `reports/phase-s2.2-c42-generalization.md` §0.
- **SSIM baseline:** SepSSIM is the default SSIM in Reference V1 and is
  classified `C44 = BASELINE_CONSTITUENT` (not an incremental optimization)
  — `reports/phase-s2.2-c42-generalization.md` §10.
- **Critical fix noted:** an initial training attempt used
  `scripts/epic05/phase7/gaussian_model.py`, which creates a new optimizer
  after each topology change (losing Adam momentum) and uses wrong
  densification/pruning parameters, causing catastrophic quality degradation
  (Room PSNR 20→12). Switching to `baseline/reference_v1/gaussian_model.py`
  resolved it (`reports/c42-completion-batch.md` §1.1).
- **Environment:** PyTorch 2.7.1+cu118, gsplat 1.5.3, A100-PCIE-40GB (108
  SMs), seed=42, 1080p (1920×1080), Mip-NeRF 360 (bicycle, garden, room)
  (`reports/c42-completion-batch.md` §7; `reports/a100_validation/c42_downsample_ssim_a100.md` §1).

The **baseline** against which C42 is measured is therefore scale=1.0 (full-
resolution SSIM) under the identical Reference V1 codebase — i.e., C42 at
scale=1.0 must reduce exactly to the native loss. This was verified by
**loss-equivalence tests** on all three 13-scene baselines:
- Speedy-Splat: L1 diff=0, DSSIM diff=0, grad cosine=1.0, grad rel L2=0 —
  bit-identical (`reports/speedysplat-c42-13scene-final.md` §3).
- Faster-GS: loss diff=0, L1 diff=0, DSSIM diff=0, grad cosine=1.0, grad
  rel L2=0 — exact (`reports/fastergs-c42-13scene-final.md` §3).
- FastGS: forward diff=0.0 (exact); backward grad cosine=0.9997 (below
  0.999999 threshold, attributed to `fused_ssim` CUDA kernel non-determinism)
  (`reports/fastgs-c42-13scene-final.md` §3).

---

## 3. Experiments

C42 spans a large experiment tree. Speedup definitions used:
- **D-SSIM speedup** = isolated D-SSIM fwd+bwd time ratio (kernel-only on the
  loss).
- **E2E speedup** = full training-iteration wall-clock ratio (render + loss +
  bwd + optimizer).
- **Wall-time speedup** = total training wall-clock ratio (includes
  densification, which changes Gaussian population).
Formulae: reduction `r = 1 − T_C42/T_base`; throughput `S = T_base/T_C42`
(`reports/phase-s2.2-c42-generalization.md` §5).

### 3.1 RTX 5070 exploratory screening (NOT for final claims)

`reports/rtx5070/c42_hypothesis_matrix.md` designed six candidates (A cache,
B separable conv, C torch.compile, D downsampled SSIM, E adaptive schedule,
F hybrid). Candidate **D** became "C42." RTX 5070 results
(`reports/a100_validation/final_c42_decision.md` §2): scale 0.50 gave D-SSIM
4.18×, E2E +38.2%, but rotation cosine 0.921 — later shown to be a
**diverged-checkpoint artifact** (PSNR=12.03). RTX 5070 data is explicitly
exploratory, not for final claims (`final_c42_decision.md` §7).

### 3.2 A100 isolated downsampled-SSIM benchmark

`reports/a100_validation/c42_downsample_ssim_a100.md`; JSON
`results/a100/phase-c42/c42_downsample_ssim_a100.json`. Room scene,
896,969 Gaussians, checkpoint PSNR=20.56, 5 cameras, 30 warmup / 50 measure.

| Scale | Img size | D-SSIM fwd+bwd (ms) | D-SSIM speedup | E2E (ms) | E2E speedup | Min grad cosine |
|-------|----------|---------------------|----------------|----------|-------------|-----------------|
| 1.00 | 1080×1920 | 79.58 | 1.00× | 95.30 | — | 1.0000 |
| 0.75 | 810×1440 | 47.71 | 1.67× | 63.10 | +33.8% | 0.9939 |
| 0.50 | 540×960 | 22.48 | 3.54× | 37.46 | +60.7% | 0.9854 |
| 0.25 | 270×480 | 2.64 | 30.18× | 17.50 | +81.6% | 0.9469 (FAIL) |

Key fields in JSON: `isolated_timing.<s>.full_ms`, `end_to_end_timing.<s>.mean_ms`,
`cosine_similarity.<s>.<param>.{mean,min}`, `memory.<s>.e2e_peak_MB`.
Forward dominates backward (75.2 vs 5.1 ms at scale 1.0) because cuDNN
dispatches 63 forward kernels on A100. Memory savings modest (−299 MB / −10.4%
at 0.5).

### 3.3 P0 training validation (scale=0.50, 10K, room) — FAIL

`reports/a100_validation/c42_p0_training_validation.md`; JSON
`results/a100/phase-c42/c42_p0_training_validation.json`. A vs B, 10K iters,
seed=42, densification enabled.

| Gate | Threshold | Result | Verdict |
|------|-----------|--------|---------|
| PSNR drop | <0.2 dB | −0.01 dB | PASS |
| SSIM drop | <0.005 | −0.0062 | FAIL |
| Gaussian count | <10% | +17.8% | FAIL |
| Speedup | >40% | +60.2% | PASS |

Decision: **FAIL** (2/4). PSNR identical (14.93 both) but SSIM marginally
over gate and Gaussian count diverges (+17.8%) because downsampled SSIM
provides smoother gradients → less pruning pressure. Both runs diverge
(PSNR 21.8→14.9) but A-vs-B comparison held valid.

### 3.4 P1 training validation (scale=0.75, 10K, room) — PASS

`reports/a100_validation/c42_p1_training_validation_0.75.md`; JSON
`results/a100/phase-c42/c42_p1_training_validation_0.75.json`.

| Gate | Threshold | A (1.0) | B (0.75) | Result | Verdict |
|------|-----------|---------|----------|--------|---------|
| PSNR | <0.2 dB | 14.76 | 15.32 | +0.56 dB | PASS |
| SSIM | <0.005 | 0.5996 | 0.6047 | +0.0051 | PASS |
| GS | <10% | 980,807 | 974,097 | −0.7% | PASS |
| Speedup | >30% | 94.1 ms | 62.4 ms | +33.6% | PASS |

Decision: **PASS — all 4 gates.** Scale=0.75 *improves* quality (regularizer
effect) and preserves topology. This report declares scale=0.75 the primary
candidate.

### 3.5 P2 30K validation (scale=0.75, room) — PASS

`reports/a100_validation/candidate_parallel_screening.md` §C42-P2; JSON
`results/a100/phase-c42/c42_p2_training_validation_30k.json`. 30K iters:
dPSNR=+0.04, dSSIM=+0.0029, dGS=−7.7%, speedup +33.6% (from P1). All gates
pass. (Both runs diverge PSNR 21.8→12.7; comparison valid.)

### 3.6 Track A scale sweep + multi-scene (5K, 3 scenes, scale=0.75)

`reports/a100_validation/track_a_c42_scale_sweep.md`; JSONs
`results/a100/phase-c42/track_a_scale_{1.0,0.75,0.5,0.25}.json`,
`track_a_multiscene_{bicycle,garden}.json`. Gradient cosine + magnitude +
topology at iters 1000/3000/5000.

Cross-scene (5K, scale=0.75): room +33.7% (dPSNR +0.59, dSSIM +0.0066, dGS
+7.2%); bicycle +30.1% (dPSNR +0.24, dSSIM +0.0055, dGS −3.4%); garden +33.7%
(dPSNR +0.13, dSSIM +0.0001, dGS +1.4%). **All 3 scenes pass all 4 gates.**
Scale=0.25 dropped (xyz cosine 0.834, ΔPSNR −1.93 dB room, topology +17%).

### 3.7 Track B ablation + the discrepancy (P1 +33.6% vs Track B +6.4%)

`reports/c42_discrepancy_analysis.md`; JSONs
`results/a100/phase-c42/track_b_ablation_{10,0875,075,0625,05}.json`. Track B
re-ran the scale ablation and got only +6.4% at scale=0.75 (vs P1's +33.6%).

**Root cause: configuration mismatch (primary) + training phase (secondary).**
The old P1 script called `model.prune()` every 100 iters (final GS ~960K); the
new `track_b_ablation.py` only called `prune_and_reset` once at iter 3000,
so clone+split exploded GS to 23.8M. C42 speedup is Gaussian-count-dependent:
+37.6% at 500K GS, +16.5% at 20M GS (controlled profiler,
`results/a100/phase-c42/c42_gs_count_profile.json`,
`scripts/phase-c42/c42_gs_count_profiler.py`). Timing definitions are
equivalent (both per-iteration wall clock, same speedup formula). Both
results are correct for their configs; Track B's Pareto *relative* comparison
between scales is still valid.

### 3.8 S2.2 cross-scene generalization (30K, scale=0.5, 3 scenes)

`reports/phase-s2.2-c42-generalization.md`; JSONs under
`results/reference_v1/s22/{room,garden,bicycle}/`.

| Scene | Throughput | ΔPSNR | ΔSSIM | ΔLPIPS | ΔN% | Classification |
|-------|-----------|-------|-------|--------|-----|----------------|
| Room | 1.90× (47.5% reduction) | +0.24 | −0.0078 | N/A | −21.7% | QUALITY_PASS |
| Garden | 1.68× (40.5%) | −0.40 | −0.0224 | +0.0255 | −12.2% | QUALITY_TRADEOFF |
| Bicycle | 1.60× (37.3%) | −0.04 | −0.0213 | +0.0197 | −13.2% | QUALITY_TRADEOFF |

Speedup attribution (`reports/phase-s2.2-c42-generalization.md` §6):
~57–60% direct loss-side saving (cheaper SSIM), ~40–44% indirect Gaussian-
population reduction (fewer Gaussians → faster raster). DS0.5 SepSSIM is
consistently ~9.2 ms regardless of Gaussian count. Per-scene decisions:
Room KEEP, Garden MODIFY, Bicycle KEEP → global `C42_GENERALIZATION = MODIFY`.

### 3.9 S2.3 Pareto frontier (30K, scales {1.0, 0.75, 0.625, 0.5})

`reports/phase-s2.3-c42-pareto-frontier.md`; JSONs under
`results/reference_v1/s23/`. Fixed-tensor loss benchmark
(`fixed_tensor_loss_benchmark.json`): 1.0=33.15 ms, 0.75=19.41 ms (1.71×),
0.625=13.92 ms (2.38×), 0.5=9.24 ms (3.58×). Non-monotonic finding: on
Bicycle, scale=0.625 is **Pareto-dominated** by 0.5 (slower *and* worse
PSNR/SSIM/LPIPS). Final classification `C42_PARETO_KEEP`; scale=0.5
recommended default; Garden has a thin frontier with an intermediate 0.75
option. Paper-ready claim: "tunable efficiency–quality frontier … 1.6–1.9×
throughput."

### 3.10 13-scene baseline benchmarks (Speedy-Splat, FastGS, Faster-GS)

See §4. Per-scene PSNR/SSIM/LPIPS/N_gaussians/wall_time tables are in the
three `*-c42-13scene-final.md` reports; aggregate JSONs
`results/speedy-splat-all-metrics.json`, `results/fastgs-all-metrics.json`,
`results/faster-gs-all-metrics.json`. Each ran 30K iters, A100, λ(DSSIM)=0.2,
λ(L1)=0.8, scale_factor=0.5, mode="area".

---

## 4. Baseline Comparisons (C42 vs Speedy-Splat / FastGS / Faster-GS)

All three use scale_factor=0.5, 30K iters, A100. Joint quality criterion =
PSNR within ±0.3 dB **and** SSIM within ±0.02. Classification bands:
BROADLY_ADDITIVE ≥70%, PARTIALLY_ADDITIVE 40–70%.

### 4.1 Speedy-Splat — PARTIALLY_ADDITIVE

`reports/speedysplat-c42-13scene-final.md` (baseline commit `34c45c6`,
C42 commits `07f4712`, `aa75d8d`). Quality preserved (joint) **9/13 = 69.2%**
(0.8 pp short of BROADLY); PSNR within ±0.3: 12/13; SSIM within ±0.02: 10/13.
Gaussian reduction 12/13 (mean −15.7% excl. garden). **Geomean wall speedup
1.084×** (9/13 faster). Mean ΔPSNR −0.090, ΔSSIM −0.0125 (all 13 negative),
ΔLPIPS +0.0099. **Binding constraint = SSIM**; indoor 4/4 preserved, T&T 0/2
(worst truck: −0.282 dB, −0.0215 SSIM, −41.1% Gaussians, 1.251×).

### 4.2 FastGS — PARTIALLY_ADDITIVE (but SLOWER)

`reports/fastgs-c42-13scene-final.md` (baseline `44e02a5`, C42 commits
`8b47521`, `d4d33b6`). Quality preserved (joint) **8/13 = 61.5%**. Gaussian
reduction **13/13 = 100%**, mean −13.15%. **Geomean wall speedup 0.924×
(C42 ~8% SLOWER)** — `F.interpolate` overhead not offset because FastGS uses a
fused_ssim CUDA kernel (~130–145 it/s) so SSIM is already cheap. Mean ΔPSNR
−0.175, ΔSSIM −0.0135 (all 13 negative). Indoor 4/4 preserved; stump worst
(−0.44 dB, −0.044 SSIM).

### 4.3 Faster-GS — PARTIALLY_ADDITIVE (but quality-improving)

`reports/fastergs-c42-13scene-final.md` (baseline `3cb0b75`, NeRFICG; C42
patch in `Loss.py`/`Trainer.py`). Quality preserved (strict joint) **8/13 =
61.5%**; non-degraded **11/13 = 84.6%**. Gaussian reduction 9/13
(inconsistent — 4 increase). **Geomean wall speedup 0.998× (parity)**. **Mean
ΔPSNR +0.19 dB, ΔSSIM 0.000, ΔLPIPS −0.006 — first baseline where C42
net-improves quality** (Room +1.99 dB, truck +0.64 dB). Sole genuine
degradation: stump (−0.94 dB, −0.040 SSIM).

### 4.4 Cross-baseline conclusion

C42's effect is **baseline-dependent**. On Speedy-Splat (aggressive pruning)
it gives modest wall speedup + consistent Gaussian reduction at an SSIM cost.
On FastGS (fused SSIM kernel) it gives no wall benefit (overhead dominates)
but strong Gaussian reduction. On Faster-GS (fused_dssim, frequent
densification) it net-*improves* quality but inconsistently reduces Gaussians.
SSIM degradation is the systematic, mechanism-intrinsic cost across all three;
stump is the consistent worst-case scene. Garden is a data-quality outlier in
all three (manually-created COLMAP) and should be excluded.

---

## 5. Adaptive Scale Experiments (λ / oracle / fixed-point analysis)

### 5.1 Adaptive Structural Supervision feasibility

`reports/c42-adaptive-feasibility.md` (date 2026-09-15). Treats per-scene
scale selection as a constrained optimization: feasible set
`F_j(τ) = {s : |SSIM_1.0 − SSIM_s| ≤ τ}`; per-scene oracle
`s_j* = argmin_{s∈F_j(τ)} C(s)`; best fixed feasible = cheapest scale in the
global intersection. **Key correction:** a prior DROP compared the oracle
against fixed-0.5, but fixed-0.5 *violates* the τ constraint on Garden/Bicycle
at τ<0.025 — invalid. Corrected comparator is the best *globally feasible*
fixed policy.

Tau sweep (30K, fixed-tensor costs: 1.0=33.15, 0.75=19.41, 0.625=13.92,
0.5=9.24 ms):
- τ=0.020: oracle (Room→0.5, Garden→0.75, Bicycle→1.0), avg cost 20.60 ms;
  best fixed feasible = 1.0 (33.15 ms); **advantage 12.55 ms (37.9%)**.
- τ=0.010–0.015: advantage 7.97 ms (24.0%).
- τ≥0.025: advantage collapses to 0 (global 0.5 becomes feasible).

Decision: **`C42_ADAPTIVE_MODIFY`** — oracle opportunity is real, but the
current HF-fraction predictor FAILS (33% constraint-violation rate at
τ=0.020; predicts 0.625 for Bicycle, which violates). Counterfactual
limitation stated: the oracle is a **fixed-trajectory counterfactual
opportunity proxy**, *neither* a formal upper nor lower bound on online
adaptive training.

### 5.2 B6 robustness gate — PASS

`reports/c42-adaptive-feasibility.md` §B6. UNKNOWN-scale sensitivity: over
all 16 feasible/infeasible assignments of 4 missing 30K SSIM measurements,
min advantage at τ=0.020 = 3.39 ms (>0.5 ms materiality). Multi-metric:
advantage survives PSNR+SSIM identically (12.55 ms); adding LPIPS≤0.03
reduces to 4.58 ms (Room LPIPS was missing, now resolved to 12.55 ms in
completion batch). Highest-value missing measurement: Garden 0.625 (max swing
6.41 ms). `C42_ORACLE_ROBUSTNESS = PASS`; predictor search justified.

### 5.3 B7 falsification, B9 strong fixed-boundary — MODIFY

`reports/c42-b9-strong-fixed-boundary.md`. B9 trained four new Bicycle scales
(0.80, 0.85, 0.90, 0.95) at 30K and benchmarked loss cost (SepSSIM, A100,
1080p, backward-pass only): 0.50=2.42 ms, 0.625=3.54, 0.75=4.92, 0.80=5.51,
0.85=6.24, 0.90=7.00, 0.95=7.77, 1.00=8.34 ms. **Only scale 1.0 is globally
feasible at τ=0.020** — Bicycle is the binding constraint (dSSIM at 0.90=0.0206,
0.95=0.0205, both marginal fails; dSSIM non-monotonic). Adaptive oracle:
Room→0.5 (2.42), Garden→0.75 (4.92), Bicycle→1.0 (8.34), avg 5.23 ms vs strong
fixed baseline 8.34 ms → **advantage 3.11 ms (37.3%)**. Decision gate
2.0–5.0 ms → **MODIFY**. (Relaxing τ to 0.021 would make Bicycle 0.90
feasible and drop advantage to 1.77 ms = DROP territory.)

### 5.4 Completion batch — final adaptive decision = KEEP

`reports/c42-completion-batch.md`; JSON
`results/c42_adaptive/completion_batch/final_decision.json`. Three fresh 30K
runs filled missing scales (Bicycle 0.75, Room 0.75, Garden 0.625). Unified
re-eval of all 11 checkpoints. At τ=0.020, full-metric (PSNR+SSIM+LPIPS)
advantage = **12.55 ms (37.8%)** > 5 ms KEEP threshold.
`C42_FINAL_STATUS = KEEP`. Bicycle 0.75 dSSIM=−0.0225 (NOT feasible at τ=0.020),
so global 0.75 is blocked and the advantage stays at 12.55 ms (stronger than
B6 predicted). Per-scene oracle: Room→0.5, Garden→0.75, Bicycle→1.0.

Note the **fixed-point analysis** the task asks about corresponds to the
**fixed-tensor loss benchmark** (isolated SSIM fwd+bwd on frozen 1080p random
tensors, scene-independent) and the **fixed-trajectory counterfactual**
wording — both in `reports/c42-adaptive-feasibility.md` §B6-E/B6-F and
`reports/phase-s2.3-c42-pareto-frontier.md` §6.

---

## 6. Final Decision

C42 has **multiple "final" decisions that evolve by phase** — the fixed-scale
C42 is broadly KEPT for speed; the adaptive variant oscillates:

| Report / file | Decision | Basis |
|---------------|----------|-------|
| `reports/a100_validation/final_c42_decision.md` §8 | **KEEP — scale 0.50 primary (+60.7%), 0.75 fallback (+33.8%)** | A100 gradient cosine + isolated/E2E timing; "highest-impact D-SSIM optimization" |
| `reports/phase-s2.3-c42-pareto-frontier.md` §9 | `C42_PARETO_KEEP`; scale=0.5 recommended default | 30K Pareto across 3 scenes |
| `reports/phase-s2.2-c42-generalization.md` §8 | `C42_GENERALIZATION = MODIFY` | Garden fails all 3 quality gates |
| `reports/c42-adaptive-feasibility.md` §8 | `C42_ADAPTIVE_MODIFY` | Oracle real (12.55 ms), predictor fails |
| `reports/c42-b9-strong-fixed-boundary.md` §B9-H | **MODIFY** (3.11 ms, 2.0–5.0 ms band) | Strong fixed-boundary search |
| `reports/c42-completion-batch.md` §5 / `results/c42_adaptive/completion_batch/final_decision.json` | **`C42_FINAL_STATUS = KEEP`** | Full-metric advantage 12.55 ms > 5 ms |

**Most recent / authoritative statement** (`results/c42_adaptive/completion_batch/final_decision.json`):
> `"C42_FINAL_STATUS": "KEEP"` — "Full-metric (PSNR+SSIM+LPIPS) advantage =
> 12.55ms at tau=0.020, exceeds 5ms threshold."

And the fixed-scale deployment statement (`reports/a100_validation/final_c42_decision.md` §8):
> "On A100-PCIE-40GB with gsplat 1.5.3, downsampled SSIM at scale 0.50
> achieves +60.7% total training speedup (95.3 → 37.5 ms/iter) with gradient
> direction preserved >98.5% for all parameter groups. … Scale 0.75 is
> validated as a fallback with +33.8% speedup and >99.4% gradient
> preservation."

### Commits

`git log --oneline --all -i --grep=c42` (and broader searches for
dssim/downsample/structural/sepssim) did **not** surface C42-named commit
subjects in this benchmark repo — C42 work was committed under descriptive
names and lives largely in **external patch repos** referenced by the 13-scene
reports. Concrete commit references found in reports:

- **Benchmark repo git HEAD (completion batch):** `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`; git describe `baseline/reference-v1-absgrad-3-g32ab80e` (`reports/c42-completion-batch.md` §7).
- **Canonical Reference V1 commit:** `ad18916` (tag `baseline/reference-v1-absgrad`) (`reports/phase-s2.2-c42-generalization.md` §0).
- **Speedy-Splat C42 patch commits** (repo `j-alex-hanson/speedy-splat`, branch `experiment/c42-on-speedysplat`): `07f4712`, `aa75d8d`, plus a GUI fix (`reports/speedysplat-c42-13scene-final.md` §2.2, §10).
- **FastGS C42 patch commits** (repo `fastgs/FastGS`, branch `experiment/c42-on-fastgs`): `8b47521`, `d4d33b6`, plus a GUI fix (`reports/fastgs-c42-13scene-final.md` §2.2, §10).
- **Faster-GS:** patch in `Loss.py` (C42_SCALE attribute + conditional downsample) and `Trainer.py` (C42_SCALE=1.0 in LOSS ConfigParameterList); baseline commit `3cb0b75` (`reports/fastergs-c42-13scene-final.md` §2, §10).

A limitation: I could not enumerate every in-repo commit that touched C42
reports/scripts because their subjects are not C42-keyed; the references
above are the ones explicitly cited in the evidence files.

---

## 7. Contradictions and Unresolved Items

1. **Task hypothesis vs reality.** The brief assumed C42 was "scale/gaussian
   quantization or compression." All evidence shows C42 is downsampled-SSIM
   (loss-side). No quantization/bit-width/clustering of Gaussians is involved.

2. **"Primary" scale flips between reports.** `final_c42_decision.md` declares
   scale **0.50 primary** (+60.7%) and 0.75 fallback — based on *isolated/E2E
   timing + gradient cosine* only. But the actual *training-validation*
   reports (`c42_p0_training_validation.md` P0, `c42_p1_training_validation_0.75.md`
   P1, `track_a_c42_scale_sweep.md`) show scale 0.5 **FAILS** the SSIM and
   Gaussian-count gates (P0: dSSIM −0.0062, dGS +17.8%) while scale 0.75
   **PASSES all gates** and is declared "primary candidate." The
   "primary/0.50" label in `final_c42_decision.md` is not reconciled with the
   P0/P1 training gate results. (`final_c42_decision.md` §6 even lists
   end-to-end training validation as "remaining work, not in scope.")

3. **P1 (+33.6%) vs Track B (+6.4%) discrepancy.** Resolved in
   `reports/c42_discrepancy_analysis.md` as a pruning-schedule configuration
   mismatch (896K vs 23.8M Gaussians), not a measurement error — but it means
   the headline +33.6% number is configuration-dependent and only holds under
   continuous pruning.

4. **Adaptive decision oscillates KEEP → MODIFY.** `c42-completion-batch.md`
   says `KEEP` (12.55 ms) while the *later* `c42-b9-strong-fixed-boundary.md`
   says `MODIFY` (3.11 ms) after searching a stronger fixed baseline. The two
   use different fixed comparators (best fixed *tested* vs best fixed
   *feasible including new 0.80–0.95 scales*). Which is "final" depends on
   which comparator is canonical — not settled in the files.

5. **Counterfactual bound wording corrected mid-stream.** `c42-adaptive-feasibility.md`
   §B6-F changes "upper bound" to "neither upper nor lower bound" — earlier
   reports' "upper bound" claims are superseded.

6. **Wall-time speedup does not generalize to fused-SSIM baselines.** On
   Reference V1 / Speedy-Splat C42 gives 1.084× wall speedup, but on FastGS
   (fused_ssim CUDA kernel) C42 is **0.924× (slower)** and on Faster-GS
   0.998× (parity) — the `F.interpolate` overhead dominates when SSIM is
   already cheap (`reports/fastgs-c42-13scene-final.md` §8–9). The
   "+60.7% E2E" headline is Reference-V1/A100-specific.

7. **Both 10K and 30K room runs diverge** (PSNR 21.8→12.7 over 30K) on this
   gsplat version (`c42_post_validation_audit.md`, `candidate_parallel_screening.md`
   §C42-P2). A-vs-B comparisons are held valid because both diverge equally,
   but absolute quality numbers (e.g., PSNR 12.68) are not production-quality
   — a confound for any quality-preservation claim.

8. **Garden data-quality outlier.** All three baselines flag garden as using
   manually-created COLMAP (PSNR ≈12–13 for both native and C42); its
   anomalous Gaussian/wall-time behavior is a data artifact, not C42 — but it
   is included in aggregate means in the reports.

9. **Stump is a consistent, unmitigated degradation** across all three
   baselines (Speedy-Splat ΔSSIM −0.019, FastGS −0.044/−0.44 dB, Faster-GS
   −0.040/−0.94 dB) — a mechanism limitation for high-frequency outdoor
   foliage with no proposed mitigation beyond "use a less aggressive scale."

10. **Task path mismatches.** Several requested paths don't exist as written:
    `reports/phase-a100/` → `reports/a100_validation/`; `results/phase-a42/`
    and `results/phase-c42/` → `results/a100/phase-c42/`; `configs/**/*c42*`
    finds no config files (C42 config is inline in scripts and
    `baseline/reference_v1/config.py`). The digest used the actual paths.
