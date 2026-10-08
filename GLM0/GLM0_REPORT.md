# GLM0 Report — Gaussian Lifecycle Management Feasibility Audit

**Date:** 2026-10-01 (Asia/Shanghai). **Host:** EPIC-05, 8x A100-SXM4-80GB.
**Verdict: GLM0_PROMISING** — enter GLM1 (lifecycle-aware optimizer scheduling).
**Scope:** analysis only. Frozen C0 trainer, seed 42, 30K iters, canonical settings.
No production optimizer / densification / pruning / VOS / LAVOS / ReproTopo code changed.

---

## 1. Executive summary

GLM0 tests whether Gaussian **lifecycle** (birth -> growth -> mature -> inactive -> death)
is a new, exploitable 3DGS training-optimization dimension **beyond visibility**.

- **C1 Stable lifecycle pattern — PASS.** An unsupervised GMM (BIC k=4-5) finds a
  consistent NEW/MATURE/INACTIVE structure on all 3 scenes; an operational 3-stage rule
  is stable across threshold choices; the stage transition matrix is diagonally dominant
  (stages persist).
- **C2 Lifecycle != visibility — PASS.** corr(stage, vis_frac) max 0.43 (not
  deterministic); INACTIVE gaussians are sometimes visible; the additional headroom comes
  from **visible-but-mature** gaussians that a visibility oracle updates every iteration.
- **C3 additional_safe_skip_headroom >= 10% — PASS.** **12.84%** pooled (room 12.8%,
  bicycle 10.6%, garden 15.7%) at MATURE update-frequency 2; robust 12.6-14.5% across
  visibility/age thresholds.

**Headline:** visibility methods already skip the 73.65% of optimizer updates spent on
invisible gaussians. Lifecycle adds a further **12.84%** safe-skip headroom (= 48.7% of
the visible-update residual) by reducing the update frequency of visible-but-mature
gaussians — a dimension orthogonal to visibility.

---

## 2. Environment and protocol

- Frozen C0 recipe: seed 42, 30000 iters, densify 500-15000 @100, opacity reset every
  3000 (<15000), SH oneup every 1000, tile 16, loss = 0.8*L1 + 0.2*(1-SSIM) fused,
  `--eps2d 0.3`, eval SepSSIM + LPIPS(vgg). Renderer: F9 + SCALAR_ADJOINT + H8_MR, PX=2.
- Scenes: room (311 cams), garden (185 cams), bicycle (194 cams). garden traced fresh this
  campaign (final eval PSNR 29.01 / SSIM 0.8814 / LPIPS 0.1686); room + bicycle reuse the
  validated D0 frozen-C0 traces.
- Instrumentation is metadata-only (UID/lineage, per-iter visibility, cost samples); it
  consumes no training RNG and alters no semantic tensor. See GLM0_ENVIRONMENT.json.

---

## 3. Phase 1 — Gaussian lifecycle trace (GLM0_TRACE.json)

Per-Gaussian birth / survival / activity / death recovered from the frozen-C0 traces.

| Scene | N_init | N_final | births (clone/split) | deaths (opacity/world/split-parent) | alive@30K | lifetime median (p90) | alive frac |
|---|---|---|---|---|---|---|---|
| room | 112,627 | 835,653 | 1,059,732 / 443,774 | 556,919 / 1,674 / 221,887 | 835,653 | 1,700 (6,100) | 0.517 |
| bicycle | 54,275 | 3,681,789 | 3,995,698 / 2,516,784 | 1,615,161 / 11,415 / 1,258,392 | 3,681,789 | 1,600 (6,000) | 0.561 |
| garden | 138,766 | 2,568,130 | 2,581,131 / 1,704,272 | 999,555 / 4,348 / 852,136 | 2,568,130 | 1,400 (6,300) | 0.580 |

- **Births** are overwhelmingly clone/split (densification); split-parent removals mirror
  split children (topology arithmetic consistent). Screen prune is structurally
  unreachable on C0 (max_radii2D zeroed before the prune mask) — all non-opacity prunes
  are world prunes.
- **Lifetime** is short-tailed: median ~1,400-1,700 iters, p90 ~6,000; ~52-58% of
  gaussians survive to 30K. Most churn (birth+death) is confined to the densify window
  (<=14,900); after 15K the population is frozen.
- **Activity** spans a wide visibility spectrum (vis_frac 0 -> 1) with contribution
  (tiles) strongly tied to visibility (log-corr 0.80-0.91).

---

## 4. Phase 2 — Lifecycle stage discovery (GLM0_STAGE_DISCOVERY.json)

Unsupervised GMM (diagonal covariance) on standardized lifetime features
(log age, vis_frac, log tiles, parent opacity). BIC selects k=4-5 (fine, continuum-like
structure); the operational lifecycle is the goal 3-stage granularity:

- **INACTIVE:** vis_frac < 0.05 (low contribution)
- **NEW:** vis_frac >= 0.05 and age < 3000 (young, ramping)
- **MATURE:** vis_frac >= 0.05 and age >= 3000 (established, contributing)

| Scene | NEW | MATURE | INACTIVE | median vis_frac (NEW/MATURE/INACTIVE) |
|---|---|---|---|---|
| room | 0.264 | 0.566 | 0.170 | 0.318 / 0.303 / 0.019 |
| bicycle | 0.240 | 0.620 | 0.140 | 0.150 / 0.117 / 0.028 |
| garden | 0.222 | 0.553 | 0.225 | 0.300 / 0.318 / 0.029 |

- **Q1 (young/mature/inactive distribution):** yes — a low-contribution INACTIVE tail
  (~14-23%), a dominant established MATURE cohort (~55-62%), and a young NEW cohort
  (~22-26%).
- **Q2 (stage durations):** NEW concentrates at age < 3000; MATURE/INACTIVE span the full
  lifetime. Median lifetime by stage tracks the age split.
- **Q3 (transitions):** the early->late visibility transition matrix is diagonally
  dominant (persistence 0.85-0.99 on the diagonal); gaussians predominantly hold their
  visibility level, with a low->mid->high progression. Stages are persistent, not
  rapidly flickering.
- **visibility_relation:** corr(stage, vis_frac) 0.22-0.43 (related, not reducible).
- **gradient_relation:** corr(stage, grad) weak; gradient is bimodal (many newborns die
  before accumulating gradient), confirming gradient alone is not the lifecycle signal.

---

## 5. Phase 3 — Lifecycle headroom oracle (GLM0_HEADROOM.json)

Pure accounting on the 1,500 sampled iterations/scene (no real skip). Per-Gaussian
3-stage -> skip weight: INACTIVE skip (w=1), MATURE update every `freq` iters (w=1-1/freq),
NEW full update (w=0).

| Config | VOS headroom | GLM headroom | **additional** | additional / visible-residual |
|---|---|---|---|---|
| freq=1 (INACTIVE skip only) | 0.7365 | 0.7413 | **0.0048** | 0.018 |
| **freq=2 (primary)** | 0.7365 | 0.8649 | **0.1284** | **0.487** |
| freq=4 | 0.7365 | 0.9267 | **0.1902** | 0.712 |

Per-scene additional headroom (freq=2): room 0.1284, bicycle 0.1062, garden 0.1574.
Sensitivity (freq=2): TV in {0.03,0.05,0.10,0.15} -> 0.1265-0.1446; TA in {1500,3000,5000}
-> 0.1263-0.1302. Stable.

**Interpretation.** The visibility oracle (VOS) already removes 73.65% of updates
(invisible gaussians). The additional 12.84% is dominated by the **MATURE frequency-
reduction** lever (INACTIVE-skip alone is only 0.48%, because INACTIVE gaussians are mostly
invisible and VOS already skips them). Lifecycle reaches the visible-but-mature majority
that visibility cannot distinguish from young-but-visible gaussians.

---

## 6. Phase 4 — Novelty positioning (GLM0_POSITIONING.md)

Closest prior art (repo-verified): **GS-Scale** (ASPLOS 2026) defers zero-gradient /
invisible Gaussians via a counter; **Sparse Adam / AdamW-GS** (ICLR 2026) is visibility-
aware sparsity; **VOS-GS** (this repo) virtualizes invisible-optimizer state; **LAVOS**
(lineage-aware exact VOS) and **ReproTopo** address lineage fidelity / topology
correctness, not scheduling; **FastGS** is densification acceleration. None reduces the
update frequency of **visible-but-mature** Gaussians. GLM0 opening = the maturity axis on
top of the visibility baseline. Caveats: the INACTIVE-skip sub-lever overlaps prior art;
the 12.84% is a theoretical oracle requiring GLM1 quality validation.

---

## 7. Verdict and criteria (GLM0_VERDICT.json)

**GLM0_PROMISING** (C1 AND C2 AND C3). No human override; all from trace + oracle + novelty
audit.

- C1 stable lifecycle pattern: PASS
- C2 lifecycle != visibility: PASS
- C3 additional_safe_skip_headroom >= 10%: PASS (12.84%)

**Paper decision:** proceed to **GLM1 — lifecycle-aware optimizer scheduling**: a
quality-gated scheduler that reduces update frequency for visible-but-mature Gaussians on
top of the VOS visibility baseline, with PSNR/SSIM/LPIPS non-inferiority gates.

---

## 8. Caveats and limitations

1. The headroom is a **theoretical oracle** assuming mature-Gaussian updates can be halved
   without quality loss. GLM1 must validate with a quality gate (as VOS did). If mature
   reduction is quality-unsafe, realized headroom collapses toward the INACTIVE-only 0.48%.
2. The effect is **scene-dependent** (garden 15.7% > room 12.8% > bicycle 10.6%); all pass
   the 10% gate at freq=2, but bicycle is the marginal case.
3. Frozen-C0 run-to-run nondeterminism (~1% N spread) applies; the oracle uses per-scene
   sampled-iteration visibility, so the ratio is seed-42 frozen and internally consistent.
4. The 3-stage rule thresholds (TV=0.05, TA=3000) are operational discretizations of a
   continuum; sensitivity analysis shows the headroom is robust to these choices.

---

## 9. Deliverables and reproducibility

GLM0/ contains: GLM0_ENVIRONMENT.json, GLM0_TRACE.json, GLM0_STAGE_DISCOVERY.json,
GLM0_HEADROOM.json, GLM0_POSITIONING.md, GLM0_VERDICT.json, GLM0_REPORT.md,
MANIFEST.sha256. Working artifacts (per-uid tables, cost aggregates, stage profiles) and
all scripts are under /root/glm0/{work,artifacts,scripts} on EPIC-05. The frozen-C0 trace
infrastructure reuses the validated D0 pattern; garden was traced fresh this campaign.
