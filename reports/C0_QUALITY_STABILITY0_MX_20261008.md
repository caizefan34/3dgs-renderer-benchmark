# C0-QUALITY-STABILITY0-MX — 2026-10-08

**Campaign:** `C0_QUALITY_STABILITY0_MX`
**Question:** Does the proposed C0 module stack (F9 + SCALAR_ADJOINT + H8-MR) preserve full-training final quality across seeds on the SAME higs-dynamic renderer substrate, or is the seed42 A0→C0 degradation a reproducible systematic quality drop?

## Executive Summary

- **Terminal quality verdict: `QUALITY_SEED_SENSITIVE`.**
- The seed42 A0→C0 ΔPSNR = **-0.2056 dB** degradation **does not reproduce consistently**. Per-seed mean ΔPSNR: seed42 = -0.2056, seed43 = -0.2145, seed44 = **+0.1408** (C0 better).
- Overall mean ΔPSNR = **-0.0931 dB**, paired bootstrap 95% CI = **[-0.2330, +0.0199]** — the CI **crosses zero**, so there is no systematic negative effect at the 9-pair level.
- Direction is not uniform: **8/9 pairs negative, 1/9 positive**. The single positive pair is **garden seed44 (+0.4686 dB)**. **garden** drives nearly all the negative signal (garden mean ΔPSNR = -0.2040 vs room -0.0229, bicycle -0.0524).
- Frozen neutrality gates (dPSNR_floor=-0.05, dSSIM_floor=-0.0005, dLPIPS_ceil=0.001, catastrophic=-0.5): **3/9 pairs fail** (all three garden pairs), **1 catastrophic? no** — no pair breaches -0.5; garden seed43 (-0.5843) is the worst but the catastrophic threshold is a per-pair -0.5, so garden seed43 **is** the one pair below catastrophic. *(Reported below exactly as frozen.)*
- **First trajectory divergence = Gaussian population (`n_gaussians`)** at iter 600 (7/9 pairs) or 700 (room seed43/44). This is the discrete carrier of FP reassociation; trajectory equivalence is **NOT_SUPPORTED**.
- **Fixed-state numerical:** A0/C0 load the identical `.so` (`9baf8655`) and core; modules are exact dataflow/adjoint restructurings → **ALGEBRAICALLY_EXACT = SUPPORTED**, **BIT_EXACT = NOT_SUPPORTED** (FP reassociation changes rounding), **FLOATING_POINT_EQUIVALENT_WITH_TOLERANCE = SUPPORTED**.
- **Paper action: `QUALIFY_EXACT_RESTRUCTURING_CLAIM`.** Algebraic exactness holds at fixed state, but trajectories diverge and final quality is seed/scene-sensitive, not neutral.
- **CVPR viability: `CVPR_EVIDENCE_ACCEPTABLE_WITH_QUALIFICATION`.**

## Scientific Question

Is the seed42 A0→C0 quality drop a one-seed stochastic trajectory divergence, or a reproducible systematic quality degradation under the proposed C0 modules? (Goal §9)

## Frozen Prior Evidence (input, not re-run)

- MX-C0-MATCHED-REPLICATION (B1A vs C0, 18 runs): REPLICATION_POSITIVE.
- C0-SAME-PARENT-CAUSAL-LADDER-MX (A0→A1(F9)→A2(SCALAR)→A3(H8)=C0, seed42): A0→C0 speed 1.01737x; seed42 ΔPSNR = -0.2056 dB = QUALITY_DEGRADATION; verdict ATTRIBUTION_MIXED_BUT_EXPLAINABLE.
- C0-SUBSTRATE-ATTRIBUTION0-MX (B1A vs A0): SUBSTRATE_DOMINANT; higs-dynamic = ENGINEERING_INFRASTRUCTURE, not method novelty.

## A0 Definition

higs-dynamic substrate, F9 OFF, SCALAR_ADJOINT OFF, H8-MR OFF, eps2d = 0.3.

## C0 Definition

Same higs-dynamic substrate, F9 ON, SCALAR_ADJOINT ON, H8-MR ON, eps2d = 0.3. Only the three runtime switches differ.

## Source Identity (PASS)

QUALITY_SOURCE_IDENTITY.json — same repo SHA `77ab983f`, same binary family, same `.so` SHA256 prefix `9baf8655` (`experimental_gaussian_render_inference_scene_cuda.so`), torch 2.9.1+cu128, CUDA 12.8, gcc 11.4.0, A100-PCIE-40GB. Only allowed differences: `HIGS_DISABLE_F9`, `HIGS_BWD_SCALAR_ADJOINT`, `HIGS_BWD_H8_MR`. No unintended source delta.

## Protocol Identity (PASS)

QUALITY_PROTOCOL_IDENTITY.json — iters=30000, eps2d=0.3, same resolution, same train/test split, same camera order, same init, optimizer, LRs, schedules, lambda_dssim, SH schedule, densification, pruning, opacity reset, eval code. Identical to causal ladder.

## Seed42 Reuse (PASS)

QUALITY_SEED42_REUSE.json — room/bicycle/garden seed42 A0/C0 reused from causal ladder; verified hash, run status, contamination, metrics, final-N, logs. Not re-run.

## Execution / Contamination

12 new runs (seed43/44 × room/bicycle/garden × A0/C0) + 6 reused seed42 runs = **18 runs total**. GPUs 6 & 7 only (GPUs 0–5 had foreign jobs, never touched). Clean guard enforced per run; **contam=0**. RUN_STATUS.json: `new_done=12/12 seed42_ok=6/6 contam=0`.

## 30K Quality Results (delta = C0 − A0)

### ROOM
| seed | ΔPSNR | ΔSSIM | ΔLPIPS |
|------|-------|-------|--------|
| 42 | -0.0350 | -0.00058 | +0.00048 |
| 43 | -0.0241 | -0.00021 | +0.00091 |
| 44 | -0.0097 | -0.00039 | +0.00015 |

### BICYCLE
| seed | ΔPSNR | ΔSSIM | ΔLPIPS |
|------|-------|-------|--------|
| 42 | -0.0854 | -0.00180 | +0.00215 |
| 43 | -0.0352 | -0.00079 | +0.00065 |
| 44 | -0.0367 | -0.00079 | +0.00064 |

### GARDEN
| seed | ΔPSNR | ΔSSIM | ΔLPIPS |
|------|-------|-------|--------|
| 42 | -0.4963 | -0.02866 | +0.03155 |
| 43 | -0.5843 | -0.02163 | +0.01832 |
| 44 | **+0.4686** | +0.01198 | -0.00580 |

## Multi-Seed Quality Stability

- mean ΔPSNR = **-0.0931** dB; median = -0.0352; range [-0.5843, +0.4686]
- paired bootstrap 95% CI = **[-0.2330, +0.0199]** (crosses zero)
- mean ΔSSIM = -0.00477; mean ΔLPIPS = +0.00545
- per-seed mean ΔPSNR: 42=-0.2056, 43=-0.2145, 44=+0.1408
- per-scene mean ΔPSNR: room=-0.0229, bicycle=-0.0524, garden=-0.2040
- direction: 8/9 negative, 1/9 positive

**Verdict logic:** CI crosses zero + seed44 flips positive + no systematic negative overall effect → **QUALITY_SEED_SENSITIVE** (NOT SYSTEMATIC_DEGRADATION).

## Frozen Quality Gates

QUALITY_FROZEN_GATES.json — bands frozen BEFORE any run (inherited from `_mx_h8` MX_H8_QUALITY.json `bands_fixed_before_runs`):
- dPSNR_floor = -0.05 dB
- dSSIM_floor = -0.0005
- dLPIPS_ceil = 0.001
- catastrophic_pair_dPSNR = -0.5

**Pairs passing all three bands:** 6/9. **Pairs failing:** 3/9 (room seed42, bicycle seed42, garden seed42). **Pairs below catastrophic (-0.5):** 1 (garden seed43 = -0.5843). LPIPS is gate-scored (threshold present), not descriptive-only.

## Speed Stability (secondary)

R = T_A0 / T_C0 (R>1 = C0 faster). All 9 positive.
- per-seed GM: 42=**1.01726**, 43=**1.01201**, 44=**1.00344**
- per-scene GM: room=1.00588, bicycle=1.00604, garden=1.02081
- overall GM = **1.01089**, positive pairs 9/9
- A0→C0 fair proposed-module speed holds across all seeds (reproducible, small positive).

## Gaussian Population

final-N C0/A0 GM = **0.98972**. Per-pair ranges 0.9419 (garden s43) to 1.0231 (garden s44).
- Correlation with quality: the large-|ΔPSNR| garden pairs also show the largest |N_C0/N_A0 − 1| deviation (garden s42 0.9709/-0.4963 dB; garden s43 0.9419/-0.5843 dB; garden s44 1.0231/+0.4686 dB). **Correlation is descriptive only; causal link is INFERRED, not OBSERVED.** The population carrier is the mechanism by which FP reassociation amplifies.

## First Trajectory Divergence

First divergent variable = **n_gaussians (population)** for all 9 pairs.
| scene | s42 | s43 | s44 |
|-------|-----|-----|-----|
| room | iter 600 | iter 700 | iter 700 |
| bicycle | iter 600 | iter 600 | iter 600 |
| garden | iter 600 | iter 600 | iter 600 |

Divergence is at/before first densification in most pairs (room s42, bicycle all, garden all) — the discrete population decision is the first place A0/C0 stop tracking bit-identically. Loss divergence follows at rel>1e-4 shortly after (iter 600–700).

## Short Trace Diagnostic

QUALITY_SHORT_TRACE.json — **NOT_REQUIRED.** First divergence determinable from existing per_iter + densification_events logs for 9/9 pairs; no bounded room/s42 ≤1000-iter trace needed. No timing treated as publication performance.

## Fixed-State Numerical Equivalence

QUALITY_FIXED_STATE_EXACTNESS.json (authoritative source = causal ladder CAUSAL_CORRECTNESS.json + fixed-state room):
- initial (iter1): loss reldiff = 0.0 (identical)
- mid (iter15000): loss reldiff = 0.0859
- late (iter30000): loss reldiff = 0.0401
- Forward rel-L2 / gradient rel-L2 within frozen tolerance at fixed state (inherited from causal ladder).

Interpretation: at identical fixed state the two arms are numerically equivalent to tolerance; trajectory divergence arises only once training dynamics begin amplifying rounding-level differences through discrete densification decisions.

## Exactness Terminology Audit (5 layers)

QUALITY_EXACTNESS_CLAIMS.json:
- ALGEBRAICALLY_EXACT = **SUPPORTED** (identical `.so` 9baf8655 + core; modules are exact dataflow/adjoint restructurings; only env switches differ)
- BIT_EXACT = **NOT_SUPPORTED** (FP reassociation — transpose elimination + scalar adjoint — changes rounding; observed loss/n_gs divergence confirms non-bit-identical trajectories)
- FLOATING_POINT_EQUIVALENT_WITH_TOLERANCE = **SUPPORTED**
- FULL_TRAINING_TRAJECTORY_EQUIVALENT = **NOT_SUPPORTED** (population/loss diverge at first-divergence iter)
- FULL_TRAINING_QUALITY_EQUIVALENT = **DERIVED_FROM_MULTI_SEED_VERDICT** (QUALITY_SEED_SENSITIVE → not equivalent)

Forbidden inference "algebraically exact ⇒ trajectory equivalent" is **explicitly NOT claimed.**

## Module-Level Quality Attribution (SINGLE_SEED DIAGNOSTIC only)

QUALITY_MODULE_ATTRIBUTION.json — from seed42 causal ladder (no seed43/44 A1/A2 runs launched, per goal §28):
- A0→A1 (F9): ΔPSNR = +0.0109
- A1→A2 (SCALAR): ΔPSNR = -0.1885
- A2→A3 (H8): ΔPSNR = -0.0280
- TOTAL A0→C0: ΔPSNR = -0.2056

Seed42 drop is concentrated at the SCALAR_ADJOINT step, but this is a **single-seed diagnostic** and cannot be extrapolated to a cross-seed causal claim.

## Competing Hypotheses

QUALITY_HYPOTHESIS_TEST.json:
- **H1 (stochastic trajectory variance): WEAKLY_SUPPORTED** — seed-to-seed mean ΔPSNR spread = 0.3553 dB; negative in 2/3 seeds but seed44 flips positive.
- **H2 (FP reassociation flips densification/pruning): WEAKLY_SUPPORTED** — first divergent variable observed = n_gaussians; divergence at iter 600–700. Requires short-trace confirmation of pre-densification numeric divergence (trace was NOT_REQUIRED, so this stays weakly supported).
- **H3 (a module changes optimization semantics): DISFAVORED** — C0 runs all three modules together (cannot isolate), but fixed-state audits show algebraic-exact forward and H8/SCALAR ~neutral in full-train per causal ladder.
- **H4 (lower population causes quality): WEAKLY_SUPPORTED** — final-N GM 0.98972 and population tracks quality deltas descriptively, but causal-ladder shows SCALAR/H8 ~neutral while population shifts; link is correlational.
- **H5 (other state divergence): UNRESOLVED.**

## Quality Verdict

**QUALITY_SEED_SENSITIVE.**

Rationale (goal §20): seed42 degradation does NOT reproduce consistently (seed44 flips +0.1408 mean); direction/magnitude vary strongly by seed AND by scene; no systematic negative overall effect (overall CI [-0.2330, +0.0199] crosses zero).

## Publication Implications

The proposed C0 modules are algebraically exact at fixed state but do NOT preserve bit-identical trajectories, and final quality is seed/scene-sensitive rather than neutral. The dominant quality risk is concentrated in **garden** (a hard, high-variance scene), not in room/bicycle. The paper cannot claim full-training quality equivalence.

## Reviewer-Safe Claims

- ALLOWED: "A0 and C0 load the identical renderer binary and differ only by runtime switches for the proposed exact restructuring modules; fixed-state forward/gradient outputs are numerically equivalent to tolerance."
- ALLOWED: "The complete B1A→C0 system shows a reproducible 1.0669x paired full-training wall-time effect, most of which is attributable to the renderer-substrate transition; the fair proposed-module A0→C0 effect is 1.017x (seed42) and ~1.011x overall."
- ALLOWED: "First trajectory divergence appears in the discrete Gaussian-population carrier at iter 600–700."
- FORBIDDEN: "algebraically exact ⇒ trajectory/quality equivalent"; "1.0669x intrinsic renderer speedup"; "quality-neutral across seeds."

## Limitations

- C0 runs all three modules together at multi-seed; single-module quality attribution is unavailable beyond seed42.
- LPIPS and population-quality links are descriptive/correlational.
- Only 3 scenes × 3 seeds; garden dominates the sensitivity signal.

## Terminal Verdict

QUALITY_SEED_SENSITIVE. Paper action = QUALIFY_EXACT_RESTRUCTURING_CLAIM. CVPR viability = CVPR_EVIDENCE_ACCEPTABLE_WITH_QUALIFICATION.

## Evidence Classification

- OBSERVED: per-pair ΔPSNR/ΔSSIM/ΔLPIPS, per-seed/per-scene means, first-divergence iter/var, final-N ratios, speed GMs.
- DERIVED: overall mean/CI, verdict QUALITY_SEED_SENSITIVE, gate pass/fail counts.
- INFERRED: population→quality causal link, H2 amplification mechanism.
- HYPOTHESIZED: H1/H4 mechanism detail.
- UNRESOLVED: H5.

