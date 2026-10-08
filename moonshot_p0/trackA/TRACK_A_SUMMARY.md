# Track A — PSC-GS / Projection-Space Consensus GS (P0) — VERDICT: **DROP**

**Date:** 2026-09-30  **Host:** EPIC-05 A100 (NVIDIA A100-SXM4-80GB, dsw-933622-64cc845c7d-lqswc)

## Scientific question
Can high-frequency view-indexed projected-state updates be periodically synchronized back to ONE
shared 3D Gaussian state? (H2 = PSC promotes; H1 = image-rate 3D consistency required, PSC does not promote.)

## Prior state: BLOCKED (resolved)
The frozen PSC-P0 packet was previously **BLOCKED** with `blocker_class=EXECUTION_HOST_MISMATCH`:
the prior reprobe ran on the local WSL Ubuntu-24.04 host with an RTX 5070 Laptop GPU, where the frozen
Linux artifacts under /root/epic05_* were inaccessible (path NOT_FOUND, missing bytes are NOT a hash
failure). Scientific verdict state was NONE / NOT_STARTED.

## Resolution: host provenance + artifact identity now VERIFIED on EPIC-05
EPIC-05 is the intended formal A100 host. Re-probing as root, ALL frozen identities match byte-for-byte:

| identity | expected sha256 | on-disk | match |
|---|---|---|---|
| room@25K checkpoint | b31e5a6b...32fc047d | b31e5a6b...32fc047d | YES |
| C0 extension | d26cccf4...113f11* | d26cccf4...b7dacf3197258a513f11 | YES |
| core gsplat_cuda.so | 935ea6c0...d1ff37e | 935ea6c0...d1ff37e | YES |
| trainer p12 | 675b0267...49050c5 | 675b0267...49050c5 | YES |
| frozen worktree | /root/epic05_frozen/higs_c0_final30k_worktree | present | YES |

\*The prereg constant for c0_extension is 63 hex chars (malformed, not a valid SHA-256). The probe
binds it to the authoritative P12 source-identity value (64 chars) and records this as a DOCUMENTED
DEVIATION (a prereg transcription typo), NOT a FAIL->PASS upgrade.

## Execution: frozen probe run to completion (with bug fixes)
The frozen probe `psc_p0_probe.py` had never executed (it stopped at the host gate). On EPIC-05 it
reached the scientific stages but hit several latent implementation bugs (the packet was untested on
the correct host). The following MINIMAL fixes were applied to make the frozen instrument runnable;
the verdict GATES (`psc_p0_analyze.py`) and the scientific protocol were NOT altered:

1. `compute_J_cohort`: pass CONTIGUOUS means/quats/scales to `fully_fused_projection` (matches the
   proven `view_data` pattern; non-contiguous slices triggered a gsplat shape-dispatch error).
2. `conics_to_Sigma`: NaN/Inf-safe clamp of the conic diagonal + `pinv` (degenerate/measure-zero
   gaussians produced singular 2x2 conics; `R_int` is scale-invariant so the clamp is neutral).
3. `grad_conic_to_Sigma`: `pinv` instead of `solve` for the conic->Sigma Jacobian (singular Jf on
   degenerate gaussians).
4. `r_int_and_loo`: removed spurious `.unsqueeze(-1)`/`.squeeze(-1)` on three einsums (operand-dim
   mismatch; the equations as written could not execute — a latent logic bug).

These fixes complete the intended measurement; they do not change the gate thresholds or the analyzer's
mechanical verdict logic.

## Results (room@25K, 8 frozen cameras via farthest-point sampling [19,140,219,73,40,31,255,13])
- Timing (camera 19, 100 reps): T_screen=25.26 ms, T_global=1.01 ms (proj 0.25 + vjp 0.28 + opt 0.49),
  T_native=26.28 ms, f_global=0.039, S_max=1.040.
- **S_upper(K) = (T_screen+T_global)/(T_screen+T_global/K):** K=2 -> 1.020, K=4 -> 1.030, **K=8 -> 1.035**.
- R_int_energy (primary): B2=0.998, B4=0.949, B8=0.744.
- Leave-one-out (primary, B8): P_hold_pos=0.513, median_C_hold=0.000.

## Mechanical verdict (frozen `psc_p0_analyze.py`): **PSC_P0_DROP**
Drop rules fired:
- **S_upper_8 < 1.30** (1.035) — the projection-space consensus gives almost no amortization: the
  "global" geometry/opacity step is only ~3.9% of native, so amortizing it behind K cheap projected
  steps yields only ~4% speedup, far below the 1.30 threshold.
- **P_hold_pos_B8 <= 0.55** (0.513) — leave-one-out position consistency is below threshold: the
  high-frequency projected updates do not hold a consistent shared 3D state across views.

Strong checks: S_upper_8>=1.50 FALSE; R_int_energy_B8>=0.65 TRUE; P_hold_pos_B8>=0.75 FALSE;
R_int_energy_B8>=0.5*R_int_B2 TRUE. Precedence DROP > STRONG_PROMOTE > INCONCLUSIVE => **DROP**.

## Interpretation
H1 is supported: image-rate 3D consistency is required. The PSC mechanism does not deliver the
expected projection-space amortization (S_upper ~1.04) and the multi-view 3D sync is inconsistent
(P_hold_pos 0.513). The R_int_energy (integrability) is high at small B (0.998 at B=2) but decays
(0.744 at B=8), consistent with projected updates being locally integrable but not globally
synchronizable. No threshold tuned; gates applied as pre-registered.

## Deliverables
PSC_P0_VERDICT.json (DROP), PSC_METRICS.json, PSC_P0_REPORT.md, TIMING_SUMMARY.json,
TIMING_RAW.json, CAMERA_SET.json, COHORT.json, FROZEN_IDENTITIES.json (all_match=true),
ENVIRONMENT.json, ANALYZER_STDOUT.txt, PSC_RAW_{PRIMARY,DIAGNOSTIC}.npz, MANIFEST.sha256,
psc_p0_probe.py (bug-fixed), psc_p0_analyze.py (frozen). Prior BLOCKED reprobe preserved in
PSC_P0_RUN/_blocked_archive/ on EPIC-05 and in git history locally.
