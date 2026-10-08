# H0-FROZEN PREREG (HomotopyGS, reduced frozen-checkpoint cohort)

## 0. Authority and namespace
- Supersedes NOTHING. This is a SEPARATE preregistered experiment with its OWN
  analyzer (`h0_frozen_analyze.py`) and verdict namespace `H0_FROZEN_*`.
- The original H0 prereg/analyzer verdicts (`H0_CHECKPOINT_BLOCKER`,
  `H0_R5K_REGEN_IDENTITY_FAIL`) are PRESERVED and NOT overwritten.
- Rationale (user decision A): C0 training is cross-run non-deterministic
  (custom CUDA rasterizer backward `atomicAdd`; torch determinism APIs cannot
  control it). Bit-exact regeneration of the canonical trajectory is
  UNSATISFIABLE. We therefore do NOT relax the identity gate and do NOT call
  regenerated checkpoints canonical. Instead we run H0 over the cells that
  genuinely exist in the frozen trajectory.

## 1. Cells (frozen checkpoints that EXIST)
- room@15000, room@25000, bicycle@15000, bicycle@25000
- Checkpoints read-only from `/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{iter}.pt`
- The 5K cells are ABSENT and are NOT substituted. No p13x/VOS substitution.

## 2. Reduced-cohort caveat (pre-registered, not post-hoc)
- Original H0 grid = 6 cells (incl. room@5K, bicycle@5K). H0-FROZEN = 4 cells.
- This is a STRICT REDUCTION. The gates are UNCHANGED (section 5). A
  H0-FROZEN_PROMOTE is therefore evidence over 4 mature checkpoints, NOT over
  the full 6-cell grid. It is labeled as such in every downstream report.
- If H0-FROZEN_DROP, HomotopyGS is CLOSED on the available frozen cohort.

## 3. Arms / K / folds / offsets (unchanged from H0_PREREG)
- Arms: NATIVE, E2D, OPA.  K in {4, 8}.  Folds A,B.  Offsets {0,1,2,3}.
- 8 paired rollouts per (cell, family, K).  Camera folds =
  ids=[round(n*(i+0.5)/8) for i in 8] (room=[19,58,97,136,175,214,253,292],
  bicycle=[12,36,61,85,109,133,158,182]); fold A update=c0..c3/heldout=c4..c7,
  fold B reverse.
- Continuation schedules (verbatim):
  - E2D eps2d mult: K4=[2.00,1.60,1.30,1.00]; K8=[2.00,1.85,1.70,1.55,1.40,1.25,1.10,1.00]
  - OPA gamma:      K4=[0.60,0.75,0.90,1.00]; K8=[0.60,0.68,0.76,0.84,0.90,0.94,0.97,1.00]
- Native endpoint eval after every step (measurement only, excluded from step timing).
- `--time` REQUIRED (E_hold gate needs step_times_ms).

## 4. Parity gate (reused frozen artifact)
- `H0_PARITY.json` cells are EXACTLY the 4 H0-FROZEN cells, all
  e2d_parity_pass=opa_parity_pass=true; global parity_pass=true. Per-cell
  parity_pass = e2d AND opa. Binary identities match prereg pins (c0_ext
  d26cccf4.., core 935ea6c0..).

## 5. Gates (ORIGINAL H0 gates, verbatim, unchanged)
- One fixed family (E2D or OPA) must satisfy ALL 4 cells.
- STRONG (per cell): parity_pass; K4 p_hold_med>=1.20/p10>=0.85/e_hold_med>=1.15;
  K8 p_hold_med>=1.30/p10>=1.00/e_hold_med>=1.20/p_design_med>=1.20/
  psnr_med>=-0.03/psnr_worst>=-0.10/base_non_descent<=2/nan_inf==0/n_rollouts==8.
- PROMOTE (per cell): parity_pass; K4 p_hold_med>=1.08/p10>=0.70/e_hold_med>=1.03;
  K8 p_hold_med>=1.15/p10>=0.80/e_hold_med>=1.08/p_design_med>=1.10/
  psnr_med>=-0.05/psnr_worst>=-0.15/base_non_descent<=2/nan_inf==0/n_rollouts==8.
- Else H0_FROZEN_DROP.

## 6. Non-determinism note (pre-registered)
- Each rollout starts from the frozen checkpoint (master.restore()); the K-step
  short rollouts diverge only slightly across runs. BASE_NON_DESCT<=2 and
  median/p10 over 8 paired rollouts are the designed robustness. The analyzer
  owns the verdict; no human override.

## 7. Pipeline
- h0_rollout.py (per cell, 1 GPU) -> h0_summarize.py --cells <4 dirs> ->
  h0_frozen_analyze.py H0_SUMMARY.json H0_PARITY.json -> H0_FROZEN_VERDICT.json.
- All artifacts under /root/epic05_h0frozen/ with sha256 manifest.