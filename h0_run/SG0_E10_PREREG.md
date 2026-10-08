# SplatMG SG0-E10 PREREG (reduced 10K cohort, E0 viability)

## 0. Authority and namespace
- SEPARATE preregistered experiment. Verdict namespace `SG0E10_*`. Does NOT
  overwrite the original SG0 verdict (`SG0_BASELINE_UNSTABLE`, preserved) or the
  original SG0_PREREG.
- Original sg0_analyze.py is NOT reused (it targets 20K/25K + oracle). This
  experiment has its OWN analyzer(s).

## 1. Why 10K (reduced cohort)
- Original SG0-EARLY planned room/bicycle @ {5K, 10K}. Under user decision A,
  canonical 5K does NOT exist and MUST NOT be substituted with regenerated 5K.
- Canonical 10K EXISTS and is byte-exact: room@10000=9aa00f0d…,
  bicycle@10000=26bf0b95…. SG0-E10 = room@10000 + bicycle@10000 (2 cells).
- STRICT REDUCTION of the 4-cell grid. Gates UNCHANGED. A SPLATMG_CLOSED here is
  terminal for the SplatMG-Early line on the available cohort.

## 2. Camera folds (frozen, identical to SG0/H0)
- ids=[round(n*(i+0.5)/8) for i in 8]; room=[19,58,97,136,175,214,253,292],
  bicycle=[12,36,61,85,109,133,158,182]. Fold A design=c0..c3/holdout=c4..c7;
  Fold B reverse. 8 rollout orders (2 folds x 4 cyclic offsets).

## 3. Stage E0 (this prereg's first gate) — native baseline viability
- Run ONLY native Adam K=4 and K=8 on the design cameras (frozen optimizer state,
  clip_norm=1.0, exact a0_lib renderer/loss). NO hierarchy, NO oracle.
- For each rollout: base_non_descent = NOT (LK_holdout < L0_holdout).
- Metric: base_non_descent_K8_count per cell (out of 8).
- GATE (section C1): EVERY cell must have base_non_descent_K8 <= 2/8.
  - ANY cell fails -> SPLATMG_CLOSED (terminal; do NOT search 3K/7K/12K/other
    cameras/other K).
  - ALL cells pass -> SG0E10_E0_PASS.

## 4. Stage C2 (coarse-space oracle) — NOT run here
- Only after E0_PASS, and as a SEPARATE later stage: original SG0 oracle
  (L32/L128 x SIM13/AFF22) on the 10K cohort, with the original SG0 gates
  (sg0_analyze.py thresholds) over the 2 cells. That stage gets its OWN prereg
  addendum and analyzer before execution. E0_PASS does NOT imply C2 will run
  automatically; it only authorizes designing C2.

## 5. Non-determinism note (pre-registered)
- Each rollout starts from the frozen 10K checkpoint; cross-run non-determinism
  (atomicAdd) makes single rollouts noisy. base_non_descent_count over 8 rollouts
  is the designed robustness. The analyzer owns the verdict; no human override.
- LAW-MATURE-HELDOUT context: H0-FROZEN showed mature (15K/25K) native held-out
  non-descent. 10K is earlier; E0 tests whether 10K is early enough for native
  held-out descent. If E0 fails, that is consistent with the law and CLOSES
  SplatMG-Early on the available cohort (no rescue).

## 6. Pipeline
- sg0e10_e0.py (per cell, 1 GPU) -> sg0e10_e0_finalize.py (aggregate 2 cells) ->
  sg0e10_e0_analyze.py -> SG0E10_E0_VERDICT.json.
- Artifacts under /root/epic05_sg0e10/ with sha256 manifest.