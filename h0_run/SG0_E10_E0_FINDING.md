# SplatMG SG0-E10 — E0 Viability Finding (2026-09-30)

## STATUS
SPLATMG_CLOSED  (analyzer-owned; namespace SG0E10_*)

Terminal for the SplatMG-Early line on the available frozen cohort. The
coarse-space oracle (section C2) was NOT run, per prereg section C1 (E0 fail ->
stop; do not search other checkpoints/cameras/K). Original SG0 verdict
(SG0_BASELINE_UNSTABLE on 20K/25K) is PRESERVED and NOT overwritten.

## SOURCE IDENTITY (live re-verified, byte-exact)
- c0_ext .so  d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- core .so    935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- n1snap trainer 49a79538d5418cdf795b8a8f9919b2e3f9bc5f7550c7f966b030e3ae14d8dbf4
- room@10000    9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4
- bicycle@10000 26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9
- renderer a0_lib (frozen MOONSHOT0); loss 0.8*L1+0.2*(1-SSIM) fused; seed 0; clip_grad_norm=1.0

## EXPERIMENT
- 2 cells (room@10000, bicycle@10000) x {native Adam K=4,K=8} x 8 rollout orders
  (2 folds x 4 cyclic offsets). EPIC-05 A100-SXM4-80GB, torch 2.9.1+cu128.
- NO hierarchy, NO oracle (E0 viability only). Frozen camera folds:
  room=[19,58,97,136,175,214,253,292], bicycle=[12,36,61,85,109,133,158,182].
- base_non_descent = NOT(LK_holdout < L0_holdout), native Adam on the 4 design
  cameras cycling the fold order.

## PRIMARY METRIC (prereg E0 gate)
- base_non_descent_K8_count per cell; gate = <= 2/8 on EVERY cell.
- room@10000:    K4=1/8, K8=4/8  -> FAIL (>2/8)
- bicycle@10000: K4=8/8, K8=8/8  -> FAIL

## GATES (section C1)
- E0 viability: EVERY cell base_non_descent_K8 <= 2/8.
- room@10000 K8=4 > 2 -> FAIL. bicycle@10000 K8=8 > 2 -> FAIL.
- ANY cell fails -> SPLATMG_CLOSED (terminal). Both fail.

## VERDICT
SPLATMG_CLOSED (analyzer-owned). failed_cells=[bicycle@10000, room@10000].
E0_PASS not reached; section-C2 oracle NOT authorized/run.

## INTERPRETATION
- OBSERVED: bicycle@10000 native single/few-step Adam NEVER reduces held-out loss
  (0/8 descent at K4 and K8; D_hold all negative, -0.0003..-0.0014). room@10000
  descends at K4 (7/8) but NOT at K8 (4/8): after cycling the same 4 design
  cameras a 2nd time, held-out reverses (design-set overfitting).
- EVIDENCE: This is the SAME mechanism as H0-FROZEN and the original
  SG0_BASELINE_UNSTABLE (LAW-MATURE-HELDOUT). It extends to 10K for bicycle and
  shows the design-overfitting reversal for room at K8.
- HYPOTHESIS: The K-step held-out-descent probe is unreliable because native
  Adam on a small design camera set does not generalize to held-out views at
  these checkpoints; the P_hold denominator is non-positive.
- TEST: Frozen E0 viability gate over 2 canonical 10K cells x 8 rollouts.
- CONCLUSION: SplatMG cannot demonstrate coarse-space oracle gain where the
  native baseline itself does not descend on held-out. CLOSED on the available
  cohort. No rescue, no checkpoint/camera/K search (prereg C1).

## NEXT ACTION
- Per prereg: SPLATMG_CLOSED is terminal. Record in CLOSED registry.
- All three executable tracks now resolved: HomotopyGS H0-FROZEN=DROP,
  SplatMG SG0-E10=CLOSED, AtomBirth AB0=BLOCKED. Per directive, proceed to
  Track D (baseline/novelty/CLOSED audit) and then the negative-evidence
  synthesis / bottleneck-model rebuild.

## ARTIFACTS
- remote /root/epic05_sg0e10/: SG0_E10_PREREG.md, sg0e10_e0.py,
  sg0e10_e0_analyze.py, sg0e10_e0_finalize.py, SG0E10_E0_SUMMARY.json,
  SG0E10_E0_VERDICT.json, cells/SG0E10_E0_{room,bicycle}@10000.json, run_e0.sh,
  run_e0.log
- reused: /root/epic05_sg0/sg0_lib.py + /root/epic05_moonshot/a0/a0_lib.py
- local mirror: 3dgs-renderer-benchmark/h0_run/