# HomotopyGS H0-FROZEN — Finding (2026-09-30)

## STATUS
H0_FROZEN_DROP  (analyzer-owned; namespace H0_FROZEN_*)

This is a SEPARATE preregistered experiment (H0_FROZEN_PREREG.md) with its OWN
analyzer. It does NOT overwrite the preserved original verdicts
H0_CHECKPOINT_BLOCKER / H0_R5K_REGEN_IDENTITY_FAIL. HomotopyGS is CLOSED on the
available frozen cohort (4 mature checkpoints).

## SOURCE IDENTITY (live re-verified, byte-exact)
- c0_ext .so  d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- core .so    935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- n1snap trainer (publication_trainer_p12_n1snap.py)
             49a79538d5418cdf795b8a8f9919b2e3f9bc5f7550c7f966b030e3ae14d8dbf4
- room@15000   b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd
- room@25000   b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d
- bicycle@15000 273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7
- bicycle@25000 73ae5436cc44a2c25dca26cb226e522047465e65b671717ac043578cbd7e21ea
- H0_PARITY.json parity_pass = true (all 4 cells; e2d AND opa endpoint parity)

## EXPERIMENT
- 4 cells x {NATIVE,E2D,OPA} x K{4,8} x folds{A,B} x offsets{0,1,2,3}
  = 48 rollouts/cell, 192 total. EPIC-05 A100-SXM4-80GB, torch 2.9.1+cu128,
  cuda 12.8, seed 42, 1080p, tile16, lambda_dssim 0.2, densify 500..15000.
- Camera folds (frozen): ids=[round(n*(i+0.5)/8)]; room=[19,58,97,136,175,214,253,292],
  bicycle=[12,36,61,85,109,133,158,182]. Fold A update=c0..c3/heldout=c4..c7; B reverse.
- Checkpoints read-only from /root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{iter}.pt.
- Rollouts from each frozen checkpoint are subject to cross-run non-determinism
  (atomicAdd); BASE_NON_DESCT<=2 and median/p10 over 8 paired rollouts are the
  designed robustness. --time used (E_hold needs step_times_ms).

## PRIMARY METRICS (prereg H0 gates; median over 8 paired rollouts)
- room@15000   K4 E2D p_hold=None(8/8 non-desc) OPA=None ; K8 both None(8/8)
- room@25000   K4 E2D p_hold=1.076 p10=0.274 e_hold=1.10 bnd=3 ; OPA p_hold=-6.11 bnd=3
               K8 E2D=None(8/8) OPA=None(8/8)
- bicycle@15000 K4 E2D p_hold=0.487 p10=-1.84 bnd=4 ; OPA p_hold=-7.01 bnd=4
               K8 E2D p_hold=-0.833 bnd=5 ; OPA p_hold=-27.13 bnd=5
- bicycle@25000 K4/K8 E2D=None(8/8) OPA=None(8/8)

## GATES (unchanged original H0 gates; None = BASE_NON_DESCT = FAIL)
- STRONG: needs all 4 cells K4 p_hold_med>=1.20/p10>=0.85/e_hold>=1.15 AND K8
  p_hold_med>=1.30/p10>=1.00/e_hold>=1.20/p_design>=1.20/psnr_med>=-0.03/
  worst>=-0.10/bnd<=2/nan=0/n=8.
- PROMOTE: K4 p_hold_med>=1.08/p10>=0.70/e_hold>=1.03 AND K8 med>=1.15/p10>=0.80/
  e_hold>=1.08/p_design>=1.10/psnr_med>=-0.05/worst>=-0.15/bnd<=2/nan=0/n=8.
- Per-cell pass: E2D fails (room@15000/bicycle@25000 None@K8; bicycle@15000 p_hold<1;
  room@25000 bnd=3>2 and p10=0.274<0.70). OPA fails everywhere (p_hold hugely negative).
- No fixed family satisfies all 4 cells.

## VERDICT
H0_FROZEN_DROP (analyzer-owned). selected_fixed_family=null. HomotopyGS CLOSED
on the available frozen cohort.

## INTERPRETATION
- OBSERVED: At mature checkpoints (15K/25K), native single/few-step Adam on ONE
  design camera produces near-zero or NEGATIVE held-out loss descent
  (room@25K K4 +0.0001, bicycle@25K -0.0002), while the DESIGN objective keeps
  descending. BASE_NON_DESCT=8/8 in 6 of 8 (cell,K) blocks.
- EVIDENCE: Where native does descend (room@25K K4, bicycle@15K K4/K8), E2D
  p_hold_median is 1.076 / 0.487 / -0.833 (<=1, i.e. homotopy NOT better than
  native), and OPA is catastrophic (-6.1 to -27.1). Arms are provably active
  (E2D/OPA p_hold != 1.0), so this is not a silently-native artifact.
- HYPOTHESIS: HomotopyGS (E2D eps2d anneal, OPA opacity anneal) provides no
  reliable held-out improvement over native Adam at maturity.
- TEST: The frozen prereg gates over 4 cells x 2 families x K{4,8}.
- CONCLUSION: The held-out P_hold denominator is unreliable at maturity because
  native K-step is itself at a held-out plateau. This is the SAME root cause as
  SG0_BASELINE_UNSTABLE. HomotopyGS cannot demonstrate held-out gain where the
  native baseline does not descend. CLOSED.

## NEXT ACTION
- Per prereg: H0_FROZEN_DROP -> HomotopyGS permanently CLOSED on frozen cohort.
- Proceed to SG0-E10 (P3): canonical 10K E0 viability (native Adam K4/K8
  BASE_NON_DESCT<=2/8). NOTE: H0-FROZEN already shows mature-checkpoint native
  held-out non-descent; SG0-E10 E0 is the designed gate to confirm/deny whether
  10K is early enough for native to descend on held-out. If E0 fails on any cell
  -> SPLATMG_CLOSED immediately (no oracle).
- AB0 remains BLOCKED (needs separately preregistered stochastic-replicate
  early-stage protocol; canonical 5K unavailable).

## ARTIFACTS
- remote /root/epic05_h0frozen/: H0_FROZEN_PREREG.md, h0_frozen_analyze.py,
  h0frozen_run.sh, H0_SUMMARY.json, H0_FROZEN_VERDICT.json,
  cells/{room,bicycle}@{15000,25000}/H0_ROLLOUTS.json, run.log
- reused: /root/epic05_h0/H0_PARITY.json (parity gate), h0_rollout.py, h0_summarize.py
- local mirror: 3dgs-renderer-benchmark/h0_run/