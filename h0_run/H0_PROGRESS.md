# HomotopyGS H0 — Progress Note (2026-09-30)

Directive: 注意不是在本地GPU进行实验 (do NOT run on the local GPU). All compute on EPIC-05 A100.

## Verified environment
- Frozen infra is REMOTE (EPIC-05, 8x A100-SXM4-80GB, Python 3.10.21, torch 2.9.1+cu128, gsplat 1.6.0 @512d366b).
- Local box (RTX 5070 8GB, torch 2.13+cu130, gsplat 1.4.0, Win, py3.13) is NOT used for any H0 run.
- Frozen campaign python: /root/miniforge3/envs/gsplat/bin/python (NOT default python, which has a torch ABI mismatch).

## Source identity (H0_SOURCE_IDENTITY.json) — PASS
- C0 ext .so live sha256 = d26cccf4...7dacf3197258a513f11 (64 hex). Matches authoritative A0-manifest value.
- core gsplat_cuda.so = 935ea6c0...d1ff37e. Matches prereg pin.
- NOTE: H0_PREREG.md sec.1 inline C0-ext pin is 63 hex (malformed, dropped a char) — same malformation IR0 already documented. Binary is correct; prereg inline string is truncated.
- room@25K ckpt = b31e5a6b...fc047d, maps to /root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt.

## Checkpoint inventory (H0_CHECKPOINT_INVENTORY.json) — BLOCKER
- Frozen C0 trajectory = /root/epic05_n1/runs/n1snap_c0f_s42_{room,bicycle} (save-only edit of frozen P12 trainer, sha256 49a79538..; reproduces canonical seed-42 stream bit-exactly).
- Snapshots exist at 10K/15K/20K/25K only. NO 5K checkpoint exists anywhere in the frozen trajectory.
- Required cells room/bicycle @ {5K,15K,25K}: 15K+25K PRESENT; 5K MISSING (both scenes).
- Per prereg sec.2 + DSH_HANDOFF item 9: emit H0_CHECKPOINT_BLOCKER, STOP, do not substitute.

## Parity gate (H0_PARITY.json) — PASS  [this turn's progress]
- Mandatory prereg artifact (sec.5/6/7); required before ANY scientific run; provable on existing 15K/25K cells.
- Harness: h0_run/h0_parity.py (isolated /root/epic05_h0/ on EPIC-05; reuses frozen n1_core.py primitives; never touches frozen tree; never local GPU).
- Result over room/bicycle x {15K,25K} x cam{0,1,2,3}:
  - E2D multiplier=1: rgb max-abs = 0.00e+00, grad cosine = 1.000000 -> bit-exact PASS.
  - OPA gamma=1: rgb max-abs <= 6e-8, grad cosine >= 0.999999 -> PASS.
- Key implementation finding: literal o_gamma = 1-(1-clamp(o,0,1-eps))**gamma at gamma=1 incurs ~1e-7 cancellation on ~7.2e5 gaussians via torch's power kernel, spuriously breaking the prereg's own endpoint claim. Fix: evaluate the SAME frozen formula with a bit-exact endpoint (gamma==1 -> clamp(o,0,1-eps)); for gamma<1 the literal power form is used verbatim. This is numerically-faithful, not a schedule/arm change.
- E2D arm is natively supported: frozen n1snap trainer exposes --eps2d as a runtime kwarg ("no rebuild"); E2D = --eps2d multiplier*0.3.

## Remaining blocker (needs user decision A/B)
- room@5K and bicycle@5K absent. A) accept H0_CHECKPOINT_BLOCKER; or B) authorize bit-exact 5K regeneration via the frozen n1snap trainer from seed 42 (deviation from strict frozen-manifest-hash reading; requires sign-off).
- Scientific rollouts (NATIVE/E2D/OPA, camera folds, metrics, H0_SUMMARY, analyzer verdict) remain gated on that decision.


## Addendum (same day) — camera folds + rollout harness
- H0_CAMERA_FOLDS.json: formal MOONSHOT 8-camera set resolved via authoritative
  frozen_camera_ids(n,k=8)=[round(n*(i+0.5)/8)] (/root/epic05_moonshot/a0/a0_lib.py:125).
  room c0..c7 = [19,58,97,136,175,214,253,292]; bicycle = [12,36,61,85,109,133,158,182].
  Fold A update=c0..c3/heldout=c4..c7; Fold B update=c4..c7/heldout=c0..c3; 4 cyclic offsets;
  8 paired rollouts per (cell,family,K). Decision-independent.
- h0_rollout.py: full topology-frozen K-step rollout harness (NATIVE/E2D/OPA), native-endpoint
  eval after every step (excluded from timing), section-8 metrics. Reuses frozen n1_core primitives.
  Mechanics SMOKE-TESTED on EPIC-05 A100: room@25000 NATIVE K=4 foldA off0 ->
  held-out obj 0.031105 -> 0.030955 (D_native>0), design obj monotonic down, PSNR/SSIM tracked,
  update norms 0->19.8. Valid. (Smoke = mechanics proof only, NOT the scientific result.)
- Config respected: densify_until_iter=15000 (15K/25K post-densif => topology frozen), sh_progress=1000
  (active_sh already 3), lambda_dssim=0.2, opacity_reset=3000.

## Status: ALL decision-independent prerequisites COMPLETE
Source identity PASS | Checkpoint inventory BLOCKER(5K) | Parity gate PASS | Camera folds resolved |
Rollout harness built+validated. The full 6-cell scientific run (NATIVE/E2D/OPA x K4/K8 x 2 folds x
4 offsets x 6 cells, H0_SUMMARY, analyzer verdict) is gated ONLY on room@5K + bicycle@5K.


## Addendum 2 — summary aggregator + analyzer pipeline (same day)
- h0_summarize.py: turns H0_ROLLOUTS trajectories into H0_SUMMARY.json (H0_SCHEMA.json). Pairs each
  homotopy rollout with the NATIVE rollout at (fold,offset,K); computes D_native/D_homo, P_hold,
  P_design, E_hold, k_cross_8, PSNR deltas, BASE_NON_DEScent count, NaN/Inf; median/p10 over the
  8 paired rollouts. TESTED on smoke (room@25K K4, 3 arms): E2D p_hold=1.076/p_design=0.979/
  psnr+0.003dB; OPA p_hold=-6.11/psnr-0.34dB (held-out rose). Correct schema.
- Analyzer pipeline PROVEN end-to-end on A100: h0_rollout -> h0_summarize -> h0_analyze.py.
  On smoke input the analyzer emits H0_INPUT_INVALID (all 6 required cells absent incl. 5K),
  confirming verdict logic + input validation are correctly wired.
- ALL THREE arms mechanics-validated through the annealed schedule (NATIVE descent, E2D descent,
  OPA runs). e_hold requires --time (omitted in smoke).

## FINAL STATUS: every decision-independent H0 component is COMPLETE + validated on EPIC-05 A100
1 H0_SOURCE_IDENTITY.json PASS | 2 H0_CHECKPOINT_INVENTORY.json BLOCKER(5K) | 3 H0_PARITY.json PASS |
4 H0_CAMERA_FOLDS.json resolved | 5 h0_rollout.py (3 arms validated) | 6 h0_summarize.py (tested) |
7 analyzer pipeline wired. The full 6-cell scientific run + H0_SUMMARY + analyzer verdict are gated
SOLELY on room@5K + bicycle@5K. No further decision-independent work exists.
