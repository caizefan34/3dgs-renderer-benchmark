# H0-R5K IDENTITY FAIL — Root-Cause Finding (2026-09-30)

## STATUS
H0_R5K_REGEN_IDENTITY_FAIL — **reproducibility BLOCKER**, NOT a scientific DROP of HomotopyGS.
The HomotopyGS continuation hypothesis was never tested (no valid canonical 5K starting state).

## VERDICT (analyzer-owned)
- Analyzer: `h0r5k_identity_gate.py` (frozen, sha256 265addb8…)
- Verdict: **H0_R5K_REGEN_IDENTITY_FAIL** (exit 1)
- all 8 suffix comparisons (room/bicycle × 10K/15K/20K/25K) = MISMATCH
- new 5K checkpoints written but NOT gate-valid (no canonical 5K ever existed)

## SOURCE IDENTITY (pre-flight, all PASS, live-verified on EPIC-05)
- n1snap trainer: 49a79538d5418cdf795b8a8f9919b2e3f9bc5f7550c7f966b030e3ae14d8dbf4
- c0_ext .so:     d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- core gsplat .so:935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- fused_ssim .so: dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb
- env: EPIC-05, 8×A100-SXM4-80GB, torch 2.9.1+cu128, cuda 12.8, driver 570.133.20
- data: images dated 2026-08-20 (canonical run 2026-09-28) → NO data drift

## EXPERIMENT
- room/bicycle, fresh from-scratch regen of frozen P12/C0 seed-42 trajectory,
  argv = canonical verbatim + `--snapshot-iterations 5000,10000,15000,20000,25000`
  (ONLY delta = added 5000; save-only, first fires at iter 5000).
- room@GPU0 pid3124196, bicycle@GPU1 pid3124197. wall ~7.2min / ~12.1min.

## ROOT CAUSE — cross-run non-determinism of the C0 workload (PROVEN 3 ways)
1. CONTROL (verbatim canonical cmd, no 5000, GPU2) ALSO diverges from canonical →
   the added 5000 snapshot is NOT the cause.
2. Three matched-seed runs → three DIFFERENT iter-1000 Gaussian counts:
   canonical 128764 / control 128409 / regen 128182.
3. FULL determinism mode (torch.use_deterministic_algorithms(True) + TF32 off +
   CUBLAS_WORKSPACE_CONFIG=:4096:8 + cuDNN deterministic, all confirmed "enabled")
   → det_a ckpt_3000 = 80516c43… ≠ det_b = 9ef4f035… (same script/seed/scene, GPU3 vs GPU4).
   The non-determinism lives in the CUSTOM CUDA rasterizer backward (non-deterministic
   atomicAdd gradient accumulation), which torch's determinism registry does NOT control.
- Divergence visible at 5K already: regen GS=509791 PSNR=27.40 vs canonical GS=518840 PSNR=26.65.
- torch defaults on this box: matmul.allow_tf32=False, cudnn.allow_tf32=True, deterministic=False.

## INTERPRETATION
- OBSERVED: fresh regeneration never reproduces canonical checkpoint bytes; even
  determinism-forced reruns disagree.
- EVIDENCE: control + triplicated GS counts + determinism-mode self-inconsistency.
- HYPOTHESIS (confirmed): C0 training is cross-run non-deterministic via custom-kernel
  atomicAdd; the "canonical trajectory" is a single non-reproducible sample.
- TEST: the 3-way probe above.
- CONCLUSION: the bit-exact identity gate is UNSATISFIABLE with the frozen binaries.
  The directive's premise ("N1 proves bit-exact canonical-stream reproduction") was a
  misreading: N1 proved camera-RNG-stream determinism + save/restore transparency WITHIN
  a run, NOT cross-run weight reproducibility.

## WHY THIS IS A BLOCKER, NOT A DROP
- Per protocol rule "不允许把 BLOCKED 改写成 DROP": the HomotopyGS mechanism was never
  exercised (no NATIVE/E2D/OPA arms ran). Closing it as a scientific DROP would be wrong.
- This is an infrastructure/reproducibility closure of the H0-R5K *approach*.

## PROGRAM-LEVEL IMPACT
- AB0 (needs canonical 5K) and SG0-Early (needs canonical 5K) hit the SAME wall.
- Tracks needing a NEW checkpoint on the canonical trajectory are all blocked.
- Tracks using ONLY existing frozen checkpoints (H0 15K/25K cells; SG0 10K; AB0 15K) are unaffected.

## ARTIFACTS
- local: h0_run/h0r5k_identity_gate.py, h0_run/h0r5k_regen.sh, CLOSED_DIRECTIONS_20260930.md
- remote: /root/epic05_h0r5k/{h0r5k_identity_gate.py,h0r5k_regen.sh,H0_R5K_IDENTITY.json,
  determinism_probe.py, runs/r5k_c0f_s42_{room,bicycle}, det_room_{a,b}, control/, logs/}
- manifest: regen ckpt sha256 in H0_R5K_IDENTITY.json; determinism probe ckpts hashed above.
