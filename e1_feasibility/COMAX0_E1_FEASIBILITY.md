# COMAX0 / MX — E1 Full-Quality Feasibility Analysis

Date: 2026-10-06
Host: bms-39468022-001 (8x A100-40GB sm_80)
Experiment: E1_FULL_QUALITY_FEASIBILITY
Scope: numerics/cost feasibility only. NO performance or quality PASS is claimed or implied.
Master verdict UNCHANGED: INCONCLUSIVE_BUDGET.

## Question
Given the REAL measured per-step cost of the E1 numerics stack on the frozen room checkpoint,
is the full preregistered E1 quality run (1000-step paired refinement, baseline vs CoMax,
PSNR/SSIM/LPIPS on the frozen test split) completable within the remaining 12h GPU budget?

## OBSERVED
- Budget charged: 35412.6s (9.84h) of 12h; remaining 7787.4s = 129.8 min across 49 counted runs.
- All 8 GPUs were fully idle (0 MiB / 0%) at probe time; no train/train_tf.py co-resident.
- The E1 determinism prerequisite already PASSED under the shared-GPU amendment
  (COMAX0_E1_CLAMP_DETERMINISM): CLAMP production renderer bitwise-deterministic + NaN-free on 5/5 views.
- A bounded per-step timing probe (probe_e1_stepcost.py) on idle GPU 2 measured the real cost of
  the E1 training step for both branches at 1080p (924550 gaussians):
  * setup (GT cache 2460 MB + gsplat JIT compile + model load): ~90s
  * timing loop: 6 baseline steps COMPLETED; 6 comax steps NOT completed in 176s.
  * comax per-step cost: >29.3s (lower bound = 176s / 6, before disk-stall overhead); 0 of 6 done.
  * GPU util stayed 0-33% during comax steps: the checkpointed reference comax renderer
    (comax_render_eta_ckpt) per-Kmax torch.where loop is CPU-bound, not GPU-bound.
- The non-checkpoint comax_render_eta OOMs at 1080p on a CLEAN idle GPU (39.4 GiB used,
  full-graph retention ~30-40 GB). The checkpointed twin is REQUIRED at 1080p regardless of co-residency.
- Root filesystem /dev/sda1 is 100% full (434G/439G, 0 bytes free). The probe process
  intermittently stalled in State D (do_wait/__flush_work) because the kernel cannot flush
  dirty page-cache pages to a full root; GT images are read from root, so page-cache pressure
  from other tenants stalls even reads.

## EVIDENCE
- COMAX0_E1_FEASIBILITY.json (machine-readable; this directory).
- probe_stepcost.log (GT cache log; no JSON emitted because the probe was killed at a conclusive lower bound).
- Process snapshot at kill: python child 362127, elapsed 266s, RSS up to 11.7 GB, GPU2 1635 MiB,
  State cycling Rl/Dl; 6 comax steps未完成.
- Budget accounting: charged 35412.6s, remaining 129.8 min; probe charged NO budget (standalone, not via run_on_mx.py).
- Prior co-residency OOM root cause: comax_eta_ckpt.py:91 torch.where (COMAX0_E1_CKPT_PREP);
  the ckpt twin exists specifically to remove the ~30-40 GB full-graph retention.
- Prior H0 verdict: MARGINAL_HEADROOM (~2.0x ceiling, realistic <2x) per COMAX0_H0.

## INFERENCE
- The E1 comax reference oracle (comax_render_eta_ckpt) costs >29.3s/step at 1080p on room.
- Full E1 requires 1100 comax steps (100 determinism + 1000 training) plus baseline 1000 steps plus eval.
  Lower bound: 1100 x 29.3s = 32230s = 8.95h for comax alone; ~9.6h including baseline+eval.
- Remaining budget is 2.16h. The comax reference oracle alone exceeds budget by ~4.1x;
  the full E1 run exceeds it by ~4.4x. E1 full quality is therefore BUDGET-INFEASIBLE.
- The bottleneck is the unoptimized per-Kmax Python torch.where loop in the reference renderer,
  NOT GPU idleness (GPU util was low because the loop is CPU-bound) and NOT co-residency
  (the probe ran on a fully idle GPU). The full-root disk would further degrade a multi-hour run.
- The determinism prerequisite passing does NOT help: it is a gate, not a substitute for the
  1000-step quality comparison, which cannot complete in budget.

## HYPOTHESIS
- FUTURE_HYPOTHESIS (VECTORIZED_COMAX_REFERENCE): replacing the per-Kmax Python loop with a
  segmented-scan / vectorized equivalent (as the CUDA envelope already implements) would make the
  reference oracle orders of magnitude faster and could bring E1 within budget. This is a METHOD
  CHANGE, explicitly forbidden in this budget-constrained pass, and would require re-proving
  bitwise-exactness against the current depth-order oracle before any quality result is valid.
- The non-ckpt OOM at 1080p (even on a clean GPU) indicates any future 1080p E1 must use the
  checkpointed twin; this is now a hard requirement, not an optional shared-GPU mitigation.

## NOT_MEASURED
- E1 quality deltas (delta_psnr_db / delta_ssim / delta_lpips): NOT_MEASURED (run cannot complete).
- Baseline per-step cost: estimated ~2s/step (gsplat, GPU-bound); not separately isolated because
  the probe was killed during the comax loop. Does not affect the infeasibility conclusion.
- Wall-clock / throughput / bandwidth: NOT_MEASURED and NOT claimed (performance_evidence_valid=false).
- E2 / E3: NOT_RUN (out of scope for this feasibility pass; H0 already flags marginal headroom).

## CONCLUSION
E1 full-quality numerics are BUDGET-INFEASIBLE on MX under the frozen preregistration: the CoMax
reference oracle needs >29.3s/step (~8.95h for 1100 steps) versus 2.16h of remaining budget.
The determinism prerequisite PASS is preserved but does not substitute for the quality comparison.
The master verdict remains INCONCLUSIVE_BUDGET (unchanged). A full-root-disk environmental
blocker and the non-ckpt 1080p OOM are documented as additional findings. No timing/performance
claim is made. A vectorized reference renderer is recorded as a FUTURE_HYPOTHESIS only.
