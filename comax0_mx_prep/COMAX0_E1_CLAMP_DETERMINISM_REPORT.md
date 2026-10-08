# COMAX0 E1 — CLAMP Production Renderer Determinism (E1_NUMERICS_SHARED_GPU_AMENDMENT)

Date: 2026-10-06 (MX host bms-39468022-001, 8x A100-40GB sm_80)
Stage: E1 numerical/correctness evidence ONLY. NO timing/performance claim.
Amendment: E1_NUMERICS_SHARED_GPU_AMENDMENT (shared_gpu_numerics_only=true, performance_evidence_valid=false).

## OBSERVED
Two render-only determinism gates were run on the frozen room scene (seed 17, 1080p test split, n_test=39),
rendering each representative view TWICE under no_grad and comparing the two rendered stacks:

- NON-CLAMP oracle (`comax_render_eta_ckpt`) — prior experiment COMAX0_E1_SHARED_DET2:
  2/5 views deterministic; 3/5 views (globals 96, 240, 304) NON-deterministic-NaN.
  Failure mode: the NaN MASK varies run-to-run; non-NaN pixels are bitwise-exact (max_abs_diff_excl_nan=0.0).
  Root cause (per DET2): inf from non-PSD conics -> float reassociation in the per-k scan -> NaN mask toggles.
  VERDICT = BLOCKED_SHARED_GPU_NUMERICS.

- CLAMP production renderer (`comax_render_eta_ckpt_clamp`, power-clamp power>=0 + TF32-off) — THIS experiment (e1_clamp_det2_20261006):
  5/5 views deterministic. All views: nan_count_run_a = nan_count_run_b = 0, hash_match = true,
  max_abs_diff_excl_nan = 0.0, view_pass = true. Global gate pass = true.
  Includes the exact views that failed under the oracle (96, 240, 304): now fully finite and bitwise-deterministic.
  Peak reserved GPU memory = 648 MiB (peak alloc 590.5 MiB). Run elapsed = 2762.6 s (46.0 min).

Raw evidence: e1_clamp_det2_20261006/e1/{DETERMINISM.json,results.jsonl,RUN_STATUS.json,GUARD.json,console.log}.

## EVIDENCE
- Renderer: `comax_render_eta_ckpt_clamp` (package/COMAX0_MX_DSH/scripts/comax_gs/comax_eta_ckpt_clamp.py),
  the memory-efficient (per-chunk gradient-checkpointed) production twin used for E1 numerics.
- Numerical guards: COMAX0_POWER_CLAMP=1 (clamps per-Gaussian power to >=0 so non-PSD conics cannot make exp() overflow),
  TF32 disabled (matmul + cudnn).
- Selection: shared launcher run_e1_shared_ckpt.py, lowest time-avg utilization among GPUs 1-5 whose sole
  co-resident compute is the user's own train/*.py. Selected GPU2 (avg util 31.6%, 16939 MiB free, feasible>=6000),
  co-resident train_tf.py PID 3200555 (owner-verified, untouched). The co-resident train job ended on its own
  mid-run (after ~10 min); GPU2 then ran free. Determinism is the worker's self-consistency, so this does not
  invalidate the comparison; before/after GPU state recorded in RUN_STATUS.json.
- Comparison logic identical to e1_determinism.determinism_check / DET2 (sha256 of the rendered stack, tolerance = prereg e2_rgb_abs_max = 2e-5).
- Worker SHA256: e1_clamp_det.py = a17edfd10ab093d7fe35bbc35088de027b2c13b6810fe2b53df6dc92105f90ee.
- Launcher SHA256: run_e1_shared_ckpt.py (with COMAX0_E1_ENTRY / COMAX0_E1_FREE_MEM_MIN_MIB env overrides; defaults preserve prior behavior).

## INFERENCE
The power-clamp (power >= 0) removes the inf->NaN source entirely, so the per-k scan no longer produces a
run-to-run-varying NaN mask. The clamp renderer is therefore bitwise-deterministic (hash_match=true, max_abs_diff=0.0)
on every tested view, including the high-overdraw views that were non-deterministic under the oracle. This indicates
the nondeterminism is a numerical-overflow artifact of the non-clamp oracle, NOT an inherent property of the COMAX
envelope scan, and NOT a shared-GPU artifact.

## HYPOTHESIS
- The clamp renderer is a faithful, deterministic production path for E1 numerics (forward render). This is being
  cross-checked against the non-clamp oracle on co-finite pixels (e1_clamp_vs_oracle, CLAMP_VS_ORACLE.json) to confirm
  it does not DIVERGE from the trusted oracle where the oracle is finite (required for the E2 correctness gate).
- The E1 QUALITY pilot (paired refinement, PSNR/SSIM/LPIPS deltas) remains BUDGET-BLOCKED: the reference/Python scan
  renderer is ~minutes per render, so a multi-hundred-step paired refinement does not fit the launch budget. This is
  a compute blocker, unchanged by the determinism result.

## NOT_MEASURED
- Training-inclusive determinism (render-only gate only).
- Full 39-view determinism sweep (5-view representative subset: clean 0,40; former-NaN 96,240,304).
- Any wall-clock / kernel / throughput / bandwidth claim (forbidden by the amendment; performance_evidence_valid=false).
- E1 quality deltas (budget-blocked). E2 timing (no clean guarded GPU + reducer gates unmet). E3 (not reached).

## CONCLUSION
The E1 determinism PREREQUISITE is SATISFIED for the CLAMP production renderer: render-only, bitwise-deterministic,
NaN-free, within tolerance, on a shared GPU under the amendment (peak 648 MiB). The non-clamp oracle remains
NaN-nondeterministic (DET2). The master protocol verdict is UNCHANGED: e1 = INCONCLUSIVE_BUDGET (the quality pilot is
budget-blocked, so the reducer never reaches e2); master = INCONCLUSIVE_BUDGET. No speed claim is made or implied.
