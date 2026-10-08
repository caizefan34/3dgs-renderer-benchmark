# DSH task: execute P13-X Discovery exactly as preregistered

Read `P13X_PREREG.md` first. Treat it as immutable preregistration.

## Inputs to reuse, not rewrite
- `/root/epic05_p12/publication_trainer_p12.py`
- `/root/epic05_frozen/higs_c0_final30k_worktree`
- frozen C0/core/fused-SSIM binaries and hashes recorded by P12
- P12 recipe, datasets, contamination policy, scheduler conventions

## First action
Create `/root/epic05_p13x/` and copy the P12 trainer to `publication_trainer_p13x.py`. Keep the original untouched. Generate `P13X_SOURCE_IDENTITY.json` before modifying behavior.

## Implementation constraint
Instrumentation must be side-channel only. It must not alter camera sampling, optimizer values, densification decisions, RNG state used by training, renderer configuration, loss semantics, or timed measurements.
Use an independent diagnostic RNG or deterministic anchor-camera list.

## Required engineering order
1. Implement X0 analyzer from P12 artifacts and verify the 15K-30K constant-N claim.
2. Implement X1 using true training cameras and actual renderer visibility outputs.
3. Expose exact renderer intersection/tile-order intermediates for X2. If gsplat/C0 already returns suitable metadata, consume it. Otherwise add a debug-only CUDA/API path. Never substitute Python footprint approximations.
4. Implement X3 attribute-delta + dirty-work accounting and replay correctness check.
5. Implement X4 by retaining image gradient from the exact fused loss and computing fixed 16x16 DCT statistics outside timed regions.
6. Implement X5 as fixed-state dual backward/shadow-step probes. Clone optimizer state; never contaminate the training state.
7. Only then measure X6 oracle.

## Preflight before long runs
Run 220 iterations on room for baseline P12 trainer vs instrumented P13-X with diagnostics disabled. Require loss checkpoints 1/100/200 to match within P12-style FP tolerances and renderer/binary hashes to match. Then run diagnostics enabled for 300 iterations and verify training RNG/camera sequence is unchanged.

## Execution scale
Start with one scene (`room`) and only 15K->16K diagnostic window if checkpoint/resume exists; otherwise one full 16K run. Evaluate gates before running bicycle/garden. Do not launch three 30K runs blindly.

## Verdict discipline
Every report section must use OBSERVED / EVIDENCE / HYPOTHESIS / TEST / CONCLUSION.
Allowed final verdicts: `DELTA_GO`, `ADJOINT_GO`, `BOTH_GO`, `INVESTIGATE`, `DROP`.
No novelty language and no >3x claim unless X6 conservative oracle passes >=4x and later implementation validates it.
