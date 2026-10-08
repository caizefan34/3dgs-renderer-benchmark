# Real-state finite-edit screening protocol

Specified before the first completed paired-branch result, 2026-09-26.

Purpose: test whether conditioning finite split scores on feasible old-parameter
image responses improves prediction of short-horizon reconstruction benefit.
This is an expensive offline opportunity probe, not a deployed training method.

## Fixed development configuration

- P11 `room/state_2000.pt`, 260,530 Gaussians; saved Adam history explicitly loaded.
- Pinned gmain rasterizer from P11; original clamped RGB and 0.8 L1 + 0.2 fused
  D-SSIM (`same` padding); original full 1920x1080 resolution.
- First three unique future camera IDs form scoring set A, the next three B.
  B is excluded from continuation updates. B may have been seen during the
  original checkpoint training; this is NOT an unseen-view generalization test.
- Candidate parents satisfy native large-scale split eligibility, visibility
  in at least two A views and projection inside the first A view.
- Partition the first A view into 4x4 cells; take the 32 highest absgrad eligible
  parents per qualifying cell, then the eight groups with highest mean absgrad.
  Equal group size; no candidate uses B or continuation outcome to be generated.
- Each edit invokes native two-child split, including parent removal, random
  offsets and inherited opacity/appearance. Survivor Adam moments must be exactly
  preserved; newborn moments zero, with the native group step value retained.
- One no-edit arm, its identical replay, and eight edit arms each take the same
  64 Adam updates from the same checkpoint. Evaluation at 0, 8, 32 and 64 updates.
  No additional topology operation, opacity reset or SH upgrade in this window.

## Scores compared before seeing outcome

1. Mean A absgrad, normalized by visibility count.
2. First-view rectangular high-error footprint count (L1 > 0.02). Explicitly a
   simple proxy, NOT a reproduction of FastGS VCD.
3. Unconditional finite-edit sampled MSE gain.
4. Conditional MSE gain after projecting out three no-edit Adam image secants
   (updates 1,4,8). These are observed finite responses, not a true Jacobian.
5. Conditional MSE gain using ten real local central-difference directions:
   shared xyz (3), log-scale (3), opacity logit (1), DC (3). Perturbations are
   1% of median activated scale for xyz and 0.01 for the other parameters.
6. The combined local and Adam-secant basis. Columns are normalized before SVD;
   relative singular cutoff is 1e-6. No full old-parameter tangent-space claim.
7. Actual immediate A loss decrease under the original loss, as an additional
   control for changing the proxy objective to MSE.

Scores 3–6 use all RGB channels at every eighth pixel along each image axis in
A, identical across arms. Scores 4–6 use the signed finite-edit formula, not
an optimizable scalar interpolation between before/after images. Outcome labels
use the full-resolution original loss in B after identical continuation.

## Interpretation and continue gate

Report Spearman ordering and the average B loss improvement of each score's
top three groups. Report every group's signed gains at every horizon, and the
absolute difference between the repeated no-edit runs as a diagnostic noise
scale. One repeat does not estimate the full variance distribution.

For a positive opportunity signal at 64 updates, a conditional score must choose
better groups than the raw score and cheap controls by more than the observed
repeat difference. Eight development candidates cannot establish significance,
novelty, FastGS superiority, complete-training quality or speedup. Passing only
allows independent checkpoints/scenes and stronger official method controls.

If a candidate starts harmful and later helps, record immediate-score false
rejection. If rankings remain the same, do not credit conditioning merely because
the chosen splits eventually improve. If all candidates are harmful, investigate
the horizon/candidate initialization before declaring capacity control impossible.

Count setup, image loading, finite-difference renders, baseline lookahead,
optimizer restoration, continuation and evaluation in recorded wall time.
No offline-oracle cost is excluded to manufacture a speedup.

## Execution safeguards and known setup findings

- `GaussianModel.restore()` restores parameters but omits saved optimizer state;
  the probe explicitly calls `optimizer.load_state_dict()` and checks it.
- Driver PIDs differ from container PIDs on this host. A minimal CUDA probe
  observed namespace PID 1611901 and driver PID 334315. The experiment attributes
  the sole driver process across empty-GPU -> own CUDA initialization, then
  rejects any process-set change before each arm and at completion.
- Earlier attempts stopped before paired results: GPU 5 was already occupied;
  the first GPU 7 run hit the PID-namespace check. Both handles were confirmed
  terminal before restarting. Frozen P11/P12 code was not modified.

Reproduction on EPIC-05 uses the existing pinned Python environment:

```text
/root/miniforge3/envs/gsplat/bin/python -u real_topology_probe.py --gpu 7 --scene room --iteration 2000 --groups 8 --group-size 32 --steps 64 --out room_2000_topology.json
```

## Development extensions after first results

The same source and parameters were then applied to garden@2000 (8 groups of
32, 64 updates). Room@2000 was extended to 512 updates after observing horizon
sign reversals at 64 updates. Neither extension is called an independent
confirmation campaign. The 512-update run keeps the same candidates and score
construction, freezes further topology, and evaluates at 0,8,32,512; no
retuning against the observed candidate labels is introduced.
