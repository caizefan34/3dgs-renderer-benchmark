# HomotopyGS H0 — Native-Objective Continuation Oracle (FROZEN)

## 0. Scientific question

Can a smoother intermediate renderer produce parameter updates that descend the *native* full-resolution 3DGS objective faster than ordinary native Adam?

The experiment deliberately distinguishes:

TRAIN renderer:
    may be smoothed/tempered during a homotopy step.

EVAL renderer:
    ALWAYS the exact native renderer.

A positive result therefore requires better progress on the exact endpoint objective, not merely lower loss under the altered renderer.

## 1. Source / baseline identity

Use frozen P12/C0 semantics:
- loss: `0.8*L1 + 0.2*(1-SSIM)`
- full-resolution fused SSIM
- seed 42
- tile16
- black background
- densification schedule in the parent trajectory: 500..15000

Known frozen identities:
- C0 extension SHA256:
  `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11`
- core `gsplat_cuda.so` SHA256:
  `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e`
- room@25K checkpoint SHA256:
  `b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d`

Recover COMPLETE hashes for all additional checkpoints/camera inventories from existing frozen manifests.
Fail closed on mismatch.

## 2. Checkpoints

Required cells:

- room@5K
- room@15K
- room@25K
- bicycle@5K
- bicycle@15K
- bicycle@25K

The purpose is to cover:
- active densification regime;
- densification boundary / mature geometry;
- post-densification refinement.

If an exact checkpoint does not exist in the frozen trajectory:
    H0_CHECKPOINT_BLOCKER
    STOP.
Do not substitute a nearby iteration.

Inside H0 short rollouts:
- topology is frozen;
- no densify / prune / clone / split;
- optimizer state from the checkpoint must be preserved exactly.

A 15K rollout is a local counterfactual diagnostic if the original trainer would execute a topology event at that exact step; record this explicitly.

## 3. Camera folds / rollouts

Use exactly 8 frozen cameras per cell, preferably the same formal MOONSHOT camera set.

Ordered list:
    c0,c1,c2,c3,c4,c5,c6,c7

Fold A:
- update sequence pool c0..c3
- held-out eval c4..c7

Fold B:
- update sequence pool c4..c7
- held-out eval c0..c3

Within each fold use four fixed cyclic offsets.

For K=4:
- one four-camera cycle.

For K=8:
- same four-camera cycle repeated twice.

Thus each cell/family/K has 8 paired rollouts.

All branches start from byte-identical model + optimizer state.

## 4. Native baseline arm

`NATIVE(K)`:
- run ordinary C0 Adam for K topology-frozen steps;
- use the fixed update-camera sequence;
- after EVERY optimizer step, evaluate all 8 cameras with the exact native renderer.

Evaluation is forward-only and is excluded from training-step timing.

Record:
- native design loss trajectory;
- native held-out loss trajectory;
- PSNR/SSIM trajectory;
- training-step CUDA/wall timing;
- parameter family update norms.

## 5. Family E2D — screen-space filter continuation

Let the exact native rasterizer use its frozen native 2D filter variance `eps2d_native`.
First verify the exact numeric native value from source/runtime metadata.

For homotopy step j:

    eps2d_j = multiplier_j * eps2d_native

Frozen schedules:

K=4:
    [2.00, 1.60, 1.30, 1.00]

K=8:
    [2.00, 1.85, 1.70, 1.55, 1.40, 1.25, 1.10, 1.00]

Everything else remains native.

Important:
- this is full-resolution rendering;
- enlarged filter/support is allowed because this arm intentionally defines a smoother intermediate renderer;
- step K ends at EXACT native `eps2d`;
- `multiplier=1.0` must pass native parity before any scientific run.

### E2D parity gate

On all required cells and at least one frozen camera/cell:
- multiplier=1 output/loss must match the native branch within the normal run-to-run floating envelope;
- gradients must pass the project's existing correctness tolerance.

If not:
    H0_E2D_IMPLEMENTATION_FAIL
and E2D is removed from scientific consideration.
Do not relax tolerances.

## 6. Family OPA — opacity/transmission continuation

Let native per-Gaussian opacity after the standard sigmoid be:

    o in (0,1)

Define training-only effective opacity:

    o_gamma = 1 - (1 - clamp(o, 0, 1-eps)) ** gamma

with:
    eps = 1e-7

For gamma < 1:
- effective opacity is reduced;
- deeper contributions receive more signal;
- gamma=1 is exactly the native opacity mapping up to the explicit clamp domain.

Frozen schedules:

K=4:
    [0.60, 0.75, 0.90, 1.00]

K=8:
    [0.60, 0.68, 0.76, 0.84, 0.90, 0.94, 0.97, 1.00]

The transformation MUST remain differentiable back to the original native opacity parameter.

Do not modify:
- per-pixel Gaussian exponent;
- support threshold;
- alpha clamp inside the rasterizer;
- loss;
- learning rate;
- Adam betas/eps.

### OPA parity gate

At gamma=1:
- RGB/loss parity with native;
- native opacity gradient parity;
- all other gradient groups parity.

If not:
    H0_OPA_IMPLEMENTATION_FAIL
and OPA is removed from scientific consideration.

## 7. Native endpoint evaluation

After every training update in every arm:
- evaluate the state using EXACT native C0 renderer:
    E2D multiplier = 1
    OPA gamma = 1
- evaluate all 8 frozen cameras.

No optimizer update during evaluation.

Primary metrics use this native evaluation ONLY.

## 8. Metrics

For rollout r and K define held-out native progress:

    D_native = L0_hold - L_nativeK_hold
    D_homo   = L0_hold - L_homoK_hold

When D_native > 0:

    P_hold = D_homo / D_native

Likewise:
    P_design

Do not clamp ratios.
Negative and >1 values are valid.

### Progress-per-training-time efficiency

Let:
    T_native = sum training-step time of native K-step branch
    T_homo   = sum training-step time of homotopy K-step branch

Native evaluation overhead is excluded from both.

Define:

    E_hold = P_hold * (T_native / T_homo)

This is a local progress-per-second ratio, not an E2E speedup claim.

### Crossing metric

Let:
    target = native held-out loss after K=8 steps.

Using native endpoint evaluations after each homotopy step, record:

    k_cross_8

= earliest j <= 8 at which homotopy held-out loss <= target.

If never reached:
    null.

This is descriptive in H0, not a direct verdict threshold.

### Quality guards

Record after K:
- held-out PSNR delta: homotopy - native
- held-out SSIM delta
- design PSNR/SSIM
- NaN/Inf
- Gaussian parameter validity.

## 9. Timing protocol

For scientific step timing:
- warm the altered renderer separately before each arm;
- CUDA synchronize;
- exclude all 8-camera evaluation work;
- report median/p10/p90 of actual training steps;
- record intersections / accepted-hit counts if exposed, because E2D may increase raster workload.

H0 does not require publication-grade microbenchmarking, but timing must be comparable enough to reject a "faster convergence" mechanism whose individual steps become much slower.

## 10. Mechanical verdict

Analyzer considers candidate family in:
    {E2D, OPA}

A SINGLE family must satisfy a gate across ALL SIX cells.
No per-cell family selection.
No joint schedule rescue.

Rollout ratios with D_native <= 0 are marked `BASE_NON_DESCENT` and excluded from ratio aggregates, but their counts are reported.

If >2/8 K=8 rollouts are BASE_NON_DESCENT in any cell:
    that cell/family fails.

### H0_STRONG_PROMOTE

Exists one fixed family such that EVERY cell satisfies:

K=4:
- median P_hold >= 1.20
- p10 P_hold >= 0.85
- median E_hold >= 1.15

K=8:
- median P_hold >= 1.30
- p10 P_hold >= 1.00
- median E_hold >= 1.20
- median P_design >= 1.20
- median held-out PSNR delta >= -0.03 dB
- worst rollout held-out PSNR delta >= -0.10 dB
- BASE_NON_DESCENT <= 2/8
- NaN/Inf count = 0

### H0_PROMOTE

Exists one fixed family such that EVERY cell satisfies:

K=4:
- median P_hold >= 1.08
- p10 P_hold >= 0.70
- median E_hold >= 1.03

K=8:
- median P_hold >= 1.15
- p10 P_hold >= 0.80
- median E_hold >= 1.08
- median P_design >= 1.10
- median held-out PSNR delta >= -0.05 dB
- worst rollout held-out PSNR delta >= -0.15 dB
- BASE_NON_DESCENT <= 2/8
- NaN/Inf count = 0

### H0_DROP

Otherwise.

The strong gate is intentionally demanding because short local continuation that only improves native progress by a few percent is not enough to justify a paper-scale schedule.

## 11. Forbidden rescue

After results are seen, do NOT:
- change schedules;
- add a joint E2D+OPA arm;
- change camera sets;
- replace 5K/15K/25K;
- alter learning rates;
- add coarse-resolution loss;
- add loss curriculum;
- change densification;
- select family per scene;
- reinterpret altered-renderer loss as native progress.

A failed family stays failed.

## 12. Required artifacts

- H0_SOURCE_IDENTITY.json
- H0_ENVIRONMENT.json
- H0_CHECKPOINT_INVENTORY.json
- H0_CAMERA_FOLDS.json
- H0_PARITY.json
- H0_NATIVE_TRAJECTORIES.json
- H0_E2D_TRAJECTORIES.json
- H0_OPA_TRAJECTORIES.json
- H0_TIMING.json
- H0_SUMMARY.json
- H0_VERDICT.json
- H0_REPORT.md
- raw logs
- immutable SHA256 manifest

## 13. Interpretation

PROMOTE proves only:
- a preregistered renderer continuation family improves short-horizon progress on the native objective across the required states;
- the signal is large enough to justify H1 schedule design/full-trajectory testing.

It does NOT prove:
- full training is faster;
- final quality is preserved;
- 2x/3x E2E acceleration;
- novelty;
- superiority over Faster-GS/FastGS/DashGaussian/3DGS-LM.

Those require later stages.
