# SplatMG SG0 — Future-Displacement Coarse-Space Oracle (FROZEN)

## 0. Scientific question

Does standard 3DGS optimization contain a low-dimensional, spatially coherent collective correction that can shortcut several expensive full-resolution Adam iterations?

SG0 deliberately gives the coarse method unfair future information:
it observes the actual displacement produced by K ordinary Adam steps and asks how much of that displacement/progress can be represented by a fixed multilevel coarse basis.

Therefore SG0 is an ORACLE UPPER BOUND, not an algorithm.

Failure is decisive for this hierarchy/basis family.
Success only justifies designing a deployable coarse solver in SG1.

## 1. Frozen objective and source

Use the frozen C0 training semantics and ORIGINAL full-resolution loss:

    0.8 * L1 + 0.2 * (1 - SSIM)

No image downsampling.
No low-resolution loss.
No surrogate supervision.
No densification inside the SG0 rollout.
No pruning/clone/split.
No renderer modification.

Primary cells:
- room @ 20K
- room @ 25K
- bicycle @ 20K
- bicycle @ 25K

Use the same frozen checkpoints and 8 frozen cameras/cell used by the formal MOONSHOT/A0 campaign when available.

Known identities include:
- C0 extension:
  d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- core gsplat_cuda.so:
  935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- room@25K checkpoint:
  b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d

Recover all other COMPLETE expected hashes from the formal manifests.
Fail closed on identity mismatch.

## 2. Camera folds and rollouts

For each cell, take the preregistered ordered 8-camera list:

    c0,c1,c2,c3,c4,c5,c6,c7

Two fixed folds:

Fold A:
- design/update cameras = c0,c1,c2,c3
- held-out evaluation cameras = c4,c5,c6,c7

Fold B:
- design/update cameras = c4,c5,c6,c7
- held-out evaluation cameras = c0,c1,c2,c3

Within each fold run four cyclic offsets.

For design cameras [a,b,c,d], offsets are:
- [a,b,c,d]
- [b,c,d,a]
- [c,d,a,b]
- [d,a,b,c]

For K=4 use one cycle.
For K=8 repeat the same four-camera cycle twice.

Thus each cell has:
- 2 folds
- 4 offsets
= 8 independent rollouts for K=4 and 8 rollouts for K=8.

All rollouts start from an identical deep copy of checkpoint + optimizer state.

## 3. Baseline trajectory

For every rollout:

1. Save theta_0 and optimizer_0.
2. Evaluate original full-resolution loss/PSNR/SSIM on all 8 cameras.
3. Run exact ordinary native Adam for K steps using the frozen design sequence.
4. Save theta_K.
5. Evaluate the same 8 cameras.

No densification.

Record:
- design mean loss before/after
- held-out mean loss before/after
- PSNR/SSIM before/after
- exact parameter displacement theta_K - theta_0

If held-out loss does not improve after K native Adam steps:
    mark `BASE_NON_DESCENT`
but retain the rollout for descriptive statistics.
It is excluded from progress-ratio denominators.

If >2/8 rollouts in ANY cell are BASE_NON_DESCENT for K=8:
    `SG0_BASELINE_UNSTABLE`
    STOP.

## 4. Deterministic multilevel hierarchy

Hierarchy is built ONLY from theta_0 3D Gaussian means.
No gradient, loss, camera residual, future displacement, or quality signal may affect clustering.

Build a deterministic balanced kd-tree:
- split node along the axis with largest coordinate range;
- split at the stable median;
- tie break by Gaussian ID;
- recurse until leaf size <=32.

Two frozen levels:

L32:
- leaf nodes, target size <=32.

L128:
- ancestor two binary levels above L32 where available;
- boundary nodes with fewer descendants remain as-is.

Record actual cluster-size distributions.

Visibility may be measured as a diagnostic but MUST NOT change cluster membership in SG0.

## 5. Coarse coordinate systems

Operate in these native/tangent coordinates:

- xyz: Euclidean displacement, 3/Gaussian
- log-scales: native unconstrained log-scale coordinates, 3/Gaussian
- rotations: SO(3) tangent displacement from q0 to qK, 3/Gaussian
- opacity: native unconstrained opacity/logit coordinate, 1/Gaussian
- SH DC only: 3/Gaussian

Higher-order SH coefficients are intentionally left to the fine level and are NOT represented by the SG0 coarse basis.

For each cluster use TWO preregistered basis families.

### SIM13

Per cluster:
- shared xyz translation: 3
- shared rotation tangent: 3
- shared log-scale delta: 3
- shared opacity delta: 1
- shared SH-DC delta: 3

Total: 13 coarse DoF/cluster.

### AFF22

Per cluster:
- xyz affine field around cluster centroid:
    delta_x_i = t + A (x_i - center)
  where t has 3 DoF and A has 9 DoF
- shared rotation tangent: 3
- shared log-scale delta: 3
- shared opacity delta: 1
- shared SH-DC delta: 3

Total: 22 coarse DoF/cluster.

No additional basis family may be introduced after results are seen.

## 6. Future-displacement oracle projection

For each rollout and K:

1. Convert theta_K - theta_0 into the coordinate representation in Section 5.
2. For each {level,basis} in:
       {L32,L128} x {SIM13,AFF22}
   solve the independent least-squares projection of the FUTURE displacement
   onto the coarse basis.

This creates:
    delta_oracle = Pi_H (theta_K - theta_0)

where H is the fixed coarse space.

This uses future information by design and is an upper-bound opportunity test.

Reconstruct a valid model:
- xyz: add projected delta
- log-scales: add projected delta
- rotations: exp-map projected tangent and compose with q0, then normalize
- opacity native coordinate: add projected delta
- SH DC: add projected delta
- higher-order SH: unchanged from theta_0

No optimizer state is advanced for the oracle state.

## 7. Oracle evaluation

Evaluate the oracle-jump state on all 8 cameras with the ORIGINAL full-resolution loss.

Primary held-out progress capture:

    P_hold =
      (L0_hold - L_oracle_hold)
      /
      (L0_hold - LK_hold)

only when denominator > 0.

Primary design progress capture:

    P_design =
      (L0_design - L_oracle_design)
      /
      (L0_design - LK_design)

Also record:
- PSNR/SSIM deltas
- per-parameter-family displacement-energy capture
- total displacement-energy capture
- coarse DoF count
- represented fine DoF count
- coarse_dof_ratio

Clamp NO ratios. Values >1 and negative values are valid observations.

## 8. Controls

### Identity control
Project onto an identity/per-Gaussian basis for the SAME represented coordinate groups.
It must reconstruct represented future displacement to numerical tolerance.
If not:
    SG0_PROJECTION_BUG
    STOP.

### Random-cluster diagnostic
Create one deterministic random partition matched to the L32 cluster-size multiset, seed=20260930.
Evaluate SIM13 only.
This is descriptive and does not enter the verdict.

Purpose: determine whether spatial hierarchy matters beyond generic parameter averaging.

## 9. Frozen verdict

The analyzer considers only:
- K=4 and K=8
- levels L32/L128
- bases SIM13/AFF22

A SINGLE fixed `(level,basis)` must satisfy the gates in ALL FOUR cells.
No per-cell selection.

For each cell, aggregate across the 8 rollouts, excluding BASE_NON_DESCENT denominators.

### SG0_STRONG_PROMOTE

Exists one fixed `(level,basis)` such that every cell has:

K=4:
- median P_hold >= 0.85
- p10 P_hold >= 0.60

K=8:
- median P_hold >= 0.80
- p10 P_hold >= 0.55

and:
- median P_design(K=8) >= 0.80
- median coarse_dof_ratio <= 0.05
- oracle held-out loss improves over theta_0 in >=7/8 K=8 rollouts
- BASE_NON_DESCENT <=2/8

### SG0_PROMOTE

Exists one fixed `(level,basis)` such that every cell has:

K=4:
- median P_hold >= 0.75
- p10 P_hold >= 0.40

K=8:
- median P_hold >= 0.65
- p10 P_hold >= 0.30

and:
- median P_design(K=8) >= 0.70
- median coarse_dof_ratio <= 0.10
- oracle held-out loss improves over theta_0 in >=6/8 K=8 rollouts
- BASE_NON_DESCENT <=2/8

### SG0_DROP

Otherwise.

This is intentionally demanding because the oracle has future information.
A weak oracle result is not enough to justify a real solver.

## 10. What success means

PROMOTE means:

The actual multi-step optimization trajectory contains a low-dimensional spatially coherent collective component large enough to justify designing a deployable coarse correction.

It does NOT prove:
- a real coarse solver can discover the oracle displacement;
- fewer wall-clock iterations are achieved;
- a V-cycle is stable during densification;
- novelty is established;
- 2x/3x training acceleration is realized.

## 11. What failure means

DROP means this specific hierarchy + low-order basis family does not contain enough future optimization progress, even with oracle future information.

Do not rescue by:
- downsampling the image/loss;
- gradient-dependent clustering;
- using future displacement to build clusters;
- adding dozens of new basis modes;
- per-scene basis choice;
- relaxing thresholds.

## 12. Required outputs

- SG0_SOURCE_IDENTITY.json
- SG0_ENVIRONMENT.json
- SG0_CAMERA_FOLDS.json
- SG0_HIERARCHY.json
- SG0_BASELINE_ROLLOUTS.json
- SG0_ORACLE_PROJECTIONS.json
- SG0_CONTROLS.json
- SG0_SUMMARY.json
- SG0_VERDICT.json
- SG0_REPORT.md
- raw logs
- SHA256 manifest
