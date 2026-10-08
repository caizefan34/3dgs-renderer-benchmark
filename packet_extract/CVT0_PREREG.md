# CVT-GS CVT0 — Cross-View Update-Transfer Oracle (FROZEN)

## 0. Hypothesis

For the same model state, a one-view Adam update from camera i may have very different effects on other cameras j.

Define:

    M_ij = L_j(theta_0) - L_j(theta_i^+)

where theta_i^+ is the exact native one-step Adam state produced using ONLY source camera i.

Positive M_ij means the update from i improves view j.

CVT0 tests whether selecting i by peer-view transfer offers a large, consistent advantage over:
- uniform-random source selection;
- highest-current-loss source selection.

No deployable predictor is proposed in CVT0.

## 1. Frozen system

Use frozen C0 source/renderer/trainer semantics:
- original full-resolution loss:
    0.8*L1 + 0.2*(1-SSIM)
- exact checkpoint optimizer state
- native Adam
- tile16
- black background
- topology frozen during the one-step probe

Known identities:
- C0 extension SHA256:
  d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- core gsplat_cuda.so:
  935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e

Recover COMPLETE checkpoint/source/dataset hashes from authoritative manifests.
Fail closed on mismatch.

## 2. Required cells

Use existing exact frozen checkpoints:

- room@10K
- room@15K
- room@20K
- bicycle@10K
- bicycle@15K
- bicycle@20K

Do not substitute another iteration.

No densification/prune/clone/split inside a probe.

## 3. Correct train / true-holdout semantics

The corrected SB0 true-holdout protocol is mandatory.

For each scene:
- TRUE HOLDOUT = canonical cameras whose original dataset index satisfies `i % 8 == 0`;
- TRAIN POOL = all remaining cameras.

No true-holdout camera may generate an update or participate in oracle source selection.

Assert:
    train_pool ∩ true_holdout = empty

and save exact camera IDs.

## 4. Deterministic source panels

From the ordered TRAIN POOL of N cameras:

Select 24 stratified positions:

    q_k = floor((k + 0.5) * N / 24),  k=0..23

using the camera at train_pool[q_k].
If duplicate positions occur, advance deterministically to the next unused camera.

Assign:
- Panel 0 = selected k with k mod 3 == 0
- Panel 1 = selected k with k mod 3 == 1
- Panel 2 = selected k with k mod 3 == 2

Each panel must contain exactly 8 unique train cameras.

TRUE-HOLDOUT evaluation panel:
select 8 stratified cameras from the ordered true-holdout pool using the analogous midpoint quantiles.

Freeze and save all IDs before scientific execution.

## 5. Base evaluation

For each cell and each panel:

From the identical checkpoint + optimizer state:
- evaluate original native loss on all 8 panel cameras;
- evaluate original native loss / PSNR / SSIM on the 8 true-holdout cameras.

No gradients yet.

## 6. Source-update matrix

For each source camera i in the 8-camera panel:

1. Deep-clone model + optimizer state from theta_0.
2. Run ONE exact native forward/backward/Adam update using camera i.
3. Topology remains frozen.
4. Evaluate the resulting theta_i^+ with forward only on:
   - all 8 panel cameras;
   - all 8 true-holdout cameras.
5. Save all losses/PSNR/SSIM.

Construct:

    M_ij = L_j(theta_0) - L_j(theta_i^+)

Primary peer-transfer score:

    T_i = mean_{j != i, j in panel} M_ij

True-heldout transfer:

    H_i = mean_{h in true_holdout_panel}
          [L_h(theta_0) - L_h(theta_i^+)]

Self-improvement:

    S_i = L_i(theta_0) - L_i(theta_i^+)

## 7. Three source selectors

### RANDOM expectation
No random draw is needed.
The expected random transfer is exactly:

    T_random = mean_i T_i
    H_random = mean_i H_i

### HARD-LOSS
Select:

    i_hard = argmax_i L_i(theta_0)

tie break by camera ID.

Record:
    T_hard = T_{i_hard}
    H_hard = H_{i_hard}

This represents the simplest error-driven source rule and is especially important because TurboGS already uses a high-error anchor.

### TRANSFER ORACLE
Select using TRAIN-panel peers only:

    i_oracle = argmax_i T_i

tie break by camera ID.

True holdout is NEVER used for selection.

Record:
    T_oracle
    H_oracle
    S_oracle

## 8. Primary opportunity metrics

Let:

    A = mean_i |T_i| + 1e-12

Define normalized peer advantages:

    G_random = (T_oracle - T_random) / A
    G_hard   = (T_oracle - T_hard)   / A

For true holdout:

    B = mean_i |H_i| + 1e-12
    G_true = (H_oracle - H_random) / B

Oracle peer-positive fraction:

    F_pos = fraction_{j != i_oracle}(M_i_oracle,j > 0)

Rank-transfer generalization:

    rho_TH = Spearman({T_i}, {H_i})

Also report ordinary ratios only when denominators are positive:

    R_random = T_oracle / T_random
    R_hard   = T_oracle / T_hard
    R_true   = H_oracle / H_random

These ratios are descriptive and do NOT replace normalized primary metrics.

## 9. Mechanical verdict

Aggregate the 3 panels inside each cell.

A SINGLE fixed protocol already exists; no selector family is chosen post hoc.

### CVT0_STRONG_PROMOTE

EVERY one of 6 cells satisfies:

- median(T_oracle) > 0
- median G_random >= 0.60
- median G_hard >= 0.35
- median G_true >= 0.40
- H_oracle > 0 in all 3 panels
- median F_pos >= 0.75
- median rho_TH >= 0.40
- all 3 panels complete with no NaN/Inf

### CVT0_PROMOTE

EVERY cell satisfies:

- median(T_oracle) > 0
- median G_random >= 0.35
- median G_hard >= 0.20
- median G_true >= 0.20
- H_oracle > 0 in >=2/3 panels
- median F_pos >= 0.65
- median rho_TH >= 0.20
- all panels complete

### CVT0_DROP

Otherwise.

The `G_hard` gate is mandatory.
If transfer-aware selection cannot materially beat the highest-loss camera, the distinct research mechanism is not justified because error-aware view sampling already exists.

## 10. Controls / diagnostics

Report, but do not use for selection:
- pairwise camera pose distance vs M_ij;
- view-frustum / visible-Gaussian overlap vs M_ij if cheaply available;
- source current loss vs T_i;
- source self-improvement S_i vs T_i.

These determine whether transfer is merely geometry proximity or hard-view selection.

## 11. Timing

Measure:
- one native training step;
- one forward-only evaluation;
- full CVT0 matrix construction time.

CVT0 makes NO E2E speedup claim.
A deployable CVT1 must predict source utility without trying all source updates.

## 12. Forbidden rescue

After results:
- no different checkpoint;
- no different camera panels;
- no selecting with true holdout;
- no multi-view batch;
- no learned selector;
- no changing loss;
- no threshold relaxation;
- no redefining hard-loss baseline.

DROP means the transfer-scheduling mechanism is closed.

## 13. Required outputs

- CVT0_SOURCE_IDENTITY.json
- CVT0_ENVIRONMENT.json
- CVT0_CAMERA_SEMANTICS.json
- CVT0_PANELS.json
- CVT0_BASE_LOSSES.json
- CVT0_TRANSFER_MATRICES.json
- CVT0_SELECTORS.json
- CVT0_TRUE_HOLDOUT.json
- CVT0_DIAGNOSTICS.json
- CVT0_TIMING.json
- CVT0_SUMMARY.json
- CVT0_VERDICT.json
- CVT0_REPORT.md
- raw logs
- SHA256 manifest
