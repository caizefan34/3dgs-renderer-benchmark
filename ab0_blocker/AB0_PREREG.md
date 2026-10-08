# AtomBirth-GS AB0 — Parent-Free Atom Opportunity Oracle (FROZEN)

## 0. Scientific question

At an active densification stage, can a parent-free Gaussian atom proposed directly from the current multi-view photometric residual produce substantially more held-out native-loss reduction per added primitive than a parent-derived clone baseline?

AB0 intentionally gives AtomBirth a strong but fixed search budget.
If even this oracle-style search cannot beat parent-derived birth, STOP.

## 1. Frozen source / objective

Use the frozen C0 training semantics and ORIGINAL full-resolution loss:

    0.8 * L1 + 0.2 * (1 - SSIM)

No image downsampling.
No Gaussian pyramid.
No external depth estimator.
No MVS / optical-flow / semantic model.
No densification inside the short AB0 optimization.
No pruning.
No topology changes except the single test atom inserted by AB0.

Required cells:
- room@5K
- room@15K
- bicycle@5K
- bicycle@15K

These are chosen because standard densification is active / near its end.

If an exact checkpoint does not exist:
    AB0_CHECKPOINT_BLOCKER
    STOP.
Do not substitute.

Use exact checkpoint optimizer state for baseline measurements, but the base scene is frozen during candidate-only atom fitting.

Known C0 identities:
- C0 extension:
  d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
- core gsplat_cuda.so:
  935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e

Recover all checkpoint / camera hashes from frozen manifests.

## 2. Camera folds

Use the same frozen 8-camera list/cell where available:

    c0,c1,c2,c3,c4,c5,c6,c7

Fold A:
- design: c0,c1,c2,c3
- held-out: c4,c5,c6,c7

Fold B:
- design: c4,c5,c6,c7
- held-out: c0,c1,c2,c3

Within each fold use four cyclic source-camera offsets.
Thus each cell has 8 rollouts.

The source camera used for candidate generation is the first camera in the cyclic design order.
The remaining three design cameras are used for candidate scoring/fitting.

## 3. Base measurements

For every rollout, at theta_0:

- exact native RGB on all 8 cameras;
- exact original loss;
- image adjoint dL/dRGB on the source camera;
- native rendered expected depth or equivalent compositing depth;
- native accepted-hit / contribution metadata if already available.

No model parameter is changed.

## 4. Parent-free candidate generation

### 4.1 Source-pixel selection

On the source camera compute scalar photometric-need score:

    s_p = || dL/dRGB_p ||_2

Run deterministic non-maximum suppression:
- window radius: 24 pixels
- retain top 16 source pixels
- tie break by linear pixel index

No GT-error-only alternative score is allowed after results are seen.

### 4.2 Depth hypotheses

For each selected pixel with valid positive native rendered depth z:

    z_h in {0.80*z, 0.90*z, 1.00*z, 1.10*z, 1.25*z}

If native expected depth is unavailable, recover the exact native weighted/contribution depth from existing renderer outputs.
Do not use external depth.

Backproject each `(pixel, z_h)` to a 3D center using the frozen calibrated camera.

### 4.3 Initial size hypotheses

For each center create isotropic world-space scales corresponding approximately to source-view projected Gaussian sigma:

    sigma_px in {1.5, 3.0, 6.0}

Use:
    s_world = sigma_px * z_h / f_mean

where `f_mean = (fx + fy)/2`.

Set all 3 log-scales to log(s_world).
Initial rotation = identity quaternion.
Because the initial covariance is isotropic, rotation is irrelevant initially.

### 4.4 Initial opacity / appearance

Initial effective opacity:
    alpha0 = 0.05
converted through the exact native inverse-opacity parameterization.

Initial SH:
- DC initialized from source-camera GT RGB at the selected pixel using the exact renderer SH convention;
- all higher-order coefficients = 0.

This yields at most:
    16 * 5 * 3 = 240 parent-free candidates per rollout.

## 5. Cheap multi-view screening

AB0 must not native-render all 240 candidates through full candidate optimization.

For every candidate, project it into all four design cameras.

Reject if:
- behind any required design camera;
- outside image in >2/4 design views;
- projected sigma <0.5 px or >20 px in >=2 design views;
- NaN/Inf.

Compute a fixed cheap screen score using the BASE images/adjoints only.

For each valid projected footprint, approximate the infinitesimal candidate image direction:

    dI_p ≈ G_p * alpha0 * (c_init - I_base_p)

where G_p is the candidate 2D Gaussian footprint under the current camera.

Score:

    score = - sum_design < dL/dRGB, dI >

Higher is better.

This is ONLY a ranking proxy.

Keep top 16 parent-free candidates by score.
Tie break by candidate generation index.

If fewer than 8 survive:
    mark rollout INSUFFICIENT_CANDIDATES.

No additional candidate-generation rescue is allowed.

## 6. Parent-derived control pool

Construct a matched baseline pool from current visible Gaussians.

For the same source camera:
- rank visible Gaussians by native densification signal used by the frozen trainer at that state;
- keep top 16 valid parents.

For each parent create ONE clone-like candidate atom:
- xyz initialized exactly at parent center;
- scale/rotation copied from parent;
- opacity initialized to alpha0=0.05, not copied;
- SH DC copied from parent;
- higher SH initialized to parent values up to the currently active degree.

The original parent remains present.
This creates an "add one child near existing support" baseline.

Do NOT perform destructive split/removal in AB0.
The purpose is matched single-atom birth opportunity, not reproducing the full trainer topology operation.

## 7. Candidate-only exact fitting

The base scene is frozen for all candidate fitting.

For each of the:
- top 16 parent-free candidates;
- 16 parent-derived candidates;

make an independent base-scene clone and optimize ONLY the candidate atom.

Trainable candidate parameters:
- xyz
- log-scales
- rotation
- opacity
- active SH coefficients

Use exactly 8 candidate-only Adam steps on the four design cameras in cyclic order.

Learning rates:
- use the frozen trainer's per-family learning rates at this global iteration;
- candidate Adam starts with zero moments and step=0;
- no LR tuning;
- no damping tuning.

No base Gaussian parameter may update.

After step 8 evaluate:
- design mean native loss;
- held-out mean native loss;
- PSNR/SSIM on all 8 cameras;
- candidate visibility count;
- candidate alpha/scale validity.

## 8. Deployable selection simulation

The method is NOT allowed to select by held-out loss.

For each pool separately:
- select the single candidate with lowest FINAL DESIGN mean native loss after its 8 fitting steps;
- tie break by original pool order.

Call:
- `ATOM_SELECTED`
- `PARENT_SELECTED`

Then reveal held-out performance only for these selected candidates.

This prevents an oracle that picks the best candidate using held-out cameras.

## 9. Baseline full-model progress control

From the identical theta_0 + optimizer state, run 8 ordinary native full-model Adam steps on the same four-camera cyclic design order, with topology frozen.

Evaluate held-out native loss.

Define:
    D_full8 = L0_hold - L_full8_hold

This is NOT a matched-cost comparison.
It only contextualizes how much global progress one inserted atom can create.

## 10. Metrics

For the deployed-selected atom:

    D_atom = L0_hold - L_atom_hold
    D_parent = L0_hold - L_parent_hold

Primary ratio when D_parent > 0:

    R_parent = D_atom / D_parent

If D_parent <= 0:
- report denominator as non-positive;
- do not fabricate a ratio.

Full-step progress fraction when D_full8 > 0:

    P_full8 = D_atom / D_full8

Also record:
- design loss improvement;
- held-out PSNR delta;
- held-out SSIM delta;
- candidate visible views;
- nearest-neighbor normalized distance to existing Gaussians:

    d_norm = ||x_new - x_nn|| / max(mean_scale_nn, 1e-8)

This distance is descriptive only and does not enter selection.

## 11. Mechanical verdict

A SINGLE parent-free protocol is already fixed; no family selection exists.

Per cell aggregate 8 rollouts.

Rollouts with:
- INSUFFICIENT_CANDIDATES
- NaN/Inf
are failures.

### AB0_STRONG_PROMOTE

EVERY one of the 4 cells satisfies:
- ATOM_SELECTED held-out improvement positive in >=7/8 rollouts;
- parent-derived selected held-out improvement positive in >=4/8 rollouts;
- among rollouts with D_parent>0:
    median R_parent >= 2.0
    p10 R_parent >= 1.0
- median P_full8 >= 0.20
- median held-out PSNR delta(atom vs base) > 0
- worst held-out PSNR delta(atom vs base) >= -0.05 dB
- median candidate visibility >= 4/8 cameras
- failure count <=1/8

### AB0_PROMOTE

EVERY cell satisfies:
- ATOM_SELECTED held-out improvement positive in >=6/8;
- among D_parent>0 rollouts:
    median R_parent >= 1.50
    p10 R_parent >= 0.75
- median P_full8 >= 0.10
- median held-out PSNR delta >= -0.01 dB
- worst held-out PSNR delta >= -0.10 dB
- median candidate visibility >= 3/8
- failure count <=2/8

### AB0_DROP

Otherwise.

The full8 progress fraction threshold is intentionally modest:
AB0 tests whether ONE atom can account for a nontrivial fraction of eight global steps, not whether a single atom replaces all optimization.

## 12. Controls

### Random-pixel control
For one fixed rollout per cell:
- same candidate generation machinery;
- 16 source pixels sampled uniformly with seed=20260930;
- identical depth/scale hypotheses and fitting.

Descriptive only.

### Existing-center control
For one fixed rollout per cell:
- parent-free generation but force center to the nearest existing Gaussian center;
- keep new scale/opacity/color initialization.

Descriptive only.

These help distinguish "residual-guided new support" from merely "another atom".

## 13. Timing / cost

Record:
- candidate generation time;
- cheap screen time;
- 32 candidate-fitting total time;
- standard one training-step time.

AB0 does NOT use these for a speedup claim.
A later AB1 must reduce search cost dramatically before any E2E claim.

## 14. Forbidden rescue

After results:
- no external depth/MVS;
- no more source pixels;
- no more depth hypotheses;
- no learned candidate proposal;
- no semantic/edge network;
- no held-out candidate selection;
- no per-scene thresholds;
- no changing alpha0;
- no changing fitting steps/LRs;
- no destructive split baseline substitution.

A DROP stays DROP.

## 15. Required artifacts

- AB0_SOURCE_IDENTITY.json
- AB0_ENVIRONMENT.json
- AB0_CHECKPOINT_INVENTORY.json
- AB0_CAMERA_FOLDS.json
- AB0_BASE_RENDER_STATS.json
- AB0_CANDIDATE_POOLS.json
- AB0_SCREENING.json
- AB0_FITTING_RESULTS.json
- AB0_SELECTED_RESULTS.json
- AB0_FULL8_CONTROL.json
- AB0_CONTROLS.json
- AB0_TIMING.json
- AB0_SUMMARY.json
- AB0_VERDICT.json
- AB0_REPORT.md
- raw logs
- immutable SHA256 manifest

## 16. Interpretation

PROMOTE means:
- a parent-free, residual/adjoint-guided atom proposal contains substantially more native held-out descent per added primitive than a matched parent-derived add-one-atom baseline;
- the effect is consistent enough to justify AB1 search-efficiency / actual densification experiments.

It does NOT prove:
- end-to-end training acceleration;
- lower final Gaussian count;
- production search cost is acceptable;
- superiority to EDGS/SteepGS/ResGS;
- novelty.

Those require later stages.
