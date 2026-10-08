# OC0 — Occlusion–Gradient Conflict Census Preregistration

Frozen date: 2026-09-29

## Research question

In ordinary static full-resolution 3DGS training, is cross-view gradient
conflict systematically associated with changes in a Gaussian's alpha-
compositing / occlusion role across cameras?

HarmoGS already establishes that cross-view gradient conflict exists and can
be manipulated. OC0 does NOT claim first discovery of gradient conflict.

The new question is whether conflict can be predicted *before a second
backward* from renderer state.

---

# Scope

Baseline:
- frozen P12/C0

Scenes:
- room
- bicycle

Checkpoints:
- 20K
- 25K

Cameras:
- use the MG0 frozen 64-camera training set;
- choose every second camera in frozen order => 32-camera OC0 subset;
- freeze before renderer-feature collection.

Gaussians per cell:
- 1024 highest-V_j rows from MG0-S;
- 1024 uniform nonzero-V_j rows using seed 20260929;
- union/de-duplicate, then freeze.

No formal test views.
No training continuation.
No method implementation.

---

# Gradient conflict labels

Reuse MG0 full-resolution original-loss gradients.

Use Adam-preconditioned concatenated row gradients.

For every Gaussian and every pair of OC0 cameras where BOTH row gradients have
nonzero norm, compute:

    cosine_ab

Primary continuous disagreement target:

    Y_ab = 1 - cosine_ab

Primary binary conflict:

    conflict_ab = (cosine_ab < 0)

Severe conflict diagnostic:

    severe_ab = (cosine_ab < -0.25)

Do not tune these thresholds.

For every Gaussian also report:
- negative-pair fraction;
- severe-pair fraction;
- directional dispersion:
      1 - ||mean(normalized gradients)||

---

# Renderer-state features

A diagnostic replay may be instrumented, but must not change training.

For each selected Gaussian-camera pair collect summaries over native accepted
pixel interactions:

- `visible`: any accepted hit;
- `accepted_hits`;
- `mean_alpha`;
- `mean_T_before` (transmittance immediately before this Gaussian);
- `p10_T_before`;
- `p50_T_before`;
- `mean_depth_percentile` within the native sorted list for accepted hits;
- `mean_prior_contrib_count`;
- `mean_remaining_contrib_count`;
- projected radius / footprint summary.

If a quantity cannot be extracted exactly from native replay, mark it missing;
do not replace it with an invented proxy without a prereg amendment.

Primary pairwise renderer features for camera pair (a,b):

- abs(log(mean_T_a+eps) - log(mean_T_b+eps))
- abs(mean_depth_percentile_a - mean_depth_percentile_b)
- abs(log(accepted_hits_a+1) - log(accepted_hits_b+1))
- abs(mean_alpha_a - mean_alpha_b)
- visibility-role mismatch indicator
- difference in prior contributor count

All are computed without using pair gradients.

---

# Analysis

## A. Pairwise univariate association

For every feature independently:
- Spearman(feature, Y_ab)
- AUC(feature -> conflict_ab), orienting sign only from the room-20K training
  cell and freezing it thereafter.

Report per cell and pooled.

## B. Frozen multivariate predictor

Features:
exactly the six primary pairwise renderer features above.

Model:
L2 logistic regression, C=1.0, class_weight=None.

No hyperparameter tuning.

Cross-scene evaluation:
- train room 20K -> test bicycle 20K and bicycle 25K;
- train bicycle 20K -> test room 20K and room 25K.

Also train on 20K and test same-scene 25K to test temporal transfer.

Primary:
- ROC AUC;
- average precision;
- Brier score;
- top-decile conflict enrichment versus base rate.

No gradient-derived feature is allowed.

---

# Discovery labels

## `OC0_STRONG_OCCLUSION_CONFLICT_STRUCTURE`

Requires all of:

1. conflict is nontrivial:
   median negative-pair fraction among the high-V sample >= 0.10 in all cells;

2. at least one T/depth-role feature has |Spearman| >= 0.35 in all four cells;

3. every cross-scene/temporal predictor test has:
   ROC AUC >= 0.75;

4. top-decile predicted-risk pairs have >= 2.0x the base conflict rate in every
   test cell.

## `OC0_MODERATE_STRUCTURE`

Requires:
- nontrivial conflict; AND
- all cross-scene/temporal AUC >= 0.65; AND
- top-decile enrichment >= 1.5x.

## `OC0_NO_PREDICTABLE_OCCLUSION_STRUCTURE`

Otherwise.

A positive OC0 result is a correlation/predictability result, not causal proof.

---

# Why OC0 matters

If strong:
the next method should not spend a second backward to discover conflict.
It should use renderer-state conflict risk to decide which Gaussians need
variance correction / conservative updates.

If negative:
drop the "occlusion drives harmful SGD" story and keep OneView-GS as a generic
camera-variance method.
