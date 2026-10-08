# MG0-S — Variance Concentration + Low-Rank Sidecar Preregistration

Frozen date: 2026-09-29

## Data reuse

Reuse MG0's frozen:
- P12/C0 identity;
- room + bicycle;
- 64 training cameras;
- 20K and 25K checkpoints;
- full-resolution original-loss gradients.

Do not add formal test views.

The sidecar should run while MG0 already has the 64-view gradients available,
to avoid duplicate full-resolution backward work.

---

# Part A — Gradient-Variance Concentration Law

For every Gaussian row j, in Adam-preconditioned coordinates, compute:

    V_j = mean_v ||P(g_v,j) - mean_u P(g_u,j)||^2

Zeros from cameras where the row receives zero gradient remain zeros; this is
the actual camera-sampling gradient distribution.

Compute the cumulative fraction of total variance energy captured by the top:

- 0.1%
- 0.25%
- 0.5%
- 1%
- 2%
- 5%
- 10%
- 20%

of Gaussian rows ranked by V_j.

Also report:
- Gini coefficient of V_j;
- effective support = (sum V)^2 / sum(V^2);
- fraction of rows accounting for 50%, 80%, 90%, 95% of variance;
- the same statistics per parameter group.

Do this independently for:
- room 20K
- room 25K
- bicycle 20K
- bicycle 25K

No threshold selection after data.

## Strong L1 pattern

Call the pattern `L1_STRONG_CONCENTRATION` only if all four cells satisfy:

- top 2% rows capture >= 70% of total Adam-coordinate variance; AND
- <= 10% rows capture >= 90% of variance.

## Moderate L1 pattern

`L1_MODERATE_CONCENTRATION` if all four cells satisfy:

- top 5% capture >= 60%; AND
- <= 20% rows capture >= 90%.

Otherwise:
`L1_DIFFUSE_VARIANCE`.

These are descriptive discovery labels, not method gates.

---

# Part B — View-Gradient Low-Rank Law

Use a deterministic 4096-row analysis set per scene/checkpoint:

- 2048 rows from the highest-variance portion, using the frozen V_j ranking;
- 2048 uniform rows from all nonzero-variance Gaussians using a frozen seed.

Freeze row IDs before singular-value analysis.

For every selected Gaussian j, construct the 64 x D concatenated
Adam-preconditioned gradient matrix:

    X_j[v,:] = P(g_v,j)

Center across views:

    Xc_j = X_j - mean_v X_j

Compute singular values.

Report two versions:

1. `all_views`: zeros for invisible/no-gradient cameras remain in X.
2. `visible_only`: use only cameras with nonzero row-gradient norm, requiring
   >= 8 visible cameras; otherwise label insufficient.

For each mode, report:
- rank needed for 80%, 90%, 95% centered energy;
- rank-1/2/4/8/16 explained energy;
- participation/effective rank:
      r_eff = (sum s^2)^2 / sum s^4
- distributions separately for hot and uniform samples.

Also report parameter-group matrices for:
- geometry = xyz + scale + quaternion
- opacity
- appearance = SH/DC + SH/rest

Do not interpret scalar opacity rank as evidence for the global law.

## Strong L2 pattern

`L2_STRONG_LOW_RANK` only if, in all four scene/checkpoint cells:

- median visible-only rank-8 explained energy >= 90%; AND
- median visible-only r_eff <= 8; AND
- the uniform sample also has median rank-8 energy >= 80%.

## Moderate L2 pattern

`L2_MODERATE_LOW_RANK` if all four cells satisfy:

- median rank-16 explained energy >= 90%; AND
- median r_eff <= 16.

Otherwise:
`L2_NO_GENERAL_LOW_RANK`.

---

# Combined interpretation

The scientifically important positive case is not merely L1 or L2 alone.

`OPT_STRUCTURE_STRONG` requires:
- L1 strong or moderate; AND
- L2 strong or moderate;
- both on room and bicycle at 20K and 25K.

This would justify a later OneView-GS design that explicitly combines sparse
hot-set correction with low-rank camera-conditioned state.

No production implementation is authorized by MG0-S alone.
