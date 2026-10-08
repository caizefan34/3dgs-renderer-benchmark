# A0-v2 Preregistration — Exact Backward Coefficient Tensor Census

Frozen date: 2026-09-29

## 0. Scope

Baseline:
- frozen P12/C0

Scenes:
- room
- bicycle

Checkpoints:
- 20K
- 25K

Views:
- 8 deterministic training cameras per scene/checkpoint
- uniformly spread over canonical train-camera order
- frozen before instrumentation results

Protocol:
- full effective resolution
- original 0.8*L1 + 0.2*(1-SSIM)
- training views only

No garden.
No continuation.
No formal test views.
No production CUDA optimization.

---

## 1. Exact algebra gate

Instrument an isolated diagnostic copy of the frozen C0 raster backward.

Before collecting structure data, document the exact native equations and expose
an algebraically exact sufficient-state factorization for every accepted hit.

At minimum identify:

    q_alpha[p,i]

such that the full contribution of alpha to local parameter gradients is:

    q_alpha[p,i] * d alpha[p,i] / d local_parameter

before expanding the local alpha derivative.

And:

    q_color[p,i,c]

such that the local color / SH derivative contribution factors through
`q_color`.

If a single exact scalar q_alpha does not exist under the frozen kernel's
clamping / saturation / recurrence semantics, export the minimal exact vector
or tuple instead and document the identity.

NO approximate proxy is allowed.

If the factorization cannot be written exactly:
    `A0_ALGEBRA_BLOCKER`
and STOP A0.

---

## 2. Native parity gate

The instrumented diagnostic backward must match uninstrumented frozen C0 on
the same frozen views:

- RGB
- scalar original loss
- xyz gradient
- scale gradient
- quaternion gradient
- opacity gradient
- SH/DC gradient
- SH/rest gradient
- screen-space gradient fields used by the trainer

within the existing C0 numerical envelope.

If parity fails:
    `A0_PARITY_BLOCKER`
and STOP.

---

## 3. Frozen bucket selection

Using ONLY native accepted-hit counts:

Per scene/checkpoint:
- rank all nonempty 16x16 tile × native backward-bucket objects;
- select top 256 by accepted-hit count;
- select 256 deterministic uniform nonempty buckets using seed 20260929;
- freeze before coefficient analysis.

No gradient magnitude, loss, or basis reconstruction enters selection.

Write:
`A0_BUCKETS.json`

---

## 4. Exact coefficient tensors

For every selected bucket create dense 16x16 spatial tensors with zeros at
nonaccepted pixels.

For each local Gaussian column i:

    Q_alpha[:, i]
    Q_color_r[:, i]
    Q_color_g[:, i]
    Q_color_b[:, i]

plus:
- accepted mask
- saturation/clamp state if relevant
- native local Gaussian ID
- accepted-hit count
- exact selected-bucket native gradient contribution by parameter group

The dense pixel dimension is always 256.

Use FP32 export for primary analysis.

---

# 5. Part A — ORACLE intrinsic rank

For every coefficient family independently, treat the bucket matrix as:

    Q in R^(256 x m)

where m <= native bucket width.

Centering:
DO NOT center Q; the DC component is part of the backward signal.

Compute exact SVD offline.

For r in {1,2,4,8,16,32} report:

    retained_energy_oracle(r)
      = ||Q_r||_F^2 / ||Q||_F^2

and:
- effective rank = (sum s^2)^2 / sum s^4
- r90, r95, r99, r995

Color:
also analyze a joint matrix

    Q_color_joint in R^(256 x (3m))

so low rank cannot be created by treating RGB channels independently.

## Oracle strong pattern

Across BOTH scenes and BOTH checkpoints, on high-work buckets:

- median r995 <= 8 for q_alpha; AND
- median r995 <= 8 for joint q_color; AND
- p90 r99 <= 16 for both.

`A0_ORACLE_LOW_RANK_STRONG`

## Oracle moderate

- median r99 <= 16 for both q_alpha and joint q_color.

`A0_ORACLE_LOW_RANK_MODERATE`

Otherwise:
`A0_ORACLE_HIGH_RANK`

If ORACLE is high-rank, STOP before any A1 derivation.

---

# 6. Part B — fixed deployable spatial basis

Use a SINGLE frozen 2D orthonormal DCT-II basis on the 16x16 pixel grid.

Basis ordering:
- increasing total spatial frequency fx+fy;
- tie break fx then fy.

Use prefixes:

    r in {4,8,16,32}

Same basis for every scene/view/bucket/coefficient.

No learned basis.
No PCA/SVD basis at runtime.
No scene-specific basis selection.

Reconstruct:

    Q_hat_r = B_r B_r^T Q

Report:
- retained coefficient energy
- relative Frobenius error
- p95/max absolute coefficient error

## Fixed-basis strong pattern

There exists one SINGLE r <= 16 that satisfies in all four scene/checkpoint
cells on high-work buckets:

q_alpha:
- median retained energy >= .995
- p10 retained energy >= .98

joint q_color:
- median retained energy >= .995
- p10 retained energy >= .98

Then:
`A0_FIXED_BASIS_STRONG`

Moderate:
one single r<=16 with median retention >=.98 for both families in all cells.

Otherwise:
`A0_FIXED_BASIS_WEAK`

---

# 7. Reconstructed native VJP fidelity

For every r, substitute ONLY the reconstructed coefficient tensors into an
offline exact replay of the selected-bucket native local derivative factors.

Everything else is native/exact.

Aggregate selected-bucket gradients per view and compare with exact selected-
bucket native gradients.

Report for:
- xyz
- scale
- quaternion
- opacity
- SH/DC
- SH/rest

Metrics:
- cosine
- relative L2 error
- norm ratio
- max abs error

Also report whole selected-bucket concatenated gradient cosine.

## VJP strong

A SINGLE r<=16 must satisfy in all four cells:

- every parameter group median cosine >= .995
- every parameter group median relative L2 error <= .05
- concatenated gradient cosine >= .998
- no parameter group median norm ratio outside [0.95,1.05]

`A0_VJP_STRONG`

Moderate:
- all group median cosine >= .99
- concatenated cosine >= .995
- all group median relative L2 error <= .10

Otherwise:
`A0_VJP_WEAK`

---

# 8. Conservative work opportunity proxy

For each bucket and fixed r:

    proxy_ratio =
        r * (# local Gaussian columns with >=1 accepted hit)
        / (# accepted pixel-Gaussian hits)

This assumes one basis-moment operation per basis coefficient per active
Gaussian column and gives ZERO credit for overheads.

Report pooled and per-bucket distribution.

Strong opportunity requires for the SAME r used by fixed-basis/VJP gates:

- pooled proxy_ratio <= .35 on high-work buckets
- >=75% high-work buckets have ratio < .50

Moderate:
- pooled ratio <= .50

This is NOT measured speedup.

---

# 9. Final A0-v2 verdict

`A0_STRONG_ADJOINT_TENSOR_SIGNAL` requires:
- exact algebra gate pass
- parity pass
- ORACLE strong or moderate
- FIXED_BASIS strong
- VJP strong
- strong cost opportunity

`A0_MODERATE_ADJOINT_TENSOR_SIGNAL` requires:
- exact algebra + parity
- ORACLE not high-rank
- FIXED_BASIS moderate
- VJP moderate
- moderate cost opportunity

`A0_STRUCTURE_EXISTS_ENCODING_UNSOLVED`:
- ORACLE strong/moderate
- but fixed-basis or VJP gate fails

This is scientifically useful: low-rank structure exists but DCT encoding is
not sufficient. One encoding-design iteration is allowed, with no production
CUDA yet.

`A0_NO_USEFUL_ADJOINT_TENSOR_STRUCTURE`:
- ORACLE high-rank
OR
- oracle/fixed/VJP structure is too weak for even moderate cost opportunity.

`A0_ALGEBRA_BLOCKER`
`A0_PARITY_BLOCKER`

No threshold tuning.
No basis search after seeing results.
No full training.
