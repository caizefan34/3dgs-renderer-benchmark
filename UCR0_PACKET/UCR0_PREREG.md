# UCR0 — Unbiased Control-Variate Raster Feasibility Oracle (FROZEN)

## 0. Question

Can a cheap deterministic proxy for the standard C0 image objective act as a control variate so that
only 5–10% of exact full-resolution loss units require exact residual gradients, while the resulting
gradient estimator remains unbiased, substantially lowers variance versus plain sparse exact-unit
sampling, and has a cost model compatible with >=3x full-training speed?

This is a statistical/mechanism P0. It does NOT implement production sparse CUDA and cannot make an
E2E speedup claim.

## 1. Cells

- room@10K
- room@15K
- bicycle@10K
- bicycle@15K

For each cell use 4 deterministic midpoint-stratified TRAIN source cameras.
Evaluate candidate states on:
- source camera
- 3 deterministic peer TRAIN cameras
- 8 corrected true-holdout cameras

No camera fishing.

## 2. Exact loss-unit decomposition

C0 loss remains exactly:
    L = 0.8 * L1 + 0.2 * (1 - SSIM)

Define loss units as 16x16 SSIM-center tiles with the required 5-pixel halo.
Each image-center pixel's L1 and SSIM-center contribution belongs to exactly one unit.

Before UCR probes, implement a reference loss-map decomposition and require:
- scalar full-image loss relative error <= 1e-6 vs frozen fused loss;
- 16 fixed optimizer-scaled Rademacher directional derivatives:
  median relative error <= 1e-4, max <= 5e-4.

Fail => UCR0_LOSS_DECOMPOSITION_FAIL.

## 3. Frozen proxy

One proxy only:
Q4 = render at quarter linear resolution (W/4,H/4), same C0 representation/objective,
then bilinearly lift RGB and loss sensitivity to full-resolution loss units.

P17 already showed coarse gradients are poor as direct replacements. UCR0 specifically tests whether
Q4 is useful ONLY as a control variate with exact residual correction.

No alternate proxy after results.

## 4. Estimator

Let exact unit gradients be g_u and proxy unit gradients be c_u.
Let C = sum_u c_u.

For uniform Bernoulli inclusion with probability q:
    G_hat = C + sum_{u in S} (g_u - c_u) / q

q is frozen to:
- 0.05
- 0.10

Mandatory comparator at the same q:
    HT_hat = sum_{u in S} g_u / q

No importance sampling in UCR0.

## 5. Monte-Carlo protocol

Per source camera and q:
- 16 frozen Bernoulli masks, generated before seeing results.
- full parameter gradient may be computed with the existing renderer even if this does not realize
  the target sparse wall cost; this is a P0 statistical oracle.
- cost prediction MUST use HM0 measured unit/raster work, not the unoptimized masked-backward wall time.

Compare gradients in optimizer-unit coordinates:
each parameter group divided by its frozen native learning rate before dot products/norms.

## 6. Candidate Adam state

For every mask:
- clone identical theta0 + optimizer state;
- inject G_hat as the current gradient;
- execute exactly one native Adam update;
- no moment reset;
- no LR changes.

Reference:
- exact full C0 gradient + one native Adam update from same state.

Evaluate source, peers and true-holdout.

## 7. Metrics

Per fixed q:
- empirical gradient bias norm / ||G_exact||
- optimizer-unit cosine(G_hat,G_exact): median and p10
- norm ratio
- variance MSE / ||G_exact||^2
- variance-reduction factor vs HT_hat at same q
- exact-Adam source progress capture
- peer mean-loss delta(candidate - exact-Adam)
- true-holdout PSNR delta(candidate - exact-Adam)
- HM0-predicted proxy+residual cost fraction
- predicted ideal E2E speed upper bound

## 8. Mechanical verdict

A SINGLE fixed q must pass ALL 4 cells.

### UCR0_STRONG_PROMOTE
- loss decomposition PASS
- bias <= 0.02
- cosine median >= 0.90
- cosine p10 >= 0.80
- variance reduction vs HT >= 2.0x
- median exact-Adam source progress capture >= 0.85
- p10 progress capture >= 0.70
- median peer mean-loss delta <= +0.5% of exact-Adam improvement magnitude
- median true-holdout PSNR delta >= -0.02 dB
- worst true-holdout PSNR delta >= -0.08 dB
- predicted cost fraction <= 0.25
- predicted ideal E2E speed >= 3.0x
- no NaN/Inf

### UCR0_PROMOTE
- bias <= 0.03
- cosine median >= 0.85
- cosine p10 >= 0.70
- variance reduction vs HT >= 1.5x
- median progress capture >= 0.75
- p10 >= 0.55
- median true-holdout PSNR delta >= -0.04 dB
- worst >= -0.12 dB
- predicted cost fraction <= 0.30
- predicted ideal E2E speed >= 3.0x
- no NaN/Inf

Else UCR0_DROP.

If q=0.05 and q=0.10 differ, the analyzer may select a passing q only if that SAME q passes every cell.
No per-scene selection.

## 9. Interpretation boundary

PROMOTE means only:
the statistical estimator appears sufficiently accurate/low-variance under an HM0-based cost oracle
to justify UCR1 sparse exact-residual CUDA work.

It does NOT establish real 3x speedup.

## 10. Forbidden rescue

No:
- q outside {0.05,0.10}
- importance sampling
- alternate proxy/resolution
- low-res loss as the training objective
- threshold relaxation
- full training
- local GPU

## 11. Required outputs

UCR0_ENVIRONMENT.json
UCR0_SOURCE_IDENTITY.json
UCR0_CAMERA_SEMANTICS.json
UCR0_LOSS_PARITY.json
UCR0_RAW.json
UCR0_SUMMARY.json
UCR0_VERDICT.json
UCR0_REPORT.md
MANIFEST.sha256
